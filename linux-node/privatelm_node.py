#!/usr/bin/env python3
"""privatelm-node: a Linux PC (with its GPUs) in a PrivateLM phone cluster.

  privatelm-node detect               devices, free memory, bandwidth
  privatelm-node worker               lend this PC's GPU(s) to phone masters
  privatelm-node serve -m model.gguf  run a model here, optionally with
                                      phones (or other PCs) as workers, and
                                      serve it as an OpenAI-compatible API
  privatelm-node scan                 list workers and servers on the LAN

Workers advertise themselves exactly like worker phones do
(_privatelm._tcp, TXT ram/bw/model/id/v), so the app's "Scan nearby" finds a
PC like any other phone. Phones and PC must run the same llama.cpp RPC
protocol (rpc7); build.sh builds the commit the app vendors.

The RPC protocol has no authentication: run workers only on a network you
trust (home LAN, or a VPN such as NetBird/WireGuard with --host set to the
VPN address).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import socket
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

RPC_VERSION = "rpc7"
WORKER_TYPE = "_privatelm._tcp.local."
SERVER_TYPE = "_privatelm-api._tcp.local."
DEFAULT_RPC_PORT = 50052
DEFAULT_API_PORT = 8080

BIN_DIR = Path(os.environ.get("PRIVATELM_NODE_BIN", Path(__file__).resolve().parent))
CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "privatelm-node"
CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "privatelm-node"

# Share of a device's free memory the planner may fill. Phones: the same
# 0.75 the app uses for workers. GPUs keep room for the KV cache and
# compute buffers; the host keeps room for the OS and llama-server.
GPU_SHARE = 0.95  # of free VRAM; GPU_RESERVE_MB comes off on top
PHONE_SHARE = 0.75
PC_RAM_SHARE = 0.9  # of what a PC RAM worker offers (already minus its reserve)
HOST_SHARE = 0.8
UNKNOWN_PHONE_SPEED = 4.0  # GB/s, as in the app


def log(msg: str) -> None:
    print(f"[privatelm-node] {msg}", file=sys.stderr, flush=True)


def die(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    log(msg)
    sys.exit(1)


def find_bin(name: str) -> str:
    for cand in (BIN_DIR / name, BIN_DIR.parent / "build" / "bin" / name):
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    found = shutil.which(name)
    if found:
        return found
    die(f"{name} not found next to privatelm-node or in PATH. Run build.sh first.")


# ── Devices ──────────────────────────────────────────────────────────────────


@dataclass
class Device:
    name: str
    description: str
    backend: str
    type: str
    free_mb: int
    total_mb: int
    bandwidth: float
    device_id: str = ""

    @property
    def is_gpu(self) -> bool:
        return self.type in ("gpu", "igpu")


def usable_gpus(devices: list[Device]) -> list[Device]:
    """The GPUs llama.cpp would use, the same way it picks them: each card
    once (a card seen by CUDA and Vulkan, or ROCm and Vulkan, has one PCI ID;
    the first backend wins), and integrated GPUs only without a dedicated
    one."""
    gpus: list[Device] = []
    for d in devices:
        if d.type != "gpu":
            continue
        if d.device_id and any(g.device_id == d.device_id for g in gpus):
            continue
        gpus.append(d)
    if gpus:
        return gpus
    igpus = [d for d in devices if d.type == "igpu"]
    return [d for d in igpus if d.backend == igpus[0].backend] if igpus else []


def probe(bench: bool = True, refresh: bool = False) -> list[Device]:
    """Local devices. Bandwidth is measured once and cached; free memory is
    re-read every time."""
    cache = CACHE_DIR / "probe.json"
    cached: dict[str, float] = {}
    if cache.exists() and not refresh:
        try:
            cached = {d["name"]: d["bandwidth_gbps"] for d in json.loads(cache.read_text())["devices"]}
        except (ValueError, KeyError):
            cached = {}
    need_bench = bench and (refresh or not cached)
    args = [find_bin("privatelm-probe")] + ([] if need_bench else ["--no-bench"])
    if need_bench:
        log("Measuring memory bandwidth of each device (a few seconds)...")
    out = subprocess.run(args, check=True, capture_output=True, text=True).stdout
    data = json.loads(out)
    if need_bench:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(out)
    devices = []
    for d in data["devices"]:
        bw = d["bandwidth_gbps"] if need_bench else cached.get(d["name"], d["bandwidth_gbps"])
        devices.append(Device(d["name"], d["description"], d["backend"], d["type"],
                              d["free_mb"], d["total_mb"], bw, d.get("device_id", "")))
    return devices


def host_available_mb() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0


def machine_id() -> str:
    """This PC's lasting worker ID (the app finds a worker again by it when
    its IP changes)."""
    for p in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            raw = Path(p).read_text().strip()
            if raw:
                return hashlib.sha1(f"privatelm:{raw}".encode()).hexdigest()[:16]
        except OSError:
            continue
    path = CONFIG_DIR / "id"
    if not path.exists():
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_hex(8))
    return path.read_text().strip()


def lan_address() -> str:
    """The address other devices reach this PC at (the default route's)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 9))  # no packet is sent
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


# ── mDNS ─────────────────────────────────────────────────────────────────────


class Advertiser:
    """Registers one DNS-SD service via python-zeroconf, or avahi-publish."""

    def __init__(self, service_type: str, name: str, port: int, txt: dict[str, str], address: str):
        self.service_type, self.name, self.port = service_type, name, port
        self.txt, self.address = txt, address
        self._zc = self._info = self._proc = None

    def start(self) -> bool:
        try:
            from zeroconf import ServiceInfo, Zeroconf  # type: ignore

            self._info = ServiceInfo(
                self.service_type,
                f"{self.name}.{self.service_type}",
                addresses=[socket.inet_aton(self.address)],
                port=self.port,
                properties=self.txt,
                server=f"{socket.gethostname()}.local.",
            )
            self._zc = Zeroconf(interfaces=[self.address])
            self._zc.register_service(self._info, allow_name_change=True)
            return True
        except ImportError:
            pass
        avahi = shutil.which("avahi-publish-service") or shutil.which("avahi-publish")
        if avahi:
            stype = self.service_type.replace(".local.", "").rstrip(".")
            cmd = [avahi] + ([] if avahi.endswith("-service") else ["-s"])
            cmd += [self.name, stype, str(self.port)] + [f"{k}={v}" for k, v in self.txt.items()]
            self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        log("No mDNS: install python3-zeroconf (or avahi-utils). Pair by QR code or IP instead.")
        return False

    def stop(self) -> None:
        if self._zc and self._info:
            try:
                self._zc.unregister_service(self._info)
            finally:
                self._zc.close()
        if self._proc:
            self._proc.terminate()


@dataclass
class Peer:
    kind: str  # "worker" or "server"
    name: str
    host: str
    port: int
    txt: dict[str, str] = field(default_factory=dict)

    @property
    def endpoint(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def ram_mb(self) -> Optional[int]:
        v = self.txt.get("ram")
        return int(v) if v and v.isdigit() else None

    @property
    def bandwidth(self) -> Optional[float]:
        try:
            return float(self.txt["bw"])
        except (KeyError, ValueError):
            return None

    @property
    def is_gpu(self) -> bool:
        return self.txt.get("kind") == "gpu"


def scan(timeout: float = 4.0) -> list[Peer]:
    try:
        from zeroconf import ServiceBrowser, Zeroconf  # type: ignore
    except ImportError:
        return _scan_avahi()

    found: dict[tuple[str, str], Peer] = {}
    zc = Zeroconf()

    class Listener:
        def add_service(self, zc_, type_, name):
            info = zc_.get_service_info(type_, name, timeout=2000)
            if not info:
                return
            addrs = [a for a in info.parsed_addresses() if ":" not in a]
            if not addrs:
                return
            txt = {k.decode(): (v.decode() if v else "") for k, v in info.properties.items()}
            kind = "server" if type_ == SERVER_TYPE else "worker"
            found[(kind, name)] = Peer(kind, name.split("." + type_)[0], addrs[0], info.port, txt)

        def update_service(self, *a):
            pass

        def remove_service(self, *a):
            pass

    browsers = [ServiceBrowser(zc, t, Listener()) for t in (WORKER_TYPE, SERVER_TYPE)]
    try:
        time.sleep(timeout)
    finally:
        for b in browsers:
            b.cancel()
        zc.close()
    return sorted(found.values(), key=lambda p: (p.kind, p.name))


def _scan_avahi() -> list[Peer]:
    if not shutil.which("avahi-browse"):
        die("Scanning needs python3-zeroconf or avahi-utils.")
    peers = []
    for stype, kind in (("_privatelm._tcp", "worker"), ("_privatelm-api._tcp", "server")):
        out = subprocess.run(["avahi-browse", "-rpt", stype], capture_output=True, text=True).stdout
        for line in out.splitlines():
            f = line.split(";")
            if len(f) < 10 or f[0] != "=" or f[2] != "IPv4":
                continue
            txt = dict(re.findall(r'"([^"=]+)=([^"]*)"', f[9]))
            peers.append(Peer(kind, f[3].replace("\\032", " "), f[7], int(f[8]), txt))
    return peers


def worker_qr(host: str, port: int, name: str, ram: int, bw: float, wid: str, kind: str) -> str:
    from urllib.parse import urlencode

    q = urlencode({"host": host, "port": port, "name": name, "ram": ram, "bw": f"{bw:.1f}",
                   "id": wid, "kind": kind, "v": RPC_VERSION})
    return f"privatelm://worker?{q}"


def show_qr(data: str) -> None:
    print(f"\nPairing code (Phone cluster > Add worker > Scan QR):\n  {data}\n")
    try:
        import qrcode  # type: ignore

        qr = qrcode.QRCode(border=1)
        qr.add_data(data)
        qr.print_ascii(invert=True)
    except ImportError:
        if shutil.which("qrencode"):
            subprocess.run(["qrencode", "-t", "ANSIUTF8", data])
        else:
            print("  (pip install qrcode, or apt install qrencode, to show it as a QR code)")


# ── GGUF ─────────────────────────────────────────────────────────────────────


@dataclass
class ModelInfo:
    n_layer: int
    n_embd: int
    n_vocab: int
    layer_mb: float   # weights of one transformer block
    kv_layer_mb: float  # KV cache of one block at the chosen context
    output_mb: float  # output head (or tied token embedding)
    size_mb: float
    embd_mb: float = 0.0  # token embedding: always in this PC's RAM
    tied: bool = False  # output head shares the token embedding
    n_expert: int = 0  # MoE: experts per layer
    n_expert_used: int = 0  # MoE: experts that run per token
    expert_mb: float = 0.0  # MoE: all experts of one layer
    n_nextn: int = 0  # MTP: draft layers at the end of the model

    def per_layer_mb(self) -> float:
        """Memory per layer: all weights plus KV cache."""
        return self.layer_mb + self.kv_layer_mb

    @property
    def is_moe(self) -> bool:
        return self.n_expert > 1 and self.expert_mb > 0

    def read_mb(self) -> float:
        """Weights read per layer and token: an MoE layer only reads the
        experts that run, which is why experts in RAM are cheap."""
        if not self.is_moe:
            return self.layer_mb
        used = max(1, self.n_expert_used) / self.n_expert
        return self.layer_mb - self.expert_mb + self.expert_mb * used


def gguf_info(path: str, ctx: int) -> ModelInfo:
    """Layer count, sizes and KV cache size from the GGUF header and tensor
    table (tensor sizes are exact: taken from the data offsets)."""
    scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            die(f"{path} is not a GGUF file")
        version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
        if version < 2:
            die("GGUF v1 files are not supported")

        def read_str() -> str:
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")

        def read_val(t: int):
            if t in scalar:
                fmt = "<" + scalar[t]
                return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
            if t == 8:
                return read_str()
            if t == 9:
                et, n = struct.unpack("<IQ", f.read(12))
                if et in scalar:
                    f.seek(struct.calcsize("<" + scalar[et]) * n, 1)
                else:
                    for _ in range(n):
                        read_val(et)
                return ("array", n)
            die(f"Unknown GGUF value type {t}")

        kv: dict = {}
        for _ in range(n_kv):
            key = read_str()
            (t,) = struct.unpack("<I", f.read(4))
            kv[key] = read_val(t)

        tensors = []
        for _ in range(n_tensors):
            name = read_str()
            (nd,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
            _ttype, offset = struct.unpack("<IQ", f.read(12))
            tensors.append((offset, name, dims))
        align = kv.get("general.alignment", 32) or 32
        data_start = (f.tell() + align - 1) // align * align

    arch = kv.get("general.architecture")
    a = lambda k, d=None: kv.get(f"{arch}.{k}", d)  # noqa: E731
    n_layer = a("block_count")
    if not isinstance(n_layer, int) or n_layer <= 0:
        die("Could not read the model's layer count")

    tensors.sort()
    sizes: dict[str, int] = {}
    for i, (off, name, dims) in enumerate(tensors):
        nxt = tensors[i + 1][0] if i + 1 < len(tensors) else file_size - data_start
        sizes[name] = nxt - off
    blk = sum(v for k, v in sizes.items() if k.startswith("blk."))
    out = sizes.get("output.weight", sizes.get("token_embd.weight", 0))

    n_embd = a("embedding_length", 0) or 0
    n_head = a("attention.head_count", 1)
    n_head_kv = a("attention.head_count_kv", n_head)
    n_head = n_head if isinstance(n_head, int) and n_head > 0 else 1
    n_head_kv = n_head_kv if isinstance(n_head_kv, int) and n_head_kv > 0 else n_head
    k_dim = a("attention.key_length", n_embd // n_head if n_embd else 128)
    v_dim = a("attention.value_length", k_dim)
    vocab = kv.get("tokenizer.ggml.tokens")
    n_vocab = vocab[1] if isinstance(vocab, tuple) else a("vocab_size", 0) or 0
    kv_layer = ctx * n_head_kv * (k_dim + v_dim) * 2  # f16 K and V
    exps = sum(v for k, v in sizes.items() if re.match(r"^blk\.\d+\.ffn_(up|down|gate|gate_up)_(ch|)exps", k))
    n_expert = a("expert_count", 0)
    n_used = a("expert_used_count", 0)
    nextn = a("nextn_predict_layers", 0)
    return ModelInfo(n_layer, n_embd, n_vocab, blk / n_layer / 2**20, kv_layer / 2**20,
                     out / 2**20, file_size / 2**20,
                     sizes.get("token_embd.weight", 0) / 2**20, "output.weight" not in sizes,
                     n_expert if isinstance(n_expert, int) else 0,
                     n_used if isinstance(n_used, int) else 0,
                     exps / n_layer / 2**20, nextn if isinstance(nextn, int) else 0)


# ── Planning ─────────────────────────────────────────────────────────────────
#
# llama.cpp runs a split model one device after another: per token, every
# device works through its layers in turn, and every device boundary costs a
# network round trip. So adding a slower device never makes it faster; it
# only buys memory. The planner therefore:
#   - fills the fastest devices first (memory bandwidth = token speed),
#   - leaves room on each GPU for its KV cache and compute buffers,
#   - estimates the time per token and, in "auto" mode, drops each worker
#     whose network hop costs more than it saves,
#   - keeps the output head (n_vocab scores per token) on this PC unless
#     shipping the scores is cheaper, so they don't cross the wire.

GPU_RESERVE_MB = 400  # compute buffers per GPU, beyond weights and KV
MTP_RESERVE_MB = 1300  # MTP: second context and draft buffers, on the output device
HOST_RESERVE_MB = 1024
NET_MBPS_PC = 110.0  # 1 Gbit/s Ethernet
NET_MBPS_PHONE = 25.0  # Wi-Fi
HOP_OVERHEAD_S = 0.002  # per worker per token: graph upload, sync (conservative)


@dataclass
class Unit:
    label: str
    capacity_layers: int
    speed: float  # GB/s
    devices: list[str]  # llama.cpp device names; [] for the host CPU
    endpoint: Optional[str] = None  # RPC workers
    rtt_s: float = 0.0
    net_mbps: float = NET_MBPS_PC
    device_caps: list[int] = field(default_factory=list)
    layers: int = 0
    min_layers: int = 0  # e.g. the MTP tail that must stay on this PC
    status_port: int = 0  # worker's monitoring endpoint

    @property
    def remote(self) -> bool:
        return self.endpoint is not None


def fill_fastest_first(units: list[Unit], n_layer: int) -> int:
    """Assigns layers to the fastest units first; returns layers left over."""
    left = n_layer
    for u in units:
        u.layers = min(u.min_layers, u.capacity_layers, left)
        left -= u.layers
    for u in sorted(units, key=lambda u: -u.speed):
        more = min(u.capacity_layers - u.layers, left)
        u.layers += more
        left -= more
    return left


def token_time(units: list[Unit], m: ModelInfo, out_unit: Optional[Unit]) -> float:
    t = 0.0
    for u in units:
        if u.layers:
            t += u.layers * m.read_mb() / 1024 / u.speed
            if u.remote:
                t += 2 * u.rtt_s + HOP_OVERHEAD_S + 2 * m.n_embd * 4 / 2**20 / u.net_mbps
    if out_unit is not None:
        t += m.output_mb / 1024 / out_unit.speed
        if out_unit.remote:
            t += m.n_vocab * 4 / 2**20 / out_unit.net_mbps
    return t


def plan(units: list[Unit], m: ModelInfo, prune: bool) -> list[Unit]:
    """Fills fastest first; with prune, drops workers that don't pay off.
    Returns the units that got layers."""
    active = list(units)
    for u in units:
        u.layers = 0
    if fill_fastest_first(active, m.n_layer) > 0:
        return []
    best = token_time(active, m, None)
    improved = prune
    while improved:
        improved = False
        for u in sorted([u for u in active if u.remote and u.layers], key=lambda u: u.speed):
            trial = [x for x in active if x is not u]
            snapshot = {id(x): x.layers for x in active}
            if fill_fastest_first(trial, m.n_layer) == 0:
                t = token_time(trial, m, None)
                if t < best:
                    u.layers, active, best, improved = 0, trial, t, True
                    break
            for x in active:
                x.layers = snapshot[id(x)]
    return [u for u in active if u.layers]


def split_integer(total: int, weights: list[float]) -> list[int]:
    """total split by weights into integers (largest remainder)."""
    s = sum(weights)
    if total <= 0 or s <= 0:
        return [0] * len(weights)
    raw = [total * w / s for w in weights]
    out = [int(r) for r in raw]
    for i in sorted(range(len(raw)), key=lambda i: out[i] - raw[i])[: total - sum(out)]:
        out[i] += 1
    return out


def measure_rtt(endpoint: str) -> float:
    host, _, port = endpoint.rpartition(":")
    best = 0.05
    for _ in range(3):
        t0 = time.perf_counter()
        try:
            with socket.create_connection((host, int(port)), timeout=2):
                pass
        except OSError:
            return 0.05
        best = min(best, time.perf_counter() - t0)
    return best


def list_devices(rpc: list[str]) -> list[tuple[str, str, int]]:
    """(name, description, free MiB) of the devices llama-server would use,
    in its order: RPC workers first, then local GPUs (llama.cpp puts RPC
    devices in front to keep the output layer, and its large logits, local).
    Free memory of RPC devices comes from the workers themselves."""
    cmd = [find_bin("llama-server")]
    if rpc:
        cmd += ["--rpc", ",".join(rpc)]
    out = subprocess.run(cmd + ["--list-devices"], capture_output=True, text=True, timeout=120).stdout
    devs = [(n, d, int(f)) for n, d, f in
            re.findall(r"^\s+(\S+): (.*) \(\d+ MiB, (\d+) MiB free\)$", out, re.M)]
    return [d for d in devs if d[0].startswith("RPC")] + [d for d in devs if not d[0].startswith("RPC")]


def _on_worker(name: str, description: str, endpoint: str) -> bool:
    """Whether a listed device belongs to the RPC worker at endpoint. RPC
    devices are named RPC<n> per worker, with the endpoint in the
    description (or, in some versions, in the name: RPC<n>[host:port])."""
    return name.startswith("RPC") and (
        name.endswith(f"[{endpoint}]") or description.split(" ")[0] == endpoint)


def reachable(endpoint: str, timeout: float = 2.0) -> bool:
    """llama.cpp aborts on a worker it can't reach, so check first."""
    host, _, port = endpoint.rpartition(":")
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except (OSError, ValueError):
        return False


# ── Commands ─────────────────────────────────────────────────────────────────


def cmd_detect(args) -> None:
    devices = probe(bench=True, refresh=True)
    print(f"{'Device':<10} {'Backend':<8} {'Type':<6} {'Free':>9} {'Total':>9} {'Bandwidth':>11}  Description")
    for d in devices:
        print(f"{d.name:<10} {d.backend:<8} {d.type:<6} {d.free_mb:>6} MB {d.total_mb:>6} MB "
              f"{d.bandwidth:>7.1f} GB/s  {d.description}")
    print(f"\nHost RAM available: {host_available_mb()} MB · worker ID {machine_id()} · {RPC_VERSION}")


def pick_worker_devices(devices: list[Device], wanted: Optional[str]) -> list[Device]:
    if wanted:
        names = [n.strip() for n in wanted.split(",") if n.strip()]
        by_name = {d.name: d for d in devices}
        missing = [n for n in names if n not in by_name]
        if missing:
            die(f"Unknown device(s) {', '.join(missing)}. Have: {', '.join(by_name)}")
        return [by_name[n] for n in names]
    return usable_gpus(devices) or [d for d in devices if d.type == "cpu"][:1]


def cmd_worker(args) -> None:
    devices = probe(bench=not args.no_bench)
    chosen = pick_worker_devices(devices, args.devices)
    if not chosen:
        die("No usable device found")
    gpu = any(d.is_gpu for d in chosen)
    # What the master may fill: free VRAM, or free RAM for a CPU worker.
    ram_mb = sum(d.free_mb for d in chosen) if gpu else _offered_ram(args)
    # Layers on one worker run one after another on its devices, so the
    # slowest device sets the pace.
    bw = min((d.bandwidth for d in chosen if d.bandwidth > 0), default=0.0)
    address = args.advertise_ip or (args.host if args.host not in ("0.0.0.0", "") else lan_address())
    label = chosen[0].description + (f" x{len(chosen)}" if len(chosen) > 1 else "")
    host = socket.gethostname()
    wid = machine_id()

    log(f"Serving {', '.join(f'{d.name} ({d.description}, {d.free_mb} MB free)' for d in chosen)}")
    log(f"Offering {ram_mb} MB at {bw:.1f} GB/s on {address}:{args.port}")
    procs = [_rpc_server(args, args.port, [d.name for d in chosen])]
    rpc_info = [{"port": args.port, "kind": "gpu" if gpu else "cpu", "offered_mb": ram_mb, "bw": bw,
                 "devices": [{"name": d.name, "description": d.description, "free_mb": d.free_mb,
                              "bandwidth": d.bandwidth} for d in chosen]}]
    advs: list[Advertiser] = []
    txt = {"ram": str(ram_mb), "bw": f"{bw:.1f}", "model": f"{host} · {label}", "v": RPC_VERSION,
           "id": wid, "kind": "gpu" if gpu else "cpu",
           "backend": ",".join(sorted({d.backend for d in chosen}))}
    if not args.no_status:
        txt["stat"] = str(args.status_port)
    _advertise(args, advs, f"PrivateLM {host}", args.port, txt, address)
    show_qr(worker_qr(address, args.port, f"{host} · {label}", ram_mb, bw, wid, txt["kind"]))

    # The PC's RAM as a second, separate worker: masters weigh each worker
    # by its own speed, so slow RAM never drags down the GPU's share, and
    # phones (which split a worker evenly across its devices) see it as one
    # more device of its own.
    if args.ram:
        cpu = next((d for d in devices if d.type == "cpu"), None)
        if not gpu or cpu is None:
            log("--ram: this worker already serves its CPU/RAM, skipping the second worker")
        else:
            ram_port = args.ram_port or args.port + 1
            offered = _offered_ram(args)
            log(f"Offering {offered} MB RAM at {cpu.bandwidth:.1f} GB/s on {address}:{ram_port}")
            procs.append(_rpc_server(args, ram_port, ["CPU"]))
            rpc_info.append({"port": ram_port, "kind": "cpu", "offered_mb": offered, "bw": cpu.bandwidth,
                             "devices": [{"name": "CPU", "description": cpu.description,
                                          "free_mb": offered, "bandwidth": cpu.bandwidth}]})
            rtxt = {"ram": str(offered), "bw": f"{cpu.bandwidth:.1f}", "model": f"{host} · RAM",
                    "v": RPC_VERSION, "id": f"{wid}-ram", "kind": "cpu", "backend": "CPU"}
            if not args.no_status:
                rtxt["stat"] = str(args.status_port)
            _advertise(args, advs, f"PrivateLM {host} RAM", ram_port, rtxt, address)
            show_qr(worker_qr(address, ram_port, f"{host} · RAM", offered, cpu.bandwidth,
                              f"{wid}-ram", "cpu"))
    if not args.no_status:
        start_worker_status(args, rpc_info, procs, devices)
    _wait(procs, advs)


def _offered_ram(args) -> int:
    """Free RAM this PC lends, minus what it keeps for itself."""
    mb = max(0, host_available_mb() - int(args.ram_reserve * 1024))
    if args.ram_limit:
        mb = min(mb, int(args.ram_limit * 1024))
    return mb


def _rpc_server(args, port: int, devices: list[str]) -> subprocess.Popen:
    cmd = [find_bin("ggml-rpc-server"), "-H", args.host, "-p", str(port), "-d", ",".join(devices)]
    if args.cache:
        cmd.append("-c")
    if args.threads:
        cmd += ["-t", str(args.threads)]
    return subprocess.Popen(cmd)


def _advertise(args, advs: list, name: str, port: int, txt: dict, address: str) -> None:
    if args.no_advertise:
        return
    adv = Advertiser(WORKER_TYPE, name, port, txt, address)
    if adv.start():
        advs.append(adv)
        log(f"Advertised {name} on the LAN (phones: Phone cluster > Find nearby)")


def _find_workers(args) -> list[Peer]:
    workers: list[Peer] = []
    if args.workers not in ("off", "auto", "all"):
        # Workers by address; their speed and kind still come from mDNS
        # when they advertise (a GPU worker is far faster than the default).
        known = {p.endpoint: p for p in scan(min(args.scan_timeout, 3.0)) if p.kind == "worker"}
        for ep in args.workers.split(","):
            h, _, p = ep.strip().rpartition(":")
            peer = Peer("worker", ep.strip(), h or ep.strip(), int(p) if h else DEFAULT_RPC_PORT)
            if peer.endpoint in known:
                peer.txt = dict(known[peer.endpoint].txt)
                peer.name = known[peer.endpoint].name
            workers.append(peer)
    elif args.workers != "off":
        log("Looking for workers on the LAN...")
        me = machine_id()
        for p in scan(args.scan_timeout):
            if p.kind != "worker" or p.txt.get("id") == me:
                continue
            if p.txt.get("v", RPC_VERSION) != RPC_VERSION:
                log(f"Skipping {p.name}: protocol {p.txt.get('v')} (this node speaks {RPC_VERSION})")
                continue
            workers.append(p)
    for w in list(workers):
        if not reachable(w.endpoint):
            log(f"Worker {w.endpoint} not reachable, leaving it out")
            workers.remove(w)
    return workers


def _remote_units(workers: list[Peer], per_layer: float) -> list[Unit]:
    """One unit per worker, with room in layers per device as the workers
    report their free memory over RPC."""
    reported = list_devices([w.endpoint for w in workers]) if workers else []
    remote: list[Unit] = []
    for w in workers:
        devs_w = [(n, f) for n, d, f in reported if _on_worker(n, d, w.endpoint)]
        if not devs_w:
            log(f"Worker {w.endpoint} reported no devices, leaving it out")
            continue
        if w.is_gpu:
            caps = [int((f * GPU_SHARE - GPU_RESERVE_MB) // per_layer) for _, f in devs_w]
        else:
            offered = sum(f for _, f in devs_w)
            if w.ram_mb:
                offered = min(offered, w.ram_mb)
            budget = offered * (PC_RAM_SHARE if w.txt.get("kind") == "cpu" else PHONE_SHARE)
            total_f = sum(f for _, f in devs_w) or 1
            caps = [int(budget * f / total_f // per_layer) for _, f in devs_w]
        caps = [max(0, c) for c in caps]
        u = Unit(w.txt.get("model") or w.name, sum(caps), w.bandwidth or UNKNOWN_PHONE_SPEED, [],
                 endpoint=w.endpoint, rtt_s=measure_rtt(w.endpoint),
                 net_mbps=NET_MBPS_PC if w.txt.get("kind") else NET_MBPS_PHONE, device_caps=caps,
                 status_port=int(w.txt.get("stat") or STATUS_PORT))
        log(f"Worker {u.label} ({w.endpoint}): room for {u.capacity_layers} layers, "
            f"{u.speed:.0f} GB/s, {u.rtt_s*1000:.1f} ms round trip")
        remote.append(u)
    return remote


@dataclass
class MoePlan:
    counts: list[int]  # layers per local GPU, registry order
    n_cpu_moe: int  # experts of the first N layers stay in RAM
    time_s: float


def moe_local_plan(m: ModelInfo, gpus: list[Device], budgets: list[float], host_budget: float,
                   host_speed: float) -> Optional[MoePlan]:
    """MoE on this PC alone: attention, shared weights and KV cache of every
    layer on the GPU(s), as many layers' experts as fit too, the rest of the
    experts in RAM. Only the experts that run are read per token, so RAM
    costs little; workers would add a network hop for nothing."""
    if not gpus or not m.is_moe:
        return None
    dense = m.layer_mb - m.expert_mb + m.kv_layer_mb
    for k in range(m.n_layer + 1):
        counts = [0] * len(gpus)
        used = [0.0] * len(gpus)
        g = 0
        ok = True
        for il in range(m.n_layer):
            need = dense + (m.expert_mb if il >= k else 0)
            while g < len(gpus) and used[g] + need > budgets[g]:
                g += 1
            if g == len(gpus):
                ok = False
                break
            used[g] += need
            counts[g] += 1
        if not ok:
            continue
        if k * m.expert_mb + m.embd_mb > host_budget:
            return None
        active = m.expert_mb * max(1, m.n_expert_used) / m.n_expert
        t = 0.0
        il = 0
        for gi, c in enumerate(counts):
            sp = gpus[gi].bandwidth or 100.0
            for _ in range(c):
                t += (m.layer_mb - m.expert_mb) / 1024 / sp
                t += active / 1024 / (host_speed if il < k else sp)
                il += 1
        t += m.output_mb / 1024 / (gpus[-1].bandwidth or 100.0)
        return MoePlan(counts, k, t)
    return None


@dataclass
class ServerPlan:
    cmd: list[str]
    name: str
    key: str
    estimate_tps: float
    info: dict = field(default_factory=dict)  # the plan, for the dashboard


def build_server(args, quiet: bool = False) -> ServerPlan:
    """Plans the split and returns the llama-server command."""
    say = (lambda *a, **k: None) if quiet else print
    model = str(Path(args.model).expanduser().resolve())
    if not Path(model).is_file():
        die(f"No such model: {model}")
    m = gguf_info(model, args.ctx)
    if args.mtp and not m.n_nextn:
        die("This model has no MTP layers (nextn_predict_layers = 0). Use a model with MTP, "
            "or leave out --mtp.")
    per_layer = m.per_layer_mb()
    logits_mb = m.n_vocab * 4 * 512 / 2**20  # output buffer for a 512-token batch
    out_extra = m.output_mb + logits_mb + (MTP_RESERVE_MB if args.mtp else 0)

    devices = probe(bench=True)
    local_gpus = usable_gpus(devices)
    cpu = next((d for d in devices if d.type == "cpu"), None)
    host_speed = cpu.bandwidth if cpu and cpu.bandwidth else 20.0

    # The output head (and the MTP draft layers, which sit at the tail)
    # stays on this PC: on the last local GPU in llama.cpp's order that has
    # room, so the tail of the model ends here; else in RAM.
    out_gpu = None
    for d in reversed(local_gpus):
        if d.free_mb * GPU_SHARE - GPU_RESERVE_MB >= out_extra + (m.n_nextn * per_layer if args.mtp else 0):
            out_gpu = d
            break
    budgets = [d.free_mb * GPU_SHARE - GPU_RESERVE_MB - (out_extra if d is out_gpu else 0)
               for d in local_gpus]
    gpus = [Unit(f"{d.name} {d.description}", max(0, int(b // per_layer)), d.bandwidth or 100.0,
                 [d.name], device_caps=[max(0, int(b // per_layer))],
                 min_layers=m.n_nextn if (args.mtp and d is out_gpu) else 0)
            for d, b in zip(local_gpus, budgets)]
    host_budget = host_available_mb() * HOST_SHARE - HOST_RESERVE_MB
    if args.ram_limit is not None:
        host_budget = min(host_budget, args.ram_limit * 1024)
    host_room = host_budget - m.embd_mb - (0 if out_gpu else out_extra)
    host = Unit("This PC (RAM)", max(0, int(host_room // per_layer)), host_speed, [])

    # MoE on this PC alone (experts in RAM), if it fits and isn't slower.
    moe = None
    if m.is_moe and out_gpu is not None and out_gpu is local_gpus[-1] and args.workers in ("auto", "off"):
        # experts not on the GPU live in RAM: the RAM budget is for them
        moe = moe_local_plan(m, local_gpus, budgets, host_budget, host_speed)

    workers = _find_workers(args) if not (moe and args.workers == "off") else []
    remote = _remote_units(workers, per_layer)
    prune = args.workers == "auto"
    used = plan(gpus + remote + [host], m, prune)
    layer_time = token_time(used, m, host if out_gpu is None else gpus[local_gpus.index(out_gpu)]) \
        if used else float("inf")

    # MoE that fits here (experts in RAM) stays here in auto mode: measured
    # setups show MoE over RPC about even on generation, since only a few
    # experts run per token; the time model would overrate the workers.
    if moe:
        if used and any(u.remote and u.layers for u in used):
            log(f"MoE fits on this PC: staying here (est. {1/moe.time_s:.1f} tokens/s; with workers the "
                f"estimate is {1/layer_time:.1f}). Compare for real: bench with --workers off and all.")
        return _moe_command(args, model, m, local_gpus, moe, say)
    if not used:
        total = sum(u.capacity_layers for u in gpus + remote + [host])
        die(f"Model needs {m.n_layer} layers of ~{per_layer:.0f} MB (weights + KV cache at "
            f"-c {args.ctx}); all devices together have room for {total}. "
            "Add workers, lower --ctx, or use a smaller quantization.")
    dropped = [u for u in remote if not u.layers]
    if dropped and prune:
        log("Not using (slower with them than without): " + ", ".join(u.label for u in dropped)
            + ". --workers all includes every worker that has room.")
    used_remote = [u for u in remote if u.layers]
    out_unit: Unit = next((g for g in gpus if out_gpu and out_gpu.name in g.devices), host)
    if out_gpu is None and used_remote and not args.mtp:
        # No GPU here: a worker GPU may compute the head faster even after
        # shipping the scores back.
        best_r = max(used_remote, key=lambda u: u.speed)
        here = m.output_mb / 1024 / host.speed
        there = m.output_mb / 1024 / best_r.speed + m.n_vocab * 4 / 2**20 / best_r.net_mbps
        spare = (best_r.capacity_layers - best_r.layers) * per_layer
        if there < here and spare >= m.output_mb + logits_mb:
            out_unit = best_r
    if args.mtp and out_gpu is None:
        log("MTP without a GPU on this PC: the draft layers run on the last worker, "
            "which costs extra network round trips.")
    used_remote.sort(key=lambda u: u is out_unit)  # the worker holding the head goes last
    rpc = [u.endpoint for u in used_remote]

    devs = list_devices(rpc)
    local_names = {d.name for d in local_gpus}
    devs = [d for d in devs if d[0].startswith("RPC") or d[0] in local_names]
    counts: list[int] = []
    owner_of: list[Optional[Unit]] = []
    seen: dict[str, int] = {}
    for name, desc, _free in devs:
        if name.startswith("RPC"):
            u = next((u for u in used_remote if _on_worker(name, desc, u.endpoint)), None)
            if u is None:
                counts.append(0)
                owner_of.append(None)
                continue
            k = seen.get(u.endpoint, 0)
            seen[u.endpoint] = k + 1
            shares = split_integer(u.layers, [float(c) for c in u.device_caps])
            counts.append(shares[k] if k < len(shares) else 0)
            owner_of.append(u)
        else:
            u = next((g for g in gpus if name in g.devices), None)
            counts.append(u.layers if u else 0)
            owner_of.append(u)
    for u in used_remote:
        if seen.get(u.endpoint, 0) != len(u.device_caps):
            die(f"Worker {u.endpoint} changed its devices while planning; start again.")

    # llama.cpp offloads the last -ngl of n_layer + 1 slots, the output layer
    # being the last slot: whenever any layer is offloaded, the output layer
    # goes with it, to the last device. So it gets its own slot there, and
    # -ot pins its weights (and with them its matmul) where it belongs.
    segments = []
    if host.layers:
        segments.append({"machine": "master", "device": "RAM (CPU)", "layers": host.layers})
    for (name, desc, _f), c, own in zip(devs, counts, owner_of):
        if not c:
            continue
        if name.startswith("RPC") and own is not None:
            segments.append({"machine": own.endpoint.rsplit(":", 1)[0], "device": own.label, "layers": c})
        else:
            segments.append({"machine": "master", "device": own.label if own else name, "layers": c})
    n_off = sum(counts)
    extra: list[str] = []
    ngl = 0
    if n_off:
        ngl = n_off + 1
        last = max(i for i, c in enumerate(counts) if c > 0)
        counts[last] += 1
        pattern = "^token_embd\\.weight$" if m.tied else "^output\\.weight$"
        if out_unit is host:
            extra += ["-ot", f"{pattern}=CPU"]
        elif out_unit is not owner_of[last] and out_gpu is not None and not out_unit.remote:
            extra += ["-ot", f"{pattern}={out_gpu.name}"]

    est = token_time(used, m, out_unit)
    hops = len(used_remote)
    net_kb = hops * (2 * m.n_embd * 4) / 1024 + (m.n_vocab * 4 / 1024 if out_unit.remote else 0)
    say("\nPlan (layers split across devices):")
    for u in sorted(used, key=lambda u: -u.speed):
        say(f"  {u.label:<42} {u.layers:>3} layers  {u.layers * per_layer / 1024:5.1f} GB  "
            f"({u.speed:.0f} GB/s{', %.1f ms RTT' % (u.rtt_s * 1000) if u.remote else ''})")
    where = "this PC's RAM" if out_unit is host else out_unit.label
    say(f"  Output head: {where}" + ("  (scores cross the network every token)" if out_unit.remote else ""))
    _say_model(say, m, per_layer, args)
    if moe:
        say(f"  (MoE on this PC alone would give ~{1/moe.time_s:.1f} tokens/s)")
    say(f"  Estimate: ~{1/est:.1f} tokens/s{' before MTP' if args.mtp else ''}, {hops} network hop(s), "
        f"~{net_kb:.0f} KB/token on the wire (check with: privatelm-node bench)\n")

    cmd = [find_bin("llama-server"), "-m", model, "-c", str(args.ctx), "--fit", "off", "-ngl", str(ngl)]
    if any(counts):
        cmd += ["--tensor-split", ",".join(str(c) for c in counts)]
    cmd += extra
    if rpc:
        cmd += ["--rpc", ",".join(rpc)]
    info = _plan_info(args, model, m, "layers", 1 / est, hops, net_kb, segments, used_remote,
                      {"machine": "master" if not out_unit.remote else out_unit.endpoint.rsplit(":", 1)[0],
                       "label": where})
    return _finish_command(args, cmd, model, 1 / est, info)


def _plan_info(args, model: str, m: ModelInfo, mode: str, tps: float, hops: int, net_kb: float,
               segments: list[dict], used_remote: list[Unit], output: dict, moe_cpu_layers: int = 0) -> dict:
    machines = [{"id": "master", "role": "master", "host": socket.gethostname(),
                 "layers": sum(s["layers"] for s in segments if s["machine"] == "master")}]
    for u in used_remote:
        hid = u.endpoint.rsplit(":", 1)[0]
        mach = next((x for x in machines if x["id"] == hid), None)
        if mach is None:
            mach = {"id": hid, "role": "worker", "host": hid, "label": u.label, "endpoints": [],
                    "status_port": u.status_port, "layers": 0, "rtt_ms": round(u.rtt_s * 1000, 2)}
            machines.append(mach)
        mach["endpoints"].append(u.endpoint)
        mach["layers"] += u.layers
    return {"model": Path(model).stem, "n_layer": m.n_layer, "kind": "moe" if m.is_moe else "dense",
            "n_expert": m.n_expert, "n_expert_used": m.n_expert_used, "mode": mode, "mtp": args.mtp,
            "ctx": args.ctx, "estimate_tps": round(tps, 1), "hops": hops, "net_kb_est": round(net_kb),
            "per_layer_mb": round(m.per_layer_mb()), "size_gb": round(m.size_mb / 1024, 1),
            "segments": segments, "output": output, "moe_cpu_layers": moe_cpu_layers,
            "machines": machines}


def _say_model(say, m: ModelInfo, per_layer: float, args) -> None:
    kind = (f"MoE, {m.n_expert_used} of {m.n_expert} experts per token" if m.is_moe else "dense")
    say(f"  Model: {m.n_layer} layers ({kind}), {per_layer:.0f} MB each incl. KV cache for -c {args.ctx}"
        + (f"; MTP depth {args.mtp}" if args.mtp else (f"; has MTP (try --mtp 1)" if m.n_nextn else "")))


def _moe_command(args, model: str, m: ModelInfo, gpus: list[Device], moe: MoePlan, say) -> ServerPlan:
    say("\nPlan (MoE on this PC: experts that don't fit on the GPU stay in RAM):")
    for d, c in zip(gpus, moe.counts):
        if c:
            say(f"  {d.name} {d.description:<36} {c:>3} layers ({d.bandwidth:.0f} GB/s)")
    if moe.n_cpu_moe:
        say(f"  Experts of the first {moe.n_cpu_moe} layers: this PC's RAM "
            f"({moe.n_cpu_moe * m.expert_mb / 1024:.1f} GB)")
    say(f"  Output head: {gpus[-1].name}")
    _say_model(say, m, m.per_layer_mb(), args)
    say(f"  Estimate: ~{1/moe.time_s:.1f} tokens/s{' before MTP' if args.mtp else ''}, no network hops\n")
    segments = [{"machine": "master", "device": f"{d.name} {d.description}", "layers": c}
                for d, c in zip(gpus, moe.counts) if c]
    info = _plan_info(args, model, m, "moe", 1 / moe.time_s, 0, 0, segments, [],
                      {"machine": "master", "label": gpus[-1].name}, moe.n_cpu_moe)
    counts = list(moe.counts)
    counts[-1] += 1  # output slot on the last GPU
    cmd = [find_bin("llama-server"), "-m", model, "-c", str(args.ctx), "--fit", "off",
           "-ngl", str(m.n_layer + 1)]
    if len(gpus) > 1:
        cmd += ["--tensor-split", ",".join(str(c) for c in counts)]
    if moe.n_cpu_moe:
        cmd += ["--n-cpu-moe", str(moe.n_cpu_moe)]
    return _finish_command(args, cmd, model, 1 / moe.time_s, info)


def _finish_command(args, cmd: list[str], model: str, tps: float, info: Optional[dict] = None) -> ServerPlan:
    cmd += ["--metrics"]  # for the dashboard
    if args.mtp:
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(args.mtp)]
    key = args.api_key if args.api_key is not None else _api_key()
    cmd += ["--host", args.host, "--port", str(args.port)]
    if key:
        cmd += ["--api-key", key]
    cmd += args.extra
    return ServerPlan(cmd, Path(model).stem, key, tps, info or {})


def cmd_serve(args) -> None:
    sp = build_server(args)
    log(" ".join(sp.cmd).replace(sp.key, "***") if sp.key else " ".join(sp.cmd))
    proc = subprocess.Popen(sp.cmd)

    address = args.advertise_ip or (args.host if args.host not in ("0.0.0.0", "") else lan_address())
    adv = None
    if not args.no_advertise:
        adv = Advertiser(SERVER_TYPE, f"PrivateLM {socket.gethostname()}", args.port,
                         {"path": "/v1", "model": sp.name, "auth": "1" if sp.key else "0", "v": "1"}, address)
        adv.start()
    from urllib.parse import urlencode

    params = {"url": f"http://{address}:{args.port}/v1", "model": sp.name, "name": socket.gethostname()}
    if sp.key:
        params["key"] = sp.key
    print(f"\nOpenAI-compatible API: http://{address}:{args.port}/v1  (model: {sp.name})")
    if not args.no_dashboard:
        dash = Dashboard(args, sp, proc, probe(bench=False))
        if dash.start():
            print(f"Dashboard:             http://{address}:{args.dashboard_port}/"
                  + ("  (log in with the API key)" if sp.key else ""))
    show_qr(f"privatelm://server?{urlencode(params)}")
    _wait(proc, adv)


def cmd_tune_mtp(args) -> None:
    """Starts the server once per MTP depth and measures it at the
    temperature you actually use: a depth tuned at temperature 0 (what most
    benchmarks run) can be slower at temperature 1, because fewer drafted
    tokens are accepted."""
    depths = [int(x) for x in args.depths.split(",")]
    args.host, args.no_advertise = "127.0.0.1", True
    if any(depths) and not gguf_info(str(Path(args.model).expanduser()), args.ctx).n_nextn:
        die("This model has no MTP layers (nextn_predict_layers = 0); MTP needs a model that ships them.")
    results = []
    for depth in depths:
        args.mtp = depth
        sp = build_server(args, quiet=depth != depths[0])
        base = f"http://127.0.0.1:{args.port}"
        log(f"MTP depth {depth}: starting server...")
        proc = subprocess.Popen(sp.cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            if not _wait_healthy(base, proc, timeout=args.load_timeout):
                log(f"depth {depth}: server did not start (try it with privatelm-node serve --mtp {depth})")
                continue
            r = measure(base, sp.key, None, args.temp, args.tokens, args.context, runs=args.runs)
            results.append((depth, r))
            acc = f", accepted {r['acceptance']*100:.0f}%" if r.get("acceptance") is not None else ""
            log(f"depth {depth}: {r['tps']:.1f} tokens/s{acc}")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
    if not results:
        die("No depth could be measured.")
    best = max(results, key=lambda x: x[1]["tps"])
    print(f"\nMTP at temperature {args.temp}, context ~{args.context} tokens:")
    for depth, r in results:
        acc = f"{r['acceptance']*100:5.0f}%" if r.get("acceptance") is not None else "    -"
        print(f"  depth {depth}: {r['tps']:6.1f} tokens/s   accepted {acc}" + ("   <- best" if (depth, r) == best else ""))
    print(f"\nUse: privatelm-node serve -m {args.model} --mtp {best[0]}" if best[0] else
          "\nMTP doesn't pay off at this temperature: serve without --mtp.")


def _wait_healthy(base: str, proc: subprocess.Popen, timeout: float) -> bool:
    import urllib.request

    t_end = time.time() + timeout
    while time.time() < t_end:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(base + "/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(2)
    return False


def cmd_scan(args) -> None:
    peers = scan(args.timeout)
    if not peers:
        print("Nothing found. mDNS may be blocked (guest Wi-Fi, client isolation, VPN).")
        return
    for p in peers:
        if p.kind == "worker":
            kind = p.txt.get("kind", "phone")
            note = "" if p.txt.get("v") == RPC_VERSION else f"  !! protocol {p.txt.get('v')}"
            print(f"worker  {p.endpoint:<22} {kind:<5} {p.ram_mb or '?':>6} MB "
                  f"{p.bandwidth or 0:>6.1f} GB/s  {p.txt.get('model', p.name)}{note}")
        else:
            print(f"server  {p.endpoint:<22} http://{p.endpoint}{p.txt.get('path', '/v1')}  "
                  f"model {p.txt.get('model', '?')}")


def _net_bytes(ifaces: Optional[list[str]]) -> tuple[int, int]:
    """(received, sent) bytes over all interfaces but loopback, or the given
    ones. Both directions: a one-way counter misses the scores coming back."""
    rx = tx = 0
    for line in Path("/proc/net/dev").read_text().splitlines()[2:]:
        name, data = line.split(":", 1)
        name = name.strip()
        if (ifaces and name not in ifaces) or (not ifaces and name == "lo"):
            continue
        f = data.split()
        rx += int(f[0])
        tx += int(f[8])
    return rx, tx


def _http(base: str, key: str):
    import urllib.request

    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    def call(path: str, body: Optional[dict] = None) -> dict:
        import urllib.error

        req = urllib.request.Request(base + path, headers=headers,
                                     data=json.dumps(body).encode() if body is not None else None)
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            if e.code == 401:
                die("The server wants an API key: --api-key KEY (privatelm-node serve printed it)")
            die(f"Server answered {e.code} on {path}: {detail}")

    return call


FILLER = ("The lighthouse keeper wrote in his log every evening, noting the weather, the ships "
          "that passed, the state of the lamp and the small events of the island. ")


def measure(base: str, key: str, ifaces: Optional[list[str]], temp: float, tokens: int,
            context: int, runs: int = 1) -> dict:
    """Generation speed and network traffic per token (both directions:
    the difference between a 1-token and an N-token run, so the prompt's
    traffic drops out). context = rough prompt length in tokens: a full
    context is slower, and that's where workers' extra memory pays off."""
    call = _http(base, key)
    try:
        meta = (call("/v1/models").get("data") or [{}])[0].get("meta") or {}
    except OSError as e:
        die(f"No server at {base}: {e}")
    tail = "\nContinue the log with a new entry:"
    prompt = tail
    if context > 0:
        try:
            n_ctx = call("/props").get("default_generation_settings", {}).get("n_ctx", 0)
        except SystemExit:
            n_ctx = 0
        if n_ctx:
            room = n_ctx - tokens - 64
            if context > room:
                log(f"--context {context} doesn't fit the server's {n_ctx}-token context with "
                    f"{tokens} new tokens; using {max(0, room)}")
                context = max(0, room)
        per = len(call("/tokenize", {"content": FILLER}).get("tokens", [])) or 30
        prompt = FILLER * max(0, context // per) + tail

    def run(n: int) -> tuple[dict, int, int]:
        rx0, tx0 = _net_bytes(ifaces)
        res = call("/completion", {"prompt": prompt, "n_predict": n, "ignore_eos": True,
                                   "cache_prompt": False, "temperature": temp, "seed": 42})
        rx1, tx1 = _net_bytes(ifaces)
        return res.get("timings", {}), rx1 - rx0, tx1 - tx0

    run(4)  # warm-up
    _, rx1, tx1 = run(1)
    tps, per_rx, per_tx, drafted, accepted, n = [], 0.0, 0.0, 0, 0, tokens
    for _ in range(max(1, runs)):
        t, rxn, txn = run(tokens)
        n = t.get("predicted_n") or tokens
        tps.append(t.get("predicted_per_second", 0.0))
        per_rx += max(0, rxn - rx1) / max(1, n - 1)
        per_tx += max(0, txn - tx1) / max(1, n - 1)
        drafted += t.get("draft_n", 0) or 0
        accepted += t.get("draft_n_accepted", 0) or 0
    k = max(1, runs)
    return {"tps": sum(tps) / len(tps), "per_rx": per_rx / k, "per_tx": per_tx / k, "n": n,
            "prompt_tokens": None, "acceptance": (accepted / drafted) if drafted else None,
            "n_vocab": meta.get("n_vocab", 0), "n_embd": meta.get("n_embd", 0)}


def cmd_bench(args) -> None:
    base = args.url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    key = args.api_key if args.api_key is not None else (
        _api_key() if (CONFIG_DIR / "api_key").exists() else "")
    ifaces = args.iface.split(",") if args.iface else None
    log(f"Measuring at temperature {args.temp}, context ~{args.context} tokens...")
    r = measure(base, key, ifaces, args.temp, args.tokens, args.context, args.runs)
    per = r["per_rx"] + r["per_tx"]
    print(f"Generation:   {r['tps']:.1f} tokens/s ({r['n']} tokens, temperature {args.temp}, "
          f"context ~{args.context})")
    if r["acceptance"] is not None:
        print(f"MTP/draft:    {r['acceptance']*100:.0f}% of drafted tokens accepted "
              "(tune the depth at this temperature: privatelm-node tune-mtp)")
    print(f"Network:      {per/1024:.0f} KB per token  (in {r['per_rx']/1024:.0f} KB, out {r['per_tx']/1024:.0f} KB)")
    n_vocab, n_embd = r["n_vocab"], r["n_embd"]
    if not n_vocab:
        return
    logits = n_vocab * 4
    act = max(1, n_embd) * 4
    print(f"Reference:    one hand-off of activations ~{act/1024:.0f} KB, "
          f"output scores {logits/1024:.0f} KB ({n_vocab} tokens x 4 bytes)")
    if per < 2048:
        print("Verdict:      no traffic per token: the model runs on this PC alone.")
    elif r["per_rx"] >= 0.8 * logits:
        print("Verdict:      the output head sits on a worker: its scores come back every token.\n"
              "              Restart with privatelm-node serve (it keeps the head on this PC),\n"
              "              or give the head's device the last place in -dev / --rpc order.")
    elif per > 64 * act + 256 * 1024:
        print("Verdict:      far more traffic than a few hand-offs: layers are probably\n"
              "              interleaved between machines. Check the device order (-dev).")
    else:
        print("Verdict:      fine: activations hand-offs only, the output head is local.")


def _api_key() -> str:
    path = CONFIG_DIR / "api_key"
    if not path.exists():
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_urlsafe(24))
        path.chmod(0o600)
    return path.read_text().strip()


def _wait(procs, advs) -> None:
    """Runs until a child exits or we're stopped, then stops everything."""
    if isinstance(procs, subprocess.Popen):
        procs = [procs]
    if advs is None:
        advs = []
    elif isinstance(advs, Advertiser):
        advs = [advs]
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    code = 0
    try:
        while not stopping:
            done = [p for p in procs if p.poll() is not None]
            if done:
                code = done[0].returncode or 1
                break
            time.sleep(0.5)
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        for a in advs:
            a.stop()
    sys.exit(0 if stopping else code)


# ── Monitoring ───────────────────────────────────────────────────────────────
#
# Workers serve their live stats on a small read-only HTTP endpoint
# (/status, port 50060). The master's dashboard (serve, port 8090) polls
# llama-server's metrics, its own stats and every worker's, keeps ten
# minutes of history and serves dashboard.html plus /api/status.

STATUS_PORT = 50060
DASHBOARD_PORT = 8090
HISTORY_POINTS = 300  # x 2 s = 10 minutes


class SystemStats:
    """CPU, RAM, network and GPU readings of this PC, with rates computed
    between calls."""

    def __init__(self, devices: Optional[list[Device]] = None):
        self.devices = devices or []
        self._cpu = self._read_cpu()
        self._net = (time.time(), _net_bytes(None))

    @staticmethod
    def _read_cpu() -> tuple[int, int]:
        f = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
        vals = [int(x) for x in f]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
        return sum(vals), idle

    def sample(self) -> dict:
        total, idle = self._read_cpu()
        dt, di = total - self._cpu[0], idle - self._cpu[1]
        self._cpu = (total, idle)
        cpu = 100.0 * (1 - di / dt) if dt > 0 else 0.0
        now = time.time()
        rx, tx = _net_bytes(None)
        t0, (rx0, tx0) = self._net
        self._net = (now, (rx, tx))
        span = max(1e-3, now - t0)
        mem = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            if k in ("MemTotal", "MemAvailable"):
                mem[k] = int(v.split()[0]) // 1024
        return {
            "host": socket.gethostname(),
            "cpu_pct": round(cpu, 1),
            "cpu_name": next((d.description for d in self.devices if d.type == "cpu"), ""),
            "load": os.getloadavg()[0],
            "ram_total_mb": mem.get("MemTotal", 0),
            "ram_used_mb": mem.get("MemTotal", 0) - mem.get("MemAvailable", 0),
            "net_rx_kbps": round((rx - rx0) / span / 1024, 1),
            "net_tx_kbps": round((tx - tx0) / span / 1024, 1),
            "net_rx_total": rx,
            "net_tx_total": tx,
            "gpus": self.gpus(),
        }

    def gpus(self) -> list[dict]:
        out: list[dict] = []
        if shutil.which("nvidia-smi"):
            try:
                res = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,"
                     "temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout
                for line in res.strip().splitlines():
                    f = [x.strip() for x in line.split(",")]
                    out.append({"name": f[0], "vendor": "NVIDIA", "util_pct": _num(f[1]),
                                "mem_used_mb": _num(f[2]), "mem_total_mb": _num(f[3]),
                                "temp_c": _num(f[4]), "power_w": _num(f[5])})
            except (OSError, subprocess.SubprocessError, IndexError):
                pass
        names = {d.device_id.lower(): d.description for d in self.devices if d.device_id}
        for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
            dev = card / "device"
            if not (dev / "vendor").exists() or "-" in card.name:
                continue
            vendor = (dev / "vendor").read_text().strip()
            if vendor not in ("0x1002", "0x8086"):
                continue  # NVIDIA comes from nvidia-smi
            slot = ""
            for line in _read(dev / "uevent").splitlines():
                if line.startswith("PCI_SLOT_NAME="):
                    slot = line.split("=", 1)[1].lower()
            g = {"name": names.get(slot, "AMD GPU" if vendor == "0x1002" else "Intel GPU"),
                 "vendor": "AMD" if vendor == "0x1002" else "Intel",
                 "util_pct": _num(_read(dev / "gpu_busy_percent")),
                 "mem_used_mb": _num(_read(dev / "mem_info_vram_used"), 2**20),
                 "mem_total_mb": _num(_read(dev / "mem_info_vram_total"), 2**20),
                 "temp_c": None, "power_w": None}
            for hw in (dev / "hwmon").glob("hwmon*"):
                g["temp_c"] = _num(_read(hw / "temp1_input"), 1000)
                g["power_w"] = _num(_read(hw / "power1_average"), 1e6)
            if g["mem_total_mb"] or g["util_pct"] is not None:
                out.append(g)
        return out


def _read(p: Path) -> str:
    try:
        return p.read_text().strip()
    except OSError:
        return ""


def _num(s, div: float = 1.0) -> Optional[float]:
    try:
        return round(float(s) / div, 1)
    except (TypeError, ValueError):
        return None


def tcp_peers(port: int) -> list[str]:
    """IPs with an established connection to this local TCP port."""
    peers = []
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        for line in _read(Path(path)).splitlines()[1:]:
            f = line.split()
            if len(f) < 4 or f[3] != "01":
                continue
            local, remote = f[1], f[2]
            if int(local.rsplit(":", 1)[1], 16) != port:
                continue
            ip_hex = remote.rsplit(":", 1)[0]
            if len(ip_hex) == 8:
                peers.append(socket.inet_ntoa(struct.pack("<I", int(ip_hex, 16))))
            elif ip_hex.startswith("0000000000000000FFFF0000"):
                peers.append(socket.inet_ntoa(struct.pack("<I", int(ip_hex[24:], 16))))
    return sorted(set(peers))


def _start_http(host: str, port: int, handler_cls) -> Optional[object]:
    from http.server import ThreadingHTTPServer
    import threading

    try:
        srv = ThreadingHTTPServer((host, port), handler_cls)
    except OSError as e:
        log(f"Port {port} busy ({e}); no web endpoint")
        return None
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _json_handler(get_routes: dict, post_routes: Optional[dict] = None, key: str = "",
                  public: tuple = ()):
    from http.server import BaseHTTPRequestHandler

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _authorized(self, path: str) -> bool:
            if not key or path in public:
                return True
            got = self.headers.get("Authorization", "")
            return secrets.compare_digest(got.encode(), f"Bearer {key}".encode())

        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _route(self, routes: dict) -> None:
            path = self.path.split("?", 1)[0]
            fn = routes.get(path)
            if fn is None:
                return self._send(404, b'{"error":"not found"}')
            if not self._authorized(path):
                return self._send(401, b'{"error":"API key required"}')
            res = fn()
            if isinstance(res, tuple):
                return self._send(200, res[0], res[1])
            self._send(200, json.dumps(res).encode())

        def do_GET(self):
            self._route(get_routes)

        def do_POST(self):
            self._route(post_routes or {})

    return H


def start_worker_status(args, rpc_info: list[dict], procs: list[subprocess.Popen],
                        devices: list[Device]) -> None:
    stats = SystemStats(devices)

    def status() -> dict:
        rpc = []
        for info, p in zip(rpc_info, procs):
            rpc.append(dict(info, alive=p.poll() is None, peers=tcp_peers(info["port"])))
        return {"role": "worker", "host": socket.gethostname(), "version": RPC_VERSION,
                "rpc": rpc, "stats": stats.sample()}

    if _start_http(args.host, args.status_port, _json_handler({"/status": status})):
        log(f"Status for the master's dashboard on port {args.status_port}")


class Dashboard:
    """Collects the cluster's state every 2 s for the master's dashboard."""

    def __init__(self, args, sp: "ServerPlan", proc: subprocess.Popen, devices: list[Device]):
        import threading

        self.args, self.sp, self.proc = args, sp, proc
        self.base = f"http://127.0.0.1:{args.port}"
        self.stats = SystemStats(devices)
        self.started = time.time()
        self.lock = threading.Lock()
        self.history: dict[str, list] = {"t": [], "tps": [], "net_kbps": [], "gpu": {}}
        self.state: dict = {}
        self.workers: dict[str, dict] = {}
        self.lan: list[dict] = []
        self.bench: dict = {"running": False}
        self._last_metrics: Optional[tuple[float, float, int]] = None
        self._last_tps = 0.0
        self._kb_per_token: Optional[float] = None

    # Collection ------------------------------------------------------------

    def _llama(self) -> dict:
        import urllib.request

        out = {"status": "starting", "metrics": {}}
        if self.proc.poll() is not None:
            out["status"] = "stopped"
            return out
        try:
            with urllib.request.urlopen(self.base + "/health", timeout=2) as r:
                out["status"] = "ready" if r.status == 200 else "loading"
        except OSError as e:
            out["status"] = "loading" if "503" in str(e) else "starting"
            return out
        try:
            req = urllib.request.Request(self.base + "/metrics",
                                         headers={"Authorization": f"Bearer {self.sp.key}"} if self.sp.key else {})
            with urllib.request.urlopen(req, timeout=2) as r:
                for line in r.read().decode().splitlines():
                    if line and not line.startswith("#"):
                        k, _, v = line.rpartition(" ")
                        out["metrics"][k.replace("llamacpp:", "")] = _num(v)
        except OSError:
            pass
        return out

    def _worker(self, w: dict) -> dict:
        import urllib.request

        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(f"http://{w['host']}:{w['status_port']}/status", timeout=1.5) as r:
                data = json.loads(r.read())
            return {"online": True, "rtt_ms": round((time.perf_counter() - t0) * 1000, 1), **data}
        except (OSError, ValueError):
            rpc_ok = any(reachable(ep, 1.0) for ep in w["endpoints"])
            return {"online": rpc_ok, "status_missing": rpc_ok, "rtt_ms": None}

    def collect(self) -> None:
        llama = self._llama()
        stats = self.stats.sample()
        m = llama["metrics"]
        now = time.time()
        predicted = m.get("tokens_predicted_total")
        net_total = stats["net_rx_total"] + stats["net_tx_total"]
        tps = 0.0
        if predicted is not None:
            if self._last_metrics:
                t0, p0, n0 = self._last_metrics
                dp = predicted - p0
                if dp > 0:
                    tps = dp / max(1e-3, now - t0)
                    self._last_tps = tps
                    self._kb_per_token = (net_total - n0) / dp / 1024
            self._last_metrics = (now, predicted, net_total)
        workers = {}
        for w in self.sp.info.get("machines", []):
            if w["role"] == "worker":
                workers[w["id"]] = {**w, **self._worker(w)}
        with self.lock:
            self.state = {"llama": llama, "stats": stats, "tps_now": round(tps, 2),
                          "tps_last": round(self._last_tps, 2),
                          "kb_per_token": None if self._kb_per_token is None else round(self._kb_per_token, 1)}
            self.workers = workers
            h = self.history
            h["t"].append(int(now))
            h["tps"].append(round(tps, 2))
            h["net_kbps"].append(round(stats["net_rx_kbps"] + stats["net_tx_kbps"], 1))
            gpu_now = {"master": [g.get("util_pct") for g in stats["gpus"]]}
            for wid, w in workers.items():
                gpu_now[wid] = [g.get("util_pct") for g in (w.get("stats") or {}).get("gpus", [])]
            for k, v in gpu_now.items():
                vals = [x for x in v if x is not None]
                h["gpu"].setdefault(k, [None] * (len(h["t"]) - 1)).append(max(vals) if vals else None)
            for k in h["gpu"]:
                if len(h["gpu"][k]) < len(h["t"]):
                    h["gpu"][k].append(None)
            for k in ("t", "tps", "net_kbps"):
                del h[k][:-HISTORY_POINTS]
            for k in h["gpu"]:
                del h["gpu"][k][:-HISTORY_POINTS]

    def scan_lan(self) -> None:
        used = {ep for w in self.sp.info.get("machines", []) for ep in w.get("endpoints", [])}
        try:
            peers = scan(3.0)
        except Exception:  # noqa: BLE001 - mDNS problems must not stop the dashboard
            return
        lan = [{"kind": p.kind, "name": p.txt.get("model") or p.name, "endpoint": p.endpoint,
                "ram_mb": p.ram_mb, "bw": p.bandwidth, "type": p.txt.get("kind", "phone" if p.kind == "worker" else ""),
                "in_use": p.endpoint in used, "compatible": p.txt.get("v", RPC_VERSION) == RPC_VERSION}
               for p in peers if p.txt.get("id") != machine_id() or p.kind == "server"]
        with self.lock:
            self.lan = lan

    def loop(self) -> None:
        last_scan = 0.0
        while self.proc.poll() is None:
            try:
                self.collect()
                if time.time() - last_scan > 30:
                    last_scan = time.time()
                    import threading
                    threading.Thread(target=self.scan_lan, daemon=True).start()
            except Exception as e:  # noqa: BLE001 - keep monitoring alive
                log(f"Dashboard: {e}")
            time.sleep(2)

    # API ---------------------------------------------------------------------

    def status(self) -> dict:
        with self.lock:
            return {"plan": self.sp.info, "master": {**self.state, "uptime_s": int(time.time() - self.started),
                                                      "host": socket.gethostname()},
                    "workers": self.workers, "lan": self.lan, "history": self.history,
                    "bench": self.bench}

    def run_bench(self) -> dict:
        import threading

        with self.lock:
            if self.bench.get("running"):
                return self.bench
            self.bench = {"running": True, "started": int(time.time())}

        def go():
            try:
                r = measure(self.base, self.sp.key, None, 0.7, 64, 0, 1)
                res = {"running": False, "finished": int(time.time()), "tps": round(r["tps"], 1),
                       "kb_per_token": round((r["per_rx"] + r["per_tx"]) / 1024, 1),
                       "acceptance": r["acceptance"]}
            except SystemExit:
                res = {"running": False, "error": "Messung fehlgeschlagen (Server bereit?)"}
            with self.lock:
                self.bench = res

        threading.Thread(target=go, daemon=True).start()
        return self.bench

    def start(self) -> Optional[str]:
        import threading

        html_path = next((p for p in (BIN_DIR / "dashboard.html", Path(__file__).resolve().parent / "dashboard.html")
                          if p.exists()), None)
        page = html_path.read_bytes() if html_path else b"<p>dashboard.html missing</p>"
        handler = _json_handler({"/": lambda: (page, "text/html; charset=utf-8"),
                                 "/api/status": self.status},
                                {"/api/bench": self.run_bench}, key=self.sp.key, public=("/",))
        if not _start_http(self.args.host, self.args.dashboard_port, handler):
            return None
        threading.Thread(target=self.loop, daemon=True).start()
        return f"{self.args.dashboard_port}"


def main() -> None:
    ap = argparse.ArgumentParser(prog="privatelm-node", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("detect", help="list devices, free memory and bandwidth").set_defaults(fn=cmd_detect)

    w = sub.add_parser("worker", help="lend this PC's GPU(s) to phone/PC masters")
    w.add_argument("--host", default="0.0.0.0", help="address to listen on (default all)")
    w.add_argument("--port", type=int, default=DEFAULT_RPC_PORT)
    w.add_argument("--devices", help="e.g. CUDA0,CUDA1 or Vulkan0 (default: all GPUs)")
    w.add_argument("--threads", type=int, help="CPU threads (CPU worker only)")
    w.add_argument("--cache", action="store_true", help="cache model tensors on disk (faster reloads)")
    w.add_argument("--advertise-ip", help="address announced via mDNS/QR (e.g. the VPN address)")
    w.add_argument("--no-advertise", action="store_true")
    w.add_argument("--no-bench", action="store_true", help="skip the bandwidth test")
    w.add_argument("--ram", action="store_true",
                   help="also lend this PC's RAM, as a second worker on port+1")
    w.add_argument("--ram-port", type=int, help="port of the RAM worker (default: port+1)")
    w.add_argument("--ram-reserve", type=float, default=4.0,
                   help="GB of RAM kept free for this PC (default 4)")
    w.add_argument("--ram-limit", type=float, help="lend at most this many GB of RAM")
    w.add_argument("--status-port", type=int, default=STATUS_PORT,
                   help=f"read-only status for the master's dashboard (default {STATUS_PORT})")
    w.add_argument("--no-status", action="store_true")
    w.set_defaults(fn=cmd_worker)

    s = sub.add_parser("serve", help="run a model here and serve an OpenAI-compatible API")
    s.add_argument("-m", "--model", required=True)
    s.add_argument("--workers", default="auto",
                   help="auto (workers only where they pay off), all (every worker with room), off, or host:port,...")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=DEFAULT_API_PORT)
    s.add_argument("--ctx", type=int, default=8192)
    s.add_argument("--api-key", help="default: generated once, kept in ~/.config/privatelm-node")
    s.add_argument("--mtp", type=int, default=0,
                   help="MTP draft depth (models with MTP layers; find the best with tune-mtp)")
    s.add_argument("--ram-limit", type=float,
                   help="GB of this PC's RAM the model may use (layers that don't fit go to workers)")
    s.add_argument("--dashboard-port", type=int, default=DASHBOARD_PORT,
                   help=f"web dashboard (default {DASHBOARD_PORT}; same API key as the server)")
    s.add_argument("--no-dashboard", action="store_true")
    s.add_argument("--advertise-ip")
    s.add_argument("--no-advertise", action="store_true")
    s.add_argument("--scan-timeout", type=float, default=4.0)
    s.add_argument("extra", nargs=argparse.REMAINDER, help="-- further llama-server options")
    s.set_defaults(fn=cmd_serve)

    b = sub.add_parser("bench", help="measure tokens/s and network traffic per token (run on the master)")
    b.add_argument("--url", default=f"http://127.0.0.1:{DEFAULT_API_PORT}")
    b.add_argument("--api-key", help="default: the key privatelm-node serve created")
    b.add_argument("--tokens", type=int, default=64)
    b.add_argument("--temp", type=float, default=0.7, help="the temperature you actually use (default 0.7)")
    b.add_argument("--context", type=int, default=0, help="fill the prompt with ~N tokens first")
    b.add_argument("--runs", type=int, default=1)
    b.add_argument("--iface", help="count only these interfaces, e.g. eth0 (default: all but lo)")
    b.set_defaults(fn=cmd_bench)

    t = sub.add_parser("tune-mtp", help="find the best MTP depth at your temperature (stop the server first)")
    t.add_argument("-m", "--model", required=True)
    t.add_argument("--depths", default="0,1,2,3")
    t.add_argument("--temp", type=float, default=0.7, help="the temperature you actually use (default 0.7)")
    t.add_argument("--context", type=int, default=0, help="fill the prompt with ~N tokens first")
    t.add_argument("--tokens", type=int, default=128)
    t.add_argument("--runs", type=int, default=2)
    t.add_argument("--workers", default="auto")
    t.add_argument("--ctx", type=int, default=8192)
    t.add_argument("--ram-limit", type=float)
    t.add_argument("--port", type=int, default=18080)
    t.add_argument("--load-timeout", type=float, default=900)
    t.add_argument("--scan-timeout", type=float, default=4.0)
    t.set_defaults(fn=cmd_tune_mtp, api_key="", extra=[], mtp=0)

    sc = sub.add_parser("scan", help="list workers and servers on the LAN")
    sc.add_argument("--timeout", type=float, default=4.0)
    sc.set_defaults(fn=cmd_scan)

    args = ap.parse_args()
    if getattr(args, "extra", None) and args.extra[:1] == ["--"]:
        args.extra = args.extra[1:]
    args.fn(args)


if __name__ == "__main__":
    main()
