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
GPU_SHARE = 0.9
PHONE_SHARE = 0.75
HOST_SHARE = 0.7
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

    @property
    def is_gpu(self) -> bool:
        return self.type in ("gpu", "igpu")


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
                              d["free_mb"], d["total_mb"], bw))
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


def gguf_block_count(path: str) -> int:
    """The model's transformer block count, from its GGUF header."""
    scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            die(f"{path} is not a GGUF file")
        version, _n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
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
                return None
            die(f"Unknown GGUF value type {t}")

        kv = {}
        for _ in range(n_kv):
            key = read_str()
            (t,) = struct.unpack("<I", f.read(4))
            val = read_val(t)
            if key == "general.architecture" or key.endswith(".block_count"):
                kv[key] = val
        arch = kv.get("general.architecture")
        count = kv.get(f"{arch}.block_count")
        if not isinstance(count, int) or count <= 0:
            die("Could not read the model's layer count")
        return count


# ── Planning (mirrors the app's ClusterService) ─────────────────────────────


@dataclass
class Unit:
    label: str
    capacity_mb: float
    speed: float
    devices: list[str]  # llama.cpp device names; [] for the host CPU
    endpoint: Optional[str] = None  # RPC workers
    assigned_mb: float = 0.0


def fill_fastest_first(units: list[Unit], need_mb: float) -> None:
    left = need_mb
    for u in sorted(units, key=lambda u: -u.speed):
        u.assigned_mb = min(u.capacity_mb, left)
        left -= u.assigned_mb


def split_by_speed(units: list[Unit], need_mb: float) -> None:
    open_, left = list(units), need_mb
    while open_ and left > 0.5:
        speed_sum = sum(u.speed for u in open_) or 1.0
        capped = [u for u in open_ if u.assigned_mb + left * u.speed / speed_sum >= u.capacity_mb]
        if not capped:
            for u in open_:
                u.assigned_mb += left * u.speed / speed_sum
            left = 0
        else:
            for u in capped:
                left -= u.capacity_mb - u.assigned_mb
                u.assigned_mb = u.capacity_mb
            open_ = [u for u in open_ if u not in capped]


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
    gpus = [d for d in devices if d.type == "gpu"] or [d for d in devices if d.type == "igpu"]
    return gpus or [d for d in devices if d.type == "cpu"][:1]


def cmd_worker(args) -> None:
    devices = probe(bench=not args.no_bench)
    chosen = pick_worker_devices(devices, args.devices)
    if not chosen:
        die("No usable device found")
    gpu = any(d.is_gpu for d in chosen)
    # What the master may fill: free VRAM, or free RAM for a CPU worker.
    ram_mb = sum(d.free_mb for d in chosen) if gpu else host_available_mb()
    # Layers on one worker run one after another on its devices, so the
    # slowest device sets the pace.
    bw = min((d.bandwidth for d in chosen if d.bandwidth > 0), default=0.0)
    address = args.advertise_ip or (args.host if args.host not in ("0.0.0.0", "") else lan_address())
    label = chosen[0].description + (f" x{len(chosen)}" if len(chosen) > 1 else "")
    host = socket.gethostname()
    wid = machine_id()

    log(f"Serving {', '.join(f'{d.name} ({d.description})' for d in chosen)}")
    log(f"Offering {ram_mb} MB at {bw:.1f} GB/s on {address}:{args.port}")
    cmd = [find_bin("ggml-rpc-server"), "-H", args.host, "-p", str(args.port),
           "-d", ",".join(d.name for d in chosen)]
    if args.cache:
        cmd.append("-c")
    if args.threads:
        cmd += ["-t", str(args.threads)]
    proc = subprocess.Popen(cmd)

    adv = None
    if not args.no_advertise:
        txt = {"ram": str(ram_mb), "bw": f"{bw:.1f}", "model": f"{host} · {label}", "v": RPC_VERSION,
               "id": wid, "kind": "gpu" if gpu else "cpu",
               "backend": ",".join(sorted({d.backend for d in chosen}))}
        adv = Advertiser(WORKER_TYPE, f"PrivateLM {host}", args.port, txt, address)
        if adv.start():
            log("Advertised on the LAN: phones find this PC under Phone cluster > Scan nearby")
    show_qr(worker_qr(address, args.port, f"{host} · {label}", ram_mb, bw, wid, "gpu" if gpu else "cpu"))
    _wait(proc, adv)


def cmd_serve(args) -> None:
    model = str(Path(args.model).expanduser().resolve())
    if not Path(model).is_file():
        die(f"No such model: {model}")
    model_mb = Path(model).stat().st_size / 2**20
    need_mb = model_mb * 1.1  # + KV cache and working buffers
    n_layer = gguf_block_count(model)

    devices = probe(bench=True)
    local_gpus = [d for d in devices if d.type == "gpu"] or [d for d in devices if d.type == "igpu"]
    cpu = next((d for d in devices if d.type == "cpu"), None)
    units = [Unit(f"{d.name} {d.description}", d.free_mb * GPU_SHARE, d.bandwidth or 100.0, [d.name])
             for d in local_gpus]
    host = Unit("This PC (CPU/RAM)", host_available_mb() * HOST_SHARE,
                (cpu.bandwidth if cpu and cpu.bandwidth else 20.0), [])

    workers: list[Peer] = []
    if args.workers not in ("off", "auto", "all"):
        for ep in args.workers.split(","):
            h, _, p = ep.strip().rpartition(":")
            workers.append(Peer("worker", ep, h or ep, int(p) if h else DEFAULT_RPC_PORT))
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
    # Free memory as the workers report it over RPC (also fills it in for
    # workers given by address, which have no mDNS record).
    if workers:
        reported = list_devices([w.endpoint for w in workers])
        for w in workers:
            free = sum(f for n, d, f in reported if _on_worker(n, d, w.endpoint))
            if free:
                w.txt["ram"] = str(free)
    for w in workers:
        log(f"Worker {w.name}: {w.ram_mb or '?'} MB free, {w.bandwidth or '?'} GB/s")

    local_total = sum(u.capacity_mb for u in units) + host.capacity_mb
    if args.workers == "auto" and need_mb <= local_total:
        if workers:
            log(f"Model ({model_mb/1024:.1f} GB) fits on this PC: leaving workers out "
                "(faster, no network in the loop). Use --workers all to include them.")
        workers = []
    remote = [Unit(w.name, (w.ram_mb or 2000) * (GPU_SHARE if w.is_gpu else PHONE_SHARE),
                   w.bandwidth or UNKNOWN_PHONE_SPEED, [], endpoint=w.endpoint) for w in workers]

    all_units = units + remote + [host]
    total = sum(u.capacity_mb for u in all_units)
    if need_mb > total:
        die(f"Model needs ~{need_mb/1024:.1f} GB, all devices together have ~{total/1024:.1f} GB free.")
    explicit = args.workers not in ("off", "auto", "all")
    if (args.workers == "all" or explicit) and remote:
        split_by_speed(all_units, need_mb)
    else:
        fill_fastest_first(all_units, need_mb)

    used_remote = [u for u in remote if u.assigned_mb > 0]
    rpc = [u.endpoint for u in used_remote]
    # Device names per RPC worker (a PC worker may serve several GPUs).
    devs = list_devices(rpc)
    for u in used_remote:
        u.devices = [n for n, d, _ in devs if _on_worker(n, d, u.endpoint)]
        if not u.devices:
            die(f"Worker {u.endpoint} reported no devices (is it running {RPC_VERSION}?)")
    split = []
    for name, desc, _ in devs:
        if name.startswith("RPC"):
            owner = next((u for u in used_remote if _on_worker(name, desc, u.endpoint)), None)
        else:
            owner = next((u for u in units if name in u.devices), None)
        split.append(owner.assigned_mb / len(owner.devices) if owner else 0.0)

    offloaded = sum(split)
    local_layers = round(n_layer * host.assigned_mb / need_mb) if need_mb else 0
    ngl = 0 if offloaded <= 0 else n_layer + 1 - max(0, min(n_layer, local_layers))

    print("\nPlan:")
    for u in all_units:
        if u.assigned_mb > 0:
            size = f"{u.assigned_mb/1024:6.1f} GB" if u.assigned_mb >= 1024 else f"{u.assigned_mb:6.0f} MB"
            print(f"  {u.label:<38} {size}  ({u.speed:.0f} GB/s)")
    print(f"  {n_layer} layers, {ngl} offloaded to GPUs/workers\n")

    key = args.api_key if args.api_key is not None else _api_key()
    cmd = [find_bin("llama-server"), "-m", model, "--host", args.host, "--port", str(args.port),
           "-c", str(args.ctx), "--fit", "off", "-ngl", str(ngl)]
    if any(split):
        cmd += ["--tensor-split", ",".join(f"{s:.0f}" for s in split)]
    if rpc:
        cmd += ["--rpc", ",".join(rpc)]
    if key:
        cmd += ["--api-key", key]
    cmd += args.extra
    log(" ".join(cmd).replace(key, "***") if key else " ".join(cmd))
    proc = subprocess.Popen(cmd)

    address = args.advertise_ip or (args.host if args.host not in ("0.0.0.0", "") else lan_address())
    name = Path(model).stem
    adv = None
    if not args.no_advertise:
        adv = Advertiser(SERVER_TYPE, f"PrivateLM {socket.gethostname()}", args.port,
                         {"path": "/v1", "model": name, "auth": "1" if key else "0", "v": "1"}, address)
        adv.start()
    from urllib.parse import urlencode

    params = {"url": f"http://{address}:{args.port}/v1", "model": name, "name": socket.gethostname()}
    if key:
        params["key"] = key
    print(f"\nOpenAI-compatible API: http://{address}:{args.port}/v1  (model: {name})")
    show_qr(f"privatelm://server?{urlencode(params)}")
    _wait(proc, adv)


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


def _api_key() -> str:
    path = CONFIG_DIR / "api_key"
    if not path.exists():
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_urlsafe(24))
        path.chmod(0o600)
    return path.read_text().strip()


def _wait(proc: subprocess.Popen, adv: Optional[Advertiser]) -> None:
    def stop(*_):
        proc.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        code = proc.wait()
    finally:
        if adv:
            adv.stop()
    sys.exit(code)


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
    w.set_defaults(fn=cmd_worker)

    s = sub.add_parser("serve", help="run a model here and serve an OpenAI-compatible API")
    s.add_argument("-m", "--model", required=True)
    s.add_argument("--workers", default="auto",
                   help="auto (only if the model doesn't fit here), all, off, or host:port,...")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=DEFAULT_API_PORT)
    s.add_argument("--ctx", type=int, default=8192)
    s.add_argument("--api-key", help="default: generated once, kept in ~/.config/privatelm-node")
    s.add_argument("--advertise-ip")
    s.add_argument("--no-advertise", action="store_true")
    s.add_argument("--scan-timeout", type=float, default=4.0)
    s.add_argument("extra", nargs=argparse.REMAINDER, help="-- further llama-server options")
    s.set_defaults(fn=cmd_serve)

    sc = sub.add_parser("scan", help="list workers and servers on the LAN")
    sc.add_argument("--timeout", type=float, default=4.0)
    sc.set_defaults(fn=cmd_scan)

    args = ap.parse_args()
    if getattr(args, "extra", None) and args.extra[:1] == ["--"]:
        args.extra = args.extra[1:]
    args.fn(args)


if __name__ == "__main__":
    main()
