#!/usr/bin/env python3
"""myrmid: a Linux PC (with its GPUs) in a MyrmidLM cluster of PCs and phones.

  myrmid detect               devices, free memory, bandwidth
  myrmid worker               lend this PC's GPU(s) to a master
  myrmid serve -m model.gguf  run a model here, optionally with
                              phones (or other PCs) as workers, and
                              serve it as an OpenAI-compatible API
  myrmid scan                 list workers and servers on the LAN
  myrmid update               versions and the cluster-wide update

Workers advertise themselves exactly like worker phones do
(_privatelm._tcp, TXT ram/bw/model/id/v), so the app's "Scan nearby" finds a
PC like any other phone. The service types, the privatelm:// QR codes and
the ids keep their PrivateLM names: they are the protocol the app (and
phones running older builds) speak, MyrmidLM is a fork of PrivateLM. Phones and PC must run the same llama.cpp RPC
protocol (rpc9); build.sh builds the commit the app vendors.

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

RPC_VERSION = "rpc9"
WORKER_TYPE = "_privatelm._tcp.local."
SERVER_TYPE = "_privatelm-api._tcp.local."
HOST_TYPE = "_privatelm-host._tcp.local."  # PCs that load whole models for a master's router
DEFAULT_RPC_PORT = 50052
DEFAULT_API_PORT = 8080

BIN_DIR = Path(os.environ.get("MYRMID_BIN") or os.environ.get("PRIVATELM_NODE_BIN") or Path(__file__).resolve().parent)


def _data_dir(base: Path) -> Path:
    """base/myrmid. Before the rename this was base/privatelm-node: move it
    over (keys, calibration, model choice), or keep using it if that fails."""
    new, old = base / "myrmid", base / "privatelm-node"
    if not new.exists() and old.is_dir():
        try:
            old.rename(new)
        except OSError:
            return old
    return new


def env(name: str, default: Optional[str] = None) -> Optional[str]:
    """MYRMID_<name>, or the PRIVATELM_<name> it was called before."""
    return os.environ.get(f"MYRMID_{name}") or os.environ.get(f"PRIVATELM_{name}") or default


CACHE_DIR = _data_dir(Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")))
CONFIG_DIR = _data_dir(Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")))
RUN_DIR = env("RUN_DIR", "/run/myrmid")  # RuntimeDirectory of the services: requests to the root helpers

# Share of a device's free memory the planner may fill. Phones: the same
# 0.75 the app uses for workers. GPUs keep room for the KV cache and
# compute buffers; the host keeps room for the OS and llama-server.
GPU_SHARE = 0.95  # of free VRAM; GPU_RESERVE_MB comes off on top
PHONE_SHARE = 0.75
PC_RAM_SHARE = 0.9  # of what a PC RAM worker offers (already minus its reserve)
HOST_SHARE = 0.9
UNKNOWN_PHONE_SPEED = 4.0  # GB/s, as in the app
# Phones measure their memory bandwidth with a copy loop, but llama.cpp on a
# phone CPU is held back by compute and thermal throttling long before that:
# a phone reaching about half of it is typical. Without this a phone looks
# faster than a laptop and gets the whole model.
PHONE_SPEED_FACTOR = 0.5


def log(msg: str) -> None:
    print(f"[myrmid] {msg}", file=sys.stderr, flush=True)


PLAN_FAILED = 2  # the model doesn't fit: restarting won't help (systemd: RestartPreventExitStatus=2)


class PlanTooBig(Exception):
    pass


def die(msg: str, code: int = 1) -> "NoReturn":  # type: ignore[name-defined]
    log(msg)
    sys.exit(code)


def find_bin(name: str) -> str:
    for cand in (BIN_DIR / name, BIN_DIR.parent / "build" / "bin" / name):
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    found = shutil.which(name)
    if found:
        return found
    die(f"{name} not found next to myrmid or in PATH. Run build.sh first.")


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
    args = [find_bin("myrmid-probe")] + ([] if need_bench else ["--no-bench"])
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
    its IP changes). MYRMID_MACHINE_ID sets one, e.g. for cloned VMs that
    share /etc/machine-id."""
    if env("MACHINE_ID"):
        return env("MACHINE_ID")
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
    """The address other devices reach this PC at: its fastest link into
    the LAN (cable before Wi-Fi, see best_link)."""
    try:
        return best_address()
    except Exception:  # noqa: BLE001 - odd interfaces: the default route still works
        return _default_route_address()


def _default_route_address() -> str:
    """The address of the default route's interface."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 9))  # no packet is sent
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


# ── Network: the fastest connection ─────────────────────────────────────────
#
# A PC with cable and Wi-Fi announces the address of its fastest link:
# wired before Wi-Fi, the faster wired one first, but only links into the
# same network as the default route (a second cable to a NAS is no way for
# the master to reach it). A watcher checks every 20 s; when the best link
# changes (cable plugged in or pulled), the mDNS announcements move to the
# new address and the master re-plans onto it.

VIRTUAL_IFACE = re.compile(r"^(lo|docker|br-|virbr|veth|vnet|tun|tap|wg|wt\d|tailscale|zt|ham|vmnet|vboxnet|"
                           r"cni|flannel|cali|kube|lxc|lxd|incus|podman)")
NET_WATCH_S = 20
NET_STATE: dict = {"address": "", "link": ""}  # what this PC announces now, for status pages


def _ifreq_ip(sock, name: str, req: int) -> Optional[str]:
    import fcntl

    try:
        return socket.inet_ntoa(fcntl.ioctl(sock.fileno(), req, struct.pack("256s", name[:15].encode()))[20:24])
    except OSError:
        return None


def local_links() -> list[dict]:
    """IPv4 links that are up: address, interface, wired or Wi-Fi, speed."""
    import ipaddress

    out = []
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for _i, name in socket.if_nameindex():
            if VIRTUAL_IFACE.match(name):
                continue
            base = Path(f"/sys/class/net/{name}")
            if _read(base / "operstate").strip() not in ("up", "unknown") or _read(base / "carrier").strip() == "0":
                continue
            ip = _ifreq_ip(s, name, 0x8915)  # SIOCGIFADDR
            mask = _ifreq_ip(s, name, 0x891B)  # SIOCGIFNETMASK
            if not ip or ip.startswith("127."):
                continue
            wifi = (base / "wireless").exists() or (base / "phy80211").exists()
            try:
                speed = max(0, int(_read(base / "speed").strip() or 0)) if not wifi else 0
            except ValueError:
                speed = 0
            net = ipaddress.ip_network(f"{ip}/{mask or '255.255.255.0'}", strict=False)
            out.append({"ip": ip, "iface": name, "kind": "wifi" if wifi else "wired", "speed": speed, "net": net})
    finally:
        s.close()
    return out


def link_label(link: Optional[dict]) -> str:
    if not link:
        return ""
    if link["kind"] == "wifi":
        return f"WLAN ({link['iface']})"
    sp = link["speed"]
    return f"Kabel ({link['iface']}" + (f", {sp // 1000} Gbit/s)" if sp >= 1000 else f", {sp} Mbit/s)" if sp else ")")


def best_link() -> Optional[dict]:
    """The fastest link into the default route's network."""
    import ipaddress

    links = local_links()
    default = _default_route_address()
    if not links:
        return None
    dnet = next((ln["net"] for ln in links if ln["ip"] == default), None)
    same = [ln for ln in links if dnet is None or ipaddress.ip_address(ln["ip"]) in dnet] or links
    return max(same, key=lambda ln: (ln["kind"] == "wired", ln["speed"], ln["ip"] == default))


def best_address() -> str:
    ln = best_link()
    return ln["ip"] if ln else _default_route_address()


def all_lan_addresses() -> list[str]:
    """Every address of this PC in its LAN (for firewalls on the workers)."""
    import ipaddress

    links = local_links()
    default = _default_route_address()
    dnet = next((ln["net"] for ln in links if ln["ip"] == default), None)
    return [ln["ip"] for ln in links if dnet is None or ipaddress.ip_address(ln["ip"]) in dnet]


class NetWatch:
    """Moves this PC's mDNS announcements to its fastest link when that changes."""

    def __init__(self, advs: list, fixed: bool, on_change=None):
        self.advs, self.fixed, self.on_change = advs, fixed, on_change
        ln = best_link()
        NET_STATE.update(address=ln["ip"] if ln else _default_route_address(), link=link_label(ln))

    def start(self) -> None:
        import threading

        if self.fixed:  # --advertise-ip or --host set: the user chose the address
            return
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self) -> None:
        while True:
            time.sleep(NET_WATCH_S)
            try:
                ln = best_link()
            except Exception:  # noqa: BLE001 - try again next round
                continue
            if not ln or ln["ip"] == NET_STATE["address"]:
                continue
            old = f"{NET_STATE['address']} ({NET_STATE['link'] or '?'})"
            NET_STATE.update(address=ln["ip"], link=link_label(ln))
            log(f"Network: now {ln['ip']} over {NET_STATE['link']} (was {old}); moving the announcements")
            mac, bcast = iface_info(ln["ip"])
            for adv in self.advs:
                txt = dict(adv.txt, link=ln["kind"])
                if mac:  # Wake-on-LAN needs the wired card's MAC
                    txt.update(mac=mac, bcast=bcast)
                adv.move(ln["ip"], txt)
            if self.on_change:
                self.on_change(ln["ip"])


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
        self._zc = self._info = self._proc = None

    def move(self, address: str, txt: Optional[dict] = None) -> None:
        """Announces the service on another address (and TXT) from now on."""
        try:
            self.stop()
        except Exception:  # noqa: BLE001 - the old interface may be gone already
            self._zc = self._info = self._proc = None
        self.address = address
        if txt is not None:
            self.txt = txt
        self.start()


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
            kind = {SERVER_TYPE: "server", HOST_TYPE: "host"}.get(type_, "worker")
            found[(kind, name)] = Peer(kind, name.split("." + type_)[0], addrs[0], info.port, txt)

        def update_service(self, *a):
            pass

        def remove_service(self, *a):
            pass

    browsers = [ServiceBrowser(zc, t, Listener()) for t in (WORKER_TYPE, SERVER_TYPE, HOST_TYPE)]
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
    for stype, kind in (("_privatelm._tcp", "worker"), ("_privatelm-api._tcp", "server"),
                        ("_privatelm-host._tcp", "host")):
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
HOST_RESERVE_MB = 768
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
    status_port: int = 0  # worker's monitoring endpoint (0: none, e.g. phones)
    mem_mb: list[float] = field(default_factory=list)  # room per device, for re-planning at another -c
    key: str = ""  # identity for calibration (see Calibration)
    base_speed: float = 0.0  # speed before calibration

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
    mac, bcast = iface_info(address)
    if mac:  # for Wake-on-LAN by the master
        txt.update(mac=mac, bcast=bcast)
    ln = next((x for x in local_links() if x["ip"] == address), None)
    if ln:
        txt["link"] = ln["kind"]
        log(f"Network: {address} over {link_label(ln)}")
    pairing = Pairing(read_cluster_key(args.key, args.key_file), args.pair_from)
    if not args.no_status:
        txt["stat"] = str(args.status_port)
        if pairing.waiting:  # the master sees it and gives this PC its key
            txt["pair"] = "1"
            log("No cluster key yet: the first master that sees this PC takes it into its cluster"
                + (f" (only {args.pair_from})" if args.pair_from else " (in the next hour)"))
    _advertise(args, advs, f"MyrmidLM {host}", args.port, txt, address)
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
            # An integrated GPU lends the same RAM: offer only what's left.
            offered -= sum(d.free_mb for d in chosen if d.type == "igpu")
        if gpu and cpu is not None and offered < 1024:
            log("--ram: the integrated GPU already lends this PC's RAM; no separate RAM worker")
        elif gpu and cpu is not None:
            log(f"Offering {offered} MB RAM at {cpu.bandwidth:.1f} GB/s on {address}:{ram_port}")
            procs.append(_rpc_server(args, ram_port, ["CPU"]))
            rpc_info.append({"port": ram_port, "kind": "cpu", "offered_mb": offered, "bw": cpu.bandwidth,
                             "devices": [{"name": "CPU", "description": cpu.description,
                                          "free_mb": offered, "bandwidth": cpu.bandwidth}]})
            rtxt = {"ram": str(offered), "bw": f"{cpu.bandwidth:.1f}", "model": f"{host} · RAM",
                    "v": RPC_VERSION, "id": f"{wid}-ram", "kind": "cpu", "backend": "CPU",
                    **({"mac": mac, "bcast": bcast} if mac else {})}
            if not args.no_status:
                rtxt["stat"] = str(args.status_port)
                if pairing.waiting:
                    rtxt["pair"] = "1"
            _advertise(args, advs, f"MyrmidLM {host} RAM", ram_port, rtxt, address)
            show_qr(worker_qr(address, ram_port, f"{host} · RAM", offered, cpu.bandwidth,
                              f"{wid}-ram", "cpu"))
    if not args.no_status:
        start_worker_status(args, rpc_info, procs, devices, pairing, advs)
    NetWatch(advs, fixed=bool(args.advertise_ip) or args.host not in ("0.0.0.0", "")).start()
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
        phone = not w.txt.get("kind")  # PC workers say gpu or cpu
        if w.is_gpu:
            mem = [f * GPU_SHARE - GPU_RESERVE_MB for _, f in devs_w]
        else:
            offered = sum(f for _, f in devs_w)
            if w.ram_mb:
                offered = min(offered, w.ram_mb)
            budget = offered * (PHONE_SHARE if phone else PC_RAM_SHARE)
            total_f = sum(f for _, f in devs_w) or 1
            mem = [budget * f / total_f for _, f in devs_w]
        mem = [max(0.0, x) for x in mem]
        caps = [int(x // per_layer) for x in mem]
        speed = (w.bandwidth * PHONE_SPEED_FACTOR if phone else w.bandwidth) if w.bandwidth else UNKNOWN_PHONE_SPEED
        u = Unit(w.txt.get("model") or w.name, sum(caps), speed, [],
                 endpoint=w.endpoint, rtt_s=measure_rtt(w.endpoint),
                 net_mbps=NET_MBPS_PHONE if phone else NET_MBPS_PC, device_caps=caps,
                 status_port=0 if phone else int(w.txt.get("stat") or STATUS_PORT), mem_mb=mem,
                 key=f"worker/{w.txt.get('id') or w.host}/{w.port}")
        log(f"Worker {u.label} ({w.endpoint}): room for {u.capacity_layers} layers, "
            f"{u.speed:.0f} GB/s{' (phone: half of the measured %.0f)' % w.bandwidth if phone and w.bandwidth else ''}, "
            f"{u.rtt_s*1000:.1f} ms round trip")
        remote.append(u)
    return remote


@dataclass
class MoePlan:
    counts: list[int]  # layers per local GPU, registry order
    n_cpu_moe: int  # experts of the first N layers stay in RAM
    time_s: float
    cache_mb: int = 0  # >0: all experts in RAM (-cmoe), this much GPU memory caches the most used
    cpu_share: float = 0.0  # share of cache misses the CPU computes instead of uploading
    hit_est: float = 0.0


# MoE expert cache (llama.cpp b12191+): with the experts in RAM, the GPU keeps
# the most used ones (LRFU-like) and the CPU computes a share of the misses
# instead of uploading them. Usage is skewed, so a cache holding a fraction f
# of all experts serves more than f of the reads; sqrt(f) is a cautious guess
# (the PR measured 72-89 % hits). myrmid tune-moe measures it.
MOE_CACHE_MARGIN_MB = 256  # per GPU, for compute buffers that grow with the cache
MOE_CACHE_MIN_MB = 512
MOE_CPU_SHARE = 0.5  # default until tune-moe measured this PC (mmap'd experts: the CPU often reads faster)
MOE_TUNE_STATE = CONFIG_DIR / "moe-tune.json"


def moe_tuned(model: str) -> dict:
    try:
        return json.loads(MOE_TUNE_STATE.read_text()).get(str(Path(model).resolve()), {})
    except (OSError, ValueError):
        return {}


def moe_cache_plan(m: ModelInfo, gpus: list[Device], budgets: list[float], host_budget: float,
                   host_speed: float, cpu_share: float, max_mb: Optional[int] = None) -> Optional[MoePlan]:
    """All experts in RAM, attention/shared weights/KV on the GPU(s), the
    GPU memory left over as expert cache."""
    dense = m.layer_mb - m.expert_mb + m.kv_layer_mb
    counts = [0] * len(gpus)
    used = [0.0] * len(gpus)
    g = 0
    for _ in range(m.n_layer):
        while g < len(gpus) and used[g] + dense > budgets[g]:
            g += 1
        if g == len(gpus):
            return None
        used[g] += dense
        counts[g] += 1
    if m.n_layer * m.expert_mb + m.embd_mb > host_budget:
        return None
    cache = int(sum(max(0.0, b - u - MOE_CACHE_MARGIN_MB) for b, u in zip(budgets, used)))
    if max_mb is not None:
        cache = min(cache, max_mb)
    if cache < MOE_CACHE_MIN_MB:
        return None
    hit = min(1.0, (cache / max(1.0, m.n_layer * m.expert_mb)) ** 0.5)
    active = m.expert_mb * max(1, m.n_expert_used) / m.n_expert
    sp = min((d.bandwidth or 100.0) for d, c in zip(gpus, counts) if c)
    t = m.n_layer * ((m.layer_mb - m.expert_mb) / 1024 / sp + active / 1024 * (hit / sp + (1 - hit) / host_speed))
    t += m.output_mb / 1024 / (gpus[-1].bandwidth or 100.0)
    return MoePlan(counts, 0, t, cache_mb=cache, cpu_share=cpu_share, hit_est=hit)


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
    # For re-planning without touching busy devices: every unit with the
    # room it had before the server started, the model at this -c, and the
    # unit that computes the output head.
    units: list = field(default_factory=list)
    model_info: Optional["ModelInfo"] = None
    out_label: str = ""
    model_path: str = ""

    def endpoints(self) -> list[str]:
        return [ep for mach in self.info.get("machines", []) for ep in mach.get("endpoints", [])]


CTX_STEPS = (8192, 4096, 2048)  # without --ctx: the first that fits


def build_server_fitting(args, quiet: bool = False, exit_on_fail: bool = True) -> ServerPlan:
    """build_server, halving the context (down to 2048) when the model
    doesn't fit and no --ctx was given: the KV cache is often what's over.
    exit_on_fail=False raises PlanTooBig instead of ending the program."""
    auto = args.ctx is None
    if auto:
        args.ctx = CTX_STEPS[0]
    while True:
        try:
            return build_server(args, quiet)
        except PlanTooBig as e:
            if not auto or args.ctx <= CTX_STEPS[-1]:
                if auto:
                    args.ctx = None  # the next try starts at the top again
                if not exit_on_fail:
                    raise
                die(str(e) + ("" if auto else " (--ctx was set; leave it out to shrink it automatically)"),
                    PLAN_FAILED)
            log(f"Doesn't fit with -c {args.ctx}; trying -c {args.ctx // 2}.")
            args.ctx //= 2


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
    logits_mb = m.n_vocab * 4 * 8 / 2**20  # scores for the last tokens of a batch only
    out_extra = m.output_mb + logits_mb + (MTP_RESERVE_MB if args.mtp else 0)

    devices = probe(bench=True)
    for d in devices:  # memory kept for the router's fast model
        if d.name in getattr(args, "reserve_mb", {}):
            d.free_mb = max(0, int(d.free_mb - args.reserve_mb[d.name]))
    local_gpus = usable_gpus(devices)
    only = getattr(args, "only_devices", None)
    if only:  # a host with one model per GPU: this model gets its GPU (and its share of the RAM)
        local_gpus = [d for d in local_gpus if d.name in only]
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
    host_budget = host_available_mb() * HOST_SHARE - HOST_RESERVE_MB
    host_budget -= getattr(args, "reserve_mb", {}).get("__host__", 0)  # router's fast model in RAM
    if args.ram_limit is not None:
        host_budget = min(host_budget, args.ram_limit * 1024)
    # An integrated GPU has no memory of its own: what it reports as free is
    # the same RAM the CPU uses. Whatever it gets comes off the RAM budget.
    for i, d in enumerate(local_gpus):
        if d.type == "igpu":
            budgets[i] = max(0.0, min(budgets[i], host_budget))
            host_budget -= budgets[i]
    gpus = [Unit(f"{d.name} {d.description}", max(0, int(b // per_layer)), d.bandwidth or 100.0,
                 [d.name], device_caps=[max(0, int(b // per_layer))],
                 min_layers=m.n_nextn if (args.mtp and d is out_gpu) else 0, mem_mb=[max(0.0, b)])
            for d, b in zip(local_gpus, budgets)]
    host_room = host_budget - m.embd_mb - (0 if out_gpu else out_extra)
    host = Unit("This PC (RAM)", max(0, int(host_room // per_layer)), host_speed, [], mem_mb=[max(0.0, host_room)])

    # MoE on this PC alone (experts in RAM), if it fits and isn't slower.
    moe = None
    if m.is_moe and out_gpu is not None and out_gpu is local_gpus[-1] and args.workers in ("auto", "off"):
        # experts not on the GPU live in RAM: the RAM budget is for them
        moe = moe_local_plan(m, local_gpus, budgets, host_budget, host_speed)
        mode = str(getattr(args, "moe_cache", "auto") or "auto").lower()
        if moe and moe.n_cpu_moe and mode != "off":
            # Some experts have to live in RAM anyway: a cache over all of them
            # beats a fixed split, since usage is skewed (llama.cpp PR #30112, #30966).
            tuned = moe_tuned(model)
            share = getattr(args, "moe_cpu_share", None)
            share = tuned.get("share", MOE_CPU_SHARE) if share is None else share
            cap = int(mode) if mode.isdigit() else tuned.get("mib")
            cached = moe_cache_plan(m, local_gpus, budgets, host_budget, host_speed, share, cap)
            if cached:
                moe = cached

    workers = _find_workers(args) if not (moe and args.workers == "off") else []
    remote = _remote_units(workers, per_layer)
    me = socket.gethostname()
    for d, u in zip(local_gpus, gpus):
        u.key = f"{me}/{d.description}/{d.device_id or d.name}"
    host.key = f"{me}/cpu"
    apply_calibration(gpus + remote + [host], Calibration())
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
        raise PlanTooBig(f"Model needs {m.n_layer} layers of ~{per_layer:.0f} MB (weights + KV cache at "
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
        f"~{net_kb:.0f} KB/token on the wire (check with: myrmid bench)\n")

    cmd = [find_bin("llama-server"), "-m", model, "-c", str(args.ctx), "--fit", "off", "-ngl", str(ngl)]
    if any(counts):
        cmd += ["--tensor-split", ",".join(str(c) for c in counts)]
    cmd += extra
    if rpc:
        cmd += ["--rpc", ",".join(rpc)]
    info = _plan_info(args, model, m, "layers", 1 / est, hops, net_kb, segments, used_remote,
                      {"machine": "master" if not out_unit.remote else out_unit.endpoint.rsplit(":", 1)[0],
                       "label": where})
    sp = _finish_command(args, cmd, model, 1 / est, info)
    sp.units, sp.model_info, sp.out_label, sp.model_path = gpus + remote + [host], m, out_unit.label, model
    return sp


def _plan_info(args, model: str, m: ModelInfo, mode: str, tps: float, hops: int, net_kb: float,
               segments: list[dict], used_remote: list[Unit], output: dict, moe_cpu_layers: int = 0) -> dict:
    machines = [{"id": "master", "role": "master", "host": socket.gethostname(),
                 "layers": sum(s["layers"] for s in segments if s["machine"] == "master")}]
    for u in used_remote:
        hid = u.endpoint.rsplit(":", 1)[0]
        mach = next((x for x in machines if x["id"] == hid), None)
        if mach is None:
            mach = {"id": hid, "role": "worker", "host": hid, "label": u.label, "endpoints": [],
                    "status_port": u.status_port, "phone": not u.status_port, "layers": 0,
                    "rtt_ms": round(u.rtt_s * 1000, 2)}
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
    if moe.cache_mb:
        say(f"  All experts in RAM ({m.n_layer * m.expert_mb / 1024:.1f} GB); GPU expert cache "
            f"{moe.cache_mb / 1024:.1f} GB (est. {moe.hit_est * 100:.0f} % hits), CPU computes "
            f"{moe.cpu_share:g} of the misses")
    elif moe.n_cpu_moe:
        say(f"  Experts of the first {moe.n_cpu_moe} layers: this PC's RAM "
            f"({moe.n_cpu_moe * m.expert_mb / 1024:.1f} GB)")
    say(f"  Output head: {gpus[-1].name}")
    _say_model(say, m, m.per_layer_mb(), args)
    say(f"  Estimate: ~{1/moe.time_s:.1f} tokens/s{' before MTP' if args.mtp else ''}, no network hops\n")
    segments = [{"machine": "master", "device": f"{d.name} {d.description}", "layers": c}
                for d, c in zip(gpus, moe.counts) if c]
    info = _plan_info(args, model, m, "moe", 1 / moe.time_s, 0, 0, segments, [],
                      {"machine": "master", "label": gpus[-1].name}, moe.n_cpu_moe)
    if moe.cache_mb:
        info["moe_cache"] = {"mib": moe.cache_mb, "cpu_share": moe.cpu_share, "hit_est": round(moe.hit_est, 2),
                             "tuned": bool(moe_tuned(model))}
    counts = list(moe.counts)
    counts[-1] += 1  # output slot on the last GPU
    cmd = [find_bin("llama-server"), "-m", model, "-c", str(args.ctx), "--fit", "off",
           "-ngl", str(m.n_layer + 1)]
    if len(gpus) > 1:
        cmd += ["--tensor-split", ",".join(str(c) for c in counts)]
    if moe.cache_mb:
        cmd += ["--cpu-moe", "--moe-cache-mib", str(moe.cache_mb), "--moe-cache-cpu-share", f"{moe.cpu_share:g}"]
    elif moe.n_cpu_moe:
        cmd += ["--n-cpu-moe", str(moe.n_cpu_moe)]
    return _finish_command(args, cmd, model, 1 / moe.time_s, info)


def _finish_command(args, cmd: list[str], model: str, tps: float, info: Optional[dict] = None) -> ServerPlan:
    cmd += ["--metrics"]  # for the dashboard
    if args.mtp:
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(args.mtp)]
    elif getattr(args, "draft_opts", None):
        cmd += args.draft_opts
        if info is not None:
            info["draft"] = args.draft_info
    key = args.api_key if args.api_key is not None else _api_key()
    cmd += ["--host", args.host, "--port", str(args.port)]
    if key:
        cmd += ["--api-key", key]
    cmd += args.extra
    return ServerPlan(cmd, Path(model).stem, key, tps, info or {})


MODEL_STATE = CONFIG_DIR / "model.json"


def restore_model_choice(args) -> None:
    """The model picked in the dashboard wins over the service's -m, as long
    as -m is still what it was then (a changed -m is a new decision)."""
    try:
        d = json.loads(MODEL_STATE.read_text())
    except (OSError, ValueError):
        return
    cli = str(Path(args.model).expanduser().resolve())
    if d.get("cli") == cli and d.get("model") and Path(d["model"]).is_file() and d["model"] != cli:
        log(f"Model chosen in the dashboard: {Path(d['model']).name} (the service's -m is {Path(cli).name})")
        args.cli_model = cli
        args.model = d["model"]


def save_model_choice(args, path: str) -> None:
    cli = getattr(args, "cli_model", None) or str(Path(args.model).expanduser().resolve())
    args.cli_model = cli
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_STATE.write_text(json.dumps({"cli": cli, "model": path}))
    except OSError as e:
        log(f"Could not save the model choice ({e})")


def setup_draft(args) -> None:
    """(Re)computes the draft model for args.model and its memory reserve."""
    args.draft_opts, args.draft_info = [], {}
    reserve = dict(getattr(args, "fast_reserve", None) or {})
    if args.draft_model and args.draft_model.lower() not in ("off", "none", "no", "0"):
        if args.mtp:
            log("--mtp and --draft-model: the model's own MTP layers are used, no draft model")
        else:
            draft = find_draft(args, str(Path(args.model).expanduser().resolve()))
            if draft:
                opts, res, args.draft_info = plan_draft(args, draft, probe(bench=True))
                args.draft_opts = opts
                for k, v in res.items():
                    reserve[k] = reserve.get(k, 0) + v
                log(f"Draft model {args.draft_info['model']} ({args.draft_info['size_gb']} GB) on "
                    f"{args.draft_info['device']}, up to {args.draft_max} tokens ahead")
    args.reserve_mb = reserve


def cmd_serve(args) -> None:
    has_fast = bool(args.fast_model or args.fast_url)
    public_host, public_port = args.host, args.port
    fast_cmd: Optional[list[str]] = None
    fast_info: dict = {}
    if has_fast:
        if args.fast_model:
            fast_cmd, args.reserve_mb, fast_info = plan_fast(args, probe(bench=True))
        else:
            fast_info = {"model": args.fast_url, "device": "eigener Server", "ctx": args.fast_ctx}
    restore_model_choice(args)
    args.fast_reserve = dict(getattr(args, "reserve_mb", None) or {})
    setup_draft(args)
    # The router takes the public port; the big model listens locally on +1
    # (the fast one on +2). The router is always there: hosts may join later.
    args.host, args.port = "127.0.0.1", public_port + 1

    sup = Supervisor(args, _rebalance_initial(args.rebalance))
    if not sup.start_plan(initial=True):
        log(f"{sup.reason} Waiting for workers (checking every {REBALANCE_SCAN_S} s; "
            "the dashboard shows the state).")
    sp = sup.sp
    big_base = f"http://127.0.0.1:{args.port}"
    fast_proc = None
    backends = [Backend("big", big_base, sp.key, sp.name, group=sp.name, where=socket.gethostname(),
                        ctx=args.ctx or 0, est_tps=sp.estimate_tps or 0.0)]
    if has_fast:
        if fast_cmd:
            if sp.key:
                fast_cmd += ["--api-key", sp.key]
            print(f"Fast model: {fast_info['model']} ({fast_info['size_gb']} GB) on {fast_info['device']}, "
                  f"context {args.fast_ctx}\n")
            fast_proc = subprocess.Popen(fast_cmd)
            backends.append(Backend("fast", f"http://127.0.0.1:{fast_info['port']}", sp.key, fast_info["model"],
                                    group=fast_info["model"], where=socket.gethostname(), ctx=args.fast_ctx))
        else:
            url = args.fast_url.rstrip("/")
            backends.append(Backend("fast", url[:-3] if url.endswith("/v1") else url, args.fast_key or "", url,
                                    group="fast-url", where=url, ctx=args.fast_ctx))
        sp.info["fast"] = fast_info
    enabled, mode = _router_initial(args.router, True)
    if args.router_mode:
        mode = args.router_mode
    public_name = args.model_name
    router = Router(backends, args.fast_ctx, enabled, mode, public_name, sp.key)
    router.on_timings = sup.on_timings
    router.embedder = Embedder(public_port + 3)
    if mode == "examples":
        router.embedder.ensure()
    try:
        router.prefer_fast = bool(json.loads(ROUTER_STATE.read_text()).get("prefer_fast"))
    except (OSError, ValueError, AttributeError):
        pass

    def new_model(sp_new) -> None:
        b = router.big
        b.label = b.group = sp_new.name
    sup.on_new_model = new_model
    if not router.start(public_host, public_port):
        sup.stop_server()
        die(f"Port {public_port} is busy.")
    log(f"Router on port {public_port}: rules {'on' if enabled else 'off'} ({mode}); model name for clients: "
        f"{public_name}, or a model's own name")
    args.host, args.port = public_host, public_port
    hosts = None
    if sp.key:
        hosts = HostManager(args, router, sp.key, sup.known)
        router.waker = hosts.wake_and_wait
        hosts.start()
    elif args.assign or args.hosts:
        die("Hosts need an API key on the master (it is their cluster key); leave out --api-key \"\".")

    address = args.advertise_ip or (args.host if args.host not in ("0.0.0.0", "") else lan_address())
    adv = None
    if not args.no_advertise:
        adv = Advertiser(SERVER_TYPE, f"MyrmidLM {socket.gethostname()}", args.port,
                         {"path": "/v1", "model": public_name, "auth": "1" if sp.key else "0", "v": "1",
                          "id": machine_id(), "addrs": ",".join(all_lan_addresses())[:200]}, address)
        adv.start()
        NetWatch([adv], fixed=bool(args.advertise_ip) or args.host not in ("0.0.0.0", "")).start()
    from urllib.parse import urlencode

    params = {"url": f"http://{address}:{args.port}/v1", "model": public_name, "name": socket.gethostname()}
    if sp.key:
        params["key"] = sp.key
    print(f"\nOpenAI-compatible API: http://{address}:{args.port}/v1  (model: {public_name})")
    dash = None
    if not args.no_dashboard:
        dash = Dashboard(args, sup, probe(bench=False), router=router, big_base=big_base, hosts=hosts)
    updater = UpdateCoordinator(sup, router, hosts, dash, sp.key)
    updater.start()
    if sp.key and args.auto_pair == "on":
        Pairer(sp.key, sup).start()
    if dash:
        dash.updater = updater
        if dash.start():
            print(f"Dashboard:             http://{address}:{args.dashboard_port}/"
                  + ("  (log in with the API key)" if sp.key else ""))
    show_qr(f"privatelm://server?{urlencode(params)}")
    if sup.enabled:
        log("Rebalancing on: new or lost workers lead to a new plan when the server is idle "
            "(--rebalance off or the dashboard switches it off)")
    sup.run([adv] if adv else [], [fast_proc] if fast_proc else [])


def cmd_tune_moe(args) -> None:
    """Runs llama.cpp's moe-cache-tune.py on the plan serve would use here:
    it starts the server a few times at two cache sizes and three CPU shares,
    fits this PC's upload and CPU speeds, and recommends both values. serve
    uses them from then on (moe-tune.json)."""
    script = find_bin("moe-cache-tune.py")
    if not Path(script).is_file():
        die("moe-cache-tune.py not found next to llama-server; run build.sh again")
    args.host, args.no_advertise, args.workers = "127.0.0.1", True, "off"
    args.moe_cache, args.moe_cpu_share = "auto", None
    try:
        sp = build_server(args)
    except PlanTooBig as e:
        die(str(e), PLAN_FAILED)
    cmd, skip = [], 0
    for i, a in enumerate(sp.cmd):  # the script sets cache, share, host and port itself
        if skip:
            skip -= 1
            continue
        if a in ("--moe-cache-mib", "--moe-cache-cpu-share", "--host", "--port", "--api-key"):
            skip = 1
            continue
        cmd.append(a)
    if "--cpu-moe" not in cmd:
        die("This model doesn't need the MoE cache here: it fits on the GPU(s), or it isn't MoE.")
    out = CONFIG_DIR / "moe-tune-runs"
    out.mkdir(parents=True, exist_ok=True)
    log("Tuning the MoE cache: the server starts 6 times, this takes a few minutes (mostly loading)...")
    r = subprocess.run([sys.executable, script, "--out", str(out)] + (["--max-mib", str(args.max_mib)]
                       if args.max_mib else []) + ["--"] + cmd, capture_output=True, text=True)
    print(r.stdout[-4000:])
    found = re.search(r"--moe-cache-mib (\d+) --moe-cache-cpu-share ([0-9.]+)", r.stdout)
    if r.returncode != 0 or not found:
        die("Tuning failed:\n" + r.stderr[-2000:])
    mib, share = int(found.group(1)), float(found.group(2))
    try:
        data = json.loads(MOE_TUNE_STATE.read_text())
    except (OSError, ValueError):
        data = {}
    data[str(Path(args.model).expanduser().resolve())] = {"mib": mib, "share": share, "at": int(time.time())}
    MOE_TUNE_STATE.write_text(json.dumps(data, indent=1))
    print(f"\nSaved: cache {mib} MiB, CPU share {share:g}. serve uses them for this model from now on "
          "(restart the service).")


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
        try:
            sp = build_server(args, quiet=depth != depths[0])
        except PlanTooBig as e:
            die(str(e), PLAN_FAILED)
        base = f"http://127.0.0.1:{args.port}"
        log(f"MTP depth {depth}: starting server...")
        proc = subprocess.Popen(sp.cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            if not _wait_healthy(base, proc, timeout=args.load_timeout):
                log(f"depth {depth}: server did not start (try it with myrmid serve --mtp {depth})")
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
    print(f"\nUse: myrmid serve -m {args.model} --mtp {best[0]}" if best[0] else
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
    if args.json:
        print(json.dumps([{"kind": p.kind, "name": p.name, "host": p.host, "port": p.port, "txt": p.txt}
                          for p in peers]))
        return
    if not peers:
        print("Nothing found. mDNS may be blocked (guest Wi-Fi, client isolation, VPN).")
        return
    for p in peers:
        if p.kind == "worker":
            kind = p.txt.get("kind", "phone")
            note = "" if p.txt.get("v") == RPC_VERSION else \
                f"  !! speaks {p.txt.get('v') or '?'}, this cluster {RPC_VERSION}: " + \
                ("needs the MyrmidLM app of this release" if kind == "phone" else "update it") + " (not used)"
            print(f"worker  {p.endpoint:<22} {kind:<5} {p.ram_mb or '?':>6} MB "
                  f"{p.bandwidth or 0:>6.1f} GB/s  {p.txt.get('model', p.name)}{note}")
        elif p.kind == "host":
            gpu = p.txt.get("gpu") or "?"
            state = "  (waits to be taken into a cluster)" if p.txt.get("pair") == "1" else ""
            print(f"host    {p.endpoint:<22} {p.txt.get('host') or p.name}, {gpu}, "
                  f"models for the router on port {p.txt.get('api', HOST_API_PORT)}{state}")
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


def _http(base: str, key: str, route: str = "auto", timeout: float = 900):
    import urllib.request

    headers = {"Content-Type": "application/json"}
    if route != "auto":
        headers["X-Myrmid-Route"] = route  # with the router: measure this model
    if key:
        headers["Authorization"] = f"Bearer {key}"

    def call(path: str, body: Optional[dict] = None) -> dict:
        import urllib.error

        req = urllib.request.Request(base + path, headers=headers,
                                     data=json.dumps(body).encode() if body is not None else None)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            if e.code == 401:
                die("The server wants an API key: --api-key KEY (myrmid serve printed it)")
            die(f"Server answered {e.code} on {path}: {detail}")

    return call


FILLER = ("The lighthouse keeper wrote in his log every evening, noting the weather, the ships "
          "that passed, the state of the lamp and the small events of the island. ")


def measure(base: str, key: str, ifaces: Optional[list[str]], temp: float, tokens: int,
            context: int, runs: int = 1, route: str = "big", timeout: float = 900) -> dict:
    """Generation speed and network traffic per token (both directions:
    the difference between a 1-token and an N-token run, so the prompt's
    traffic drops out). context = rough prompt length in tokens: a full
    context is slower, and that's where workers' extra memory pays off."""
    call = _http(base, key, route, timeout)
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
    return {"tps": sum(tps) / len(tps), "per_rx": per_rx / k, "per_tx": per_tx / k, "n": n, "timings": t,
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
    r = measure(base, key, ifaces, args.temp, args.tokens, args.context, args.runs, args.route)
    per = r["per_rx"] + r["per_tx"]
    print(f"Generation:   {r['tps']:.1f} tokens/s ({r['n']} tokens, temperature {args.temp}, "
          f"context ~{args.context})")
    if r["acceptance"] is not None:
        print(f"MTP/draft:    {r['acceptance']*100:.0f}% of drafted tokens accepted "
              "(MTP depth: myrmid tune-mtp; draft model: serve --draft-max)")
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
              "              Restart with myrmid serve (it keeps the head on this PC),\n"
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


def _wait(procs, advs, also_stop: Optional[list] = None) -> None:
    """Runs until a child exits or we're stopped, then stops everything
    (also_stop: helpers whose exit doesn't end the run)."""
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
        procs = list(procs) + list(also_stop or [])
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


# ── Calibration ──────────────────────────────────────────────────────────────
#
# The planner estimates each device's speed from its memory bandwidth. Real
# speed differs (compute, drivers, throttling, the phone's CPU), so the
# master measures: every finished request on its model reports how long one
# pass of the model took (llama-server's timings; with a draft model or MTP
# per verified pass). Per plan that gives one equation
#     measured = network + sum over devices (layers x read / speed) x k
# with one unknown factor k per device. Several plans (rebalancing adds
# them) pin the factors down; ridge regression towards 1 keeps a device's
# factor at its estimate until measurements say otherwise. The planner then
# uses speed / k. Factors live in calibration.json, per device.

CALIB_STATE = CONFIG_DIR / "calibration.json"
CALIB_RIDGE = 0.05  # pull towards the estimate (relative units)
CALIB_MIN_TOKENS = 16  # shorter answers say little
CALIB_K_RANGE = (0.2, 5.0)


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting (small systems)."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[p][c]) < 1e-12:
            continue
        m[c], m[p] = m[p], m[c]
        for r in range(n):
            if r != c and m[r][c]:
                f = m[r][c] / m[c][c]
                m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    return [m[i][n] / m[i][i] if abs(m[i][i]) > 1e-12 else 1.0 for i in range(n)]


class Calibration:
    def __init__(self, path: Path = CALIB_STATE):
        import threading

        self.path = path
        self.lock = threading.Lock()
        self.k: dict[str, float] = {}  # device key -> time factor (2: twice as slow as estimated)
        self.labels: dict[str, str] = {}
        self.plans: dict[str, dict] = {}  # plan signature -> a, other, t (smoothed), n, est
        try:
            d = json.loads(path.read_text())
            self.k, self.labels, self.plans = d.get("k", {}), d.get("labels", {}), d.get("plans", {})
        except (OSError, ValueError):
            pass

    def factor(self, key: str) -> float:
        """Speed multiplier for the planner (1/k)."""
        return 1.0 / self.k.get(key, 1.0)

    def save(self) -> None:
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"k": self.k, "labels": self.labels, "plans": self.plans}, indent=1))
        except OSError as e:
            log(f"Calibration: could not save ({e})")

    @staticmethod
    def terms(units: list[Unit], m: ModelInfo, out_label: str) -> tuple[dict[str, float], float]:
        """Per device key: seconds per pass at the estimated speed (a), and
        the part that doesn't depend on device speed (network)."""
        a: dict[str, float] = {}
        other = 0.0
        for u in units:
            if not u.key:
                continue
            base = u.base_speed or u.speed
            if u.layers:
                a[u.key] = a.get(u.key, 0.0) + u.layers * m.read_mb() / 1024 / base
                if u.remote:
                    other += 2 * u.rtt_s + HOP_OVERHEAD_S + 2 * m.n_embd * 4 / 2**20 / u.net_mbps
            if u.label == out_label:
                a[u.key] = a.get(u.key, 0.0) + m.output_mb / 1024 / base
                if u.remote:
                    other += m.n_vocab * 4 / 2**20 / u.net_mbps
        return a, other

    @staticmethod
    def signature(units: list[Unit]) -> str:
        return ",".join(sorted(f"{u.key}={u.layers}" for u in units if u.layers and u.key))

    def predicted(self, units: list[Unit], m: ModelInfo, out_label: str) -> float:
        a, other = self.terms(units, m, out_label)
        return other + sum(v * self.k.get(key, 1.0) for key, v in a.items())

    def observe(self, units: list[Unit], m: ModelInfo, out_label: str, seconds_per_pass: float) -> Optional[dict]:
        """Adds a measurement of the plan that is running; returns the plan's
        record (estimate before, measured) when the factors changed."""
        a, other = self.terms(units, m, out_label)
        if not a or seconds_per_pass <= 0:
            return None
        sig = self.signature(units)
        with self.lock:
            for u in units:
                if u.key:
                    self.labels[u.key] = u.label
            rec = self.plans.get(sig) or {"a": a, "other": other, "t": seconds_per_pass, "n": 0}
            n = rec["n"] + 1
            rec["t"] = rec["t"] + (seconds_per_pass - rec["t"]) / min(n, 20)  # running mean, then smoothing
            rec.update(a=a, other=other, n=n, at=int(time.time()))
            self.plans[sig] = rec
            if len(self.plans) > 30:
                oldest = min(self.plans, key=lambda s: self.plans[s].get("at", 0))
                del self.plans[oldest]
            self._fit()
        return rec

    def _fit(self) -> None:
        keys = sorted({k for rec in self.plans.values() for k in rec["a"]})
        if not keys:
            return
        idx = {k: i for i, k in enumerate(keys)}
        n = len(keys)
        ata = [[0.0] * n for _ in range(n)]
        atb = [0.0] * n
        for rec in self.plans.values():
            t = rec["t"]
            w = min(rec["n"], 10) ** 0.5  # repeated measurements count more, up to a point
            row = [0.0] * n
            for k, v in rec["a"].items():
                row[idx[k]] = v / t
            r = (t - rec["other"]) / t
            for i in range(n):
                if row[i]:
                    atb[i] += w * row[i] * r
                    for j in range(n):
                        ata[i][j] += w * row[i] * row[j]
        for i in range(n):
            ata[i][i] += CALIB_RIDGE
            atb[i] += CALIB_RIDGE * 1.0
        x = _solve(ata, atb)
        lo, hi = CALIB_K_RANGE
        for k, v in zip(keys, x):
            self.k[k] = round(min(hi, max(lo, v)), 3)

    def measured(self, units: list[Unit]) -> Optional[float]:
        rec = self.plans.get(self.signature(units))
        return rec["t"] if rec and rec["n"] >= 3 else None

    def reset(self) -> None:
        with self.lock:
            self.k, self.labels, self.plans = {}, {}, {}
        self.save()


def timings_pass_seconds(t: dict) -> Optional[float]:
    """Seconds per model pass from llama-server's timings: with speculative
    decoding several tokens come out of one pass."""
    n = t.get("predicted_n") or 0
    ms = t.get("predicted_ms") or 0
    if n < CALIB_MIN_TOKENS or ms <= 0:
        return None
    passes = n - (t.get("draft_n_accepted") or 0)
    return ms / 1000 / max(1, passes)


def cmd_calibration(args) -> None:
    c = Calibration()
    if args.reset:
        c.reset()
        print("Calibration reset: the planner uses the estimated speeds again.")
        return
    if not c.k:
        print("No measurements yet. They come from requests to the model (myrmid bench is one).")
        return
    print(f"{'Device':<52} {'measured vs. estimate':>22}")
    for key, k in sorted(c.k.items(), key=lambda x: x[1]):
        print(f"{c.labels.get(key, key)[:52]:<52} {f'{1/k:.2f}x as fast':>22}")
    print(f"\n{len(c.plans)} plan(s) measured. Reset with: myrmid calibration --reset")


# ── Rebalancing ──────────────────────────────────────────────────────────────
#
# The split is planned when llama-server starts. With rebalancing on (the
# default), serve keeps watching the LAN:
#   - a new worker that makes the model faster (by REBALANCE_GAIN) or lets
#     it run with a longer context: new plan as soon as no request runs;
#   - a worker that drops out (llama-server ends when it loses one, or the
#     worker stops answering): new plan with the ones left;
#   - a model that doesn't fit (yet): wait for workers instead of exiting.
# A new plan means reloading the model: workers without a tensor cache
# (phones) get their layers sent again, which over Wi-Fi takes minutes.
# The candidate plan is estimated from the room every device had before the
# server started, so nothing busy is probed while the model is loaded.

REBALANCE_STATE = CONFIG_DIR / "rebalance.json"
REBALANCE_GAIN = 1.15  # estimated speed-up a new plan must bring
REBALANCE_SCAN_S = 30
REBALANCE_MIN_GAP_S = 300  # between switches for speed (lost workers: at once)
REBALANCE_LOST_SCANS = 3  # missed scans before a worker counts as gone
REBALANCE_RETRY_S = 300  # waiting for workers: re-plan at least this often


def _rebalance_initial(arg: str) -> bool:
    if arg in ("on", "off"):
        return arg == "on"
    try:
        return bool(json.loads(REBALANCE_STATE.read_text()).get("enabled", True))
    except (OSError, ValueError):
        return True


def tcp_connected(endpoint: str) -> bool:
    """Whether this PC holds an established TCP connection to host:port
    (llama-server keeps one open to every RPC worker it uses)."""
    host, _, port = endpoint.rpartition(":")
    try:
        want_ip, want_port = socket.inet_aton(socket.gethostbyname(host)), int(port)
    except (OSError, ValueError):
        return False
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        for line in _read(Path(path)).splitlines()[1:]:
            f = line.split()
            if len(f) < 4 or f[3] != "01":
                continue
            ip_hex, _, port_hex = f[2].rpartition(":")
            if int(port_hex, 16) != want_port:
                continue
            if len(ip_hex) == 32 and ip_hex.startswith("0000000000000000FFFF0000"):
                ip_hex = ip_hex[24:]
            if len(ip_hex) == 8 and struct.pack("<I", int(ip_hex, 16)) == want_ip:
                return True
    return False


def apply_calibration(units: list[Unit], calib: "Calibration", quiet: bool = False) -> None:
    """Planner speed = estimated speed x measured factor (see Calibration)."""
    for u in units:
        u.base_speed = u.base_speed or u.speed
        f = calib.factor(u.key) if u.key else 1.0
        u.speed = u.base_speed * f
        if abs(f - 1) > 0.05 and not quiet:
            log(f"Calibration: {u.label} measured at {f:.2f}x its estimate ({u.base_speed:.0f} -> {u.speed:.0f} GB/s)")


def estimate_plan_full(units: list[Unit], model_path: str, ctx: int, out_label: str, prune: bool,
                       calib: Optional["Calibration"] = None) -> Optional[tuple[float, list[Unit], ModelInfo]]:
    """(tokens/s the planner expects, the units it would use, the model at
    -c ctx), or None if the model doesn't fit. With calib, speeds use the
    current factors."""
    import dataclasses

    m = gguf_info(model_path, ctx)
    per = m.per_layer_mb()
    trial = []
    for u in units:
        caps = [max(0, int(mb // per)) for mb in u.mem_mb] or [u.capacity_layers]
        trial.append(dataclasses.replace(u, capacity_layers=sum(caps), device_caps=caps, layers=0))
    if calib is not None:
        apply_calibration(trial, calib, quiet=True)
    used = plan(trial, m, prune)
    if not used:
        return None
    out = next((u for u in trial if u.label == out_label and not u.remote), None)
    return 1 / token_time(used, m, out), trial, m


def estimate_plan(units: list[Unit], model_path: str, ctx: int, out_label: str,
                  prune: bool) -> Optional[float]:
    """Tokens/s the planner expects for these units at -c ctx, or None if
    the model doesn't fit."""
    r = estimate_plan_full(units, model_path, ctx, out_label, prune)
    return r[0] if r else None


class Supervisor:
    """Owns llama-server for serve: starts it, re-plans when workers come
    or go (if enabled), and waits for workers while the model doesn't fit."""

    def __init__(self, args, enabled: bool):
        import threading

        self.args = args
        self.ctx_auto = args.ctx is None
        self.enabled = enabled
        self.sp: Optional[ServerPlan] = None
        self.proc: Optional[subprocess.Popen] = None
        self.state = "starting"  # running | waiting | switching | stopped
        self.reason = ""
        self.pending: Optional[dict] = None
        self.events: list[dict] = []
        self.lock = threading.RLock()
        self.stopping = False
        self.now_requested = False
        self.last_switch = time.time()
        self.last_scan = 0.0
        self.last_try = 0.0
        self.tried_with: frozenset = frozenset()
        self.evaluated: set = set()  # sets of new workers found not worth a switch
        self.missing: dict[str, int] = {}
        self.lan_seen: set[str] = set()  # worker endpoints in the last scan
        self.ever_seen: set[str] = set()
        self.exit_code = 0
        self.calib = Calibration()
        self.known = Known()
        self.model_job: dict = {}
        self.on_new_model = None  # callback(sp) after a model change
        self.last_calib_check = 0.0
        self.last_calib_save = 0.0
        # llama-server's own address (behind the router, if there is one)
        self.big_host, self.big_port = args.host, args.port

    # Events and state -------------------------------------------------------

    def event(self, en: str, de: str) -> None:
        log(en)
        with self.lock:
            self.events.append({"t": int(time.time()), "text": de})
            del self.events[:-20]

    def set_enabled(self, on: bool) -> None:
        with self.lock:
            self.enabled = on
            if not on:
                self.pending = None
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            REBALANCE_STATE.write_text(json.dumps({"enabled": on}))
        except OSError:
            pass
        self.event(f"Rebalancing {'on' if on else 'off'}", f"Automatische Neuverteilung {'an' if on else 'aus'}")

    def api(self, body: dict) -> dict:
        if "enabled" in body:
            self.set_enabled(bool(body["enabled"]))
        if body.get("now"):
            with self.lock:
                if self.state == "waiting":
                    self.last_try = 0.0
                    self.tried_with = frozenset({"?"})
                elif self.state == "running":
                    self.now_requested = True
        return self.info()

    def info(self) -> dict:
        with self.lock:
            return {"enabled": self.enabled, "state": self.state, "reason": self.reason,
                    "model_job": self.model_progress(),
                    "pending": self.pending, "now_requested": self.now_requested,
                    "events": list(self.events[-8:]), "ctx_auto": self.ctx_auto,
                    "gain_pct": round((REBALANCE_GAIN - 1) * 100)}

    # Server ------------------------------------------------------------------

    def placeholder(self) -> ServerPlan:
        """Plan stand-in while waiting for workers, so router and dashboard run."""
        a = self.args
        model = str(Path(a.model).expanduser().resolve())
        key = a.api_key if a.api_key is not None else _api_key()
        try:
            m = gguf_info(model, a.ctx or CTX_STEPS[-1])
            info = {"model": Path(model).stem, "n_layer": m.n_layer, "kind": "moe" if m.is_moe else "dense",
                    "n_expert": m.n_expert, "n_expert_used": m.n_expert_used, "mode": "waiting",
                    "mtp": a.mtp, "ctx": a.ctx or CTX_STEPS[-1], "estimate_tps": 0, "hops": 0, "net_kb_est": 0,
                    "per_layer_mb": round(m.per_layer_mb()), "size_gb": round(m.size_mb / 1024, 1),
                    "segments": [], "output": {"machine": "master", "label": ""}, "moe_cpu_layers": 0,
                    "machines": [{"id": "master", "role": "master", "host": socket.gethostname(), "layers": 0}]}
        except Exception:  # noqa: BLE001 - a broken file shows up in the log anyway
            info = {"model": Path(model).stem, "mode": "waiting", "segments": [], "machines": []}
        return ServerPlan([], Path(model).stem, key, 0.0, info)

    def start_plan(self, quiet: bool = False, initial: bool = False) -> bool:
        """Plans and starts llama-server; False (state waiting) if the model
        doesn't fit with the workers there are. Only the first plan with
        rebalancing off ends the program instead (exit 2)."""
        a = self.args
        if self.ctx_auto:
            a.ctx = None
        public = (a.host, a.port)
        a.host, a.port = self.big_host, self.big_port
        try:
            sp = build_server_fitting(a, quiet, exit_on_fail=initial and not self.enabled)
        except PlanTooBig as e:
            with self.lock:
                if self.sp is None:
                    self.sp = self.placeholder()
                else:
                    self.sp.info = {**self.placeholder().info}
                self.state, self.reason, self.proc = "waiting", str(e), None
            return False
        finally:
            a.host, a.port = public
        log(" ".join(sp.cmd).replace(sp.key, "***") if sp.key else " ".join(sp.cmd))
        proc = subprocess.Popen(sp.cmd)
        with self.lock:
            self.sp, self.proc, self.state, self.reason = sp, proc, "running", ""
            self.pending, self.now_requested = None, False
            self.evaluated, self.missing = set(), {}
        return True

    def stop_server(self) -> None:
        p = self.proc
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()

    def busy(self) -> bool:
        """Whether llama-server is answering a request right now."""
        import urllib.request

        sp = self.sp
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{self.big_port}/metrics",
                                         headers={"Authorization": f"Bearer {sp.key}"} if sp and sp.key else {})
            with urllib.request.urlopen(req, timeout=2) as r:
                for line in r.read().decode().splitlines():
                    if line.startswith("llamacpp:requests_processing"):
                        return (_num(line.rpartition(" ")[2]) or 0) > 0
        except OSError:
            pass  # loading or gone: nothing to interrupt
        return False

    def replan(self, en: str, de: str) -> None:
        self.event(en, de)
        with self.lock:
            self.state = "switching"
        self.stop_server()
        self.last_switch = time.time()
        if self.start_plan():
            n = len(self.sp.info.get("machines", [])) - 1
            self.event(f"New plan running: {n} worker(s), ~{self.sp.estimate_tps:.1f} tokens/s estimated, "
                       f"-c {self.args.ctx}",
                       f"Neuer Plan läuft: {n} Worker, geschätzt {self.sp.estimate_tps:.1f} Token/s, "
                       f"Kontext {self.args.ctx}")
        else:
            self.event("Model doesn't fit with the workers left; waiting for workers",
                       "Modell passt mit den verbliebenen Workern nicht. Wartet auf Worker.")

    # Switching the model ---------------------------------------------------------

    def model_api(self, body: dict) -> dict:
        import threading

        spec = str(body.get("model") or "").strip()
        if not spec:
            return {"error": "model missing"}
        with self.lock:
            if self.model_job and self.model_job.get("state") in ("resolving", "downloading", "loading"):
                return {"error": "a model change is already running"}
            self.model_job = {"state": "resolving", "model": spec, "t": int(time.time())}
        threading.Thread(target=self._switch_model, args=(spec,), daemon=True).start()
        return self.info()

    def _local_model(self, spec: str) -> Optional[Path]:
        p = Path(spec).expanduser()
        if p.is_file():
            return p.resolve()
        want = spec.lower().removesuffix(".gguf")
        dirs = [CONFIG_DIR / "models"]
        cands = [c for d in dirs if d.is_dir() for c in d.glob("*.gguf")] + [Path(m["path"]) for m in local_models()]
        return next((c.resolve() for c in cands if c.stem.lower().split("-00001-of-")[0] == want), None)

    def _switch_model(self, spec: str) -> None:
        a = self.args
        try:
            path = self._local_model(spec)
            if path is None:
                dest = Path("/srv/models") if os.access("/srv/models", os.W_OK) else CONFIG_DIR / "models"

                def resolved(info: dict) -> None:
                    with self.lock:
                        self.model_job.update(state="downloading", files=info["files"], dir=info["dir"],
                                              size_mb=info["size_mb"], name=info["name"])

                path = fetch_model(spec, dest, quiet=True, on_resolved=resolved)
            path = Path(path).resolve()
            with self.lock:
                self.model_job.update(state="loading", path=str(path))
            self.event(f"Switching the model to {path.name}", f"Wechsle das Modell: {path.name}")
            save_model_choice(a, str(path))
            a.model = str(path)
            if a.mtp and not gguf_info(str(path), CTX_STEPS[-1]).n_nextn:
                log("The new model has no MTP layers: MTP off")
                a.mtp = 0
            setup_draft(a)
            self.replan(f"Loading {path.name}", f"Lade {path.name}.")
            with self.lock:
                ok = self.state == "running"
                self.model_job.update(state="done" if ok else "error",
                                      error="" if ok else (self.reason or "doesn't fit"))
            if ok and self.on_new_model:
                self.on_new_model(self.sp)
        except (ModelError, PlanTooBig) as e:
            with self.lock:
                self.model_job.update(state="error", error=str(e))
        except Exception as e:  # noqa: BLE001 - report it in the dashboard
            with self.lock:
                self.model_job.update(state="error", error=f"{type(e).__name__}: {e}")

    def model_progress(self) -> Optional[dict]:
        j = self.model_job
        if not j:
            return None
        out = {k: v for k, v in j.items() if k not in ("files",)}
        if j.get("state") == "downloading" and j.get("files"):
            done = 0
            for f in j["files"]:
                dest = Path(j["dir"]) / Path(f["file"]).name
                part = dest.with_suffix(dest.suffix + ".part")
                done += dest.stat().st_size if dest.exists() else part.stat().st_size if part.exists() else 0
            out["done_mb"] = done // 2**20
        return out

    # Measurements --------------------------------------------------------------

    def on_timings(self, timings: dict) -> None:
        """A request on the big model finished alone: learn from its speed."""
        sp = self.sp
        if self.state != "running" or sp is None or not sp.units or sp.model_info is None:
            return
        spp = timings_pass_seconds(timings)
        if spp is None:
            return
        rec = self.calib.observe(sp.units, sp.model_info, sp.out_label, spp)
        if rec is None:
            return
        now = time.time()
        if now - self.last_calib_save > 60:
            self.last_calib_save = now
            self.calib.save()
        if rec["n"] in (3, 10, 30):
            est = sp.estimate_tps
            self.event(f"Measured {1/rec['t']:.1f} passes/s on this plan (estimate {est:.1f}, {rec['n']} requests)",
                       f"Gemessen: {1/rec['t']:.1f} Durchläufe/s mit diesem Plan (Schätzung {est:.1f}, "
                       f"{rec['n']} Anfragen)")
        if rec["n"] >= 5 and now - self.last_calib_check > 120:
            self.last_calib_check = now
            self.check_calibrated(rec)

    def check_calibrated(self, rec: dict) -> None:
        """Would another split be faster with what the measurements say?"""
        sp, a = self.sp, self.args
        if not self.enabled or self.pending or sp.info.get("mode") != "layers":
            return
        r = estimate_plan_full(sp.units, sp.model_path, a.ctx, sp.out_label, a.workers == "auto", self.calib)
        if r is None:
            return
        tps_new, trial, m = r
        new_sig = Calibration.signature(trial)
        if new_sig == Calibration.signature(sp.units):
            return
        cur = 1 / rec["t"]
        measured_new = self.calib.measured(trial)
        if measured_new:
            tps_new = 1 / measured_new  # tried before: trust the measurement
        else:
            # compare model with model: the fit reproduces this plan's measurement
            tps_new = tps_new * cur * self.calib.predicted(sp.units, sp.model_info, sp.out_label)
        if tps_new < cur * REBALANCE_GAIN:
            return
        with self.lock:
            self.pending = {"kind": "gain", "t": int(time.time()),
                            "text": f"Messung: andere Aufteilung schneller ({cur:.1f} → {tps_new:.1f} Durchläufe/s)"}
        self.event(f"Measurements: another split looks faster ({cur:.1f} -> {tps_new:.1f}); re-planning when idle",
                   f"Messung: Eine andere Aufteilung ist voraussichtlich schneller ({cur:.1f} → {tps_new:.1f}). "
                   "Neuverteilung, sobald der Server frei ist.")

    def calib_info(self) -> dict:
        sp = self.sp
        if sp is None or not sp.units or sp.model_info is None:
            return {}
        a, _ = Calibration.terms(sp.units, sp.model_info, sp.out_label)
        machines: dict[str, list] = {}
        for u in sp.units:
            if u.key in a:
                mid = u.endpoint.rsplit(":", 1)[0] if u.remote else "master"
                machines.setdefault(mid, []).append((a[u.key], self.calib.k.get(u.key, 1.0)))
        rec = self.calib.plans.get(Calibration.signature(sp.units))
        return {"measured_tps": round(1 / rec["t"], 2) if rec else None, "n": rec["n"] if rec else 0,
                "machines": {mid: round(sum(w for w, _ in v) / sum(w * k for w, k in v), 2)
                             for mid, v in machines.items() if sum(w for w, _ in v)}}

    # Watching the LAN --------------------------------------------------------

    def _lan_workers(self) -> Optional[list[Peer]]:
        a = self.args
        try:
            peers = scan(3.0)
        except Exception:  # noqa: BLE001 - mDNS trouble: try again next time
            return None
        me = machine_id()
        peers = [p for p in peers if p.kind == "worker" and p.txt.get("id") != me
                 and p.txt.get("v", RPC_VERSION) == RPC_VERSION]
        if a.workers not in ("auto", "all", "off"):
            wanted = set()
            for ep in a.workers.split(","):
                h, _, port = ep.strip().rpartition(":")
                wanted.add(ep.strip() if h else f"{ep.strip()}:{DEFAULT_RPC_PORT}")
            peers = [p for p in peers if p.endpoint in wanted]
        with self.lock:
            self.lan_seen = {p.endpoint for p in peers}
            self.ever_seen |= self.lan_seen
        for p in peers:
            self.known.note(p.txt.get("id") or p.endpoint, "worker", p.txt.get("model") or p.name, p.host,
                            {**p.txt, "port": p.port, "model": ""})
        return peers

    def wake_workers(self) -> None:
        """Model doesn't fit: wake the known workers that are not in the LAN."""
        seen_hosts = {ep.rsplit(":", 1)[0] for ep in self.lan_seen}
        names = []
        for mid, rec in list(self.known.machines.items()):
            if rec.get("kind") == "worker" and rec.get("ip") not in seen_hosts and self.known.wake(mid):
                names.append(rec.get("name") or rec["ip"])
                seen_hosts.add(rec.get("ip"))  # a PC's GPU and RAM worker share one wake-up
        if names:
            self.event(f"Waking workers (Wake-on-LAN): {', '.join(names)}",
                       f"Wecke Worker per Wake-on-LAN: {', '.join(names)}")

    def _alive(self, ep: str, machines: list[dict]) -> bool:
        import urllib.request

        if ep in self.lan_seen:
            return True
        mach = next((m for m in machines if ep in m.get("endpoints", [])), {})
        if mach.get("status_port"):
            try:
                with urllib.request.urlopen(f"http://{mach['host']}:{mach['status_port']}/status", timeout=3):
                    return True
            except OSError:
                return False
        # a phone given by address, mDNS blocked: llama-server's connection
        return ep not in self.ever_seen and tcp_connected(ep)

    def check_workers(self) -> None:
        a, sp = self.args, self.sp
        if a.workers == "off" or sp is None:
            return
        peers = self._lan_workers()
        if peers is None:
            return
        used = set(sp.endpoints())
        machines = sp.info.get("machines", [])
        # A worker that now announces another address (cable plugged in):
        # re-plan onto it as soon as the server is idle.
        moved = []
        for u in sp.units:
            if not (u.remote and u.layers and u.key.startswith("worker/")):
                continue
            wid, port = u.key.split("/")[1], u.endpoint.rsplit(":", 1)[1]
            peer = next((p for p in peers if (p.txt.get("id") or p.host) == wid and str(p.port) == port), None)
            if peer and peer.endpoint != u.endpoint:
                link = {"wired": "Kabel", "wifi": "WLAN"}.get(peer.txt.get("link", ""), "neue Adresse")
                moved.append(f"{u.label}: {u.endpoint} → {peer.endpoint} ({link})")
        if moved and not self.pending:
            with self.lock:
                self.pending = {"kind": "moved", "t": int(time.time()), "text": "Verbindung gewechselt: " + "; ".join(moved)}
            self.event(f"Worker connection changed: {'; '.join(moved)}; re-planning when idle",
                       f"Verbindung gewechselt: {'; '.join(moved)}. Neuverteilung, sobald der Server frei ist.")
            return
        for ep in used:
            if self._alive(ep, machines):
                self.missing.pop(ep, None)
            else:
                self.missing[ep] = self.missing.get(ep, 0) + 1
        lost = sorted(ep for ep, n in self.missing.items() if n >= REBALANCE_LOST_SCANS)
        if lost and not (self.pending and self.pending.get("lost")):
            with self.lock:
                self.pending = {"kind": "lost", "lost": lost, "t": int(time.time()),
                                "text": f"Worker nicht mehr erreichbar: {', '.join(lost)}"}
            self.event(f"Worker(s) gone: {', '.join(lost)}; re-planning when idle",
                       f"Worker nicht mehr erreichbar: {', '.join(lost)}. Neuverteilung, sobald der Server frei ist.")
            return
        if sp.info.get("mode") != "layers" or not sp.units or self.pending:
            return
        new = [p for p in peers if p.endpoint not in used]
        key = frozenset(p.endpoint for p in new)
        if not new or key in self.evaluated:
            return
        self.evaluated.add(key)
        self.evaluate(new)

    def evaluate(self, new: list[Peer]) -> None:
        sp, a = self.sp, self.args
        extra = [u for u in _remote_units(new, sp.model_info.per_layer_mb()) if u.capacity_layers or any(u.mem_mb)]
        if not extra:
            return
        prune = a.workers == "auto"
        cur_ctx = a.ctx
        cur = estimate_plan(sp.units, sp.model_path, cur_ctx, sp.out_label, prune) or sp.estimate_tps
        names = ", ".join(u.label for u in extra)
        noun = "Neuer Worker" if len(extra) == 1 else "Neue Worker"
        for ctx in (CTX_STEPS if self.ctx_auto else (cur_ctx,)):
            if ctx < cur_ctx:
                break
            tps = estimate_plan(sp.units + extra, sp.model_path, ctx, sp.out_label, prune)
            if tps is None:
                continue
            if ctx > cur_ctx:
                if tps < cur * 0.8:  # a longer context, but much slower: try a shorter one
                    continue
                why = f"Kontext {cur_ctx} → {ctx}"
            elif tps >= cur * REBALANCE_GAIN:
                why = f"geschätzt {cur:.1f} → {tps:.1f} Token/s"
            else:
                log(f"New worker(s) {names}: ~{tps:.1f} vs {cur:.1f} tokens/s now; not worth a reload")
                return
            with self.lock:
                self.pending = {"kind": "gain", "t": int(time.time()), "workers": [u.endpoint for u in extra],
                                "text": f"{noun} {names}: {why}", "tps": round(tps, 1), "ctx": ctx}
            self.event(f"New worker(s) {names}: {why}; re-planning when idle",
                       f"{noun} {names}: {why}. Neuverteilung, sobald der Server frei ist.")
            return

    # Main loop -----------------------------------------------------------------

    def run(self, advs: list, also_stop: list) -> None:
        def stop(*_):
            self.stopping = True

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            while not self.stopping:
                now = time.time()
                p = self.proc
                if self.state == "running" and p is not None and p.poll() is not None:
                    if not self.enabled:
                        self.exit_code = p.returncode or 1
                        break
                    time.sleep(3)  # let a lost worker drop out of the listings
                    self.replan(f"llama-server ended (code {p.returncode}); planning again",
                                f"Server beendet (Code {p.returncode}). Plane neu.")
                    continue
                if self.state == "waiting":
                    if now - self.last_scan > REBALANCE_SCAN_S:
                        self.last_scan = now
                        peers = self._lan_workers() or []
                        self.wake_workers()
                        seen = frozenset(p.endpoint for p in peers)
                        if seen != self.tried_with or now - self.last_try > REBALANCE_RETRY_S:
                            self.tried_with, self.last_try = seen, now
                            if self.start_plan(quiet=True):
                                n = len(self.sp.info.get("machines", [])) - 1
                                self.event(f"Model fits now ({n} worker(s)); server started",
                                           f"Modell passt jetzt ({n} Worker). Server gestartet.")
                elif self.state == "running" and self.enabled:
                    if now - self.last_scan > REBALANCE_SCAN_S:
                        self.last_scan = now
                        try:
                            self.check_workers()
                        except Exception as e:  # noqa: BLE001 - watching must not stop serving
                            log(f"Rebalance check: {e}")
                    pend = self.pending
                    due = self.now_requested or (pend and (pend["kind"] in ("lost", "moved")
                                                           or now - self.last_switch > REBALANCE_MIN_GAP_S))
                    if due and not self.busy():
                        if self.now_requested:
                            self.replan("Re-planning (requested)", "Neuverteilung von Hand angestoßen.")
                        else:
                            self.replan(f"Re-planning: {pend['text']}", f"Verteile neu: {pend['text']}")
                elif self.state == "running" and self.now_requested and not self.busy():
                    self.replan("Re-planning (requested)", "Neuverteilung von Hand angestoßen.")
                time.sleep(1)
        finally:
            self.state = "stopped"
            self.calib.save()
            for q in [self.proc] + list(also_stop):
                if q is not None and q.poll() is None:
                    q.terminate()
            for q in [self.proc] + list(also_stop):
                if q is not None:
                    try:
                        q.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        q.kill()
            for adv in advs:
                adv.stop()
        sys.exit(0 if self.stopping else self.exit_code)


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
            k = key() if callable(key) else key  # a worker gets its key while running (pairing)
            if not k or path in public:
                return True
            got = self.headers.get("Authorization", "")
            return secrets.compare_digest(got.encode(), f"Bearer {k}".encode())

        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _route(self, routes: dict, post: bool = False) -> None:
            path = self.path.split("?", 1)[0]
            fn = routes.get(path)
            if fn is None:
                return self._send(404, b'{"error":"not found"}')
            if not self._authorized(path):
                return self._send(401, b'{"error":"API key required"}')
            if post:
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(n)) if n else {}
                except ValueError:
                    body = {}
                body = body if isinstance(body, dict) else {}
                if path.endswith("/pair"):  # pairing checks who asks
                    body["_client"] = self.client_address[0]
                res = fn(body)
            else:
                res = fn()
            if isinstance(res, tuple):
                return self._send(200, res[0], res[1])
            self._send(200, json.dumps(res).encode())

        def do_GET(self):
            self._route(get_routes)

        def do_POST(self):
            self._route(post_routes or {}, post=True)

    return H


def start_worker_status(args, rpc_info: list[dict], procs: list[subprocess.Popen],
                        devices: list[Device], pairing: "Pairing", advs: Optional[list] = None) -> None:
    stats = SystemStats(devices)

    def status() -> dict:
        rpc = []
        for info, p in zip(rpc_info, procs):
            rpc.append(dict(info, alive=p.poll() is None, peers=tcp_peers(info["port"])))
        key = pairing.key
        return {"role": "worker", "host": socket.gethostname(), "version": RPC_VERSION, "release": own_version(),
                "rpc": rpc, "stats": stats.sample(), "address": NET_STATE["address"], "link": NET_STATE["link"],
                "switchable": bool(key) and can_switch_role(), "keyed": bool(key), "update": update_brief(),
                **pairing.status()}

    # Status stays open (no secrets in it); switching the role or updating needs the cluster key.
    keyed = {"/role": lambda body: request_role(str(body.get("role") or "")), "/update": request_update}
    post = dict(keyed) if pairing.key else {"/pair": pairing.pair}

    def paired(_key: str) -> None:  # from now on the master may switch and update this PC
        post.pop("/pair", None)
        post.update(keyed)
        for adv in advs or []:
            adv.move(adv.address, {k: v for k, v in adv.txt.items() if k != "pair"})

    pairing.on_paired = paired
    if _start_http(args.host, args.status_port, _json_handler({"/status": status}, post, key=lambda: pairing.key,
                                                               public=("/status",))):
        log(f"Status for the master's dashboard on port {args.status_port}")


BENCH_TIMEOUT_S = 120  # the dashboard's test, per request


class Dashboard:
    """Collects the cluster's state every 2 s for the master's dashboard."""

    def __init__(self, args, sup: "Supervisor", devices: list[Device],
                 router: Optional["Router"] = None, big_base: Optional[str] = None,
                 hosts: Optional["HostManager"] = None):
        import threading

        self.args, self.sup, self.router, self.hosts = args, sup, router, hosts
        self.base = big_base or f"http://127.0.0.1:{args.port}"
        self._fast_last: Optional[tuple[float, float]] = None
        self._fast_tps_last = 0.0
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
        self.fast_state: dict = {}
        self.updater: Optional["UpdateCoordinator"] = None

    @property
    def sp(self) -> "ServerPlan":
        return self.sup.sp

    @property
    def proc(self) -> Optional[subprocess.Popen]:
        return self.sup.proc

    # Collection ------------------------------------------------------------

    def _llama(self) -> dict:
        import urllib.request

        out = {"status": "starting", "metrics": {}}
        if self.sup.state in ("waiting", "switching"):
            out["status"] = self.sup.state
            return out
        if self.proc is None or self.proc.poll() is not None:
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

        if not w.get("status_port"):
            # Phones have no status endpoint, and probing their RPC port
            # doesn't work while the master is connected (it serves one
            # client at a time). llama-server's connection and mDNS tell.
            connected = any(tcp_connected(ep) for ep in w["endpoints"])
            listed = any(ep in self.sup.lan_seen for ep in w["endpoints"])
            return {"online": connected or listed, "connected": connected, "phone": True, "rtt_ms": None}
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(f"http://{w['host']}:{w['status_port']}/status", timeout=1.5) as r:
                data = json.loads(r.read())
            return {"online": True, "rtt_ms": round((time.perf_counter() - t0) * 1000, 1), **data,
                    "ctl": f"{w['host']}:{w['status_port']}"}
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
        self.fast_state = self._fast() if self.router and self.router.fast else {}
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
            h.setdefault("tps_fast", [None] * (len(h["t"]) - 1)).append(
                round(self.fast_state.get("tps_now", 0.0), 2) if self.fast_state else None)
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
            for k in ("t", "tps", "net_kbps", "tps_fast"):
                del h[k][:-HISTORY_POINTS]
            for k in h["gpu"]:
                del h["gpu"][k][:-HISTORY_POINTS]

    def _is_me(self, p: Peer) -> bool:
        """This master's own announcements: its workers (same machine id)
        and its own API (by id, or by address and port from older versions)."""
        if p.txt.get("id") == machine_id():
            return True
        return p.kind == "server" and p.port == self.args.port and p.host in (
            "127.0.0.1", self.args.advertise_ip, lan_address())

    def scan_lan(self) -> None:
        used = {ep for w in self.sp.info.get("machines", []) for ep in w.get("endpoints", [])}
        try:
            peers = scan(3.0)
        except Exception:  # noqa: BLE001 - mDNS problems must not stop the dashboard
            return
        lan = [{"kind": p.kind, "name": p.txt.get("model") or p.name, "endpoint": p.endpoint,
                "ram_mb": p.ram_mb, "bw": p.bandwidth, "type": p.txt.get("kind", "phone" if p.kind == "worker" else ""),
                "in_use": p.endpoint in used, "compatible": p.txt.get("v", RPC_VERSION) == RPC_VERSION,
                "ctl": f"{p.host}:{p.txt['stat']}" if p.txt.get("stat") else ""}
               for p in peers if not self._is_me(p) and p.kind != "host"]  # hosts have their own panel
        seen_ips = {p.host for p in peers}
        used_ips = {ep.rsplit(":", 1)[0] for ep in used}
        if self.hosts:
            # A PC that became a host: its old worker entry is not a sleeping worker
            host_info = self.hosts.info()["hosts"]
            seen_ips |= {h["addr"].rsplit(":", 1)[0] for h in host_info}
            host_macs = {(self.sup.known.machines.get(h["id"]) or {}).get("mac") for h in host_info} - {None, ""}
            self.hosts.worker_ips = {p.host for p in peers if p.kind == "worker" and p.txt.get("stat")}
        else:
            host_macs = set()
        for mid, rec in self.sup.known.machines.items():
            if rec.get("kind") == "worker" and rec.get("mac") and rec.get("mac") in host_macs:
                continue
            if rec.get("kind") == "worker" and rec.get("ip") not in seen_ips and rec.get("ip") not in used_ips:
                seen_ips.add(rec.get("ip"))  # GPU and RAM worker of one PC: one row, one wake-up
                lan.append({"kind": "worker", "name": rec.get("name") or rec["ip"], "endpoint": rec["ip"],
                            "ram_mb": 0, "bw": 0, "type": "off", "in_use": False, "compatible": True,
                            "wake_id": mid, "waking": self.sup.known.recently_woken(mid)})
        with self.lock:
            self.lan = lan

    def loop(self) -> None:
        last_scan = 0.0
        while not self.sup.stopping:
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

    def set_router(self, body: dict) -> dict:
        if not self.router:
            return {"available": False}
        return self.router.set(body)

    def _fast(self) -> dict:
        """The router's fast model: status and generation speed."""
        import urllib.request

        r = self.router
        out = {"status": r.fast.status, "tps_now": 0.0, "tps_last": self._fast_tps_last, "tokens": None}
        try:
            req = urllib.request.Request(r.fast.base + "/metrics",
                                         headers={"Authorization": f"Bearer {r.fast.key}"} if r.fast.key else {})
            with urllib.request.urlopen(req, timeout=2) as resp:
                for line in resp.read().decode().splitlines():
                    if line.startswith("llamacpp:tokens_predicted_total"):
                        out["tokens"] = _num(line.rpartition(" ")[2])
        except OSError:
            return out
        now = time.time()
        if out["tokens"] is not None:
            if self._fast_last and out["tokens"] > self._fast_last[1]:
                out["tps_now"] = (out["tokens"] - self._fast_last[1]) / max(1e-3, now - self._fast_last[0])
                self._fast_tps_last = out["tps_now"]
                out["tps_last"] = self._fast_tps_last
            self._fast_last = (now, out["tokens"])
        return out

    def status(self) -> dict:
        router = None
        if self.router:
            router = {**self.router.info(), **{"fast_live": self.fast_state}}
        with self.lock:
            self._bench_watchdog()
            return {"router": router or {"available": False}, "rebalance": self.sup.info(),
                    "calibration": self.sup.calib_info(),
                    "hosts": self.hosts.info() if self.hosts else {"hosts": [], "catalog": [], "off": True},
                    "plan": self.sp.info, "master": {**self.state, "uptime_s": int(time.time() - self.started),
                                                      "host": socket.gethostname()},
                    "workers": self.workers, "lan": self.lan, "history": self.history,
                    "bench": self.bench, "models": self.models_list(),
                    "update": self.updater.info() if self.updater
                    else {"installed": False, "release": own_version(), "rpc": RPC_VERSION}}

    def run_bench(self) -> dict:
        import threading

        with self.lock:
            self._bench_watchdog()
            if self.bench.get("running"):
                return self.bench
            if self.sup.state != "running" or (self.state.get("llama") or {}).get("status") != "ready":
                self.bench = {"running": False, "error": "Der Server ist gerade nicht bereit (lädt oder verteilt neu)."}
                return self.bench
            self.bench = {"running": True, "started": int(time.time())}

        def go():
            try:
                r = measure(self.base, self.sp.key, None, 0.7, 64, 0, 1, timeout=BENCH_TIMEOUT_S)
                self.sup.on_timings(r.get("timings") or {})  # the test goes past the router: learn from it here
                res = {"running": False, "finished": int(time.time()), "tps": round(r["tps"], 1),
                       "kb_per_token": round((r["per_rx"] + r["per_tx"]) / 1024, 1),
                       "acceptance": r["acceptance"]}
            except BaseException as e:  # noqa: BLE001 - any failure must end "running", or the button stays grey
                why = "Server antwortet nicht" if isinstance(e, OSError) else "Messung fehlgeschlagen"
                res = {"running": False, "finished": int(time.time()), "error": f"{why} ({type(e).__name__})."}
                log(f"Dashboard test: {type(e).__name__}: {e}")
            with self.lock:
                self.bench = res

        threading.Thread(target=go, daemon=True).start()
        return self.bench

    def switch_role(self, body: dict) -> dict:
        """Asks a worker or host PC to change its role (it needs the
        installer's switch; the request carries the cluster key)."""
        import urllib.error
        import urllib.request

        ctl, role = str(body.get("ctl") or ""), str(body.get("role") or "")
        allowed = {w.get("ctl") for w in self.workers.values() if w.get("ctl")}
        allowed |= {d.get("ctl") for d in self.lan if d.get("ctl")}
        if self.hosts:
            allowed |= {h["addr"] for h in self.hosts.info()["hosts"]}
        if ctl not in allowed:
            return {"ok": False, "error": "unknown machine"}
        try:
            req = urllib.request.Request(f"http://{ctl}/role", data=json.dumps({"role": role}).encode(),
                                         headers={"Authorization": f"Bearer {self.sp.key}",
                                                  "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as r:
                res = json.loads(r.read())
        except urllib.error.HTTPError as e:
            res = {"ok": False, "error": "the machine has another key (or none): run the install script on it "
                                         "with this master's key" if e.code in (401, 404) else f"HTTP {e.code}"}
        except (OSError, ValueError) as e:
            res = {"ok": False, "error": f"not reachable: {e}"}
        if res.get("ok"):
            self.sup.event(f"{ctl} switches to {role}", f"{ctl} wird {'Host' if role == 'host' else 'Worker'}.")
        return res

    def models_list(self) -> dict:
        """Models to offer in the dashboard: on this PC and from the catalog (cached for a minute)."""
        now = time.time()
        if now - getattr(self, "_models_at", 0) > 60:
            self._models_at = now
            try:
                local = [{"name": Path(m["path"]).stem, "path": m["path"], "size_gb": round(m["size_mb"] / 1024, 1)}
                         for m in local_models()]
                extra = CONFIG_DIR / "models"
                if extra.is_dir():
                    local += [{"name": p.stem, "path": str(p), "size_gb": round(p.stat().st_size / 2**30, 1)}
                              for p in extra.glob("*.gguf") if not any(m["path"] == str(p) for m in local)]
            except Exception:  # noqa: BLE001
                local = []
            self._models = {"local": local, "catalog": [{"key": k, "desc": d} for k, _r, _q, d in CATALOG]}
        return self._models

    def _bench_watchdog(self) -> None:
        """A test that hangs (server stuck) frees the button after a while."""
        b = self.bench
        if b.get("running") and time.time() - b.get("started", 0) > 3 * BENCH_TIMEOUT_S:
            self.bench = {"running": False, "finished": int(time.time()), "error": "Test abgebrochen: keine Antwort."}

    def start(self) -> Optional[str]:
        import threading

        html_path = next((p for p in (BIN_DIR / "dashboard.html", Path(__file__).resolve().parent / "dashboard.html")
                          if p.exists()), None)
        page = html_path.read_bytes() if html_path else b"<p>dashboard.html missing</p>"
        handler = _json_handler({"/": lambda: (page, "text/html; charset=utf-8"),
                                 "/api/status": self.status},
                                {"/api/bench": lambda body: self.run_bench(),
                                 "/api/router": self.set_router,
                                 "/api/rebalance": self.sup.api,
                                 "/api/role": self.switch_role,
                                 "/api/model": self.sup.model_api,
                                 "/api/update": lambda b: self.updater.api(b) if self.updater else {"ok": False},
                                 "/api/wake": lambda b: {"ok": self.sup.known.wake(str(b.get("id") or ""), force=True)},
                                 "/api/hosts": self.hosts.api if self.hosts else (lambda b: {"error": "no hosts"})},
                                key=self.sp.key, public=("/",))
        if not _start_http(self.args.host, self.args.dashboard_port, handler):
            return None
        threading.Thread(target=self.loop, daemon=True).start()
        return f"{self.args.dashboard_port}"


# ── Wake-on-LAN and sleep ────────────────────────────────────────────────────
#
# Workers and hosts announce their MAC address; the master remembers every
# machine it has seen (known.json), so it can wake one that is off or
# asleep: a worker when the model doesn't fit without it, a host when a
# request asks for its model. Hosts can sleep after a while without
# requests: the master tells them to, they stop llama-server and suspend
# (through a small root unit, see myrmid-suspend.path), and after
# waking up they load their model again.

KNOWN_STATE = CONFIG_DIR / "known.json"
WAKE_WAIT_S = 240  # how long a request waits for a host to wake up and load
WAKE_REPEAT_S = 300  # don't wake the same machine more often (unless asked)
SUSPEND_FILE = f"{RUN_DIR}/suspend"


def iface_info(ip: str) -> tuple[str, str]:
    """(MAC, broadcast address) of the interface that has this IPv4 address."""
    import fcntl

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for _i, name in socket.if_nameindex():
            req = struct.pack("256s", name[:15].encode())
            try:
                addr = socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915, req)[20:24])  # SIOCGIFADDR
            except OSError:
                continue
            if addr != ip:
                continue
            mac = _read(Path(f"/sys/class/net/{name}/address")).strip()
            try:
                bcast = socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8919, req)[20:24])  # SIOCGIFBRDADDR
            except OSError:
                bcast = ""
            wired = not Path(f"/sys/class/net/{name}/wireless").exists()
            return (mac if mac and mac != "00:00:00:00:00:00" and wired else ""), bcast
    finally:
        s.close()
    return "", ""


def send_magic(mac: str, ip: str = "", bcast: str = "") -> bool:
    """Wake-on-LAN: the magic packet to the subnet's broadcast, the global
    broadcast and the last address (ports 9 and 7)."""
    hexmac = re.sub(r"[^0-9a-fA-F]", "", mac)
    if len(hexmac) != 12:
        return False
    packet = b"\xff" * 6 + bytes.fromhex(hexmac) * 16
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sent = False
    for target in [t for t in (bcast, "255.255.255.255", ip) if t]:
        for port in (9, 7):
            try:
                s.sendto(packet, (target, port))
                sent = True
            except OSError:
                pass
    s.close()
    return sent


class Known:
    """Machines the master has seen, with what it takes to wake them."""

    def __init__(self):
        import threading

        self.lock = threading.Lock()
        self.machines: dict[str, dict] = {}
        self.woken: dict[str, float] = {}
        try:
            self.machines = json.loads(KNOWN_STATE.read_text()).get("machines", {})
        except (OSError, ValueError):
            pass

    def note(self, mid: str, kind: str, name: str, ip: str, txt_or_status: dict) -> None:
        mac = txt_or_status.get("mac") or ""
        if not mac:
            return
        rec = {"kind": kind, "name": name, "ip": ip, "mac": mac, "bcast": txt_or_status.get("bcast") or "",
               "seen": int(time.time()), "port": txt_or_status.get("port"),
               "model": txt_or_status.get("model") or "", "api_port": txt_or_status.get("api_port"),
               "path": txt_or_status.get("path") or "", "pc": txt_or_status.get("pc") or ""}
        with self.lock:
            old = self.machines.get(mid)
            changed = not old or any(old.get(k) != rec[k]
                                     for k in ("mac", "ip", "name", "bcast", "port", "model", "path", "pc"))
            if changed or rec["seen"] - old.get("seen", 0) > 3600:
                self.machines[mid] = rec
                self._save()

    def _save(self) -> None:
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            KNOWN_STATE.write_text(json.dumps({"machines": self.machines}, indent=1))
        except OSError:
            pass

    def forget(self, mid: str) -> None:
        with self.lock:
            if self.machines.pop(mid, None) is not None:
                self._save()

    def wake(self, mid: str, force: bool = False) -> bool:
        rec = self.machines.get(mid)
        if not rec:
            return False
        now = time.time()
        if not force and now - self.woken.get(mid, 0) < WAKE_REPEAT_S:
            return False
        self.woken[mid] = now
        ok = send_magic(rec["mac"], rec.get("ip", ""), rec.get("bcast", ""))
        if ok:
            log(f"Wake-on-LAN sent to {rec['name']} ({rec['mac']})")
        return ok

    def recently_woken(self, mid: str) -> bool:
        return time.time() - self.woken.get(mid, 0) < WAKE_WAIT_S


def request_suspend() -> tuple[bool, str]:
    """Suspends this PC: through the root helper unit (a file it watches),
    or directly when allowed (root, or a desktop session)."""
    custom = env("SUSPEND_CMD")  # e.g. "systemctl hibernate"
    if custom:
        import shlex

        r = subprocess.run(shlex.split(custom), capture_output=True, text=True)
        return r.returncode == 0, (r.stderr.strip() or custom)[:200]
    p = Path(SUSPEND_FILE)
    if p.parent.is_dir() and os.access(p.parent, os.W_OK) and \
            Path("/etc/systemd/system/myrmid-suspend.path").exists():
        p.write_text(str(int(time.time())))
        return True, "helper"
    r = subprocess.run(["systemctl", "suspend"], capture_output=True, text=True)
    if r.returncode == 0:
        return True, "systemctl"
    return False, (r.stderr.strip() or "systemctl suspend failed")[:200]


# ── Switching a PC between worker and host ─────────────────────────────────
#
# Both services are installed on a node PC; one runs. The master asks the
# running one (authenticated with the cluster key) to switch; the service has
# no root rights, so it leaves the wanted role in a file that a small root
# unit watches (myrmid-role.path): it stops one service and starts the other.

ROLE_FILE = f"{RUN_DIR}/role"
ROLE_HELPER = Path("/etc/systemd/system/myrmid-role.path")


def can_switch_role() -> bool:
    d = Path(ROLE_FILE).parent
    return ROLE_HELPER.exists() and d.is_dir() and os.access(d, os.W_OK)


def request_role(role: str) -> dict:
    if role not in ("worker", "host"):
        return {"ok": False, "error": "role must be worker or host"}
    if not can_switch_role():
        return {"ok": False, "error": "this PC can't switch roles yet: run the install script on it once "
                                      "(it sets up both services and the switch)"}
    Path(ROLE_FILE).write_text(role)
    log(f"Switching this PC to {role} (the other service starts in a moment)")
    return {"ok": True, "role": role}


# ── Updates ──────────────────────────────────────────────────────────────────
#
# Releases are the git tags vX.Y.Z of the MyrmidLM repo (llama.cpp moves only
# with a release: LLAMA_CPP_COMMIT is part of it). Each PC builds a new
# release in the background into /opt/myrmid/releases/X.Y.Z while its
# services keep running the old one, then switches the link
# /opt/myrmid/current and restarts them. If they don't come up, it switches
# back. The master switches its cluster together, once every PC has the
# release built and no request is running: a master and its workers must
# speak the same RPC protocol, phones included (they get the new app from
# the same GitHub release).
#
# Building and switching need root: myrmid-update.timer (daily) runs
# "myrmid update auto"; the services, without root, ask with a request file
# in /run/myrmid that myrmid-update-request.path hands to a root service.

PREFIX = Path(env("PREFIX", "/opt/myrmid"))
UPDATE_DIR = Path(env("UPDATE_DIR", "/var/lib/myrmid-update"))
UPDATE_STATE = UPDATE_DIR / "state.json"
UPDATE_LOG = UPDATE_DIR / "build.log"
UPDATE_CONF = Path(env("CONF", "/etc/myrmid/myrmid.conf"))
UPDATE_REQUEST = f"{RUN_DIR}/update"
UNIT_DIR = Path(env("UNIT_DIR", "/etc/systemd/system"))
UPDATE_HELPER = UNIT_DIR / "myrmid-update-request.path"
CLUSTER_KEY_FILE = Path(env("CLUSTER_KEY_FILE", "/etc/myrmid/cluster_key"))
DEFAULT_REPO = "https://github.com/Miki84/MyrmidLM.git"
SERVICES = ("myrmid-server", "myrmid-worker", "myrmid-host")
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
RELEASE_RE = re.compile(r"^(\d+\.\d+\.\d+|main-[0-9a-f]{7,40})$")
UPDATE_ACTIONS = ("check", "prepare", "activate", "rollback", "auto")
UPDATE_HEALTH_S = 30  # services must stay up this long after a switch
UPDATE_SELF_AFTER_S = 7 * 86400  # a worker whose master never asks switches itself after a week
MIN_BUILD_FREE_GB = 3


class UpdateError(Exception):
    pass


def own_version() -> str:
    """The release this program belongs to (VERSION next to it, or in the checkout)."""
    for p in (BIN_DIR / "VERSION", Path(__file__).resolve().parent.parent / "VERSION"):
        try:
            v = p.read_text().strip()
            if v:
                return v
        except OSError:
            pass
    return "dev"


def vkey(v: Optional[str]) -> tuple:
    return tuple(int(x) for x in v.split(".")) if v and VERSION_RE.match(v) else (-1,)


def is_newer(a: Optional[str], b: Optional[str]) -> bool:
    """a newer than b: by number for releases; a build of main is newer when it differs."""
    if not a:
        return False
    if VERSION_RE.match(a) and b and VERSION_RE.match(b):
        return vkey(a) > vkey(b)
    return a != b


def read_conf() -> dict:
    conf = {"REPO": DEFAULT_REPO, "CHANNEL": "stable", "AUTO": "on", "BACKENDS": "auto",
            "CUDA_BIN": "", "ROCM_TARGETS": "", "LLAMA_CPP_REPO": ""}
    try:
        for line in UPDATE_CONF.read_text().splitlines():
            k, sep, v = line.strip().partition("=")
            if sep and k and not k.startswith("#"):
                conf[k.strip()] = v.strip().strip('"')
    except OSError:
        pass
    return conf


def write_conf_value(key: str, value: str) -> None:
    lines = []
    try:
        lines = UPDATE_CONF.read_text().splitlines()
    except OSError:
        pass
    lines = [ln for ln in lines if ln.split("=", 1)[0].strip() != key] + [f"{key}={value}"]
    UPDATE_CONF.parent.mkdir(parents=True, exist_ok=True)
    UPDATE_CONF.write_text("\n".join(lines) + "\n")


def read_update_state() -> dict:
    try:
        st = json.loads(UPDATE_STATE.read_text())
        return st if isinstance(st, dict) else {}
    except (OSError, ValueError):
        return {}


def write_update_state(**kw) -> dict:
    """Merges kw into the state (a build and a request may write at the same time)."""
    import fcntl

    UPDATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(UPDATE_DIR / "state.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        st = read_update_state()
        st.update(kw)
        st["at"] = int(time.time())
        st["auto"] = read_conf()["AUTO"] == "on"
        tmp = UPDATE_STATE.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(st, indent=1))
        os.chmod(tmp, 0o644)  # the services (without root) read it
        tmp.replace(UPDATE_STATE)
    return st


def is_bad(st: dict, v: Optional[str]) -> bool:
    """A release that failed here or was rolled back: never switched to automatically again."""
    return bool(v) and (v == st.get("bad") or v in (st.get("bad_all") or []))


def mark_bad(v: str) -> dict:
    st = read_update_state()
    return write_update_state(bad=v, bad_all=sorted(set(st.get("bad_all") or []) | {v}))


def update_brief() -> dict:
    """What a PC tells its master about updates (in /status)."""
    st = read_update_state()
    out = {k: st.get(k) for k in ("current", "latest", "prepared", "phase", "error", "bad", "bad_all", "auto")}
    out["release"] = own_version()
    out["can_update"] = UPDATE_HELPER.exists() and os.access(RUN_DIR, os.W_OK)
    return out


def current_release() -> str:
    link = PREFIX / "current"
    if link.is_symlink():
        return Path(os.readlink(link)).name
    return own_version()


def _releases() -> list[str]:
    d = PREFIX / "releases"
    return sorted((p.name for p in d.iterdir() if p.is_dir() and RELEASE_RE.match(p.name)),
                  key=lambda v: (vkey(v), v)) if d.is_dir() else []


def _git(*a: str, check: bool = True) -> str:
    r = subprocess.run(["git", "-C", str(PREFIX / "src"), *a], capture_output=True, text=True, timeout=900)
    if check and r.returncode:
        raise UpdateError(f"git {' '.join(a)}: {(r.stderr or r.stdout).strip()[-300:]}")
    return r.stdout.strip()


def _systemctl(*a: str) -> subprocess.CompletedProcess:
    return subprocess.run(["systemctl", *a], capture_output=True, text=True)


def _unit_active(u: str) -> bool:
    return shutil.which("systemctl") is not None and _systemctl("is-active", "--quiet", u).returncode == 0


def _unit_enabled(u: str) -> bool:
    return shutil.which("systemctl") is not None and _systemctl("is-enabled", "--quiet", u).returncode == 0


def _lock(name: str):
    """An exclusive lock (None when another update holds it)."""
    import fcntl

    UPDATE_DIR.mkdir(parents=True, exist_ok=True)
    f = open(UPDATE_DIR / f"{name}.lock", "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    return f


def update_check() -> dict:
    """Asks the repo for the newest release (tags vX.Y.Z, or main with CHANNEL=main)."""
    conf = read_conf()
    if not (PREFIX / "src" / ".git").is_dir():
        raise UpdateError(f"{PREFIX / 'src'} is not a git checkout: install with install.sh")
    st = read_update_state()
    write_update_state(phase="checking" if st.get("phase") != "building" else "building")
    try:
        _git("remote", "set-url", "origin", conf["REPO"])
        if conf["CHANNEL"] == "main":
            _git("fetch", "--quiet", "--force", "origin", "+main:refs/remotes/origin/main")
            ref = "origin/main"
            latest = "main-" + _git("rev-parse", "--short=8", ref)
        else:
            _git("fetch", "--quiet", "--force", "--tags", "origin")
            tags = [t[1:] for t in _git("tag", "-l", "v*").split() if VERSION_RE.match(t[1:])]
            latest = max(tags, key=vkey, default=None)
            ref = f"v{latest}" if latest else None
    except (UpdateError, OSError, subprocess.TimeoutExpired) as e:
        write_update_state(phase="idle" if st.get("phase") != "building" else "building",
                           error=f"Prüfen fehlgeschlagen: {e}")
        raise UpdateError(str(e)) from e
    rpc = None
    if ref:
        m = re.search(r'^RPC_VERSION = "([^"]+)"', _git("show", f"{ref}:node/myrmid.py", check=False), re.M)
        rpc = m.group(1) if m else None
    cur = current_release()
    phase = read_update_state().get("phase")
    return write_update_state(phase="building" if phase == "building" else "idle", latest=latest, latest_rpc=rpc,
                              latest_ref=ref, checked=int(time.time()), current=cur, error="",
                              newer=is_newer(latest, cur), channel=conf["CHANNEL"])


def update_prepare(version: Optional[str] = None) -> dict:
    """Builds a release next to the running one (the services keep running)."""
    conf = read_conf()
    st = read_update_state()
    if not version:
        st = update_check()
        if not st.get("newer"):
            return st
        version = st["latest"]
    if not RELEASE_RE.match(version):
        raise UpdateError(f"not a release: {version}")
    dest = PREFIX / "releases" / version
    if dest.is_dir():
        return write_update_state(prepared=version, prepared_at=st.get("prepared_at") or int(time.time()),
                                  phase="idle", error="")
    lock = _lock("build")
    if lock is None:
        return write_update_state()  # another build runs
    try:
        ref = f"v{version}" if VERSION_RE.match(version) else (st.get("latest_ref") or "origin/main")
        if not _git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", check=False):
            _git("fetch", "--quiet", "--force", "--tags", "origin")  # a release the master knows, not yet here
            if not _git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", check=False):
                raise UpdateError(f"release {version} is not in {read_conf()['REPO']}")
        if version.startswith("main-") and not _git("rev-parse", "--short=8", ref).startswith(version[5:]):
            raise UpdateError(f"{version} is no longer the newest commit on main: check again")
        free_gb = shutil.disk_usage(PREFIX).free / 2**30
        if free_gb < MIN_BUILD_FREE_GB:
            raise UpdateError(f"only {free_gb:.1f} GB free under {PREFIX}; building needs {MIN_BUILD_FREE_GB} GB")
        write_update_state(phase="building", building=version, build_started=int(time.time()), error="")
        log(f"Update: building {version} (log: {UPDATE_LOG})")
        _git("checkout", "--quiet", "--force", "--detach", ref)
        _git("clean", "-fdq")  # leaves the ignored build tree: only what changed is built again
        part = dest.with_name(version + ".part")
        shutil.rmtree(part, ignore_errors=True)
        envp = dict(os.environ)
        if conf["CUDA_BIN"]:
            envp["PATH"] = f"{conf['CUDA_BIN']}:{envp.get('PATH', '')}"
        if conf["LLAMA_CPP_REPO"]:
            envp["LLAMA_CPP_REPO"] = conf["LLAMA_CPP_REPO"]
        cmd = ["nice", "-n", "19"] + (["ionice", "-c3"] if shutil.which("ionice") else [])
        cmd += ["bash", str(PREFIX / "src" / "node" / "build.sh"), "--backends", conf["BACKENDS"],
                "--install", str(part)]
        if conf["ROCM_TARGETS"]:
            cmd += ["--rocm-targets", conf["ROCM_TARGETS"]]
        with open(UPDATE_LOG, "w") as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=envp, cwd=str(PREFIX / "src")).returncode
        if rc:
            raise UpdateError(f"Bauen von {version} fehlgeschlagen (Log: {UPDATE_LOG})")
        (part / "bin" / "VERSION").write_text(version + "\n")
        part.rename(dest)
        log(f"Update: {version} is built and ready")
        return write_update_state(prepared=version, prepared_at=int(time.time()), phase="idle", building=None,
                                  error="")
    except (UpdateError, OSError, subprocess.SubprocessError) as e:
        write_update_state(phase="failed", building=None, error=str(e))
        raise UpdateError(str(e)) from e
    finally:
        lock.close()


def _switch_link(version: str) -> None:
    link, tmp = PREFIX / "current", PREFIX / ".current.tmp"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    os.symlink(f"releases/{version}", tmp)
    os.replace(tmp, link)  # atomic: a service starting now gets the old or the new release, never none


def _install_units(release: Path) -> None:
    """The release's own helper units (update, role switch, sleep) and the
    role-switch script; the services themselves stay as the installer wrote them."""
    src = release / "systemd"
    if not src.is_dir():
        return
    changed = False
    for f in sorted(src.iterdir()):
        if f.suffix not in (".path", ".service", ".timer") or f.stem in SERVICES:
            continue
        dst = UNIT_DIR / f.name
        is_update = f.name.startswith("myrmid-update")
        if not (is_update or dst.exists()):
            continue  # not set up on this PC (e.g. sleep on a master)
        if not dst.exists() or dst.read_bytes() != f.read_bytes():
            shutil.copyfile(f, dst)
            changed = True
    role = src / "myrmid-role.sh"
    dst = Path("/usr/local/sbin/myrmid-role")
    if role.is_file() and dst.exists() and dst.read_bytes() != role.read_bytes():
        shutil.copyfile(role, dst)
        os.chmod(dst, 0o755)
    if changed:
        _systemctl("daemon-reload")


def _restarts(u: str) -> int:
    out = _systemctl("show", "-p", "NRestarts", "--value", u).stdout.strip()
    return int(out) if out.isdigit() else 0


def _healthy(units: list[str], before: dict[str, int]) -> tuple[bool, str]:
    end = time.time() + UPDATE_HEALTH_S
    while time.time() < end:
        for u in units:
            state = _systemctl("is-active", u).stdout.strip()
            if state in ("failed", "inactive") or _restarts(u) > before.get(u, 0):
                return False, f"{u} läuft nicht ({state})"
        time.sleep(2)
    for u in units:
        if _systemctl("is-active", u).stdout.strip() != "active":
            return False, f"{u} ist nicht hochgekommen"
    return True, ""


def update_activate(version: Optional[str] = None, *, rollback: bool = False) -> dict:
    """Switches to a built release and restarts the running services; back to
    the previous one if they don't stay up."""
    st = read_update_state()
    version = version or st.get("prepared")
    if not version or not RELEASE_RE.match(version):
        raise UpdateError("no release built: run myrmid update prepare")
    dest = PREFIX / "releases" / version
    if not dest.is_dir():
        raise UpdateError(f"{version} is not built ({dest} missing)")
    prev = current_release()
    if prev == version and (PREFIX / "current").is_symlink():
        return write_update_state(current=version, phase="idle", prepared=None if st.get("prepared") == version
                                  else st.get("prepared"))
    r = subprocess.run([sys.executable, str(dest / "bin" / "myrmid"), "--version"], capture_output=True, text=True)
    if r.returncode:
        raise UpdateError(f"{version} doesn't start: {r.stderr.strip()[-200:]}")
    lock = _lock("switch")
    if lock is None:
        raise UpdateError("another switch is running")
    try:
        running = [u for u in SERVICES if _unit_active(u)]
        write_update_state(phase="activating", activating=version, error="")
        log(f"Update: {prev} -> {version}; restarting {', '.join(running) or 'nothing'}")
        _switch_link(version)
        _install_units(dest)
        before = {u: _restarts(u) for u in running}
        if running:
            _systemctl("restart", *running)
        ok, why = _healthy(running, before) if running else (True, "")
        if not ok:
            log(f"Update: {version} failed ({why}); back to {prev}")
            if (PREFIX / "releases" / prev).is_dir():
                _switch_link(prev)
                _systemctl("restart", *running)
            mark_bad(version)
            write_update_state(phase="failed", activating=None, current=current_release(),
                               error=f"{version}: {why}. Zurück auf {prev}.")
            raise UpdateError(f"{version}: {why}; back on {prev}")
        keep = {version, prev, read_update_state().get("prepared")}
        for old in _releases():
            if old not in keep:
                shutil.rmtree(PREFIX / "releases" / old, ignore_errors=True)
        log(f"Update: now on {version}")
        st = read_update_state()
        bad_all = [v for v in (st.get("bad_all") or []) if v != version]  # switched to by hand: good again
        return write_update_state(phase="idle", activating=None, current=version, previous=prev, error="",
                                  prepared=None if st.get("prepared") == version else st.get("prepared"),
                                  bad=st.get("bad") if rollback or st.get("bad") != version else None,
                                  bad_all=bad_all, newer=is_newer(st.get("latest"), version))
    finally:
        lock.close()


def update_rollback() -> dict:
    st = read_update_state()
    cur = current_release()
    prev = st.get("previous")
    if not prev or prev == cur or not (PREFIX / "releases" / prev).is_dir():
        older = [v for v in _releases() if v != cur]
        prev = older[-1] if older else None
    if not prev:
        raise UpdateError("no earlier release on this PC")
    mark_bad(cur)  # the timer must not switch straight back
    return update_activate(prev, rollback=True)


def update_auto() -> dict:
    """The daily timer: check, build a new release, and switch where nobody
    else does. A master's server switches its cluster itself (when idle); a
    worker or host with the cluster key waits for its master; one without
    the key can't be asked, so it switches itself (at the timer's hour)."""
    conf = read_conf()
    st = update_check()
    if st.get("newer") and not is_bad(st, st.get("latest")):
        st = update_prepare(st["latest"])
    prep = st.get("prepared")
    if conf["AUTO"] != "on" or not prep or prep == st.get("current") or is_bad(st, prep):
        return st
    if _unit_enabled("myrmid-server"):
        if not _unit_active("myrmid-server"):
            return update_activate(prep)  # nothing running to coordinate
        return st  # the server switches the cluster when every PC is ready
    has_key = CLUSTER_KEY_FILE.exists() or any(
        p.exists() for p in (Path("/var/lib/private/myrmid/myrmid/cluster_key"), Path("/var/lib/myrmid/myrmid/cluster_key")))
    if not has_key or time.time() - (st.get("prepared_at") or time.time()) > UPDATE_SELF_AFTER_S:
        return update_activate(prep)
    return st


def update_apply_request() -> dict:
    """Runs what a service asked for (myrmid-update-request.service, as root)."""
    p = Path(UPDATE_REQUEST)
    try:
        req = json.loads(p.read_text())
    except (OSError, ValueError):
        req = {}
    finally:
        p.unlink(missing_ok=True)
    action, version = str(req.get("action") or ""), req.get("version")
    if action not in UPDATE_ACTIONS:
        raise UpdateError(f"unknown request: {action or '(none)'}")
    if version is not None and not RELEASE_RE.match(str(version)):
        raise UpdateError(f"not a release: {version}")
    log(f"Update request: {action} {version or ''}".rstrip())
    if action == "check":
        return update_check()
    if action == "prepare":
        return update_prepare(version)
    if action == "activate":
        return update_activate(version)
    if action == "rollback":
        return update_rollback()
    write_conf_value("AUTO", "on" if req.get("value") else "off")
    return write_update_state()


def request_update(body: dict) -> dict:
    """A service asks the root helper (the master's dashboard, or a master asking a worker)."""
    action, version = str(body.get("action") or ""), body.get("version")
    if action not in UPDATE_ACTIONS:
        return {"ok": False, "error": f"action must be one of {', '.join(UPDATE_ACTIONS)}"}
    if version is not None and not RELEASE_RE.match(str(version)):
        return {"ok": False, "error": "not a release"}
    if not (UPDATE_HELPER.exists() and os.access(RUN_DIR, os.W_OK)):
        return {"ok": False, "error": "this PC can't update itself yet: install it with install.sh"}
    if Path(UPDATE_REQUEST).exists():
        return {"ok": False, "error": "a request is still running"}
    tmp = Path(f"{UPDATE_REQUEST}.tmp")
    tmp.write_text(json.dumps({"action": action, "version": version, "value": bool(body.get("value"))}))
    tmp.replace(UPDATE_REQUEST)
    return {"ok": True}


def cmd_update(args) -> None:
    what = args.what
    if what == "status":
        st = read_update_state()
        if args.json:
            print(json.dumps({**st, "running": own_version(), "releases": _releases()}, indent=1))
            return
        conf = read_conf()
        print(f"Running:   {own_version()}  (installed: {current_release()}, releases: {', '.join(_releases()) or '-'})")
        print(f"Newest:    {st.get('latest') or '?'} ({conf['CHANNEL']}, {conf['REPO']})"
              + (f", checked {time.strftime('%Y-%m-%d %H:%M', time.localtime(st['checked']))}" if st.get("checked") else ""))
        if st.get("prepared"):
            print(f"Built:     {st['prepared']} (switches with the cluster)")
        print(f"State:     {st.get('phase') or 'idle'}" + (f" - {st['error']}" if st.get("error") else ""))
        print(f"Automatic: {'on' if conf['AUTO'] == 'on' else 'off'} (myrmid update auto on|off)")
        return
    if os.geteuid() != 0:
        die(f"Updates build into {PREFIX} and restart the services: run it with sudo")
    try:
        if what == "check":
            st = update_check()
            print(f"Newest: {st.get('latest') or 'none'}; running {current_release()}"
                  + ("; an update is available" if st.get("newer") else ""))
        elif what == "prepare":
            st = update_prepare(args.version)
            print(f"Built: {st.get('prepared') or 'nothing new'}")
        elif what == "activate":
            st = update_activate(args.version)
            print(f"Now on {st.get('current')}")
        elif what == "rollback":
            st = update_rollback()
            print(f"Back on {st.get('current')}")
        elif what == "auto":
            if args.version in ("on", "off"):
                write_conf_value("AUTO", args.version)
                write_update_state()
                print(f"Automatic updates {args.version}")
            else:
                st = update_auto()
                print(f"Running {st.get('current')}; newest {st.get('latest') or '?'}; built {st.get('prepared') or '-'}")
        elif what == "apply-request":
            update_apply_request()
    except UpdateError as e:
        die(str(e))


class UpdateCoordinator:
    """On the master: brings every PC of the cluster to a new release and
    switches them together, when all have it built and no request runs."""

    IDLE_S = 120  # no request for this long (and as long after the start: the LAN scan must find the PCs first)
    WAIT_SWITCH_S = 240  # for each PC to come back on the new release

    def __init__(self, sup: "Supervisor", router: "Router", hosts: Optional["HostManager"],
                 dash: Optional["Dashboard"], key: str):
        import threading

        self.sup, self.router, self.hosts, self.dash, self.key = sup, router, hosts, dash, key
        self.lock = threading.Lock()
        self.nodes: list[dict] = []
        self.status = ""
        self.busy = False
        self.asked: dict[tuple, float] = {}  # (ctl, action, version) -> when
        self.started = time.time()

    # The PCs ----------------------------------------------------------------

    def _get(self, ctl: str, path: str = "/status", timeout: float = 3) -> dict:
        import urllib.request

        req = urllib.request.Request(f"http://{ctl}{path}", headers={"Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def _post(self, ctl: str, body: dict) -> dict:
        import urllib.error
        import urllib.request

        req = urllib.request.Request(f"http://{ctl}/update", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {self.key}",
                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            return {"ok": False, "error": "kein Cluster-Key (oder ein anderer) auf diesem PC" if e.code in (401, 404)
                    else f"HTTP {e.code}"}
        except (OSError, ValueError) as e:
            return {"ok": False, "error": f"nicht erreichbar: {e}"}

    def _ctls(self) -> dict[str, dict]:
        """Every PC worker and host the master knows: ctl address -> facts."""
        out: dict[str, dict] = {}
        for m in self.sup.sp.info.get("machines", []):
            if m.get("role") == "worker" and m.get("status_port") and m.get("host"):
                out[f"{m['host']}:{m['status_port']}"] = {"kind": "worker", "in_plan": True}
        if self.dash:
            with self.dash.lock:
                lan = list(self.dash.lan)
            for d in lan:
                if d.get("ctl"):  # PC workers (phones have no status port)
                    out.setdefault(d["ctl"], {"kind": "worker", "in_plan": bool(d.get("in_use"))})
        if self.hosts:
            for h in self.hosts.info()["hosts"]:
                if not h.get("asleep") and h.get("state") != "unreachable":
                    out[h["addr"]] = {"kind": "host", "in_plan": False, "id": h["id"]}
        return out

    def poll(self) -> list[dict]:
        nodes = []
        for ctl, facts in self._ctls().items():
            try:
                st = self._get(ctl)
            except (OSError, ValueError):
                continue
            up = st.get("update") or {}
            nodes.append({"ctl": ctl, "name": st.get("host") or ctl, **facts,
                          "release": up.get("release") or st.get("release") or "?",
                          "prepared": up.get("prepared"), "phase": up.get("phase") or "",
                          "error": up.get("error") or "", "bad": up.get("bad"), "bad_all": up.get("bad_all") or [],
                          "can_update": bool(up.get("can_update")),
                          "keyed": bool(st.get("keyed") or facts["kind"] == "host")})
        with self.lock:
            self.nodes = nodes
        return nodes

    def _phones_in_plan(self) -> bool:
        return any(m.get("role") == "worker" and not m.get("status_port") for m in self.sup.sp.info.get("machines", []))

    def _idle(self) -> bool:
        r = self.router
        with r.lock:
            running = any(b.inflight for b in r.backends)
        quiet_since = max(getattr(r, "last_request", 0), self.started)
        return not running and time.time() - quiet_since > self.IDLE_S and self.sup.state in ("running", "waiting")

    def _ask(self, n: dict, action: str, version: Optional[str], every_s: float = 1800) -> None:
        k = (n["ctl"], action, version)
        if time.time() - self.asked.get(k, 0) < every_s:
            return
        self.asked[k] = time.time()
        res = self._post(n["ctl"], {"action": action, "version": version})
        if not res.get("ok"):
            log(f"Update: {n['name']} can't {action}: {res.get('error')}")

    # Deciding ---------------------------------------------------------------

    def target(self) -> tuple[Optional[str], dict]:
        st = read_update_state()
        mine = own_version()
        prep = st.get("prepared")
        ok = prep and is_newer(prep, mine) and not is_bad(st, prep) and (PREFIX / "releases" / prep).is_dir()
        return (prep if ok else None), st

    def tick(self, force: bool = False) -> None:
        if self.busy:
            return
        target, st = self.target()
        mine = own_version()
        nodes = self.poll()
        if not st:
            self.status = "Updates sind auf diesem PC nicht eingerichtet (mit install.sh installieren)."
            return
        if not target:
            # PCs behind the master (were off, or failed before): bring them up to its release
            for n in nodes:
                if n["release"] == mine or not n["can_update"] or not n["keyed"] or mine in n["bad_all"]:
                    continue  # (a release that failed on a PC is not pushed onto it again)
                if n["prepared"] == mine and (self._idle() or not n["in_plan"]):
                    self._ask(n, "activate", mine, every_s=600)
                elif n["prepared"] != mine and n["phase"] != "building" and VERSION_RE.match(mine):
                    self._ask(n, "prepare", mine)
            self.status = ""
            return
        for n in nodes:  # let every PC build the release now, not at its own timer
            if n["release"] != target and n["prepared"] != target and n["phase"] != "building" \
                    and n["can_update"] and n["keyed"]:
                self._ask(n, "prepare", target)
        waiting = [n["name"] for n in nodes if n["release"] != target and n["prepared"] != target and n["keyed"]]
        rpc_change = st.get("latest") == target and st.get("latest_rpc") not in (None, RPC_VERSION)
        if not force:
            if not st.get("auto"):
                self.status = f"{target} ist gebaut. Automatisch umstellen ist aus: im Dashboard auf Jetzt umstellen klicken."
                return
            if waiting:
                self.status = f"{target}: warte, bis es auf {', '.join(waiting)} gebaut ist."
                return
            if rpc_change and self._phones_in_plan():
                self.status = (f"{target} bringt ein neues RPC-Protokoll ({st.get('latest_rpc')}): die Handys im Plan "
                               "brauchen erst die neue App. Danach hier auf Jetzt umstellen klicken.")
                return
            if not self._idle():
                self.status = f"{target}: wird umgestellt, sobald {self.IDLE_S // 60} Minuten lang keine Anfrage läuft."
                return
        import threading

        self.busy = True
        threading.Thread(target=self.switch, args=(target, nodes), daemon=True).start()

    def switch(self, target: str, nodes: list[dict], action: str = "activate") -> None:
        try:
            what = "Zurück auf die vorige Version" if action == "rollback" else f"Umstellen auf {target}"
            self.status = f"{what}: die PCs werden neu gestartet ..."
            self.sup.event(f"update: {action} {target}", f"{what} (alle PCs zusammen).")
            todo = [n for n in nodes if n["keyed"] and n["can_update"]
                    and (action == "rollback" or (n["release"] != target and n["prepared"] == target))]
            for n in todo:
                res = self._post(n["ctl"], {"action": action, "version": None if action == "rollback" else target})
                if not res.get("ok"):
                    log(f"Update: {n['name']}: {res.get('error')}")
            end = time.time() + self.WAIT_SWITCH_S
            left = {n["ctl"]: n for n in todo}
            while left and time.time() < end:
                time.sleep(5)
                for ctl, n in list(left.items()):
                    try:
                        st = self._get(ctl, timeout=2)
                    except (OSError, ValueError):
                        continue  # restarting
                    up = st.get("update") or {}
                    rel = up.get("release")
                    if (action == "rollback" and rel and rel != n["release"]) or rel == target \
                            or up.get("phase") == "failed":
                        if up.get("phase") == "failed":
                            log(f"Update: {n['name']} failed ({up.get('error')}); it stays on its release")
                        left.pop(ctl)
            for n in left.values():
                log(f"Update: {n['name']} didn't report back; it is switched when it answers again")
            res = request_update({"action": action, "version": None if action == "rollback" else target})
            if not res.get("ok"):
                self.status = f"Der Master kann sich nicht umstellen: {res.get('error')}"
            else:
                self.status = f"{what}: jetzt der Master (der Server startet neu) ..."
        finally:
            self.busy = False

    # Dashboard --------------------------------------------------------------

    def info(self) -> dict:
        target, st = self.target()
        with self.lock:
            nodes = list(self.nodes)
        return {"release": own_version(), "rpc": RPC_VERSION, "installed": bool(st), "target": target,
                "latest": st.get("latest"), "latest_rpc": st.get("latest_rpc"), "prepared": st.get("prepared"),
                "phase": st.get("phase") or "", "error": st.get("error") or "", "auto": bool(st.get("auto")),
                "checked": st.get("checked"), "channel": st.get("channel") or "stable",
                "previous": st.get("previous"), "status": self.status, "busy": self.busy,
                "phones_in_plan": self._phones_in_plan(), "nodes": nodes,
                "can_update": UPDATE_HELPER.exists() and os.access(RUN_DIR, os.W_OK)}

    def api(self, body: dict) -> dict:
        action = str(body.get("action") or "")
        if action == "check":  # check and build: here and on every PC
            res = request_update({"action": "prepare"})
            for n in self.nodes:
                if n["keyed"] and n["can_update"] and n["phase"] != "building":
                    self._ask(n, "prepare", None, every_s=0)
        elif action == "now":
            target, _ = self.target()
            if not target:
                return {"ok": False, "error": "keine neue Version gebaut", **self.info()}
            self.tick(force=True)
            res = {"ok": True}
        elif action == "rollback":
            if self.busy:
                return {"ok": False, "error": "ein Umstellen läuft gerade", **self.info()}
            import threading

            self.busy = True
            threading.Thread(target=self.switch, args=(own_version(), self.poll(), "rollback"), daemon=True).start()
            res = {"ok": True}
        elif action == "auto":
            res = request_update({"action": "auto", "value": bool(body.get("value"))})
        else:
            res = {"ok": False, "error": "unknown action"}
        return {**res, **self.info()}

    def loop(self) -> None:
        time.sleep(20)  # let the dashboard see the workers first
        while not self.sup.stopping:
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001 - updates must never stop the server
                log(f"Update: {type(e).__name__}: {e}")
            time.sleep(30)

    def start(self) -> None:
        import threading

        threading.Thread(target=self.loop, daemon=True).start()


# ── Hosts ────────────────────────────────────────────────────────────────────
#
# A host is a PC that loads a whole model on its own devices and answers for
# a master's router: three PCs with one A4000 each serve three models, or
# the same model three times with the requests spread over them. The master
# decides which host loads what (dashboard or --assign); the host downloads
# the model itself, plans it on its own GPUs and RAM, and starts llama-server
# with the cluster key (the master's API key). Its control API needs the
# same key: whoever has it can make the host download and run models.

HOST_CTL_PORT = 50070
HOST_API_PORT = 8081
HOST_STATE = CONFIG_DIR / "host.json"
HOST_LAYOUT = CONFIG_DIR / "host-layout.json"
HOST_INTERNAL_PORT = 18081  # the llama-servers listen on 127.0.0.1 from here up; the API port forwards to them
HOST_LAYOUTS = ("split", "per_gpu")  # one model over all GPUs, or one model per GPU


PAIRED_KEY = CONFIG_DIR / "cluster_key"  # the key a master gave this PC (pairing)
PAIR_WINDOW_S = 3600  # without --pair-from, the first hour after the start


def read_cluster_key(key: Optional[str], key_file: Optional[str]) -> str:
    """--key, --key-file, the systemd credential cluster_key, or the key a
    master gave this PC when it took it into its cluster."""
    if key:
        return key.strip()
    paths = [Path(key_file).expanduser()] if key_file else []
    if os.environ.get("CREDENTIALS_DIRECTORY"):
        paths.append(Path(os.environ["CREDENTIALS_DIRECTORY"]) / "cluster_key")
    paths.append(PAIRED_KEY)
    for p in paths:
        try:
            k = p.read_text().strip()
            if k:
                return k
        except OSError:
            continue
    return ""


class Pairing:
    """A worker or host installed without the cluster key: the first master
    that sees it gives it its key (myrmid serve does that on its own), from
    then on the PC is that master's. Who may: the master's addresses the
    installer found (--pair-from), or anyone in the first hour after the
    start (restart the service to open that hour again)."""

    def __init__(self, key: str, allow: Optional[str], on_paired=None):
        import threading

        self.key = key
        self.allow = [a.strip() for a in (allow or "").split(",") if a.strip()]
        self.started = time.time()
        self.on_paired = on_paired
        self.lock = threading.Lock()

    @property
    def waiting(self) -> bool:
        return not self.key and (bool(self.allow) or time.time() - self.started < PAIR_WINDOW_S)

    def pair(self, body: dict) -> dict:
        who = str(body.get("_client") or "")
        with self.lock:
            if self.key:
                return {"ok": False, "error": "already in a cluster"}
            if self.allow and who not in self.allow:
                log(f"Pairing: refused {who} (only {', '.join(self.allow)})")
                return {"ok": False, "error": "only the master given at the installation may take this PC"}
            if not self.allow and time.time() - self.started > PAIR_WINDOW_S:
                return {"ok": False, "error": "the hour for taking this PC is over: restart its service"}
            key = str(body.get("key") or "").strip()
            if len(key) < 16:
                return {"ok": False, "error": "no key"}
            try:
                CONFIG_DIR.mkdir(parents=True, exist_ok=True)
                tmp = PAIRED_KEY.with_suffix(".tmp")
                tmp.write_text(key + "\n")
                os.chmod(tmp, 0o600)
                tmp.replace(PAIRED_KEY)
            except OSError as e:
                return {"ok": False, "error": f"could not save the key: {e}"}
            self.key = key
        log(f"Pairing: now in the cluster of {body.get('master') or who} ({who})")
        if self.on_paired:
            self.on_paired(key)
        return {"ok": True}

    def status(self) -> dict:
        return {"paired": bool(self.key), "pairing": self.waiting}


class Pairer:
    """On the master: takes new workers and hosts (installed without the key)
    into the cluster by giving them its key."""

    def __init__(self, key: str, sup: Optional["Supervisor"] = None):
        self.key, self.sup = key, sup
        self.tried: dict[str, float] = {}

    def once(self) -> None:
        import urllib.request

        for p in scan(3.0):
            if p.txt.get("pair") != "1" or p.kind not in ("worker", "host"):
                continue
            port = p.port if p.kind == "host" else int(p.txt.get("stat") or 0)
            if not port:
                continue
            ctl = f"{p.host}:{port}"
            if time.time() - self.tried.get(ctl, 0) < 60:
                continue
            self.tried[ctl] = time.time()
            body = json.dumps({"key": self.key, "master": socket.gethostname()}).encode()
            req = urllib.request.Request(f"http://{ctl}/pair", data=body, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=5) as r:
                    res = json.loads(r.read())
            except (OSError, ValueError) as e:
                log(f"Pairing: {p.name} ({ctl}) doesn't answer: {e}")
                continue
            name = p.txt.get("host") or p.name
            if res.get("ok"):
                log(f"Pairing: {name} ({p.kind}) is now in this cluster")
                if self.sup:
                    self.sup.event(f"paired {name}", f"{name} aufgenommen ({'Host' if p.kind == 'host' else 'Worker'}):"
                                   " der Master kann ihn jetzt umschalten und aktualisieren.")
            elif "already" not in str(res.get("error")):
                log(f"Pairing: {name} ({ctl}): {res.get('error')}")

    def loop(self) -> None:
        while True:
            try:
                self.once()
            except Exception as e:  # noqa: BLE001 - never stops the server
                log(f"Pairing: {type(e).__name__}: {e}")
            time.sleep(20)

    def start(self) -> None:
        import threading

        threading.Thread(target=self.loop, daemon=True).start()


def host_id(port: int) -> str:
    return machine_id() if port == HOST_CTL_PORT else f"{machine_id()}-{port}"


class HostAgent:
    """Loads the model the master assigned and keeps it running: on all of
    this PC's GPUs, or (slot) on one of them, when each GPU has its own model."""

    def __init__(self, args, key: str, slot: Optional[int] = None, gpu: Optional["Device"] = None,
                 ram_limit_gb: Optional[float] = None, stats: Optional["SystemStats"] = None):
        import threading

        self.args, self.key = args, key
        self.slot, self.gpu, self.ram_limit_gb = slot, gpu, ram_limit_gb
        self.models_dir = Path(args.models_dir).expanduser()
        self.lock = threading.RLock()
        self.pc_id = host_id(args.port)
        self.id = self.pc_id if slot is None else f"{self.pc_id}-gpu{slot}"
        self.path = "" if slot is None else f"/gpu{slot}"  # prefix on the control and API ports
        self.api_port = HOST_INTERNAL_PORT + (slot or 0)  # llama-server, on 127.0.0.1
        self.state_file = HOST_STATE if slot is None else CONFIG_DIR / f"host-gpu{slot}.json"
        self.closed = False  # replaced by another layout
        self.node: Optional["HostNode"] = None
        self.state, self.error = "idle", ""
        self.assignment: Optional[dict] = None  # model, name, role, ctx
        self.proc: Optional[subprocess.Popen] = None
        self.model_path: Optional[Path] = None
        self.ctx: Optional[int] = None
        self.estimate_tps: Optional[float] = None
        self.download: Optional[dict] = None  # files being fetched, for the progress
        self.job = 0  # a newer assignment makes older jobs stop
        self.restarts = 0
        self.paths: dict[str, str] = {}  # assigned model -> file, so a restart needs no lookup
        self.mac, self.bcast = "", ""
        self.stats = stats or SystemStats(probe(bench=False))

    @property
    def label(self) -> str:
        return "Host" if self.slot is None else f"Host GPU {self.slot}"

    # State -------------------------------------------------------------------

    def _set(self, state: str, error: str = "") -> None:
        with self.lock:
            self.state, self.error = state, error
        log(f"{self.label}: {state}{': ' + error if error else ''}")

    def _save(self) -> None:
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps({"assignment": self.assignment, "paths": self.paths}))
        except OSError as e:
            log(f"{self.label}: could not save the assignment ({e})")

    def saved(self) -> dict:
        try:
            d = json.loads(self.state_file.read_text())
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def restore(self, inherit: Optional[dict] = None) -> None:
        """Loads the last assignment again after a restart, without the master.
        inherit: what the previous layout's first host ran (switching layouts
        keeps that model running on the new first host)."""
        own = self.saved()
        saved = inherit if inherit and inherit.get("assignment") else own
        a, self.paths = saved.get("assignment"), {**(own.get("paths") or {}), **(saved.get("paths") or {})}
        if a and a.get("model"):
            log(f"{self.label}: loading the last assigned model again ({a['model']})")
            self.assign(a)

    def budget_mb(self) -> int:
        local, _ = memory_budget(False)
        return local

    def _metrics(self) -> dict:
        import urllib.request

        out: dict = {}
        if not self.proc or self.proc.poll() is not None:
            return out
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{self.api_port}/metrics",
                                         headers={"Authorization": f"Bearer {self.key}"})
            with urllib.request.urlopen(req, timeout=2) as r:
                for line in r.read().decode().splitlines():
                    if line.startswith("llamacpp:") and " " in line:
                        k, _, v = line.rpartition(" ")
                        out[k.replace("llamacpp:", "")] = _num(v)
        except OSError:
            pass
        return out

    def _progress(self) -> Optional[dict]:
        d = self.download
        if not d:
            return None
        done = 0
        for f in d["files"]:
            dest = Path(d["dir"]) / Path(f["file"]).name
            part = dest.with_suffix(dest.suffix + ".part")
            done += dest.stat().st_size if dest.exists() else part.stat().st_size if part.exists() else 0
        return {"name": d["name"], "total_mb": d["size_mb"], "done_mb": done // 2**20}

    def status(self) -> dict:
        node = self.node
        with self.lock:
            name = socket.gethostname() if self.slot is None else f"{socket.gethostname()} · GPU {self.slot}"
            st = {"id": self.id, "host": socket.gethostname(), "name": name, "v": 1, "state": self.state,
                  "error": self.error, "pc": self.pc_id, "path": self.path, "slot": self.slot,
                  "gpu": f"{self.gpu.name} {self.gpu.description}" if self.gpu else None,
                  "assignment": self.assignment, "model": self.model_path.stem if self.model_path else None,
                  "ctx": self.ctx, "estimate_tps": self.estimate_tps, "api_port": self.args.api_port,
                  "progress": self._progress() if self.state == "downloading" else None,
                  "mac": self.mac, "bcast": self.bcast,
                  "can_sleep": os.geteuid() == 0 or Path("/etc/systemd/system/myrmid-suspend.path").exists(),
                  "switchable": can_switch_role(), "keyed": True, "release": own_version(),
                  "update": update_brief()}
        st["metrics"] = self._metrics()
        st["stats"] = self.stats.sample()
        st["models"] = sorted({p.stem.split("-00001-of-")[0] for p in self.models_dir.glob("*.gguf")}) \
            if self.models_dir.is_dir() else []
        if node:  # the PC's layout: the master lists every slot and can switch it
            st.update(node.layout_info())
            st["layout_owner"] = bool(node.agents) and node.agents[0] is self
        return st

    # Commands ------------------------------------------------------------------

    def assign(self, body: dict) -> dict:
        import threading

        model = str(body.get("model") or "").strip()
        if not model:
            return {"error": "model missing"}
        a = {"model": model, "name": str(body.get("name") or "").strip(),
             "role": body.get("role") if body.get("role") in ("big", "fast", "extra") else "big",
             "ctx": int(body["ctx"]) if str(body.get("ctx") or "").isdigit() else None}
        with self.lock:
            self.job += 1
            self.assignment, self.restarts = a, 0
            job = self.job
        self._save()
        threading.Thread(target=self._run, args=(job, a), daemon=True).start()
        return self.status()

    def unload(self, _body: Optional[dict] = None) -> dict:
        with self.lock:
            self.job += 1
            self.assignment = None
        self._save()
        self.stop_server()
        with self.lock:
            self.model_path, self.ctx, self.estimate_tps, self.download = None, None, None, None
        self._set("idle")
        return self.status()

    def stop_server(self) -> None:
        p = self.proc
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        self.proc = None

    def sleep(self, _body: Optional[dict] = None) -> dict:
        """Suspends this PC (all its GPUs' models): see HostNode.sleep."""
        if self.node:
            return self.node.sleep(self)
        return self.status()

    def busy(self) -> bool:
        with self.lock:
            return self.state in ("downloading", "loading", "preparing")

    def park(self) -> None:
        """Stops llama-server before the PC sleeps; resumed() loads it again."""
        with self.lock:
            self.job += 1  # stops a restart in progress
        self.stop_server()
        self._set("asleep")

    def resumed(self) -> None:
        """Back from sleep (or suspended by someone else): load the model again."""
        with self.lock:
            a = self.assignment
            dead = self.proc is None or self.proc.poll() is not None
        log(f"{self.label}: woke up")
        if a and (self.state == "asleep" or dead):
            self.assign(a)
        elif not a:
            self._set("idle")

    def _local(self, spec: str) -> Optional[Path]:
        """An assigned model that is already on this PC: a path, or the name
        of a GGUF in the models folder or the usual places."""
        p = Path(spec).expanduser()
        if p.is_file():
            return p.resolve()
        known = self.paths.get(spec)
        if known and Path(known).is_file():
            return Path(known)
        want = spec.lower().removesuffix(".gguf")
        cands = list(self.models_dir.glob("*.gguf")) if self.models_dir.is_dir() else []
        cands += [Path(m["path"]) for m in local_models()]
        for c in cands:
            if c.stem.lower().split("-00001-of-")[0] == want:
                return c.resolve()
        return None

    def _run(self, job: int, a: dict) -> None:
        def current() -> bool:
            return job == self.job

        try:
            self._set("preparing")
            self.stop_server()
            path = self._local(a["model"])
            if path is None:
                self._set("downloading")

                def resolved(info: dict) -> None:
                    with self.lock:
                        self.download = info

                path = fetch_model(a["model"], self.models_dir, quiet=True, on_resolved=resolved,
                                   max_mb=self.budget_mb())
                with self.lock:
                    self.paths[a["model"]] = str(path)
                self._save()
            if not current():
                return
            self._set("loading")
            ns = argparse.Namespace(model=str(path), ctx=a.get("ctx"), mtp=0, workers="off",
                                    ram_limit=self.ram_limit_gb, api_key=self.key, host="127.0.0.1",
                                    port=self.api_port, extra=list(self.args.extra or []), scan_timeout=1.0,
                                    only_devices=[self.gpu.name] if self.gpu else None)
            sp = build_server_fitting(ns, quiet=True, exit_on_fail=False)
            if self.gpu:  # without it llama.cpp spreads the model over every GPU it sees
                sp.cmd += ["--device", self.gpu.name]
            if not current():
                return
            log(" ".join(sp.cmd).replace(self.key, "***"))
            proc = subprocess.Popen(sp.cmd)
            with self.lock:
                self.proc, self.model_path, self.ctx, self.estimate_tps = proc, path, ns.ctx, round(sp.estimate_tps, 1)
                self.download = None
            if _wait_healthy(f"http://127.0.0.1:{self.api_port}", proc, timeout=3600) and current():
                self._set("ready")
            elif current():
                self._set("error", f"llama-server did not start (code {proc.poll()}); see the log")
        except (ModelError, PlanTooBig) as e:
            if current():
                self._set("error", str(e))
        except Exception as e:  # noqa: BLE001 - report it to the master instead of dying
            if current():
                self._set("error", f"{type(e).__name__}: {e}")

    def watch(self) -> None:
        """Restarts llama-server if it ends on its own (up to 3 times in a row)."""
        while not self.closed:
            time.sleep(5)
            with self.lock:
                dead = self.state == "ready" and self.proc is not None and self.proc.poll() is not None
                a = self.assignment
            if dead and a:
                code = self.proc.poll()
                self.restarts += 1
                if self.restarts > 3:
                    self._set("error", f"llama-server keeps ending (code {code}); not restarting again")
                    continue
                self._set("error", f"llama-server ended (code {code}); restarting in 30 s")
                time.sleep(30)
                with self.lock:
                    same = self.assignment is a and not self.closed
                    if same:
                        self.job += 1
                        job = self.job
                if same:
                    self._run(job, a)

    def close(self) -> None:
        """For a layout change: stop for good (the saved assignment stays)."""
        with self.lock:
            self.closed = True
            self.job += 1
        self.stop_server()


class HostNode:
    """The host on one PC: one HostAgent over all GPUs (layout "split"), or
    one per dedicated GPU ("per_gpu"), switched from the master's dashboard.
    The control port answers for each (/gpuN/status, /gpuN/assign, ...; the
    first also without prefix), the API port forwards /gpuN/... to that
    GPU's llama-server on 127.0.0.1. So the firewall needs no new ports."""

    def __init__(self, args, key: str):
        import threading

        self.args, self.key = args, key
        self.lock = threading.RLock()
        self.devices = probe(bench=False)
        self.gpus = [d for d in usable_gpus(self.devices) if d.type == "gpu"]
        self.stats = SystemStats(self.devices)
        self.get_routes: dict = {}
        self.post_routes: dict = {}
        self.agents: list[HostAgent] = []
        self.mac, self.bcast = "", ""
        self.switching = False
        try:
            self.layout = json.loads(HOST_LAYOUT.read_text()).get("layout", "split")
        except (OSError, ValueError, AttributeError):
            self.layout = "split"
        if self.layout not in HOST_LAYOUTS or (self.layout == "per_gpu" and len(self.gpus) < 2):
            self.layout = "split"

    def _make(self) -> list[HostAgent]:
        if self.layout == "per_gpu":
            ram_gb = memory_budget(False)[0] / 1024 / len(self.gpus)  # models that spill share the RAM
            agents = [HostAgent(self.args, self.key, slot=i, gpu=g, ram_limit_gb=ram_gb, stats=self.stats)
                      for i, g in enumerate(self.gpus)]
        else:
            agents = [HostAgent(self.args, self.key, stats=self.stats)]
        for a in agents:
            a.node, a.mac, a.bcast = self, self.mac, self.bcast
        return agents

    def _set_routes(self) -> None:
        gets, posts = {}, {}
        for i, a in enumerate(self.agents):
            for prefix in ([""] if i == 0 else []) + ([a.path] if a.path else []):
                gets[f"{prefix}/status"] = a.status
                posts.update({f"{prefix}/assign": a.assign, f"{prefix}/unload": a.unload,
                              f"{prefix}/sleep": a.sleep})
        posts.update({"/layout": self.set_layout, "/role": lambda body: request_role(str(body.get("role") or "")),
                      "/update": request_update})
        # the handler holds these dicts: change them in place
        self.get_routes.clear()
        self.get_routes.update(gets)
        self.post_routes.clear()
        self.post_routes.update(posts)

    def start(self) -> None:
        import threading

        with self.lock:
            self.agents = self._make()
            self._set_routes()
        for a in self.agents:
            threading.Thread(target=a.watch, daemon=True).start()
            a.restore()
        threading.Thread(target=self.watch_resume, daemon=True).start()

    def layout_info(self) -> dict:
        return {"layout": self.layout, "switching": self.switching,
                "slots": [{"id": a.id, "path": a.path} for a in self.agents],
                "gpus": [{"name": g.name, "description": g.description, "total_mb": g.total_mb} for g in self.gpus]}

    def set_layout(self, body: dict) -> dict:
        """Switches between one model over all GPUs and one per GPU. The
        models stop and load again; the first GPU (or all of them) keeps the
        model the other layout had there, if it has nothing of its own yet."""
        import threading

        layout = str(body.get("layout") or "")
        if layout not in HOST_LAYOUTS:
            return {"ok": False, "error": f"layout must be one of {', '.join(HOST_LAYOUTS)}"}
        if layout == "per_gpu" and len(self.gpus) < 2:
            return {"ok": False, "error": "this PC has fewer than two GPUs"}
        if layout == self.layout:
            return {"ok": True, **self.layout_info()}
        if self.switching:
            return {"ok": False, "error": "a switch is running"}
        self.switching = True

        def run() -> None:
            try:
                old = self.agents
                first = old[0].saved() if old else {}
                for a in old:
                    a.close()
                with self.lock:
                    self.layout = layout
                    try:
                        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
                        HOST_LAYOUT.write_text(json.dumps({"layout": layout}))
                    except OSError as e:
                        log(f"Host: could not save the layout ({e})")
                    self.agents = self._make()
                    self._set_routes()
                log(f"Host: layout {layout} ({len(self.agents)} model(s))")
                for i, a in enumerate(self.agents):
                    threading.Thread(target=a.watch, daemon=True).start()
                    a.restore(inherit=first if i == 0 else None)
            finally:
                self.switching = False

        threading.Thread(target=run, daemon=True).start()
        return {"ok": True, **self.layout_info()}

    def sleep(self, asking: HostAgent) -> dict:
        """Stops every model and suspends this PC (the master wakes it by
        Wake-on-LAN; the models load again after waking up)."""
        busy = [a for a in self.agents if a.busy()]
        if busy:
            return {**asking.status(), "error": f"busy ({busy[0].state})"}
        for a in self.agents:
            a.park()
        ok, how = request_suspend()
        if not ok:
            log(f"Host: could not suspend ({how}); loading the models again")
            for a in self.agents:
                if a.assignment:
                    a.assign(a.assignment)
            return {**asking.status(), "error": f"suspend failed: {how}"}
        log(f"Host: suspending ({how})")
        return asking.status()

    def watch_resume(self) -> None:
        """Notices waking up (the wall clock jumps, the monotonic one doesn't)."""
        offset = time.time() - time.monotonic()
        while True:
            time.sleep(5)
            now_offset = time.time() - time.monotonic()
            if now_offset - offset > 20:
                for a in list(self.agents):
                    a.resumed()
            offset = now_offset

    def backend_port(self, path: str) -> tuple[Optional[int], str]:
        """The llama-server for a request on the API port, and the path for it."""
        agents = self.agents
        for a in agents:
            if a.path and (path == a.path or path.startswith(a.path + "/")):
                return a.api_port, path[len(a.path):] or "/"
        if path.startswith("/gpu"):
            return None, path  # a GPU of the other layout
        return (agents[0].api_port if agents else None), path

    def stop(self) -> None:
        for a in self.agents:
            a.close()


def _host_wait_for_master(args) -> None:
    """A host without the cluster key can't run models: it waits until a
    master takes it into its cluster (gives it its key), then goes on."""
    import threading
    from http.server import ThreadingHTTPServer

    got = threading.Event()
    pairing = Pairing("", args.pair_from, on_paired=lambda _k: got.set())

    def status() -> dict:
        return {"id": host_id(args.port), "host": socket.gethostname(), "state": "pairing", "v": 1,
                "release": own_version(), **pairing.status()}

    try:
        srv = ThreadingHTTPServer((args.host, args.port),
                                  _json_handler({"/status": status}, {"/pair": pairing.pair}))
    except OSError:
        die(f"Port {args.port} is busy.")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    address = args.advertise_ip or (args.host if args.host not in ("0.0.0.0", "") else lan_address())
    adv = None
    if not args.no_advertise:
        adv = Advertiser(HOST_TYPE, f"MyrmidLM Host {socket.gethostname()}", args.port,
                         {"id": host_id(args.port), "host": socket.gethostname(), "api": str(args.api_port),
                          "v": "1", "pair": "1"}, address)
        adv.start()
    log("No cluster key yet: waiting for a master to take this host into its cluster"
        + (f" (only {args.pair_from})" if args.pair_from else " (in the next hour; then restart the service)"))
    while not got.wait(5):
        if not pairing.waiting:
            die("No master took this host in the first hour. Restart the service, or give it the key: "
                "--key-file (on the master: myrmid key)")
    if adv:
        adv.stop()
    srv.shutdown()
    srv.server_close()


def _host_proxy_handler(node: HostNode):
    """The host's API port: passes each request to the llama-server it is for
    (streaming, like the router), so per-GPU models need no extra ports."""
    import http.client
    from http.server import BaseHTTPRequestHandler

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _pass(self, method: str) -> None:
            port, path = node.backend_port(self.path)
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else None
            if port is None:
                body = b'{"error":{"message":"no model on this path"}}'
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() in ("content-type", "accept", "authorization")}
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
            try:
                try:
                    conn.request(method, path, body=raw, headers=headers)
                    resp = conn.getresponse()
                except OSError as e:
                    body = json.dumps({"error": {"message": f"model not running: {e}"}}).encode()
                    self.send_response(502)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                try:
                    self.send_response(resp.status)
                    for k, v in resp.getheaders():
                        if k.lower() not in ("transfer-encoding", "content-length", "connection"):
                            self.send_header(k, v)
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    while True:
                        chunk = resp.read1(65536)
                        if not chunk:
                            break
                        self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True  # client went away; closing the backend stops generating
            finally:
                conn.close()

        def do_GET(self):
            self._pass("GET")

        def do_POST(self):
            self._pass("POST")

    return H


def cmd_host(args) -> None:
    import threading

    key = read_cluster_key(args.key, args.key_file)
    if not key:
        _host_wait_for_master(args)  # returns with the key a master gave this PC
        key = read_cluster_key(args.key, args.key_file)
    node = HostNode(args, key)
    if not _start_http(args.host, args.port, _json_handler(node.get_routes, node.post_routes, key=key)):
        die(f"Port {args.port} is busy.")
    if not _start_http(args.host, args.api_port, _host_proxy_handler(node)):
        die(f"Port {args.api_port} is busy.")
    devices = probe(bench=True)
    gpus = usable_gpus(devices)
    address = args.advertise_ip or (args.host if args.host not in ("0.0.0.0", "") else lan_address())
    vram = sum(d.total_mb for d in gpus if d.type == "gpu")
    node.mac, node.bcast = iface_info(address)
    gpu_txt = ", ".join(sorted({d.description for d in gpus})) or "CPU"
    log(f"Host {socket.gethostname()}: control API on {address}:{args.port}, models in "
        f"{Path(args.models_dir).expanduser()}, API on port {args.api_port}; {gpu_txt}; "
        + ("one model per GPU" if node.layout == "per_gpu" else "one model over all GPUs"))
    advs = []
    if not args.no_advertise:
        adv = Advertiser(HOST_TYPE, f"MyrmidLM Host {socket.gethostname()}", args.port,
                         {"id": host_id(args.port), "host": socket.gethostname(), "api": str(args.api_port),
                          "gpu": gpu_txt[:120], "vram": str(vram), "v": "1",
                          **({"mac": node.mac, "bcast": node.bcast} if node.mac else {})}, address)
        adv.start()
        advs.append(adv)
    NetWatch(advs, fixed=bool(args.advertise_ip) or args.host not in ("0.0.0.0", "")).start()
    node.start()

    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while not stopping:
        time.sleep(0.5)
    node.stop()
    for a in advs:
        a.stop()


def cmd_key(args) -> None:
    """Prints the master's API key, which hosts use as the cluster key: the
    service's (installed with the script) if there is one, else this user's."""
    for service in (Path("/var/lib/myrmid/myrmid/api_key"), Path("/var/lib/private/myrmid/myrmid/api_key"),
                    # before the rename, or moved by install.sh but the service hasn't started since:
                    Path("/var/lib/private/myrmid/privatelm-node/api_key"),
                    Path("/var/lib/privatelm-node/privatelm-node/api_key")):
        try:
            if service.is_file():
                print(service.read_text().strip())
                return
        except OSError:
            die(f"{service} is the service's key; read it with sudo: sudo {sys.argv[0]} key")
    print(_api_key())


# ── Router ───────────────────────────────────────────────────────────────────
#
# With a fast model (--fast-model or --fast-url), serve puts a router on the
# public API port: one endpoint, one model name, and each request goes to
# the fast model (short, simple) or the big one (everything else). The big
# model may be split across workers; the fast one runs on a single device,
# so simple answers come back with no network round trips.
#
# Overrides per request: model "fast" or "big", or header X-Myrmid-Route.
# On/off and the decision method can be changed live from the dashboard.

ROUTER_STATE = CONFIG_DIR / "router.json"

HARD_WORDS = re.compile(
    r"(```|\b(code|programm\w*|skript|script|funktion|function|klasse|class|debug\w*|bug|fehler\w*|"
    r"refactor\w*|implementier\w*|analys\w*|analyz\w*|vergleich\w*|compare|beweis\w*|prove|herleit\w*|"
    r"berechn\w*|calculate|schritt für schritt|step by step|ausführlich|in detail|entwirf|entwurf|design|"
    r"architektur|architecture|strategie|strategy|konzept|zusammenfass\w*|summari[sz]\w*|übersetze den|"
    r"translate the following|review|optimier\w*|sql|regex|kubernetes|ansible|docker|yaml|json)\b)",
    re.IGNORECASE)
ROUTER_PROMPT = (
    "Du entscheidest, welches Sprachmodell eine Anfrage beantwortet. Antworte nur mit einem "
    "Buchstaben: E, wenn ein kleines Modell sie gut beantworten kann (Smalltalk, kurze Fakten, "
    "kurze Umformulierungen, einfache Fragen), S, wenn sie ein großes Modell braucht "
    "(Programmieren, Analyse, Mathematik, lange Texte, mehrstufiges Denken, Fachwissen).")


def _message_text(content) -> tuple[str, bool]:
    """Text of a message content, and whether it carries an image."""
    if isinstance(content, str):
        return content, False
    text, image = [], False
    for part in content or []:
        if isinstance(part, dict):
            if part.get("type") == "text":
                text.append(part.get("text", ""))
            elif part.get("type") in ("image_url", "input_image", "image"):
                image = True
    return "\n".join(text), image


@dataclass
class Backend:
    name: str  # role: "big", "fast", or "extra" (only by its model name)
    base: str  # http://host:port
    key: str
    label: str  # model name for display
    healthy: bool = False
    status: str = "starting"
    group: str = ""  # name clients can ask for (model field or header)
    where: str = "Master"  # machine
    source: str = "local"  # "local" or the host's id
    ctx: int = 0
    inflight: int = 0  # requests running now
    served: int = 0
    asleep: bool = False  # a host that sleeps (or is off) and can be woken
    est_tps: float = 0.0  # the planner's estimate (tokens/s)
    ctx_at: float = 0.0  # when ctx was last read from the server
    tps: float = 0.0  # measured on requests it ran alone (moving average)

    @property
    def speed(self) -> float:
        return self.tps or self.est_tps


ROLE_WORDS = {"fast": ("fast", "schnell", "small"), "big": ("big", "large", "groß", "gross")}

# Routing by examples: a small embedding model compares each request with
# example sentences; the closest category decides (fast, big, or a model by
# its client name, e.g. a coder model for code). Edited in the dashboard.
ROUTER_EXAMPLES = CONFIG_DIR / "router-examples.json"
EMBED_MODEL = "ggml-org/embeddinggemma-300M-GGUF:Q8_0"  # 300 MB, many languages, runs on the CPU
DEFAULT_EXAMPLES = {
    "threshold": 0.05,
    "categories": [
        {"name": "leicht", "target": "fast", "examples": [
            "Was ist die Hauptstadt von Frankreich?", "Übersetze das ins Englische.",
            "Fasse diesen Absatz in einem Satz zusammen.", "Korrigiere die Rechtschreibung.",
            "Wie spät ist es in New York?", "Schreib eine kurze Antwort auf diese Mail.",
            "What does this word mean?", "Give me a synonym for fast."]},
        {"name": "schwer", "target": "big", "examples": [
            "Refaktoriere dieses Modul und erkläre die Änderungen.", "Finde den Fehler in diesem Code.",
            "Entwirf eine Architektur für ein verteiltes System.", "Beweise, dass die Aussage stimmt.",
            "Vergleiche die Vor- und Nachteile und gib eine begründete Empfehlung.",
            "Schreib ein Skript, das die Logdateien auswertet und Berichte erstellt.",
            "Analyze this contract and list the risks.", "Plan the migration step by step."]},
    ],
}


class ContextTooLong(Exception):
    """No model the request may go to has room for it."""

    def __init__(self, need: int, biggest: int, what: str):
        super().__init__(f"{what}: about {need} tokens, the largest context is {biggest}")
        self.need, self.biggest, self.what = need, biggest, what


class Embedder:
    """The router's embedding model: its own small llama-server on the CPU of
    the master, started (and downloaded) the first time routing by examples
    is switched on."""

    def __init__(self, port: int):
        import threading

        self.port = port
        self.state, self.error = "off", ""
        self.proc: Optional[subprocess.Popen] = None
        self.lock = threading.Lock()
        self.cache: dict[str, list[float]] = {}

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def ensure(self) -> None:
        import threading

        with self.lock:
            if self.state in ("downloading", "loading", "ready"):
                return
            self.state, self.error = "downloading", ""
        threading.Thread(target=self._start, daemon=True).start()

    def _start(self) -> None:
        try:
            dest = Path("/srv/models") if os.access("/srv/models", os.W_OK) else CONFIG_DIR / "models"
            path = Path(env("EMBED_MODEL") or "")
            if not path.is_file():
                path = fetch_model(env("EMBED_MODEL") or EMBED_MODEL, dest, quiet=True)
            self.state = "loading"
            cmd = [str(find_bin("llama-server")), "-m", str(path), "--embeddings", "--pooling", "mean", "--host", "127.0.0.1",
                   "--port", str(self.port), "-ngl", "0", "-c", "2048", "-b", "2048", "-ub", "2048",
                   "--threads", str(max(1, min(4, (os.cpu_count() or 2) // 2)))]
            log(f"Router: embedding model {path.name} on the CPU (port {self.port})")
            self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if not _wait_healthy(self.base, self.proc, timeout=300):
                raise RuntimeError("the embedding model did not start")
            self.state = "ready"
        except Exception as e:  # noqa: BLE001 - routing falls back to the rules
            self.state, self.error = "error", str(e)[:300]
            log(f"Router: embedding model failed: {e}")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Normalized vectors; examples stay cached."""
        import math
        import urllib.request

        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if todo:
            req = urllib.request.Request(self.base + "/v1/embeddings",
                                         data=json.dumps({"input": todo}).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read())["data"]
            for t, d in zip(todo, sorted(data, key=lambda x: x.get("index", 0))):
                v = d["embedding"]
                n = math.sqrt(sum(x * x for x in v)) or 1.0
                self.cache[t] = [x / n for x in v]
            if len(self.cache) > 2000:  # requests come and go, examples stay
                self.cache = {k: v for k, v in list(self.cache.items())[-1000:]}
        return [self.cache[t] for t in texts]


class Router:
    """One endpoint in front of every model: the master's own (big, and the
    optional fast one) and those the hosts load. Requests go by model name
    when they ask for one, else by the rules (fast or big); within a group
    the backend with the fewest running requests gets the next one."""

    def __init__(self, backends: list[Backend], fast_ctx: int, enabled: bool, mode: str,
                 public_name: str, key: str):
        import threading

        self.backends = list(backends)
        self.fast_ctx = fast_ctx
        self.enabled, self.mode = enabled, mode
        self.public_name, self.key = public_name, key
        self.lock = threading.Lock()
        self.counts = {"fast": 0, "big": 0, "extra": 0}
        self.recent: list[dict] = []
        self.on_timings = None  # callback(timings) for requests the master's big model ran alone
        self.waker = None  # callback(backend) -> backend once awake, or None: wake a sleeping host
        self.last_request = 0.0  # when the last request came in (updates wait for a quiet moment)
        self.prefer_fast = False  # within a group: the fastest first, slower ones only when it is busy
        self.examples: dict = json.loads(json.dumps(DEFAULT_EXAMPLES))
        try:
            saved = json.loads(ROUTER_EXAMPLES.read_text())
            if isinstance(saved, dict) and isinstance(saved.get("categories"), list):
                self.examples = saved
        except (OSError, ValueError):
            pass
        self.embedder: Optional[Embedder] = None  # set by serve; started when mode is "examples"

    @property
    def big(self) -> Backend:
        return next((b for b in self.backends if b.name == "big" and b.source == "local"), self.backends[0])

    @property
    def fast(self) -> Optional[Backend]:
        return next((b for b in self.backends if b.name == "fast" and b.source == "local"), None)

    def set_hosts(self, hosts: list[Backend]) -> None:
        """Replaces the hosts' backends, keeping health and counters of the
        ones that stay."""
        with self.lock:
            old = {(b.source, b.base): b for b in self.backends if b.source != "local"}
            keep = [b for b in self.backends if b.source == "local"]
            for h in hosts:
                prev = old.get((h.source, h.base))
                if prev:
                    prev.name, prev.label, prev.group, prev.where, prev.ctx = h.name, h.label, h.group, h.where, h.ctx
                    if prev.label != h.label or prev.est_tps != h.est_tps:
                        prev.est_tps = h.est_tps
                    if prev.asleep != h.asleep:
                        prev.asleep, prev.healthy = h.asleep, False
                        prev.status = "asleep" if h.asleep else "starting"
                    keep.append(prev)
                else:
                    keep.append(h)
            self.backends = keep

    # State -----------------------------------------------------------------

    def save(self) -> None:
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            ROUTER_STATE.write_text(json.dumps({"enabled": self.enabled, "mode": self.mode,
                                                "prefer_fast": self.prefer_fast}))
        except OSError as e:
            log(f"Router: could not save the setting ({e})")

    def set(self, body: dict) -> dict:
        with self.lock:
            if "enabled" in body:
                self.enabled = bool(body["enabled"])
            if body.get("mode") in ("heuristic", "model", "examples"):
                self.mode = body["mode"]
            if "prefer_fast" in body:
                self.prefer_fast = bool(body["prefer_fast"])
        if isinstance(body.get("examples"), dict):
            err = self._set_examples(body["examples"])
            if err:
                return {**self.info(), "error": err}
        if body.get("test") is not None:
            return {**self.info(), "test": self.test_examples(str(body["test"]))}
        if self.mode == "examples" and self.embedder:
            self.embedder.ensure()
        self.save()
        log(f"Router {'on' if self.enabled else 'off'} ({self.mode}); "
            f"{'fastest first' if self.prefer_fast else 'fewest running requests'} within a group")
        return self.info()

    def _set_examples(self, ex: dict) -> str:
        cats = []
        for c in ex.get("categories") or []:
            name = str(c.get("name") or "").strip()[:40]
            target = str(c.get("target") or "").strip()[:80]
            lines = [str(x).strip()[:500] for x in (c.get("examples") or []) if str(x).strip()][:50]
            if not name or not target or not lines:
                continue
            cats.append({"name": name, "target": target, "examples": lines})
        if not any(c["target"] == "big" for c in cats):
            return "Mindestens eine Kategorie muss ans große Modell gehen (Ziel „groß“)."
        try:
            th = min(0.5, max(0.0, float(ex.get("threshold", 0.05))))
        except (TypeError, ValueError):
            th = 0.05
        with self.lock:
            self.examples = {"threshold": th, "categories": cats[:12]}
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            ROUTER_EXAMPLES.write_text(json.dumps(self.examples, indent=1, ensure_ascii=False))
        except OSError as e:
            return f"could not save: {e}"
        return ""

    def score_examples(self, text: str) -> Optional[list[dict]]:
        """Each category's best similarity to the text, best first (None: no embedding model)."""
        e = self.embedder
        if not e or e.state != "ready" or not text.strip():
            return None
        cats = self.examples.get("categories") or []
        lines = [x for c in cats for x in c["examples"]]
        try:
            vecs = e.embed(lines + [text[:2000]])
            e.error = ""
        except (OSError, ValueError, KeyError) as x:
            e.error = f"Embedding fehlgeschlagen: {x}"[:300]
            return None
        q = vecs[-1]
        out, i = [], 0
        for c in cats:
            n = len(c["examples"])
            best = max((sum(a * b for a, b in zip(q, v)) for v in vecs[i:i + n]), default=0.0)
            out.append({"name": c["name"], "target": c["target"], "score": round(best, 3)})
            i += n
        return sorted(out, key=lambda x: -x["score"])

    def decide_examples(self, text: str) -> Optional[tuple[str, str]]:
        """(target, reason): fast, big or a client name. Fast only with a clear
        lead over the rest; unsure means big."""
        scores = self.score_examples(text)
        if not scores:
            return None
        best = scores[0]
        rest = max((x["score"] for x in scores if x["target"] != best["target"]), default=0.0)
        why = f"Beispiele: {best['name']} ({best['score']:.2f})"
        if best["target"] == "fast" and best["score"] - rest < float(self.examples.get("threshold", 0.05)):
            return "big", f"Beispiele: unsicher ({best['name']} {best['score']:.2f} / {rest:.2f})"
        return best["target"], why

    def test_examples(self, text: str) -> dict:
        scores = self.score_examples(text)
        if scores is None:
            e = self.embedder
            return {"error": (e.error if e and e.state == "ready" and e.error else "")
                    or "Das Embedding-Modell ist noch nicht bereit."}
        target, why = self.decide_examples(text) or ("big", "")
        return {"scores": scores, "target": target, "reason": why}

    def need(self, body: dict) -> tuple[int, int]:
        """Rough tokens a request needs (prompt, tools, room to answer): a
        careful estimate for choosing, and an optimistic one for refusing."""
        msgs = body.get("messages")
        if msgs is None:
            msgs = [{"content": body.get("prompt", "")}]
        chars = sum(len(_message_text(m.get("content"))[0]) + 16 for m in msgs)
        chars += len(json.dumps(body.get("tools") or body.get("functions") or []))
        try:
            answer = min(2048, int(body.get("max_tokens") or body.get("max_completion_tokens") or 256))
        except (TypeError, ValueError):
            answer = 256
        return int(chars / 3.2) + 64 + answer, int(chars / 4.2) + 16

    def info(self) -> dict:
        def brief(b: Optional[Backend]) -> Optional[dict]:
            return {"model": b.label, "status": b.status, "base": b.base} if b else None

        with self.lock:
            fast = self.fast or next((b for b in self.backends if b.name == "fast"), None)
            emb = self.embedder
            return {"available": True, "enabled": self.enabled, "mode": self.mode, "prefer_fast": self.prefer_fast,
                    "examples": self.examples,
                    "embedder": {"state": emb.state, "error": emb.error, "model": EMBED_MODEL} if emb else None,
                    "fast": brief(fast), "big": brief(self.big),
                    "has_fast": any(b.name == "fast" for b in self.backends),
                    "counts": dict(self.counts), "recent": list(self.recent[-12:]),
                    "backends": [{"role": b.name, "group": b.group, "model": b.label, "where": b.where,
                                  "source": b.source, "status": "asleep" if b.asleep else b.status,
                                  "inflight": b.inflight, "served": b.served, "ctx": b.ctx,
                                  "tps": round(b.speed, 1), "measured": b.tps > 0} for b in self.backends]}

    def watch_health(self) -> None:
        import urllib.request

        while True:
            for b in list(self.backends):
                if b.asleep:
                    b.healthy, b.status = False, "asleep"
                    continue
                try:
                    with urllib.request.urlopen(b.base + "/health", timeout=2) as r:
                        b.healthy, b.status = r.status == 200, "ready"
                except OSError as e:
                    b.healthy = False
                    b.status = "loading" if "503" in str(e) else "down"
                if b.healthy and time.time() - b.ctx_at > 60:
                    # what the server really has (a planner may have shrunk it to fit)
                    b.ctx_at = time.time()
                    try:
                        req = urllib.request.Request(b.base + "/props", headers={"Authorization": f"Bearer {b.key}"}
                                                     if b.key else {})
                        with urllib.request.urlopen(req, timeout=3) as r:
                            n = int((json.loads(r.read()).get("default_generation_settings") or {}).get("n_ctx") or 0)
                        if n > 0:
                            b.ctx = n
                    except (OSError, ValueError, TypeError):
                        pass
            time.sleep(3)

    # Decision ----------------------------------------------------------------

    def choose(self, cands: list[Backend]) -> Optional[Backend]:
        """The healthy backend with the fewest running requests (then the
        fewest served, so equal ones take turns). With "fastest first": the
        one that should finish soonest, (running + 1) / speed: a single
        request goes to the fastest, a slower one helps once the fastest is
        busy enough (llama-server runs several requests at once, each slower)."""
        ok = [b for b in cands if b.healthy]
        if not ok:
            return None
        known = [b.speed for b in ok if b.speed > 0]
        if self.prefer_fast and known and len(ok) > 1:
            slowest = min(known)  # no measurement or estimate yet: count it as the slowest
            return min(ok, key=lambda b: ((b.inflight + 1) / (b.speed or slowest), b.served))
        return min(ok, key=lambda b: (b.inflight, b.served))

    def role(self, role: str) -> list[Backend]:
        return [b for b in self.backends if b.name == role]

    def sleeper(self, cands: list[Backend]) -> Optional[Backend]:
        """A sleeping member to wake when none is ready."""
        return next((b for b in cands if b.asleep), None)

    @staticmethod
    def fits(cands: list[Backend], need: int) -> list[Backend]:
        """The ones with room for the request (unknown context: assumed to fit)."""
        return [b for b in cands if not need or not b.ctx or need <= b.ctx]

    def _too_long(self, cands: list[Backend], loose: int, what: str) -> None:
        """Refuses when even the optimistic estimate fits none (and says so like OpenAI does)."""
        known = [b.ctx for b in cands if b.ctx]
        if known and len(known) == len(cands) and loose > max(known):
            raise ContextTooLong(loose, max(known), what)

    def _pick_role(self, role: str, reason: str, need: int = 0, loose: int = 0) -> tuple[Backend, str]:
        cands = self.fits(self.role(role), need)
        if self.role(role) and not cands:
            reason += f", {'schnelles' if role == 'fast' else 'großes'} Modell hat zu wenig Kontext"
        b = self.choose(cands)
        if b:
            return b, reason
        b = self.sleeper(cands)
        if b and self.waker:
            return b, f"{reason}, Host wird geweckt"
        other = "big" if role == "fast" else "fast"
        b = self.choose(self.fits(self.role(other), need))
        if b:
            return b, reason if not cands and self.role(role) else \
                f"{reason}, aber {'schnelles' if role == 'fast' else 'großes'} Modell nicht bereit"
        b = self.choose(self.fits(self.backends, need))
        if b:
            return b, reason + ", nichts anderes bereit"
        self._too_long([x for x in self.backends if x.healthy or x.asleep], loose, "every model")
        # nothing fits the careful estimate: the one with the most room tries
        roomy = sorted((x for x in self.backends if x.healthy), key=lambda x: -(x.ctx or 10**9))
        return (roomy[0] if roomy else self.big), reason + ", größter Kontext"

    def _pick_group(self, group: list[Backend], reason: str, need: int, loose: int) -> tuple[Backend, str]:
        fitting = self.fits(group, need)
        b = self.choose(fitting)
        if b:
            return b, reason
        b = self.sleeper(fitting)
        if b and self.waker:
            return b, f"{reason}, Host wird geweckt"
        if not fitting:
            self._too_long(group, loose, group[0].group or group[0].label)
            roomy = sorted((x for x in group if x.healthy), key=lambda x: -(x.ctx or 10**9))
            if roomy:
                return roomy[0], reason + ", größter Kontext"
        return group[0], f"{reason}, aber nicht bereit"

    def group(self, name: str) -> list[Backend]:
        name = name.lower()
        return [b for b in self.backends if name in (b.group.lower(), b.label.lower())]

    def decide(self, body: dict, headers) -> tuple[Backend, str]:
        """Which model answers. Raises ContextTooLong when none has room."""
        need, loose = self.need(body)
        forced = ((headers.get("X-Myrmid-Route") or headers.get("X-PrivateLM-Route") or "").strip()
                  or str(body.get("model", "")).strip()).lower()
        # "privatelm" was the router's name before the rename: clients set up then keep working
        if forced and forced not in (self.public_name.lower(), "privatelm"):
            group = self.group(forced)
            if group:
                return self._pick_group(group, "Modell angefordert", need, loose)
            for role, words in ROLE_WORDS.items():
                if forced in words:
                    return self._pick_role(role, "angefordert", need, loose)
        if not self.enabled:
            return self._pick_role("big", "Router aus", need, loose)
        by_example = None
        if self.mode == "examples":
            msgs = body.get("messages") or [{"role": "user", "content": body.get("prompt", "")}]
            users = [_message_text(m.get("content"))[0] for m in msgs if m.get("role", "user") == "user"]
            by_example = self.decide_examples(users[-1] if users else "")
            if by_example and by_example[0] not in ("fast", "big"):
                group = self.group(by_example[0])  # a category for a model by name, e.g. code
                if group:
                    return self._pick_group(group, by_example[1], need, loose)
                by_example = ("big", by_example[1] + f", „{by_example[0]}“ nicht da")
        if not self.choose(self.fits(self.role("fast"), need)):
            return self._pick_role("big", "kein schnelles Modell bereit" if not self.choose(self.role("fast"))
                                   else "Kontext zu groß fürs schnelle Modell", need, loose)
        if not self.choose(self.role("big")):
            return self._pick_role("fast", "großes Modell nicht bereit", need, loose)
        route, reason = self.classify(body)
        if route == "fast" and self.mode == "model":
            route, reason = self.ask_fast_model(body, reason)
        elif route == "fast" and self.mode == "examples":
            route, reason = by_example or ("fast", reason + " (Beispiele nicht bereit)")
        return self._pick_role(route, reason, need, loose)

    def classify(self, body: dict) -> tuple[str, str]:
        """Rules first (what the small model can't do), then the shape of the
        request. Errs towards the big model: a wrong 'big' costs time, a
        wrong 'fast' costs quality."""
        msgs = body.get("messages")
        if msgs is None:
            msgs = [{"role": "user", "content": body.get("prompt", "")}]
        if body.get("tools") or body.get("functions"):
            return "big", "Werkzeuge (Tools)"
        texts, image = [], False
        for m in msgs:
            t, img = _message_text(m.get("content"))
            texts.append(t)
            image = image or img
        if image:
            return "big", "Bild in der Anfrage"
        total = sum(len(t) for t in texts)
        fast_ctx = min((b.ctx for b in self.role("fast") if b.ctx), default=self.fast_ctx)
        if total / 3.5 > fast_ctx * 0.6:
            return "big", "langer Kontext"
        max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or body.get("n_predict") or 0
        if isinstance(max_tokens, int) and max_tokens > 1024:
            return "big", "lange Antwort verlangt"
        if (body.get("response_format") or {}).get("type") == "json_schema":
            return "big", "strukturierte Ausgabe"
        users = [t for m, t in zip(msgs, texts) if m.get("role") == "user"]
        last = users[-1] if users else (texts[-1] if texts else "")
        if len(users) > 8:
            return "big", "langes Gespräch"
        if len(last) > 600:
            return "big", "lange Anfrage"
        if HARD_WORDS.search(last):
            return "big", "anspruchsvolle Aufgabe"
        return "fast", "kurz und einfach"

    def ask_fast_model(self, body: dict, reason: str) -> tuple[str, str]:
        """Lets the fast model judge the request (one token, a few ms)."""
        import urllib.request

        fast = self.choose(self.role("fast"))
        if fast is None:
            return "fast", reason
        msgs = body.get("messages") or [{"role": "user", "content": body.get("prompt", "")}]
        users = [_message_text(m.get("content"))[0] for m in msgs if m.get("role", "user") == "user"]
        q = (users[-1] if users else "")[:1500]
        req = {"messages": [{"role": "system", "content": ROUTER_PROMPT}, {"role": "user", "content": q}],
               "max_tokens": 1, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
        try:
            r = urllib.request.Request(fast.base + "/v1/chat/completions", data=json.dumps(req).encode(),
                                       headers={"Content-Type": "application/json",
                                                **({"Authorization": f"Bearer {fast.key}"} if fast.key else {})})
            with urllib.request.urlopen(r, timeout=3) as resp:
                ans = json.loads(resp.read())["choices"][0]["message"].get("content", "").strip().upper()
        except (OSError, ValueError, KeyError, IndexError):
            return "fast", reason + " (Modell nicht befragt)"
        if ans.startswith("S"):
            return "big", "Modell: schwer"
        if ans.startswith("E"):
            return "fast", "Modell: einfach"
        return "fast", reason

    def record(self, backend: Backend, reason: str, chars: int, stream: bool) -> None:
        with self.lock:
            self.counts[backend.name] = self.counts.get(backend.name, 0) + 1
            self.recent.append({"t": int(time.time()), "route": backend.name, "reason": reason,
                                "model": backend.group or backend.label, "where": backend.where,
                                "chars": chars, "stream": stream})
            del self.recent[:-50]

    # HTTP --------------------------------------------------------------------

    def handler(self):
        from http.server import BaseHTTPRequestHandler
        import http.client
        from urllib.parse import urlsplit

        router = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _auth(self) -> bool:
                if not router.key or self.path.startswith("/health"):
                    return True
                got = self.headers.get("Authorization", "")
                return secrets.compare_digest(got.encode(), f"Bearer {router.key}".encode())

            def _json(self, code: int, obj: dict) -> None:
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if not self._auth():
                    return self._json(401, {"error": {"message": "Invalid API Key", "type": "authentication_error"}})
                path = self.path.split("?", 1)[0]
                if path in ("/v1/models", "/models"):
                    now = int(time.time())
                    data = [{"id": router.public_name, "object": "model", "owned_by": "myrmid", "created": now}]
                    for role in ("fast", "big"):
                        bs = router.role(role)
                        if bs:
                            data.append({"id": role, "object": "model", "owned_by": "myrmid", "created": now,
                                         "meta": {"models": sorted({b.label for b in bs})}})
                    seen = set()
                    for b in router.backends:
                        g = b.group or b.label
                        if g.lower() in seen or g.lower() in (router.public_name.lower(), "fast", "big"):
                            continue
                        seen.add(g.lower())
                        where = sorted({x.where for x in router.backends if (x.group or x.label) == g})
                        data.append({"id": g, "object": "model", "owned_by": "myrmid", "created": now,
                                     "meta": {"model": b.label, "hosts": where}})
                    return self._json(200, {"object": "list", "data": data})
                if path == "/health":
                    ok = any(b.healthy for b in router.backends)
                    return self._json(200 if ok else 503, {"status": "ok" if ok else "loading"})
                self._forward(router.big, "GET", None, "")

            def do_POST(self):
                if not self._auth():
                    return self._json(401, {"error": {"message": "Invalid API Key", "type": "authentication_error"}})
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n else b""
                path = self.path.split("?", 1)[0]
                body = {}
                try:
                    body = json.loads(raw) if raw else {}
                except ValueError:
                    pass
                if path.rstrip("/") in ("/v1/chat/completions", "/chat/completions", "/v1/completions",
                                        "/completions", "/completion") and isinstance(body, dict):
                    try:
                        backend, reason = router.decide(body, self.headers)
                    except ContextTooLong as e:
                        # In OpenAI's words, so clients (opencode, ...) shorten the conversation themselves
                        return self._json(400, {"error": {
                            "message": f"This model's maximum context length is {e.biggest} tokens. However, your "
                                       f"messages resulted in about {e.need} tokens. (MyrmidLM: no model has room "
                                       f"for this request; give one more context in the dashboard.)",
                            "type": "invalid_request_error", "param": "messages",
                            "code": "context_length_exceeded"}})
                    chars = len(json.dumps(body.get("messages") or body.get("prompt") or ""))
                    router.record(backend, reason, chars, bool(body.get("stream")))
                else:
                    backend, reason = router._pick_role("big", "keine Chat-Anfrage")
                if backend.asleep and router.waker:
                    woke = router.waker(backend)
                    if woke is None:
                        return self._json(503, {"error": {"message": f"{backend.where} sleeps and did not wake up "
                                                                     f"within {WAKE_WAIT_S} s"}})
                    backend = woke
                self._forward(backend, "POST", raw, reason)

            def _forward(self, backend: Backend, method: str, raw: Optional[bytes], reason: str) -> None:
                u = urlsplit(backend.base)
                conn_cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
                conn = conn_cls(u.hostname, u.port or (443 if u.scheme == "https" else 80), timeout=900)
                headers = {"Content-Type": self.headers.get("Content-Type", "application/json"),
                           "Accept": self.headers.get("Accept", "*/*")}
                if backend.key:
                    headers["Authorization"] = f"Bearer {backend.key}"
                path = (u.path.rstrip("/") + self.path) if u.path not in ("", "/") else self.path
                with router.lock:
                    alone = backend.inflight == 0
                    backend.inflight += 1
                    backend.served += 1
                    router.last_request = time.time()
                tail = b""
                try:
                    try:
                        conn.request(method, path, body=raw, headers=headers)
                        resp = conn.getresponse()
                    except OSError as e:
                        return self._json(502, {"error": {"message": f"{backend.where} ({backend.label}) "
                                                                     f"unreachable: {e}"}})
                    try:
                        self.send_response(resp.status)
                        self.send_header("Content-Type", resp.getheader("Content-Type", "application/json"))
                        self.send_header("X-Myrmid-Route", backend.name)
                        self.send_header("X-Myrmid-Backend", f"{backend.group or backend.label}@{backend.where}"
                                         .encode("ascii", "replace").decode())
                        self.send_header("X-Myrmid-Reason", reason.encode("ascii", "replace").decode())
                        self.send_header("Transfer-Encoding", "chunked")
                        self.end_headers()
                        while True:
                            chunk = resp.read1(65536)
                            if not chunk:
                                break
                            self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                            self.wfile.flush()
                            tail = (tail + chunk)[-16384:]
                        self.wfile.write(b"0\r\n\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        pass  # client went away; closing the backend stops its generation
                finally:
                    conn.close()
                    with router.lock:
                        alone = alone and backend.inflight == 1
                        backend.inflight -= 1
                    found = re.findall(rb'"timings"\s*:\s*(\{[^{}]*\})', tail) if alone and tail else []
                    if found:
                        try:
                            t = json.loads(found[-1])
                            tps = float(t.get("predicted_per_second") or 0)
                            if tps > 0 and (t.get("predicted_n") or 0) >= 8:  # the speed for "fastest first"
                                backend.tps = tps if not backend.tps else 0.7 * backend.tps + 0.3 * tps
                            if router.on_timings and backend.source == "local" and backend.name == "big":
                                router.on_timings(t)
                        except Exception as e:  # noqa: BLE001 - measuring must never break serving
                            log(f"Calibration: {e}")

        return H

    def start(self, host: str, port: int) -> bool:
        import threading

        threading.Thread(target=self.watch_health, daemon=True).start()
        return _start_http(host, port, self.handler()) is not None


# ── Host manager (master) ───────────────────────────────────────────────────
#
# Finds the hosts (mDNS, or --hosts), polls their state every 5 s, pushes
# the model each one should load (from the dashboard or --assign, kept in
# hosts.json) and gives the router a backend for every host that is ready.
# A host keeps its model when the master restarts or forgets it.

HOSTS_STATE = CONFIG_DIR / "hosts.json"


class HostManager:
    def __init__(self, args, router: Router, key: str, known: Optional[Known] = None):
        import threading

        # Addresses where a PC worker answers right now (the dashboard's LAN scan):
        # a sleeping host there is that PC from before it became a worker.
        self.worker_ips: set[str] = set()

        self.args, self.router, self.key = args, router, key
        self.known = known or Known()
        self.sleep_min: dict[str, int] = {}  # host id -> minutes without requests before it sleeps
        self.last_active: dict[str, float] = {}
        self.last_served: dict[str, int] = {}
        self.lock = threading.RLock()
        self.hosts: dict[str, dict] = {}  # id -> addr, port, name, status, seen, error
        self.assign: dict[str, Optional[dict]] = {}  # id -> wanted assignment (None: unload)
        self.pending_cli = [self._parse_assign(x) for x in (args.assign or [])]
        self.extra = []
        for ep in (args.hosts or "").split(","):
            ep = ep.strip()
            if ep:
                h, _, p = ep.rpartition(":")
                self.extra.append((h or ep, int(p) if h else HOST_CTL_PORT))
        try:
            saved = json.loads(HOSTS_STATE.read_text())
            self.assign, self.sleep_min = saved.get("assign", {}), saved.get("sleep", {})
        except (OSError, ValueError):
            pass
        # hosts seen before that are off or asleep now: listed, and woken on demand
        for mid, rec in self.known.machines.items():
            if rec.get("kind") == "host" and rec.get("ip") and rec.get("port"):
                self.hosts[mid] = {"addr": rec["ip"], "port": int(rec["port"]), "path": rec.get("path") or "",
                                   "name": rec.get("name", rec["ip"]), "status": {}, "seen": 0, "error": "",
                                   "asleep": True, "pc": rec.get("pc") or mid}
        self.last_scan = 0.0
        self.rescan = False  # a PC changed its layout: ask every host again
        self.pushed: dict[str, tuple] = {}  # id -> (assignment, time) last sent

    @staticmethod
    def _parse_assign(text: str) -> dict:
        """HOST=MODEL[,role=big|fast|extra][,name=NAME][,ctx=N]"""
        target, _, rest = text.partition("=")
        parts = [p.strip() for p in rest.split(",")]
        a = {"match": target.strip(), "model": parts[0], "role": "big", "name": "", "ctx": None}
        for p in parts[1:]:
            k, _, v = p.partition("=")
            if k == "role" and v in ("big", "fast", "extra"):
                a["role"] = v
            elif k == "name":
                a["name"] = v
            elif k == "ctx" and v.isdigit():
                a["ctx"] = int(v)
        if not a["match"] or not a["model"]:
            die(f"--assign {text}: expected HOST=MODEL[,role=big|fast|extra][,name=NAME][,ctx=N]")
        return a

    def save(self) -> None:
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            HOSTS_STATE.write_text(json.dumps({"assign": self.assign, "sleep": self.sleep_min}))
        except OSError as e:
            log(f"Hosts: could not save the assignments ({e})")

    # Talking to hosts ----------------------------------------------------------

    def _call(self, h: dict, path: str, body: Optional[dict] = None, timeout: float = 5) -> dict:
        import urllib.request

        req = urllib.request.Request(f"http://{h['addr']}:{h['port']}{h.get('path') or ''}{path}",
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Authorization": f"Bearer {self.key}",
                                              "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def discover(self) -> None:
        found = []
        try:
            found = [(p.host, p.port) for p in scan(3.0) if p.kind == "host"]
        except Exception:  # noqa: BLE001 - mDNS trouble: --hosts still works
            pass
        for addr, port in found + self.extra:
            # A PC with one model per GPU has a host for each GPU (path /gpuN); the
            # first answers without a path too and lists them all.
            known_here = [h for h in self.hosts.values() if h["addr"] == addr and h["port"] == port]
            if known_here and all(not h.get("asleep") for h in known_here) and \
                    time.time() - min(h.get("seen", 0) for h in known_here) < 60 and not self.rescan:
                continue
            try:
                root = self._call({"addr": addr, "port": port}, "/status")
            except Exception as e:  # noqa: BLE001
                if not known_here:
                    log(f"Host {addr}:{port} doesn't answer ({e}); a wrong key gives HTTP 401")
                continue
            if root.get("state") == "pairing":
                continue  # waits for its key (the pairer gives it this master's)
            for slot in root.get("slots") or [{"id": root["id"], "path": ""}]:
                if slot["id"] in self.hosts and not self.hosts[slot["id"]].get("asleep"):
                    continue
                try:
                    st = root if slot["id"] == root["id"] else \
                        self._call({"addr": addr, "port": port, "path": slot["path"]}, "/status")
                except Exception:  # noqa: BLE001 - switching layouts: next time
                    continue
                name = st.get("name") or st.get("host", addr)
                with self.lock:
                    self.hosts[st["id"]] = {"addr": addr, "port": port, "path": slot["path"], "name": name,
                                            "status": st, "seen": time.time(), "error": "",
                                            "asleep": st.get("state") == "asleep", "pc": st.get("pc") or st["id"]}
                self.known.note(st["id"], "host", name, addr, {**st, "port": port, "path": slot["path"]})
                log(f"Host {name} ({addr}:{port}{slot['path']}) found: {st.get('state')}"
                    + (f", {st.get('model')}" if st.get("model") else ""))
        self.rescan = False

    def _prune(self, st: dict) -> None:
        """A PC switched its layout: its hosts that no longer exist go (their
        assignment stays, for when it switches back)."""
        pc, slots = st.get("pc"), st.get("slots")
        if not pc or slots is None:
            return
        valid = {x["id"] for x in slots}
        with self.lock:
            gone = [hid for hid, h in self.hosts.items() if h.get("pc") == pc and hid not in valid]
            for hid in gone:
                log(f"Host {self.hosts[hid]['name']}: gone (the PC now has {len(valid)} model slot(s))")
                del self.hosts[hid]
                self.known.forget(hid)
        if len(valid) > sum(1 for h in self.hosts.values() if h.get("pc") == pc):
            self.rescan = True
            self.last_scan = 0  # find the new ones now

    def poll(self) -> None:
        for hid, h in list(self.hosts.items()):
            if hid not in self.hosts:
                continue  # pruned by a sibling
            try:
                st = self._call(h, "/status", timeout=3)
                if st.get("id") != hid:  # the PC switched layouts: this path belongs to another host now
                    self._prune(st)
                    if hid in self.hosts and st.get("pc") == self.hosts[hid].get("pc"):
                        with self.lock:
                            self.hosts.pop(hid, None)
                        self.known.forget(hid)
                    self.rescan, self.last_scan = True, 0
                    continue
                with self.lock:
                    h.update(status=st, seen=time.time(), error="", asleep=st.get("state") == "asleep",
                             pc=st.get("pc") or hid)
                self.known.note(hid, "host", h["name"], h["addr"], {**st, "port": h["port"], "path": h.get("path", "")})
                self._prune(st)
            except Exception as e:  # noqa: BLE001
                with self.lock:
                    if hid in self.known.machines:
                        h["asleep"] = True  # off or asleep: it can be woken
                        h["error"] = ""
                    else:
                        h["error"] = str(e)[:160]
                        if time.time() - h["seen"] > 300:  # gone for 5 minutes: forget it until it's back
                            del self.hosts[hid]
                continue
            self._apply_cli(hid, h)
            self.ensure(hid)

    def _apply_cli(self, hid: str, h: dict) -> None:
        for a in list(self.pending_cli):
            if a["match"] in (hid, h["name"], h["addr"], f"{h['addr']}:{h['port']}"):
                self.assign[hid] = {k: a[k] for k in ("model", "role", "name", "ctx")}
                self.pending_cli.remove(a)
                self.save()

    def ensure(self, hid: str) -> None:
        """Sends the wanted assignment when the host has another one."""
        h = self.hosts.get(hid)
        if not h or hid not in self.assign:
            return
        want = self.assign[hid]
        have = (h["status"] or {}).get("assignment")
        if want is None:
            if have:
                self._send(hid, "/unload", {})
            return
        keys = ("model", "role", "name", "ctx")
        if have and all(have.get(k) == want.get(k) for k in keys):
            return
        last = self.pushed.get(hid)
        if last and last[0] == json.dumps(want, sort_keys=True) and time.time() - last[1] < 60:
            return  # sent a moment ago; the host reports it soon
        self._send(hid, "/assign", want)

    def _send(self, hid: str, path: str, body: dict) -> None:
        h = self.hosts[hid]
        try:
            st = self._call(h, path, body, timeout=10)
            with self.lock:
                h["status"] = st
            self.pushed[hid] = (json.dumps(body, sort_keys=True), time.time())
            log(f"Host {h['name']}: {path.strip('/')}" + (f" {body['model']}" if body.get("model") else ""))
        except Exception as e:  # noqa: BLE001
            h["error"] = f"{path}: {e}"[:160]

    def update_router(self) -> None:
        backends = []
        with self.lock:
            for hid, h in self.hosts.items():
                st = h.get("status") or {}
                rec = self.known.machines.get(hid, {})
                if h.get("asleep") and h["addr"] in self.worker_ips:
                    continue  # the PC is a worker now
                if h.get("asleep"):
                    # still offered: the router wakes it when a request needs it
                    a = (st.get("assignment") or self.assign.get(hid) or {})
                    label = st.get("model") or rec.get("model") or ""
                    if not a or not label:
                        continue
                    backends.append(Backend(a.get("role") or "big",
                                            f"http://{h['addr']}:{st.get('api_port') or rec.get('api_port') or HOST_API_PORT}"
                                            f"{h.get('path') or ''}",
                                            self.key, label, group=a.get("name") or label, where=h["name"],
                                            source=hid, ctx=st.get("ctx") or 0, asleep=True, status="asleep"))
                    continue
                a = st.get("assignment") or {}
                if st.get("state") != "ready" or not st.get("model") or h.get("error"):
                    continue
                backends.append(Backend(a.get("role") or "big",
                                        f"http://{h['addr']}:{st.get('api_port', HOST_API_PORT)}{h.get('path') or ''}",
                                        self.key, st["model"], group=a.get("name") or st["model"],
                                        where=h["name"], source=hid, ctx=st.get("ctx") or 0,
                                        est_tps=float(st.get("estimate_tps") or 0)))
        self.router.set_hosts(backends)

    # Sleep and wake ------------------------------------------------------------

    def check_sleep(self) -> None:
        """Hosts with a sleep time that had no request for that long go to sleep."""
        now = time.time()
        for hid, h in list(self.hosts.items()):
            st = h.get("status") or {}
            mins = int(self.sleep_min.get(hid, self.args.host_sleep or 0) or 0)
            served = sum(b.served for b in self.router.backends if b.source == hid)
            busy = any(b.inflight for b in self.router.backends if b.source == hid) or \
                (st.get("metrics") or {}).get("requests_processing")
            if busy or served != self.last_served.get(hid) or st.get("state") != "ready":
                self.last_active[hid] = now
                self.last_served[hid] = served
                continue
            if not mins or h.get("asleep") or not st.get("can_sleep") or hid not in self.known.machines:
                continue
            if now - self.last_active.setdefault(hid, now) > mins * 60:
                # a PC with a model per GPU sleeps only when all of them may
                siblings = [x for x, y in self.hosts.items() if y.get("pc") == h.get("pc") and x != hid]
                if all(self._may_sleep(x, now) for x in siblings):
                    self.sleep(hid, f"{mins} min without requests")

    def _may_sleep(self, hid: str, now: float) -> bool:
        h = self.hosts[hid]
        st = h.get("status") or {}
        mins = int(self.sleep_min.get(hid, self.args.host_sleep or 0) or 0)
        if h.get("asleep") or not st.get("assignment"):
            return True
        return bool(mins) and st.get("state") == "ready" and now - self.last_active.get(hid, now) > mins * 60

    def sleep(self, hid: str, why: str) -> None:
        h = self.hosts.get(hid)
        if not h:
            return
        log(f"Host {h['name']}: going to sleep ({why})")
        self._send(hid, "/sleep", {})
        if (h.get("status") or {}).get("state") == "asleep":
            with self.lock:
                for y in self.hosts.values():  # the whole PC sleeps
                    if y.get("pc") == h.get("pc"):
                        y["asleep"] = True
            self.update_router()

    def wake(self, hid: str, force: bool = False) -> bool:
        ok = self.known.wake(hid, force)
        if ok:
            self.last_active[hid] = time.time()
        return ok

    def wake_and_wait(self, backend: Backend) -> Optional[Backend]:
        """For the router: wakes the host behind a backend and waits until its
        model answers (the request waits that long). Returns the backend to
        use then (while loading, the host has none, so it is a new one)."""
        hid = backend.source
        if not self.wake(hid) and not self.known.recently_woken(hid):
            return None
        deadline = time.time() + WAKE_WAIT_S
        while time.time() < deadline:
            ready = next((b for b in self.router.backends if b.source == hid and b.healthy and not b.asleep), None)
            if ready:
                return ready
            time.sleep(2)
        return None

    def loop(self) -> None:
        while True:
            try:
                if time.time() - self.last_scan > 30:
                    self.last_scan = time.time()
                    self.discover()
                self.poll()
                self.update_router()
                self.check_sleep()
            except Exception as e:  # noqa: BLE001 - keep managing
                log(f"Hosts: {e}")
            time.sleep(5)

    def start(self) -> None:
        import threading

        threading.Thread(target=self.loop, daemon=True).start()

    # Dashboard ---------------------------------------------------------------

    def api(self, body: dict) -> dict:
        import threading

        hid = str(body.get("host") or "")
        if hid not in self.hosts:
            return {"error": "unknown host"}
        if body.get("wake"):
            self.wake(hid, force=True)
            return self.info()
        if body.get("layout"):
            h = dict(self.hosts[hid], path="")  # the PC's first host answers for the layout
            try:
                res = self._call(h, "/layout", {"layout": str(body["layout"])}, timeout=10)
            except Exception as e:  # noqa: BLE001
                return {**self.info(), "error": f"layout: {e}"}
            if not res.get("ok"):
                return {**self.info(), "error": res.get("error") or "layout failed"}
            log(f"Host {h['name']}: layout {body['layout']}")
            # The model the PC's first host ran keeps running on the new first host
            # (the host does that too; the master must not push the old wish back).
            pc = self.hosts[hid].get("pc") or hid
            with self.lock:
                old_first = next((x for x, y in self.hosts.items()
                                  if y.get("pc") == pc and (y.get("status") or {}).get("layout_owner")), None)
                want = self.assign.get(old_first) if old_first else None
                if want is None and old_first:
                    want = (self.hosts[old_first].get("status") or {}).get("assignment")
                new_first = pc if body["layout"] == "split" else f"{pc}-gpu0"
                if want:
                    self.assign[new_first] = {k: want.get(k) for k in ("model", "role", "name", "ctx")}
                    self.pushed.pop(new_first, None)
            self.save()
            self.rescan, self.last_scan = True, 0
            return self.info()
        if body.get("sleep"):
            threading.Thread(target=self.sleep, args=(hid, "from the dashboard"), daemon=True).start()
            return self.info()
        if "sleep_min" in body:
            v = str(body.get("sleep_min") or "0")
            self.sleep_min[hid] = int(v) if v.isdigit() else 0
            self.last_active[hid] = time.time()
            self.save()
            return self.info()
        if body.get("unload"):
            self.assign[hid] = None
        else:
            model = str(body.get("model") or "").strip()
            if not model:
                return {"error": "model missing"}
            ctx = body.get("ctx")
            self.assign[hid] = {"model": model, "role": body.get("role") if body.get("role") in ("big", "fast", "extra")
                                else "big", "name": str(body.get("name") or "").strip(),
                                "ctx": int(ctx) if str(ctx or "").isdigit() else None}
        self.save()
        self.pushed.pop(hid, None)
        threading.Thread(target=self.ensure, args=(hid,), daemon=True).start()
        return self.info()

    def info(self) -> dict:
        with self.lock:
            hosts = []
            for hid, h in self.hosts.items():
                st = h.get("status") or {}
                if h.get("asleep") and h["addr"] in self.worker_ips:
                    continue  # the PC is a worker now; switched back to host, it shows up again
                hosts.append({"id": hid, "name": h["name"], "addr": f"{h['addr']}:{h['port']}",
                              "error": h.get("error", ""), "want": self.assign.get(hid, "–"),
                              "asleep": bool(h.get("asleep")), "wakeable": hid in self.known.machines,
                              "waking": self.known.recently_woken(hid),
                              "sleep_min": int(self.sleep_min.get(hid, self.args.host_sleep or 0) or 0),
                              "can_sleep": bool((h.get("status") or {}).get("can_sleep")),
                              "switchable": bool((h.get("status") or {}).get("switchable")),
                              "pc": h.get("pc"), "path": h.get("path") or "",
                              **{k: st.get(k) for k in ("state", "error", "assignment", "model", "ctx",
                                                       "estimate_tps", "progress", "models", "stats", "layout",
                                                       "gpus", "layout_owner", "slot", "gpu", "switching")},
                              "busy": (st.get("metrics") or {}).get("requests_processing")})
                if h.get("error"):
                    hosts[-1]["state"], hosts[-1]["error"] = "unreachable", h["error"]
                elif h.get("asleep"):
                    hosts[-1]["state"] = "asleep"
        return {"hosts": sorted(hosts, key=lambda x: x["name"]),
                "catalog": [{"key": k, "desc": d} for k, _r, _q, d in CATALOG]}


def _router_initial(arg: str, has_fast: bool) -> tuple[bool, str]:
    saved = {}
    try:
        saved = json.loads(ROUTER_STATE.read_text())
    except (OSError, ValueError):
        pass
    mode = saved.get("mode", "heuristic")
    if arg == "on":
        return True, mode
    if arg == "off":
        return False, mode
    return bool(saved.get("enabled", has_fast)), mode


def plan_fast(args, devices: list[Device]) -> tuple[list[str], dict, dict]:
    """Where the fast model runs: --fast-device, or the local GPU it fits on
    best (smallest that fits), or the CPU. Returns the llama-server command,
    the VRAM to keep free on that GPU for the big model's planning, and info."""
    model = str(Path(args.fast_model).expanduser().resolve())
    if not Path(model).is_file():
        die(f"No such model: {model}")
    m = gguf_info(model, args.fast_ctx)
    need = m.size_mb + m.kv_layer_mb * m.n_layer + GPU_RESERVE_MB
    gpus = usable_gpus(devices)
    dev: Optional[Device] = None
    if args.fast_device:
        if args.fast_device.upper() != "CPU":
            dev = next((d for d in devices if d.name == args.fast_device), None)
            if dev is None:
                die(f"--fast-device {args.fast_device}: no such device ({', '.join(d.name for d in devices)})")
    else:
        fits = [d for d in gpus if d.free_mb * GPU_SHARE >= need]
        dev = min(fits, key=lambda d: d.free_mb) if fits else None
        if dev is None:
            log(f"Fast model ({need/1024:.1f} GB with cache) fits on no GPU here: it runs on the CPU.")
    port = args.port + 2
    cmd = [find_bin("llama-server"), "-m", model, "-c", str(args.fast_ctx), "--fit", "off",
           "--host", "127.0.0.1", "--port", str(port), "--metrics"]
    cmd += ["-ngl", "999", "--device", dev.name] if dev else ["-ngl", "0", "--device", "none"]
    reserve = {dev.name: need} if dev else {}
    if dev is None or dev.type == "igpu":
        reserve["__host__"] = need  # on the CPU or an iGPU it lives in this PC's RAM
    info = {"model": Path(model).stem, "device": f"{dev.name} {dev.description}" if dev else "CPU",
            "size_gb": round(m.size_mb / 1024, 1), "ctx": args.fast_ctx, "port": port,
            "need_gb": round(need / 1024, 1)}
    return cmd, reserve, info


# ── Draft model (speculative decoding) ──────────────────────────────────────
#
# A small model of the same family (same tokenizer) writes a few tokens
# ahead; the big model checks them in one pass and keeps the ones it would
# have written. Same output, often 1.5-2x the tokens per pass. It pays most
# when the big model is split over the network: every pass costs network
# round trips, and the draft runs on the master without any.

DRAFT_CATALOG = [
    # key, repo, quant, for which architectures (prefix match; vocab is checked anyway)
    ("qwen3.5-0.8b", "unsloth/Qwen3.5-0.8B-GGUF", "Q8_0", ("qwen35", "qwen36", "qwen37", "qwen38")),
    ("qwen3-0.6b", "bartowski/Qwen_Qwen3-0.6B-GGUF", "Q8_0", ("qwen3",)),
    ("llama-3.2-1b", "bartowski/Llama-3.2-1B-Instruct-GGUF", "Q8_0", ("llama",)),
]
DRAFT_MAX_SHARE = 0.12  # a draft larger than this share of the big model rarely pays off
DRAFT_MIN_MB = 500  # the smallest catalog drafts (Q8_0) are about this size
DRAFT_N_MAX = 8


def gguf_vocab(path: str) -> Optional[dict]:
    """Tokenizer type, BOS/EOS and the token list from a GGUF header."""
    scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    want = {"tokenizer.ggml.model", "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.add_bos_token", "tokenizer.ggml.add_eos_token", "tokenizer.ggml.tokens"}
    out: dict = {}
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return None
            _v, _nt, n_kv = struct.unpack("<IQQ", f.read(20))

            def read_str() -> str:
                (n,) = struct.unpack("<Q", f.read(8))
                return f.read(n).decode("utf-8", "replace")

            def read_val(t: int, keep: bool):
                if t in scalar:
                    fmt = "<" + scalar[t]
                    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
                if t == 8:
                    return read_str()
                if t == 9:
                    et, n = struct.unpack("<IQ", f.read(12))
                    if et in scalar and not keep:
                        f.seek(struct.calcsize("<" + scalar[et]) * n, 1)
                        return None
                    vals = [read_val(et, keep) for _ in range(n)]
                    return vals if keep else None
                raise ValueError(f"GGUF value type {t}")

            for _ in range(n_kv):
                key = read_str()
                (t,) = struct.unpack("<I", f.read(4))
                v = read_val(t, key in want)
                if key in want:
                    out[key] = v
                if len(out) == len(want):
                    break
    except (OSError, struct.error, ValueError):
        return None
    return out


def vocab_compatible(target: str, draft: str) -> tuple[bool, str]:
    """llama.cpp's own check (common_speculative_are_compatible): same
    tokenizer type, same BOS/EOS, sizes within 128, same tokens from id 5."""
    a, b = gguf_vocab(target), gguf_vocab(draft)
    if not a or not b:
        return False, "could not read the tokenizer"
    if a.get("tokenizer.ggml.model") != b.get("tokenizer.ggml.model"):
        return False, f"tokenizer {a.get('tokenizer.ggml.model')} vs {b.get('tokenizer.ggml.model')}"
    for k, d in (("add_bos_token", False), ("add_eos_token", False)):
        if bool(a.get(f"tokenizer.ggml.{k}", d)) != bool(b.get(f"tokenizer.ggml.{k}", d)):
            return False, k.replace("_", " ") + " differs"
    for k, flag in (("bos_token_id", "add_bos_token"), ("eos_token_id", "add_eos_token")):
        if a.get(f"tokenizer.ggml.{flag}") and a.get(f"tokenizer.ggml.{k}") != b.get(f"tokenizer.ggml.{k}"):
            return False, k.replace("_", " ") + " differs"
    ta, tb = a.get("tokenizer.ggml.tokens") or [], b.get("tokenizer.ggml.tokens") or []
    if not ta or not tb or abs(len(ta) - len(tb)) > 128:
        return False, f"vocab size {len(ta)} vs {len(tb)}"
    n = min(len(ta), len(tb))
    if ta[5:n] != tb[5:n]:
        return False, "different tokens"
    return True, ""


def find_draft(args, main: str) -> Optional[str]:
    """The draft model for --draft-model: a path, or auto (a fitting GGUF on
    this PC, else a small model of the same family from the catalog)."""
    choice = (args.draft_model or "off").strip()
    if choice.lower() in ("off", "none", "no", "0", ""):
        return None
    if choice.lower() != "auto":
        p = Path(choice).expanduser()
        if not p.is_file():
            log(f"Draft model {choice}: no such file; running without")
            return None
        ok, why = vocab_compatible(main, str(p))
        if not ok:
            log(f"Draft model {p.name} doesn't match {Path(main).name} ({why}); running without")
            return None
        return str(p.resolve())
    main_mb = os.path.getsize(main) / 2**20
    cands = [Path(e["path"]) for e in local_models()
             if e["size_mb"] <= main_mb * DRAFT_MAX_SHARE and Path(e["path"]).resolve() != Path(main).resolve()]
    for p in sorted(cands, key=lambda c: -c.stat().st_size):  # the largest that is small enough
        if vocab_compatible(main, str(p))[0]:
            log(f"Draft model: {p.name} (found on this PC)")
            return str(p.resolve())
    arch = _gguf_arch(main)
    entry = next((e for e in DRAFT_CATALOG if any(arch == a or (a != "qwen3" and arch.startswith(a))
                                                  or (a == "qwen3" and arch in ("qwen3", "qwen3moe"))
                                                  for a in e[3])), None)
    if entry is not None and main_mb * DRAFT_MAX_SHARE < DRAFT_MIN_MB:
        # 0.6B drafting for 1.7B: the draft costs nearly as much as it saves
        log(f"No draft model: {Path(main).name} is small ({main_mb / 1024:.1f} GB); a draft pays off from "
            f"about {DRAFT_MIN_MB / DRAFT_MAX_SHARE / 1024:.0f} GB on")
        return None
    if entry is None:
        log(f"No draft model known for architecture {arch or '?'}; give one with --draft-model PATH")
        return None
    dest = Path(main).parent if os.access(Path(main).parent, os.W_OK) else CONFIG_DIR / "models"
    try:
        p = fetch_model(f"{entry[1]}:{entry[2]}", dest, quiet=True)
    except ModelError as e:
        log(f"Draft model {entry[0]}: {e}; running without")
        return None
    ok, why = vocab_compatible(main, str(p))
    if not ok:
        log(f"Draft model {p.name} doesn't match {Path(main).name} ({why}); running without")
        return None
    log(f"Draft model: {p.name} (downloaded)")
    return str(p)


def plan_draft(args, path: str, devices: list[Device]) -> tuple[list[str], dict, dict]:
    """Where the draft runs: the local GPU with the most room (it is small,
    and every pass needs it), else the CPU. Returns its llama-server
    options, the memory to keep free for it, and info."""
    m = gguf_info(path, args.ctx or CTX_STEPS[0])
    need = m.size_mb + m.kv_layer_mb * m.n_layer + 300
    gpus = [d for d in usable_gpus(devices) if d.free_mb * GPU_SHARE - (getattr(args, "reserve_mb", None) or {}).get(d.name, 0) >= need]
    dev = max(gpus, key=lambda d: d.free_mb) if gpus else None
    opts = ["--spec-type", "draft-simple", "-md", path, "--spec-draft-n-max", str(args.draft_max)]
    opts += ["-devd", dev.name, "-ngld", "all"] if dev else ["-devd", "none", "-ngld", "0"]
    reserve = {dev.name: need} if dev else {}
    if dev is None or dev.type == "igpu":
        reserve["__host__"] = need
    info = {"model": Path(path).stem, "device": f"{dev.name} {dev.description}" if dev else "CPU",
            "size_gb": round(m.size_mb / 1024, 2), "n_max": args.draft_max}
    return opts, reserve, info


# ── Models ───────────────────────────────────────────────────────────────────
#
# myrmid models   GGUF files on this PC plus a small catalog, each
#                         rated: fits here / needs workers / too big, and
#                         whether this build knows its architecture.
# myrmid download fetches a catalog model (or repo[:quant]) from
#                         Hugging Face, resumable, checked by SHA-256.
#
# The catalog names repositories only; file names, sizes and architectures
# are looked up at run time, so it doesn't go stale with renamed files.

CATALOG = [
    # key, Hugging Face repo, quantization, description
    ("qwen3-1.7b", "bartowski/Qwen_Qwen3-1.7B-GGUF", "Q4_K_M", "sehr klein: Router, schwache Rechner"),
    ("qwen3-4b", "bartowski/Qwen_Qwen3-4B-GGUF", "Q4_K_M", "klein: Alltag, Router"),
    ("qwen3.5-9b", "unsloth/Qwen3.5-9B-GGUF", "Q4_K_M", "mittel"),
    ("qwen3-14b", "bartowski/Qwen_Qwen3-14B-GGUF", "Q4_K_M", "mittel bis groß"),
    ("qwen3.8-27b", "unsloth/Qwen3.8-27B-GGUF", "Q4_K_M", "groß, dicht: lohnt den Verbund"),
    ("qwen3-32b", "bartowski/Qwen_Qwen3-32B-GGUF", "Q4_K_M", "groß, dicht: lohnt den Verbund"),
    ("qwen3.6-35b-a3b", "unsloth/Qwen3.6-35B-A3B-GGUF", "UD-Q4_K_M", "MoE: schnell bei viel RAM im Master"),
]
MODEL_DIRS = ["~/models", "~/Models", "~/Downloads", "/srv/models", "/opt/models", "~/.cache/llama.cpp",
              "~/.cache/huggingface/hub", "~/.lmstudio/models", "~/.cache/lm-studio/models"]
SPLIT_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
HF_BASE = env("HF_BASE", "https://huggingface.co").rstrip("/")  # or a mirror


def supported_archs() -> Optional[set]:
    """Architectures this build's llama.cpp knows (written by build.sh)."""
    for p in (BIN_DIR / "supported-archs.txt", BIN_DIR.parent / "build" / "bin" / "supported-archs.txt"):
        if p.exists():
            return {x.strip() for x in p.read_text().splitlines() if x.strip()}
    return None


def _gguf_arch(path: str) -> str:
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return ""
            _v, _n, n_kv = struct.unpack("<IQQ", f.read(20))
            for _ in range(min(n_kv, 8)):
                (kl,) = struct.unpack("<Q", f.read(8))
                key = f.read(kl).decode("utf-8", "replace")
                (t,) = struct.unpack("<I", f.read(4))
                if t != 8:
                    return ""
                (vl,) = struct.unpack("<Q", f.read(8))
                val = f.read(vl).decode("utf-8", "replace")
                if key == "general.architecture":
                    return val
    except (OSError, struct.error):
        pass
    return ""


def local_models() -> list[dict]:
    seen, out = set(), []
    for d in MODEL_DIRS:
        root = Path(os.path.expanduser(d))
        try:
            if not root.is_dir():
                continue
        except OSError:  # e.g. a service without a home: ~ is /, and /.cache belongs to root
            continue
        for dirpath, dirnames, files in os.walk(root, followlinks=True):
            if dirpath.count(os.sep) - str(root).count(os.sep) > 6:
                dirnames[:] = []
                continue
            for fn in files:
                if not fn.lower().endswith(".gguf") or "mmproj" in fn.lower():
                    continue
                m = SPLIT_RE.search(fn)
                if m and m.group(1) != "00001":
                    continue
                p = Path(dirpath) / fn
                try:
                    real = p.resolve()
                    size = real.stat().st_size
                except OSError:
                    continue
                if size < 50 * 2**20 or real in seen:
                    continue
                seen.add(real)
                if m:  # all parts together
                    size = sum(q.stat().st_size for q in real.parent.glob(SPLIT_RE.sub("-*-of-" + m.group(2) + ".gguf", real.name)))
                out.append({"source": "local", "name": p.stem, "path": str(real), "size_mb": size // 2**20,
                            "arch": _gguf_arch(str(real))})
    return sorted(out, key=lambda x: x["size_mb"])


def hf_resolve(repo: str, quant: str) -> dict:
    """The files of one quantization in a Hugging Face repo, with sizes,
    SHA-256 and the GGUF architecture."""
    import urllib.request

    headers = {"User-Agent": "myrmid"}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    req = urllib.request.Request(f"{HF_BASE}/api/models/{repo}?blobs=true", headers=headers)
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read())
    q = quant.lower()
    files = []
    for sib in data.get("siblings", []):
        fn = sib.get("rfilename", "")
        low = fn.lower()
        if not low.endswith(".gguf") or "mmproj" in low:
            continue
        base = low.rsplit("/", 1)[-1]
        if not (base.endswith(f"-{q}.gguf") or base.endswith(f".{q}.gguf") or f"-{q}-0000" in base
                or f"/{q}/" in f"/{low}" or base == f"{q}.gguf"):
            continue
        lfs = sib.get("lfs") or {}
        files.append({"file": fn, "size": sib.get("size") or lfs.get("size") or 0, "sha256": lfs.get("sha256")})
    if not files:
        raise ValueError(f"no {quant} GGUF in {repo}")
    # One model: a single file, or the parts of one split file.
    files.sort(key=lambda f: f["file"])
    first = files[0]
    m = SPLIT_RE.search(first["file"])
    if m:
        stem = SPLIT_RE.sub("", first["file"])
        files = [f for f in files if SPLIT_RE.sub("", f["file"]) == stem]
    else:
        files = [first]
    arch = (data.get("gguf") or {}).get("architecture", "")
    return {"repo": repo, "quant": quant, "files": files, "size_mb": sum(f["size"] for f in files) // 2**20,
            "arch": arch, "name": Path(first["file"]).stem.split("-00001-of-")[0]}


def memory_budget(use_workers: bool) -> tuple[int, int]:
    """MB a model may use on this PC, and on the workers found on the LAN."""
    devices = probe(bench=False)
    host = host_available_mb() * HOST_SHARE - HOST_RESERVE_MB
    local = max(0.0, host)
    for d in usable_gpus(devices):
        if d.type == "gpu":
            local += max(0.0, d.free_mb * GPU_SHARE - GPU_RESERVE_MB)
    remote = 0.0
    if use_workers:
        me = machine_id()
        for p in scan(3.0):
            if p.kind == "worker" and p.txt.get("id", "").split("-ram")[0] != me and p.ram_mb:
                share = GPU_SHARE if p.is_gpu else PC_RAM_SHARE if p.txt.get("kind") == "cpu" else PHONE_SHARE
                remote += p.ram_mb * share
    return int(local), int(remote)


def _rate(size_mb: int, local: int, remote: int) -> str:
    need = size_mb * 1.1 + 600  # + KV cache for ~8k tokens and buffers
    if need <= local:
        return "here"
    if need <= local + remote:
        return "workers"
    return "too_big"


def list_models(use_workers: bool = True, small_only: bool = False) -> dict:
    local, remote = memory_budget(use_workers)
    archs = supported_archs()
    entries = []
    for m in local_models():
        entries.append(m)
    for key, repo, quant, desc in CATALOG:
        e = {"source": "catalog", "key": key, "repo": repo, "quant": quant, "desc": desc}
        try:
            r = hf_resolve(repo, quant)
            e.update(name=r["name"], size_mb=r["size_mb"], arch=r["arch"])
        except Exception as ex:  # noqa: BLE001 - offline, renamed repo: show it, don't fail
            e.update(name=key, size_mb=0, arch="", error=str(ex)[:120])
        entries.append(e)
    for e in entries:
        e["fit"] = _rate(e["size_mb"], local, remote) if e.get("size_mb") else "unknown"
        e["arch_ok"] = None if not e.get("arch") or archs is None else e["arch"] in archs
        if e["source"] == "catalog" and any(x["source"] == "local" and x["name"].lower() == e["name"].lower()
                                              for x in entries):
            e["have"] = True
    if small_only:
        entries = [e for e in entries if 0 < e.get("size_mb", 0) <= 5 * 1024]
    usable = [e for e in entries if e.get("size_mb") and e["arch_ok"] is not False and not e.get("error")]
    here = [e for e in usable if e["fit"] == "here"]
    if small_only:
        rec = min(here or usable, key=lambda e: e["size_mb"], default=None)
    else:
        rec = max(here, key=lambda e: e["size_mb"], default=None) or min(usable, key=lambda e: e["size_mb"], default=None)
    return {"local_mb": local, "workers_mb": remote, "models": entries,
            "recommended": entries.index(rec) if rec else None}


def cmd_models(args) -> None:
    res = list_models(not args.no_workers, args.small)
    if args.json:
        print(json.dumps(res))
        return
    fit = {"here": "passt auf diesen PC", "workers": "nur mit Workern", "too_big": "zu groß",
           "unknown": "unbekannt"}
    print(f"Platz für Modelle: {res['local_mb']/1024:.1f} GB auf diesem PC"
          + (f", {res['workers_mb']/1024:.1f} GB auf Workern im Netz" if res['workers_mb'] else "") + "\n")
    for i, e in enumerate(res["models"], 1):
        size = f"{e['size_mb']/1024:5.1f} GB" if e.get("size_mb") else "    –   "
        state = e.get("error") and "nicht abrufbar" or fit[e["fit"]]
        if e.get("arch_ok") is False:
            state = f"Architektur {e['arch']} braucht neuere llama.cpp"
        where = e.get("path") or f"{e['repo']} ({e['quant']})"
        mark = " *" if res["recommended"] == i - 1 else ""
        print(f"{i:>2}. {e['name']:<34} {size}  {state:<26} {where}{mark}")
    if res["recommended"] is not None:
        print("\n* Empfehlung. Herunterladen mit: myrmid download <Name aus der Liste oder Nummer>")


class ModelError(Exception):
    pass


def resolve_spec(target: str) -> tuple[str, str]:
    """(repo, quant) for a catalog name, a number from `models`, or user/repo[:QUANT]."""
    for key, repo, quant, _d in CATALOG:
        if target.lower() in (key, repo.lower()):
            return repo, quant
    repo, _, quant = target.partition(":")
    if "/" not in repo:
        raise ModelError(f"Unknown model {target}: a catalog name (myrmid models), or user/repo[:QUANT]")
    return repo, quant or "Q4_K_M"


def fetch_model(target: str, out_dir: Path, force: bool = False, quiet: bool = False,
                on_resolved=None, max_mb: Optional[int] = None) -> Path:
    """Downloads a model (resumable, checked against its SHA-256) and returns
    the path of its first file. on_resolved(info) is called once the files
    are known; max_mb refuses models that won't fit before downloading."""
    spec = resolve_spec(target)
    try:
        r = hf_resolve(*spec)
    except Exception as e:  # noqa: BLE001
        raise ModelError(f"Could not look up {spec[0]} ({spec[1]}) on Hugging Face: {e}") from e
    archs = supported_archs()
    if r["arch"] and archs is not None and r["arch"] not in archs and not force:
        raise ModelError(f"{r['name']} uses architecture {r['arch']}, which this llama.cpp build doesn't know. "
                         "Pick another model, or --force.")
    if max_mb is not None and r["size_mb"] * 1.1 + 600 > max_mb:
        raise ModelError(f"{r['name']} needs about {(r['size_mb'] * 1.1 + 600) / 1024:.1f} GB, "
                         f"this PC has {max_mb / 1024:.1f} GB for models")
    out_dir.mkdir(parents=True, exist_ok=True)
    if on_resolved:
        on_resolved({**r, "dir": str(out_dir)})
    paths = []
    for f in r["files"]:
        dest = out_dir / Path(f["file"]).name
        paths.append(dest)
        if dest.exists() and (not f["size"] or dest.stat().st_size == f["size"]):
            log(f"{dest.name} is already there")
            continue
        url = f"{HF_BASE}/{r['repo']}/resolve/main/{f['file']}"
        part = dest.with_suffix(dest.suffix + ".part")
        log(f"Downloading {dest.name} ({f['size']/2**30:.1f} GB) to {out_dir}")
        if shutil.which("curl"):
            cmd = ["curl", "-fL", "--retry", "5", "-C", "-", "-o", str(part), url]
            if quiet:
                cmd.insert(1, "-sS")
            if os.environ.get("HF_TOKEN"):
                cmd[1:1] = ["-H", f"Authorization: Bearer {os.environ['HF_TOKEN']}"]
            if subprocess.run(cmd).returncode != 0:
                raise ModelError(f"Download failed; run the same command again to resume ({part})")
        else:
            _download_urllib(url, part)
        if f.get("sha256"):
            log("Checking SHA-256...")
            h = hashlib.sha256()
            with open(part, "rb") as fh:
                for chunk in iter(lambda: fh.read(8 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != f["sha256"]:
                part.unlink(missing_ok=True)
                raise ModelError(f"{dest.name}: checksum mismatch, file deleted; please try again")
        part.rename(dest)
        dest.chmod(0o644)
    return paths[0]


def cmd_download(args) -> None:
    target = args.model
    if target.isdigit():
        e = list_models(False)["models"][int(target) - 1]
        if e["source"] == "local":
            print(e["path"])
            return
        target = f"{e['repo']}:{e['quant']}"
    try:
        print(fetch_model(target, Path(args.dir).expanduser(), args.force))
    except ModelError as e:
        die(str(e))


def _download_urllib(url: str, part: Path) -> None:
    import urllib.request

    have = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "myrmid", **({"Range": f"bytes={have}-"} if have else {})}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r, \
            open(part, "ab" if have and r.status == 206 else "wb") as out:
        shutil.copyfileobj(r, out, 8 << 20)


def main() -> None:
    ap = argparse.ArgumentParser(prog="myrmid", description=__doc__.split("\n\n")[0])
    ap.add_argument("--version", action="version", version=f"MyrmidLM {own_version()} ({RPC_VERSION})")
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
    w.add_argument("--key", help="cluster key (the master's API key): lets the master switch this PC to host. "
                                 "Without it the first master that sees this PC gives it its key")
    w.add_argument("--key-file", help="file with the cluster key")
    w.add_argument("--pair-from", metavar="IP[,IP]",
                   help="without a key: only this master may take the PC into its cluster "
                        "(otherwise any master, in the first hour after the start)")
    w.set_defaults(fn=cmd_worker)

    h = sub.add_parser("host", help="load whole models for a master's router (the master picks the model)")
    h.add_argument("--key", help="cluster key: the master's API key (better: --key-file). "
                                 "Without it the first master that sees this PC gives it its key")
    h.add_argument("--key-file", help="file with the cluster key")
    h.add_argument("--pair-from", metavar="IP[,IP]",
                   help="without a key: only this master may take the PC into its cluster")
    h.add_argument("--host", default="0.0.0.0", help="address to listen on (default all)")
    h.add_argument("--port", type=int, default=HOST_CTL_PORT, help=f"control API (default {HOST_CTL_PORT})")
    h.add_argument("--api-port", type=int, default=HOST_API_PORT,
                   help=f"the models' API for the master (default {HOST_API_PORT}; the llama-servers "
                        f"themselves listen on 127.0.0.1:{HOST_INTERNAL_PORT} and up)")
    h.add_argument("--models-dir", default="/srv/models" if os.access("/srv/models", os.W_OK) else "~/models",
                   help="where downloaded models go")
    h.add_argument("--advertise-ip")
    h.add_argument("--no-advertise", action="store_true")
    h.add_argument("extra", nargs=argparse.REMAINDER, help="-- further llama-server options")
    h.set_defaults(fn=cmd_host)

    c = sub.add_parser("calibration", help="show (or --reset) what the planner learned from measurements")
    c.add_argument("--reset", action="store_true", help="forget all measurements")
    c.set_defaults(fn=cmd_calibration)

    k = sub.add_parser("key", help="print the master's API key (the cluster key for hosts)")
    k.set_defaults(fn=cmd_key)

    s = sub.add_parser("serve", help="run a model here and serve an OpenAI-compatible API")
    s.add_argument("-m", "--model", required=True)
    s.add_argument("--workers", default="auto",
                   help="auto (workers only where they pay off), all (every worker with room), off, or host:port,...")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=DEFAULT_API_PORT)
    s.add_argument("--ctx", type=int, default=None,
                   help="context length (default 8192, shrunk automatically down to 2048 if it doesn't fit)")
    s.add_argument("--api-key", help="default: generated once, kept in ~/.config/myrmid")
    s.add_argument("--mtp", type=int, default=0,
                   help="MTP draft depth (models with MTP layers; find the best with tune-mtp)")
    s.add_argument("--ram-limit", type=float,
                   help="GB of this PC's RAM the model may use (layers that don't fit go to workers)")
    s.add_argument("--fast-model", help="router: small model for simple requests (runs on one device here)")
    s.add_argument("--fast-url", help="router: use an existing OpenAI-compatible server as the fast model")
    s.add_argument("--fast-key", help="API key for --fast-url")
    s.add_argument("--fast-device", help="device for --fast-model, e.g. CUDA0 or CPU (default: best fit)")
    s.add_argument("--fast-ctx", type=int, default=8192, help="context of the fast model (default 8192)")
    s.add_argument("--router", choices=["auto", "on", "off"], default="auto",
                   help="route requests between the models (auto: last setting from the dashboard, else on)")
    s.add_argument("--router-mode", choices=["heuristic", "model", "examples"],
                   help="heuristic: rules on the request; model: the fast model judges (default: last setting)")
    s.add_argument("--model-name", default="myrmid", help="model name clients see with the router")
    s.add_argument("--moe-cache", default="auto",
                   help="MoE models whose experts don't all fit on the GPU: auto (GPU memory left over caches "
                        "the most used experts), MIB (at most this much), or off (fixed split, --n-cpu-moe)")
    s.add_argument("--moe-cpu-share", type=float,
                   help=f"share of expert-cache misses the CPU computes (default: from tune-moe, else {MOE_CPU_SHARE})")
    s.add_argument("--draft-model", default="off",
                   help="small model of the same family for speculative decoding: PATH, auto (find or "
                        "download one), or off (default)")
    s.add_argument("--draft-max", type=int, default=DRAFT_N_MAX,
                   help=f"tokens the draft writes ahead (default {DRAFT_N_MAX})")
    s.add_argument("--assign", action="append", metavar="HOST=MODEL[,role=..][,name=..][,ctx=N]",
                   help="have a host load a whole model (repeatable): HOST is its name or IP, MODEL a catalog "
                        "name, user/repo[:QUANT] or a file on the host; role big (default), fast, or extra "
                        "(only by name); name is what clients ask for")
    s.add_argument("--hosts", help="hosts mDNS doesn't find: IP[:PORT],... (default port 50070)")
    s.add_argument("--auto-pair", choices=("on", "off"), default="on",
                   help="give new workers and hosts (installed without the key) this master's key, so it can "
                        "switch and update them (default on)")
    s.add_argument("--host-sleep", type=int, default=0, metavar="MIN",
                   help="hosts sleep after MIN minutes without requests and are woken by Wake-on-LAN "
                        "when a request needs them (default 0: never; per host in the dashboard)")
    s.add_argument("--rebalance", choices=["auto", "on", "off"], default="auto",
                   help="re-plan when workers come or go, and wait for workers if the model doesn't fit "
                        "(auto: last setting from the dashboard, else on; off: plan once, exit 2 if it doesn't fit)")
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
    b.add_argument("--api-key", help="default: the key myrmid serve created")
    b.add_argument("--tokens", type=int, default=64)
    b.add_argument("--temp", type=float, default=0.7, help="the temperature you actually use (default 0.7)")
    b.add_argument("--context", type=int, default=0, help="fill the prompt with ~N tokens first")
    b.add_argument("--runs", type=int, default=1)
    b.add_argument("--route", choices=["big", "fast", "auto"], default="big",
                   help="with the router: which model to measure (default big)")
    b.add_argument("--iface", help="count only these interfaces, e.g. eth0 (default: all but lo)")
    b.set_defaults(fn=cmd_bench)

    tm = sub.add_parser("tune-moe", help="measure the best MoE expert cache and CPU share here (stop the server first)")
    tm.add_argument("-m", "--model", required=True)
    tm.add_argument("--ctx", type=int, default=8192)
    tm.add_argument("--ram-limit", type=float)
    tm.add_argument("--max-mib", type=int, help="largest cache to try (default: what fits)")
    tm.add_argument("--port", type=int, default=18080)
    tm.add_argument("--scan-timeout", type=float, default=1.0)
    tm.set_defaults(fn=cmd_tune_moe, api_key="", extra=[], mtp=0)

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

    mo = sub.add_parser("models", help="models on this PC and to download, rated by fit")
    mo.add_argument("--json", action="store_true")
    mo.add_argument("--small", action="store_true", help="only models up to 5 GB (e.g. for the router)")
    mo.add_argument("--no-workers", action="store_true", help="rate by this PC alone")
    mo.set_defaults(fn=cmd_models)

    dl = sub.add_parser("download", help="download a model (catalog name, number from 'models', or user/repo[:QUANT])")
    dl.add_argument("model")
    dl.add_argument("--dir", default="/srv/models" if os.access("/srv/models", os.W_OK) else "~/models")
    dl.add_argument("--force", action="store_true", help="download even if the architecture looks unsupported")
    dl.set_defaults(fn=cmd_download)

    sc = sub.add_parser("scan", help="list workers and servers on the LAN")
    sc.add_argument("--timeout", type=float, default=4.0)
    sc.add_argument("--json", action="store_true")
    sc.set_defaults(fn=cmd_scan)

    up = sub.add_parser("update", help="versions and updates (check, prepare, activate need sudo)",
                        description="status: this PC's release and the newest one. check: ask the repo. "
                                    "prepare [VERSION]: build it next to the running one. activate [VERSION]: "
                                    "switch to it (the master's dashboard does this for the whole cluster). "
                                    "rollback: back to the previous release. auto [on|off]: the daily timer's "
                                    "run, or switch automatic updates on or off.")
    up.add_argument("what", nargs="?", default="status",
                    choices=("status", "check", "prepare", "activate", "rollback", "auto", "apply-request"))
    up.add_argument("version", nargs="?", help="release (e.g. 1.2.0); for auto: on or off")
    up.add_argument("--json", action="store_true")
    up.set_defaults(fn=cmd_update)

    args = ap.parse_args()
    if getattr(args, "extra", None) and args.extra[:1] == ["--"]:
        args.extra = args.extra[1:]
    args.fn(args)


if __name__ == "__main__":
    main()
