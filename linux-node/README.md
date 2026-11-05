# PrivateLM Linux Node

Bringt Linux-PCs mit GPU (NVIDIA, AMD, Intel) in den PrivateLM-Cluster. Ein PC kann:

| Rolle | Befehl | Was passiert |
|---|---|---|
| **GPU-Worker** | `privatelm-node worker` | Der PC leiht seine GPU(s) einem Handy-Master. Er taucht in der App unter *Phone cluster → Find nearby* auf wie ein Handy, bekommt aber wegen der viel höheren Speicherbandbreite den größten Anteil der Layer. |
| **Master / Server** | `privatelm-node serve -m model.gguf` | Das Modell läuft auf dem PC (GPU + RAM), optional mit Handys oder weiteren PCs als Workern. Bereitgestellt als OpenAI-kompatible API; die App verbindet sich per QR-Code oder *Find PCs*. |
| **Diagnose** | `privatelm-node detect` / `scan` | Geräte, freier Speicher, gemessene Bandbreite / alle Worker und Server im LAN. |

Technisch nutzt alles das RPC-Backend von llama.cpp, das die App bereits für den Handy-Cluster verwendet. Der Node wird aus **exakt demselben llama.cpp-Commit** gebaut wie die App (`LLAMA_CPP_COMMIT`, plus die beiden Logging-Patches der App unter `patches/`). Das ist Pflicht: RPC-Protokoll und Op-Liste müssen bitgenau passen, sonst bricht llama.cpp beim Verbinden ab. Beim Update der App auf eine neue llama.cpp-Version muss `LLAMA_CPP_COMMIT` mitziehen (Protokollkennung `rpc7` in der App und in `privatelm_node.py`).

## Bauen

Voraussetzungen: `cmake`, `g++`, `git`, Python 3.9+, plus das Toolkit der jeweiligen GPU.

| GPU | Pakete (Debian/Ubuntu) | Build |
|---|---|---|
| NVIDIA | CUDA Toolkit (`nvcc`) | `./build.sh --backends cuda` |
| AMD (alle, auch ältere 4-GB-Karten wie RX 470/570/580) | `libvulkan-dev glslc mesa-vulkan-drivers` | `./build.sh --backends vulkan` |
| AMD, neuere Karten (RDNA2/3, Vega) | ROCm (`hipcc`, `rocminfo`) | `./build.sh --backends rocm` |
| Intel (Arc, iGPU) | `libvulkan-dev glslc mesa-vulkan-drivers` | `./build.sh --backends vulkan` (oder `sycl` mit oneAPI) |
| NVIDIA + AMD im selben PC | CUDA + Vulkan-Pakete | `./build.sh --backends cuda,vulkan` |
| ohne GPU | – | `./build.sh --backends cpu` |

Ohne `--backends` erkennt `build.sh` die GPUs selbst: NVIDIA mit installiertem CUDA-Toolkit → CUDA, sonst Vulkan; AMD und Intel → Vulkan. ROCm unterstützt ältere AMD-Karten (Polaris, gfx803) nicht mehr, Vulkan läuft auf allen und liegt bei der Textgenerierung nahe an ROCm. Mehrere Backends werden als Module gebaut und zur Laufzeit geladen, ein fehlender Treiber verhindert also nicht den Start. Sieht eine NVIDIA-Karte sowohl CUDA als auch Vulkan, wird sie trotzdem nur einmal genutzt (Abgleich über die PCI-Adresse, wie in llama.cpp). Die CUDA-Architekturen der eingebauten Karten liest `build.sh` aus `nvidia-smi` (z. B. 86 für A2000/3070 Ti), das spart viel Build-Zeit.

```bash
cd linux-node
./build.sh --backends cuda --install /opt/privatelm-node
sudo apt install python3-zeroconf          # mDNS (alternativ: avahi-utils)
pip install qrcode                          # optional: QR-Code im Terminal
```

## Nutzung

**PC als GPU-Worker für ein Handy:**

```bash
/opt/privatelm-node/bin/privatelm-node worker --cache
```

Misst beim ersten Start die Bandbreite (gecacht), startet `ggml-rpc-server` auf Port 50052, kündigt sich per mDNS an und zeigt einen QR-Code. In der App: *Phone cluster → Master → Find nearby* oder *Scan a worker's QR code*. `--cache` legt Modell-Tensoren auf der Platte ab, wiederholte Ladevorgänge gehen dann deutlich schneller. `--devices CUDA0,Vulkan1` wählt bestimmte GPUs. Mehrere GPUs in einem PC werden nach ihrem freien Speicher belastet, eine 4-GB-Karte neben einer 12-GB-Karte bekommt also ein Drittel.

**RAM mitverleihen:**

```bash
privatelm-node worker --cache --ram                     # GPU auf 50052, RAM auf 50053
privatelm-node worker --cache --ram --ram-limit 24      # höchstens 24 GB RAM verleihen
```

Der RAM erscheint als zweiter, eigener Worker („<hostname> RAM“). So bewertet der Master ihn mit seiner eigenen (niedrigeren) Geschwindigkeit, und er wird erst gefüllt, wenn alle GPUs voll sind. 4 GB bleiben standardmäßig für den PC selbst frei (`--ram-reserve`).

**Mehrere PCs mit kleinen GPUs zu einem großen Modell:** Auf jedem PC `privatelm-node worker --cache --ram`, auf dem PC mit der größten GPU `privatelm-node serve -m modell.gguf --workers all`. Der Planer füllt zuerst die schnellsten GPUs, dann die übrigen GPUs, dann RAM.

**PC als Server, Handys als Zusatzspeicher:**

```bash
privatelm-node serve -m ~/models/qwen3-32b-q4_k_m.gguf            # Handys nur, wenn es lokal nicht passt
privatelm-node serve -m model.gguf --workers all                   # alle gefundenen Worker einbeziehen
privatelm-node serve -m model.gguf --workers 192.168.1.20:50052    # bestimmte Worker
privatelm-node serve -m model.gguf -- --jinja -fa on               # weitere llama-server-Optionen
```

Der Planer verteilt wie die App: schnellste Geräte zuerst (GPU > PC-RAM > Handys), jeweils bis zum freien Speicher. Handys kommen bei `auto` nur dazu, wenn GPU + RAM nicht reichen, denn jedes Token läuft dann einmal über das WLAN. Der API-Key wird beim ersten Start erzeugt (`~/.config/privatelm-node/api_key`) und steckt im QR-Code. In der App: *Phone cluster → Scan PC QR code* (oder *Find PCs*, dann Key eingeben). Der Chat läuft danach als eigenes Custom-Provider-Profil „PC · <hostname>“.

**Als Dienst:** `systemd/privatelm-worker.service` bzw. `privatelm-server.service` nach `/etc/systemd/system/` kopieren, Pfade anpassen, `systemctl enable --now …`.

## Netzwerk und Sicherheit

- Das RPC-Protokoll von llama.cpp hat **keine Authentifizierung**. Worker nur im vertrauenswürdigen LAN oder über ein VPN betreiben. Mit `--host <LAN- oder VPN-IP>` lauscht der Worker nur dort; Port 50052 nicht ins Internet freigeben.
- Die API (`serve`) ist per API-Key geschützt, läuft aber unverschlüsselt über HTTP. Außerhalb des LANs nur über VPN (z. B. NetBird/WireGuard) oder hinter einem TLS-Reverse-Proxy.
- mDNS funktioniert nicht über VPNs und in Gast-WLANs mit Client-Isolation. Dann per QR-Code oder IP koppeln, mit `--advertise-ip <VPN-IP>` steht die richtige Adresse im QR-Code.
- Ein Worker bedient immer nur **einen** Master gleichzeitig.

## Grenzen

- Handy-Master gibt es nur auf Android (wie im Original); iOS-Geräte sind Worker und können sich per QR-Code mit einem PC-Server verbinden.
- Leistung im Cluster: Pro Token gibt es Netzwerk-Roundtrips zwischen den Geräten. Ein Handy-Master mit PC-GPU-Worker ist deutlich schneller als ein reiner Handy-Cluster, aber langsamer als das Modell direkt auf dem PC (`serve`). Für große Modelle ist `serve` auf dem PC mit Handy-Chat der schnellste Weg.
- Getestet in einer CPU-Umgebung: Build, Bandbreitenmessung, mDNS-Ankündigung und -Suche, Layer-Offload auf einen RPC-Worker und Chat-Completion über die API. GPU-Builds (CUDA/ROCm/Vulkan/SYCL) nutzen die Standard-Optionen von llama.cpp, sind hier aber mangels GPU nicht gelaufen.
