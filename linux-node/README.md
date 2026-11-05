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
| AMD | ROCm (`hipcc`, `rocminfo`) | `./build.sh --backends rocm` (Ziel-Arch wird per `rocminfo` erkannt, sonst `--rocm-targets gfx1100`) |
| Alle (inkl. Intel Arc/iGPU) | `libvulkan-dev glslc` | `./build.sh --backends vulkan` |
| Intel (beste Leistung) | Intel oneAPI Base Toolkit | `./build.sh --backends sycl` |
| ohne GPU | – | `./build.sh --backends cpu` |

Ohne `--backends` erkennt `build.sh` die vorhandenen Toolkits selbst. Mehrere Backends lassen sich kombinieren (`--backends cuda,vulkan`): Sie werden als Module gebaut und zur Laufzeit geladen, ein fehlender Treiber verhindert also nicht den Start. CUDA-Architekturen bei Bedarf über `CUDA_ARCHS="86"` (z. B. RTX 30xx/A4000) einschränken, das spart viel Build-Zeit.

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

Misst beim ersten Start die Bandbreite (gecacht), startet `ggml-rpc-server` auf Port 50052, kündigt sich per mDNS an und zeigt einen QR-Code. In der App: *Phone cluster → Master → Find nearby* oder *Scan a worker's QR code*. `--cache` legt Modell-Tensoren auf der Platte ab, wiederholte Ladevorgänge gehen dann deutlich schneller. `--devices CUDA0,CUDA1` wählt bestimmte GPUs.

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
