# PrivateLM Linux Node

Bringt Linux-PCs mit GPU (NVIDIA, AMD, Intel) in den PrivateLM-Cluster. Ein PC kann:

| Rolle | Befehl | Was passiert |
|---|---|---|
| **GPU-Worker** | `privatelm-node worker` | Der PC leiht seine GPU(s) einem Handy-Master. Er taucht in der App unter *Phone cluster → Find nearby* auf wie ein Handy, bekommt aber wegen der viel höheren Speicherbandbreite den größten Anteil der Layer. |
| **Master / Server** | `privatelm-node serve -m model.gguf` | Das Modell läuft auf dem PC (GPU + RAM), optional mit Handys oder weiteren PCs als Workern. Bereitgestellt als OpenAI-kompatible API; die App verbindet sich per QR-Code oder *Find PCs*. |
| **Diagnose** | `privatelm-node detect` / `scan` / `bench` | Geräte, freier Speicher, gemessene Bandbreite / alle Worker und Server im LAN / Token pro Sekunde und Netzwerkverkehr pro Token. |

Technisch nutzt alles das RPC-Backend von llama.cpp, das die App bereits für den Handy-Cluster verwendet. Der Node wird aus **exakt demselben llama.cpp-Commit** gebaut wie die App (`LLAMA_CPP_COMMIT`, plus die beiden Logging-Patches der App unter `patches/`). Das ist Pflicht: RPC-Protokoll und Op-Liste müssen bitgenau passen, sonst bricht llama.cpp beim Verbinden ab. Beim Update der App auf eine neue llama.cpp-Version muss `LLAMA_CPP_COMMIT` mitziehen (Protokollkennung `rpc7` in der App und in `privatelm_node.py`).

## Bauen

Voraussetzungen: `cmake`, `g++`, `git`, Python 3.9+, plus das Toolkit der jeweiligen GPU.

| GPU | Pakete (Debian/Ubuntu) | Build |
|---|---|---|
| NVIDIA | CUDA Toolkit (`nvcc`) | `./build.sh --backends cuda` |
| AMD (alle, auch ältere 4-GB-Karten wie RX 470/570/580) | `libvulkan-dev glslc spirv-headers mesa-vulkan-drivers` | `./build.sh --backends vulkan` |
| AMD, neuere Karten (RDNA2/3, Vega) | ROCm (`hipcc`, `rocminfo`) | `./build.sh --backends rocm` |
| Intel (Arc, iGPU) | `libvulkan-dev glslc spirv-headers mesa-vulkan-drivers` | `./build.sh --backends vulkan` (oder `sycl` mit oneAPI) |
| NVIDIA + AMD im selben PC | CUDA + Vulkan-Pakete | `./build.sh --backends cuda,vulkan` |
| ohne GPU | – | `./build.sh --backends cpu` |

Ohne `--backends` erkennt `build.sh` die GPUs selbst: NVIDIA mit installiertem CUDA-Toolkit → CUDA, sonst Vulkan; AMD und Intel → Vulkan. ROCm unterstützt ältere AMD-Karten (Polaris, gfx803) nicht mehr, Vulkan läuft auf allen und liegt bei der Textgenerierung nahe an ROCm. Mehrere Backends werden als Module gebaut und zur Laufzeit geladen, ein fehlender Treiber verhindert also nicht den Start. Sieht eine NVIDIA-Karte sowohl CUDA als auch Vulkan, wird sie trotzdem nur einmal genutzt (Abgleich über die PCI-Adresse, wie in llama.cpp). Die CUDA-Architekturen der eingebauten Karten liest `build.sh` aus `nvidia-smi` (z. B. 86 für A2000/3070 Ti), das spart viel Build-Zeit.

```bash
cd linux-node
./build.sh --backends cuda --install /opt/privatelm-node
sudo apt install python3-zeroconf          # mDNS (alternativ: avahi-utils)
pip install qrcode                          # optional: QR-Code im Terminal
```

## Modell wählen

```bash
privatelm-node models                  # vorhandene GGUF-Dateien und Katalog, bewertet nach Speicher
privatelm-node download qwen3-4b       # Katalogname, Nummer aus der Liste oder user/repo[:QUANT]
```

`models` sucht GGUF-Dateien in den üblichen Ordnern (`~/models`, `~/Downloads`, `/srv/models`, Hugging-Face- und LM-Studio-Cache) und zeigt einen kleinen Katalog (Qwen3 1.7B bis 32B, Qwen3.5 9B, Qwen3.8 27B, Qwen3.6 35B-A3B). Jeder Eintrag wird bewertet: passt auf diesen PC, nur mit Workern, zu groß; dazu, ob diese llama.cpp-Version die Architektur kennt. Dateinamen, Größen und Architekturen kommen zur Laufzeit von Hugging Face. `download` setzt abgebrochene Downloads fort und prüft die SHA-256-Summe; `HF_TOKEN` für gesperrte Modelle, `PRIVATELM_HF_BASE` für einen Spiegel. Das Installationsskript nutzt beides für seine Auswahl (`--model auto` wählt ohne Rückfrage das größte passende Modell).

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

So plant `serve`:

- **Schnellste Geräte zuerst.** Ein aufgeteiltes Modell läuft Gerät für Gerät nacheinander: Pro Token addieren sich die Rechenzeiten, und jeder Worker kostet eine Netzwerk-Runde. Ein langsameres Gerät bringt also Speicher, nie Tempo. Der Planer füllt deshalb die Geräte nach Speicherbandbreite und rechnet in ganzen Layern.
- **Platz für KV-Cache und Puffer.** Pro Layer zählen Gewichte plus KV-Cache für die gewählte Kontextlänge (`--ctx`), dazu 400 MB Rechenpuffer pro GPU. Eine bis zum Rand gefüllte Karte wird sonst langsamer oder lädt gar nicht.
- **Zeitmodell bei `--workers auto`:** Die Netzwerk-Latenz zu jedem Worker wird gemessen; ein Worker wird weggelassen, wenn das Modell ohne ihn passt und die geschätzte Zeit pro Token ohne ihn kürzer ist. `--workers all` nimmt jeden Worker mit Platz, füllt aber ebenfalls die schnellsten zuerst.
- **Ausgabeschicht bleibt lokal.** Sie berechnet pro Token einen Wert für jedes Wort des Vokabulars (z. B. 152.000 × 4 Byte ≈ 600 KB). Liegt sie auf einem Worker, kommt dieser Vektor bei jedem Token übers Netz. `serve` hält sie per `-ot` auf der schnellsten lokalen GPU oder im RAM (llama.cpp würde sie sonst mit den letzten Layern auf das letzte Gerät legen).
- **MoE-Modelle** (z. B. Qwen3-30B-A3B): Pro Token rechnen nur wenige Experten, Experten im RAM kosten daher wenig. Passt das Modell auf den Master, legt `serve` bei `auto` Attention, KV-Cache und so viele Experten wie möglich auf die GPU, den Rest der Experten in den RAM (`--n-cpu-moe`), ohne Worker. Gemessene Setups zeigen MoE über RPC bei der Generierung ungefähr gleichauf, der Verbund lohnt dort vor allem für lange Prompts. Mit `--workers all` werden Worker trotzdem genutzt; der Plan zeigt beide Schätzungen.
- **MTP** (Multi-Token-Prediction, Modelle mit eingebauten Draft-Layern): `--mtp N` rät N Tokens voraus und prüft sie gemeinsam. Die Draft-Layer sitzen am Ende des Modells und bleiben wie die Ausgabeschicht auf dem Master; für den zweiten Kontext werden 1,3 GB reserviert. Die beste Tiefe hängt von der Temperatur ab (bei Temperatur 1 werden weniger Vorhersagen angenommen, Tiefe 1 kann dann schneller sein als 3). `privatelm-node tune-mtp -m modell.gguf --temp 0.7` probiert die Tiefen 0 bis 3 bei Ihrer Temperatur durch; vorher den laufenden Server stoppen, die Worker bedienen nur einen Master.
- `--ram-limit GB` begrenzt den RAM, den der Master selbst für das Modell hergibt.

Der Plan wird vor dem Start mit geschätzten Token/s, Anzahl Netzwerk-Hops und erwartetem Verkehr pro Token angezeigt.

**Messen:** Auf dem Master, während `serve` läuft:

```bash
privatelm-node bench                          # Token/s und KB pro Token (Ein- und Ausgang)
privatelm-node bench --temp 1.0 --context 8000  # bei Ihrer Temperatur und mit gefülltem Kontext
```

Messen Sie mit Ihrer echten Temperatur und einem gefüllten Kontext: Ein kurzer Test mit leerem Kontext unterschätzt den Nutzen der Worker, deren Hauptbeitrag oft der Platz für den KV-Cache ist. Bei MTP zeigt `bench` die Annahmequote der Vorhersagen. Gezählt werden beide Richtungen; ein Zähler nur für den Ausgang würde die zurückkommenden Ausgabe-Vektoren übersehen. Richtwerte: ein paar bis einige zehn KB pro Token und Worker sind normal. Etwa Vokabular × 4 Byte heißt, die Ausgabeschicht liegt auf einem Worker. Megabytes heißen, die Layer sind zwischen den Rechnern verschachtelt. Der API-Key wird beim ersten Start erzeugt (`~/.config/privatelm-node/api_key`) und steckt im QR-Code. In der App: *Phone cluster → Scan PC QR code* (oder *Find PCs*, dann Key eingeben). Der Chat läuft danach als eigenes Custom-Provider-Profil „PC · <hostname>“.

**Router (zwei Modelle, ein Endpunkt):** Mit einem kleinen Zweitmodell beantwortet der Master einfache Anfragen sofort, ohne Netzwerk-Runden, und schickt alles andere an das große (verteilte) Modell:

```bash
privatelm-node serve -m gross.gguf --fast-model klein.gguf          # Router an
privatelm-node serve -m gross.gguf --fast-model klein.gguf --router off
privatelm-node serve -m gross.gguf --fast-url http://10.0.0.5:8080/v1 --fast-key KEY   # vorhandener Server als schnelles Modell
```

- Clients sehen einen Endpunkt und den Modellnamen `privatelm` (`--model-name`). Der Router läuft auf dem API-Port, die beiden llama-server lokal auf Port+1 (groß) und Port+2 (schnell).
- Das schnelle Modell läuft auf einem Gerät: `--fast-device CUDA0` oder `CPU`, sonst auf der lokalen GPU, auf die es am besten passt. Sein Speicher (Gewichte, KV-Cache für `--fast-ctx`, Puffer) wird bei der Planung des großen Modells freigehalten.
- **Entscheidung „Regeln“** (Standard): an das große Modell gehen Anfragen mit Werkzeugen, Bildern, langem Kontext, langer gewünschter Antwort, strukturierter Ausgabe, langen Gesprächen, langen Fragen und Begriffen wie Code, Skript, Analyse, Vergleich, Berechnung, Docker, SQL. Alles andere geht an das schnelle. **„Schnelles Modell“**: Das kleine Modell bewertet jede Anfrage mit einem Token (einfach/schwer); die festen Regeln (Werkzeuge, Bilder, langer Kontext) gelten weiter.
- Gezielt ansprechen: Modellname `fast` oder `big`, oder Header `X-PrivateLM-Route: fast|big`. Jede Antwort trägt `X-PrivateLM-Route` und `X-PrivateLM-Reason`.
- **Ein- und ausschalten:** im Dashboard (Schalter und Wahl der Entscheidungsart, wirkt sofort und bleibt über Neustarts gespeichert) oder per `--router on|off`; `--router auto` (Standard) nimmt die letzte Einstellung aus dem Dashboard. Aus heißt: alles an das große Modell.
- `bench` misst mit Router standardmäßig das große Modell (`--route fast` für das schnelle).

**Dashboard:** `serve` startet auf Port 8090 eine Weboberfläche (`http://<Master-IP>:8090/`, Anmeldung mit dem API-Key). Sie zeigt die Modell-Aufteilung Layer für Layer in Ausführungsreihenfolge, Live-Token/s, Netzwerkverkehr pro Token, den Verlauf der letzten zehn Minuten und für Master und jeden Worker CPU, RAM, Netz sowie je GPU Auslastung, VRAM, Temperatur und Leistung (NVIDIA über `nvidia-smi`, AMD über die Treiberwerte). Die Worker liefern ihre Messwerte über einen kleinen Nur-Lese-Endpunkt auf Port 50060, den nur der Master erreichen muss. Ein Knopf startet einen Kurztest. Optionen: `--dashboard-port`, `--no-dashboard` (serve), `--status-port`, `--no-status` (worker).

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
