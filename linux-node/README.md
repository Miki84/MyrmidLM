# PrivateLM Linux Node

Bringt Linux-PCs mit GPU (NVIDIA, AMD, Intel) in den PrivateLM-Cluster. Ein PC kann:

| Rolle | Befehl | Was passiert |
|---|---|---|
| **GPU-Worker** | `privatelm-node worker` | Der PC leiht seine GPU(s) einem Handy-Master. Er taucht in der App unter *Phone cluster → Find nearby* auf wie ein Handy, bekommt aber wegen der viel höheren Speicherbandbreite den größten Anteil der Layer. |
| **Master / Server** | `privatelm-node serve -m model.gguf` | Das Modell läuft auf dem PC (GPU + RAM), optional mit Handys oder weiteren PCs als Workern. Bereitgestellt als OpenAI-kompatible API; die App verbindet sich per QR-Code oder *Find PCs*. |
| **Diagnose** | `privatelm-node detect` / `scan` / `bench` | Geräte, freier Speicher, gemessene Bandbreite / alle Worker und Server im LAN / Token pro Sekunde und Netzwerkverkehr pro Token. |

Technisch nutzt alles das RPC-Backend von llama.cpp, das die App bereits für den Handy-Cluster verwendet. Der Node wird aus **exakt demselben llama.cpp-Commit** gebaut wie die App (derzeit b12233 vom 9.11.2026, RPC-Protokoll 9; ältere Worker und App-Versionen mit RPC 7 erscheinen als „andere Version“ und werden nicht genutzt) (`LLAMA_CPP_COMMIT`, plus die beiden Logging-Patches der App unter `patches/`). Das ist Pflicht: RPC-Protokoll und Op-Liste müssen bitgenau passen, sonst bricht llama.cpp beim Verbinden ab. Beim Update der App auf eine neue llama.cpp-Version muss `LLAMA_CPP_COMMIT` mitziehen (Protokollkennung `rpc9` in der App und in `privatelm_node.py`).

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
- **Handys** messen ihre Speicherbandbreite mit einer Kopierschleife. llama.cpp erreicht auf einem Handy-Prozessor wegen Rechenleistung und Drosselung etwa die Hälfte davon; der Planer rechnet deshalb mit der Hälfte. Sonst sähe ein Handy schneller aus als ein Laptop und bekäme das ganze Modell.
- **Kontext:** Ohne `--ctx` startet `serve` mit 8.192 Tokens und halbiert bis 2.048, wenn das Modell sonst nicht passt.

**Neuverteilung (`--rebalance`, Standard an):** `serve` schaut alle 30 Sekunden ins Netz und plant neu, sobald keine Anfrage läuft,

- wenn ein neuer Worker das Modell geschätzt um mindestens 15 % schneller macht oder einen längeren Kontext erlaubt (ohne `--ctx`),
- wenn ein Worker wegfällt (llama-server endet dann, oder der Worker ist dreimal hintereinander nicht zu sehen),
- und wenn das Modell nicht passt: Dann wartet `serve` auf Worker, statt sich zu beenden, und startet von selbst, sobald genug da sind.

Neu planen heißt neu laden. Worker ohne Tensor-Cache (Handys) bekommen ihre Layer erneut übers Netz; über WLAN dauert das einige Minuten, PC-Worker mit `--cache` laden schnell. Zwischen zwei Neuverteilungen für mehr Tempo liegen mindestens 5 Minuten; ein Worker, der nichts bringt, wird nur einmal geprüft. Die Schätzung nutzt den Speicher, den jedes Gerät vor dem Start frei hatte, daher stört die Prüfung den laufenden Server nicht. Im Dashboard: Schalter, Knopf „Jetzt neu verteilen“ und die letzten Ereignisse. `--rebalance on|off` setzt es beim Start, `auto` (Standard) nimmt die letzte Einstellung aus dem Dashboard. Mit `off` wird einmal geplant, und ein Modell, das nicht passt, beendet den Dienst mit Exit-Code 2 (die systemd-Units starten ihn dann nicht endlos neu).

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

- Clients sehen einen Endpunkt und den Modellnamen `privatelm` (`--model-name`), dazu jedes Modell unter seinem eigenen Namen. Der Router läuft immer auf dem API-Port, auch ohne Zweitmodell (Hosts können später dazukommen); die llama-server laufen lokal auf Port+1 (groß) und Port+2 (schnell).
- Das schnelle Modell läuft auf einem Gerät: `--fast-device CUDA0` oder `CPU`, sonst auf der lokalen GPU, auf die es am besten passt. Sein Speicher (Gewichte, KV-Cache für `--fast-ctx`, Puffer) wird bei der Planung des großen Modells freigehalten.
- **Entscheidung „Regeln“** (Standard): an das große Modell gehen Anfragen mit Werkzeugen, Bildern, langem Kontext, langer gewünschter Antwort, strukturierter Ausgabe, langen Gesprächen, langen Fragen und Begriffen wie Code, Skript, Analyse, Vergleich, Berechnung, Docker, SQL. Alles andere geht an das schnelle. **„Schnelles Modell“**: Das kleine Modell bewertet jede Anfrage mit einem Token (einfach/schwer); die festen Regeln (Werkzeuge, Bilder, langer Kontext) gelten weiter.
- Gezielt ansprechen: Modellname `fast` oder `big`, oder Header `X-PrivateLM-Route: fast|big`. Jede Antwort trägt `X-PrivateLM-Route` und `X-PrivateLM-Reason`.
- **Ein- und ausschalten:** im Dashboard (Schalter und Wahl der Entscheidungsart, wirkt sofort und bleibt über Neustarts gespeichert) oder per `--router on|off`; `--router auto` (Standard) nimmt die letzte Einstellung aus dem Dashboard. Aus heißt: alles an das große Modell.
- `bench` misst mit Router standardmäßig das große Modell (`--route fast` für das schnelle).

**Hosts (mehrere ganze Modelle, ein Endpunkt):** Ein Host ist ein PC, der ein ganzes Modell auf seine eigene GPU lädt, etwa drei PCs mit je einer A4000. Der Router des Masters verteilt die Anfragen auf sie. Anders als Worker (ein Modell über mehrere Rechner verteilt) rechnet jeder Host für sich, ohne Netzwerk-Runden pro Token.

```bash
# auf jedem Host (der Schlüssel ist der API-Key des Masters: dort privatelm-node key)
privatelm-node host --key-file /etc/privatelm-node/cluster_key
# auf dem Master: welcher Host was lädt (oder im Dashboard, Bereich Hosts)
privatelm-node serve -m gross.gguf --assign "pc2=qwen3-32b,name=qwen32" --assign "pc3=qwen3-32b,name=qwen32" \
                     --assign "pc4=bartowski/Qwen2.5-Coder-14B-Instruct-GGUF:Q4_K_M,name=coder,role=extra"
```

- **Zuweisen:** im Dashboard (Modell, Name, Rolle, Kontext, „Laden“/„Entladen“) oder mit `--assign HOST=MODELL[,role=big|fast|extra][,name=NAME][,ctx=N]`. HOST ist Rechnername, IP oder IP:Port; MODELL ein Katalogname, `user/repo:QUANT` oder eine Datei auf dem Host. Der Host lädt das Modell selbst herunter (fortsetzbar, SHA-256 geprüft) nach `--models-dir`, plant es auf seinen GPUs und startet llama-server auf Port 8081. Passt es nicht, meldet er das, bevor er herunterlädt. Die Zuweisung merken sich Master (`hosts.json`) und Host; nach einem Neustart lädt der Host sein Modell wieder, auch ohne Master.
- **Verteilen:** Hosts mit demselben Namen bilden eine Gruppe; jede Anfrage geht an das Mitglied mit den wenigsten laufenden Anfragen. Clients wählen per Modellname (`"model": "qwen32"`, `/v1/models` listet alle) oder schicken `privatelm`: Dann entscheiden die Regeln zwischen den Rollen „schnell“ und „groß“; „nur per Name“ (`extra`) bekommt nur Anfragen, die es beim Namen nennen. Das eigene Modell des Masters ist ein Mitglied der Rolle „groß“. Jede Antwort trägt `X-PrivateLM-Backend: name@rechner`.
- **Sicherheit:** Die Steuerung der Hosts (Port 50070) und ihre Modell-API (8081) verlangen den API-Key des Masters; wer ihn hat, kann Modelle laden lassen. Das Installationsskript (Rolle 3) legt den Schlüssel root-lesbar in `/etc/privatelm-node/cluster_key` ab und öffnet beide Ports nur für den Master.
- Hosts brauchen eigenen Speicher für ihr Modell; Worker brauchen kein Modell (der Master schickt ihnen ihre Layer).

**MoE-Expertencache (llama.cpp ab b12191):** Passen bei einem MoE-Modell nicht alle Experten auf die GPU(s), legt der Planer alle Experten in den RAM (`--cpu-moe`) und nutzt den übrigen GPU-Speicher als Cache für die meistgenutzten (`--moe-cache-mib`). Das schlägt die feste Aufteilung mit `--n-cpu-moe`, weil Experten ungleich oft gebraucht werden (llama.cpp misst 72 bis 89 % Treffer). Fehlgriffe rechnet zum Teil die CPU direkt aus dem RAM, statt die Experten über PCIe hochzuladen (`--moe-cache-cpu-share`, Standard 0,5). `privatelm-node tune-moe -m MODELL` misst mit llama.cpps `moe-cache-tune.py` Cache-Größe und CPU-Anteil für diesen PC (Server vorher stoppen, dauert einige Minuten) und speichert sie; `serve` nutzt sie danach. `--moe-cache off` schaltet zurück auf die feste Aufteilung, `--moe-cache MIB` begrenzt den Cache. Gilt für MoE-Modelle, die auf einem PC laufen (Master ohne Worker, Hosts).

**Planer lernt aus Messungen:** Die Geschwindigkeit eines Geräts schätzt der Planer aus seiner Speicherbandbreite. Jede Anfrage an das Modell des Masters, die allein läuft, liefert eine Messung (llama-servers `timings`; mit Entwurfsmodell oder MTP pro geprüftem Durchlauf). Pro Plan ergibt das eine Gleichung mit einem unbekannten Faktor je Gerät; mehrere Pläne (Neuverteilung erzeugt sie) legen die Faktoren fest, eine Ridge-Regression hält ein Gerät bei seiner Schätzung, bis Messungen etwas anderes sagen. Der Planer rechnet danach mit den gemessenen Geschwindigkeiten (`calibration.json`). Sagt die Messung, dass eine andere Aufteilung mindestens 15 % schneller wäre, verteilt `serve` neu (mit Neuverteilung an); ein schon gemessener Plan zählt mit seinem Messwert. `privatelm-node calibration` zeigt die Faktoren, `--reset` vergisst sie. Das Dashboard zeigt die Messung neben der Schätzung und pro Rechner den Faktor.

**Entwurfsmodell (`--draft-model auto|DATEI`):** Ein kleines Modell derselben Familie (gleicher Tokenizer, z. B. Qwen3-0.6B zu Qwen3-32B) schreibt bis zu `--draft-max` (8) Tokens vor, das große prüft sie in einem Durchlauf. Gleiche Ausgabe, oft 1,5- bis 2-mal so viele Tokens pro Durchlauf; am meisten, wenn das Modell über das Netz verteilt ist, denn jeder Durchlauf kostet Netzwerk-Runden. `auto` nimmt eine passende kleine GGUF-Datei auf diesem PC (höchstens 12 % des großen Modells) oder lädt eine aus dem Katalog (Qwen3.5-0.8B, Qwen3-0.6B, Llama-3.2-1B). Den Tokenizer prüft `serve` vorher wie llama.cpp; passt er nicht, läuft es ohne. Das Entwurfsmodell läuft auf der lokalen GPU mit dem meisten Platz (sonst auf der CPU), sein Speicher geht vom Plan des großen ab. Mit `--mtp` wird es nicht genutzt. `bench` zeigt die Trefferquote.

**Schnellste Verbindung:** Hat ein PC Kabel und WLAN, meldet er sich mit der Adresse seiner schnellsten Verbindung ins LAN: Kabel vor WLAN, das schnellere Kabel zuerst (nur Verbindungen ins selbe Netz wie die Standardroute; ein zweites Kabel zum NAS zählt nicht). Alle 20 Sekunden prüft er das; wird ein Kabel eingesteckt oder gezogen, zieht seine mDNS-Ankündigung auf die neue Adresse um. Der Master merkt, dass ein Worker im Plan jetzt unter einer anderen Adresse erreichbar ist, und plant um, sobald keine Anfrage läuft (die Layer werden dabei neu übertragen, über Kabel schnell). Hosts übernimmt der Router ebenso. Feste Adressen (`--advertise-ip`, `--host`) bleiben fest. Firewall: Das Installationsskript findet den Master im Netz und gibt die Worker-Ports für alle seine Adressen frei (Kabel und WLAN); bekommt der Master später eine neue Verbindung, das Skript auf den Workern noch einmal laufen lassen (oder `ufw allow from NEUE-IP`).

**Wake-on-LAN und Schlafen:** Worker und Hosts melden ihre MAC-Adresse (nur Kabel); der Master merkt sich jeden Rechner (`known.json`) und weckt ihn per Magic Packet,
- einen Worker, wenn das Modell ohne ihn nicht passt (der Dienst wartet dann auf Worker),
- einen Host, wenn eine Anfrage sein Modell braucht: Die Anfrage wartet, bis er wach ist und sein Modell geladen hat (bis 240 s),
- und von Hand im Dashboard („Aufwecken“, auch für ausgeschaltete Worker in „Weitere Geräte im Netzwerk“).

Hosts legen sich nach `--host-sleep MIN` Minuten ohne Anfrage schlafen (pro Host im Dashboard einstellbar, 0 = nie): Der Master sagt es dem Host, der beendet llama-server und geht in Standby, über `privatelm-suspend.path`/`.service` (kleine Root-Units, das Installationsskript richtet sie ein). Nach dem Aufwachen lädt der Host sein Modell von selbst wieder. Worker im Plan bleiben wach. Das Installationsskript schaltet Wake-on-LAN auf der Netzwerkkarte ein (ethtool, dauerhaft über NetworkManager oder `privatelm-wol.service`); im BIOS muss es ebenfalls an sein.

**Dashboard:** `serve` startet auf Port 8090 eine Weboberfläche (`http://<Master-IP>:8090/`, Anmeldung mit dem API-Key). Sie zeigt die Modell-Aufteilung Layer für Layer in Ausführungsreihenfolge, Live-Token/s, Netzwerkverkehr pro Token, den Verlauf der letzten zehn Minuten und für Master und jeden Worker CPU, RAM, Netz sowie je GPU Auslastung, VRAM, Temperatur und Leistung (NVIDIA über `nvidia-smi`, AMD über die Treiberwerte). Die Worker liefern ihre Messwerte über einen kleinen Nur-Lese-Endpunkt auf Port 50060, den nur der Master erreichen muss. Handys haben keinen Status-Endpunkt: Ihre Karte zeigt, ob der Master mit ihnen verbunden ist und ob sie im Netz zu sehen sind. Ein Knopf startet einen Kurztest. Optionen: `--dashboard-port`, `--no-dashboard` (serve), `--status-port`, `--no-status` (worker).

**Als Dienst:** `systemd/privatelm-worker.service`, `privatelm-host.service` bzw. `privatelm-server.service` nach `/etc/systemd/system/` kopieren, Pfade anpassen, `systemctl enable --now …`.

## Netzwerk und Sicherheit

- Das RPC-Protokoll von llama.cpp hat **keine Authentifizierung**. Worker nur im vertrauenswürdigen LAN oder über ein VPN betreiben. Mit `--host <LAN- oder VPN-IP>` lauscht der Worker nur dort; Port 50052 nicht ins Internet freigeben.
- Die API (`serve`) ist per API-Key geschützt, läuft aber unverschlüsselt über HTTP. Außerhalb des LANs nur über VPN (z. B. NetBird/WireGuard) oder hinter einem TLS-Reverse-Proxy.
- mDNS funktioniert nicht über VPNs und in Gast-WLANs mit Client-Isolation. Dann per QR-Code oder IP koppeln, mit `--advertise-ip <VPN-IP>` steht die richtige Adresse im QR-Code.
- Ein Worker bedient immer nur **einen** Master gleichzeitig.

## Grenzen

- Handy-Master gibt es nur auf Android (wie im Original); iOS-Geräte sind Worker und können sich per QR-Code mit einem PC-Server verbinden.
- Leistung im Cluster: Pro Token gibt es Netzwerk-Roundtrips zwischen den Geräten. Ein Handy-Master mit PC-GPU-Worker ist deutlich schneller als ein reiner Handy-Cluster, aber langsamer als das Modell direkt auf dem PC (`serve`). Für große Modelle ist `serve` auf dem PC mit Handy-Chat der schnellste Weg.
- Getestet in einer CPU-Umgebung: Build, Bandbreitenmessung, mDNS-Ankündigung und -Suche, Layer-Offload auf einen RPC-Worker und Chat-Completion über die API. GPU-Builds (CUDA/ROCm/Vulkan/SYCL) nutzen die Standard-Optionen von llama.cpp, sind hier aber mangels GPU nicht gelaufen.
