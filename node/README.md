# MyrmidLM Node

Bringt Linux-PCs mit GPU (NVIDIA, AMD, Intel) in den MyrmidLM-Cluster. Installation und Updates: siehe [README im Hauptordner](../README.md) (`install.sh`). Ein PC kann:

| Rolle | Befehl | Was passiert |
|---|---|---|
| **GPU-Worker** | `myrmid worker` | Der PC leiht seine GPU(s) einem Handy-Master. Er taucht in der App unter *Phone cluster → Find nearby* auf wie ein Handy, bekommt aber wegen der viel höheren Speicherbandbreite den größten Anteil der Layer. |
| **Master / Server** | `myrmid serve -m model.gguf` | Das Modell läuft auf dem PC (GPU + RAM), optional mit Handys oder weiteren PCs als Workern. Bereitgestellt als OpenAI-kompatible API; die App verbindet sich per QR-Code oder *Find PCs*. |
| **Diagnose** | `myrmid detect` / `scan` / `bench` | Geräte, freier Speicher, gemessene Bandbreite / alle Worker und Server im LAN / Token pro Sekunde und Netzwerkverkehr pro Token. |

Technisch nutzt alles das RPC-Backend von llama.cpp, das die App bereits für den Handy-Cluster verwendet. Der Node wird aus **exakt demselben llama.cpp-Commit** gebaut wie die App (derzeit b12233 vom 9.11.2026, RPC-Protokoll 9; ältere Worker und App-Versionen mit RPC 7 erscheinen als „andere Version“ und werden nicht genutzt) (`LLAMA_CPP_COMMIT`, plus die beiden Logging-Patches der App unter `patches/`). Das ist Pflicht: RPC-Protokoll und Op-Liste müssen bitgenau passen, sonst bricht llama.cpp beim Verbinden ab. Beim Update der App auf eine neue llama.cpp-Version muss `LLAMA_CPP_COMMIT` mitziehen (Protokollkennung `rpc9` in der App und in `myrmid.py`).

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

Normalerweise erledigt das `install.sh` (Treiber, CUDA, Bauen, Dienste). Von Hand:

```bash
cd node
./build.sh --backends cuda --install /opt/myrmid/releases/dev
sudo apt install python3-zeroconf          # mDNS (alternativ: avahi-utils)
pip install qrcode                          # optional: QR-Code im Terminal
```

## Modell wählen

```bash
myrmid models                  # vorhandene GGUF-Dateien und Katalog, bewertet nach Speicher
myrmid download qwen3-4b       # Katalogname, Nummer aus der Liste oder user/repo[:QUANT]
```

`models` sucht GGUF-Dateien in den üblichen Ordnern (`~/models`, `~/Downloads`, `/srv/models`, Hugging-Face- und LM-Studio-Cache) und zeigt einen kleinen Katalog (Qwen3 1.7B bis 32B, Qwen3.5 9B, Qwen3.8 27B, Qwen3.6 35B-A3B). Jeder Eintrag wird bewertet: passt auf diesen PC, nur mit Workern, zu groß; dazu, ob diese llama.cpp-Version die Architektur kennt. Dateinamen, Größen und Architekturen kommen zur Laufzeit von Hugging Face. `download` setzt abgebrochene Downloads fort und prüft die SHA-256-Summe; `HF_TOKEN` für gesperrte Modelle, `MYRMID_HF_BASE` für einen Spiegel. Das Installationsskript nutzt beides für seine Auswahl (`--model auto` wählt ohne Rückfrage das größte passende Modell).

## Nutzung

**PC als GPU-Worker für ein Handy:**

```bash
myrmid worker --cache
```

Misst beim ersten Start die Bandbreite (gecacht), startet `ggml-rpc-server` auf Port 50052, kündigt sich per mDNS an und zeigt einen QR-Code. In der App: *Phone cluster → Master → Find nearby* oder *Scan a worker's QR code*. `--cache` legt Modell-Tensoren auf der Platte ab, wiederholte Ladevorgänge gehen dann deutlich schneller. `--devices CUDA0,Vulkan1` wählt bestimmte GPUs. Mehrere GPUs in einem PC werden nach ihrem freien Speicher belastet, eine 4-GB-Karte neben einer 12-GB-Karte bekommt also ein Drittel.

**RAM mitverleihen:**

```bash
myrmid worker --cache --ram                     # GPU auf 50052, RAM auf 50053
myrmid worker --cache --ram --ram-limit 24      # höchstens 24 GB RAM verleihen
```

Der RAM erscheint als zweiter, eigener Worker („<hostname> RAM“). So bewertet der Master ihn mit seiner eigenen (niedrigeren) Geschwindigkeit, und er wird erst gefüllt, wenn alle GPUs voll sind. 4 GB bleiben standardmäßig für den PC selbst frei (`--ram-reserve`).

**Mehrere PCs mit kleinen GPUs zu einem großen Modell:** Auf jedem PC `myrmid worker --cache --ram`, auf dem PC mit der größten GPU `myrmid serve -m modell.gguf --workers all`. Der Planer füllt zuerst die schnellsten GPUs, dann die übrigen GPUs, dann RAM.

**PC als Server, Handys als Zusatzspeicher:**

```bash
myrmid serve -m ~/models/qwen3-32b-q4_k_m.gguf            # Handys nur, wenn es lokal nicht passt
myrmid serve -m model.gguf --workers all                   # alle gefundenen Worker einbeziehen
myrmid serve -m model.gguf --workers 192.168.1.20:50052    # bestimmte Worker
myrmid serve -m model.gguf -- --jinja -fa on               # weitere llama-server-Optionen
```

So plant `serve`:

- **Schnellste Geräte zuerst.** Ein aufgeteiltes Modell läuft Gerät für Gerät nacheinander: Pro Token addieren sich die Rechenzeiten, und jeder Worker kostet eine Netzwerk-Runde. Ein langsameres Gerät bringt also Speicher, nie Tempo. Der Planer füllt deshalb die Geräte nach Speicherbandbreite und rechnet in ganzen Layern.
- **Platz für KV-Cache und Puffer.** Pro Layer zählen Gewichte plus KV-Cache für die gewählte Kontextlänge (`--ctx`), dazu 400 MB Rechenpuffer pro GPU. Eine bis zum Rand gefüllte Karte wird sonst langsamer oder lädt gar nicht.
- **Zeitmodell bei `--workers auto`:** Die Netzwerk-Latenz zu jedem Worker wird gemessen; ein Worker wird weggelassen, wenn das Modell ohne ihn passt und die geschätzte Zeit pro Token ohne ihn kürzer ist. `--workers all` nimmt jeden Worker mit Platz, füllt aber ebenfalls die schnellsten zuerst.
- **Ausgabeschicht bleibt lokal.** Sie berechnet pro Token einen Wert für jedes Wort des Vokabulars (z. B. 152.000 × 4 Byte ≈ 600 KB). Liegt sie auf einem Worker, kommt dieser Vektor bei jedem Token übers Netz. `serve` hält sie per `-ot` auf der schnellsten lokalen GPU oder im RAM (llama.cpp würde sie sonst mit den letzten Layern auf das letzte Gerät legen).
- **MoE-Modelle** (z. B. Qwen3-30B-A3B): Pro Token rechnen nur wenige Experten, Experten im RAM kosten daher wenig. Passt das Modell auf den Master, legt `serve` bei `auto` Attention, KV-Cache und so viele Experten wie möglich auf die GPU, den Rest der Experten in den RAM (`--n-cpu-moe`), ohne Worker. Gemessene Setups zeigen MoE über RPC bei der Generierung ungefähr gleichauf, der Verbund lohnt dort vor allem für lange Prompts. Mit `--workers all` werden Worker trotzdem genutzt; der Plan zeigt beide Schätzungen.
- **MTP** (Multi-Token-Prediction, Modelle mit eingebauten Draft-Layern): `--mtp N` rät N Tokens voraus und prüft sie gemeinsam. Die Draft-Layer sitzen am Ende des Modells und bleiben wie die Ausgabeschicht auf dem Master; für den zweiten Kontext werden 1,3 GB reserviert. Die beste Tiefe hängt von der Temperatur ab (bei Temperatur 1 werden weniger Vorhersagen angenommen, Tiefe 1 kann dann schneller sein als 3). `myrmid tune-mtp -m modell.gguf --temp 0.7` probiert die Tiefen 0 bis 3 bei Ihrer Temperatur durch; vorher den laufenden Server stoppen, die Worker bedienen nur einen Master.
- `--ram-limit GB` begrenzt den RAM, den der Master selbst für das Modell hergibt.
- **Handys** messen ihre Speicherbandbreite mit einer Kopierschleife. llama.cpp erreicht auf einem Handy-Prozessor wegen Rechenleistung und Drosselung etwa die Hälfte davon; der Planer rechnet deshalb mit der Hälfte. Sonst sähe ein Handy schneller aus als ein Laptop und bekäme das ganze Modell.
- **Kontext:** Ohne `--ctx` startet `serve` mit 8.192 Tokens und halbiert bis 2.048, wenn das Modell sonst nicht passt. Im Dashboard („Modell und Kontext“) lässt sich das Maximum anheben (16.384 bis 131.072, höchstens was das Modell kann); die Wahl gilt vor `--ctx` und wird in `ctx.json` gespeichert.
- **KV-Cache:** Standard f16. Mit q8_0 (Dashboard oder `--kv-cache q8_0`) braucht der Kontext halb so viel Speicher bei kaum messbarem Qualitätsverlust, q4_0 ein Viertel, aber ungenauer. Der Planer rechnet mit dem gewählten Format, Flash Attention wird dafür eingeschaltet. Gilt für das große und (ab dem nächsten Neustart) das schnelle Modell; Hosts haben die Option in ihrer Karte.
- **Autotune:** Im Dashboard („Modell und Kontext“ für den Master, Knopf in jeder Host-Karte) oder mit `myrmid tune -m MODELL` (Dienst vorher stoppen). Startet das geplante Modell mehrmals mit anderen Einstellungen (Batch-Größe `-ub`, CPU-Threads, bei MoE der CPU-Anteil am Experten-Cache), misst je einen langen Prompt plus Antwort und behält die schnellste Variante. Gespeichert in `tune.json` pro Modell und GPU-Slot; jeder spätere Start nutzt sie. Läuft nur für Modelle, die auf einem PC laufen (mit Workern entscheidet das Netz).
- **Qwen3.8-Flash-Next (125B MoE):** Im Katalog als `qwen3.8-flash-next` (UD-IQ3_XXS, ≈ 82 GB). Gedacht für einen Host mit GPU und ≥ 96 GB RAM: Attention und KV-Cache auf der GPU, Experten über den MoE-Cache in RAM und VRAM, die n-gram-Tabelle bleibt im RAM (bei Knappheit liest das System sie von der SSD). Danach Autotune laufen lassen. Geteilte GGUF-Dateien (`-00001-of-0000N`) plant MyrmidLM jetzt über alle Teile.
- **Nur Router (Master ohne eigenes Modell):** `myrmid serve -m none` oder im Dashboard unter „Modell und Kontext“ den Schalter „Eigenes Modell auf diesem PC“ ausschalten (gespeichert in `master.json`, gilt vor `-m`). Der Master übernimmt dann nur Dashboard, Router, Kopplung, Updates und die Steuerung der Hosts; alle Modelle laufen auf Hosts oder externen Servern. Passt in eine kleine VM ohne GPU (Netz gebrückt, nicht NAT, damit die Erkennung per mDNS funktioniert). Ein im Dashboard gewähltes Modell schaltet das eigene Modell wieder ein. Im Installer: Modellauswahl „n“ bzw. `--model none`.
- **Externe Server:** Im Router-Bereich („Externe Server“) lassen sich andere OpenAI-kompatible Server eintragen, z. B. Strata oder vLLM: Name, Adresse, optional Schlüssel, Modellname und Rolle (nur per Name, schnell, groß). Clients erreichen sie über den Namen; gespeichert in `external.json`.

**Neuverteilung (`--rebalance`, Standard an):** `serve` schaut alle 30 Sekunden ins Netz und plant neu, sobald keine Anfrage läuft,

- wenn ein neuer Worker das Modell geschätzt um mindestens 15 % schneller macht oder einen längeren Kontext erlaubt (ohne `--ctx`),
- wenn ein Worker wegfällt (llama-server endet dann, oder der Worker ist dreimal hintereinander nicht zu sehen),
- und wenn das Modell nicht passt: Dann wartet `serve` auf Worker, statt sich zu beenden, und startet von selbst, sobald genug da sind.

Neu planen heißt neu laden. Worker ohne Tensor-Cache (Handys) bekommen ihre Layer erneut übers Netz; über WLAN dauert das einige Minuten, PC-Worker mit `--cache` laden schnell. Zwischen zwei Neuverteilungen für mehr Tempo liegen mindestens 5 Minuten; ein Worker, der nichts bringt, wird nur einmal geprüft. Die Schätzung nutzt den Speicher, den jedes Gerät vor dem Start frei hatte, daher stört die Prüfung den laufenden Server nicht. Im Dashboard: Schalter, Knopf „Jetzt neu verteilen“ und die letzten Ereignisse. `--rebalance on|off` setzt es beim Start, `auto` (Standard) nimmt die letzte Einstellung aus dem Dashboard. Mit `off` wird einmal geplant, und ein Modell, das nicht passt, beendet den Dienst mit Exit-Code 2 (die systemd-Units starten ihn dann nicht endlos neu).

Der Plan wird vor dem Start mit geschätzten Token/s, Anzahl Netzwerk-Hops und erwartetem Verkehr pro Token angezeigt.

**Messen:** Auf dem Master, während `serve` läuft:

```bash
myrmid bench                          # Token/s und KB pro Token (Ein- und Ausgang)
myrmid bench --temp 1.0 --context 8000  # bei Ihrer Temperatur und mit gefülltem Kontext
```

Messen Sie mit Ihrer echten Temperatur und einem gefüllten Kontext: Ein kurzer Test mit leerem Kontext unterschätzt den Nutzen der Worker, deren Hauptbeitrag oft der Platz für den KV-Cache ist. Bei MTP zeigt `bench` die Annahmequote der Vorhersagen. Gezählt werden beide Richtungen; ein Zähler nur für den Ausgang würde die zurückkommenden Ausgabe-Vektoren übersehen. Richtwerte: ein paar bis einige zehn KB pro Token und Worker sind normal. Etwa Vokabular × 4 Byte heißt, die Ausgabeschicht liegt auf einem Worker. Megabytes heißen, die Layer sind zwischen den Rechnern verschachtelt. Der API-Key wird beim ersten Start erzeugt (`~/.config/myrmid/api_key; als Dienst `/var/lib/myrmid/myrmid/api_key`, anzeigen mit `sudo myrmid key``) und steckt im QR-Code. In der App: *Phone cluster → Scan PC QR code* (oder *Find PCs*, dann Key eingeben). Der Chat läuft danach als eigenes Custom-Provider-Profil „PC · <hostname>“.

**Router (zwei Modelle, ein Endpunkt):** Mit einem kleinen Zweitmodell beantwortet der Master einfache Anfragen sofort, ohne Netzwerk-Runden, und schickt alles andere an das große (verteilte) Modell:

```bash
myrmid serve -m gross.gguf --fast-model klein.gguf          # Router an
myrmid serve -m gross.gguf --fast-model klein.gguf --router off
myrmid serve -m gross.gguf --fast-url http://10.0.0.5:8080/v1 --fast-key KEY   # vorhandener Server als schnelles Modell
```

- Clients sehen einen Endpunkt und den Modellnamen `myrmid` (`--model-name`; das alte `privatelm` geht weiter), dazu jedes Modell unter seinem eigenen Namen. Der Router läuft immer auf dem API-Port, auch ohne Zweitmodell (Hosts können später dazukommen); die llama-server laufen lokal auf Port+1 (groß) und Port+2 (schnell).
- Das schnelle Modell läuft auf einem Gerät: `--fast-device CUDA0` oder `CPU`, sonst auf der lokalen GPU, auf die es am besten passt. Sein Speicher (Gewichte, KV-Cache für `--fast-ctx`, Puffer) wird bei der Planung des großen Modells freigehalten.
- **Entscheidung „Regeln“** (Standard): an das große Modell gehen Anfragen mit Werkzeugen, Bildern, langem Kontext, langer gewünschter Antwort, strukturierter Ausgabe, langen Gesprächen, langen Fragen und Begriffen wie Code, Skript, Analyse, Vergleich, Berechnung, Docker, SQL. Alles andere geht an das schnelle. **„Schnelles Modell“**: Das kleine Modell bewertet jede Anfrage mit einem Token (einfach/schwer); die festen Regeln (Werkzeuge, Bilder, langer Kontext) gelten weiter.
- **Entscheidung „Beispiele“**: Ein kleines Embedding-Modell (embeddinggemma-300M, ≈ 300 MB, läuft auf der CPU des Masters und wird beim ersten Einschalten geladen) vergleicht die letzte Frage mit Beispielsätzen, die du im Dashboard pflegst. Die Kategorie mit dem ähnlichsten Satz entscheidet: Ziel „schnell“, „groß“ oder ein Modell per Name (z. B. eine Kategorie „code“ → Coder-Modell). Ans schnelle Modell geht eine Anfrage nur mit klarem Vorsprung (einstellbar), sonst ans große. Im Dashboard lässt sich jeder Satz testen. Gespeichert in `router-examples.json`; eigenes Embedding-Modell über `MYRMID_EMBED_MODEL` (Pfad oder `user/repo:QUANT`).
- **Kontext**: Der Router liest von jedem Modell, wie viel Kontext es wirklich hat, schätzt die Größe jeder Anfrage und schickt sie nur an Modelle, in die sie passt. Passt sie in keines, antwortet er wie OpenAI mit `context_length_exceeded` (HTTP 400); Clients wie opencode kürzen das Gespräch dann selbst.
- Gezielt ansprechen: Modellname `fast` oder `big`, oder Header `X-Myrmid-Route: fast|big`. Jede Antwort trägt `X-Myrmid-Route` und `X-Myrmid-Reason`.
- **Ein- und ausschalten:** im Dashboard (Schalter und Wahl der Entscheidungsart, wirkt sofort und bleibt über Neustarts gespeichert) oder per `--router on|off`; `--router auto` (Standard) nimmt die letzte Einstellung aus dem Dashboard. Aus heißt: alles an das große Modell.
- `bench` misst mit Router standardmäßig das große Modell (`--route fast` für das schnelle).

**Hosts (mehrere ganze Modelle, ein Endpunkt):** Ein Host ist ein PC, der ein ganzes Modell auf seine eigene GPU lädt, etwa drei PCs mit je einer A4000. Der Router des Masters verteilt die Anfragen auf sie. Anders als Worker (ein Modell über mehrere Rechner verteilt) rechnet jeder Host für sich, ohne Netzwerk-Runden pro Token.

```bash
# auf jedem Host (der Schlüssel ist der API-Key des Masters: dort myrmid key)
myrmid host --key-file /etc/myrmid/cluster_key
# auf dem Master: welcher Host was lädt (oder im Dashboard, Bereich Hosts)
myrmid serve -m gross.gguf --assign "pc2=qwen3-32b,name=qwen32" --assign "pc3=qwen3-32b,name=qwen32" \
                     --assign "pc4=bartowski/Qwen2.5-Coder-14B-Instruct-GGUF:Q4_K_M,name=coder,role=extra"
```

- **Zuweisen:** im Dashboard (Modell, Name, Rolle, Kontext, „Laden“/„Entladen“) oder mit `--assign HOST=MODELL[,role=big|fast|extra][,name=NAME][,ctx=N]`. HOST ist Rechnername, IP oder IP:Port; MODELL ein Katalogname, `user/repo:QUANT` oder eine Datei auf dem Host. Der Host lädt das Modell selbst herunter (fortsetzbar, SHA-256 geprüft) nach `--models-dir`, plant es auf seinen GPUs und stellt es auf Port 8081 bereit. Passt es nicht, meldet er das, bevor er herunterlädt. Die Zuweisung merken sich Master (`hosts.json`) und Host; nach einem Neustart lädt der Host sein Modell wieder, auch ohne Master.
- **Verteilen:** Hosts mit demselben Namen bilden eine Gruppe; jede Anfrage geht an das Mitglied mit den wenigsten laufenden Anfragen. Mit dem Schalter **„Schnellstes Gerät zuerst“** (Dashboard, Bereich Router) geht sie stattdessen an das Mitglied, das am schnellsten fertig wird, (laufende Anfragen + 1) / Token pro Sekunde: Einzelne Anfragen bekommt das schnellste Gerät, ein langsameres hilft erst bei Andrang. Das Tempo misst der Router an jeder Antwort, die ein Gerät allein gerechnet hat (vorher gilt die Schätzung des Planers). Clients wählen per Modellname (`"model": "qwen32"`, `/v1/models` listet alle) oder schicken `myrmid`: Dann entscheiden die Regeln zwischen den Rollen „schnell“ und „groß“; „nur per Name“ (`extra`) bekommt nur Anfragen, die es beim Namen nennen. Das eigene Modell des Masters ist ein Mitglied der Rolle „groß“. Jede Antwort trägt `X-Myrmid-Backend: name@rechner`.
- **Sicherheit:** Die Steuerung der Hosts (Port 50070) und ihre Modell-API (8081) verlangen den API-Key des Masters; wer ihn hat, kann Modelle laden lassen. Das Installationsskript (Rolle 3) legt den Schlüssel root-lesbar in `/etc/myrmid/cluster_key` ab und öffnet beide Ports nur für den Master.
- **Mehrere GPUs in einem Host:** Standard ist ein Modell über alle GPUs des PCs (doppelt so viel VRAM, eine Antwort wird aber nicht schneller). Im Dashboard lässt sich auf der Karte des Hosts auf **„Jede GPU ein eigenes Modell“** umschalten: Der PC erscheint dann als ein Host pro GPU („PC · GPU 0“, „PC · GPU 1“), jede GPU bekommt ihr eigenes Modell (llama-server mit `--device`), Modelle, die nicht ganz in den VRAM passen, teilen sich den RAM. Gleicher Name für Clients auf mehreren GPUs ergibt ein gemeinsames Modell, auf das der Router die Anfragen verteilt (doppelter Durchsatz bei gleichzeitigen Anfragen); verschiedene Namen ergeben getrennte Modelle, auch bei derselben Modelldatei. Beim Umschalten läuft das Modell der ersten GPU (bzw. das bisherige) weiter, die anderen GPUs behalten, was sie vorher hatten. Ports ändern sich nicht: Steuerung (50070) und Modell-API (8081) leiten `/gpuN/…` an die jeweilige GPU weiter, die llama-server selbst laufen auf `127.0.0.1:18081` und folgende. Ein PC mit Modell pro GPU schläft erst, wenn alle seine GPUs lange genug keine Anfrage hatten. Die Wahl merkt sich der Host (`host-layout.json`).
- **Parallele Anfragen:** Jedes Modell hat 4 Plätze (llama.cpp-Standard) und rechnet bis zu 4 Anfragen gleichzeitig; sie teilen sich den Kontext.
- **Aufnehmen ohne Key:** Ein Worker oder Host ohne Cluster-Schlüssel meldet sich mit `pair=1` im LAN. Der Master (`serve`, abschaltbar mit `--auto-pair off`) gibt ihm seinen API-Key (`POST /pair`), der PC speichert ihn (`cluster_key` im Datenordner) und gehört ab dann zu diesem Master. Aufnehmen darf nur, wer in `--pair-from` steht (das Installationsskript trägt die gefundenen Adressen des Masters ein), sonst jeder Master in der ersten Stunde nach dem Start. Ein Host ohne Schlüssel wartet so lange, bevor er Modelle lädt.
- Hosts brauchen eigenen Speicher für ihr Modell; Worker brauchen kein Modell (der Master schickt ihnen ihre Layer).

**MoE-Expertencache (llama.cpp ab b12191):** Passen bei einem MoE-Modell nicht alle Experten auf die GPU(s), legt der Planer alle Experten in den RAM (`--cpu-moe`) und nutzt den übrigen GPU-Speicher als Cache für die meistgenutzten (`--moe-cache-mib`). Das schlägt die feste Aufteilung mit `--n-cpu-moe`, weil Experten ungleich oft gebraucht werden (llama.cpp misst 72 bis 89 % Treffer). Fehlgriffe rechnet zum Teil die CPU direkt aus dem RAM, statt die Experten über PCIe hochzuladen (`--moe-cache-cpu-share`, Standard 0,5). `myrmid tune-moe -m MODELL` misst mit llama.cpps `moe-cache-tune.py` Cache-Größe und CPU-Anteil für diesen PC (Server vorher stoppen, dauert einige Minuten) und speichert sie; `serve` nutzt sie danach. `--moe-cache off` schaltet zurück auf die feste Aufteilung, `--moe-cache MIB` begrenzt den Cache. Gilt für MoE-Modelle, die auf einem PC laufen (Master ohne Worker, Hosts).

**Planer lernt aus Messungen:** Die Geschwindigkeit eines Geräts schätzt der Planer aus seiner Speicherbandbreite. Jede Anfrage an das Modell des Masters, die allein läuft, liefert eine Messung (llama-servers `timings`; mit Entwurfsmodell oder MTP pro geprüftem Durchlauf). Pro Plan ergibt das eine Gleichung mit einem unbekannten Faktor je Gerät; mehrere Pläne (Neuverteilung erzeugt sie) legen die Faktoren fest, eine Ridge-Regression hält ein Gerät bei seiner Schätzung, bis Messungen etwas anderes sagen. Der Planer rechnet danach mit den gemessenen Geschwindigkeiten (`calibration.json`). Sagt die Messung, dass eine andere Aufteilung mindestens 15 % schneller wäre, verteilt `serve` neu (mit Neuverteilung an); ein schon gemessener Plan zählt mit seinem Messwert. `myrmid calibration` zeigt die Faktoren, `--reset` vergisst sie. Das Dashboard zeigt die Messung neben der Schätzung und pro Rechner den Faktor.

**Entwurfsmodell (`--draft-model auto|DATEI`):** Ein kleines Modell derselben Familie (gleicher Tokenizer, z. B. Qwen3-0.6B zu Qwen3-32B) schreibt bis zu `--draft-max` (8) Tokens vor, das große prüft sie in einem Durchlauf. Gleiche Ausgabe, oft 1,5- bis 2-mal so viele Tokens pro Durchlauf; am meisten, wenn das Modell über das Netz verteilt ist, denn jeder Durchlauf kostet Netzwerk-Runden. `auto` nimmt eine passende kleine GGUF-Datei auf diesem PC (höchstens 12 % des großen Modells) oder lädt eine aus dem Katalog (Qwen3.5-0.8B, Qwen3-0.6B, Llama-3.2-1B). Den Tokenizer prüft `serve` vorher wie llama.cpp; passt er nicht, läuft es ohne. Das Entwurfsmodell läuft auf der lokalen GPU mit dem meisten Platz (sonst auf der CPU), sein Speicher geht vom Plan des großen ab. Mit `--mtp` wird es nicht genutzt. `bench` zeigt die Trefferquote.

**Schnellste Verbindung:** Hat ein PC Kabel und WLAN, meldet er sich mit der Adresse seiner schnellsten Verbindung ins LAN: Kabel vor WLAN, das schnellere Kabel zuerst (nur Verbindungen ins selbe Netz wie die Standardroute; ein zweites Kabel zum NAS zählt nicht). Alle 20 Sekunden prüft er das; wird ein Kabel eingesteckt oder gezogen, zieht seine mDNS-Ankündigung auf die neue Adresse um. Der Master merkt, dass ein Worker im Plan jetzt unter einer anderen Adresse erreichbar ist, und plant um, sobald keine Anfrage läuft (die Layer werden dabei neu übertragen, über Kabel schnell). Hosts übernimmt der Router ebenso. Feste Adressen (`--advertise-ip`, `--host`) bleiben fest. Firewall: Das Installationsskript findet den Master im Netz und gibt die Worker-Ports für alle seine Adressen frei (Kabel und WLAN); bekommt der Master später eine neue Verbindung, das Skript auf den Workern noch einmal laufen lassen (oder `ufw allow from NEUE-IP`).

**Wake-on-LAN und Schlafen:** Worker und Hosts melden ihre MAC-Adresse (nur Kabel); der Master merkt sich jeden Rechner (`known.json`) und weckt ihn per Magic Packet,
- einen Worker, wenn das Modell ohne ihn nicht passt (der Dienst wartet dann auf Worker),
- einen Host, wenn eine Anfrage sein Modell braucht: Die Anfrage wartet, bis er wach ist und sein Modell geladen hat (bis 240 s),
- und von Hand im Dashboard („Aufwecken“, auch für ausgeschaltete Worker in „Weitere Geräte im Netzwerk“).

Hosts legen sich nach `--host-sleep MIN` Minuten ohne Anfrage schlafen (pro Host im Dashboard einstellbar, 0 = nie): Der Master sagt es dem Host, der beendet llama-server und geht in Standby, über `myrmid-suspend.path`/`.service` (kleine Root-Units, das Installationsskript richtet sie ein). Nach dem Aufwachen lädt der Host sein Modell von selbst wieder. Worker im Plan bleiben wach. Das Installationsskript schaltet Wake-on-LAN auf der Netzwerkkarte ein (ethtool, dauerhaft über NetworkManager oder `myrmid-wol.service`); im BIOS muss es ebenfalls an sein.

**Im Dashboard umstellen:**
- *Modell des Masters:* „Modell wechseln“ unter dem Modellnamen. Lokale GGUF-Dateien oder Katalogname bzw. `user/repo:QUANT` (fehlende werden heruntergeladen, Fortschritt im Dashboard), danach plant der Master neu. MTP und Entwurfsmodell werden für das neue Modell neu bestimmt. Die Wahl gilt auch nach einem Neustart (`model.json`), solange das `-m` des Dienstes gleich bleibt; ein geändertes `-m` gewinnt.
- *Rolle eines PCs:* Worker und Hosts lassen sich gegenseitig umschalten („Zum Host machen“ / „Zum Worker machen“ auf der Karte des Rechners). Das Installationsskript richtet dafür auf Worker- und Host-PCs beide Dienste ein (einer läuft), legt den API-Key des Masters als Cluster-Schlüssel ab und installiert `myrmid-role.path`: Der laufende Dienst schreibt die gewünschte Rolle nach `/run/myrmid/role`, ein kleiner Root-Dienst stoppt den einen und startet den anderen. Die Anfrage des Masters ist mit dem Cluster-Schlüssel gesichert; ohne Schlüssel (Worker ohne Key installiert) bleibt der PC, was er ist.

**Dashboard:** `serve` startet auf Port 8090 eine Weboberfläche (`http://<Master-IP>:8090/`, Anmeldung mit dem API-Key). Sie zeigt die Modell-Aufteilung Layer für Layer in Ausführungsreihenfolge, Live-Token/s, Netzwerkverkehr pro Token, den Verlauf der letzten zehn Minuten und für Master und jeden Worker CPU, RAM, Netz sowie je GPU Auslastung, VRAM, Temperatur und Leistung (NVIDIA über `nvidia-smi`, AMD über die Treiberwerte). Die Worker liefern ihre Messwerte über einen kleinen Nur-Lese-Endpunkt auf Port 50060, den nur der Master erreichen muss. Handys haben keinen Status-Endpunkt: Ihre Karte zeigt, ob der Master mit ihnen verbunden ist und ob sie im Netz zu sehen sind. Ein Knopf startet einen Kurztest. Optionen: `--dashboard-port`, `--no-dashboard` (serve), `--status-port`, `--no-status` (worker).

**Updates:** siehe [README im Hauptordner](../README.md#updates). Kurz: Jeder PC baut neue Releases nachts neben der laufenden Version (`/opt/myrmid/releases/X.Y.Z`), der Master stellt alle zusammen um, sobald alle sie haben und keine Anfrage läuft. `myrmid update` zeigt den Stand.

**Als Dienst:** Das richtet `install.sh` ein. Von Hand: `systemd/myrmid-worker.service`, `myrmid-host.service` bzw. `myrmid-server.service` nach `/etc/systemd/system/` kopieren, Pfade anpassen, `systemctl enable --now …`; für Updates zusätzlich `myrmid-update.timer`, `myrmid-update.service`, `myrmid-update-request.path` und `.service`.

## Netzwerk und Sicherheit

- Das RPC-Protokoll von llama.cpp hat **keine Authentifizierung**. Worker nur im vertrauenswürdigen LAN oder über ein VPN betreiben. Mit `--host <LAN- oder VPN-IP>` lauscht der Worker nur dort; Port 50052 nicht ins Internet freigeben.
- Die API (`serve`) ist per API-Key geschützt, läuft aber unverschlüsselt über HTTP. Außerhalb des LANs nur über VPN (z. B. NetBird/WireGuard) oder hinter einem TLS-Reverse-Proxy.
- mDNS funktioniert nicht über VPNs und in Gast-WLANs mit Client-Isolation. Dann per QR-Code oder IP koppeln, mit `--advertise-ip <VPN-IP>` steht die richtige Adresse im QR-Code.
- Ein Worker bedient immer nur **einen** Master gleichzeitig.

## Grenzen

- Handy-Master gibt es nur auf Android (wie im Original); iOS-Geräte sind Worker und können sich per QR-Code mit einem PC-Server verbinden.
- Leistung im Cluster: Pro Token gibt es Netzwerk-Roundtrips zwischen den Geräten. Ein Handy-Master mit PC-GPU-Worker ist deutlich schneller als ein reiner Handy-Cluster, aber langsamer als das Modell direkt auf dem PC (`serve`). Für große Modelle ist `serve` auf dem PC mit Handy-Chat der schnellste Weg.
- Getestet in einer CPU-Umgebung: Build, Bandbreitenmessung, mDNS-Ankündigung und -Suche, Layer-Offload auf einen RPC-Worker und Chat-Completion über die API. GPU-Builds (CUDA/ROCm/Vulkan/SYCL) nutzen die Standard-Optionen von llama.cpp, sind hier aber mangels GPU nicht gelaufen.
