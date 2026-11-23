# MyrmidLM Node

Bringt Linux-PCs mit GPU (NVIDIA, AMD, Intel) in den MyrmidLM-Cluster. Installation und Updates: siehe [README im Hauptordner](../README.md) (`install.sh`). Ein PC kann:

| Rolle | Befehl | Was passiert |
|---|---|---|
| **GPU-Worker** | `myrmid worker` | Der PC leiht seine GPU(s) einem Handy-Master. Er taucht in der App unter *Phone cluster → Find nearby* auf wie ein Handy, bekommt aber wegen der viel höheren Speicherbandbreite den größten Anteil der Layer. |
| **Master / Server** | `myrmid serve -m model.gguf` | Das Modell läuft auf dem PC (GPU + RAM), optional mit Handys oder weiteren PCs als Workern. Bereitgestellt als OpenAI-kompatible API; die App verbindet sich per QR-Code oder *Find PCs*. |
| **Diagnose** | `myrmid detect` / `scan` / `bench` | Geräte, freier Speicher, gemessene Bandbreite / alle Worker und Server im LAN / Token pro Sekunde und Netzwerkverkehr pro Token. |

Technisch nutzt alles das RPC-Backend von llama.cpp, das die App bereits für den Handy-Cluster verwendet. Der Node wird aus **exakt demselben llama.cpp-Commit** gebaut wie die App (derzeit llama.cpp v1.0.0 vom 13.11.2026, RPC-Protokoll 10; ältere Worker und App-Versionen mit RPC 9 oder älter erscheinen als „andere Version“ und werden nicht genutzt) (`LLAMA_CPP_COMMIT`, plus die beiden Logging-Patches der App unter `patches/`). Das ist Pflicht: RPC-Protokoll und Op-Liste müssen bitgenau passen, sonst bricht llama.cpp beim Verbinden ab. Beim Update der App auf eine neue llama.cpp-Version muss `LLAMA_CPP_COMMIT` mitziehen (Protokollkennung `rpc10` in der App und in `myrmid.py`).

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

Ohne `--backends` erkennt `build.sh` die GPUs selbst: NVIDIA mit installiertem CUDA-Toolkit → CUDA, sonst Vulkan; AMD und Intel → Vulkan. ROCm unterstützt ältere AMD-Karten (Polaris, gfx803) nicht mehr, Vulkan läuft auf allen und liegt bei der Textgenerierung nahe an ROCm. Mehrere Backends werden als Module gebaut und zur Laufzeit geladen, ein fehlender Treiber verhindert also nicht den Start. Sieht eine NVIDIA-Karte sowohl CUDA als auch Vulkan, wird sie trotzdem nur einmal genutzt (Abgleich über die PCI-Adresse, wie in llama.cpp). Die CUDA-Architekturen der eingebauten Karten liest `build.sh` aus `nvidia-smi` (z. B. 86 für A2000/3070 Ti), das spart viel Build-Zeit. Ältere Karten (Maxwell, Pascal, Volta) erkennt es zusätzlich am Chipnamen aus `lspci`, weil ein Treiber neuer als 580 sie in `nvidia-smi` nicht mehr anzeigt. Mit CUDA 13 baut `build.sh` ohne diese Karten und warnt; für sie braucht es CUDA 12.9 und Treiber 580 mit proprietärem Kernel-Modul (das offene Modul kann erst Turing). `install.sh` erkennt eine solche Karte auch neben neueren Karten und bietet den Wechsel auf Treiber 580 an. Gibt es CUDA 12.9 nicht als Paket (Ubuntu 26.04 hat nur CUDA 13), baut `install.sh` auf Wunsch im CUDA-12.9-Container; die nächtlichen Updates dann auch.

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
- **Hinweis zu 2.0:** Ab MyrmidLM 2.0 nutzt der Cluster llama.cpp 1.0 mit RPC-Protokoll 10. Alle PCs aktualisieren sich gemeinsam; Handys brauchen die App ab 2.0, ältere Apps erscheinen als „andere Version“.
- **Autotune:** Im Dashboard („Modell und Kontext“ für den Master, Knopf in jeder Host-Karte) oder mit `myrmid tune -m MODELL` (Dienst vorher stoppen). Bei MoE-Modellen mit Experten im RAM misst er zuerst mit llama.cpps eigenem Test (`moe-cache-tune.py`, wie `myrmid tune-moe`) Größe und CPU-Anteil des Experten-Caches. Dann startet er das geplante Modell mehrmals mit anderen Einstellungen (Batch-Größe `-ub`, CPU-Threads, bei MoE Lademodus `-lm dio`/`mlock` und CPU-Anteil 0,5/0,95), misst je einen langen Prompt plus Antwort und behält die schnellste Variante. Läuft eine Variante wegen GPU-Speicher nicht (größeres `-ub` braucht mehr Rechenpuffer neben dem Experten-Cache), misst er sie noch einmal mit 75 % bzw. 50 % des Caches. Beim Laden plant ein Host den Rechenpuffer der ersten GPU mit ein (er wächst mit `-ub` und mit dem Kontext). Passt llama-server trotzdem nicht in den GPU-Speicher, versucht der Host es mit weniger Experten-Cache, dann mit `-ub 512`, dann mit kürzerem Kontext (je ein Viertel weniger, auch bei fest eingestelltem Kontext); die Host-Karte sagt dann z. B. „Kontext 131.072 passte nicht in den GPU-Speicher, geladen mit 98.304“. `-lm mlock` braucht `LimitMEMLOCK=infinity` im Dienst; ein Update ergänzt das bei älteren Installationen als Drop-in. Gespeichert in `tune.json` pro Modell und GPU-Slot; jeder spätere Start nutzt sie. Läuft nur für Modelle, die auf einem PC laufen (mit Workern entscheidet das Netz). Bei mehreren GPUs im PC probiert er außerdem das Modell über alle GPUs zu verteilen (liest lange Prompts schneller, auch wenn es auf eine passt) und bei dichten Modellen Tensor Split (`-sm tensor`: schreibt schneller, liest langsamer; braucht schnelle PCIe-Anbindung und f16-KV-Cache). Die PCIe-Anbindung jeder GPU steht im Dashboard; weniger Lanes als die Karte kann sind orange markiert.
- **Qwen3.8-Flash-Next (125B MoE):** Im Katalog als `qwen3.8-flash-next` (UD-IQ3_XXS, ≈ 82 GB). Gedacht für einen Host mit GPU und ≥ 96 GB RAM: Attention und KV-Cache auf der GPU, Experten über den MoE-Cache in RAM und VRAM, die n-gram-Tabelle bleibt im RAM (bei Knappheit liest das System sie von der SSD). Danach Autotune laufen lassen. Geteilte GGUF-Dateien (`-00001-of-0000N`) plant MyrmidLM jetzt über alle Teile.
- **Nur Router (Master ohne eigenes Modell):** `myrmid serve -m none` oder im Dashboard unter „Modell und Kontext“ den Schalter „Eigenes Modell auf diesem PC“ ausschalten (gespeichert in `master.json`, gilt vor `-m`). Der Master übernimmt dann nur Dashboard, Router, Kopplung, Updates und die Steuerung der Hosts; alle Modelle laufen auf Hosts oder externen Servern. Passt in eine kleine VM ohne GPU (Netz gebrückt, nicht NAT, damit die Erkennung per mDNS funktioniert). Ein im Dashboard gewähltes Modell schaltet das eigene Modell wieder ein. Im Installer: Modellauswahl „n“ bzw. `--model none`.
- **Test-Chat:** Knopf „Chat testen“ oben im Dashboard (oder `…:8090/#chat`). Geht über die API des Masters mit Router, wie opencode. Unter jeder Antwort: welches Modell auf welchem Rechner geantwortet hat, die Entscheidung des Routers, Lese- und Schreibgeschwindigkeit, belegter Kontext. „Vergleich“ schickt dieselbe Frage an zwei Modelle nebeneinander. Systemprompt, Temperatur und Antwortlänge unter „Einstellungen“; der Verlauf bleibt im Browser. Für den Alltag mit mehreren Personen eignet sich eher Open WebUI mit dem Master als OpenAI-Server.
- **Vision-Modelle (Bilder, OCR):** Modelle mit Bildteil (`mmproj`) werden erkannt: Beim Download über `user/repo:QUANT` oder den Katalog (`qwen3-vl-4b`, `qwen3-vl-8b`) lädt MyrmidLM den Bildteil (bevorzugt F16) mit und startet llama-server mit `--mmproj`; liegt er schon neben dem Modell, wird er gefunden. Anfragen mit Bildern schickt der Router (`model: myrmid`) nur an Modelle, die Bilder lesen; eins per Name zu wählen geht wie bei jedem Modell (z. B. Name `ocr`, Rolle „extra“). Ohne Vision-Modell kommt ein klarer Fehler (`model_not_vision`). Im Test-Chat Bilder per 📎, Einfügen oder Ziehen anhängen.
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
- **Verteilen:** Hosts mit demselben Namen bilden eine Gruppe; jede Anfrage geht an das Mitglied mit den wenigsten laufenden Anfragen. Mit dem Schalter **„Schnellstes Gerät zuerst“** (Dashboard, Bereich Router) geht sie stattdessen an das Mitglied, das am schnellsten fertig wird, (laufende Anfragen + 1) / Token pro Sekunde: Einzelne Anfragen bekommt das schnellste Gerät, ein langsameres hilft erst bei Andrang. Das Tempo misst der Router an jeder Antwort, die ein Gerät allein gerechnet hat (vorher gilt die Schätzung des Planers). Clients wählen per Modellname (`"model": "qwen32"`, `/v1/models` listet alle) oder schicken `myrmid`: Dann entscheiden die Regeln zwischen den Rollen „schnell“ und „groß“; „nur per Name“ (`extra`) bekommt nur Anfragen, die es beim Namen nennen. Das eigene Modell des Masters ist ein Mitglied der Rolle „groß“. Jede Antwort trägt `X-Myrmid-Backend: name@rechner`. Sind mehrere gleich gut (gleich viele laufende Anfragen bzw. gleich schnell fertig), antwortet das mit der **kühleren GPU** (in 5-°C-Schritten).
- **Fachgebiete** (Router, „Fachgebiete“): Anfragen eines Fachgebiets gehen an ein Modell per Name, z. B. Programmieren an ein Coder-Modell, Berichte an ein anderes. Erkannt an Stichwörtern (Wortanfänge, die letzte Nachricht zählt doppelt); wahlweise ordnet das schnelle Modell ein, wenn kein Stichwort passt. Ein Fachgebiet ohne Ziel tut nichts, ist sein Modell nicht bereit, verteilt der Router wie sonst; geht auch ohne schnelles Modell. Bilder, Hintergrundaufträge und feste Modellnamen gehen vor. Gespeichert in `router-subjects.json`.
- **Hintergrundaufträge:** Open WebUI schickt nach Antworten eigene kleine Anfragen (Titel, Tags, Folgefragen, Suchanfragen; Prompt beginnt mit `### Task:`). Bei eingeschaltetem Verteilen gehen sie ohne Einordnung ans schnelle Modell, wenn eins bereit ist, und halten das große nicht auf; ein `X-Myrmid-Route`-Header hat Vorrang.
- **Sicherheit:** Die Steuerung der Hosts (Port 50070) und ihre Modell-API (8081) verlangen den API-Key des Masters; wer ihn hat, kann Modelle laden lassen. Das Installationsskript (Rolle 3) legt den Schlüssel root-lesbar in `/etc/myrmid/cluster_key` ab und öffnet beide Ports nur für den Master.
- **Mehrere GPUs in einem Host:** Standard ist ein Modell über alle GPUs des PCs (doppelt so viel VRAM, eine Antwort wird aber nicht schneller). Im Dashboard lässt sich auf der Karte des Hosts auf **„Jede GPU ein eigenes Modell“** umschalten: Der PC erscheint dann als ein Host pro GPU („PC · GPU 0“, „PC · GPU 1“), jede GPU bekommt ihr eigenes Modell (llama-server mit `--device`), Modelle, die nicht ganz in den VRAM passen, teilen sich den RAM. Gleicher Name für Clients auf mehreren GPUs ergibt ein gemeinsames Modell, auf das der Router die Anfragen verteilt (doppelter Durchsatz bei gleichzeitigen Anfragen); verschiedene Namen ergeben getrennte Modelle, auch bei derselben Modelldatei. Beim Umschalten läuft das Modell der ersten GPU (bzw. das bisherige) weiter, die anderen GPUs behalten, was sie vorher hatten. Ports ändern sich nicht: Steuerung (50070) und Modell-API (8081) leiten `/gpuN/…` an die jeweilige GPU weiter, die llama-server selbst laufen auf `127.0.0.1:18081` und folgende. Ein PC mit Modell pro GPU schläft erst, wenn alle seine GPUs lange genug keine Anfrage hatten. Die Wahl merkt sich der Host (`host-layout.json`).
- **GPU-Gruppen** (ab drei GPUs): „GPU-Gruppen“ in der Host-Karte fasst Karten zusammen, jede Gruppe bekommt ein eigenes Modell, z. B. Gruppe 1 = RTX A4000 mit Strata, Gruppe 2 = beide RTX A2000 zusammen mit einem llama.cpp-Modell (llama-server mit `--device CUDA1,CUDA2`, über beide Karten verteilt). Vorgeschlagen werden gleiche Karten zusammen; jede GPU gehört in genau eine Gruppe, mindestens zwei Gruppen. Jede Gruppe erscheint als eigener Host („PC · Gruppe N“, Pfad `/grpN`), mit eigenem Autotune und eigenen gesicherten Konfigurationen. Werden die Gruppen geändert, startet jede Gruppe leer (Gruppe 1 übernimmt das bisherige Modell).
- **Parallele Anfragen:** Jedes Modell hat 4 Plätze (llama.cpp-Standard) und rechnet bis zu 4 Anfragen gleichzeitig; sie teilen sich den Kontext.
- **Aufnehmen ohne Key:** Ein Worker oder Host ohne Cluster-Schlüssel meldet sich mit `pair=1` im LAN. Der Master (`serve`, abschaltbar mit `--auto-pair off`) gibt ihm seinen API-Key (`POST /pair`), der PC speichert ihn (`cluster_key` im Datenordner) und gehört ab dann zu diesem Master. Aufnehmen darf nur, wer in `--pair-from` steht (das Installationsskript trägt die gefundenen Adressen des Masters ein), sonst jeder Master in der ersten Stunde nach dem Start. Ein Host ohne Schlüssel wartet so lange, bevor er Modelle lädt.
- Hosts brauchen eigenen Speicher für ihr Modell; Worker brauchen kein Modell (der Master schickt ihnen ihre Layer).

**MoE-Expertencache (llama.cpp ab b12191):** Passen bei einem MoE-Modell nicht alle Experten auf die GPU(s), legt der Planer alle Experten in den RAM (`--cpu-moe`) und nutzt den übrigen GPU-Speicher als Cache für die meistgenutzten (`--moe-cache-mib`). Das schlägt die feste Aufteilung mit `--n-cpu-moe`, weil Experten ungleich oft gebraucht werden (llama.cpp misst 72 bis 89 % Treffer). Fehlgriffe rechnet zum Teil die CPU direkt aus dem RAM, statt die Experten über PCIe hochzuladen (`--moe-cache-cpu-share`, Standard 0,5). `myrmid tune-moe -m MODELL` misst mit llama.cpps `moe-cache-tune.py` Cache-Größe und CPU-Anteil für diesen PC (Server vorher stoppen, dauert einige Minuten) und speichert sie; `serve` nutzt sie danach. `--moe-cache off` schaltet zurück auf die feste Aufteilung, `--moe-cache MIB` begrenzt den Cache. Gilt für MoE-Modelle, die auf einem PC laufen (Master ohne Worker, Hosts).

**Planer lernt aus Messungen:** Die Geschwindigkeit eines Geräts schätzt der Planer aus seiner Speicherbandbreite. Jede Anfrage an das Modell des Masters, die allein läuft, liefert eine Messung (llama-servers `timings`; mit Entwurfsmodell oder MTP pro geprüftem Durchlauf). Pro Plan ergibt das eine Gleichung mit einem unbekannten Faktor je Gerät; mehrere Pläne (Neuverteilung erzeugt sie) legen die Faktoren fest, eine Ridge-Regression hält ein Gerät bei seiner Schätzung, bis Messungen etwas anderes sagen. Der Planer rechnet danach mit den gemessenen Geschwindigkeiten (`calibration.json`). Sagt die Messung, dass eine andere Aufteilung mindestens 15 % schneller wäre, verteilt `serve` neu (mit Neuverteilung an); ein schon gemessener Plan zählt mit seinem Messwert. `myrmid calibration` zeigt die Faktoren, `--reset` vergisst sie. Das Dashboard zeigt die Messung neben der Schätzung und pro Rechner den Faktor.

**Entwurfsmodell (`--draft-model auto|DATEI`):** Ein kleines Modell derselben Familie (gleicher Tokenizer, z. B. Qwen3-0.6B zu Qwen3-32B) schreibt bis zu `--draft-max` (8) Tokens vor, das große prüft sie in einem Durchlauf. Gleiche Ausgabe, oft 1,5- bis 2-mal so viele Tokens pro Durchlauf; am meisten, wenn das Modell über das Netz verteilt ist, denn jeder Durchlauf kostet Netzwerk-Runden. `auto` nimmt eine passende kleine GGUF-Datei auf diesem PC (höchstens 12 % des großen Modells) oder lädt eine aus dem Katalog (Qwen3.5-0.8B, Qwen3-0.6B, Llama-3.2-1B). Den Tokenizer prüft `serve` vorher wie llama.cpp; passt er nicht, läuft es ohne. Das Entwurfsmodell läuft auf der lokalen GPU mit dem meisten Platz (sonst auf der CPU), sein Speicher geht vom Plan des großen ab. Mit `--mtp` wird es nicht genutzt. `bench` zeigt die Trefferquote.

**Schnellste Verbindung:** Hat ein PC Kabel und WLAN, meldet er sich mit der Adresse seiner schnellsten Verbindung ins LAN: Kabel vor WLAN, das schnellere Kabel zuerst (nur Verbindungen ins selbe Netz wie die Standardroute; ein zweites Kabel zum NAS zählt nicht). Alle 20 Sekunden prüft er das; wird ein Kabel eingesteckt oder gezogen, zieht seine mDNS-Ankündigung auf die neue Adresse um. Der Master merkt, dass ein Worker im Plan jetzt unter einer anderen Adresse erreichbar ist, und plant um, sobald keine Anfrage läuft (die Layer werden dabei neu übertragen, über Kabel schnell). Hosts übernimmt der Router ebenso. Feste Adressen (`--advertise-ip`, `--host`) bleiben fest. Firewall: Das Installationsskript findet den Master im Netz und gibt die Worker-Ports für alle seine Adressen frei (Kabel und WLAN); bekommt der Master später eine neue Verbindung, das Skript auf den Workern noch einmal laufen lassen (oder `ufw allow from NEUE-IP`).

**Wake-on-LAN und Schlafen:** Worker und Hosts melden ihre MAC-Adresse (nur Kabel); der Master merkt sich jeden Rechner (`known.json`) und weckt ihn per Magic Packet,
- einen Worker, wenn das Modell ohne ihn nicht passt (der Dienst wartet dann auf Worker),
- einen Host, wenn eine Anfrage sein Modell braucht: Die Anfrage wartet, bis er wach ist und sein Modell geladen hat (bis 240 s), Wacht er nicht auf (z. B. ausgeschaltet), antwortet ein anderes bereites Modell, und der Router versucht ihn 10 Minuten lang nicht mehr. Ein wacher Host, in den die Anfrage nach der optimistischen Schätzung passt, geht vor dem Wecken eines schlafenden,
- und von Hand im Dashboard („Aufwecken“, auch für ausgeschaltete Worker in „Weitere Geräte im Netzwerk“).

**Reasoning-Stufen:** Schickt ein Client eine Stufe, die das Chat-Template des Modells nicht kennt (z. B. Hermes `reasoning_effort: minimal`, Qwen3.8 kennt nur low, medium und xhigh), wiederholt der Router die Anfrage mit der nächstliegenden erlaubten Stufe und merkt sich das für diesen Host.

**Host entfernen:** „Entfernen“ in der Host-Karte (gilt für den ganzen PC) entlädt sein Modell, wenn er erreichbar ist, und der Master vergisst ihn samt Zuweisung und Einstellungen. Taucht der PC wieder im Netz auf, nimmt der Master ihn nicht von selbst auf; unter der Host-Liste steht er bei „Entfernte Hosts“ mit „Wieder aufnehmen“. Ganz weg ist er erst, wenn MyrmidLM auf dem PC deinstalliert ist.

**CPU-Takt:** Auf Workern und Hosts fragt der Installer, ob die CPU immer mit vollem Takt laufen soll (`myrmid-cpu-performance.service`, Governor „performance“). Bei MoE-Modellen rechnet die CPU die Experten im RAM, Energiesparen kostet dort Tempo. Nachträglich: `sudo systemctl enable --now myrmid-cpu-performance` (ab 2.4.0 liegt der Dienst nach einem Update bereit), aus mit `disable --now`.

**Bilder an/aus:** In der Host-Karte wählt „Bilder“ den Bild-Teil eines Vision-Modells: „an (GPU, wenn Platz)“ (Standard, wie unten beschrieben), „an, im RAM“ (immer auf der CPU, der VRAM bleibt für Text) oder „aus“ (nur Text, der ganze VRAM für den Experten-Cache; der Router schickt Bilder dann an ein anderes Modell).

**Bild-Teil (mmproj) im RAM, wenn VRAM fehlt:** Der Planer rechnet zwei Pläne, mit dem Bild-Teil auf der GPU und im RAM. Kostet er auf der GPU mehr als 10 % Tempo für Text (oder passt gar nicht), bleibt er im RAM (`--no-mmproj-offload`) und die CPU liest Bilder ein; die Host-Karte sagt das. Bei Modellen, die vor ihrem Bild-Teil heruntergeladen wurden, holt der Host ihn beim nächsten Laden einmalig nach. Lädt diese llama.cpp-Version den Bild-Teil nicht, startet das Modell ohne ihn (nur Text, mit Hinweis).

**Rig live:** Der Reiter „Rig live“ (oder `/#rig`) zeigt jede GPU des Clusters als Karte, nach PC gruppiert: das Modell darauf (mit Engine, Rolle, GPU-Gruppe), laufende Anfragen, Auslastung, Leistung, Temperatur und VRAM; oben GPUs online, wie viele gerade rechnen, geladene Modelle, Leistung gesamt und die wärmste Karte. Schlafende Hosts erscheinen ausgegraut.

**Hilfe:** Der Reiter „Hilfe“ (oder `/#hilfe`) sammelt die wichtigsten Befehle und Hinweise (Clients anbinden, Installation, Updates, Dienste und Logs, Modelle und GPUs, Strata, Router, Fehlerbehebung), mit Suchfeld und Kopier-Knopf an jedem Befehl; die Adresse des Masters ist schon eingesetzt.

**Design:** Der Knopf neben den Reitern wechselt zwischen automatisch (wie das System), dunkel und hell; die Wahl merkt sich der Browser.

**Monitor:** Der Reiter „Monitor“ oben im Dashboard (oder `/#monitor`) zeigt je Host oder für alle: Tempo der letzten Anfrage (schreiben, lesen), GPU-Auslastung, VRAM, GPU-Temperatur, Leistungsaufnahme, PCIe-Anbindung, CPU und RAM, jeweils mit Verlauf der letzten 5 Minuten; dazu die Kontext-Füllung der letzten Anfrage, wie viel des Prompts aus dem Cache kam, und eine Liste der letzten 200 Anfragen mit Absender (aus dem User-Agent: Open WebUI, Hermes, …), warum der Router so entschieden hat, wie lange das Einordnen dauerte, Prompt-, Cache- und Ausgabe-Token, Tempo und Dauer (`GET /api/monitor` auf dem Dashboard-Port). Die Werte kommen aus den „timings“ von llama-server, bei externen Servern aus „usage“.

**Gesicherte Konfigurationen:** In jeder Host-Karte unter „Gesicherte Konfigurationen“: „Aktuelle Konfiguration sichern“ speichert Zuweisung (Modell, Rolle, Name, Kontext, KV-Cache), Autotune-Werte und Cache-Messung, dazu Startbefehl und Tempo zum Vergleich; „Wiederherstellen“ setzt alles zurück und lädt neu. Vor jedem Autotune und vor „Verwerfen“ sichert der Host von selbst („vor Autotune“). Stürzt das Modell nach einem Autotune innerhalb einer Stunde zweimal ab, stellt er diese Sicherung von selbst wieder her (die Karte sagt es). Es bleiben die letzten 6 Sicherungen, davon höchstens 3 automatische; gespeichert in `snapshots.json` auf dem Host.

**Vorausraten (Speculative Decoding) auf Hosts:** In der Host-Karte unter „Vorausraten“: **MTP** nutzt eingebaute Zusatzschichten des Modells (`--spec-type draft-mtp`, bis 2 Tokens voraus; hat das Modell keine, läuft es ohne und die Karte sagt es), **Entwurfsmodell** nimmt ein kleines Modell derselben Familie (auf dem PC gefunden oder einmal geladen, auf der GPU der Gruppe mit dem meisten Platz). Die Karte zeigt, was läuft; der Monitor zeigt je Anfrage die Trefferquote (Anteil der geratenen Tokens, die das Modell übernommen hat). Strata hat sein eigenes Vorausraten (DFlash2).

**Strata als Engine:** Ein Host kann statt llama.cpp die auf dem PC installierte [Strata](https://github.com/Niko1221/Strata)-Engine nutzen. Strata spricht dieselbe API wie llama-server, läuft an dessen Stelle auf `127.0.0.1` mit dem Schlüssel des Hosts, und Router, Bilderkennung und Monitor bleiben gleich. Am einfachsten aus dem Dashboard: Auf einem Host mit NVIDIA-GPU ohne Strata bietet die Host-Karte **„Strata installieren“** (Benutzer, Modell, Kontext, KV-Cache); der Root-Helfer startet dafür `myrmid strata install … --model … --yes` als eigene Unit (`myrmid-strata-install`), die Karte zeigt den Fortschritt aus `/var/log/myrmid/strata-install.log`, danach erscheint die Engine-Auswahl. Ist Strata eingebunden, verschwindet der Bereich. Auf der Kommandozeile, 
```
sudo myrmid strata install --user NAME     # klont Strata nach ~NAME/strata, startet dessen setup.sh als NAME
```
`setup.sh` fragt wie gewohnt nach Modell, Speicherort und Einstellungen und lädt das Modell (40 bis 90 GB); Strata startet danach nicht, sondern wird gleich eingebunden. Optionen für `setup.sh` nach `--`, z. B. `-- --family qwen --model Q2_0`. Ein weiteres Modell: den Befehl noch einmal ausführen (der Klon bleibt). Ist Strata schon da (Git-Klon mit `./setup.sh`), genügt:
```
sudo myrmid strata setup            # findet /home/*/strata; sonst --dir PFAD [--user NAME]
```
Beide Ordner-Aufteilungen von Strata gehen (alles im Klon, oder ab Strata 0.2.16 in dessen `terminal/`), auch wenn ein Update die Aufteilung ändert. Der offizielle Installer von Strata (`~/.local/share/strata`, `strata launch`) wird nicht eingebunden. Das schreibt `myrmid-strata.service` (läuft als der Benutzer, dem der Strata-Ordner gehört, nicht beim Booten), den Root-Helfer `myrmid-strata-request.path`/`.service` und eine Logrotation für `/var/log/myrmid/strata.log`. Ein vorhandener `strata.service` wird gestoppt und deaktiviert (dessen `Environment`/`EnvironmentFile`-Zeilen, z. B. `STRATA_UV=1` und `~/.strata.env`, übernimmt der neue Dienst; `--keep-old` lässt ihn in Ruhe). Danach in der Host-Karte: **Engine** „Strata“, **Strata-Konfiguration** (alle `strata-*.json` im Strata-Ordner), Name, Rolle, „Laden“. Der Host stoppt dabei llama-server und umgekehrt Strata, wenn wieder llama.cpp gewählt wird; Kontext und Bilder kommen aus Strata (die Karte zeigt sie). Bei mehreren NVIDIA-Karten wählt **GPUs für Strata**, welche Karten Strata nutzt (an Strata als `--gpu 0,2`; mehrere teilen sich die Schichten). Vorbelegt ist, was die Strata-Konfiguration nennt; keine angehakt heißt ebenfalls: wie dort. Strata empfiehlt für geteilte Modelle mindestens 8 GB je Karte, die Karte markiert kleinere. „Strata aktualisieren“ führt `update.sh` (bzw. `setup.sh --update`) als dieser Benutzer aus (läuft Strata, wird es dafür gestoppt und wieder gestartet), Protokoll in `/var/log/myrmid/strata-update.log`. **Strata-Einstellungen** (Kontext, KV-Cache int8/q4_0/k8v4/fp16): Strata legt beides in der Konfiguration fest, nicht beim Start; „Übernehmen“ führt als dieser Benutzer `setup.sh --setup --no-start --yes --model … --context … --kv …` aus (ohne neuen Download, mit den gewählten GPUs), Strata stoppt dafür und startet danach neu. Protokoll ebenfalls in `strata-update.log`. Autotune gibt es nur für llama.cpp. Bei „Ein Modell über alle GPUs“ wählt man die Karten für Strata frei; bei „Jede GPU ein eigenes Modell“ oder „GPU-Gruppen“ kann eine GPU bzw. Gruppe Strata nutzen (genau deren Karten), die anderen laden daneben llama.cpp-Modelle. Je PC läuft eine Strata-Instanz. Strata legt viel ins RAM (Experten): neben weiteren Modellen auf das RAM achten. `myrmid strata` zeigt den Stand.

**Wartungsmodus:** Ein Knopf in jeder Host-Karte nimmt den PC aus dem Routing (bei mehreren GPUs alle seine Hosts). Der Router schickt ihm nichts, weckt ihn nicht und legt ihn nicht schlafen; „Aufwecken“ von Hand geht weiter. Gedacht für PCs, die länger aus sind oder an denen gearbeitet wird. Die Einstellung bleibt über Neustarts des Masters erhalten.

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
