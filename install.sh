#!/usr/bin/env bash
# MyrmidLM: Installation eines PCs als Master (Server), Worker oder Host.
#
# Von überall mit Internet:
#
#   curl -fsSL https://raw.githubusercontent.com/Miki84/MyrmidLM/main/install.sh | bash
#   curl -fsSL https://raw.githubusercontent.com/Miki84/MyrmidLM/main/install.sh | bash -s -- --role worker
#
# Oder aus einem geklonten Repo: ./install.sh. Als normaler Benutzer mit
# sudo-Rechten starten. Ein PC mit der alten Installation (privatelm-node)
# wird übernommen: Schlüssel, Messwerte, Modelle und Einstellungen bleiben.
#
# Optionen (alle optional, sonst wird gefragt):
#   --role master|worker|host  Rolle dieses PCs
#   --model DATEI|auto|none  Master: GGUF-Modell (auto: passendes wählen/laden;
#                            none: nur Router, z. B. als VM ohne GPU, Modelle auf Hosts)
#   --workers MODUS          Master: auto (Standard) | all | off | IP:Port,IP:Port
#   --master-ip IP[,IP]      Worker/Host: Ports nur für den Master öffnen (alle seine Adressen)
#   --cluster-key KEY        Worker/Host: API-Key des Masters (dort: myrmid key)
#   --draft auto|off|DATEI   Master: Entwurfsmodell für mehr Tempo (Standard: fragen)
#   --host-sleep MIN         Master: Hosts nach MIN Minuten ohne Anfrage schlafen legen
#   --port N                 Master: API-Port (Standard 8080)
#   --fast-model DATEI|auto  Master: kleines Zweitmodell für einfache Anfragen (Router)
#   --backends LISTE         z. B. cuda,vulkan (Standard: automatisch)
#   --version X.Y.Z          diese Version statt der neuesten
#   --channel stable|main    Updates aus Releases (Standard) oder jedem Stand von main
#   --no-auto-update         neue Versionen bauen, aber nur auf Klick im Dashboard umstellen
#   --repo URL               anderes Repo (Fork, Spiegel)
#   --src DIR                lokalen Klon als Quelle nehmen (ohne Download)
#   --no-service             keinen systemd-Dienst einrichten
#   --yes                    Standardantworten ohne Rückfrage
#   --dry-run                nur anzeigen, was passieren würde
#
# Das Skript kann gefahrlos erneut gestartet werden (z. B. nach einem
# Neustart für den NVIDIA-Treiber): Erledigte Schritte werden übersprungen.
set -euo pipefail

REPO_URL="https://github.com/Miki84/MyrmidLM.git"
PREFIX="/opt/myrmid"
SRC_DIR="$PREFIX/src"
CONF=/etc/myrmid/myrmid.conf
OLD_PREFIX="/opt/privatelm-node"  # before the rename

ROUTER_ONLY=0 ROLE="" MODEL="" WORKERS="" PORT="8080" BACKENDS="" MASTER_IP="" FAST_MODEL="-" CLUSTER_KEY=""
DRAFT="" HOST_SLEEP="" WANT_VERSION="" CHANNEL="" AUTO_UPDATE="on" LOCAL_SRC=""
SERVICE=1 YES=0 DRY=0

ORIG_ARGS=("$@")  # to show the exact command for the rerun after a reboot
while [[ $# -gt 0 ]]; do
    case "$1" in
        --role) ROLE="$2"; shift 2 ;;
        --version) WANT_VERSION="${2#v}"; shift 2 ;;
        --channel) CHANNEL="$2"; shift 2 ;;
        --no-auto-update) AUTO_UPDATE="off"; shift ;;
        --repo) REPO_URL="$2"; shift 2 ;;
        --src) LOCAL_SRC="$(cd "$2" && pwd)"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --master-ip) MASTER_IP="$2"; shift 2 ;;
        --cluster-key) CLUSTER_KEY="$2"; shift 2 ;;
        --draft) DRAFT="$2"; shift 2 ;;
        --host-sleep) HOST_SLEEP="$2"; shift 2 ;;
        --fast-model) FAST_MODEL="$2"; shift 2 ;;
        --backends) BACKENDS="$2"; shift 2 ;;
        --no-service) SERVICE=0; shift ;;
        --yes|-y) YES=1; shift ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) sed -n '2,34p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unbekannte Option: $1 (--help zeigt alle)" >&2; exit 2 ;;
    esac
done

# ── Hilfsfunktionen ──────────────────────────────────────────────────────────

if [[ -t 1 ]]; then B=$'\e[1m' G=$'\e[32m' Y=$'\e[33m' R=$'\e[31m' N=$'\e[0m'; else B="" G="" Y="" R="" N=""; fi
step() { echo; echo "${B}==> $*${N}"; }
info() { echo "    $*"; }
ok()   { echo "    ${G}✓${N} $*"; }
warn() { echo "    ${Y}!${N} $*" >&2; }
die()  { echo; echo "${R}Fehler:${N} $*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

SUDO=""
[[ $EUID -ne 0 ]] && SUDO="sudo"

# Rückfragen über das Terminal, auch wenn das Skript per Pipe kommt.
TTY_OK=0
if [[ $YES -eq 0 ]] && { : </dev/tty; } 2>/dev/null; then TTY_OK=1; fi

# Führt einen Befehl aus (bei --dry-run nur anzeigen).
run() {
    echo "    ${B}\$${N} $*"
    [[ $DRY -eq 1 ]] && return 0
    "$@"
}
srun() { run $SUDO "$@"; }

# ask "Frage" "Standard" -> Antwort auf stdout
ask() {
    local q="$1" def="${2:-}" a
    if [[ $TTY_OK -eq 0 ]]; then echo "$def"; return; fi
    read -r -p "    $q${def:+ [$def]}: " a </dev/tty
    echo "${a:-$def}"
}

# ask_yn "Frage" j|n -> Rückgabewert 0 = ja
ask_yn() {
    local q="$1" def="${2:-j}" a hint="[J/n]"
    [[ $def == n ]] && hint="[j/N]"
    if [[ $TTY_OK -eq 0 ]]; then a="$def"; else read -r -p "    $q $hint " a </dev/tty; fi
    a="${a:-$def}"
    [[ ${a,,} == j* || ${a,,} == y* ]]
}

# ── 1. Rolle ─────────────────────────────────────────────────────────────────

echo "${B}MyrmidLM – Installation${N}"
echo "Repo: $REPO_URL · Installationsziel: $PREFIX"

# ── Alte Installation (PrivateLM-Node) ───────────────────────────────────────
# Its settings become the defaults: same role, same model and options.
OLD_ROLE="" OLD_ARGS=""
old_unit_args() {  # old_unit_args NAME -> ExecStart arguments without the program
    local f="/etc/systemd/system/$1.service"
    [[ -f "$f" ]] || return 1
    sed -n 's|^ExecStart=[^ ]*privatelm-node ||p' "$f" | head -1
}
if [[ -d "$OLD_PREFIX" ]] || ls /etc/systemd/system/privatelm-*.service >/dev/null 2>&1; then
    for r in server worker host; do
        if systemctl is-enabled --quiet "privatelm-$r" 2>/dev/null; then
            OLD_ROLE="$r"; OLD_ARGS="$(old_unit_args "privatelm-$r" || true)"; break
        fi
    done
    [[ "$OLD_ROLE" == server ]] && OLD_ROLE=master
    [[ -z "$ROLE" && -n "$OLD_ROLE" ]] && ROLE="$OLD_ROLE"
    if [[ -z "$CLUSTER_KEY" ]] && $SUDO test -s /etc/privatelm-node/cluster_key 2>/dev/null; then
        CLUSTER_KEY="$($SUDO cat /etc/privatelm-node/cluster_key)"
    fi
fi

if [[ -z "$ROLE" ]]; then
    echo
    echo "    Welche Rolle hat dieser PC?"
    echo "      1) Master  – OpenAI-API, Dashboard und Router; mit eigenem Modell oder nur als Router (auch in einer VM)"
    echo "      2) Worker  – leiht GPU und RAM an den Master (für ein Modell, das auf mehrere PCs verteilt wird)"
    echo "      3) Host    – lädt ein ganzes Modell auf die eigene GPU; der Router des Masters schickt Anfragen hin"
    case "$(ask "Auswahl 1, 2 oder 3" 2)" in
        1|m|M|master) ROLE=master ;;
        2|w|W|worker) ROLE=worker ;;
        3|h|H|host) ROLE=host ;;
        *) die "Bitte 1 (Master), 2 (Worker) oder 3 (Host) wählen." ;;
    esac
fi
[[ "$ROLE" == master || "$ROLE" == worker || "$ROLE" == host ]] || die "--role muss master, worker oder host sein."
ok "Rolle: $ROLE"

# ── 2. System erkennen ───────────────────────────────────────────────────────

step "System erkennen"
[[ -r /etc/os-release ]] || die "/etc/os-release fehlt, Distribution unbekannt."
# shellcheck disable=SC1091
. /etc/os-release
DISTRO="${ID:-unknown}" DISTRO_VER="${VERSION_ID:-}"
if have apt-get; then PM=apt
elif have dnf; then PM=dnf
elif have pacman; then PM=pacman
else die "Kein unterstützter Paketmanager (apt, dnf oder pacman) gefunden."; fi
ok "${PRETTY_NAME:-$DISTRO} (Paketmanager: $PM)"

gpu_vendors() {
    # PCI display devices (VGA 0x0300, 3D 0x0302 as in Optimus laptops,
    # display 0x0380): found even when no driver is loaded for them.
    local d
    for d in /sys/bus/pci/devices/*; do
        [[ -r "$d/class" && -r "$d/vendor" ]] || continue
        case "$(cat "$d/class")" in 0x0300*|0x0302*|0x0380*) ;; *) continue ;; esac
        case "$(cat "$d/vendor")" in
            0x10de) echo nvidia ;;
            0x1002) echo amd ;;
            0x8086) echo intel ;;
        esac
    done | sort -u | paste -sd' ' -
}
VENDORS="$(gpu_vendors || true)"
if [[ -n "$VENDORS" ]]; then ok "GPUs: $VENDORS"; else warn "Keine GPU gefunden, es wird nur die CPU genutzt."; fi
has_vendor() { [[ " $VENDORS " == *" $1 "* ]]; }

# apt_has PACKAGE: installable here? (apt-cache show also succeeds for a name
# that other packages only refer to, e.g. nvidia-vulkan-icd on Ubuntu)
apt_has() {
    [[ "$(apt-cache policy "$1" 2>/dev/null | awk '/Candidate:/ {print $2; exit}')" =~ ^[0-9] ]]
}

pkg_install() {
    case "$PM" in
        apt) srun apt-get install -y "$@" ;;
        dnf) srun dnf install -y "$@" ;;
        pacman) srun pacman -S --needed --noconfirm "$@" ;;
    esac
}

# ── 3. Grundpakete ───────────────────────────────────────────────────────────

step "Grundpakete installieren"
case "$PM" in
    apt)
        srun apt-get update
        # Ubuntu server images may come without universe (glslc, zeroconf) and multiverse (CUDA)
        if [[ "$DISTRO" == ubuntu ]] && ! apt_has glslc; then
            info "Aktiviere die Ubuntu-Paketquellen universe und multiverse ..."
            pkg_install software-properties-common
            srun add-apt-repository -y universe
            srun add-apt-repository -y multiverse
            srun apt-get update
        fi
        pkg_install build-essential cmake git python3 python3-zeroconf qrencode pciutils curl ca-certificates ethtool ;;
    dnf)
        pkg_install gcc-c++ make cmake git python3 python3-zeroconf qrencode pciutils curl ethtool ;;
    pacman)
        pkg_install base-devel cmake git python python-zeroconf qrencode pciutils curl ethtool ;;
esac

# ── NVIDIA: Generation, Treiber, CUDA ────────────────────────────────────────

# Oldest NVIDIA generation in this PC, from the chip names lspci shows
# (GM108M = Maxwell). It decides the driver branch and the CUDA version.
nvidia_generation() {
    local chips c gen=modern
    chips="$(lspci -d 10de: 2>/dev/null | grep -E 'VGA|3D|Display' | grep -oE '\b(GF|GK|GM|GP|GV)[0-9]+' || true)"
    for c in $chips; do
        case "$c" in
            GF*) gen=fermi ;;
            GK*) [[ $gen != fermi ]] && gen=kepler ;;
            GM*) [[ $gen == modern || $gen == volta || $gen == pascal ]] && gen=maxwell ;;
            GP*) [[ $gen == modern || $gen == volta ]] && gen=pascal ;;
            GV*) [[ $gen == modern ]] && gen=volta ;;
        esac
    done
    echo "$gen"
}

# Debian: the NVIDIA driver is in contrib/non-free. Adds them to every
# Debian source (sources.list and the newer .sources files), with a backup.
debian_enable_nonfree() {
    local f changed=0
    for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.list; do
        [[ -f "$f" ]] || continue
        if grep -qE '^deb .*debian.* main' "$f" && ! grep -qE '^deb .*debian.* main.*non-free( |$)' "$f"; then
            srun cp "$f" "$f.privatelm-bak"
            srun sed -i -E '/^deb(-src)? .*debian.* main/ { / contrib( |$)/! s/$/ contrib/; / non-free( |$)/! s/$/ non-free/; / non-free-firmware( |$)/! s/$/ non-free-firmware/ }' "$f"
            changed=1
        fi
    done
    for f in /etc/apt/sources.list.d/*.sources; do
        [[ -f "$f" ]] || continue
        grep -qiE '^URIs:.*debian' "$f" || continue  # only Debian's own sources, not third-party ones
        if grep -qE '^Components:.*main' "$f" && ! grep -qE '^Components:.*non-free( |$)' "$f"; then
            srun cp "$f" "$f.privatelm-bak"
            srun sed -i -E '/^Components:/ { / contrib( |$)/! s/$/ contrib/; / non-free( |$)/! s/$/ non-free/; / non-free-firmware( |$)/! s/$/ non-free-firmware/ }' "$f"
            changed=1
        fi
    done
    if [[ $changed -eq 1 ]]; then
        ok "Paketquellen um contrib/non-free ergänzt (Sicherung: *.privatelm-bak)."
    fi
    srun apt-get update
}

fedora_enable_rpmfusion() {
    rpm -q rpmfusion-nonfree-release >/dev/null 2>&1 && return 0
    local v
    v="$(rpm -E %fedora)"
    pkg_install "https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$v.noarch.rpm" \
                "https://mirrors.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$v.noarch.rpm"
    ok "RPM Fusion eingerichtet."
}

# The driver runs, but its tool nvidia-smi is missing (Debian installs it only
# as a recommendation, minimal setups skip those): install just the tool.
install_nvidia_smi() {
    local ver=""
    [[ -r /proc/driver/nvidia/version ]] && ver="$(grep -o 'Kernel Module[^0-9]*[0-9]*' /proc/driver/nvidia/version | grep -o '[0-9]*$' || true)"
    case "$DISTRO" in
        ubuntu|pop|linuxmint|elementary|zorin)
            if [[ -n "$ver" ]] && apt_has "nvidia-utils-$ver-server" && dpkg -l | grep -q "nvidia-.*-$ver-server"; then
                pkg_install "nvidia-utils-$ver-server"
            elif [[ -n "$ver" ]]; then
                pkg_install "nvidia-utils-$ver"
            else
                return 1
            fi ;;
        debian) pkg_install nvidia-smi ;;
        fedora) pkg_install xorg-x11-drv-nvidia-cuda ;;
        arch|endeavouros|manjaro|cachyos) pkg_install nvidia-utils ;;
        *) return 1 ;;
    esac
}

install_nvidia_driver() {  # install_nvidia_driver GENERATION
    local gen="$1" helper=""
    case "$DISTRO" in
        ubuntu|pop|linuxmint|elementary|zorin)
            # picks the right branch itself, legacy cards included
            pkg_install ubuntu-drivers-common
            srun ubuntu-drivers install ;;
        debian)
            debian_enable_nonfree
            if [[ "$gen" == kepler ]]; then
                apt_has nvidia-tesla-470-driver || \
                    die "Für Kepler-Karten hat ${PRETTY_NAME:-Debian} keinen Treiber mehr (470 gibt es nur bis Debian 12)."
                pkg_install "linux-headers-$(uname -r)" nvidia-tesla-470-driver firmware-misc-nonfree
            else
                # nvidia-smi is only "recommended" there: without it the dashboard has no GPU values
                pkg_install "linux-headers-$(uname -r)" nvidia-driver nvidia-smi firmware-misc-nonfree
                apt_has nvidia-vulkan-icd && pkg_install nvidia-vulkan-icd
            fi ;;
        fedora)
            fedora_enable_rpmfusion
            pkg_install "kernel-devel-$(uname -r)" || pkg_install kernel-devel
            case "$gen" in
                kepler) pkg_install akmod-nvidia-470xx xorg-x11-drv-nvidia-470xx-cuda ;;
                maxwell|pascal|volta)
                    if dnf info akmod-nvidia-580xx >/dev/null 2>&1; then
                        pkg_install akmod-nvidia-580xx xorg-x11-drv-nvidia-580xx-cuda
                    else
                        pkg_install akmod-nvidia xorg-x11-drv-nvidia-cuda
                    fi ;;
                *) pkg_install akmod-nvidia xorg-x11-drv-nvidia-cuda ;;
            esac
            info "Kernel-Modul wird gebaut (einige Minuten)..."
            srun akmods --force || true ;;
        arch|endeavouros|manjaro|cachyos)
            if [[ "$gen" == modern ]]; then
                pkg_install nvidia-open-dkms nvidia-utils linux-headers
            else
                # Arch moved older cards to the AUR (nvidia-580xx / nvidia-470xx).
                local pkg=nvidia-580xx-dkms
                [[ "$gen" == kepler ]] && pkg=nvidia-470xx-dkms
                for h in yay paru; do have "$h" && helper="$h" && break; done
                if [[ -n "$helper" ]]; then
                    run "$helper" -S --needed --noconfirm "$pkg" "${pkg%-dkms}-utils"
                else
                    die "Ältere NVIDIA-Karten brauchen unter Arch $pkg aus dem AUR (z. B. mit yay oder paru).
       Bitte installieren, neu starten und das Skript erneut starten."
                fi
            fi ;;
        *)
            die "Für ${PRETTY_NAME:-$DISTRO} kann das Skript den NVIDIA-Treiber nicht selbst installieren.
       Bitte den Treiber nach der Anleitung der Distribution installieren, neu starten, Skript erneut starten." ;;
    esac
}

# Secure Boot: self-built kernel modules must be signed with a key the
# firmware trusts. Debian and Fedora create one; enroll it here, confirm it
# once in the blue MOK screen at the next boot.
secure_boot_hint() {
    have mokutil || return 0
    mokutil --sb-state 2>/dev/null | grep -qi "enabled" || return 0
    local key=""
    for k in /var/lib/dkms/mok.pub /etc/pki/akmods/certs/public_key.der /var/lib/shim-signed/mok/MOK.der; do
        [[ -f "$k" ]] && key="$k" && break
    done
    echo
    warn "Secure Boot ist an: Das NVIDIA-Modul muss mit einem eigenen Schlüssel signiert sein."
    if [[ -n "$key" ]] && ! mokutil --test-key "$key" 2>/dev/null | grep -q "already enrolled"; then
        info "Gleich wird ein Einmal-Passwort abgefragt. Beim Neustart erscheint ein blauer Bildschirm (MOK):"
        info 'Dort nacheinander: Enroll MOK, Continue, Yes, dieses Passwort, Reboot.'
        srun mokutil --import "$key" || warn "Schlüssel nicht eingetragen; ohne ihn lädt der Treiber nicht."
    else
        info "Ubuntu fragt dabei selbst nach einem Passwort; beim Neustart im blauen MOK-Bildschirm bestätigen."
    fi
}

# CUDA toolkit from NVIDIA's repo (apt, dnf) or the distribution (pacman).
install_cuda() {  # install_cuda PACKAGE DEFAULT_ANSWER
    local pkg="$1" def="$2" repo="" tmp
    case "$DISTRO:$DISTRO_VER" in
        ubuntu:26.04*) repo=ubuntu2604 ;;
        ubuntu:24.04*) repo=ubuntu2404 ;;
        ubuntu:22.04*) repo=ubuntu2204 ;;
        debian:12*) repo=debian12 ;;
        debian:13*) repo=debian13 ;;
        fedora:*) repo="fedora${DISTRO_VER%%.*}" ;;
    esac
    case "${UBUNTU_CODENAME:-}" in  # Mint, Pop!_OS & Co.: the Ubuntu they are built on
        resolute) repo="${repo:-ubuntu2604}" ;;
        noble) repo="${repo:-ubuntu2404}" ;;
        jammy) repo="${repo:-ubuntu2204}" ;;
    esac
    if [[ "$PM" == pacman ]]; then
        [[ "$pkg" == cuda-toolkit ]] || { info "Arch hat nur das aktuelle CUDA, das diese Karte nicht mehr kennt."; return 1; }
        ask_yn "CUDA-Toolkit installieren (ca. 4 GB, schnellste Variante für NVIDIA)?" "$def" || return 1
        pkg_install cuda
        CUDA_BIN=/opt/cuda/bin
        return 0
    fi
    if [[ -z "$repo" ]] || ! curl -fsIL "https://developer.download.nvidia.com/compute/cuda/repos/$repo/x86_64/cuda-keyring_1.1-1_all.deb" >/dev/null 2>&1; then
        # Ubuntu 26.04 and later package CUDA themselves (multiverse)
        if [[ "$PM" == apt && "$DISTRO" == ubuntu && "$pkg" == cuda-toolkit ]] && apt_has cuda-toolkit; then
            ask_yn "CUDA-Toolkit aus den Ubuntu-Paketen installieren (ca. 4 GB, schnellste Variante)?" "$def" || return 1
            pkg_install cuda-toolkit
            find_cuda_bin
            [[ -n "$CUDA_BIN" ]] || { warn "CUDA installiert, aber nvcc nicht gefunden."; return 1; }
            return 0
        fi
        info "Für ${PRETTY_NAME:-$DISTRO} hat NVIDIA kein CUDA-Paketarchiv."
        return 1
    fi
    ask_yn "CUDA-Toolkit${pkg#cuda-toolkit} aus dem NVIDIA-Repo installieren (ca. 4 GB, schnellste Variante)?" "$def" || return 1
    if [[ "$PM" == apt ]]; then
        tmp="$(mktemp -d)"
        run curl -fsSL -o "$tmp/cuda-keyring.deb" \
            "https://developer.download.nvidia.com/compute/cuda/repos/$repo/x86_64/cuda-keyring_1.1-1_all.deb"
        srun dpkg -i "$tmp/cuda-keyring.deb"
        srun apt-get update
        apt_has "$pkg" || { warn "Paket $pkg gibt es für $repo nicht."; return 1; }
    else
        srun dnf config-manager addrepo --from-repofile="https://developer.download.nvidia.com/compute/cuda/repos/$repo/x86_64/cuda-$repo.repo" 2>/dev/null || \
            srun dnf config-manager --add-repo "https://developer.download.nvidia.com/compute/cuda/repos/$repo/x86_64/cuda-$repo.repo"
    fi
    pkg_install "$pkg"
    CUDA_BIN=/usr/local/cuda/bin
    for d in /usr/local/cuda-12.9/bin /usr/local/cuda/bin; do [[ -x "$d/nvcc" ]] && CUDA_BIN="$d" && break; done
    return 0
}

find_cuda_bin() {  # sets CUDA_BIN to the newest nvcc found
    local d
    CUDA_BIN=""
    for d in /usr/local/cuda/bin $(ls -d /usr/local/cuda-*/bin 2>/dev/null | sort -V) /opt/cuda/bin /usr/lib/cuda/bin; do
        [[ -x "$d/nvcc" ]] && CUDA_BIN="$d"
    done
    have nvcc && [[ -z "$CUDA_BIN" ]] && CUDA_BIN="$(dirname "$(command -v nvcc)")"
    return 0
}

# ── 4. GPU-Treiber, CUDA, Vulkan ─────────────────────────────────────────────

USE_CUDA=0 USE_VULKAN=0
CUDA_BIN=""
find_cuda_bin

if has_vendor nvidia; then
    step "NVIDIA"
    NV_GEN="$(nvidia_generation)"
    NV_NAMES="$(lspci -d 10de: 2>/dev/null | grep -E 'VGA|3D|Display' | sed -E 's/.*: NVIDIA Corporation //' | paste -sd';' - || true)"
    info "Karte(n): ${NV_NAMES:-unbekannt} (Generation: $NV_GEN)"
    case "$NV_GEN" in
        fermi) warn "Fermi-Karten unterstützt kein aktueller NVIDIA-Treiber mehr; die Karte bleibt ungenutzt." ;;
        kepler) info "Kepler-Karte: nur der Legacy-Treiber 470, kein aktuelles CUDA. Genutzt wird sie über Vulkan." ;;
        maxwell|pascal|volta) info "Ältere Karte: Treiber bis Version 580, CUDA höchstens 12.x (CUDA 13 kennt sie nicht mehr)." ;;
    esac
    if [[ "$NV_GEN" != fermi ]] && have nvidia-smi && nvidia-smi 2>&1 | grep -q "version mismatch"; then
        warn "Der NVIDIA-Treiber wurde gerade aktualisiert, läuft aber noch in der alten Version (Driver/library version mismatch)."
        echo "    ${B}Ein Neustart ist nötig. Danach das Skript erneut starten:${N}"
        if [[ -f "${BASH_SOURCE[0]:-}" ]]; then echo "    bash $(realpath "${BASH_SOURCE[0]}") ${ORIG_ARGS[*]}"
        else echo "    curl -fsSL https://raw.githubusercontent.com/Miki84/MyrmidLM/main/install.sh | bash -s -- ${ORIG_ARGS[*]}"; fi
        if ask_yn "Jetzt neu starten?" j; then srun systemctl reboot; fi
        exit 0
    fi
    if [[ "$NV_GEN" != fermi && -r /proc/driver/nvidia/version ]] && ! have nvidia-smi; then
        info "Der NVIDIA-Treiber läuft, aber nvidia-smi fehlt (das Dashboard zeigt sonst keine GPU-Werte). Installiere es ..."
        install_nvidia_smi || warn "nvidia-smi konnte nicht installiert werden; das Modell läuft trotzdem, nur ohne GPU-Werte im Dashboard."
    fi
    if [[ "$NV_GEN" != fermi ]] && { ! have nvidia-smi || ! nvidia-smi >/dev/null 2>&1; } && [[ ! -r /proc/driver/nvidia/version ]]; then
        warn "Kein funktionierender NVIDIA-Treiber installiert (nur der freie nouveau-Treiber, den llama.cpp nicht nutzt)."
        if ask_yn "NVIDIA-Treiber jetzt installieren? Danach ist ein Neustart nötig." j; then
            install_nvidia_driver "$NV_GEN"
            secure_boot_hint
            echo
            echo "    ${B}Treiber installiert. Nach dem Neustart das Skript erneut starten:${N}"
            if [[ -f "${BASH_SOURCE[0]:-}" ]]; then echo "    bash $(realpath "${BASH_SOURCE[0]}") ${ORIG_ARGS[*]}"
            else echo "    curl -fsSL https://raw.githubusercontent.com/Miki84/MyrmidLM/main/install.sh | bash -s -- ${ORIG_ARGS[*]}"; fi
            if ask_yn "Jetzt neu starten?" j; then srun systemctl reboot; fi
            exit 0
        fi
        warn "Ohne NVIDIA-Treiber wird die NVIDIA-Karte nicht genutzt."
    elif [[ "$NV_GEN" != fermi ]]; then
        if have nvidia-smi; then ok "Treiber: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | paste -sd';' -)"
        else ok "Treiber: $(head -1 /proc/driver/nvidia/version | grep -o 'Kernel Module.*' | cut -c1-60)"; fi
        if [[ -n "$CUDA_BIN" ]]; then
            ok "CUDA-Toolkit: $("$CUDA_BIN/nvcc" --version | grep -o 'release [0-9.]*')"
            USE_CUDA=1
        elif [[ "$NV_GEN" == kepler ]]; then
            USE_VULKAN=1
        else
            # CUDA 13 needs Turing or newer; older cards get the last CUDA 12.
            cuda_pkg=cuda-toolkit
            [[ "$NV_GEN" == maxwell || "$NV_GEN" == pascal || "$NV_GEN" == volta ]] && cuda_pkg=cuda-toolkit-12-9
            vram_mb="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1 || echo 0)"
            def=j
            if [[ "$cuda_pkg" != cuda-toolkit ]] && (( ${vram_mb:-0} < 4096 )); then
                info "Die Karte hat nur ${vram_mb} MB und ist älter: CUDA bringt hier wenig gegenüber Vulkan (ca. 4 GB Download)."
                def=n
            fi
            if install_cuda "$cuda_pkg" "$def"; then USE_CUDA=1
            else
                warn "Ohne CUDA wird die NVIDIA-Karte über Vulkan genutzt (langsamer, aber funktioniert)."
                USE_VULKAN=1
            fi
        fi
    fi
fi
if has_vendor amd || has_vendor intel; then USE_VULKAN=1; fi

if [[ $USE_VULKAN -eq 1 ]]; then
    step "Vulkan (AMD, Intel$( [[ $USE_CUDA -eq 0 ]] && has_vendor nvidia && echo ", NVIDIA ohne CUDA"))"
    case "$PM" in
        apt)
            if ! apt_has glslc; then
                die "Paket glslc fehlt in ${PRETTY_NAME:-dieser Version} (z. B. Ubuntu 22.04).
       Lösung: Vulkan SDK von LunarG installieren (vulkan.lunarg.com) oder Ubuntu 24.04 verwenden.
       Danach das Skript erneut starten."
            fi
            pkg_install libvulkan-dev glslc spirv-headers mesa-vulkan-drivers vulkan-tools
            # NVIDIA without CUDA runs on Vulkan: Debian packs its Vulkan driver separately
            # (Ubuntu's driver packages bring it along).
            if has_vendor nvidia && have nvidia-smi && apt_has nvidia-vulkan-icd; then
                pkg_install nvidia-vulkan-icd
            fi ;;
        dnf) pkg_install vulkan-loader-devel glslc spirv-headers-devel mesa-vulkan-drivers vulkan-tools ;;
        pacman)
            pk=(vulkan-icd-loader vulkan-headers shaderc spirv-headers vulkan-tools)
            has_vendor amd && pk+=(vulkan-radeon)
            has_vendor intel && pk+=(vulkan-intel)
            pkg_install "${pk[@]}" ;;
    esac
    # GPU-Zugriff für den eigenen Benutzer (der Dienst bekommt ihn selbst).
    if [[ $EUID -ne 0 ]] && ! id -nG | grep -qw render; then
        srun usermod -aG render,video "$(id -un)" || true
        info "Benutzer zu den Gruppen render/video hinzugefügt (wirkt nach neuer Anmeldung)."
    fi
fi

if [[ -z "$BACKENDS" ]]; then
    b=()
    [[ $USE_CUDA -eq 1 ]] && b+=(cuda)
    [[ $USE_VULKAN -eq 1 ]] && b+=(vulkan)
    [[ ${#b[@]} -eq 0 ]] && b=(cpu)
    BACKENDS="$(IFS=,; echo "${b[*]}")"
fi
ok "Backends: $BACKENDS"

# ── 5. Quellcode ─────────────────────────────────────────────────────────────

step "Quellcode"
CHANNEL="${CHANNEL:-$(sed -n 's/^CHANNEL=//p' "$CONF" 2>/dev/null || true)}"
CHANNEL="${CHANNEL:-stable}"
[[ "$CHANNEL" == stable || "$CHANNEL" == main ]] || die "--channel muss stable oder main sein."
CONF_CHANNEL="$CHANNEL"  # without a release yet main is installed, but updates still follow releases
if [[ ! -d "$SRC_DIR/.git" ]]; then
    srun mkdir -p "$PREFIX"
    run $SUDO git clone -q "${LOCAL_SRC:-$REPO_URL}" "$SRC_DIR"
fi
[[ -n "$LOCAL_SRC" ]] && run $SUDO git -C "$SRC_DIR" remote set-url origin "$LOCAL_SRC"
if [[ $DRY -eq 0 ]]; then
    $SUDO git -C "$SRC_DIR" fetch -q --force --tags origin
    $SUDO git -C "$SRC_DIR" fetch -q --force origin +main:refs/remotes/origin/main
fi
if [[ -n "$WANT_VERSION" ]]; then
    REF="v$WANT_VERSION" VER="$WANT_VERSION"
elif [[ "$CHANNEL" == stable ]]; then
    VER="$($SUDO git -C "$SRC_DIR" tag -l 'v*' 2>/dev/null | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -1 | sed 's/^v//' || true)"
    REF="v$VER"
    if [[ -z "$VER" ]]; then
        warn "Das Repo hat noch kein Release (Tag vX.Y.Z): nehme den Stand von main; das erste Release kommt per Update."
        CHANNEL=main
    fi
fi
if [[ -z "${VER:-}" ]]; then
    REF=origin/main
    VER="main-$($SUDO git -C "$SRC_DIR" rev-parse --short=8 origin/main 2>/dev/null || echo unknown)"
fi
run $SUDO git -C "$SRC_DIR" checkout -q --force --detach "$REF"
# Updates kommen aus dem Repo (beim lokalen Klon: weiterhin aus ihm, bis --repo etwas anderes sagt).
[[ -z "$LOCAL_SRC" ]] && run $SUDO git -C "$SRC_DIR" remote set-url origin "$REPO_URL"
ok "Version $VER ($CHANNEL)"

# ── 6. Bauen ─────────────────────────────────────────────────────────────────

REL="$PREFIX/releases/$VER"
step "Bauen (CUDA dauert 15 bis 45 Minuten)"
[[ -n "$CUDA_BIN" ]] && export PATH="$CUDA_BIN:$PATH"
# Ein laufender Dienst hält das Modell im RAM; der Compiler braucht ihn jetzt.
# Bei wenig freiem Speicher Dienste anhalten, am Ende startet das Skript sie neu.
avail_mb=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo 2>/dev/null || echo 0)
if (( avail_mb < 6000 )) && have systemctl; then
    running=()
    for u in myrmid-server myrmid-host myrmid-worker privatelm-server privatelm-host privatelm-worker; do
        systemctl is-active --quiet "$u" 2>/dev/null && running+=("$u")
    done
    if (( ${#running[@]} )); then
        info "Nur ${avail_mb} MB frei: halte ${running[*]} für das Bauen an (startet am Ende wieder)."
        srun systemctl stop "${running[@]}"
        sleep 2
    fi
fi
if [[ -x "$REL/bin/myrmid" ]]; then
    ok "Version $VER ist schon gebaut."
else
    srun rm -rf "$REL.part"
    srun env PATH="$PATH" bash "$SRC_DIR/node/build.sh" --backends "$BACKENDS" --install "$REL.part"
    if [[ $DRY -eq 0 ]]; then
        echo "$VER" | $SUDO tee "$REL.part/bin/VERSION" >/dev/null
        $SUDO mv "$REL.part" "$REL"
    fi
fi
# Version umschalten (atomar) und den Befehl myrmid für alle bereitstellen.
srun ln -sfn "releases/$VER" "$PREFIX/.current.tmp"
srun mv -T "$PREFIX/.current.tmp" "$PREFIX/current"
srun ln -sfn "$PREFIX/current/bin/myrmid" /usr/local/bin/myrmid
NODE="$PREFIX/current/bin/myrmid"

# Einstellungen für die Updates: woher, welche Backends, automatisch umstellen.
if [[ $DRY -eq 0 ]]; then
    srun mkdir -p /etc/myrmid
    printf '%s\n' "# MyrmidLM (install.sh). Updates: myrmid update; ändern und speichern genügt." \
        "REPO=$REPO_URL" "CHANNEL=$CONF_CHANNEL" "AUTO=$AUTO_UPDATE" "BACKENDS=$BACKENDS" "CUDA_BIN=${CUDA_BIN:-}" \
        "$(sed -n 's/^LLAMA_CPP_REPO=/LLAMA_CPP_REPO=/p' "$CONF" 2>/dev/null || true)" \
        | sed '/^$/d' | $SUDO tee "$CONF" >/dev/null
fi

if [[ $DRY -eq 0 ]]; then
    step "Geräte prüfen"
    "$NODE" detect || warn "detect ist fehlgeschlagen, bitte Ausgabe prüfen."
fi

# ── Alte Installation übernehmen ─────────────────────────────────────────────

migrate_old() {
    local had=0 u d
    for u in privatelm-server privatelm-worker privatelm-host privatelm-suspend.path privatelm-role.path privatelm-wol; do
        if [[ -f "/etc/systemd/system/${u%.service}.service" || -f "/etc/systemd/system/$u" ]]; then had=1; fi
    done
    [[ $had -eq 1 || -d "$OLD_PREFIX" ]] || return 0
    step "Alte Installation (privatelm-node) übernehmen"
    for u in privatelm-server privatelm-worker privatelm-host privatelm-suspend.path privatelm-role.path privatelm-wol.service; do
        $SUDO systemctl disable --now "$u" >/dev/null 2>&1 || true
    done
    # Daten der Dienste (DynamicUser: /var/lib/private/NAME): API-Key, Messwerte, Host-Modelle.
    for d in /var/lib/private /var/cache/private /var/lib /var/cache; do
        if [[ -d "$d/privatelm-node" && ! -L "$d/privatelm-node" && ! -e "$d/myrmid" ]]; then
            srun mv "$d/privatelm-node" "$d/myrmid"
        fi
    done
    for d in /var/lib/privatelm-node /var/cache/privatelm-node; do
        [[ -L "$d" ]] && srun rm -f "$d"
    done
    if [[ -f /etc/privatelm-node/cluster_key && ! -f /etc/myrmid/cluster_key ]]; then
        srun mkdir -p /etc/myrmid
        srun mv /etc/privatelm-node/cluster_key /etc/myrmid/cluster_key
    fi
    srun rm -f /etc/systemd/system/privatelm-{server,worker,host,suspend,role,wol}.service \
               /etc/systemd/system/privatelm-{suspend,role}.path /usr/local/sbin/privatelm-role
    srun rmdir /etc/privatelm-node 2>/dev/null || true
    srun rm -rf "$OLD_PREFIX"
    srun systemctl daemon-reload
    ok "Übernommen: Dienste heißen jetzt myrmid-*, Daten liegen unter /var/lib/myrmid."
    info "Der alte Quellordner (z. B. ~/privatelm) wird nicht mehr gebraucht und kann weg."
}
[[ $DRY -eq 1 ]] || migrate_old

# Hilfsdienste aus der Version: Updates (alle), Schlaf und Rollenwechsel (Worker/Host).
install_static_units() {  # install_static_units NAME...
    local n
    for n in "$@"; do
        srun install -m 644 "$PREFIX/current/systemd/$n" "/etc/systemd/system/$n"
    done
    srun systemctl daemon-reload
}
if [[ $DRY -eq 0 ]]; then
    install_static_units myrmid-update.timer myrmid-update.service myrmid-update-request.path myrmid-update-request.service
    srun systemctl enable --now myrmid-update.timer myrmid-update-request.path
    ok "Updates: jede Nacht prüfen und bauen; umgestellt wird vom Master für alle zusammen ($( [[ $AUTO_UPDATE == on ]] && echo automatisch || echo auf Klick im Dashboard))."
fi

# ── 7. Netzwerk ──────────────────────────────────────────────────────────────

LAN_IP="$(ip -4 route get 1.1.1.1 2>/dev/null | grep -oP 'src \K[0-9.]+' || true)"
if [[ -z "$LAN_IP" ]]; then  # ohne iproute2
    LAN_IP="$(python3 -c "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('192.0.2.1', 9)); print(s.getsockname()[0])" 2>/dev/null || true)"
fi
LAN_NET=""
if [[ -n "$LAN_IP" ]]; then
    cidr="$(ip -o -4 addr show 2>/dev/null | awk -v ip="$LAN_IP" '{split($4,a,"/"); if (a[1]==ip) print $4}' | head -1 || true)"
    [[ -n "$cidr" ]] && LAN_NET="$(python3 -c "import ipaddress,sys; print(ipaddress.ip_interface(sys.argv[1]).network)" "$cidr")"
fi
LAN_NET="${LAN_NET:-192.168.0.0/16}"

open_ports() {  # open_ports QUELLE[,QUELLE...] "50052:50053/tcp" ...  (QUELLE: IP oder Netz)
    local src="$1" p s
    shift
    if have ufw && $SUDO ufw status 2>/dev/null | grep -q "Status: active"; then
        for s in ${src//,/ }; do
            for p in "$@"; do
                srun ufw allow from "$s" to any port "${p%/*}" proto "${p#*/}"
            done
        done
        srun ufw allow 5353/udp
        ok "ufw: Ports nur für $src geöffnet."
    elif have firewall-cmd && systemctl is-active --quiet firewalld; then
        for s in ${src//,/ }; do
            for p in "$@"; do
                srun firewall-cmd --permanent --add-rich-rule="rule family=ipv4 source address=$s port port=$(echo "${p%/*}" | tr : -) protocol=${p#*/} accept"
            done
        done
        srun firewall-cmd --permanent --add-service=mdns
        srun firewall-cmd --reload
        ok "firewalld: Ports nur für $src geöffnet."
    else
        info "Keine aktive Firewall (ufw/firewalld) gefunden. Empfehlung: Ports $(printf '%s ' "$@" | sed 's|/tcp||g; s| $||; s| |, |g') nur für $src freigeben."
    fi
}

# Every LAN address of the master (cable and Wi-Fi): it may switch between them,
# and the firewall must let all of them in. Asks the master's mDNS announcement.
find_master_addrs() {
    [[ -x "$NODE" ]] || return 0
    "$NODE" scan --json --timeout 4 2>/dev/null | python3 -c '
import json, sys
try:
    peers = json.load(sys.stdin)
except ValueError:
    sys.exit(0)
for p in peers:
    if p["kind"] == "server":
        addrs = [a for a in (p["txt"].get("addrs") or "").split(",") if a] or [p["host"]]
        name = p["name"].replace("MyrmidLM ", "").replace("PrivateLM ", "")
        print(name + " " + ",".join(dict.fromkeys(addrs)))
' || true
}

ask_master_ip() {  # sets MASTER_IP (one or more addresses, comma-separated; empty = whole LAN)
    [[ -n "$MASTER_IP" ]] && return 0
    local found name addrs
    found="$(find_master_addrs | head -1)"
    if [[ -n "$found" ]]; then
        name="${found%% *}" addrs="${found#* }"
        info "Master im Netz gefunden: $name ($addrs)"
        if ask_yn "Ports nur für diesen Master öffnen (alle seine Adressen, Kabel und WLAN)?" j; then
            MASTER_IP="$addrs"
            return 0
        fi
    fi
    MASTER_IP="$(ask "IP-Adresse(n) des Masters, mehrere mit Komma (leer = ganzes LAN $LAN_NET)" "")"
}

check_master_ip() {
    local a
    for a in ${MASTER_IP//,/ }; do
        python3 -c "import ipaddress,sys; ipaddress.ip_address(sys.argv[1])" "$a" 2>/dev/null || die "Keine gültige IP-Adresse: $a"
    done
}

# ── 8. Rolle einrichten ──────────────────────────────────────────────────────

write_unit() {  # write_unit NAME BESCHREIBUNG EXECSTART [ZUSÄTZLICHE SERVICE-ZEILEN] [on|off]
    local name="$1" desc="$2" exec="$3" more="${4:-}" active="${5:-on}" unit="/etc/systemd/system/$1.service"
    local content="[Unit]
Description=$desc
After=network-online.target avahi-daemon.service
Wants=network-online.target

[Service]
ExecStart=$exec
DynamicUser=yes
# Anfragen an die Root-Helfer (Schlaf, Rollenwechsel, Update)
RuntimeDirectory=myrmid
RuntimeDirectoryMode=0755
RuntimeDirectoryPreserve=yes
CacheDirectory=myrmid
Environment=XDG_CACHE_HOME=/var/cache/myrmid
StateDirectory=myrmid
Environment=XDG_CONFIG_HOME=/var/lib/myrmid
SupplementaryGroups=video render
Restart=on-failure
RestartSec=5
# Exit 2 = the model doesn't fit: restarting won't help
RestartPreventExitStatus=2${more:+
$more}

[Install]
WantedBy=multi-user.target"
    echo "    ${B}\$${N} schreibe $unit"
    if [[ $DRY -eq 0 ]]; then
        printf '%s\n' "$content" | $SUDO tee "$unit" >/dev/null
        srun systemctl daemon-reload
        if [[ "$active" == on ]]; then
            srun systemctl enable "$name"
            srun systemctl restart "$name"
        else  # set up, but the other role runs
            srun systemctl disable --now "$name" 2>/dev/null || true
        fi
    fi
}

# Wake-on-LAN: der Master weckt diesen PC, wenn er gebraucht wird (nur per Kabel).
enable_wol() {
    local ifc con
    ifc="$(ip -o -4 addr show 2>/dev/null | awk -v ip="${LAN_IP:-}" '$4 ~ "^"ip"/" {print $2; exit}' || true)"
    if [[ -z "$ifc" ]]; then
        warn "Wake-on-LAN: Netzwerkkarte nicht gefunden, übersprungen."
        return
    fi
    if [[ -d "/sys/class/net/$ifc/wireless" ]]; then
        warn "Wake-on-LAN geht nur per Kabel; $ifc ist WLAN. Der Master kann diesen PC nicht wecken."
        return
    fi
    if ! have ethtool || ! $SUDO ethtool "$ifc" 2>/dev/null | grep -q "Supports Wake-on:.*g"; then
        warn "Wake-on-LAN: $ifc unterstützt kein Magic Packet (oder ethtool fehlt)."
        return
    fi
    run $SUDO ethtool -s "$ifc" wol g
    con="$(nmcli -g GENERAL.CONNECTION device show "$ifc" 2>/dev/null || true)"
    if [[ -n "$con" ]]; then
        run $SUDO nmcli connection modify "$con" 802-3-ethernet.wake-on-lan magic
    else
        local unit=/etc/systemd/system/myrmid-wol.service
        if [[ $DRY -eq 0 ]]; then
            printf '%s\n' "[Unit]" "Description=Wake-on-LAN for MyrmidLM on $ifc" "After=network.target" "" \
                "[Service]" "Type=oneshot" "ExecStart=$(command -v ethtool) -s $ifc wol g" "" \
                "[Install]" "WantedBy=multi-user.target" | $SUDO tee "$unit" >/dev/null
            srun systemctl daemon-reload
            srun systemctl enable myrmid-wol.service
        fi
    fi
    ok "Wake-on-LAN an ($ifc). Falls der PC nicht aufwacht: im BIOS die Option Wake on LAN bzw. Power on by PCI-E einschalten."
}

# Worker und Host: Beide Dienste werden eingerichtet, einer läuft. Der Master
# kann im Dashboard umschalten: Der laufende Dienst legt die gewünschte Rolle
# in /run/myrmid/role ab, myrmid-role.path (root) schaltet um.
if [[ "$ROLE" == worker || "$ROLE" == host ]]; then
    step "$( [[ "$ROLE" == worker ]] && echo Worker || echo Host ) einrichten"
    info "Dieser PC bekommt beide Dienste: Worker (leiht GPU/RAM für ein verteiltes Modell) und Host"
    info "(lädt ein ganzes Modell). Einer läuft; umschalten im Dashboard des Masters."
    if [[ -z "$CLUSTER_KEY" ]] && $SUDO test -s /etc/myrmid/cluster_key 2>/dev/null; then
        CLUSTER_KEY="$($SUDO cat /etc/myrmid/cluster_key)"
        ok "API-Key des Masters ist schon hinterlegt."
    fi
    if [[ -z "$CLUSTER_KEY" ]]; then
        info "Den API-Key des Masters musst du nicht eingeben: Sobald der Master diesen PC im Netz sieht,"
        info "nimmt er ihn auf und gibt ihm seinen Schlüssel (danach kann er ihn umschalten und aktualisieren)."
    fi

    worker_args=(worker --cache)
    RAM_NOTE="" WHAT="GPU"
    if [[ "$OLD_ROLE" == worker && -n "$OLD_ARGS" ]] && ask_yn "Bisherige Worker-Einstellungen übernehmen ($OLD_ARGS)?" j; then
        read -ra worker_args <<< "$OLD_ARGS"
        [[ " $OLD_ARGS " == *" --ram "* ]] && RAM_NOTE=", RAM unter ${LAN_IP:-?}:50053"
    elif [[ "$BACKENDS" == cpu ]]; then
        WHAT="RAM/CPU"   # ohne GPU verleiht der Worker ohnehin seinen RAM
        reserve="$( [[ "$ROLE" == worker ]] && ask "GB RAM für diesen PC freihalten" 4 || echo 4)"
        worker_args+=(--ram-reserve "$reserve")
    elif [[ "$ROLE" == host ]] || ask_yn "Als Worker auch den Arbeitsspeicher (RAM) dieses PCs verleihen?" j; then
        reserve="$( [[ "$ROLE" == worker ]] && ask "GB RAM für diesen PC freihalten" 4 || echo 4)"
        worker_args+=(--ram --ram-reserve "$reserve")
        RAM_NOTE=", RAM unter ${LAN_IP:-?}:50053"
    fi
    host_args=(host --models-dir /var/lib/myrmid/models)

    # Die Ports braucht nur der Master (RPC hat keine Anmeldung).
    ask_master_ip
    check_master_ip
    open_ports "${MASTER_IP:-$LAN_NET}" "50052:50053/tcp" "50060/tcp" "50070/tcp" "8081/tcp"
    enable_wol
    # Ohne Key: nur dieser Master darf den PC aufnehmen (sonst jeder, in der ersten Stunde nach dem Start).
    if [[ -z "$CLUSTER_KEY" && -n "$MASTER_IP" ]]; then
        worker_args+=(--pair-from "$MASTER_IP")
        host_args+=(--pair-from "$MASTER_IP")
    fi

    cred=""
    if [[ -n "$CLUSTER_KEY" ]]; then
        if [[ $DRY -eq 0 ]]; then
            srun mkdir -p /etc/myrmid
            printf '%s\n' "$CLUSTER_KEY" | $SUDO tee /etc/myrmid/cluster_key >/dev/null
            srun chmod 600 /etc/myrmid/cluster_key
        fi
        ok "Schlüssel in /etc/myrmid/cluster_key (nur root lesbar, die Dienste bekommen ihn über systemd)."
        cred="LoadCredential=cluster_key:/etc/myrmid/cluster_key"
    fi

    if [[ $DRY -eq 0 ]]; then
        # Standby auf Wunsch des Masters (Host) und Rollenwechsel: kleine Root-Dienste,
        # ausgelöst durch eine Datei, die der Dienst (ohne Root-Rechte) anlegt.
        install_static_units myrmid-suspend.path myrmid-suspend.service myrmid-role.path myrmid-role.service
        srun install -m 755 "$PREFIX/current/systemd/myrmid-role.sh" /usr/local/sbin/myrmid-role
        srun systemctl enable --now myrmid-suspend.path myrmid-role.path
    fi

    if [[ $SERVICE -eq 1 ]]; then
        other=$( [[ "$ROLE" == worker ]] && echo host || echo worker )
        write_unit myrmid-worker "MyrmidLM Worker (GPU und RAM für den Master)" "$NODE ${worker_args[*]}" "$cred" \
            "$( [[ "$ROLE" == worker ]] && echo on || echo off )"
        write_unit myrmid-host "MyrmidLM Host (ganzes Modell für den Router des Masters)" "$NODE ${host_args[*]}" "$cred" \
            "$( [[ "$ROLE" == host ]] && echo on || echo off )"
        ok "Dienst myrmid-$ROLE läuft und startet künftig automatisch; myrmid-$other ist eingerichtet, aber aus."
        info "Log: journalctl -u myrmid-$ROLE -f"
    else
        if [[ "$ROLE" == worker ]]; then info "Starten mit: $NODE ${worker_args[*]}"
        else info "Starten mit: $NODE ${host_args[*]}"; fi
    fi
    [[ $DRY -eq 0 ]] && { srun "$NODE" update check >/dev/null 2>&1 || true; }
    echo
    if [[ "$ROLE" == worker ]]; then
        echo "${G}${B}Fertig.${N} Dieser PC ist Worker ($VER): $WHAT unter ${LAN_IP:-?}:50052$RAM_NOTE."
        echo "Auf dem Master prüfen mit: myrmid scan"
    else
        echo "${G}${B}Fertig.${N} Dieser PC ist Host ($VER, Steuerung ${LAN_IP:-?}:50070)."
        echo "Im Dashboard des Masters erscheint er im Bereich Hosts: dort ein Modell wählen und auf Laden klicken."
    fi
    if [[ -n "$CLUSTER_KEY" ]]; then
        echo "Rolle und Updates steuert das Dashboard des Masters."
    else
        echo "Der Master nimmt diesen PC in den nächsten Sekunden auf (er muss laufen und im selben Netz sein)."
        echo "Danach steuert er Rolle und Updates im Dashboard. Ohne Master: innerhalb einer Stunde, sonst Dienst neu starten."
    fi
    exit 0
fi

# Master
# Modellauswahl: vorhandene GGUF-Dateien und Katalog, bewertet nach Speicher.
# choose_model big|small  ->  setzt CHOSEN (Pfad; bei small leer = ohne Router)
choose_model() {
    local kind="$1" json tmp sel pick
    CHOSEN=""
    if [[ $DRY -eq 1 || ! -x "$NODE" ]]; then
        CHOSEN="$(ask "Pfad zum GGUF-Modell" "")"
        return
    fi
    info "Suche Modelle auf diesem PC und im Katalog (bewertet nach freiem Speicher hier und auf Workern) ..."
    local flags=()
    [[ "$kind" == small ]] && flags+=(--small)
    json="$("$NODE" models --json "${flags[@]}" 2>/dev/null || true)"
    if [[ -z "$json" ]]; then
        warn "Modellliste nicht verfügbar."
        CHOSEN="$(ask "Pfad zum GGUF-Modell" "")"
        return
    fi
    tmp="$(mktemp)"
    printf '%s' "$json" > "$tmp"
    python3 - "$tmp" menu "$kind" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); kind = sys.argv[3]
fit = {"here": "passt hier", "workers": "nur mit Workern", "too_big": "zu groß", "unknown": "?"}
print(f"    Platz: {d['local_mb']/1024:.1f} GB auf diesem PC" + (f", {d['workers_mb']/1024:.1f} GB auf Workern" if d["workers_mb"] else ""))
for i, e in enumerate(d["models"], 1):
    if e.get("error") or e.get("have"):
        continue
    state = fit[e["fit"]] if e.get("arch_ok") is not False else "Architektur nicht unterstützt"
    src = "vorhanden" if e["source"] == "local" else ("schon geladen" if e.get("have") else "Download")
    star = "  <- Empfehlung" if d["recommended"] == i - 1 else ""
    desc = f" ({e['desc']})" if e.get("desc") else ""
    print(f"      {i:>2}) {e['name']:<30} {e['size_mb']/1024:5.1f} GB  {state:<16} {src}{desc}{star}")
print("       p) eigenen Pfad eingeben" + ("\n       0) ohne Router" if kind == "small" else
      "\n       n) kein eigenes Modell: nur Router (z. B. VM ohne GPU; Modelle laufen auf Hosts)"))
PY
    local def
    def="$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); r=d['recommended']; print('' if r is None else r+1)" "$tmp")"
    [[ "$kind" == small && -z "$def" ]] && def=0
    if [[ "$kind" == small ]] && python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['local_mb'] < 8192 else 1)" "$tmp"; then
        info "Wenig Speicher auf diesem PC: Ein Zweitmodell würde dem großen Modell RAM wegnehmen. Empfehlung: ohne Router."
        def=0
    fi
    sel="$(ask "Auswahl" "${def:-p}")"
    if [[ "$sel" == p ]]; then
        CHOSEN="$(ask "Pfad zum GGUF-Modell" "")"
    elif [[ "$kind" == big && "$sel" == n ]]; then
        CHOSEN="none"
    elif [[ "$kind" == small && "$sel" == 0 ]]; then
        CHOSEN=""
    elif [[ "$sel" =~ ^[0-9]+$ ]]; then
        pick="$(python3 - "$tmp" "$sel" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); i = int(sys.argv[2]) - 1
if not 0 <= i < len(d["models"]) or d["models"][i].get("error"):
    sys.exit(1)
e = d["models"][i]
print("local\t" + e["path"] if e["source"] == "local" else f"catalog\t{e['repo']}:{e['quant']}")
PY
)" || die "Ungültige Auswahl: $sel"
        if [[ "${pick%%$'\t'*}" == local ]]; then
            CHOSEN="${pick#*$'\t'}"
        else
            srun mkdir -p /srv/models
            srun chown "$(id -un)" /srv/models
            info "Lade ${pick#*$'\t'} herunter (wird bei Abbruch beim nächsten Start fortgesetzt) ..."
            CHOSEN="$("$NODE" download "${pick#*$'\t'}" --dir /srv/models | tail -1)" || die "Download fehlgeschlagen"
            ok "Modell geladen: $CHOSEN"
        fi
    else
        die "Ungültige Auswahl: $sel"
    fi
    rm -f "$tmp"
}

step "Master einrichten"
REUSE=0
if [[ "$OLD_ROLE" == master && -n "$OLD_ARGS" && -z "$MODEL" ]]; then
    info "Bisherige Einstellungen: $OLD_ARGS"
    if ask_yn "So übernehmen?" j; then
        REUSE=1
        read -ra args <<< "$OLD_ARGS"
        for ((i = 0; i < ${#args[@]}; i++)); do
            case "${args[i]}" in
                -m|--model) MODEL="${args[i + 1]:-}" ;;
                --port) PORT="${args[i + 1]:-$PORT}" ;;
                --fast-model) FAST_MODEL="${args[i + 1]:-}" ;;
            esac
        done
        [[ "$FAST_MODEL" == "-" ]] && FAST_MODEL=""
        open_ports "$LAN_NET" "$PORT/tcp" "8090/tcp"
    fi
fi
if [[ $REUSE -eq 0 ]]; then
if [[ -z "$MODEL" || "$MODEL" == auto ]]; then
    [[ "$MODEL" == auto ]] && TTY_OK=0
    choose_model big
    MODEL="$CHOSEN"
fi
ROUTER_ONLY=0
if [[ "$MODEL" == none ]]; then
    ROUTER_ONLY=1
    info "Nur Router: Dieser PC lädt kein Modell; im Dashboard Hosts zuweisen (oder später ein Modell einschalten)."
    WORKERS="${WORKERS:-off}"
    [[ "$FAST_MODEL" == "-" ]] && FAST_MODEL=""
else
MODEL="${MODEL/#\~/$HOME}"
[[ $DRY -eq 1 || -f "$MODEL" ]] || die "Modell nicht gefunden: ${MODEL:-(keine Angabe)}"
MODEL="$(realpath -m "$MODEL")"
[[ "$MODEL" != *" "* ]] || die "Der Modellpfad darf keine Leerzeichen enthalten: $MODEL"
fi

# Der Dienst läuft als eigener Benutzer und muss das Modell lesen können.
service_can_read() {
    if [[ $EUID -eq 0 ]]; then runuser -u nobody -- test -r "$1"; else sudo -u nobody test -r "$1"; fi
}
if [[ $ROUTER_ONLY -eq 0 && $SERVICE -eq 1 && $DRY -eq 0 ]] && ! service_can_read "$MODEL" 2>/dev/null; then
    warn "Der Dienst kann $MODEL nicht lesen (Home-Ordner sind meist gesperrt)."
    if ask_yn "Modell nach /srv/models kopieren?" j; then
        srun mkdir -p /srv/models
        srun cp "$MODEL" /srv/models/
        MODEL="/srv/models/$(basename "$MODEL")"
        srun chmod 644 "$MODEL"
        ok "Modell liegt jetzt unter $MODEL"
    else
        warn "Dann bitte Leserechte selbst setzen, sonst startet der Dienst nicht."
    fi
fi

if [[ -z "$WORKERS" ]]; then
    echo
    echo "    Welche Worker soll der Master nutzen?"
    echo "      auto  – Worker nur, wo sie nötig sind oder es schneller machen (empfohlen)"
    echo "      all   – jeder Worker mit freiem Platz (schnellste zuerst)"
    echo "      off   – keine"
    echo "      oder feste Liste, z. B. 192.168.1.20:50052,192.168.1.20:50053"
    WORKERS="$(ask "Worker" auto)"
fi
PORT="$(ask "API-Port" "$PORT")"
MTP=0
if [[ $DRY -eq 0 && $ROUTER_ONLY -eq 0 ]] && python3 - "$MODEL" <<'PY' 2>/dev/null; then
import struct, sys
f = open(sys.argv[1], "rb")
assert f.read(4) == b"GGUF"
_, _, n_kv = struct.unpack("<IQQ", f.read(20))
sc = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
def rs():
    (n,) = struct.unpack("<Q", f.read(8)); return f.read(n).decode("utf-8", "replace")
def rv(t):
    if t in sc: return f.read(sc[t])
    if t == 8: return rs()
    et, n = struct.unpack("<IQ", f.read(12))
    if et in sc: f.seek(sc[et] * n, 1)
    else: [rv(et) for _ in range(n)]
for _ in range(n_kv):
    k = rs(); (t,) = struct.unpack("<I", f.read(4)); v = rv(t)
    if k.endswith(".nextn_predict_layers") and int.from_bytes(v, "little") > 0: sys.exit(0)
sys.exit(1)
PY
    info "Das Modell bringt MTP-Layer mit (schneller durch Vorausraten mehrerer Tokens)."
    MTP="$(ask "MTP-Tiefe (0 = aus; beste Tiefe später mit: myrmid tune-mtp)" 1)"
fi
open_ports "$LAN_NET" "$PORT/tcp" "8090/tcp"

# Router: ein kleines, schnelles Modell beantwortet einfache Anfragen sofort.
if [[ "$FAST_MODEL" == "-" ]]; then
    echo
    info "Optional: ein kleines Zweitmodell (1 bis 4B) für kurze, einfache Anfragen."
    info "Ein Router verteilt dann automatisch; im Dashboard ein- und ausschaltbar."
    if [[ $YES -eq 1 ]]; then
        FAST_MODEL=""   # ohne Rückfrage kein Router; mit --fast-model DATEI|auto einschalten
    else
        choose_model small
        FAST_MODEL="$CHOSEN"
    fi
elif [[ "$FAST_MODEL" == auto ]]; then
    TTY_OK=0
    choose_model small
    FAST_MODEL="$CHOSEN"
fi
FAST_MODEL="${FAST_MODEL/#\~/$HOME}"
if [[ -n "$FAST_MODEL" ]]; then
    [[ $DRY -eq 1 || -f "$FAST_MODEL" ]] || die "Modell nicht gefunden: $FAST_MODEL"
    FAST_MODEL="$(realpath -m "$FAST_MODEL")"
    if [[ $SERVICE -eq 1 && $DRY -eq 0 ]] && ! service_can_read "$FAST_MODEL" 2>/dev/null; then
        srun mkdir -p /srv/models
        srun cp "$FAST_MODEL" /srv/models/
        FAST_MODEL="/srv/models/$(basename "$FAST_MODEL")"
        srun chmod 644 "$FAST_MODEL"
        ok "Kleines Modell liegt jetzt unter $FAST_MODEL"
    fi
fi

# Entwurfsmodell: ein kleines Modell derselben Familie schreibt vor, das große prüft nur.
if [[ "${MTP:-0}" == 0 && -z "$DRAFT" ]]; then
    echo
    info "Optional: Entwurfsmodell (ein kleines Modell derselben Familie, z. B. 0,6B zu 32B)."
    info "Es schreibt ein paar Tokens vor, das große Modell prüft sie in einem Durchgang: oft 1,5- bis 2-mal so schnell,"
    info "besonders wenn das Modell über das Netz verteilt ist. Wird bei Bedarf heruntergeladen (unter 1 GB)."
    if [[ $YES -eq 1 ]]; then DRAFT=off; elif ask_yn "Entwurfsmodell nutzen?" j; then DRAFT=auto; else DRAFT=off; fi
fi
if [[ -z "$HOST_SLEEP" ]]; then
    HOST_SLEEP=0
    [[ $YES -eq 0 ]] && HOST_SLEEP="$(ask "Hosts nach wie vielen Minuten ohne Anfrage schlafen legen (0 = nie, später pro Host im Dashboard)" 0)"
fi

fi
if [[ $REUSE -eq 0 ]]; then
args=(serve -m "$MODEL" --workers "$WORKERS" --port "$PORT")
[[ -n "$FAST_MODEL" ]] && args+=(--fast-model "$FAST_MODEL")
[[ "${MTP:-0}" != 0 ]] && args+=(--mtp "$MTP")
[[ -n "$DRAFT" && "$DRAFT" != off ]] && args+=(--draft-model "$DRAFT")
[[ "${HOST_SLEEP:-0}" != 0 ]] && args+=(--host-sleep "$HOST_SLEEP")
fi
if [[ $SERVICE -eq 1 ]]; then
    write_unit myrmid-server "MyrmidLM Server (Modell als OpenAI-API)" "$NODE ${args[*]}"
    ok "Dienst myrmid-server läuft und startet künftig automatisch."
    KEYFILE=/var/lib/myrmid/myrmid/api_key
    if [[ $DRY -eq 0 ]]; then
        info "Warte auf den API-Key ..."
        for _ in $(seq 1 30); do $SUDO test -s "$KEYFILE" && break; sleep 2; done
        KEY="$($SUDO cat "$KEYFILE" 2>/dev/null || true)"
    fi
    info "Log (Ladefortschritt, Plan): journalctl -u myrmid-server -f"
    if [[ $DRY -eq 0 ]]; then
        sleep 8
        if ! systemctl is-active --quiet myrmid-server; then
            warn "Der Dienst läuft nicht. Letzte Meldungen:"
            $SUDO journalctl -u myrmid-server -n 6 --no-pager -o cat | sed 's/^/      /'
        elif $SUDO journalctl -u myrmid-server -n 40 --no-pager -o cat | grep -q "Waiting for workers"; then
            warn "Das Modell passt auf diesen PC allein nicht. Der Dienst wartet auf Worker und startet von selbst, sobald genug im Netz sind (Dashboard: Port 8090)."
        fi
    fi
else
    info "Starten mit: $NODE ${args[*]}"
    info "Der API-Key und der QR-Code erscheinen dann im Terminal."
    KEY=""
fi

[[ $DRY -eq 0 ]] && { srun "$NODE" update check >/dev/null 2>&1 || true; }
URL="http://${LAN_IP:-<IP>}:$PORT/v1"
NAME="$(basename "$MODEL" .gguf)"
echo
echo "${G}${B}Fertig.${N} Der Master ($VER) stellt das Modell bereit:"
echo "    Base URL:  $URL"
echo "    Dashboard: http://${LAN_IP:-<IP>}:8090/  (Anmeldung mit dem API-Key)"
echo "    Modell:    $NAME"
[[ -n "$FAST_MODEL" ]] && echo "    Router:    an (schnelles Modell: $(basename "$FAST_MODEL" .gguf)); Modellname für Clients: myrmid"
[[ -n "${KEY:-}" ]] && echo "    API-Key:   $KEY  (auch der Schlüssel für Hosts)"
if [[ -n "${KEY:-}" ]] && have qrencode; then
    QR="$(python3 -c "import sys,urllib.parse as u; print('privatelm://server?' + u.urlencode({'url':sys.argv[1],'key':sys.argv[2],'model':sys.argv[3],'name':sys.argv[4]}))" \
        "$URL" "$KEY" "$NAME" "$(hostname)")"
    echo
    echo "    QR-Code für die App (Phone cluster > Scan PC QR code):"
    qrencode -t ANSIUTF8 "$QR"
fi
echo "Das erste Laden kann je nach Modell und Workern einige Minuten dauern."
echo "Den Plan (Layer pro Gerät, geschätzte Token/s) zeigt: journalctl -u myrmid-server"
echo "Messen, sobald das Modell geladen ist: myrmid bench"
echo "Updates: Dashboard, Bereich Version (oder: myrmid update)."
