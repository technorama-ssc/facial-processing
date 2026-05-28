#!/bin/bash

# ============================================================================
# Facial Processing Project — Setup Script
# Project: /home/technorama/facial_processing/
# Code:    /home/technorama/facial_processing/Code/
# Env:     /home/technorama/main-env/  (Python 3.11)
#
# Architecture:
#   - Debian Trixie (Python 3.13 system) with custom Python 3.11 venv
#   - Uses v4l2loopback + libcamera to create /dev/video10 for OpenCV
#   - lgpio installed via pip (avoids apt 3.13 conflict)
#   - main.py runs via sudo venv python (needs GPIO + camera access)
#
# NOTE: Run displaylink_setup.sh FIRST before this script!
# ============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_DIR="$HOME/facial_processing"
CODE_DIR="$PROJECT_DIR/Code"
ENV_DIR="$HOME/main-env"
CURRENT_USER=$USER

echo "========================================================"
echo "Facial Processing Project — Environment Setup"
echo "========================================================"
echo ""
echo "Project dir : $PROJECT_DIR"
echo "Code dir    : $CODE_DIR"
echo "Virtual env : $ENV_DIR"
echo ""

# ============================================================================
# PART 0: PROJECT DIRECTORY STRUCTURE
# ============================================================================

echo -e "${BLUE}=======================================================${NC}"
echo -e "${BLUE}PART 0: Setting Up Project Directories${NC}"
echo -e "${BLUE}=======================================================${NC}"
echo ""

mkdir -p "$PROJECT_DIR"
mkdir -p "$CODE_DIR"
mkdir -p "$PROJECT_DIR/Config"

echo -e "${GREEN}Project directories ready${NC}"

# ============================================================================
# PART 0.1: PASSWORDLESS SUDO (must be early — everything below needs sudo)
# ============================================================================

echo ""
echo -e "${BLUE}=======================================================${NC}"
echo -e "${BLUE}PART 0.1: Configuring Passwordless Sudo${NC}"
echo -e "${BLUE}=======================================================${NC}"
echo ""

SUDOERS_FILE="/etc/sudoers.d/facial-processing"

printf 'Defaults:%s !use_pty\n%s ALL=(ALL) NOPASSWD: ALL\n' \
    "$CURRENT_USER" "$CURRENT_USER" \
    | sudo tee "$SUDOERS_FILE" > /dev/null

sudo chmod 0440 "$SUDOERS_FILE"

if sudo visudo -c -f "$SUDOERS_FILE" 2>/dev/null; then
    echo -e "${GREEN}Passwordless sudo configured: $SUDOERS_FILE${NC}"
else
    echo -e "${RED}Sudoers syntax error — removing bad file!${NC}"
    sudo rm -f "$SUDOERS_FILE"
    exit 1
fi

# ============================================================================
# PART 0.2: SYSTEM DEPENDENCIES
# ============================================================================

echo ""
echo -e "${BLUE}=======================================================${NC}"
echo -e "${BLUE}PART 0.2: Installing System Dependencies${NC}"
echo -e "${BLUE}=======================================================${NC}"
echo ""

sudo apt-get update

# Remove conflicting DKMS evdi entries before apt install
echo -e "${YELLOW}Cleaning up any existing DKMS evdi entries...${NC}"
sudo dkms remove evdi/1.14.15 --all 2>/dev/null || true
sudo dkms remove evdi/1.14.16 --all 2>/dev/null || true
sudo dkms remove evdi/1.14.17 --all 2>/dev/null || true

sudo apt-get install -y \
    libcap-dev \
    libopenblas-dev \
    v4l-utils \
    ffmpeg \
    python3-gpiozero

# Install v4l2loopback for virtual camera device
echo -e "${GREEN}Installing v4l2loopback...${NC}"
sudo apt-get install -y v4l2loopback-dkms v4l2loopback-utils

# Fix any broken packages
sudo apt --fix-broken install -y
sudo dpkg --configure -a

# Configure v4l2loopback to load at boot with video10
echo "v4l2loopback" | sudo tee /etc/modules-load.d/v4l2loopback.conf
echo "options v4l2loopback video_nr=10 card_label=PiCamera exclusive_caps=1" | sudo tee /etc/modprobe.d/v4l2loopback.conf

# Load v4l2loopback module now too
sudo modprobe v4l2loopback video_nr=10 card_label="PiCamera" exclusive_caps=1 || true

echo -e "${GREEN}System dependencies installed${NC}"

# ============================================================================
# PART 1: PYTHON 3.11 CHECK / INSTALL
# ============================================================================

echo ""
echo -e "${BLUE}=======================================================${NC}"
echo -e "${BLUE}PART 1: Python 3.11${NC}"
echo -e "${BLUE}=======================================================${NC}"
echo ""

if ! command -v python3.11 &> /dev/null; then
    echo -e "${RED}Python 3.11 not found — building from source (~15-20 min)...${NC}"

    sudo apt-get install -y build-essential zlib1g-dev libncurses5-dev \
        libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev \
        libsqlite3-dev wget libbz2-dev libcap-dev liblzma-dev

    cd /tmp
    if [ ! -f "Python-3.11.9.tgz" ]; then
        wget https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz
    fi
    tar -xf Python-3.11.9.tgz
    cd Python-3.11.9
    ./configure --enable-optimizations
    make -j4
    sudo make altinstall
    cd ~
    sudo rm -rf /tmp/Python-3.11.9 /tmp/Python-3.11.9.tgz || true

    if ! command -v python3.11 &> /dev/null; then
        echo -e "${RED}Python 3.11 build failed! Exiting.${NC}"
        exit 1
    fi
    echo -e "${GREEN}Python 3.11 built successfully!${NC}"
else
    echo -e "${GREEN}Python 3.11 already installed: $(python3.11 --version)${NC}"
fi

# ============================================================================
# PART 2: CREATE VIRTUAL ENVIRONMENT
# ============================================================================

echo ""
echo -e "${BLUE}=======================================================${NC}"
echo -e "${BLUE}PART 2: Creating Virtual Environment${NC}"
echo -e "${BLUE}=======================================================${NC}"
echo ""

if [ ! -d "$ENV_DIR" ]; then
    echo -e "${GREEN}Creating main-env...${NC}"
    python3.11 -m venv "$ENV_DIR"
else
    echo -e "${GREEN}Existing main-env found — keeping it${NC}"
fi

source "$ENV_DIR/bin/activate"

echo -e "${GREEN}Using: $(python --version)${NC}"

python -m pip install --upgrade pip setuptools wheel

# ============================================================================
# PART 3: INSTALL PYTHON PACKAGES (smart check)
# ============================================================================

echo ""
echo -e "${BLUE}=======================================================${NC}"
echo -e "${BLUE}PART 3: Python Packages${NC}"
echo -e "${BLUE}=======================================================${NC}"
echo ""

# --- Helper: check if a pip package is installed ---
pkg_installed() {
    python -c "import $1" 2>/dev/null && return 0 || return 1
}

# --- Define packages: "import_name|pip_name|version_spec|label" ---
declare -a PACKAGES=(
    "numpy|numpy|>=2.0|NumPy"
    "cv2|opencv-python||OpenCV"
    "mediapipe|mediapipe||MediaPipe"
    "google.protobuf|protobuf||Protobuf"
    "lgpio|lgpio||lgpio"
    "flask|flask||Flask"
)

# --- Scan which packages are installed / missing ---
declare -a INSTALLED_PKGS=()
declare -a MISSING_PKGS=()

echo -e "${BLUE}Scanning installed packages...${NC}"
echo ""

for entry in "${PACKAGES[@]}"; do
    IFS='|' read -r import_name pip_name version_spec label <<< "$entry"
    if pkg_installed "$import_name"; then
        echo -e "  ${GREEN}✓ $label — already installed${NC}"
        INSTALLED_PKGS+=("$entry")
    else
        echo -e "  ${YELLOW}✗ $label — not found${NC}"
        MISSING_PKGS+=("$entry")
    fi
done

echo ""

TOTAL=${#PACKAGES[@]}
NUM_INSTALLED=${#INSTALLED_PKGS[@]}
NUM_MISSING=${#MISSING_PKGS[@]}

# --- Install function ---
install_package() {
    local import_name="$1"
    local pip_name="$2"
    local version_spec="$3"
    local label="$4"

    echo -e "${GREEN}Installing $label...${NC}"
    pip install "${pip_name}${version_spec}"

    # Special post-install: re-pin numpy after mediapipe can downgrade it
    if [ "$import_name" = "mediapipe" ]; then
        echo -e "${YELLOW}  Re-pinning numpy>=2.0 after mediapipe install...${NC}"
        pip install "numpy>=2.0" --force-reinstall
    fi
}

# ---- CASE 1: Nothing installed → install everything silently ----
DO_REINSTALL="n"

if [ "$NUM_MISSING" -eq "$TOTAL" ]; then
    echo -e "${BLUE}No packages installed. Installing all packages...${NC}"
    echo ""
    for entry in "${PACKAGES[@]}"; do
        IFS='|' read -r import_name pip_name version_spec label <<< "$entry"
        install_package "$import_name" "$pip_name" "$version_spec" "$label"
    done

# ---- CASE 2: Some installed, some missing ----
elif [ "$NUM_MISSING" -gt 0 ] && [ "$NUM_INSTALLED" -gt 0 ]; then
    echo -e "${YELLOW}Some packages are missing. Installing missing packages automatically...${NC}"
    echo ""
    for entry in "${MISSING_PKGS[@]}"; do
        IFS='|' read -r import_name pip_name version_spec label <<< "$entry"
        install_package "$import_name" "$pip_name" "$version_spec" "$label"
    done

    echo ""
    echo -e "${YELLOW}Already-installed packages:${NC}"
    for entry in "${INSTALLED_PKGS[@]}"; do
        IFS='|' read -r import_name pip_name version_spec label <<< "$entry"
        echo -e "  • $label"
    done
    echo ""
    read -rp "$(echo -e "${YELLOW}Reinstall already-installed packages too? [y/n]: ${NC}")" DO_REINSTALL

# ---- CASE 3: Everything already installed ----
else
    echo -e "${GREEN}All packages are already installed.${NC}"
    echo ""
    read -rp "$(echo -e "${YELLOW}Reinstall all packages from scratch? [y/n]: ${NC}")" DO_REINSTALL
fi

# ---- Shared reinstall block ----
if [[ "$DO_REINSTALL" =~ ^[Yy]$ ]]; then
    echo ""
    echo -e "${BLUE}Reinstalling packages...${NC}"
    echo ""
    # In Case 3 (all installed), reinstall everything; in Case 2, just the already-installed ones
    TARGET_PKGS=("${INSTALLED_PKGS[@]}")
    [ "$NUM_INSTALLED" -eq "$TOTAL" ] && TARGET_PKGS=("${PACKAGES[@]}")
    for entry in "${TARGET_PKGS[@]}"; do
        IFS='|' read -r import_name pip_name version_spec label <<< "$entry"
        echo -e "${GREEN}Reinstalling $label...${NC}"
        pip install --force-reinstall "${pip_name}${version_spec}"
    done
    echo -e "${YELLOW}Re-pinning numpy>=2.0...${NC}"
    pip install "numpy>=2.0" --force-reinstall
else
    echo -e "${GREEN}Skipping reinstall.${NC}"
fi

# ============================================================================
# VERIFY IMPORTS
# ============================================================================

echo ""
echo -e "${BLUE}Verifying all project imports...${NC}"

python -c "import numpy;     print('  ✓ numpy',     numpy.__version__)"     || echo "  ✗ numpy FAILED"
python -c "import cv2;       print('  ✓ opencv',    cv2.__version__)"       || echo "  ✗ opencv FAILED"
python -c "import mediapipe; print('  ✓ mediapipe', mediapipe.__version__)" || echo "  ✗ mediapipe FAILED"
python -c "import lgpio;     print('  ✓ lgpio')"                            || echo "  ✗ lgpio FAILED"
python -c "import importlib.metadata; print('  ✓ flask', importlib.metadata.version('flask'))"     || echo "  ✗ flask FAILED"

echo ""
pip freeze > "$HOME/main-env-requirements.txt"
echo -e "${GREEN}Requirements saved to ~/main-env-requirements.txt${NC}"

deactivate

# ============================================================================
# CREATE STARTUP SERVICE
# ============================================================================

echo ""
echo -e "${BLUE}=======================================================${NC}"
echo -e "${BLUE}Creating Startup Service${NC}"
echo -e "${BLUE}=======================================================${NC}"
echo ""

# Launcher wrapper
cat > "/home/$CURRENT_USER/facial_processing/start.sh" << LAUNCHER
#!/bin/bash

LOG_TAG="facial_processing"

log() { echo "[\$(date '+%H:%M:%S')] \$1" | systemd-cat -t "\$LOG_TAG" -p info; }

log "Waiting for Wayland display..."
TIMEOUT=60
ELAPSED=0
until [ -S "/run/user/1000/wayland-0" ]; do
    sleep 1
    ELAPSED=\$((ELAPSED + 1))
    if [ "\$ELAPSED" -ge "\$TIMEOUT" ]; then
        log "ERROR: Wayland never became ready. Exiting."
        exit 1
    fi
done
log "Wayland ready after \${ELAPSED}s"

ELAPSED=0
until xdpyinfo -display :0 >/dev/null 2>&1; do
    sleep 1
    ELAPSED=\$((ELAPSED + 1))
    if [ "\$ELAPSED" -ge 30 ]; then
        log "WARNING: XWayland not ready, continuing anyway"
        break
    fi
done
log "XWayland ready after \${ELAPSED}s"

export DISPLAY=:0
export WAYLAND_DISPLAY=wayland-0
export XDG_RUNTIME_DIR=/run/user/1000
export GDK_BACKEND=x11
export QT_QPA_PLATFORM=xcb
export SDL_VIDEODRIVER=x11

exec /home/$CURRENT_USER/main-env/bin/python \
    /home/$CURRENT_USER/facial_processing/Code/main.py
LAUNCHER

chmod +x "/home/$CURRENT_USER/facial_processing/start.sh"
chmod +x "/home/$CURRENT_USER/facial_processing/Config/fail_save_monitoring.sh"

# Systemd service
sudo tee /etc/systemd/system/facial_processing.service > /dev/null << EOF
[Unit]
Description=Facial Processing Service
After=graphical-session.target displaylink-driver.service
Wants=graphical-session.target
StartLimitIntervalSec=0

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=/home/$CURRENT_USER/facial_processing/Code
ExecStartPre=/bin/sleep 15
ExecStart=/home/$CURRENT_USER/facial_processing/Config/fail_save_monitoring.sh
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=graphical.target
EOF

chmod +x "$HOME/facial_processing/Code/main.py" 2>/dev/null || true

sudo systemctl daemon-reload
sudo systemctl enable facial_processing.service

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo -e "${GREEN}=======================================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}=======================================================${NC}"
echo ""
echo "  ✓ $ENV_DIR (Python 3.11)"
echo "  ✓ $CODE_DIR (project code)"
echo "  ✓ Camera streaming via v4l2loopback (/dev/video10)"
echo "  ✓ facial_processing.service enabled"
echo "  ✓ Passwordless sudo configured (/etc/sudoers.d/facial-processing)"
echo ""
echo -e "${YELLOW}Note: Run displaylink_setup.sh FIRST if not done already!${NC}"
echo -e "${YELLOW}      Check camera: v4l2-ctl --list-devices${NC}"
echo "========================================================"

echo -e "${YELLOW}Rebooting in 5 seconds...${NC}"
sleep 5
sync
sudo reboot