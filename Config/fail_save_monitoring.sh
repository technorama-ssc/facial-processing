#!/bin/bash
# =============================================================================
# failsafe_monitor.sh — Failsafe & Health Monitor for Facial Processing System
# =============================================================================
# Monitors:
#   - main.py              (venv Python / OpenCV + Flask + GPIO)
#   - Flask webserver      port 5000
#   - rpicam-vid + ffmpeg  (camera pipeline → /dev/video10)
#   - /dev/video10         (v4l2loopback virtual camera device)
#   - GPIO chip            /dev/gpiochip4  (as per config.py GPIO_CHIP=4)
#   - Required Python packages in main-env
#   - Kernel module        v4l2loopback
#   - Disk space           (warn + auto-clean at <500 MB free)
#   - Memory               (warn at >90% used)
#   - CPU temperature      (warn at >80°C)
#
# Auto-fixes:
#   - Restarts dead/hung main.py
#   - Restarts dead/hung camera pipeline (rpicam-vid | ffmpeg)
#   - Kills processes holding Flask port before restarting
#   - Reloads missing v4l2loopback kernel module
#   - Reinstalls missing Python packages into main-env
#   - Cleans old logs/backups when disk is low
#   - Recreates missing log directory
#   - Repairs broken pip via ensurepip
#   - Sets fallback DISPLAY / Wayland env vars if not available
#   - Removes stale PID files
#   - Reboots if crash count exceeds MAX_CRASHES in CRASH_WINDOW seconds
# =============================================================================

set -uo pipefail

# ─────────────────────────────────────────────
# CONFIGURATION — mirrors setup.sh paths exactly


# ─────────────────────────────────────────────
CURRENT_USER=$USER
HOME_DIR="/home/$CURRENT_USER"
PROJECT_DIR="$HOME_DIR/facial_processing"
CODE_DIR="$PROJECT_DIR/Code"
MAIN_ENV="$HOME_DIR/main-env"

MAIN_APP="$CODE_DIR/main.py"
START_SH="$PROJECT_DIR/start.sh"

LOG_DIR="$HOME_DIR/logs"
LOG_FILE="$LOG_DIR/failsafe.log"

CHECK_INTERVAL=30       # seconds between health checks
MAX_CRASHES=5           # max crashes before forced reboot
CRASH_WINDOW=300        # sliding window in seconds

FLASK_PORT=5000         # webserver.py port (start_webserver default)
VIDEO_DEVICE="/dev/video10"  # v4l2loopback device used in hardware.py

DISK_WARN_MB=500
DISK_CLEAN_MB=200
MEM_WARN_PCT=90
CPU_TEMP_WARN=80

# PIDs of processes WE started
APP_PID=""

# Crash counter
APP_CRASHES=()

# ─────────────────────────────────────────────
# COLOURS
# ─────────────────────────────────────────────
if [ -t 1 ]; then
    RED='\033[0;31m'; YELLOW='\033[1;33m'
    GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
else
    RED=''; YELLOW=''; GREEN=''; CYAN=''; NC=''
fi

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
ensure_log_dir() {
    if [ ! -d "$LOG_DIR" ]; then
        mkdir -p "$LOG_DIR"
        echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Log directory created: $LOG_DIR"
    fi
}

log() {
    local level="$1"; shift
    local msg="$*"
    local ts; ts=$(date '+%Y-%m-%d %H:%M:%S')
    ensure_log_dir
    echo -e "${ts} [${level}] ${msg}" | tee -a "$LOG_FILE"

    # Rotate at 512 KB
    local size_bytes; size_bytes=$(stat -c%s "$LOG_FILE" 2>/dev/null || echo 0)
    if (( size_bytes > 524288 )); then
        local backup="${LOG_FILE}.$(date +%s).bak"
        cp "$LOG_FILE" "$backup"
        tail -1000 "$LOG_FILE" > "${LOG_FILE}.tmp" && mv "${LOG_FILE}.tmp" "$LOG_FILE"
        echo "$(date '+%Y-%m-%d %H:%M:%S') [WARN] Log rotated → $backup" >> "$LOG_FILE"
    fi
}

# Only WARN and ERROR are printed to terminal and saved to log
log_ok()   { :; }                               # silenced
log_warn() { log "${YELLOW}WARN ${NC}" "$@"; }
log_err()  { log "${RED}ERROR${NC}" "$@"; }
log_info() { :; }                               # silenced

# ─────────────────────────────────────────────
# CRASH WINDOW HELPER
# ─────────────────────────────────────────────
count_recent_crashes() {
    local -n arr=$1
    local now; now=$(date +%s)
    local fresh=()
    for ts in "${arr[@]+"${arr[@]}"}"; do
        if (( now - ts < CRASH_WINDOW )); then
            fresh+=("$ts")
        fi
    done
    arr=("${fresh[@]+"${fresh[@]}"}")
    echo "${#arr[@]}"
}

record_crash() {
    local -n arr=$1
    arr+=("$(date +%s)")
}

# ─────────────────────────────────────────────
# PORT HELPERS
# ─────────────────────────────────────────────
free_port() {
    local port="$1"
    local label="${2:-port $port}"

    if command -v fuser &>/dev/null; then
        if fuser "${port}/tcp" &>/dev/null 2>&1; then
            log_warn "Port $port held by another process — killing via fuser..."
            sudo fuser -k "${port}/tcp" 2>/dev/null || true
            sleep 1
            log_ok "Port $port cleared"
            return
        fi
    fi

    local pid
    pid=$(ss -tlnp 2>/dev/null | grep ":${port}" | grep -oP 'pid=\K[0-9]+' | head -1)

    if [ -n "$pid" ]; then
        log_warn "Port $port held by PID $pid — killing..."
        sudo kill -9 "$pid" 2>/dev/null || true
        sleep 1
        log_ok "Port $port cleared (killed PID $pid)"
    fi
}

port_listening() {
    local port="$1"
    ss -tlnp 2>/dev/null | grep -q ":${port}"
}

# ─────────────────────────────────────────────
# DISPLAY / WAYLAND HELPER
# mirrors what start.sh does at launch
# ─────────────────────────────────────────────
ensure_display() {
    if [ -n "${DISPLAY:-}" ] && xdpyinfo -display "$DISPLAY" &>/dev/null 2>&1; then
        return
    fi

    log_warn "DISPLAY not reachable — probing for a usable display..."

    for disp in :0 :1 :2; do
        if xdpyinfo -display "$disp" &>/dev/null 2>&1; then
            export DISPLAY="$disp"
            log_ok "Using fallback DISPLAY=$DISPLAY"
            return
        fi
    done

    # Wayland / XWayland fallback (matches start.sh env block)
    if [ -S "/run/user/1000/wayland-0" ]; then
        export WAYLAND_DISPLAY=wayland-0
        export XDG_RUNTIME_DIR=/run/user/1000
        export GDK_BACKEND=x11
        export QT_QPA_PLATFORM=xcb
        export SDL_VIDEODRIVER=x11
        export DISPLAY=:0
        log_warn "No X display — applied Wayland/XWayland fallback env vars"
        return
    fi

    log_err "No display found — OpenCV windows will not work"
}

# ─────────────────────────────────────────────
# STALE PID FILE CLEANUP
# ─────────────────────────────────────────────
clean_stale_pidfiles() {
    local pidfile
    for pidfile in "$HOME_DIR"/*.pid "$CODE_DIR"/*.pid /tmp/main_app.pid; do
        [ -f "$pidfile" ] || continue
        local pid; pid=$(cat "$pidfile" 2>/dev/null || echo "")
        if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
            log_warn "Removing stale PID file: $pidfile (PID $pid not running)"
            rm -f "$pidfile"
        fi
    done
}

# ─────────────────────────────────────────────
# KERNEL MODULE CHECK
# Only v4l2loopback needed (no SPI / ZMQ in this project)
# ─────────────────────────────────────────────
ensure_module() {
    local module="$1"
    local extra_args="${2:-}"

    if lsmod | grep -q "^${module}"; then
        log_ok "Kernel module $module loaded"
        return
    fi

    log_warn "Kernel module $module not loaded — loading..."
    if sudo modprobe "$module" $extra_args 2>/dev/null; then
        sleep 1
        if lsmod | grep -q "^${module}"; then
            log_ok "Kernel module $module loaded successfully"
        else
            log_err "modprobe $module returned 0 but module not visible in lsmod"
        fi
    else
        log_err "Failed to load kernel module $module — check dkms / kernel headers"
    fi
}

check_kernel_modules() {
    log_info "Checking kernel modules..."
    # Matches /etc/modprobe.d/v4l2loopback.conf from setup.sh
    ensure_module "v4l2loopback" "video_nr=10 card_label=PiCamera exclusive_caps=1"
}

# ─────────────────────────────────────────────
# PIP HEALTH CHECK
# ─────────────────────────────────────────────
ensure_pip_healthy() {
    if ! "$MAIN_ENV/bin/pip" --version &>/dev/null 2>&1; then
        log_err "pip in main-env is broken — attempting repair via ensurepip..."
        if "$MAIN_ENV/bin/python" -m ensurepip --upgrade >> "$LOG_DIR/pip_install.log" 2>&1; then
            "$MAIN_ENV/bin/python" -m pip install --upgrade pip >> "$LOG_DIR/pip_install.log" 2>&1 || true
            log_ok "pip repaired successfully"
        else
            log_err "ensurepip failed — main-env may need to be recreated with: python3.11 -m venv $MAIN_ENV"
        fi
    fi
}

# ─────────────────────────────────────────────
# PACKAGE CHECK & AUTO-REINSTALL
# Covers every import used across:
#   main.py, hardware.py, display.py, config.py,
#   utils.py, face_enhance.py, webserver.py, alignment_guide.py,
#   handle_flow.py, wrinkles.py, hair_detection.py
# ─────────────────────────────────────────────

# import_name → pip install spec
declare -A PKG_MAP=(
    # Core image / ML
    [cv2]="opencv-python"
    [numpy]="numpy>=2.0"
    [mediapipe]="mediapipe"
    # GPIO / hardware
    [lgpio]="lgpio"
    # Web server (webserver.py)
    [flask]="flask"
    # Image utilities (utils.py — PIL/Pillow for Unicode text rendering)
    [PIL]="Pillow"
    # System info (optional but checked in wrinkles / alignment)
    [scipy]="scipy"
)

reinstall_pkg() {
    local import_name="$1"
    local pip_spec="${PKG_MAP[$import_name]:-$import_name}"
    ensure_pip_healthy
    log_warn "Reinstalling $pip_spec into main-env..."
    if "$MAIN_ENV/bin/pip" install --quiet "$pip_spec" >> "$LOG_DIR/pip_install.log" 2>&1; then
        log_ok "Reinstalled $pip_spec"
    else
        log_err "Failed to reinstall $pip_spec — see $LOG_DIR/pip_install.log"
    fi
}

check_packages() {
    log_info "Checking critical Python packages in main-env..."
    ensure_pip_healthy

    local missing=()

    for pkg in "${!PKG_MAP[@]}"; do
        if ! "$MAIN_ENV/bin/python" -c "import $pkg" 2>/dev/null; then
            missing+=("$pkg")
        fi
    done

    if [ ${#missing[@]} -eq 0 ]; then
        log_ok "All critical packages present in main-env"
        return 0
    fi

    log_err "main-env missing packages: ${missing[*]}"
    for pkg in "${missing[@]}"; do
        reinstall_pkg "$pkg"
    done

    # Verify reinstalls
    local still_missing=()
    for pkg in "${missing[@]}"; do
        if ! "$MAIN_ENV/bin/python" -c "import $pkg" 2>/dev/null; then
            still_missing+=("$pkg")
        fi
    done

    if [ ${#still_missing[@]} -eq 0 ]; then
        log_ok "All missing packages reinstalled successfully"
    else
        log_err "Could NOT reinstall: ${still_missing[*]} — manual intervention required"
    fi
}

# ─────────────────────────────────────────────
# DEVICE CHECK
# GPIO chip 4 (config.py: GPIO_CHIP = 4)
# /dev/video10 (hardware.py: v4l2loopback)
# ─────────────────────────────────────────────
check_devices() {
    log_info "Checking hardware devices..."

    # GPIO chip — config.py uses GPIO_CHIP = 4
    if [ ! -e /dev/gpiochip4 ]; then
        log_err "GPIO: /dev/gpiochip4 not found (config.py GPIO_CHIP=4)"
    else
        log_ok "GPIO: /dev/gpiochip4 present"
    fi

    # v4l2loopback virtual camera device
    if [ ! -e "$VIDEO_DEVICE" ]; then
        log_warn "Camera: $VIDEO_DEVICE not found — attempting to reload v4l2loopback..."
        ensure_module "v4l2loopback" "video_nr=10 card_label=PiCamera exclusive_caps=1"
        sleep 2
        if [ -e "$VIDEO_DEVICE" ]; then
            log_ok "Camera: $VIDEO_DEVICE now available after module reload"
        else
            log_err "Camera: $VIDEO_DEVICE still missing — rpicam-vid pipeline cannot start"
        fi
    else
        log_ok "Camera: $VIDEO_DEVICE present"
    fi
}

# ─────────────────────────────────────────────
# CAMERA PIPELINE CHECK
# hardware.py launches: rpicam-vid | ffmpeg → /dev/video10
# hardware.py also has an internal watchdog, but we guard the outer process too
# ─────────────────────────────────────────────
check_camera_pipeline() {
    local rpicam_alive=false ffmpeg_alive=false

    pgrep -f "rpicam-vid" &>/dev/null && rpicam_alive=true
    pgrep -f "ffmpeg.*video10" &>/dev/null && ffmpeg_alive=true

    if $rpicam_alive && $ffmpeg_alive; then
        log_ok "Camera pipeline: rpicam-vid + ffmpeg running"
        return
    fi

    if ! $rpicam_alive; then
        log_warn "Camera pipeline: rpicam-vid not running"
    fi
    if ! $ffmpeg_alive; then
        log_warn "Camera pipeline: ffmpeg → video10 not running"
    fi

    # The pipeline is started by HardwareManager inside main.py.
    # If the pipeline is dead but main.py is alive, hardware.py's internal
    # watchdog should restart it within ~10 s.  We log and wait.
    # If main.py is also dead, watch_main_app() will restart everything.
    local app_alive=false
    { [ -n "$APP_PID" ] && kill -0 "$APP_PID" 2>/dev/null; } && app_alive=true
    pgrep -f "facial_processing/Code/main.py" &>/dev/null && app_alive=true

    if $app_alive; then
        log_warn "Camera pipeline down but main.py is alive — HardwareManager watchdog should recover it"
    else
        log_err "Camera pipeline AND main.py are both down — main.py restart will rebuild the pipeline"
    fi
}

# ─────────────────────────────────────────────
# PROCESS STARTER
# Mirrors start.sh exactly — including the Wayland/XWayland wait loops
# ─────────────────────────────────────────────
start_main_app() {
    # Clear Python cache so latest code is always used
    log_info "Clearing Python __pycache__..."
    find "$PROJECT_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find "$PROJECT_DIR" -name "*.pyc" -delete 2>/dev/null || true

    ensure_log_dir
    clean_stale_pidfiles

    # ── Wait for Wayland socket (mirrors start.sh exactly) ────────────────
    local WAYLAND_TIMEOUT=60
    local elapsed=0
    log_info "Waiting for Wayland display socket..."
    until [ -S "/run/user/1000/wayland-0" ]; do
        sleep 1
        elapsed=$(( elapsed + 1 ))
        if (( elapsed >= WAYLAND_TIMEOUT )); then
            log_err "Wayland socket never appeared after ${WAYLAND_TIMEOUT}s — aborting start"
            return 1
        fi
    done
    log_ok "Wayland ready after ${elapsed}s"

    # ── Wait for XWayland / DISPLAY :0 (mirrors start.sh exactly) ────────
    elapsed=0
    log_info "Waiting for XWayland (:0)..."
    until xdpyinfo -display :0 >/dev/null 2>&1; do
        sleep 1
        elapsed=$(( elapsed + 1 ))
        if (( elapsed >= 30 )); then
            break
        fi
    done
    if (( elapsed >= 30 )); then
        log_warn "XWayland did not confirm after 30s — continuing anyway (display may still work)"
    else
        log_ok "XWayland ready after ${elapsed}s"
    fi

    # ── Set full display environment (mirrors start.sh env block) ─────────
    export DISPLAY=:0
    export WAYLAND_DISPLAY=wayland-0
    export XDG_RUNTIME_DIR=/run/user/1000
    export GDK_BACKEND=x11
    export QT_QPA_PLATFORM=xcb
    export SDL_VIDEODRIVER=x11
    export XAUTHORITY="${XAUTHORITY:-$HOME_DIR/.Xauthority}"

    # Free Flask port before starting
    if port_listening "$FLASK_PORT"; then
        log_warn "Flask port $FLASK_PORT bound before start — freeing..."
        free_port "$FLASK_PORT" "Flask"
        sleep 1
    fi

    # Kill any stale camera processes (HardwareManager will restart them cleanly)
    sudo pkill -f "rpicam-vid" 2>/dev/null || true
    sudo pkill -f "ffmpeg.*video10" 2>/dev/null || true
    sleep 1

    log_info "Starting main.py (main-env Python 3.11)..."

    # main.py requires sudo for GPIO (lgpio gpiochip_open) + camera
    sudo \
    DISPLAY=:0 \
    WAYLAND_DISPLAY=wayland-0 \
    XDG_RUNTIME_DIR=/run/user/1000 \
    GDK_BACKEND=x11 \
    QT_QPA_PLATFORM=xcb \
    SDL_VIDEODRIVER=x11 \
    XAUTHORITY="${XAUTHORITY:-$HOME_DIR/.Xauthority}" \
    "$MAIN_ENV/bin/python" "$MAIN_APP" \
    >> "$LOG_DIR/main_app.log" 2>&1 &
    APP_PID=$!

    sleep 4   # give Flask + HardwareManager time to initialise

    if kill -0 "$APP_PID" 2>/dev/null; then
        log_ok "main.py started (PID $APP_PID)"
    else
        log_err "main.py failed immediately — check $LOG_DIR/main_app.log"
        APP_PID=""
    fi
}

# ─────────────────────────────────────────────
# FLASK PORT / HEALTH CHECK
# webserver.py exposes / and /tune; we probe / as the health endpoint
# ─────────────────────────────────────────────
FLASK_FAIL_STRIKES=0
FLASK_GRACE_CHECKS=4   # tolerate ~2 minutes of slow startup before killing

check_flask_port() {
    local responding=false

    if port_listening "$FLASK_PORT"; then
        # /filter-meta is a lightweight GET that returns JSON — good liveness probe
        if curl -sf --max-time 4 "http://localhost:${FLASK_PORT}/filter-meta" &>/dev/null; then
            responding=true
        fi
    fi

    if $responding; then
        log_ok "Flask port $FLASK_PORT responding"
        FLASK_FAIL_STRIKES=0
        return
    fi

    local app_alive=false
    { [ -n "$APP_PID" ] && kill -0 "$APP_PID" 2>/dev/null; } && app_alive=true
    pgrep -f "facial_processing/Code/main.py" &>/dev/null && app_alive=true

    if $app_alive; then
        FLASK_FAIL_STRIKES=$(( FLASK_FAIL_STRIKES + 1 ))
        log_warn "Flask not responding (main.py alive) — strike $FLASK_FAIL_STRIKES/$FLASK_GRACE_CHECKS"
        if (( FLASK_FAIL_STRIKES < FLASK_GRACE_CHECKS )); then
            return
        fi
        log_err "Flask unresponsive for $FLASK_GRACE_CHECKS checks — killing main.py"
        sudo pkill -f "facial_processing/Code/main.py" 2>/dev/null || true
        sleep 2
    else
        log_err "Flask port $FLASK_PORT not responding and main.py is down"
    fi

    free_port "$FLASK_PORT" "Flask"
    FLASK_FAIL_STRIKES=0

    record_crash APP_CRASHES
    local n; n=$(count_recent_crashes APP_CRASHES)
    log_warn "main.py crash/hang count in last ${CRASH_WINDOW}s: $n / $MAX_CRASHES"
    if (( n >= MAX_CRASHES )); then
        log_err "Too many main.py failures — rebooting in 10 seconds!"
        sleep 10
        sync
        sudo reboot
    fi
    start_main_app
}

# ─────────────────────────────────────────────
# PROCESS WATCHDOG
# ─────────────────────────────────────────────
watch_main_app() {
    local alive=false

    if [ -n "$APP_PID" ] && kill -0 "$APP_PID" 2>/dev/null; then
        alive=true
    elif pgrep -f "facial_processing/Code/main.py" &>/dev/null; then
        alive=true
    fi

    if $alive; then
        log_ok "main.py is running (PID ${APP_PID:-unknown})"
        return
    fi

    log_err "main.py is NOT running — restarting..."
    free_port "$FLASK_PORT" "Flask"

    record_crash APP_CRASHES
    local n; n=$(count_recent_crashes APP_CRASHES)
    log_warn "main.py crash count in last ${CRASH_WINDOW}s: $n / $MAX_CRASHES"
    if (( n >= MAX_CRASHES )); then
        log_err "Too many main.py crashes — rebooting in 10 seconds!"
        sleep 10
        sync
        sudo reboot
    fi
    start_main_app
}

# ─────────────────────────────────────────────
# SYSTEM HEALTH
# ─────────────────────────────────────────────
clean_old_logs() {
    log_warn "Disk critically low — cleaning old log backups and pip cache..."
    local freed=0

    for f in "$LOG_DIR"/*.bak; do
        [ -f "$f" ] || continue
        local size; size=$(du -k "$f" | cut -f1)
        rm -f "$f"
        freed=$(( freed + size ))
        log_info "Deleted log backup: $f (${size} KB)"
    done

    if [ -d "$HOME_DIR/.cache/pip" ]; then
        local cache_size; cache_size=$(du -sk "$HOME_DIR/.cache/pip" | cut -f1)
        rm -rf "$HOME_DIR/.cache/pip"
        freed=$(( freed + cache_size ))
        log_info "Cleared pip cache (${cache_size} KB)"
    fi

    # Trim large individual log files (keep last 1000 lines)
    for f in "$LOG_DIR"/*.log; do
        [ -f "$f" ] || continue
        local size_kb; size_kb=$(du -k "$f" | cut -f1)
        if (( size_kb > 51200 )); then   # > 50 MB
            tail -1000 "$f" > "${f}.tmp" && mv "${f}.tmp" "$f"
            log_info "Trimmed large log: $f (was ${size_kb} KB)"
        fi
    done

    log_ok "Disk cleanup complete — freed ~$(( freed / 1024 )) MB"
}

check_disk() {
    local free_kb; free_kb=$(df "$HOME_DIR" | awk 'NR==2 {print $4}')
    local free_mb=$(( free_kb / 1024 ))

    if (( free_mb < DISK_CLEAN_MB )); then
        log_err "Disk critically low: ${free_mb} MB free — auto-cleaning..."
        clean_old_logs
    elif (( free_mb < DISK_WARN_MB )); then
        log_warn "Disk: only ${free_mb} MB free (threshold: ${DISK_WARN_MB} MB)"
    else
        log_ok "Disk: ${free_mb} MB free"
    fi
}

check_memory() {
    local total used pct
    read -r total used _ < <(free -m | awk '/^Mem:/ {print $2, $3, $4}')
    pct=$(( used * 100 / total ))

    if (( pct > MEM_WARN_PCT )); then
        log_warn "Memory: ${pct}% used (${used}/${total} MB) — dropping page cache..."
        sync
        echo 1 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1 || true
        read -r total used _ < <(free -m | awk '/^Mem:/ {print $2, $3, $4}')
        pct=$(( used * 100 / total ))
        log_info "Memory after cache drop: ${pct}% (${used}/${total} MB)"
    else
        log_ok "Memory: ${pct}% used (${used}/${total} MB)"
    fi
}

check_cpu_temp() {
    if [ -f /sys/class/thermal/thermal_zone0/temp ]; then
        local temp_raw; temp_raw=$(cat /sys/class/thermal/thermal_zone0/temp)
        local temp_c=$(( temp_raw / 1000 ))
        if (( temp_c > CPU_TEMP_WARN )); then
            log_warn "CPU temperature: ${temp_c}°C — above ${CPU_TEMP_WARN}°C threshold"
        else
            log_ok "CPU temperature: ${temp_c}°C"
        fi
    else
        log_warn "CPU temperature: sensor not found at /sys/class/thermal/thermal_zone0/temp"
    fi
}

# ─────────────────────────────────────────────
# CLEAN SHUTDOWN
# ─────────────────────────────────────────────
_CLEANUP_DONE=0
cleanup() {
    # Guard against re-entrant calls (multiple SIGINTs)
    if (( _CLEANUP_DONE )); then return; fi
    _CLEANUP_DONE=1

    # Restore default signal handling immediately so further Ctrl+C doesn't re-trigger
    trap - SIGINT SIGTERM EXIT

    echo "Failsafe monitor shutting down — stopping child processes..."

    # Kill main.py gracefully first, then hard
    if [ -n "$APP_PID" ] && kill -0 "$APP_PID" 2>/dev/null; then
        kill -TERM "$APP_PID" 2>/dev/null || true
        sleep 3
        kill -0 "$APP_PID" 2>/dev/null && kill -9 "$APP_PID" 2>/dev/null || true
    fi

    # Mop up any remaining processes
    pkill -f "facial_processing/Code/main.py" 2>/dev/null || true
    pkill -f "rpicam-vid" 2>/dev/null || true
    pkill -f "ffmpeg.*video10" 2>/dev/null || true

    echo "Shutdown complete."
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
log_info "========================================="
log_info " Facial Processing — Failsafe Monitor"
log_info " Project : $PROJECT_DIR"
log_info " Code    : $CODE_DIR"
log_info " Env     : $MAIN_ENV"
log_info " Log     : $LOG_FILE"
log_info "========================================="

# ── One-time startup checks ───────────────────
ensure_log_dir
clean_stale_pidfiles
check_kernel_modules
check_packages
check_devices
ensure_display

# Kill any stale processes from a previous run
sudo pkill -f "facial_processing/Code/main.py" 2>/dev/null || true
sudo pkill -f "rpicam-vid" 2>/dev/null || true
sudo pkill -f "ffmpeg.*video10" 2>/dev/null || true
sleep 1

# Free Flask port in case it's still held
free_port "$FLASK_PORT" "Flask"
sleep 1

# ── Start the application ─────────────────────
start_main_app

log_info "Starting health-check loop (every ${CHECK_INTERVAL}s)..."

LOOP=0
while true; do
    LOOP=$(( LOOP + 1 ))
    log_info "--- Health check #${LOOP} @ $(date '+%H:%M:%S') ---"

    # Process liveness (every loop)
    watch_main_app

    # Camera pipeline (every loop — quick pgrep check)
    check_camera_pipeline

    # Flask liveness (every 3 loops ≈ every 90 s)
    if (( LOOP % 3 == 0 )); then
        check_flask_port
    fi

    # System health (every 6 loops ≈ every 3 min)
    if (( LOOP % 6 == 0 )); then
        check_disk
        check_memory
        check_cpu_temp
    fi

    # Kernel module + device re-check (every 30 loops ≈ every 15 min)
    if (( LOOP % 30 == 0 )); then
        check_kernel_modules
        check_devices
    fi

    # Package integrity re-check (every 120 loops ≈ every hour)
    if (( LOOP % 120 == 0 )); then
        check_packages
    fi

    sleep "$CHECK_INTERVAL"
done