#!/bin/bash
# ============================================================
# DisplayLink Dual Monitor Setup for Raspberry Pi 5
# Club3D USB-A 3.0 Dual 4K Dock
# ============================================================
# Run with: bash displaylink_setup.sh
# Make sure dock is plugged into a BLUE USB 3.0 port on the Pi!
# ============================================================

set -e

echo "========================================"
echo " DisplayLink Setup for Raspberry Pi 5"
echo "========================================"

# ── STEP 1: Install DisplayLink driver (includes evdi via apt) ─
echo ""
echo "[1/4] Installing DisplayLink driver..."

wget -q https://www.synaptics.com/sites/default/files/Ubuntu/pool/stable/main/all/synaptics-repository-keyring.deb
sudo dpkg -i synaptics-repository-keyring.deb
sudo apt update -q

# Clean up any existing DKMS evdi entries to avoid conflicts
echo "Cleaning up existing DKMS evdi entries..."
for ver in 1.14.15 1.14.16 1.14.17; do
    sudo dkms remove evdi/$ver --all 2>/dev/null || true
done

# Remove manually built evdi if present
sudo rm -f /lib/modules/$(uname -r)/kernel/drivers/gpu/drm/evdi/evdi.ko.xz

sudo apt install -y displaylink-driver || {
    echo "Fixing dependency issues..."
    sudo apt --fix-broken install -y
    sudo dpkg --configure -a || {
        # Force install displaylink if evdi apt package still broken
        DL_DEB=$(ls /var/cache/apt/archives/displaylink-driver*.deb 2>/dev/null | head -1)
        if [ -n "$DL_DEB" ]; then
            sudo dpkg --ignore-depends=evdi,libevdi1 -i "$DL_DEB"
        fi
    }
}

sudo depmod -a
sudo modprobe evdi || true
sudo systemctl restart displaylink-driver || true

echo "✓ DisplayLink driver installed"

# ── STEP 2: Remove udl from evdi softdep ─────────────────────
echo ""
echo "[2/4] Fixing evdi.conf (removing udl softdep)..."

EVDI_CONF="/etc/modprobe.d/evdi.conf"
SOFTDEP_LINE="softdep evdi pre: drm_dma_helper drm_shmem_helper v3d vc4"

if [ ! -f "$EVDI_CONF" ]; then
    echo "evdi.conf not found, creating it..."
    echo "$SOFTDEP_LINE" | sudo tee "$EVDI_CONF"
elif grep -q "softdep evdi" "$EVDI_CONF"; then
    sudo sed -i "s/softdep evdi pre:.*/$SOFTDEP_LINE/" "$EVDI_CONF"
else
    echo "$SOFTDEP_LINE" | sudo tee -a "$EVDI_CONF"
fi

echo "✓ evdi.conf fixed"

# ── STEP 3: Blacklist udl ────────────────────────────────────
echo ""
echo "[3/4] Blacklisting udl kernel driver..."

echo "blacklist udl" | sudo tee /etc/modprobe.d/blacklist-udl.conf
echo "install udl /bin/false" | sudo tee -a /etc/modprobe.d/blacklist-udl.conf

if ! grep -q "modprobe.blacklist=udl" /boot/firmware/cmdline.txt; then
    sudo sed -i 's/$/ modprobe.blacklist=udl/' /boot/firmware/cmdline.txt
fi

echo "✓ udl blacklisted"

# ── STEP 4: Apply Wayland/labwc fixes ────────────────────────
echo ""
echo "[4/4] Applying Wayland fixes..."

sudo sed -i 's/WLR_DRM_FORCE_LIBLIFTOFF=1/WLR_DRM_FORCE_LIBLIFTOFF=0/' /etc/xdg/labwc/environment

sudo mkdir -p /etc/systemd/system/displaylink-driver.service.d
cat << OVERRIDE | sudo tee /etc/systemd/system/displaylink-driver.service.d/override.conf
[Service]
ExecStartPre=/bin/sleep 15
OVERRIDE

sudo systemctl daemon-reload

echo "✓ Wayland fixes applied"

# ── Done ─────────────────────────────────────────────────────
echo ""
echo "========================================"
echo " Setup complete! Rebooting in 5s..."
echo " Both monitors should come on after boot"
echo "========================================"
sleep 5
sudo reboot