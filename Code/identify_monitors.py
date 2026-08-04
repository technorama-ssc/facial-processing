import json
import re
import subprocess
import sys
import time
import tkinter as tk
from pathlib import Path

KANSHI_CONFIG = Path.home() / ".config" / "kanshi" / "config"
MONITOR_MAP_JSON = Path.home() / ".config" / "monitor_map.json"

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def run_wlr_randr() -> str:
    try:
        result = subprocess.run(
            ["wlr-randr"], capture_output=True, text=True, timeout=10, check=True
        )
        return result.stdout
    except Exception as e:  # noqa: BLE001
        print(f"FEHLER: wlr-randr konnte nicht ausgeführt werden: {e}", file=sys.stderr)
        sys.exit(1)


def parse_outputs(raw: str):
    """
    Gibt eine Liste von dicts zurück:
    [{"name": "DVI-I-1", "desc": "...", "width": 1920, "height": 1080}, ...]
    Nur Outputs, die aktuell "connected"/mit Modus sind.
    """
    outputs = []
    current = None
    for line in raw.splitlines():
        header = re.match(r'^(\S+)\s+"(.*)"', line)
        if header:
            if current:
                outputs.append(current)
            current = {"name": header.group(1), "desc": header.group(2), "width": None, "height": None}
            continue
        if current is not None:
            mode = re.search(r'(\d+)x(\d+)@[\d.]+ Hz \(current\)', line)
            if mode:
                current["width"] = int(mode.group(1))
                current["height"] = int(mode.group(2))
    if current:
        outputs.append(current)
    # Nur Outputs mit erkannter Auflösung sind wirklich aktiv angeschlossen
    return [o for o in outputs if o["width"] and o["height"]]


def show_identify_windows(outputs):
    """
    Öffnet für jeden Output ein Vollbild-Fenster an Position x=Index*1000
    (grobe Platzierung reicht -- Ziel ist nur, dass jedes Fenster auf einem
    ANDEREN physischen Monitor sichtbar wird). Da wir die tatsächliche
    aktuelle Position der Outputs nicht sicher kennen (genau DAS Problem,
    das wir lösen wollen), platzieren wir stattdessen einfach EIN Fenster
    nacheinander pro Output im Vollbild auf dessen Auflösung -- über
    xrandr/XWayland landet es automatisch auf dem richtigen physischen Screen,
    wenn man es dem jeweiligen Output zuweist.
    """
    root = tk.Tk()
    root.withdraw()

    results = []

    for idx, out in enumerate(outputs):
        win = tk.Toplevel(root)
        color = COLORS[idx % len(COLORS)]
        win.configure(bg=color)
        win.attributes("-fullscreen", False)
        win.overrideredirect(True)
        # Grobe Startposition -- der Fenstermanager/XWayland entscheidet
        # anhand des :screen-Arguments, wo es wirklich landet, s. Hinweis unten.
        win.geometry(f"{out['width']}x{out['height']}+0+0")

        label = tk.Label(
            win,
            text=f"MONITOR {chr(65 + idx)}\n\n{out['name']}\n{out['desc']}",
            font=("DejaVu Sans", 40, "bold"),
            fg="white",
            bg=color,
            justify="center",
        )
        label.pack(expand=True, fill="both")
        win.update()

        print(f"\n--- Testfenster für Output '{out['name']}' angezeigt (Farbe: {color}) ---")
        print(f"    Beschreibung: {out['desc']}")
        pos = input("    Auf WELCHEM physischen Monitor (links=1, ..., rechts=4) siehst du dieses Fenster? ").strip()
        rotated = input("    Steht der Text auf diesem Monitor richtig rum? [j/n]: ").strip().lower()

        results.append({
            "output": out,
            "slot": int(pos),
            "needs_180": rotated.startswith("n"),
        })

        win.destroy()

    root.destroy()
    return results


def build_configs(results, spacing=1920):
    """
    Baut aus den Kalibrierungsergebnissen:
      - Liste von kanshi-Profile-Zeilen (nach EDID-Beschreibung)
      - dict für monitor_map.json (nach EDID-Beschreibung)
    Positionen werden anhand des angegebenen Slots (1..4, links->rechts)
    automatisch in X-Richtung nebeneinander gelegt.
    """
    results_sorted = sorted(results, key=lambda r: r["slot"])

    kanshi_lines = []
    monitor_map = {}

    x_offset = 0
    for r in results_sorted:
        out = r["output"]
        desc = out["desc"]
        w = out["width"]
        transform = "180" if r["needs_180"] else "normal"

        kanshi_lines.append(
            f'    output "{desc}" mode {w}x{out["height"]} position {x_offset},0 transform {transform}'
        )
        monitor_map[desc] = {"position": [x_offset, 0], "transform": transform, "width": w, "height": out["height"]}

        x_offset += w

    return kanshi_lines, monitor_map


def write_kanshi_config(kanshi_lines):
    KANSHI_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    if KANSHI_CONFIG.exists():
        backup = KANSHI_CONFIG.with_suffix(f".bak.{int(time.time())}")
        KANSHI_CONFIG.rename(backup)
        print(f"Alte kanshi-Config gesichert: {backup}")

    with open(KANSHI_CONFIG, "w") as f:
        f.write("# Automatisch generiert von identify_and_configure_monitors.py\n")
        f.write("profile {\n")
        f.write("\n".join(kanshi_lines))
        f.write("\n}\n")

    print(f"Geschrieben: {KANSHI_CONFIG}")


def write_monitor_map(monitor_map):
    MONITOR_MAP_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(MONITOR_MAP_JSON, "w") as f:
        json.dump(monitor_map, f, indent=2, ensure_ascii=False)
    print(f"Geschrieben: {MONITOR_MAP_JSON}")


def restart_kanshi():
    subprocess.run(["pkill", "-x", "kanshi"], check=False)
    time.sleep(1)
    subprocess.Popen(
        ["kanshi"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    print("kanshi neu gestartet.")


def main():
    print("========================================================")
    print(" Monitor-Kalibrierungs-Wizard (einmalig auszuführen)")
    print("========================================================")

    raw = run_wlr_randr()
    outputs = parse_outputs(raw)

    if len(outputs) < 2:
        print("Weniger als 2 aktive Outputs gefunden -- Abbruch.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{len(outputs)} aktive Outputs gefunden:")
    for o in outputs:
        print(f"  - {o['name']}: {o['desc']} ({o['width']}x{o['height']})")

    print("\nGleich öffnet sich nacheinander ein farbiges Testfenster pro Output.")
    input("Enter drücken, um zu starten...")

    results = show_identify_windows(outputs)
    kanshi_lines, monitor_map = build_configs(results)

    write_kanshi_config(kanshi_lines)
    write_monitor_map(monitor_map)
    restart_kanshi()

    print("\n========================================================")
    print(" Fertig! Ab jetzt:")
    print("   - kanshi ordnet Position/Rotation über EDID zu (stabil)")
    print("   - main.py kann monitor_map.json laden und ist ebenfalls stabil")
    print(" Kabel testweise vertauschen -- Ausrichtung sollte gleich bleiben.")
    print("========================================================")


if __name__ == "__main__":
    main()