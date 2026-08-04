"""
monitor_resolver.py

Löst das Problem, dass DVI-I-1 / DVI-I-2 (die beiden DisplayLink-Ausgänge)
bei jedem Boot/Reconnect vertauscht erkannt werden können.

Statt Positionen fest an Connector-Namen zu binden, werden sie hier an
die EDID-Beschreibung (Hersteller/Modell/Seriennummer) gebunden und beim
Programmstart per `wlr-randr` auf den JEWEILS AKTUELLEN Connector-Namen
aufgelöst.

Voraussetzung: identify_and_configure_monitors.py wurde einmal ausgeführt
und hat ~/.config/monitor_map.json erzeugt (EDID-Beschreibung -> Position).

Nutzung:
        from monitor_resolver import resolve_monitor_positions
        MONITOR_POSITIONS = resolve_monitor_positions()

    Statt der bisherigen fest verdrahteten Liste.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

MONITOR_MAP_JSON = Path.home() / ".config" / "monitor_map.json"


def _load_monitor_map() -> dict:
    if not MONITOR_MAP_JSON.exists():
        print(
            f"FEHLER: {MONITOR_MAP_JSON} nicht gefunden. "
            f"Bitte zuerst identify_and_configure_monitors.py ausführen.",
            file=sys.stderr,
        )
        return {}
    with open(MONITOR_MAP_JSON) as f:
        return json.load(f)  # {edid_desc: {"position": [x,y], ...}}


def _get_wlr_randr_output() -> str:
    try:
        result = subprocess.run(
            ["wlr-randr"], capture_output=True, text=True, timeout=10, check=True
        )
        return result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"WARNUNG: wlr-randr konnte nicht ausgeführt werden ({e}).", file=sys.stderr)
        return ""


def _parse_outputs(raw: str) -> dict[str, str]:
    """Gibt {connector_name: edid_beschreibung} zurück, z.B.
    {'DVI-I-1': 'Club3D Inc. XYZ 0x00001234 (DVI-I-1)', ...}
    """
    outputs = {}
    for line in raw.splitlines():
        m = re.match(r'^(\S+)\s+"(.*)"', line)
        if m:
            name, desc = m.group(1), m.group(2)
            outputs[name] = desc
    return outputs


def resolve_monitor_positions() -> dict[str, tuple[int, int]]:
    """
    Liefert {aktueller_connector_name: (x, y)}, passend zum GERADE
    erkannten Zustand -- unabhängig davon, ob DVI-I-1/DVI-I-2 diesmal
    vertauscht wurden. Quelle der Wahrheit: ~/.config/monitor_map.json
    (erzeugt von identify_and_configure_monitors.py).
    """
    monitor_map = _load_monitor_map()
    if not monitor_map:
        return {}

    raw = _get_wlr_randr_output()
    current_outputs = _parse_outputs(raw)  # connector_name -> desc

    resolved: dict[str, tuple[int, int]] = {}

    for desc, cfg in monitor_map.items():
        pos = tuple(cfg["position"])
        matched_name = None
        for name, current_desc in current_outputs.items():
            if desc == current_desc or desc in current_desc:
                matched_name = name
                break

        if matched_name is None:
            print(f"WARNUNG: Monitor mit Beschreibung '{desc}' aktuell nicht angeschlossen.", file=sys.stderr)
            continue

        resolved[matched_name] = pos

    return resolved


if __name__ == "__main__":
    # Quick test: python3 monitor_resolver.py
    positions = resolve_monitor_positions()
    print("Aufgelöste Monitor-Positionen:")
    for name, pos in positions.items():
        print(f"  {name}: {pos}")