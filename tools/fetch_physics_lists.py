from __future__ import annotations

import argparse
import json
import re
import urllib.request
from datetime import datetime, timezone
from typing import List


DEFAULT_URL = "https://geant4.web.cern.ch/documentation/dev/plg_html/PhysicsListGuide/physicslistguide.html"


def fetch_physics_lists(url: str) -> List[str]:
    with urllib.request.urlopen(url, timeout=60) as resp:
        html = resp.read().decode("utf-8", errors="ignore")
    # Extract reference lists from the section text
    idx = html.find("The following reference Physics Lists are available")
    block = html[idx:idx + 3000] if idx != -1 else html
    items = re.findall(r"<span class=\"(?:pre|std std-ref)\">([^<]+)</span>", block)
    items = [i.strip() for i in items if i.strip()]
    if items:
        blacklist = {"physics_lists", "G4PhysListFactory", "Geant4"}
        items = [i for i in items if i not in blacklist]
        return sorted(set(items))
    # Fallback: scan for list-like tokens
    items = re.findall(r">\s*([A-Za-z0-9_]+)\s*<", block)
    return sorted(set(items))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Geant4 reference physics lists")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out", default="knowledge/data/physics_lists.json")
    args = parser.parse_args()

    items = fetch_physics_lists(args.url)
    payload = {
        "source": args.url,
        "fetched_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "items": items,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
