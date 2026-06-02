from __future__ import annotations

import argparse
import json
import re
import urllib.request
from datetime import datetime, timezone
from typing import List


BASE_URL = "https://geant4-userdoc.web.cern.ch/UsersGuides/ForApplicationDeveloper/html/AllResources/TrackingAndPhysics/particleList.src"
CATEGORIES = ["quarks", "leptons", "mesons", "baryons", "ions", "others"]


def fetch_category(url: str) -> List[str]:
    with urllib.request.urlopen(url, timeout=60) as resp:
        html = resp.read().decode("utf-8", errors="ignore")
    names = re.findall(r"<A HREF=\"[^\"]+\">([^<]+)</A>", html, flags=re.I)
    return [n.strip() for n in names if n.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Geant4 particle names")
    parser.add_argument("--out", default="knowledge/data/particles.json")
    args = parser.parse_args()

    items = []
    for cat in CATEGORIES:
        url = f"{BASE_URL}/{cat}/index.html"
        items.extend(fetch_category(url))

    items = sorted(set(items))
    payload = {
        "source": BASE_URL,
        "fetched_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "items": items,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
