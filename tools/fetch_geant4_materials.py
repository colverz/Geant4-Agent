from __future__ import annotations

import argparse
import json
import re
import urllib.request
from datetime import datetime, timezone
from typing import List


DEFAULT_URL = "https://geant4.web.cern.ch/documentation/dev/bfad_html/ForApplicationDevelopers/Appendix/materialNames.html"


def fetch_materials(url: str) -> List[str]:
    with urllib.request.urlopen(url, timeout=60) as resp:
        html = resp.read().decode("utf-8", errors="ignore")
    names = sorted(set(re.findall(r"G4_[A-Za-z0-9_]+", html)))
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Geant4 NIST material names")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out", default="knowledge/data/materials_geant4_nist.json")
    args = parser.parse_args()

    names = fetch_materials(args.url)
    payload = {
        "source": args.url,
        "fetched_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "materials": names,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
