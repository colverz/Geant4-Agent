from __future__ import annotations

import argparse
import json
import re
import urllib.request
from datetime import datetime, timezone


DEFAULT_URL = "https://geant4.web.cern.ch/documentation/dev/bfad_html/ForApplicationDevelopers/Analysis/managers.html"
_KNOWN_TYPES = ("csv", "hdf5", "root", "xml")


def fetch_output_formats(url: str) -> list[str]:
    with urllib.request.urlopen(url, timeout=60) as resp:
        html = resp.read().decode("utf-8", errors="ignore")
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    m = re.search(r"The file types supported are ([^.]+)\.", text, flags=re.IGNORECASE)
    if m:
        found = re.findall(r"\b(csv|hdf5|root|xml)\b", m.group(1).lower())
        if found:
            return sorted(set(found))
    found = [fmt for fmt in _KNOWN_TYPES if re.search(rf"\b{fmt}\b", text, flags=re.IGNORECASE)]
    return sorted(set(found))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Geant4 official analysis output formats")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out", default="knowledge/data/output_formats.json")
    args = parser.parse_args()

    official_items = fetch_output_formats(args.url)
    payload = {
        "source": "Official Geant4 analysis manager file types, plus project-local extensions.",
        "sources": [
            {
                "url": args.url,
                "note": "Application Developers Guide: supported analysis manager file types.",
            }
        ],
        "fetched_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "official_items": official_items,
        "project_extensions": ["json"],
        "items": sorted(set(official_items + ["json"])),
        "aliases": {"h5": "hdf5"},
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
