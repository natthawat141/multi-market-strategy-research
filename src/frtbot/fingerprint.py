"""Deterministic dataset fingerprints for derived features/labels (SPEC.md 5.3)."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def fingerprint(obj: Any) -> str:
    """Return a short, stable hash for any JSON-serializable config/object.

    Used to version derived feature/label datasets so a given fingerprint
    always corresponds to the same inputs (config + code-relevant params).
    """
    canonical = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
