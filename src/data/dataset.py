"""
KTH dataset metadata: discover AVI files and parse subject, action, scenario.
Supports flat or action-subfolder layout (e.g. data/raw/walking/*.avi).
"""
from pathlib import Path
import re
from typing import List, Tuple, Optional

import pandas as pd

# Canonical action names and order (for consistent label indices)
ACTION_NAMES = ["walking", "jogging", "running", "boxing", "handwaving", "handclapping"]

# Filename patterns: personXX_action_dY_* or personXX_action_sY_* or personXX_action_scenario_*
_PERSON_ACTION_SCENARIO = re.compile(
    r"person(\d+)[_\s]+(\w+)[_\s]+[ds](\d)[_\w]*\.avi",
    re.IGNORECASE,
)
# Fallback: at least person + action
_PERSON_ACTION = re.compile(
    r"person(\d+)[_\s]+(\w+)[_\s\.].*\.avi",
    re.IGNORECASE,
)


def _normalize_action(name: str) -> str:
    name = name.lower().replace(" ", "").replace("-", "")
    if "handwav" in name or "hand_wav" in name:
        return "handwaving"
    if "handclap" in name or "hand_clap" in name:
        return "handclapping"
    for a in ACTION_NAMES:
        if a in name or name in a:
            return a
    return name


def _parse_filename(path: Path) -> Optional[Tuple[int, str, int]]:
    name = path.name
    m = _PERSON_ACTION_SCENARIO.search(name)
    if m:
        person_id = int(m.group(1))
        action = _normalize_action(m.group(2))
        scenario = int(m.group(3))
        if action in ACTION_NAMES:
            return (person_id, action, scenario)
    m = _PERSON_ACTION.search(name)
    if m:
        person_id = int(m.group(1))
        action = _normalize_action(m.group(2))
        if action in ACTION_NAMES:
            return (person_id, action, 1)
    return None


def load_metadata(data_dir: str | Path) -> pd.DataFrame:
    """
    Scan data_dir (recursively) for .avi files and build metadata table.
    Returns DataFrame with columns: video_path, subject_id, scenario, action, action_id.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return pd.DataFrame(columns=["video_path", "subject_id", "scenario", "action", "action_id"])

    rows = []
    for path in data_dir.rglob("*.avi"):
        parsed = _parse_filename(path)
        if parsed is None:
            continue
        person_id, action, scenario = parsed
        action_id = ACTION_NAMES.index(action)
        rows.append({
            "video_path": str(path.resolve()),
            "subject_id": person_id,
            "scenario": scenario,
            "action": action,
            "action_id": action_id,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["subject_id", "action", "scenario"]).reset_index(drop=True)
    return df


def get_action_label_to_id() -> dict:
    """Return mapping action_name -> action_id (0..5)."""
    return {a: i for i, a in enumerate(ACTION_NAMES)}
