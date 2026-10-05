"""Fail closed when an event contains more than one row per golfer."""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Any, Iterable


def _text(value: Any) -> str:
    text = str(value).strip() if value is not None else ""
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def _name(value: Any) -> str:
    text = _text(value)
    if "," in text:
        last, first = text.split(",", 1)
        text = f"{first} {last}"
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", text.lower())


def validate_unique_players(rows: Iterable[dict[str, Any]], *, context: str, require_ids: bool = False) -> None:
    """Reject duplicates rather than selecting arbitrary conflicting predictions.

    ID checks catch renamed players; normalized-name checks also protect exports
    which omit IDs. No outcomes or probabilities influence the decision.
    """
    ids: dict[str, list[int]] = defaultdict(list)
    names: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows, start=1):
        player_id = next((_text(row.get(key)) for key in ("dg_id", "player_id", "id") if _text(row.get(key))), "")
        player_id = re.sub(r"\.0+$", "", player_id)
        name = next((_name(row.get(key)) for key in ("player_name", "player", "Player", "Name") if _name(row.get(key))), "")
        if require_ids and not player_id:
            raise ValueError(f"{context}: missing player ID at row {index}; refusing to simulate an unidentified player")
        if not player_id and not name:
            raise ValueError(f"{context}: missing player identity at row {index}")
        if player_id:
            ids[player_id].append(index)
        if name:
            names[name].append(index)
    conflicts = [f"{kind} {key!r} at rows {positions}" for kind, groups in (("ID", ids), ("name", names)) for key, positions in groups.items() if len(positions) > 1]
    if conflicts:
        raise ValueError(f"{context}: duplicate players: {'; '.join(conflicts[:10])}. Fix the field or join upstream; do not silently drop conflicting predictions.")
