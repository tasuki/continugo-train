"""Observed champion head-to-head wins, independent of fitted rating scales."""

import json
import re
from pathlib import Path

from .rating_store import GAME_NAME
from .ratings import _atomic_csv

# Parse root properties only: comments may contain escaped brackets and fake REs.
PROPERTY = re.compile(r"\s*([A-Z]+)((?:\s*\[(?:\\.|[^\]\\])*\])+)", re.DOTALL)
VALUE = re.compile(r"\[((?:\\.|[^\]\\])*)\]", re.DOTALL)


def root_properties(path):
    # Neural move annotations can be megabytes; read only through the root.
    with path.open() as stream:
        text = ""
        while True:
            chunk = stream.read(4096)
            text += chunk
            if not text.startswith("(;"):
                raise ValueError(f"invalid SGF root: {path}")
            position = 2
            properties = {}
            while match := PROPERTY.match(text, position):
                if match[1] in properties:
                    raise ValueError(f"duplicate SGF root property: {path}")
                properties[match[1]] = [re.sub(r"\\(.)", r"\1", value, flags=re.DOTALL)
                                        for value in VALUE.findall(match[2])]
                position = match.end()
            if text[position:].lstrip().startswith((";", ")")):
                break
            if not chunk:
                raise ValueError(f"invalid SGF root properties: {path}")
    return properties


def root_result(path):
    values = root_properties(path).get("RE", [])
    if len(values) > 1:
        raise ValueError(f"invalid SGF result: {path}")
    result = values[0] if values else None
    if result not in (None, "", "0", "Void", "?") and not re.fullmatch(
        r"[BW]\+(?:\d+(?:\.\d+)?|R|Resign|T|Time|F|Forfeit)", result
    ):
        raise ValueError(f"unrecognized SGF result {result!r}: {path}")
    return result


def update(artifacts_root):
    """Atomically rebuild from SGFs; blank means no games (or self-pairing).

    Random appears once, first; model labels omit the lineage.
    Draws/truncations count as encounters, but never as wins.
    """
    root = Path(artifacts_root)
    stores = sorted(path.parent for path in root.glob("*/champions.json"))
    random = ("", "random")
    champions = [random]
    for store in stores:
        history = json.loads((store / "champions.json").read_text())["history"]
        champions.extend((store.name, entry["model"]) for entry in history
                         if entry["model"] != "random")
    if len(set(champions)) != len(champions):
        raise ValueError("duplicate champion in promotion history")
    wins, played = {}, set()
    known = set(champions)
    by_name = {}
    for key in champions:
        by_name.setdefault(key[1], []).append(key)

    def identify(store, name):
        if name == "random":
            return random
        local = (store.name, name)
        if local in known:
            return local
        candidates = by_name.get(name, [])
        if len(candidates) > 1:
            raise ValueError(f"ambiguous cross-lineage champion: {name}")
        return candidates[0] if candidates else None

    for store in [*stores, root]:
        for path in sorted((store / "sgfs").glob("*/*.sgf")):
            match = GAME_NAME.fullmatch(path.name)
            if match is None:
                if path.name.startswith("game-"):
                    continue  # An arena may be writing files not yet adopted as rated evidence.
                raise ValueError(f"unexpected rated SGF path: {path}")
            black, white = [identify(store, match[color]) for color in ("black", "white")]
            if black not in known or white not in known:
                continue
            result = root_result(path)
            played.update(((black, white), (white, black)))
            if result and result.startswith(("B+", "W+")):
                pair = (black, white) if result.startswith("B+") else (white, black)
                wins[pair] = wins.get(pair, 0) + 1
    labels = {key: key[1] for key in champions}
    columns = ["champion", *(labels[key] for key in champions)]
    rows = [
        {"champion": labels[row], **{
            labels[col]: wins.get((row, col), 0) if row != col and (row, col) in played else ""
            for col in champions
        }}
        for row in champions
    ]
    output = root / "champion-wins.csv"
    _atomic_csv(output, columns, rows)
    from .champion_wins_view import generate

    generate(output)
    return output
