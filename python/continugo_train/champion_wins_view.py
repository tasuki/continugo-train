"""Self-contained, offline HTML view of the champion wins CSV."""

import csv
import html
import json
import re
from pathlib import Path
from urllib.parse import quote

from .loop import atomic_bytes
from .rating_store import GAME_NAME

STYLE = """
:root{color-scheme:dark;font:14px system-ui,sans-serif;background:#14171d;color:#eee}
body{margin:24px}h1{font-size:22px}p{color:#b9c1ce;max-width:90ch}
#detail{min-height:2.5em;font-weight:600;color:#fff}
.scroll{overflow:auto;max-height:78vh;border:1px solid #414855}
table{border-collapse:separate;border-spacing:0;font-variant-numeric:tabular-nums}
th,td{padding:12px;text-align:center;white-space:nowrap;border-right:1px solid #414855;
border-bottom:1px solid #414855;min-width:64px}
th{background:#252b35}thead th{position:sticky;top:0;z-index:2}
tbody th{position:sticky;left:0;text-align:left;z-index:1}thead th:first-child{left:0;z-index:3}
td{background:#1e242d;cursor:pointer}td.empty{color:#7c8797}
td.chosen{box-shadow:inset 0 0 0 2px #fff}td.diagonal{cursor:default}
#games{margin-top:24px}#games li{margin:10px 0}a{color:#80c7ff}
.path{display:block;color:#929eaf;font-size:12px;overflow-wrap:anywhere}
.pending{color:#929eaf}#games ol{padding-left:24px}
td.diagonal{background:#000;color:#000}td.selected{outline:3px solid #80c7ff;outline-offset:-4px}
td.mirror{outline:3px solid #ffd080;outline-offset:-4px}
.legend{display:flex;align-items:center;gap:10px;margin:16px 0;color:#b9c1ce}
.scale{width:260px;height:14px;background:linear-gradient(to right,#8e0152,#de77ae,#f7f7f7,#7fbc41,#276419)}
th.axis{background:#3b4c60}td:focus-visible{outline:3px solid #fff;outline-offset:-4px}
"""
SCRIPT = """
const table=document.querySelector('table'),detail=document.getElementById('detail');
const hint=detail.textContent;
function clear(){table.querySelectorAll('.selected,.mirror,.axis').forEach(e=>
 e.classList.remove('selected','mirror','axis'));detail.textContent=hint;}
function show(cell){
 clear();if(!cell||!cell.matches('td[data-row]'))return;
 const r=Number(cell.dataset.row),c=Number(cell.dataset.col);
 const mirror=table.tBodies[0].rows[c].cells[r+1];
 cell.classList.add('selected');mirror.classList.add('mirror');
 table.tBodies[0].rows[r].cells[0].classList.add('axis');
 table.tHead.rows[0].cells[c+1].classList.add('axis');
 detail.textContent=cell.getAttribute('aria-label');
}
table.addEventListener('pointerover',e=>show(e.target.closest('td')));
table.addEventListener('pointerleave',()=>show(document.activeElement.closest('td')));
table.addEventListener('focusin',e=>show(e.target.closest('td')));
table.addEventListener('focusout',e=>{if(!table.contains(e.relatedTarget))clear();});
const gameData=JSON.parse(document.getElementById('game-data').textContent);
function games(cell){
 if(!cell||!cell.matches('td[data-row]'))return;
 const r=Number(cell.dataset.row),c=Number(cell.dataset.col);
 table.querySelectorAll('.chosen').forEach(e=>e.classList.remove('chosen'));
 cell.classList.add('chosen');show(cell);
 const list=document.getElementById('game-list');list.replaceChildren();
 const heading=document.getElementById('games-heading');
 heading.textContent=gameData.names[r]+' vs '+gameData.names[c];
 const records=gameData.games[Math.min(r,c)+':'+Math.max(r,c)]||[];
 document.getElementById('game-count').textContent=records.length+
  ' recorded games (both colours, including draws and unfinished games).';
 function link(href,text){const a=document.createElement('a');a.href=href;
  a.textContent=text;a.target='_blank';a.rel='noopener';return a;}
 for(const game of records){
  const li=document.createElement('li');li.append(link(game.sgf,game.name),' · ');
  if(game.nn)li.append(link(game.nn,'nn.html'));
  else{const pending=document.createElement('span');pending.className='pending';
   pending.textContent='NN report unavailable';li.append(pending);}
  const path=document.createElement('code');path.className='path';
  path.textContent=game.path;li.append(path);list.append(li);
 }
 heading.focus({preventScroll:true});
 document.getElementById('games').scrollIntoView({behavior:'smooth',block:'start'});
}
table.addEventListener('click',e=>games(e.target.closest('td')));
table.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){
 const cell=e.target.closest('td[data-row]');if(cell){e.preventDefault();games(cell);}}});
"""


def percentage_style(fraction):
    """Piecewise-linear ColorBrewer PiYG; neutral at 50%, green for row wins."""
    palette = [(142, 1, 82), (222, 119, 174), (247, 247, 247), (127, 188, 65), (39, 100, 25)]
    position = fraction * (len(palette) - 1)
    index = min(int(position), len(palette) - 2)
    weight = position - index
    rgb = tuple(
        round(a + (b - a) * weight) for a, b in zip(palette[index], palette[index + 1], strict=True)
    )
    foreground = (
        "#111"
        if sum(c * w for c, w in zip(rgb, (0.2126, 0.7152, 0.0722), strict=True)) > 140
        else "#fff"
    )
    return f"background:rgb({','.join(map(str, rgb))});color:{foreground}"


def compact_name(name):
    return re.sub(r"-r(\d+)$", lambda match: f"-r{int(match[1])}", name)


def lineage_separators(root, names):
    """Use promotion registries, not model-family changes, to find boundaries."""
    groups = {"random": "random"}
    for path in sorted(root.glob("*/champions.json")):
        for entry in json.loads(path.read_text())["history"]:
            name = entry["model"]
            if name != "random":
                groups[name] = path.parent.name
    return [
        i
        for i in range(1, len(names))
        if groups.get(names[i]) is not None
        and groups.get(names[i - 1]) is not None
        and groups[names[i]] != groups[names[i - 1]]
    ]


def game_index(root, names):
    """Embed paths only, so local-file browsing requires no server or fetch."""
    indices = {name: i for i, name in enumerate(names)}
    games = {}
    paths = [*root.glob("*/sgfs/*/*.sgf"), *root.glob("sgfs/*/*.sgf")]
    for path in sorted(paths, key=lambda p: (p.name, p.as_posix())):
        match = GAME_NAME.fullmatch(path.name)
        if path.is_symlink() or match is None:
            continue
        black, white = match["black"], match["white"]
        if black not in indices or white not in indices or black == white:
            continue
        a, b = sorted((indices[black], indices[white]))
        report = path.with_suffix(".nn.html")
        games.setdefault(f"{a}:{b}", []).append(
            {
                "name": path.name,
                "path": path.relative_to(root.parent).as_posix(),
                "sgf": quote(path.relative_to(root).as_posix(), safe="/"),
                "nn": quote(report.relative_to(root).as_posix(), safe="/")
                if report.is_file() and not report.is_symlink()
                else None,
            }
        )
    return {"names": names, "games": games}


def generate(csv_path):
    csv_path = Path(csv_path)
    with csv_path.open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        rows = list(reader)
    names = header[1:]
    if (
        header[:1] != ["champion"]
        or len(rows) != len(names)
        or any(len(row) != len(header) or row[0] != names[i] for i, row in enumerate(rows))
    ):
        raise ValueError("expected a square champion-wins CSV in matching row/column order")
    for row in rows:
        if any(value and (not value.isascii() or not value.isdigit()) for value in row[1:]):
            raise ValueError("champion wins must be nonnegative integer counts or blank")
    escape = html.escape
    separators = "".join(
        f"tr > :nth-child({index + 2}){{border-left:3px solid #9aa7b9}}"
        f"tbody tr:nth-child({index + 1}) > *{{border-top:3px solid #9aa7b9}}"
        for index in lineage_separators(csv_path.parent, names)
    )
    parts = [
        '<!doctype html><html lang="en"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>Champion head-to-head wins</title><style>",
        STYLE,
        separators,
        "</style>",
        "<h1>Champion head-to-head wins</h1>",
        (
            '<div class="legend"><span>0% · losses</span><span class="scale" aria-hidden="true"></span>'
            "<span>100% · wins (50% neutral)</span></div>"
        ),
        '<div id="detail" role="status" aria-live="polite">Hover or focus a matchup.</div>',
        '<div class="scroll"><table aria-label="Champion head-to-head wins">',
        '<thead><tr><th scope="col">Opponent →</th>',
    ]
    parts.extend(f'<th scope="col">{escape(compact_name(name))}</th>' for name in names)
    parts.append("</tr></thead><tbody>")
    for r, row in enumerate(rows):
        parts.append(f'<tr><th scope="row">{escape(compact_name(row[0]))}</th>')
        for c, value in enumerate(row[1:]):
            if r == c:
                parts.append('<td class="diagonal" aria-label="Self matchup"></td>')
                continue
            reverse = rows[c][r + 1]
            label = (
                f"{row[0]} wins: {value} · {names[c]} wins: {reverse}"
                if value != "" and reverse != ""
                else f"{row[0]} vs {names[c]}: no recorded pairing"
            )
            total = int(value) + int(reverse) if value != "" and reverse != "" else 0
            fraction = int(value) / total if total else None
            display = f"{100 * fraction:.1f}%" if fraction is not None else "—"
            style = percentage_style(fraction) if fraction is not None else ""
            if fraction is not None:
                label += f" · row share: {display} of {total} decisive games"
            elif value != "" and reverse != "":
                label += " · no decisive games (percentage undefined)"
            parts.append(
                f'<td data-row="{r}" data-col="{c}" tabindex="0" '
                f'class="{"empty" if fraction is None else "count"}" style="{style}" '
                f'title="{escape(label, quote=True)}" aria-label="{escape(label, quote=True)}">'
                f"{display}</td>"
            )
        parts.append("</tr>")
    data = json.dumps(game_index(csv_path.parent, names)).replace("<", "\\u003c")
    parts.extend(
        [
            '</tbody></table></div><section id="games" aria-labelledby="games-heading">',
            '<h2 id="games-heading" tabindex="-1">Click a matchup to list its games</h2>',
            '<p id="game-count"></p><ol id="game-list"></ol></section>',
            '<script id="game-data" type="application/json">',
            data,
            "</script><script>",
            SCRIPT,
            "</script></html>",
        ]
    )
    output = csv_path.with_suffix(".html")
    atomic_bytes(output, "".join(parts).encode())
    return output
