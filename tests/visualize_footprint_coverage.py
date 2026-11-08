"""Manual visual audit of Rust test output; no rules or raster recomputation.

Run after exporting CONTINUGO_FOOTPRINT_COVERAGE_REPORT from the Rust coverage
suite: python tests/visualize_footprint_coverage.py REPORT.json OUTPUT.html.
"""

import argparse
import html
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("report", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--force", action="store_true", help="replace an existing visual report")
ARGS = parser.parse_args()
if ARGS.output.exists() and not ARGS.force:
    parser.error(f"output already exists: {ARGS.output}")
DATA = json.loads(ARGS.report.read_text())
assert DATA["passed"]
SIDE, ORIGIN, DENSITY = DATA["side"], DATA["origin"], DATA["samples_per_radius"]
CAPPED = DATA.get("feature_cap")


def esc(value):
    return html.escape(str(value), quote=True)


def runs(rows):
    for y, encoded in enumerate(rows):
        bits = int(encoded, 16)
        x = 0
        while bits:
            while not (bits & 1):
                bits >>= 1
                x += 1
            start = x
            while bits & 1:
                bits >>= 1
                x += 1
            yield y, start, x - start


def bitmap(f, field, color, opacity=1):
    step = f["radius"] / DENSITY
    px, py = f["candidate"]
    return "".join(
        f'<rect x="{px + (x - ORIGIN - 0.5) * step:.6f}" y="{py + (y - ORIGIN - 0.5) * step:.6f}" '
        f'width="{length * step:.6f}" height="{step:.6f}" fill="{color}" opacity="{opacity}" shape-rendering="crispEdges"/>'
        for y, x, length in runs(f[field])
    )


def reference(f):
    r = f["radius"]
    result = []
    for y, left, right in f["spans"]:
        if left == right:
            result.append(f'<circle cx="{left}" cy="{y}" r="{r}" fill="#47d7b1"/>')
        else:
            result.append(
                f'<path d="M{left} {y}H{right}" stroke="#47d7b1" stroke-width="{2 * r}" stroke-linecap="round"/>'
            )
    return "".join(result)


def svg(f, mode, zoom=False):
    r = f["radius"]
    px, py = f["candidate"]
    step = r / DENSITY
    if zoom:
        view = f["zoom"]
    elif f["id"] in ("corner-eye", "correct-zero"):
        view = [-34, -34, 1068, 1068]
    else:
        view = [px - 4 * r, py - 4 * r, 8 * r, 8 * r]
    vx, vy, width, height = view
    label = {
        "centers": "Exact integer centers",
        "reference": "Continuous footprint",
        "raster": "Full verification bitset" if CAPPED is not None else "Production bitset",
    }[mode]
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vx} {vy} {width} {height}" role="img" '
            f'aria-label="{esc(f["title"] + ": " + label)}" preserveAspectRatio="xMidYMid meet">'
        ),
        f'<rect x="{vx}" y="{vy}" width="{width}" height="{height}" fill="#0b1220"/>',
        '<rect x="0" y="0" width="1000" height="1000" fill="#1b293b"/>',
    ]
    if not zoom:
        # Context only: physical candidate vs its center-adjacency annulus.
        for radius, stroke in ((2 * r, "#55759a"), ((8**0.5) * r, "#55759a")):
            parts.append(
                f'<circle cx="{px}" cy="{py}" r="{radius}" fill="none" stroke="{stroke}" '
                'stroke-dasharray="3 4" vector-effect="non-scaling-stroke" stroke-width=".7"/>'
            )
        parts.append(
            f'<rect x="{r}" y="{r}" width="{1000 - 2 * r}" height="{1000 - 2 * r}" '
            'fill="none" stroke="#668199" stroke-dasharray="4 4" vector-effect="non-scaling-stroke" stroke-width=".7"/>'
        )
    if mode == "reference":
        parts.append(reference(f))
    elif mode == "raster":
        parts.append(bitmap(f, "footprint", "#ffad4d"))
    for x, y in f["stones"]:
        parts.append(
            f'<circle cx="{x}" cy="{y}" r="{r}" fill="#68798b" fill-opacity=".72" stroke="#aab9c9" '
            'vector-effect="non-scaling-stroke" stroke-width=".7"/>'
        )
    parts.append(
        f'<circle cx="{px}" cy="{py}" r="{r}" fill="#172948" fill-opacity=".4" stroke="#93b7ff" '
        'vector-effect="non-scaling-stroke" stroke-width="1.5"/>'
    )
    if mode in ("centers", "reference"):
        for y, left, right in f["spans"]:
            if left == right:
                parts.append(
                    f'<circle cx="{left}" cy="{y}" r="{width / 180 * 1.15}" fill="#ff658c"/>'
                )
            else:
                parts.append(
                    f'<path d="M{left} {y}H{right}" stroke="#ff658c" vector-effect="non-scaling-stroke" stroke-width="1.4"/>'
                )
    else:
        parts.append(bitmap(f, "centers", "#fff4be", 0.95))
        if zoom:
            for i in range(SIDE + 1):
                x = px + (i - ORIGIN - 0.5) * step
                y = py + (i - ORIGIN - 0.5) * step
                parts.append(
                    f'<path d="M{x} {vy}V{vy + height} M{vx} {y}H{vx + width}" stroke="#162131" '
                    'vector-effect="non-scaling-stroke" stroke-width=".4"/>'
                )
        if f["id"] == "off-grid" and not zoom:
            x = px - step * 0.5
            parts.append(
                f'<path d="M{x} {vy}V{vy + height}" stroke="#a499ff" stroke-dasharray="3 4" '
                'vector-effect="non-scaling-stroke" stroke-width=".8"/>'
            )
    if f.get("highlight") and not zoom:
        x, y = f["highlight"]
        # Ring deliberately exaggerates visibility, never modifies raster data.
        parts.append(
            f'<circle cx="{x}" cy="{y}" r="{width / 180 * 5}" fill="none" stroke="#fff" '
            'vector-effect="non-scaling-stroke" stroke-width="1.2"/>'
        )
    if not f["spans"]:
        parts.append(
            f'<text x="{vx + width / 2}" y="{vy + height / 2}" fill="#fff" font-size="{width * 0.06}" '
            'text-anchor="middle">0 local centers</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


cards = []
for index, f in enumerate(DATA["fixtures"], 1):
    figures = "".join(
        f"<figure>{svg(f, mode)}<figcaption>{caption}</figcaption></figure>"
        for mode, caption in [
            ("centers", "1 · Exact centers"),
            ("reference", "2 · Continuous union"),
            (
                "raster",
                "3 · Full raster before cap" if CAPPED is not None else "3 · Actual raster union",
            ),
        ]
    )
    detail = ""
    if f.get("zoom"):
        detail = (
            '<div class="insets">'
            + "".join(
                f"<figure>{svg(f, mode, True)}<figcaption>{caption}</figcaption></figure>"
                for mode, caption in [
                    ("reference", "Reference · enlarged"),
                    ("raster", "Raster · enlarged"),
                ]
            )
            + "</div>"
        )
    legal = (
        ""
        if f["candidate_legal"] is None
        else f" · candidate {'legal' if f['candidate_legal'] else 'illegal'}"
    )
    cards.append(f'''<article id="{esc(f["id"])}"><h2>{index}. {esc(f["title"])}</h2>
<small class="kind">{esc(f["kind"])}{legal}</small><p class="note">{esc(f["note"])}</p>
<div class="views">{figures}</div>{detail}
<div class="stats"><strong class="pass">PASS · 0 dropped centers</strong><span>{f["exact_centers"]:,} exact centers → {f["occupied_center_bins"]:,} occupied bins</span>
<span>Area: reference {f["reference_area_discs"]:.4f} / raster {f["raster_area_discs"]:.4f} discs</span>
<span>Production input: {f.get("feature_value_discs", f["raster_area_discs"]):.4f} discs</span></div></article>''')

PAGE = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Local footprint coverage tests · actual Rust output</title><style>
:root{color-scheme:dark;font:14px system-ui;background:#0c1420;color:#dae5f4}*{box-sizing:border-box}body{margin:0;padding:16px;max-width:2100px;margin:auto}
h1{font-size:24px;margin:0 0 6px}h2{font-size:15px;margin:0 0 3px}p{line-height:1.4}header{border:1px solid #4d687b;border-radius:8px;padding:14px;margin-bottom:12px;background:#162438}
header p{margin:6px 0}.summary{display:flex;gap:18px;flex-wrap:wrap;font-weight:650;color:#9de6c4}.legend{display:flex;gap:14px;flex-wrap:wrap}.dot{font-weight:750}.warn{color:#ffd38f}.muted,small{color:#9fb1c5}.cases{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
article{border:1px solid #3a5068;border-radius:8px;padding:10px;background:#131f30;min-width:0}.kind{font-size:11px}.note{font-size:12px;min-height:50px;margin:5px 0 8px}.views{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px}figure{margin:0;min-width:0}figure svg{display:block;width:100%;aspect-ratio:1;background:#0b1220;border:1px solid #3d5266;border-radius:4px;cursor:zoom-in}figcaption{font-size:11px;margin-top:3px;color:#c5d5e6}
.stats{display:grid;gap:3px;font-size:12px;margin-top:8px}.pass{color:#8de8bd}.insets{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px}.insets svg{aspect-ratio:2.2;max-height:110px}.insets figcaption{font-size:10px}footer{font-size:12px;line-height:1.5;padding-top:12px;color:#aebdd0}
dialog{width:min(94vw,1000px);max-height:96vh;background:#131f30;color:#fff;border:1px solid #7a90a8;border-radius:8px}dialog svg{display:block;width:100%;max-height:82vh}dialog::backdrop{background:#000b}button{background:#263c55;color:white;border:1px solid #738ba5;border-radius:5px;padding:5px 12px;cursor:pointer}
@media(max-width:1300px){.cases{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:800px){.cases{grid-template-columns:1fr}}a{color:#a8ceff}
</style><header><h1>Do local footprint inputs miss tiny spots?</h1><p><strong>Actual passing Rust test fixtures.</strong> Exact integer centers are projected, not discovered by sampling. Every supplied exact center must keep a raster bit and produce footprint coverage.</p>
<div class="summary">SUMMARY</div><p class="legend"><span class="dot" style="color:#ff658c">Red: exact centers</span><span class="dot" style="color:#47d7b1">Green: continuous footprint</span><span class="dot" style="color:#ffad4d">Orange: actual production raster</span><span>Gray: fixed stones · blue: candidate · pale pixels: center bins</span></p>
<p class="muted">Point/line markers are enlarged for visibility, not blurred inputs. Click any panel to enlarge. Real fixtures also pass exhaustive independent fixed-obstacle checks; synthetic spans isolate binning/dilation tests.</p>
<p class="warn"><strong>Precise limit:</strong> nearby centers or tiny gaps can merge at raster resolution. A one-point pocket does not disappear, but this is not a guarantee of preserving topology, exact shapes, or all moves being selected by the coarse policy.</p></header><main class="cases">CARDS</main>
<footer>Source: <code>src/features/footprint/coverage_tests.rs</code>. Fixtures call production <code>project_span</code>, <code>dilate</code> and (for real geometry) <code>area</code>. Disc unions are checked against an independent pixel-by-pixel stamping oracle. Continuous panels are physical-radius capsules drawn from the exact test spans. Raster panels use exported production bits, not a Python reconstruction. The off-grid eye's red center is (500,500), relative offset (-66,-88) from the candidate. Existing isolated-disc, area-accuracy, collision-safe cache and serial/parallel tests remain separate.</footer>
<dialog id="zoom"><button id="close">Close</button><h2 id="zoomtitle"></h2><div id="zoomview"></div></dialog>
<script id="data" type="application/json">DATA</script><script>
const data=JSON.parse(document.querySelector('#data').textContent);for(const svg of document.querySelectorAll('article svg'))svg.addEventListener('click',()=>{const figure=svg.closest('figure'),card=svg.closest('article');document.querySelector('#zoomtitle').textContent=card.querySelector('h2').textContent+' · '+figure.querySelector('figcaption').textContent;document.querySelector('#zoomview').replaceChildren(svg.cloneNode(true));document.querySelector('#zoom').showModal()});document.querySelector('#close').onclick=()=>document.querySelector('#zoom').close();
</script></html>"""
if CAPPED is not None:
    PAGE = PAGE.replace("Orange: actual production raster", "Orange: full verification raster")
    PAGE = PAGE.replace(
        '<p class="warn">',
        f'<p class="muted"><strong>Production input cap: {CAPPED:g} discs.</strong> Full footprints are shown to verify coverage. The capped fast path must equal min(full raster, {CAPPED:g}); small pockets are unchanged.</p><p class="warn">',
        1,
    )
    PAGE = PAGE.replace(
        "Raster panels use exported production bits, not a Python reconstruction.",
        "Raster panels use exported full verification bits from the shared kernel, not a Python reconstruction. The production input uses certified early saturation and is checked against the clipped full result.",
    )
summary = (
    f"<span>PASS · {len(DATA['fixtures'])} fixtures</span>"
    f"<span>{DATA['axis_phases_checked']:,} radius/axis phases</span>"
    f"<span>{DATA['single_center_positions_checked']:,} single-center raster positions</span>"
    "<span>4 corner reflections</span><span>0 dropped centers</span>"
)
output = (
    PAGE.replace("SUMMARY", summary)
    .replace("CARDS", "".join(cards))
    .replace("DATA", json.dumps(DATA, separators=(",", ":")).replace("<", "\\u003c"))
)
ARGS.output.parent.mkdir(parents=True, exist_ok=True)
ARGS.output.write_text(output)
print(ARGS.output.resolve())
