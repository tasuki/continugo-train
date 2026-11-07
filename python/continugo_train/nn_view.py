"""Self-contained, offline neural game microscope (authoritative Rust SGF replay).

This is a checkpoint replay, not a capture of the live MCTS tree. Exact policy
covers complete top-ranked cells plus cells containing recorded visits/the move.
"""

import argparse
import base64
import hashlib
import html
import json
import math
import struct
import sys
import zlib
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from .features import COARSE_PLANES, MOVE_FEATURES, PASS, FeatureClient, inspect_game
from .model import NetworkConfig, PolicyValueNet, cell_ids, local_log_probs
from .train import feature_spec


def json_safe(value):
    return json.dumps(value, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")


def png_uri(rgba):
    """Tiny dependency-free RGBA PNG encoder; every image travels inside the HTML."""
    rgba = np.asarray(rgba, dtype=np.uint8)
    height, width, channels = rgba.shape
    if channels != 4:
        raise ValueError("expected RGBA")

    def chunk(tag, payload):
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload))
        )

    scanlines = b"".join(b"\0" + row.tobytes() for row in rgba)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">2I5B", width, height, 8, 6, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(scanlines, 6)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()


def heatmap(name, values, *, signed=False, limits=None, log=False, note=""):
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    low, high = limits or (
        (float(values[valid].min()), float(values[valid].max())) if valid.any() else (0.0, 1.0)
    )
    if signed:
        extent = max(abs(low), abs(high), 1e-12)
        low, high = -extent, extent
    safe = np.where(valid, values, 0)
    if log:
        scaled = np.log1p(np.maximum(safe, 0) / max(high, 1e-12) * 1000) / math.log(1001)
    else:
        scaled = (safe - low) / max(high - low, 1e-12)
    scaled = np.clip(scaled, 0, 1)
    if signed:
        # Negative blue, neutral dark, positive orange.
        a = np.maximum(1 - 2 * scaled, 0)[..., None]
        b = np.maximum(2 * scaled - 1, 0)[..., None]
        rgb = 24 + a * np.array([35, 120, 225]) + b * np.array([230, 125, 30])
    else:
        rgb = np.stack((30 + 220 * scaled, 25 + 180 * scaled, 55 + 40 * scaled), axis=-1)
    rgba = np.concatenate((rgb, (255 * valid)[..., None]), axis=-1).astype(np.uint8)
    return {
        "name": name,
        "src": png_uri(rgba),
        "low": low,
        "high": high,
        "note": note + (" · log intensity" if log else ""),
    }


def raster(actions, values, *, mode="mean", width=256):
    """Display-only binning: probability/visits sum; input features/logits average."""
    actions = np.asarray(actions)
    values = np.asarray(values)
    bins = (actions % 1000 * width // 1000) + width * (actions // 1000 * width // 1000)
    finite = np.isfinite(values)
    counts = np.bincount(bins[finite], minlength=width**2)
    sums = np.bincount(bins[finite], weights=values[finite], minlength=width**2).astype(np.float64)
    if mode == "mean":
        sums /= np.maximum(counts, 1)
    sums[counts == 0] = np.nan
    return sums.reshape(width, width)


def load_checkpoint(path):
    path = Path(path)
    identity = hashlib.sha256(path.read_bytes()).hexdigest()
    saved = torch.load(path, map_location="cpu", weights_only=True)
    spec = saved["feature_spec"]
    if spec != feature_spec(spec["samples_per_diameter"]):
        raise ValueError(f"{path}: incompatible feature schema")
    model = PolicyValueNet(NetworkConfig(**saved["config"])).eval()
    model.load_state_dict(saved["model"])
    return model, spec["samples_per_diameter"], identity


def action_name(action):
    return "pass" if action == PASS else f"({action % 1000}, {action // 1000})"


@torch.inference_mode()
def frame(client, sgf, ply, loaded, info):
    model, density, identity = loaded
    sample = client.load(
        sgf, ply, density=density, cells=[], view=True, file_sha256=info["file_sha256"]
    )
    view = client.last_view
    coarse = sample.coarse[None]
    trunk, pooled, spatial, pass_logit, value = model._encode(coarse)
    score = model.score_head(pooled).item()
    ownership = model.ownership_head(trunk)[0, 0].numpy()
    logits = model.coarse_logits(spatial, pass_logit)[0]
    coarse_probs = logits.softmax(0)
    resolution = sample.resolution
    # Complete cells only. Diagnostic scope is explicit, not falsely presented
    # as the live search's independently seeded exploration-cell selection.
    selected = set(
        torch.argsort(logits[:-1], descending=True, stable=True)[: model.config.top_cells].tolist()
    )
    bearing = sample.target_actions[(sample.visits > 0) & (sample.target_actions != PASS)]
    if view["action"] != PASS:
        bearing = torch.cat((bearing, torch.tensor([view["action"]])))
    selected.update(cell_ids(bearing, resolution).tolist())
    sample = client.load(
        sgf, ply, density=density, cells=sorted(selected), file_sha256=info["file_sha256"]
    )
    actions, exact, legal = sample.actions, sample.exact, sample.legal
    cells = cell_ids(actions, resolution)
    raw = (
        torch.cat(
            [
                model.score_chunk(
                    spatial,
                    exact[None, start : start + model.config.chunk_size],
                    actions[start : start + model.config.chunk_size],
                )[0]
                for start in range(0, len(actions), model.config.chunk_size)
            ]
        )
        if len(actions)
        else torch.empty(0)
    )
    local = local_log_probs(raw[None], actions, legal[None], resolution)[0].exp()
    # Same normalization as native inference: redistribute placement mass only
    # over selected nonempty cells, retaining global coarse P(pass).
    nonempty = torch.zeros(resolution**2, dtype=torch.bool)
    nonempty[cells[legal]] = True
    placement = torch.zeros(resolution**2)
    pass_prob = float(coarse_probs[-1])
    if nonempty.any():
        placement[nonempty] = logits[:-1][nonempty].softmax(0) * (1 - coarse_probs[-1])
    else:
        pass_prob = 1.0
    policy = local * placement[cells]
    mask = np.zeros(resolution**2)
    mask[list(selected)] = 1
    policy_maps = [
        heatmap("Coarse logits", logits[:-1].reshape(resolution, resolution).numpy(), signed=True),
        heatmap(
            "Coarse P(cell)", coarse_probs[:-1].reshape(resolution, resolution).numpy(), log=True
        ),
        heatmap("Diagnostic selected cells", mask.reshape(resolution, resolution), limits=(0, 1)),
        heatmap(
            "Exact raw logits (legal)",
            raster(actions, raw.masked_fill(~legal, torch.nan)),
            signed=True,
        ),
        heatmap("P(move | cell)", raster(actions[legal], local[legal], mode="sum"), log=True),
        heatmap(
            "Redistributed P(move)", raster(actions[legal], policy[legal], mode="sum"), log=True
        ),
    ]
    if len(sample.visits):
        keep = sample.target_actions != PASS
        policy_maps.append(
            heatmap(
                "MCTS visits (recorded)",
                raster(sample.target_actions[keep], sample.visits[keep], mode="sum"),
                log=True,
            )
        )
    output_maps = [heatmap("Ownership prediction (STM)", ownership, signed=True, limits=(-1, 1))]
    if sample.ownership is not None:
        output_maps.extend(
            [
                heatmap(
                    "Final ownership target (STM)", sample.ownership, signed=True, limits=(-1, 1)
                ),
                heatmap(
                    "Ownership prediction − target",
                    ownership - sample.ownership.numpy(),
                    signed=True,
                    limits=(-2, 2),
                ),
            ]
        )
    # These are the actual shared inputs to the value and score MLPs.
    output_maps.extend(
        [
            heatmap("Value & score input: pooled trunk", pooled.numpy(), signed=True),
            heatmap("Value hidden activation", model.value_head[:2](pooled).numpy(), signed=True),
            heatmap("Score hidden activation", model.score_head[:2](pooled).numpy(), signed=True),
            heatmap("Policy spatial context RMS", spatial.square().mean(1).sqrt()[0].numpy()),
        ]
    )
    top = torch.argsort(policy, descending=True)[:12]
    candidates = [
        {
            "action": action_name(int(actions[i])),
            "p": float(policy[i]),
            "logit": float(raw[i]),
            "local": float(local[i]),
            "features": exact[i].tolist(),
        }
        for i in top
        if legal[i]
    ]
    candidates.append(
        {
            "action": "pass",
            "p": pass_prob,
            "logit": float(pass_logit.item()),
            "local": None,
            "features": None,
        }
    )
    candidates.sort(key=lambda row: -row["p"])
    pass_visits = sample.visits[sample.target_actions == PASS].sum().item()
    root = info.get("root_comment") or {}
    actor = root.get("model" if sample.player == 0 else "opponent", root.get("model"))
    player_name = info.get("black_player" if sample.player == 0 else "white_player")
    counterfactual = bool(
        (actor and actor != identity)
        or (not actor and player_name and not player_name.startswith("neural-"))
    )
    weight = (root.get("search") or {}).get("score_weight", 0.5)
    return {
        "ply": ply,
        "player": sample.player,
        "model": identity,
        "board_size": sample.board_size,
        "diameter": sample.diameter,
        "resolution": resolution,
        "view": view,
        "move": action_name(view["action"]),
        "counterfactual": counterfactual,
        "actor": actor,
        "train_policy": ply in info["policy_training_plies"],
        "utility": (1 - weight) * value.item() + weight * score,
        "score_weight": weight,
        "value": value.item(),
        "points_pct": 50 * (value.item() + 1),
        "score": score,
        "score_stones": score * 1_000_000 / (math.pi * (sample.diameter / 2) ** 2),
        "outcome": sample.outcome,
        "score_target": sample.score,
        "coarse_pass": float(coarse_probs[-1]),
        "pass": pass_prob,
        "policy_mass": float(policy.sum()) + pass_prob,
        "pass_visits": pass_visits,
        "visits": int(sample.visits.sum()),
        "cells": len(selected),
        "coordinates": len(actions),
        "legal": int(legal.sum()),
        "policy": policy_maps,
        "heads": output_maps,
        "coarse": [
            heatmap(
                name,
                values,
                limits=(0, max(1.0, float(values.max()))) if i in (4, 5) else (0, 1),
                note="Raw area; not clipped" if i in (4, 5) else "",
            )
            for i, (name, values) in enumerate(zip(COARSE_PLANES, sample.coarse, strict=True))
        ],
        "exact": [
            heatmap(
                name,
                raster(actions, exact[:, i]),
                limits=(0, 1),
                note="Mean / display pixel",
            )
            for i, name in enumerate(MOVE_FEATURES)
        ],
        "context": [
            heatmap(f"Policy spatial channel {i}", channel.numpy(), signed=True)
            for i, channel in enumerate(spatial[0])
        ],
        "candidates": candidates,
    }


PAGE = r"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ContinuGo · NN microscope</title>
<style>
[hidden]{display:none!important}:root{color-scheme:dark;font:14px system-ui;background:#0d1420;color:#dbe4f3}*{box-sizing:border-box}
body{margin:0}header{position:sticky;top:0;z-index:2;background:#172134;padding:10px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
button,a,select{color:#edf4ff;background:#293950;border:1px solid #58718d;padding:5px 10px;border-radius:5px}input{flex:1;min-width:100px}button:disabled{opacity:.4}
main{padding:12px;display:grid;grid-template-columns:290px 1fr;gap:12px}.panel,section{background:#141f30;border:1px solid #2e4056;border-radius:8px;padding:10px;min-width:0}
aside{grid-row:span 3}h2{font-size:15px;margin:0 0 8px}p{line-height:1.4}.muted,small{color:#9fadc2}#board{width:100%;background:#b99563;border-radius:6px}
.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(115px,1fr));gap:6px}figure{margin:0;min-width:0;background:#101826;border-radius:4px;padding:5px}
figure img{width:100%;max-height:155px;object-fit:contain;image-rendering:pixelated;background:repeating-conic-gradient(#243044 0% 25%,#172134 0% 50%) 50%/8px 8px;cursor:zoom-in}
aside small{overflow-wrap:anywhere}figcaption{font-size:11px;overflow-wrap:anywhere}figure small{font-size:10px}#stats{font-variant-numeric:tabular-nums;line-height:1.7}
#inputs{grid-column:2;display:grid;grid-template-columns:1fr 1fr;gap:12px}#inputs .gallery{grid-template-columns:repeat(6,minmax(0,1fr))}
#inputs img{max-height:125px}details{grid-column:1/-1}table{border-collapse:collapse;width:100%;font-size:12px}th,td{text-align:right;padding:5px;border-bottom:1px solid #304159}th:first-child,td:first-child{text-align:left}
dialog{max-width:94vw;max-height:94vh;background:#172134;color:#fff;border:1px solid #62758d}dialog img{max-width:85vw;max-height:78vh;image-rendering:pixelated}dialog::backdrop{background:#000b}
@media(min-width:2100px){figure img{max-height:195px}#inputs img{max-height:160px}}@media(max-width:1200px){main{grid-template-columns:240px 1fr}#inputs{grid-template-columns:1fr}#inputs .gallery{grid-template-columns:repeat(6,1fr)}}@media(max-width:750px){main{display:block}aside,section,#inputs{margin-bottom:12px}#inputs{display:block}#inputs .gallery{grid-template-columns:repeat(3,1fr)}}
</style>
<header><strong>ContinuGo · NN microscope</strong><button id="prev">←</button><span id="position"></span><button id="next">→</button><input id="slider" type="range" min="0" value="0"><a id="sgf" download="game.sgf">SGF ↓</a></header>
<p id="notice" class="panel" style="margin:12px" role="status" hidden></p>
<details id="skipped" class="panel" style="margin:12px" hidden><summary id="skippedtitle"></summary><ul id="skiplist"></ul></details>
<main><aside class="panel"><canvas id="board" width="600" height="600"></canvas><h2 id="turn"></h2><div id="stats"></div><p class="muted">Pre-move position. Orange cross = move played. All network quantities are side-to-move relative, not always Black.</p><p class="muted">Diagnostic checkpoint replay, not a live tree trace. Complete top-ranked cells + recorded visit/move cells; live search also uses seeded exploration and independent proposals.</p><p class="muted">Blue = negative, orange = positive for signed maps. Other maps: dark → bright. Transparent checkerboard = not evaluated. Probability/visit pixels sum; exact features/logits average. Click an image to enlarge.</p><small id="provenance"></small></aside>
<section><h2>Hierarchical policy · coarse → exact → search</h2><div id="policy" class="gallery"></div></section>
<section><h2>Critics & ownership · shared trunk input, separate heads</h2><div id="heads" class="gallery"></div></section>
<div id="inputs"><section><h2>Every coarse input plane</h2><div id="coarse" class="gallery"></div></section><section><h2>Every exact move input feature</h2><div id="exact" class="gallery"></div></section></div>
<details class="panel"><summary>All projected policy-context input channels</summary><p>These shared spatial channels feed the coarse cell head and are bilinearly sampled at each exact coordinate before adding the exact-feature projection. RMS above is only a summary; these are the individual channel values.</p><div id="context" class="gallery"></div></details>
<details class="panel"><summary>Top diagnostic policy actions · exact feature values & methodology</summary><p>Local softmax uses all legal coordinates in each complete selected cell. Final policy redistributes placement mass over nonempty selected cells, preserving global coarse pass probability (or pass=1 if none). Input planes use fixed [0,1] scales except uncapped liberty areas; every map shows its range. Value is expected win/draw/loss outcome; “points” counts draws as half, not strict win probability. Score is signed final area / board area; stone units divide by stone-disc area. Ownership is auxiliary, not used in search. This report does not recompute ratings or change SGF training labels.</p><div style="overflow:auto"><table id="candidates"></table></div></details></main>
<dialog id="zoom"><button id="close">Close (Esc)</button><h2 id="zoomtitle"></h2><img id="zoomimage" alt="Enlarged heatmap"></dialog>
"""

SCRIPT = r"""
<script>
'use strict';
const data=JSON.parse(document.getElementById('data').textContent), frames=data.frames;
const $=id=>document.getElementById(id), fmt=(v,n=3)=>v==null?'unknown':Number(v).toFixed(n);
let index=0;
function gallery(id,images){$(id).replaceChildren(...images.map(image=>{const f=document.createElement('figure'),img=document.createElement('img'),c=document.createElement('figcaption'),s=document.createElement('small');img.src=image.src;img.alt=image.name;c.textContent=image.name;s.textContent=`${fmt(image.low)} … ${fmt(image.high)} ${image.note}`;img.title=s.textContent;img.onclick=()=>{$('zoomtitle').textContent=image.name+' · '+s.textContent;$('zoomimage').src=image.src;$('zoom').showModal()};f.append(img,c,s);return f}))}
function board(f){const c=$('board').getContext('2d'),scale=560/1000,xy=v=>20+v*scale;c.clearRect(0,0,600,600);c.strokeStyle='#6a5035';c.lineWidth=1;for(let i=0;i<=10;i++){c.beginPath();c.moveTo(xy(i*100),20);c.lineTo(xy(i*100),580);c.moveTo(20,xy(i*100));c.lineTo(580,xy(i*100));c.stroke()}for(const [x,y,color] of f.view.stones){c.beginPath();c.arc(xy(x),xy(y),f.diameter/2*scale,0,Math.PI*2);c.fillStyle=color?'#f3f4f7':'#10141b';c.fill();c.strokeStyle='#566';c.stroke()}if(f.view.action!==1000000){let x=xy(f.view.action%1000),y=xy(Math.floor(f.view.action/1000));c.strokeStyle='#ff701f';c.lineWidth=4;c.beginPath();c.moveTo(x-9,y);c.lineTo(x+9,y);c.moveTo(x,y-9);c.lineTo(x,y+9);c.stroke()}}
function show(i){if(!frames.length)return;index=Math.max(0,Math.min(frames.length-1,i));const f=frames[index];$('slider').value=index;$('prev').disabled=!index;$('next').disabled=index===frames.length-1;$('position').textContent=`Move ${f.ply+1} / ${data.moves||frames.length} · view ${index+1}/${frames.length}`;$('turn').textContent=`${f.player?'White':'Black'} to play ${f.move}${f.counterfactual?' · counterfactual NN':''}`;
$('stats').replaceChildren();for(const text of [`Value ${fmt(f.value)} · points ${fmt(f.points_pct,1)}%`,`Score ${fmt(f.score)} · ${fmt(f.score_stones,1)} stones`,`Head utility ${fmt(f.utility)} (score weight ${f.score_weight})`,`Policy training ${f.train_policy?'enabled':'absent/masked'}`,`Final outcome ${fmt(f.outcome,0)} · score ${fmt(f.score_target)}`,`P(pass) coarse ${fmt(f.coarse_pass,5)} / final ${fmt(f.pass,5)}`,`Search visits ${f.visits} · pass ${f.pass_visits}`,`${f.resolution}² raster · size ${f.board_size}`,`${f.cells} cells · ${f.legal} legal / ${f.coordinates} evaluated`]){const d=document.createElement('div');d.textContent=text;$('stats').append(d)}
$('provenance').textContent=`Checkpoint SHA256 ${f.model} · SGF SHA256 ${data.sgf_sha256}`;board(f);for(const name of ['policy','heads','coarse','exact','context'])gallery(name,f[name]);
const table=$('candidates');table.replaceChildren();let tr=document.createElement('tr');for(const name of ['Action','P(move)','Raw logit','P(move|cell)',...data.features]){const th=document.createElement('th');th.textContent=name;tr.append(th)}table.append(tr);for(const row of f.candidates){tr=document.createElement('tr');for(const value of [row.action,fmt(row.p,6),fmt(row.logit),fmt(row.local,6),...(row.features||data.features.map(()=>null)).map(v=>fmt(v))]){const td=document.createElement('td');td.textContent=value;tr.append(td)}table.append(tr)}}
const skipped=data.skipped||[];if(skipped.length){$('skipped').hidden=false;$('skippedtitle').textContent=`${skipped.length} moves skipped (random or unsupported network)`;for(const s of skipped){const li=document.createElement('li');li.textContent=`Move ${s.ply+1} · ${s.player?'White':'Black'}: ${s.reason}`;$('skiplist').append(li)}}
if(!frames.length){document.querySelector('main').hidden=true;$('notice').hidden=false;$('notice').textContent='No moves can be visualized with the available supported checkpoints. See the skipped-move reasons below; the original game is embedded in SGF ↓.';$('position').textContent=`0 visualizable / ${data.moves||0} moves`;for(const id of ['slider','prev','next'])$(id).disabled=true}
$('slider').max=Math.max(0,frames.length-1);$('slider').oninput=e=>show(+e.target.value);$('prev').onclick=()=>show(index-1);$('next').onclick=()=>show(index+1);$('close').onclick=()=>$('zoom').close();$('sgf').href=data.sgf;
document.addEventListener('keydown',e=>{if($('zoom').open||['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName))return;if(['ArrowLeft','ArrowRight','Home','End'].includes(e.key)){e.preventDefault();show(e.key==='Home'?0:e.key==='End'?frames.length-1:index+(e.key==='ArrowRight'?1:-1))}});show(0);
</script></html>
"""


RATED_MODE = "rated-own-actor-v1"


def _write_report(sgf, output, info, players, reasons, *, binary=None, rated=False):
    """Keep unsupported/random plies out of navigation, never substitute another net."""
    output.parent.mkdir(parents=True, exist_ok=True)
    skipped = []
    with output.open("x", encoding="utf-8") as stream:
        try:
            stream.write(PAGE)
            stream.write('<script id="data" type="application/json">')
            metadata = {
                "features": MOVE_FEATURES,
                "sgf_sha256": info["file_sha256"],
                "moves": info["moves"],
                "mode": RATED_MODE if rated else "checkpoint-replay",
                "actors": [
                    (info.get("root_comment") or {}).get(key) for key in ("model", "opponent")
                ],
                "sgf": "data:application/x-go-sgf;base64,"
                + base64.b64encode(sgf.read_bytes()).decode(),
            }
            stream.write(json_safe(metadata)[:-1] + ',"frames":[')
            count = 0
            with FeatureClient(binary) if any(players) else nullcontext() as client:
                for ply in range(info["moves"]):
                    loaded = players[ply % 2]
                    reason = reasons[ply % 2]
                    result = None
                    if loaded is not None:
                        try:
                            result = frame(client, sgf, ply, loaded, info)
                        except (ValueError, RuntimeError) as error:
                            if not rated:
                                raise
                            reason = f"Visualization unavailable: {error}"
                    if result is None:
                        skipped.append({"ply": ply, "player": ply % 2, "reason": reason})
                        continue
                    if rated and result["counterfactual"]:
                        raise ValueError("rated visualization must use the mover's own network")
                    if count:
                        stream.write(",")
                    stream.write(json_safe(result))
                    count += 1
                    print(
                        f"NN viewer: {html.escape(sgf.name)} move {ply + 1}/{info['moves']}",
                        file=sys.stderr,
                    )
            stream.write('],"skipped":' + json_safe(skipped) + "}</script>" + SCRIPT)
        except BaseException:
            output.unlink(missing_ok=True)
            raise
    return output


def generate_rated(sgf, output, info, checkpoints, *, binary=None):
    """Best-effort visualizable turns, strict identity, explicit omissions otherwise.

    `info` must already be validated with the game's matching frozen rules runtime.
    Unsupported historic architectures remain unsupported, not migrated or padded
    with invented score/ownership heads.
    """
    players, reasons = [], []
    root = info["root_comment"]
    for actor in (root["model"], root["opponent"]):
        loaded = None
        if actor == "random-no-eye-fill-v1":
            reason = "Random bot has no neural network"
        elif not checkpoints.get(actor):
            reason = f"Matching checkpoint unavailable ({actor})"
        else:
            path = Path(checkpoints[actor])
            if hashlib.sha256(path.read_bytes()).hexdigest() != actor:
                raise ValueError(f"rated checkpoint SHA256 mismatch: {path}")
            try:
                loaded = load_checkpoint(path)
                reason = ""
            except (ValueError, RuntimeError, KeyError, TypeError) as error:
                reason = f"Unsupported checkpoint architecture/features: {error}"
        players.append(loaded)
        reasons.append(reason)
    return _write_report(Path(sgf), Path(output), info, players, reasons, binary=binary, rated=True)


def generate(sgf, checkpoint, output, *, white_checkpoint=None, binary=None):
    sgf, output = Path(sgf), Path(output)
    black = load_checkpoint(checkpoint)
    white = (
        black
        if white_checkpoint is None or Path(white_checkpoint) == Path(checkpoint)
        else load_checkpoint(white_checkpoint)
    )
    info = inspect_game(sgf, black[1], binary)
    if not info["moves"]:
        raise ValueError("game has no moves")
    root = info.get("root_comment") or {}
    for loaded, key in [(black, "model"), (white, "opponent")]:
        expected = root.get(key)
        if expected and len(expected) == 64 and expected != loaded[2]:
            raise ValueError(f"checkpoint SHA256 does not match SGF {key}: {expected}")
    # Self-play records have one identity, arena records have color-specific ones.
    if not root.get("opponent") and root.get("model") == black[2] and white[2] != black[2]:
        raise ValueError("self-play requires the same checkpoint for both colors")
    players, reasons = [black, white], ["", ""]
    for color, key in enumerate(("model", "opponent")):
        if root.get(key) == "random-no-eye-fill-v1":
            players[color] = None
            reasons[color] = "Random bot has no neural network"
    return _write_report(sgf, output, info, players, reasons, binary=binary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sgf", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Black checkpoint; both colors by default"
    )
    parser.add_argument("--white-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, help="default: SGF path with .nn.html suffix")
    parser.add_argument("--feature-binary", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    try:
        path = generate(
            args.sgf,
            args.checkpoint,
            args.output or args.sgf.with_suffix(".nn.html"),
            white_checkpoint=args.white_checkpoint,
            binary=args.feature_binary,
        )
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"NN viewer: {error}\n")
    print(path)


if __name__ == "__main__":
    main()
