"""Order-independent Bradley–Terry fit, anchored at random=100.

Each observed opponent pair gets half a win and half a loss of smoothing,
keeping unbeaten records finite without pulling individuals toward random.
Only disconnected components get a N(100, 350²) prior on their mean rating;
this sets their otherwise unidentified origin, not their relative strengths.
Deviations use the inverse joint Hessian, including opponent uncertainty.
"""

import math
from collections import Counter

import numpy as np

from .ratings import COUNTS, REFERENCE_ID

# Negligible beside match uncertainty; do not chase floating-point precision.
RATING_TOLERANCE = 1.0


def fit(records):
    names = sorted({record[color] for record in records for color in ("black", "white")})
    players = [name for name in names if name != REFERENCE_ID]
    index = {name: i for i, name in enumerate(players)}
    states = {
        name: {
            "rating": 100.0,
            "deviation": 0.0 if name == REFERENCE_ID else 350.0,
            **dict.fromkeys(COUNTS, 0),
        }
        for name in names
    }
    states.setdefault(REFERENCE_ID, {"rating": 100.0, "deviation": 0.0, **dict.fromkeys(COUNTS, 0)})
    pairs = Counter()
    wins = Counter()
    neighbors = {name: set() for name in names}
    for record in records:
        black, white, result = (record[key] for key in ("black", "white", "result"))
        for name in (black, white):
            states[name]["games"] += 1
        if result is None:
            for name in (black, white):
                states[name]["truncated"] += 1
            continue
        if result == "0":
            score = 0.5
            for name in (black, white):
                states[name]["draws"] += 1
        else:
            score = float(result.startswith("B+"))
            winner, loser = (black, white) if score else (white, black)
            states[winner]["wins"] += 1
            states[loser]["losses"] += 1
        # Canonical pair order also makes floating point accumulation independent
        # of input ordering and color assignment.
        if black > white:
            black, white, score = white, black, 1 - score
        pairs[black, white] += 1
        wins[black, white] += score
        neighbors[black].add(white)
        neighbors[white].add(black)
    if not players:
        return states
    design = np.zeros((len(pairs), len(players)))
    totals, scores = [], []
    for row, ((black, white), count) in enumerate(sorted(pairs.items())):
        if black in index:
            design[row, index[black]] = 1
        if white in index:
            design[row, index[white]] -= 1
        totals.append(count + 1)
        scores.append((wins[black, white] + 0.5) / (count + 1))
    totals, scores = np.array(totals), np.array(scores)
    scale = math.log(10) / 400
    # The random-connected component needs no origin prior at all. For other
    # components, constrain only their mean; never shrink rating differences.
    prior = np.zeros((len(players), len(players)))
    unseen = set(names)
    while unseen:
        start = min(unseen)
        component, pending = set(), [start]
        while pending:
            name = pending.pop()
            if name in component:
                continue
            component.add(name)
            pending.extend(neighbors[name] - component)
        unseen -= component
        if REFERENCE_ID not in component:
            mean = np.zeros(len(players))
            for name in sorted(component):
                mean[index[name]] = 1 / len(component)
            prior += np.outer(mean, mean) / (350 * scale) ** 2
    x = np.zeros(len(players))  # rating offsets in natural log-odds units

    def objective(value):
        logits = design @ value
        return (totals * (np.logaddexp(0, logits) - scores * logits)).sum() + (
            0.5 * (value @ prior @ value)
        )

    def derivatives(value):
        logits = design @ value
        probability = np.exp(-np.logaddexp(0, -logits))
        gradient = design.T @ (totals * (probability - scores)) + prior @ value
        weights = totals * probability * (1 - probability)
        hessian = design.T @ (weights[:, None] * design) + prior
        return gradient, hessian

    for _ in range(100):
        gradient, hessian = derivatives(x)
        step = np.linalg.solve(hessian, gradient)
        # Near the optimum, objective differences round to zero before the
        # absolute gradient threshold is met. Armijo backtracking can then
        # shrink a correct Newton step to nothing forever. Check the *full*
        # correction in rating units, not the damped step or objective change:
        # those can be tiny even far from convergence. Apply this last Newton
        # correction before computing uncertainties.
        if np.max(np.abs(step)) / scale < RATING_TOLERANCE:
            x -= step
            break
        fraction = 1.0
        current = objective(x)
        while objective(x - fraction * step) > current - 1e-4 * fraction * (gradient @ step):
            fraction *= 0.5
            if fraction < 1e-12:
                raise RuntimeError("joint rating line search failed")
        x -= fraction * step
    else:
        raise RuntimeError("joint ratings did not converge")
    _, hessian = derivatives(x)
    deviations = np.sqrt(np.diag(np.linalg.inv(hessian))) / scale
    for name, i in index.items():
        states[name].update(rating=100 + float(x[i]) / scale, deviation=float(deviations[i]))
    return states
