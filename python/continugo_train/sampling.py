"""Deterministic board-budget allocation, independent of game rules."""

import math
from collections import Counter

DEFAULT_SIZE_WEIGHTS = "9:2,13:2,19:1"


def parse_weights(text):
    if text == "uniform":
        return None
    result = {}
    try:
        for item in text.split(","):
            size, weight = map(int, item.split(":"))
            if size not in range(9, 20) or not 0 <= weight <= 1000 or size in result:
                raise ValueError
            result[size] = weight
    except (ValueError, AttributeError) as error:
        raise ValueError(
            "size weights must be 'uniform' or unique size:integer pairs (sizes 9..19, weights 0..1000)"
        ) from error
    if not result:
        raise ValueError("empty size weights")
    return result


def weighted_sizes(sizes, text=DEFAULT_SIZE_WEIGHTS):
    weights = parse_weights(text)
    if weights is None:
        return list(sizes)
    counts = Counter()
    for size in sizes:
        counts[size] += weights.get(size, 1)
    entries = [(size, weight) for size, weight in counts.items() if weight]
    if not entries:
        raise ValueError("no positive-weight board sizes available")
    divisor = math.gcd(*(weight for _, weight in entries))
    entries = [(size, weight // divisor) for size, weight in entries]
    total = sum(weight for _, weight in entries)
    if total > 4096:
        raise ValueError("weighted board cycle exceeds 4096 entries")
    current = [0] * len(entries)
    result = []
    for _ in range(total):
        for index, (_, weight) in enumerate(entries):
            current[index] += weight
        best = max(range(len(entries)), key=lambda i: current[i])
        result.append(entries[best][0])
        current[best] -= total
    return result


def position_quotas(available, text):
    """Downsample surplus positions without replacement, not just loss weighting.

    Long 19-equivalent games cannot dominate extraction cost. Tiny buckets retain
    coverage where possible; unavailable sizes are renormalized, never invented.
    """
    weights = parse_weights(text)
    if weights is None:
        return dict(available)
    counts = {size: count for size, count in available.items() if count and weights.get(size, 1)}
    if not counts:
        raise ValueError("no positions with positive board-size weight")
    total_weight = sum(weights.get(size, 1) for size in counts)
    budget = min(
        sum(counts.values()),
        max(
            len(counts),
            int(min(count * total_weight / weights.get(size, 1) for size, count in counts.items())),
        ),
    )
    cycle = weighted_sizes(sorted(counts), text)
    selected = Counter()
    index = 0
    while sum(selected.values()) < budget:
        size = cycle[index % len(cycle)]
        index += 1
        if selected[size] < counts[size]:
            selected[size] += 1
    return dict(selected)
