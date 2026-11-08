//! Test-only stage decomposition; the whole-game audit checks production parity.
use super::*;
use crate::profile::Stage;
use std::time::Instant;

#[derive(Default, serde::Serialize)]
pub(in crate::features) struct Profile {
    probes: Stage,
    spans_and_projection: Stage,
    cache_lookup: Stage,
    dilation: Stage,
    cache_insert: Stage,
    probe_successes: u64,
    empty: u64,
    cache_hits: u64,
}

/// Exact integer lookup, not coordinate subsampling or a changed raster.
pub(in crate::features) struct Projection {
    reach: i32,
    bins: Vec<u8>,
}

impl Projection {
    pub(in crate::features) fn new(radius: i32) -> Self {
        let reach = (8 * radius * radius - 1).isqrt();
        Self {
            reach,
            bins: (-reach..=reach)
                .map(|delta| bin(delta, radius) as u8)
                .collect(),
        }
    }
    fn bin(&self, delta: i32) -> usize {
        self.bins[(delta + self.reach) as usize] as usize
    }
    pub(in crate::features) fn check(&self, radius: i32) {
        for delta in -self.reach..=self.reach {
            assert_eq!(self.bin(delta), bin(delta, radius));
        }
    }
}

pub(in crate::features) fn area_lookup(
    prepared: &PreparedPosition<'_>,
    spot: Spot,
    radius: i32,
    cache: &Cache,
    projection: &Projection,
) -> f32 {
    if probe_saturated(prepared, spot, radius) {
        return CAP;
    }
    let mut centers = [0u128; SIDE];
    let mut first = SIDE;
    let mut last = 0;
    prepared.visit_local_liberty_spans(spot, |span| {
        let y = projection.bin(span.y - spot.y);
        let left = projection.bin(span.left - spot.x);
        let right = projection.bin(span.right - spot.x);
        centers[y] |= (u128::MAX >> (127 - (right - left))) << left;
        first = first.min(y);
        last = last.max(y);
    });
    if first == SIDE {
        return 0.0;
    }
    let key = cache_key(&centers, first, last);
    if let Some(value) = cache.get(key, &centers) {
        return value;
    }
    let value = capped_raster_area(&centers, first, last);
    cache.insert(key, centers, value);
    value
}

pub(in crate::features) fn area(
    prepared: &PreparedPosition<'_>,
    spot: Spot,
    radius: i32,
    cache: &Cache,
    profile: &mut Profile,
) -> f32 {
    area_profiled::<false>(prepared, spot, radius, cache, None, profile)
}

pub(in crate::features) fn area_lookup_profiled(
    prepared: &PreparedPosition<'_>,
    spot: Spot,
    radius: i32,
    cache: &Cache,
    projection: &Projection,
    profile: &mut Profile,
) -> f32 {
    area_profiled::<true>(prepared, spot, radius, cache, Some(projection), profile)
}

fn area_profiled<const LOOKUP: bool>(
    prepared: &PreparedPosition<'_>,
    spot: Spot,
    radius: i32,
    cache: &Cache,
    projection: Option<&Projection>,
    profile: &mut Profile,
) -> f32 {
    let tick = Instant::now();
    let saturated = probe_saturated(prepared, spot, radius);
    profile.probes.record(tick);
    if saturated {
        profile.probe_successes += 1;
        return CAP;
    }
    let tick = Instant::now();
    let mut centers = [0u128; SIDE];
    let mut first = SIDE;
    let mut last = 0;
    prepared.visit_local_liberty_spans(spot, |span| {
        let y = if LOOKUP {
            let projection = projection.unwrap();
            let y = projection.bin(span.y - spot.y);
            let left = projection.bin(span.left - spot.x);
            let right = projection.bin(span.right - spot.x);
            centers[y] |= (u128::MAX >> (127 - (right - left))) << left;
            y
        } else {
            project_span(&mut centers, span, spot, radius)
        };
        first = first.min(y);
        last = last.max(y);
    });
    profile.spans_and_projection.record(tick);
    if first == SIDE {
        profile.empty += 1;
        return 0.0;
    }
    let tick = Instant::now();
    let key = cache_key(&centers, first, last);
    let cached = cache.get(key, &centers);
    profile.cache_lookup.record(tick);
    if let Some(value) = cached {
        profile.cache_hits += 1;
        return value;
    }
    let tick = Instant::now();
    let value = capped_raster_area(&centers, first, last);
    profile.dilation.record(tick);
    let tick = Instant::now();
    cache.insert(key, centers, value);
    profile.cache_insert.record(tick);
    value
}
