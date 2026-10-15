# Drop Slime Mold + Cymatics; smooth Terrain Flight with trees

## Summary

Joe: Slime Mold "is kind of a flop" and Cymatics is "pretty boring" — drop both. Terrain Flight
is "very cool" but the camera bobs up and down following the ground; the flight should be smooth
and unaffected by elevation while still never crashing into mountains. Add trees.

## Iteration plan

### Iteration 1
- Delete `SlimeMold.tsx`, `Cymatics.tsx`, `cymaticsMath.ts` and their `App.tsx` entries (a saved
  selection of either falls back to the default visualization).
- TerrainFlight: cruise at a near-constant altitude; avoid terrain by steering around high ground
  (horizontal look-ahead, valley seeking) and, only when a ridge can't be avoided, a slow,
  heavily smoothed, anticipatory climb/descend — no ground-following bob.
- TerrainFlight: procedural trees (forests on gentle, mid-altitude slopes, thinning toward
  snowline and shores), rendered in the raymarch within the performance budget.

## Shipped — Iteration 1

- Commit: 5ce7286 (master)
- Deploy: `flyctl deploy --remote-only` succeeded; live bundle verified: no slime-mold/cymatics, terrain-flight present.
- Flight (`Flight` class in terrainMath.ts): near-constant cruise altitude drifting over ~1 min;
  re-plans 4×/s over 13 headings ~18 s ahead (neighbour-smoothed valley seeking), bank ≤ ~18°;
  anticipatory two-stage-smoothed climb only when needed, ~14 s descent; hard floor above
  terrain + tallest tree + 0.02. Node, 6 × 60 min vs old logic: RMS vertical accel 0.057–0.089 →
  0.0009–0.0015; 10 s altitude std-dev 0.039–0.054 → 0.005–0.010; pitch −20°..+11° → −6.3°..−2.3°;
  min clearance ≥ 0.084 (≥ 0.066 above canopy); floor never engaged; no NaN.
- Trees: jittered-cell conifers/broadleaf, 4-cell lookup, only marched in the canopy band within
  2.6 units, fading to a canopy texture 1.3–2.6; biome tints; terrain shadows + short tree shadows.
  Bisection hit refinement. Estimated +3–4 % average march cost/pixel.
- Static: tsc, full eslint, build clean; all 3 shaders compile (glslangValidator ESSL 3.00).
- Not verified visually (headless). DB backup: n/a.
