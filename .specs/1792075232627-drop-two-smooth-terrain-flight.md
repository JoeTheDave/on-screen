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
