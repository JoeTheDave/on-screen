# Strange Attractors visualization

## Summary

Joe asked for a visualization of my own choosing, "super interesting". Tens of thousands of
glowing particles flow through a chaotic 3D strange attractor (Lorenz, Thomas, Aizawa,
Halvorsen). Each particle is drawn as a short streak along its velocity, colored by speed, with
additive blending and bloom so dense flow lines glow. Every 45 s the governing equations switch:
the existing cloud is not reset — it is re-interpreted in the next system's coordinates and
visibly collapses/unfurls onto the new attractor. The camera orbits throughout.

## Iteration plan

### Iteration 1
- `src/visualizations/StrangeAttractors.tsx` (Three.js + UnrealBloomPass, like OrbitingLights).
- Per attractor: derivative function, stable max step, seed point. At switch time a warm-up run
  measures the attractor's center, radius (p95) and mean speed, so every attractor is framed to
  a unit radius and moves at the same apparent pace — no hand-tuned scales.
- Warm-up samples double as a respawn pool: escaped/NaN particles respawn on the attractor.
- 20,000 particles as `LineSegments` (head = position, tail = position − velocity × trail time),
  vertex colors from a per-attractor two-hue palette by speed.
- Camera orbits (20 s/turn) with a slow elevation swing.
- Register in `App.tsx` as "Strange Attractors".
