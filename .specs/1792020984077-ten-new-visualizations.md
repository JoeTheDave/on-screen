# Ten new visualizations

## Summary

Joe, after Strange Attractors: "make 10 more. Each one should be a completely new and different
creation — original, unique, breathtaking, captivating." Ten new full-screen visualizations,
each distinct from everything already in the app (Fourier, flow field, boids, Life, Voronoi ×2,
Orbiting Lights, Sphere Field, 3D Epicycles, Wave Grid, Strange Attractors). Every one loops
forever with periodic re-randomization so an always-on screen never goes stale.

## Iteration plan

### Iteration 1
One file per visualization in `src/visualizations/`, registered in `App.tsx`:
1. **Reaction Diffusion** (`ReactionDiffusion.tsx`) — GPU Gray-Scott ping-pong; coral/fingerprint/
   mitosis patterns drifting between parameter regimes, lit as an embossed glossy relief.
2. **Aurora** (`Aurora.tsx`) — shader night sky: rippling aurora curtains, stars, mirrored in a
   still lake.
3. **Slime Mold** (`SlimeMold.tsx`) — Physarum agents depositing/sensing trails, self-organizing
   into glowing vein networks.
4. **Mandelbulb** (`Mandelbulb.tsx`) — raymarched 3D fractal with an orbiting camera and slowly
   morphing power.
5. **Galaxy Collision** (`GalaxyCollision.tsx`) — two particle spiral galaxies passing/merging,
   throwing tidal tails.
6. **Ink in Water** (`InkFluid.tsx`) — GPU stable-fluids with self-driving dye jets.
7. **Black Hole** (`BlackHole.tsx`) — raymarched gravitational lensing of an accretion disk and
   starfield.
8. **Hypercube** (`Hypercube.tsx`) — 4D polytopes rotating in 4D, projected to 3D, glowing edges.
9. **Cymatics** (`Cymatics.tsx`) — sand particles settling onto Chladni-plate nodal patterns as
   the vibration mode changes.
10. **Terrain Flight** (`TerrainFlight.tsx`) — raymarched procedural landscape flyover through a
    shifting time of day.

Validation: tsc + eslint per file; every GLSL shader compiled headlessly with glslangValidator
(ESSL 3.00) using the prefix Three.js prepends; pure math sanity-checked in node where it exists.

## Shipped — Iteration 1

- Commit: d023f49 (master)
- Deploy: `flyctl deploy --remote-only` succeeded; live bundle verified to contain all ten ids.
- Built in parallel by ten sub-agents (one file each), registered in `App.tsx` centrally.
- Validation: `tsc --noEmit`, full `eslint .`, `npm run build` clean. Every shader (≈40 across the
  ten) compiled headlessly with glslangValidator as ESSL 3.00 with the THREE.ShaderMaterial prefix.
  Node sanity checks: Galaxy integrator (energy drift 3e-7, mergers 29–43 t.u., no NaN), polytope
  vertex/edge counts (16/32, 8/24, 24/96, 120/720, 600/1200), Cymatics settling (85–93 % of grains
  on nodal lines), Terrain camera never clipping over a 60-min flight, Black Hole ray bending vs
  reference, Reaction-Diffusion regimes alive at half precision, Mandelbulb DE bounds.
- Review pass (single, bounded) fixed: GalaxyCollision bloom pass not disposed; Hypercube line
  width doubled at pixel ratio 2; InkFluid NaN/Infinity on zero-size window + cap at
  maxTextureSize; Cymatics and Mandelbulb time uniforms wrapped for always-on precision.
- Not verified: anything visual (brightness, bloom, framerate) — no browser run, per the headless rules.
- DB backup: n/a — frontend-only.
