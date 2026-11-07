# Retire eight visualizations; Reaction Diffusion without pauses

## Summary

Joe: retire (remove from the menu, keep the code) Voronoi Diagram, Circular Voronoi, Orbiting
Lights, Sphere Field, Black Hole, Terrain Flight, Atomic Bomb, Prism. Reaction Diffusion: when a
reaction finishes the screen is often still for several seconds — the next reaction should
begin immediately.

## Iteration plan

### Iteration 1
- `App.tsx`: comment out the eight imports + menu entries (Snakes-style shelving); a saved
  selection of a retired one falls back to the default. CLAUDE.md records the convention.
- `reactionDiffusionMath.ts`: TRANSFORM_HOLD 6 s → 0, RESET_HOLD 2 s → 0, SPREAD 15 s → 10 s
  (drops fill in ~9 s). The next front launches the instant the previous one has crossed the
  screen; anything still converting keeps converting under the same regime.

## Pushed, not deployed — Iteration 1

- Commit: b68edab (master), pushed.
- Deploy BLOCKED: `flyctl deploy` failed — Fly auth token expired (last login 2026-10-02; 30-day
  limit). Needs Joe to run `flyctl auth login` (browser), then `fly deploy --remote-only`.
  Live site still serves the previous build.
- Tests: none (no suite); tsc, full eslint, build clean. Not verified visually. DB backup: n/a.

## Shipped — Iteration 1

- Commit: b68edab (master); deployed after Joe re-ran `flyctl auth login`.
- Deploy: `flyctl deploy --remote-only` succeeded; live bundle verified: the eight retired ids absent, reaction-diffusion/tornado present.
- Tests: none (no suite); tsc, full eslint, build clean. Not verified visually. DB backup: n/a.
