# WebGL failure handling + Terrain Flight load cap

## Summary

Joe loaded Terrain Flight on his Linux box (Chrome, RX 7900 XT, 4K display) and got
"THREE.WebGLRenderer: A WebGL context could not be created … GL_VENDOR = Disabled,
GL_RENDERER = Disabled … BindToCurrentSequence failed", then "Uncaught Error: Error creating
WebGL context". That is Chrome's GPU process with WebGL turned off for the session — what Chrome
does after its GPU process crashes/hangs (no kernel amdgpu reset in the journal, so the driver
itself was fine). Two problems on our side: the uncaught error unmounted the whole React tree
(menu gone), and Terrain Flight had no ceiling on raymarch pixels on large screens.

## Iteration plan

### Iteration 1
- `src/components/VisualizationErrorBoundary.tsx`: wraps the current visualization in `App.tsx`
  (keyed by viz id so switching resets it). On failure it shows a short centered note; for WebGL
  errors it explains that restarting the browser restores WebGL (chrome://gpu for status). The
  menu button stays usable.
- `TerrainFlight.tsx`: cap the raymarch target at 700k pixels (Mandelbulb already caps similarly),
  so a 4K/high-DPI screen can't produce a frame heavy enough to stall the GPU process.

## Shipped — Iteration 1

- Commit: b71372e (master)
- Deploy: `flyctl deploy --remote-only` succeeded; live bundle verified to contain the boundary message.
- Tests: none (no suite); `npm run build` + eslint on touched files clean.
- DB backup: n/a — frontend-only.
