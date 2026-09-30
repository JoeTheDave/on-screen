# On-Screen — Technical Reference

## What This Is

A React + Vite static web app that uses the Screen Wake Lock API to keep the display awake, while showing full-screen generative art animations. Deployed as a Docker/nginx container on Fly.io.

## Workflow

```yaml
deployments:
  prod:
    branch: master
# No staging. Feature branches merge straight to master, which auto-deploys to prod.
```

## Intentional Deviations from STANDARDS.md

| Deviation | Why |
|-----------|-----|
| Frontend-only (no backend, no database, no auth) | There's nothing to persist server-side; the app is pure client state |
| nginx serves static build (no Express) | No API routes — nginx is lighter and simpler |
| No staging environment | Low-stakes personal project; direct-to-prod is fine |
| No test suite | No logic to unit-test; visual verification via the app itself |
| React 19 (not 18) | Latest stable at time of creation |
| No `client/` subdirectory | Single-app repo; source lives at `src/` in the root |

## Stack

| Layer | In Use |
|-------|--------|
| Frontend | React 19, TypeScript, Tailwind CSS v4, Vite |
| 3D / Canvas | Three.js (OrbitingLights), HTML Canvas 2D (all other visualizations) |
| Backend | None |
| Database | None |
| Auth | None (public URL, no login) |
| Deployment | Fly.io — `on-screen` app, `sjc` region, `https://on-screen.fly.dev` |
| Serving | nginx (static build inside Docker) |

## Repo Structure

```
on-screen/
├── src/
│   ├── components/
│   │   └── MenuModal.tsx       # Visualization picker overlay
│   ├── visualizations/
│   │   ├── FourierEpicycles.tsx
│   │   ├── ParticleFlowField.tsx
│   │   ├── FlockingBoids.tsx
│   │   ├── ConwayGameOfLife.tsx
│   │   ├── VoronoiDiagrams.tsx
│   │   ├── CircularVoronoi.tsx
│   │   ├── OrbitingLights.tsx  # Three.js; others use Canvas 2D
│   │   └── Snakes.tsx          # Shelved — commented out in App.tsx
│   ├── App.tsx                 # Root: visualization router + hidden menu button
│   ├── main.tsx                # Entry point — Screen Wake Lock setup lives here
│   └── index.css
├── public/
├── Dockerfile                  # Multi-stage: node build → nginx serve
├── nginx.conf
├── fly.toml
├── VISION.md
├── PROJECT.md
├── CLAUDE.md
└── .specs/
```

## Screen Wake Lock

Wake lock is requested in `src/main.tsx` (outside the React tree, before `createRoot`). It uses the [Screen Wake Lock API](https://developer.mozilla.org/en-US/docs/Web/API/Screen_Wake_Lock_API) — `navigator.wakeLock.request('screen')`. The lock is re-acquired on `visibilitychange` (when the tab becomes visible again) and on `window.focus`. Browser support: Chromium-based browsers and Safari 16.4+; Firefox does not support it.

## Adding a Visualization

1. Create `src/visualizations/MyViz.tsx` — export a default React component that fills `100vw × 100vh`.
2. Import it in `src/App.tsx` and add an entry to the `VISUALIZATIONS` array: `{ id: 'my-viz', name: 'My Viz', component: MyViz }`.
3. That's it — the menu and localStorage persistence pick it up automatically.

## Environment Variables

None. This is a fully static app with no runtime environment.

## Non-Obvious Conventions

- **Hidden menu trigger**: the hamburger button is off-screen by default (`-translate-x-20`). An invisible 200×200px circle in the top-left corner triggers `isHovered`, which slides the button in. This keeps the display clean during ambient use.
- **Visualization selection persistence**: `localStorage` key `on-screen-current-viz` stores the last-selected visualization ID. Cleared only by the user manually.
- **Snakes is commented out**: `Snakes.tsx` exists in the repo but is excluded from `VISUALIZATIONS` and the import is commented out. It's a development candidate, not a broken feature.
- **Three.js is only used by OrbitingLights**: all other visualizations use the HTML Canvas 2D API directly via `useRef` + `useEffect`.

## Tagged Versions

- Initial release — 2026-01 — Static site with wake lock + 7 visualizations deployed to Fly.io
