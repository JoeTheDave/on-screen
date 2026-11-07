# On-Screen — Claude Code Context

## Read these first
- `VISION.md` — what this app is and why
- `PROJECT.md` — how it's built, deployed, and what conventions apply

## Critical conventions for this project
- **Frontend-only.** No backend, no database, no auth. Do not add Express, Prisma, or any server-side infrastructure.
- **Screen Wake Lock lives in `main.tsx`**, not in any component. Keep it there.
- **Snakes is shelved** — `src/visualizations/Snakes.tsx` exists but is intentionally commented out in `App.tsx`. Do not delete it; do not re-enable it without being asked.
- **Retired visualizations** (2026-11-07): Voronoi Diagram, Circular Voronoi, Orbiting Lights, Sphere Field, Black Hole, Terrain Flight, Atomic Bomb, Prism are commented out of `App.tsx` the same way — off the menu, code kept. Same rule: don't delete, don't re-enable unless asked.

## Local Dev

```bash
npm install
npm run dev
```

App: http://localhost:5173

## Deploy

```bash
npm run deploy   # alias for: fly deploy
```

Production: https://on-screen.fly.dev
