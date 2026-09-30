# On-Screen — Vision

## Purpose

A browser-based ambient display that keeps your screen awake and fills it with generative art animations. Load it, walk away, and your screen stays on — showing something worth looking at.

## Target User

Joe Davis and a small circle of friends he shares the URL with. No registration, no accounts. Anyone with the link can use it.

## Value Proposition

The Screen Wake Lock API keeps the display alive without any OS-level configuration or third-party apps. Visitors get a gallery of canvas and Three.js animations to choose from, all full-screen and immersive. Simple enough to just work; interesting enough to keep running.

## Business Model

Personal project. No revenue model. No ads. Shared freely with a small audience.

## Competitive Landscape

Screensaver apps, browser-based generative art galleries (e.g. Shadertoy, openprocessing.org). On-screen differentiates by being dead-simple (one URL, no install) and combining the wake-lock utility with the art gallery in the same app.

## Roadmap

**Current (live)**
- Screen wake-lock via the Web Wake Lock API
- 7 generative visualizations: Fourier Epicycles, Particle Flow Field, Flocking Boids, Conway's Game of Life, Voronoi Diagrams, Circular Voronoi, Orbiting Lights
- Hidden hamburger menu (hover top-left to reveal), persistent selection via localStorage

**Possible future**
- Snakes visualization (shelved — needs more work; Opus may handle it better)
- Additional visualizations as inspiration strikes
- Light traffic mitigation if the URL ever gets shared more widely

## Out of Scope (Strategic)

- User accounts / auth
- Social features
- Backend or database
- Monetization
