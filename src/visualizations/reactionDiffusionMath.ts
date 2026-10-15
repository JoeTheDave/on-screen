// Phase planning for the Reaction Diffusion visualization. Pure math, no Three.js, so it can be
// exercised in node.
//
// The show runs in cycles. Each cycle opens with an initial spread (drops grow across the screen
// under one Gray-Scott regime), then 3-5 transformations: a new regime is released at one or a
// few points (or along a line) and sweeps outward as an advancing front, converting the pattern
// it passes over. Behind the front is the new regime, ahead of it the old one, so at any moment
// the screen holds at most two. Finally an erasing front clears the board for the next cycle.
//
// The front's distance function (frontDistance) and the nucleus hash (seedNucleus) are mirrored
// exactly in the shaders, so the planner can measure how far the front has to travel.

export type Rand = () => number

export interface Regime {
  name: string
  feed: number
  kill: number
}

// Gray-Scott regimes (Sims' scaling: dA = 1, dB = 0.5, dt = 1, 9-point Laplacian), chosen from
// four visually distinct families: mazes, spots, holes (inverted spots) and bubbling chaos
export const REGIMES: Regime[] = [
  { name: 'coral', feed: 0.0545, kill: 0.062 }, // 0 maze: branching coral
  { name: 'mitosis', feed: 0.0367, kill: 0.0649 }, // 1 spots: dividing cells
  { name: 'labyrinth', feed: 0.029, kill: 0.057 }, // 2 maze: fine fingerprint labyrinth
  { name: 'pearls', feed: 0.03, kill: 0.062 }, // 3 spots: dense stable pearls
  { name: 'holes', feed: 0.039, kill: 0.058 }, // 4 holes: solid glaze pierced by pits
  { name: 'bubbles', feed: 0.018, kill: 0.051 }, // 5 chaos: rings that swell, burst and split
  { name: 'fingerprint', feed: 0.037, kill: 0.06 }, // 6 maze: broad whorls
  { name: 'worms', feed: 0.05, kill: 0.063 }, // 7 maze: thick broken worms
]

// Transitions that were simulated (half-float CPU replica, GPU-sized grid, front speed as
// shipped) and produce a clear change while staying alive; switching parameters alone kills the
// pattern for many other pairs (e.g. coral -> mitosis). Every edge crosses families. `seeded`
// edges are ones where the old pattern melts away under the new parameters, so fresh nuclei are
// planted just behind the front and the new pattern blooms from them.
interface Transition {
  to: number
  seeded: boolean
}
const m = (to: number): Transition => ({ to, seeded: false })
const s = (to: number): Transition => ({ to, seeded: true })
export const TRANSITIONS: Transition[][] = [
  [s(1), s(3), m(4), s(5)], // coral -> dividing cells, pearls, pitted glaze, bubbles
  [m(0), m(2), m(4), m(5)], // mitosis -> spots fuse to coral, spots string into rings, inversion, bursting
  [s(1), s(3), m(4), s(5)], // labyrinth -> cells, pearls, pitted glaze, bubbles
  [m(0), m(2), m(4)], // pearls -> coral, concentric labyrinth rings, inversion
  [m(6), s(1), s(3), s(5)], // holes -> pits stretch into whorls, cells, pearls, bubbles
  [m(2), m(4), s(0), s(3)], // bubbles -> labyrinth, holes, coral, pearls
  [m(7), s(1), m(4), s(5)], // fingerprint -> worms, cells, holes, bubbles
  [s(1), m(4), s(5)], // worms -> cells, holes, bubbles
]

export const ORIGINS = 3 // front start points (unused slots repeat the first)
export const WOBBLES = 3 // plane waves that rag the front's edge
export const SEED_GAP = 12 // texels behind the front before nuclei are planted, so the old
// pattern has melted and they land on clear ground
export const SEED_BAND = 5 // texels of front travel during which a nucleus is held
export const SEED_CELL = 12 // at most one nucleus per SEED_CELL x SEED_CELL texels
export const SEED_RADIUS = 5 // texels; coral needs a disc of radius ~4 to take, so with margin
export const BLEACH_KILL = 0.08 // extra kill rate inside a bleach band: clears B in ~100 steps
export const FAR = 4096 // front radius that puts every texel behind the front
export const FRONT_SPEED = 10 // texels per sim-second the front aims for
export const MIN_SWEEP_SECONDS = 20
export const MAX_SWEEP_SECONDS = 34
export const TRANSFORM_HOLD_SECONDS = 6 // after the sweep, let the trailing edge finish converting
export const SPREAD_SECONDS = 15 // 22 drops fill the screen in ~9 s; a few seconds to admire it
export const RESET_SWEEP_SECONDS = 15
export const RESET_HOLD_SECONDS = 2
export const MIN_TRANSFORMS = 3
export const MAX_TRANSFORMS = 5
const SEEDED_DENSITY = [0.3, 0.42]
const BLEACH_CHANCE = 0.4 // seeded edges sometimes erase first and regrow in the wake
const BLEACH_WIDTH = [16, 28]
const WOBBLE_AMPLITUDE = [2, 4] // texels per wave
const WOBBLE_K = [2, 4]

export type PhaseKind = 'spread' | 'transform' | 'reset'

export interface Phase {
  kind: PhaseKind
  from: number // regime ahead of the front
  to: number // regime behind it
  palette: number // palette behind the front (ahead keeps the previous one)
  origins: Float32Array // ORIGINS x (u, v, weight x, weight y): weights (1,1) = point, (0,1) = line
  wobble: Float32Array // WOBBLES x (kx, ky, phase, amplitude in texels)
  bleach: number // width (texels) of the erasing band just behind the front, 0 = none
  seedDensity: number // share of seed cells that get a nucleus, 0 = none
  seedSalt: number // integer < 2^16
  sweepSeconds: number
  holdSeconds: number
  startRadius: number // texels; set by measureFront for the current sim size
  endRadius: number
}

const between = (rand: Rand, lo: number, hi: number) => lo + (hi - lo) * rand()
const pick = <T>(rand: Rand, items: T[]): T => items[Math.floor(rand() * items.length) % items.length]

const TAU = Math.PI * 2

// Same as the shader's frontDistance: texel distance to the nearest origin (wrapped, since the sim
// tiles), plus the ragged-edge wobble. Integer wave numbers keep the wobble seamless too.
export function frontDistance(
  origins: Float32Array,
  wobble: Float32Array,
  u: number,
  v: number,
  simW: number,
  simH: number,
): number {
  let d = Infinity
  for (let i = 0; i < ORIGINS; i++) {
    let du = u - origins[i * 4]
    let dv = v - origins[i * 4 + 1]
    du -= Math.floor(du + 0.5)
    dv -= Math.floor(dv + 0.5)
    d = Math.min(d, Math.hypot(du * simW * origins[i * 4 + 2], dv * simH * origins[i * 4 + 3]))
  }
  for (let i = 0; i < WOBBLES; i++) {
    const w = wobble.subarray(i * 4, i * 4 + 4)
    d += w[3] * Math.sin(TAU * (w[0] * u + w[1] * v) + w[2])
  }
  return d
}

// Integer hash shared with the shader (uint math; Math.imul + >>> 0 reproduce it exactly)
export function hash3(x: number, y: number, z: number): number {
  let h = (Math.imul(x, 0x9e3779b1) ^ Math.imul(y, 0x85ebca77) ^ Math.imul(z, 0xc2b2ae3d)) >>> 0
  h ^= h >>> 15
  h = Math.imul(h, 0x2c1b3c6d) >>> 0
  h ^= h >>> 12
  h = Math.imul(h, 0x297a2d39) >>> 0
  h ^= h >>> 15
  return h >>> 0
}

// Center (texels) of the nucleus covering texel (px, py), or null; same test as the sim shader
export function seedNucleus(px: number, py: number, density: number, salt: number): [number, number] | null {
  const cx = Math.floor(px / SEED_CELL)
  const cy = Math.floor(py / SEED_CELL)
  const h = hash3(cx, cy, salt)
  if ((h & 0xffff) / 65536 >= density) return null
  const jx = (cx + 0.25 + (0.5 * ((h >>> 16) & 0xff)) / 255) * SEED_CELL
  const jy = (cy + 0.25 + (0.5 * (h >>> 24)) / 255) * SEED_CELL
  return Math.hypot(px + 0.5 - jx, py + 0.5 - jy) < SEED_RADIUS ? [jx, jy] : null
}

// How far behind the front (texels) a nucleus's center must be before it is planted
export const seedStart = (bleach: number) => Math.max(bleach, SEED_GAP) + SEED_RADIUS

// One point, a cluster of two or three, or a line across the screen (horizontal or vertical)
const randomOrigins = (rand: Rand): Float32Array => {
  const o = new Float32Array(ORIGINS * 4)
  const r = rand()
  const count = r < 0.45 ? 1 : r < 0.8 ? 2 + Math.floor(rand() * 2) : 0
  if (count === 0) {
    const horizontal = rand() < 0.5
    const at = rand()
    for (let i = 0; i < ORIGINS; i++) o.set([at, at, horizontal ? 0 : 1, horizontal ? 1 : 0], i * 4)
    return o
  }
  for (let i = 0; i < ORIGINS; i++) {
    if (i < count) o.set([rand(), rand(), 1, 1], i * 4)
    else o.copyWithin(i * 4, 0, 4)
  }
  return o
}

const randomWobble = (rand: Rand): Float32Array => {
  const w = new Float32Array(WOBBLES * 4)
  for (let i = 0; i < WOBBLES; i++) {
    let kx = 0
    let ky = 0
    while (Math.hypot(kx, ky) < WOBBLE_K[0] || Math.hypot(kx, ky) > WOBBLE_K[1]) {
      kx = Math.floor(rand() * (2 * WOBBLE_K[1] + 1)) - WOBBLE_K[1]
      ky = Math.floor(rand() * (WOBBLE_K[1] + 1))
    }
    w.set([kx, ky, rand() * TAU, between(rand, WOBBLE_AMPLITUDE[0], WOBBLE_AMPLITUDE[1])], i * 4)
  }
  return w
}

const basePhase = (kind: PhaseKind, from: number, to: number, palette: number, rand: Rand): Phase => ({
  kind,
  from,
  to,
  palette,
  origins: randomOrigins(rand),
  wobble: randomWobble(rand),
  bleach: 0,
  seedDensity: 0,
  seedSalt: Math.floor(rand() * 65536),
  sweepSeconds: 0,
  holdSeconds: 0,
  startRadius: FAR,
  endRadius: FAR,
})

const otherThan = (rand: Rand, count: number, avoid: number) => {
  if (count < 2 || avoid < 0) return Math.floor(rand() * count)
  let next = Math.floor(rand() * (count - 1))
  if (next >= avoid) next++
  return next
}

// A whole cycle: spread, 3-5 transformations along verified edges (preferring regimes not yet
// seen this cycle), then the erasing reset. Palettes never repeat back to back. The spread keeps
// openingPalette (the one the previous reset brought in; -1 for a random start).
export function planCycle(rand: Rand, paletteCount: number, lastRegime: number, openingPalette: number): Phase[] {
  const first = otherThan(rand, REGIMES.length, lastRegime)
  let palette = openingPalette >= 0 ? openingPalette : Math.floor(rand() * paletteCount)
  const phases: Phase[] = [basePhase('spread', first, first, palette, rand)]
  phases[0].sweepSeconds = SPREAD_SECONDS

  const seen = new Set([first])
  let regime = first
  const transforms = MIN_TRANSFORMS + Math.floor(rand() * (MAX_TRANSFORMS - MIN_TRANSFORMS + 1))
  for (let i = 0; i < transforms; i++) {
    const options = TRANSITIONS[regime]
    const fresh = options.filter((t) => !seen.has(t.to))
    const edge = pick(rand, fresh.length > 0 ? fresh : options)
    palette = otherThan(rand, paletteCount, palette)
    const phase = basePhase('transform', regime, edge.to, palette, rand)
    phase.holdSeconds = TRANSFORM_HOLD_SECONDS
    if (edge.seeded) {
      phase.seedDensity = between(rand, SEEDED_DENSITY[0], SEEDED_DENSITY[1])
      if (rand() < BLEACH_CHANCE) phase.bleach = between(rand, BLEACH_WIDTH[0], BLEACH_WIDTH[1])
    }
    phases.push(phase)
    seen.add(edge.to)
    regime = edge.to
  }

  // The reset erases everything behind its front; the next cycle's palette rides in on it
  const reset = basePhase('reset', regime, regime, otherThan(rand, paletteCount, palette), rand)
  reset.bleach = FAR
  reset.sweepSeconds = RESET_SWEEP_SECONDS
  reset.holdSeconds = RESET_HOLD_SECONDS
  phases.push(reset)
  return phases
}

// Measures the front against the current sim size: it starts just inside its nearest point and
// ends once its trailing bands have passed the farthest texel. Call again after a resize.
export function measureFront(phase: Phase, simW: number, simH: number) {
  if (phase.kind === 'spread') {
    phase.startRadius = FAR
    phase.endRadius = FAR
    return
  }
  let lo = Infinity
  let hi = -Infinity
  const step = 4
  for (let y = 0; y < simH; y += step) {
    for (let x = 0; x < simW; x += step) {
      const d = frontDistance(phase.origins, phase.wobble, (x + 0.5) / simW, (y + 0.5) / simH, simW, simH)
      lo = Math.min(lo, d)
      hi = Math.max(hi, d)
    }
  }
  const trail =
    phase.kind === 'reset' ? 0 : phase.seedDensity > 0 ? seedStart(phase.bleach) + SEED_BAND + SEED_RADIUS : phase.bleach
  phase.startRadius = lo - 2
  phase.endRadius = hi + trail + step + 2
}

// Sets the radii and, for a transformation, a sweep time that moves the front at about
// FRONT_SPEED within the allowed range
export function startPhase(phase: Phase, simW: number, simH: number) {
  measureFront(phase, simW, simH)
  if (phase.kind === 'transform') {
    const travel = phase.endRadius - phase.startRadius
    phase.sweepSeconds = Math.min(Math.max(travel / FRONT_SPEED, MIN_SWEEP_SECONDS), MAX_SWEEP_SECONDS)
  }
}

export const phaseSeconds = (phase: Phase) => phase.sweepSeconds + phase.holdSeconds

// Front radius (texels) t sim-seconds into the phase; parked past everything once the sweep ends
export function frontRadius(phase: Phase, t: number): number {
  if (phase.kind === 'spread') return FAR
  const f = Math.min(Math.max(t / phase.sweepSeconds, 0), 1)
  return phase.startRadius + (phase.endRadius - phase.startRadius) * f
}

// Coverage probe: share of samples (red = B * 2 in 0..255) where chemical B is clearly present
export function coverage(pixels: Uint8Array): number {
  let alive = 0
  const n = pixels.length / 4
  for (let i = 0; i < n; i++) if (pixels[i * 4] > 51) alive++
  return alive / n
}
