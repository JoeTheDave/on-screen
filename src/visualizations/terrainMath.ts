// CPU twin of the GLSL terrain() in TerrainFlight.tsx, plus the camera's flight controller
// (which uses it to look ahead and steer around the high ground) and the forest placement
// (trees are real instanced models standing on this surface). It has to match the shader
// closely (it does, apart from float32/float64 rounding), so every constant here mirrors one
// in the shader.
//
// The landscape is periodic with period PERIOD in x and z. Each fbm octave stays periodic
// by using an integer "rotate and scale" matrix [[2,-1],[1,2]] (lacunarity sqrt(5), about
// 26.6 degrees of rotation) and wrapping coordinates mod PERIOD at every octave. This keeps
// every coordinate small, so float32 precision on the GPU never degrades however long the
// flight runs. The world repeats only every 256 units, which takes about half an hour of
// straight-line flight to cross.

export const PERIOD = 256
const MASK_PERIOD = 16 // region mask: 1/16 the frequency of the base octave
export const RAYMARCH_OCTAVES = 7

const mod = (x: number, p: number) => x - p * Math.floor(x / p)

// Integer hash (lowbias32-style). Same bit operations as ihash() in the shader.
function ihash(x: number, y: number): number {
  let h = (Math.imul(x, 0x8da6b343) ^ Math.imul(y, 0xd8163841)) >>> 0
  h = (h ^ (h >>> 16)) >>> 0
  h = Math.imul(h, 0x7feb352d) >>> 0
  h = (h ^ (h >>> 15)) >>> 0
  h = Math.imul(h, 0x846ca68b) >>> 0
  h = (h ^ (h >>> 16)) >>> 0
  return h
}

const hash01 = (x: number, y: number) => (ihash(x, y) >>> 8) / 16777216

// Periodic value noise with analytic derivatives (Inigo Quilez). Writes [value, d/dx, d/dy]
// into out; value is in [-1, 1].
function noised(px: number, py: number, per: number, out: Float64Array) {
  const ix = mod(Math.floor(px), per)
  const iy = mod(Math.floor(py), per)
  const fx = px - Math.floor(px)
  const fy = py - Math.floor(py)
  const ux = fx * fx * fx * (fx * (fx * 6 - 15) + 10)
  const uy = fy * fy * fy * (fy * (fy * 6 - 15) + 10)
  const dux = 30 * fx * fx * (fx * (fx - 2) + 1)
  const duy = 30 * fy * fy * (fy * (fy - 2) + 1)
  const jx = mod(ix + 1, per)
  const jy = mod(iy + 1, per)
  const a = hash01(ix, iy)
  const b = hash01(jx, iy)
  const c = hash01(ix, jy)
  const d = hash01(jx, jy)
  const k1 = b - a
  const k2 = c - a
  const k4 = a - b - c + d
  out[0] = -1 + 2 * (a + k1 * ux + k2 * uy + k4 * ux * uy)
  out[1] = 2 * dux * (k1 + k4 * uy)
  out[2] = 2 * duy * (k2 + k4 * ux)
}

const scratch = new Float64Array(3)

// Terrain height at world (x, z) using the given number of octaves.
export function terrainHeight(x: number, z: number, octaves: number): number {
  // Low-frequency region mask: mountain ranges vs rolling lowlands and lakes
  noised(mod(x / MASK_PERIOD, MASK_PERIOD), mod(z / MASK_PERIOD, MASK_PERIOD), MASK_PERIOD, scratch)
  const region = 0.5 + 0.5 * scratch[0]

  let px = mod(x, PERIOD)
  let py = mod(z, PERIOD)
  let a = 0
  let b = 1
  let dx = 0
  let dy = 0
  for (let i = 0; i < octaves; i++) {
    noised(px, py, PERIOD, scratch)
    dx += scratch[1]
    dy += scratch[2]
    // Dividing by the accumulated slope smooths steep areas: a cheap erosion look
    a += (b * scratch[0]) / (1 + dx * dx + dy * dy)
    b *= 0.47
    const nx = mod(2 * px - py, PERIOD)
    const ny = mod(px + 2 * py, PERIOD)
    px = nx
    py = ny
  }
  const s = Math.min(Math.max((region - 0.25) / 0.55, 0), 1)
  const amp = 0.16 + (0.7 - 0.16) * s * s * (3 - 2 * s)
  return amp * a + (region - 0.5) * 0.5
}

// ---------------------------------------------------------------- flight controller
//
// The camera cruises at a near-constant altitude and avoids high ground mainly by steering:
// a few times a second it scores a fan of constant-turn arcs about 18 s long and turns
// toward the one whose ground stays lowest (valleys and passes; each arc is judged with its
// neighbours, so it prefers wide corridors). Only when no corridor is clear does the
// altitude target rise, and that target is shaped for smoothness: it starts rising well
// before the obstacle (a sloped look-ahead envelope with seconds of lead), rises through a
// low-pass prefilter and a critically damped spring, and is released only slowly after.
// The filters tighten only if a big climb is suddenly needed (an "urgency" term).
// A hard floor (terrain + tallest tree + margin) at and just ahead of the camera
// guarantees no clipping even if the smooth path ever came up short.

export const TREE_MAX_HEIGHT = 0.018 // tallest possible tree (trunk base to crown top)

export interface FlightParams {
  speed: number // world units per second
  clearance: number // cruise height above the typical local ground (65th percentile)
  waterLevel: number
}

const PLAN_INTERVAL = 0.25 // seconds between re-plans
const PLAN_TURNS = 13 // candidate turn rates in the fan
const MAX_TURN = 0.16 // rad/s
const PLAN_SAMPLES = 36
const PLAN_STEP = 0.5 // seconds between samples along each arc (so 18 s ahead)
const W_BLOCKED = 4 // planner cost per unit of ground poking above the cruise altitude
const HYSTERESIS = 0.1 // cost of switching from the current turn to the opposite extreme
const PLAN_MARGIN = 0.08 // height kept above terrain + trees along the planned path
const COMMIT_SAMPLES = 100
const COMMIT_STEP = 0.1 // a fine look at the next 10 s along the path being flown
const CLIMB_SLOPE = 0.06 // obstacles further ahead count this much lower (rise / run)...
const CLIMB_LEAD = 5 // ...beyond this many seconds out, which covers the altitude filters' lag
const HARD_CLEARANCE = 0.02 // absolute floor above terrain + tallest tree
const CRUISE_SAMPLE_INTERVAL = 1
const CRUISE_TAU = 60 // cruise altitude drifts on a minutes scale
const CRUISE_PERCENTILE = 0.65 // of ground heights sampled over a ~2.6-unit disc
const PREFILTER_TAU = 1.2
const SPRING_TAU = 1.8
const URGENT_PREFILTER_TAU = 0.4
const URGENT_SPRING_TAU = 0.9
const RELEASE_TAU = 14 // how slowly a climb requirement is let go after the obstacle
const TURN_TAU = 1.6
const ROLL_TAU = 1.2
const BANK_PER_TURN = 2.0 // radians of bank per rad/s of turn
const MAX_BANK = 0.32
const BASE_PITCH = -0.08
const PITCH_TAU = 2.5
const INTENT_RELAX_TAU = 25

const wrapAngle = (a: number) => a - Math.PI * 2 * Math.floor((a + Math.PI) / (Math.PI * 2))
const ease = (dt: number, tau: number) => 1 - Math.exp(-dt / tau)
const smoothstep01 = (x: number) => {
  const t = Math.min(Math.max(x, 0), 1)
  return t * t * (3 - 2 * t)
}
const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x))

const CRUISE_RINGS = [0.6, 1.4, 2.6]
const FLOOR_AHEAD = [0.04, 0.08, 0.12]
const CRUISE_ANGLES = 8

export class Flight {
  x: number
  z: number
  y: number
  vy = 0
  heading: number
  turnRate = 0
  roll = 0
  pitch = BASE_PITCH
  floorHits = 0 // frames where the hard floor had to intervene (should stay 0)
  cruise: number // slowly drifting cruise altitude
  private cruiseGoal: number
  private need: number // held climb requirement from the planner
  private needGoal: number
  private pre: number // prefiltered altitude target
  private cmdTurn = 0
  private intent: number
  private planTimer = 0
  private urgency = 0
  private urgencyGoal = 0
  private cruiseTimer = 0
  private phases: number[]
  private samples: number[] = []
  private reqs = new Float64Array(PLAN_TURNS)
  private worsts = new Float64Array(PLAN_TURNS)
  private devs = new Float64Array(PLAN_TURNS)
  private costs = new Float64Array(PLAN_TURNS)

  constructor(x: number, z: number, heading: number, phases: number[], p: FlightParams) {
    this.x = mod(x, PERIOD)
    this.z = mod(z, PERIOD)
    this.heading = heading
    this.intent = heading
    this.phases = phases
    this.cruiseGoal = this.cruiseLevel(p)
    this.cruise = this.cruiseGoal
    this.needGoal = -Infinity
    this.need = -Infinity
    // Start above what every candidate arc would need, whichever one ends up flown; the
    // excess is released slowly
    this.plan(p)
    this.need = Math.max(this.needGoal, ...this.reqs)
    this.y = Math.max(this.cruise, this.need, this.floor(p))
    this.pre = this.y
  }

  // Ground or water surface, whichever is higher
  ground(x: number, z: number, water: number) {
    return Math.max(terrainHeight(x, z, RAYMARCH_OCTAVES), water)
  }

  // Lowest altitude the camera may ever be at: above terrain + trees here and just ahead
  floor(p: FlightParams) {
    const dx = Math.cos(this.heading)
    const dz = Math.sin(this.heading)
    let g = this.ground(this.x, this.z, p.waterLevel)
    for (const s of FLOOR_AHEAD) g = Math.max(g, this.ground(this.x + dx * s, this.z + dz * s, p.waterLevel))
    return g + TREE_MAX_HEIGHT + HARD_CLEARANCE
  }

  // Typical ground level around the camera (a percentile over a wide disc) + clearance
  private cruiseLevel(p: FlightParams) {
    const s = this.samples
    s.length = 0
    s.push(this.ground(this.x, this.z, p.waterLevel))
    for (const r of CRUISE_RINGS) {
      for (let i = 0; i < CRUISE_ANGLES; i++) {
        const a = (i / CRUISE_ANGLES) * Math.PI * 2 + r
        s.push(this.ground(this.x + Math.cos(a) * r, this.z + Math.sin(a) * r, p.waterLevel))
      }
    }
    s.sort((a, b) => a - b)
    return s[Math.floor(CRUISE_PERCENTILE * (s.length - 1))] + TREE_MAX_HEIGHT + p.clearance
  }

  // Score a fan of constant-turn arcs; pick the turn and the altitude it requires
  private plan(p: FlightParams) {
    const { reqs, worsts, devs, costs } = this
    const dist = p.speed * PLAN_STEP
    for (let k = 0; k < PLAN_TURNS; k++) {
      const w = MAX_TURN * ((2 * k) / (PLAN_TURNS - 1) - 1)
      let h = this.heading
      let px = this.x
      let pz = this.z
      let worst = -Infinity
      let req = -Infinity
      for (let j = 1; j <= PLAN_SAMPLES; j++) {
        const tj = j * PLAN_STEP
        // The turn rate eases toward w just like the real one does
        h += (w + (this.turnRate - w) * Math.exp(-tj / TURN_TAU)) * PLAN_STEP
        px += Math.cos(h) * dist
        pz += Math.sin(h) * dist
        const g = this.ground(px, pz, p.waterLevel) + TREE_MAX_HEIGHT + PLAN_MARGIN
        worst = Math.max(worst, g - this.cruise)
        req = Math.max(req, g - Math.max(0, tj - CLIMB_LEAD) * p.speed * CLIMB_SLOPE)
      }
      reqs[k] = req
      worsts[k] = worst
      devs[k] = wrapAngle(h - this.intent)
    }
    // An arc is only as clear as its neighbours: the path actually flown drifts a little
    // between them, and a wide valley beats threading a needle
    let best = 0
    for (let k = 0; k < PLAN_TURNS; k++) {
      const w = MAX_TURN * ((2 * k) / (PLAN_TURNS - 1) - 1)
      const lo = Math.max(k - 1, 0)
      const hi = Math.min(k + 1, PLAN_TURNS - 1)
      // Climb this corridor would force (near obstacles weigh most) plus anything poking
      // above the cruise altitude anywhere along it
      const climb = Math.max(reqs[k], reqs[lo], reqs[hi]) - this.cruise
      const blocked = Math.max(worsts[k], worsts[lo], worsts[hi])
      costs[k] =
        12 * Math.max(climb, 0) +
        W_BLOCKED * Math.max(blocked, 0) +
        0.25 * devs[k] * devs[k] +
        (HYSTERESIS * Math.abs(w - this.cmdTurn)) / MAX_TURN +
        0.01 * (w / MAX_TURN) ** 2
      if (costs[k] < costs[best]) best = k
    }
    const bestTurn = MAX_TURN * ((2 * best) / (PLAN_TURNS - 1) - 1)
    this.cmdTurn = bestTurn
    // The choice may flip to a neighbouring arc later, so the altitude target covers those too
    let req = reqs[best]
    req = Math.max(req, reqs[Math.max(best - 1, 0)], reqs[Math.min(best + 1, PLAN_TURNS - 1)])
    // Whatever the plan says, the next few seconds will be flown close to the current
    // turn, so that path's envelope always counts too (the filters need seconds of lead)
    let h = this.heading
    let px = this.x
    let pz = this.z
    const commitDist = p.speed * COMMIT_STEP
    let rate = 0
    for (let j = 1; j <= COMMIT_SAMPLES; j++) {
      const tj = j * COMMIT_STEP
      h += (bestTurn + (this.turnRate - bestTurn) * Math.exp(-tj / TURN_TAU)) * COMMIT_STEP
      px += Math.cos(h) * commitDist
      pz += Math.sin(h) * commitDist
      const g = this.ground(px, pz, p.waterLevel) + TREE_MAX_HEIGHT + PLAN_MARGIN
      req = Math.max(req, g - Math.max(0, tj - CLIMB_LEAD) * p.speed * CLIMB_SLOPE)
      rate = Math.max(rate, (g - this.y) / tj)
    }
    this.needGoal = req
    // Urgency: how far the climb rate this path demands exceeds a gentle one
    const gentle = p.speed * CLIMB_SLOPE
    this.urgencyGoal = smoothstep01((rate - 2 * gentle) / (2 * gentle))
  }

  step(dt: number, time: number, p: FlightParams) {
    if (!(dt > 0)) return
    // Where the flight "wants" to go drifts slowly, and relaxes toward wherever the
    // terrain has actually led it so it never tries to U-turn back
    const wander =
      0.07 * Math.sin(time * 0.031 + this.phases[0]) +
      0.05 * Math.sin(time * 0.0173 + this.phases[1]) +
      0.03 * Math.sin(time * 0.057 + this.phases[2])
    this.intent += wander * dt
    this.intent += wrapAngle(this.heading - this.intent) * ease(dt, INTENT_RELAX_TAU)
    this.intent = wrapAngle(this.intent)

    this.cruiseTimer -= dt
    if (this.cruiseTimer <= 0) {
      this.cruiseTimer += CRUISE_SAMPLE_INTERVAL
      this.cruiseGoal = this.cruiseLevel(p)
    }
    this.cruise += (this.cruiseGoal - this.cruise) * ease(dt, CRUISE_TAU)

    this.planTimer -= dt
    if (this.planTimer <= 0) {
      this.planTimer += PLAN_INTERVAL
      this.plan(p)
    }

    this.turnRate += (this.cmdTurn - this.turnRate) * ease(dt, TURN_TAU)
    this.heading = wrapAngle(this.heading + this.turnRate * dt)
    this.x = mod(this.x + Math.cos(this.heading) * p.speed * dt, PERIOD)
    this.z = mod(this.z + Math.sin(this.heading) * p.speed * dt, PERIOD)

    // Climb requirement: adopt a higher one at once (the filters below smooth it), let a
    // lower one go slowly
    if (this.needGoal > this.need) this.need = this.needGoal
    else this.need += (this.needGoal - this.need) * ease(dt, RELEASE_TAU)
    const target = Math.max(this.cruise, this.need)
    // Normally the climb is heavily smoothed; only if a big obstacle turns up late (say the
    // turn being flown sweeps into a mountainside) do the filters tighten, gradually
    this.urgency += (this.urgencyGoal - this.urgency) * ease(dt, 0.5)
    const urgency = this.urgency
    this.pre += (target - this.pre) * ease(dt, PREFILTER_TAU + (URGENT_PREFILTER_TAU - PREFILTER_TAU) * urgency)
    // Critically damped spring toward the prefiltered target (semi-implicit Euler)
    const w = 1 / (SPRING_TAU + (URGENT_SPRING_TAU - SPRING_TAU) * urgency)
    this.vy += (w * w * (this.pre - this.y) - 2 * w * this.vy) * dt
    this.y += this.vy * dt
    const fl = this.floor(p)
    if (this.y < fl) {
      this.y = fl
      this.vy = Math.max(this.vy, 0)
      this.floorHits++
    }

    const pitchTarget = BASE_PITCH + clamp((0.4 * this.vy) / p.speed, -0.03, 0.04)
    this.pitch += (pitchTarget - this.pitch) * ease(dt, PITCH_TAU)
    const rollTarget = clamp(this.turnRate * BANK_PER_TURN, -MAX_BANK, MAX_BANK)
    this.roll += (rollTarget - this.roll) * ease(dt, ROLL_TAU)
  }
}

// ---------------------------------------------------------------- forest placement
//
// Trees stand on a jittered world grid of TREE_CELL cells, one candidate per cell, so they
// never swim: everything about a candidate (position, size, species rank, color) comes from
// a hash of its cell, wrapped with the terrain period. The grid is handed out in square
// tiles of TREE_TILE world units that are populated once when they come near the camera.
//
// Only biome-independent suitability is decided here (slope, the forest/clearing pattern,
// the widest possible altitude band). Biome-dependent parts (density, water level, snow
// line, conifer share) are applied per frame on the GPU from the stored values, so a biome
// change grows and shrinks trees smoothly instead of repopulating anything.

export const TREE_CELL = 1 / 40
export const TREE_TILE = 1 // world units per tile side
export const TREE_TILE_CELLS = 40
const TREE_WRAP = PERIOD * TREE_TILE_CELLS // cells per terrain period
export const TREE_MIN_HEIGHT = 0.011
export const TREE_MAX_RADIUS = 0.62 * 1.15 * TREE_MAX_HEIGHT // widest crown (broadleaf, widest aspect)
export const TREE_SINK = 0.0015 // trunks start this far below the ground (plus slope * trunk radius)
// The highest any tree can grow: lowest snow line minus margin, highest water + tree line
const TREE_Y_MIN = -0.24 + 0.025
const TREE_Y_MAX = -0.08 + 0.95
const SLOPE_E = 0.004

// Floats stored per candidate: local x, ground y, local z, rotation, height,
// suitability, growth threshold, conifer rank, color mix, crown aspect, ground slope x / z,
// distance-thinning rank
export const TREE_FLOATS = 13

const treeScratch = new Float64Array(3)

// Forest/clearing pattern in 0..1 (integer frequency multipliers keep it periodic)
function clearing(x: number, z: number) {
  noised(mod(x * 3, PERIOD), mod(z * 3, PERIOD), PERIOD, treeScratch)
  const a = treeScratch[0]
  noised(mod(x, PERIOD), mod(z, PERIOD), PERIOD, treeScratch)
  const v = 0.65 * a + 0.35 * treeScratch[0]
  return smoothstep01((v + 0.5) / 0.5)
}

// Writes the candidate of cell (ci, cj) (unwrapped integer cell coordinates) relative to
// (ox, oz) into out at offset o. Returns false if no tree can ever grow there.
export function treeCandidate(ci: number, cj: number, ox: number, oz: number, out: Float32Array | Float64Array, o: number) {
  const wi = mod(ci, TREE_WRAP)
  const wj = mod(cj, TREE_WRAP)
  const h = ihash(wi ^ 0x9e3779b9, wj ^ 0x7f4a7c15)
  const r0 = (h & 255) / 255
  if (r0 > 0.97) return false
  const x = (ci + 0.5 + 0.75 * (((h >>> 8) & 255) / 255 - 0.5)) * TREE_CELL
  const z = (cj + 0.5 + 0.75 * (((h >>> 16) & 255) / 255 - 0.5)) * TREE_CELL
  const cl = clearing(x, z)
  if (cl <= r0) return false
  const y = terrainHeight(x, z, RAYMARCH_OCTAVES)
  if (y < TREE_Y_MIN || y > TREE_Y_MAX) return false
  const gx = (terrainHeight(x + SLOPE_E, z, RAYMARCH_OCTAVES) - y) / SLOPE_E
  const gz = (terrainHeight(x, z + SLOPE_E, RAYMARCH_OCTAVES) - y) / SLOPE_E
  const slope = Math.hypot(gx, gz)
  const s = cl * (1 - smoothstep01((slope - 0.45) / 0.4))
  if (s <= r0) return false
  const h2 = ihash(h, 0x51ed270b)
  const r1 = (h2 & 255) / 255
  const height = TREE_MIN_HEIGHT + (TREE_MAX_HEIGHT - TREE_MIN_HEIGHT) * (((h2 >>> 8) & 255) / 255)
  out[o] = x - ox
  out[o + 1] = y - TREE_SINK - slope * 0.06 * height
  out[o + 2] = z - oz
  out[o + 3] = (((h2 >>> 16) & 255) / 255) * Math.PI * 2
  out[o + 4] = height
  out[o + 5] = s
  out[o + 6] = r0
  out[o + 7] = ((h2 >>> 24) + 0.5) / 256 // conifer rank
  out[o + 8] = r1
  out[o + 9] = ((h >>> 24) & 255) / 255 // crown aspect
  out[o + 10] = gx
  out[o + 11] = gz
  out[o + 12] = (ihash(h2, 0x2c1b3c6d) >>> 8) / 16777216 // thinning rank
  return true
}

// Fills cells of rows [rowFrom, rowTo) of tile (ti, tj) into out starting at candidate
// index count, positions relative to the tile corner. Returns the new candidate count.
export function populateTreeRows(ti: number, tj: number, rowFrom: number, rowTo: number, out: Float32Array, count: number) {
  const ox = ti * TREE_TILE
  const oz = tj * TREE_TILE
  for (let j = rowFrom; j < rowTo; j++) {
    for (let i = 0; i < TREE_TILE_CELLS; i++) {
      if (treeCandidate(ti * TREE_TILE_CELLS + i, tj * TREE_TILE_CELLS + j, ox, oz, out, count * TREE_FLOATS)) count++
    }
  }
  return count
}

// ---------------------------------------------------------------- forest tiles
//
// Keeps every tile within reach of the camera populated (nearest and in-view first, a few
// rows per frame so no frame ever stalls), and on request writes the instances of the tiles
// in view into the GPU buffers. Positions are unwrapped world coordinates (the caller
// tracks the camera continuously across the terrain's wrap) relative to a floating origin.
//
// Far away only a thinned subset is drawn (and drawn larger, keeping the forest's coverage):
// candidate with thinning rank r is visible while TREE_THIN_DISTANCE / distance > r.

export const TREE_RADIUS = 11 // trees fade out completely at this distance
export const TREE_FADE_START = 8
export const TREE_THIN_DISTANCE = 1.2
const TILE_KEEP = TREE_RADIUS + 0.75 // populate tiles whose centre is this close
const TILE_DROP = TREE_RADIUS + 1.75
const THIN_MARGIN = 0.4 // world units of approach a rebuilt buffer stays valid for

interface Tile {
  ti: number
  tj: number
  key: number
  data: Float32Array | null
  count: number
  minY: number
  maxY: number
  dist: number
}

// Upper bounds of the biome-dependent growth factors over the next rebuild interval, so the
// CPU can skip candidates that are certainly invisible
export interface ForestBounds {
  density: number
  waterMin: number
  snowMax: number
}

export type BoxTest = (x0: number, y0: number, z0: number, x1: number, y1: number, z1: number) => boolean

const tileKey = (ti: number, tj: number) => (ti & 0xffff) * 65536 + (tj & 0xffff)
const smoothstep = (a: number, b: number, x: number) => smoothstep01((x - a) / (b - a))

export class Forest {
  tiles = new Map<number, Tile>()
  originX = 0
  originZ = 0
  populatedTiles = 0
  private current: Tile | null = null
  private row = 0
  private scratch = new Float32Array(TREE_TILE_CELLS * TREE_TILE_CELLS * TREE_FLOATS)
  private order: Tile[] = []

  // Adds/drops tiles around the camera (unwrapped x, z) and populates for up to budgetMs
  update(camX: number, camZ: number, fwdX: number, fwdZ: number, budgetMs: number) {
    const r = Math.ceil(TILE_KEEP / TREE_TILE) + 1
    const cti = Math.floor(camX / TREE_TILE)
    const ctj = Math.floor(camZ / TREE_TILE)
    for (let dj = -r; dj <= r; dj++) {
      for (let di = -r; di <= r; di++) {
        const ti = cti + di
        const tj = ctj + dj
        const d = Math.hypot((ti + 0.5) * TREE_TILE - camX, (tj + 0.5) * TREE_TILE - camZ)
        if (d > TILE_KEEP) continue
        const key = tileKey(ti, tj)
        if (!this.tiles.has(key)) {
          this.tiles.set(key, { ti, tj, key, data: null, count: 0, minY: 0, maxY: 0, dist: d })
        }
      }
    }
    for (const t of this.tiles.values()) {
      t.dist = Math.hypot((t.ti + 0.5) * TREE_TILE - camX, (t.tj + 0.5) * TREE_TILE - camZ)
      if (t.dist > TILE_DROP) {
        if (t === this.current) this.current = null
        if (t.data) this.populatedTiles--
        this.tiles.delete(t.key)
      }
    }

    const start = performance.now()
    while (performance.now() - start < budgetMs) {
      if (!this.current) {
        // Next tile: nearest, with tiles behind the camera counted as further away
        let best: Tile | null = null
        let bestScore = Infinity
        for (const t of this.tiles.values()) {
          if (t.data) continue
          const ahead = ((t.ti + 0.5) * TREE_TILE - camX) * fwdX + ((t.tj + 0.5) * TREE_TILE - camZ) * fwdZ
          const score = t.dist + (ahead < -1 ? TREE_RADIUS : 0)
          if (score < bestScore) {
            bestScore = score
            best = t
          }
        }
        if (!best) return
        this.current = best
        this.row = 0
        best.count = 0
      }
      const t = this.current
      const rows = Math.min(this.row + 2, TREE_TILE_CELLS)
      t.count = populateTreeRows(t.ti, t.tj, this.row, rows, this.scratch, t.count)
      this.row = rows
      if (this.row >= TREE_TILE_CELLS) {
        t.data = this.scratch.slice(0, t.count * TREE_FLOATS)
        let minY = Infinity
        let maxY = -Infinity
        for (let i = 0; i < t.count; i++) {
          minY = Math.min(minY, t.data[i * TREE_FLOATS + 1])
          maxY = Math.max(maxY, t.data[i * TREE_FLOATS + 1])
        }
        t.minY = t.count > 0 ? minY : 0
        t.maxY = t.count > 0 ? maxY : 0
        this.populatedTiles++
        this.current = null
      }
    }
  }

  // Moves the floating origin near the camera if it has drifted; returns true if it moved
  // (every instance position must then be rewritten)
  recenter(camX: number, camZ: number) {
    if (Math.abs(camX - this.originX) < 16 && Math.abs(camZ - this.originZ) < 16) return false
    this.originX = Math.round(camX / 16) * 16
    this.originZ = Math.round(camZ / 16) * 16
    return true
  }

  // Writes the instances of every populated tile that passes inView into the buffers
  // (matrix: yaw + uniform scale by tree height + translation relative to the origin;
  // tint: suitability, growth threshold, conifer rank, color mix; misc: crown aspect,
  // ground slope x/z, thinning rank). Nearest tiles first, so a full buffer drops the
  // farthest trees. Returns the instance count.
  // Tile distances are those of the last update().
  rebuild(inView: BoxTest, bounds: ForestBounds,
    matrices: Float32Array, tint: Float32Array, misc: Float32Array, capacity: number) {
    const order = this.order
    order.length = 0
    for (const t of this.tiles.values()) {
      if (!t.data || t.count === 0 || t.dist > TREE_RADIUS + 0.75) continue
      const x0 = t.ti * TREE_TILE - this.originX
      const z0 = t.tj * TREE_TILE - this.originZ
      if (!inView(x0, t.minY, z0, x0 + TREE_TILE, t.maxY + TREE_MAX_HEIGHT * 1.8, z0 + TREE_TILE)) continue
      order.push(t)
    }
    order.sort((a, b) => a.dist - b.dist)
    let n = 0
    for (const t of order) {
      const data = t.data as Float32Array
      const tx = t.ti * TREE_TILE - this.originX
      const tz = t.tj * TREE_TILE - this.originZ
      // Closest any tree of this tile can get to the camera before the next rebuild
      const near = Math.max(t.dist - 0.75 - THIN_MARGIN, 1e-3)
      const keep = Math.min(1, TREE_THIN_DISTANCE / near)
      for (let i = 0; i < t.count; i++) {
        const o = i * TREE_FLOATS
        if (data[o + 12] >= keep) continue
        const y = data[o + 1]
        const aw = y - bounds.waterMin
        const dens = bounds.density * data[o + 5] * smoothstep(0.025, 0.07, aw)
          * (1 - smoothstep(bounds.snowMax - 0.25, bounds.snowMax - 0.03, y))
        if (dens <= data[o + 6]) continue
        if (n >= capacity) return n
        const h = data[o + 4]
        const c = Math.cos(data[o + 3]) * h
        const s = Math.sin(data[o + 3]) * h
        const m = n * 16
        matrices[m] = c
        matrices[m + 1] = 0
        matrices[m + 2] = -s
        matrices[m + 3] = 0
        matrices[m + 4] = 0
        matrices[m + 5] = h
        matrices[m + 6] = 0
        matrices[m + 7] = 0
        matrices[m + 8] = s
        matrices[m + 9] = 0
        matrices[m + 10] = c
        matrices[m + 11] = 0
        matrices[m + 12] = tx + data[o]
        matrices[m + 13] = y
        matrices[m + 14] = tz + data[o + 2]
        matrices[m + 15] = 1
        const q = n * 4
        tint[q] = data[o + 5]
        tint[q + 1] = data[o + 6]
        tint[q + 2] = data[o + 7]
        tint[q + 3] = data[o + 8]
        misc[q] = data[o + 9]
        misc[q + 1] = data[o + 10]
        misc[q + 2] = data[o + 11]
        misc[q + 3] = data[o + 12]
        n++
      }
    }
    return n
  }
}
