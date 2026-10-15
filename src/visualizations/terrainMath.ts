// CPU twin of the GLSL terrain() in TerrainFlight.tsx, plus the camera's flight controller
// (which uses it to look ahead and steer around the high ground). It has to match the shader closely (it does, apart
// from float32/float64 rounding), so every constant here mirrors one in the shader.
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

export const TREE_MAX_HEIGHT = 0.018 // mirrors TREE_H in the shader (tallest possible tree)

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
