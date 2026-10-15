// Pure simulation for the Atomic Bomb visualization: city generation, blast physics (shock
// radius, overpressure, damage tiers), pooled rigid debris, dust, the mushroom cloud's shape over
// time and camera framing. No three.js in here, so all of it can be exercised headlessly.
//
// Units are meters-ish at building scale and seconds of screen time. The blast itself is
// compressed (a small town, a stylized time scale) but keeps the physical *forms*: a
// Sedov–Taylor shock that decays into a sound wave, a power-law overpressure, Rankine–Hugoniot
// wind behind the front, Friedlander positive/negative phases and damage tiers by overpressure.

export type Rng = () => number

export const mulberry32 = (seed: number): Rng => {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const range = (rng: Rng, a: number, b: number) => a + (b - a) * rng()
const clamp = (x: number, a: number, b: number) => (x < a ? a : x > b ? b : x)
export const smoothstep = (e0: number, e1: number, x: number) => {
  const t = clamp((x - e0) / (e1 - e0), 0, 1)
  return t * t * (3 - 2 * t)
}

// ---------------------------------------------------------------------------------------------
// City

export const BLOCK_PITCH = 56 // block + road, along both grid axes
export const ROAD_WIDTH = 12
export const SIDEWALK = 2.5
export const MAP_SIZE = 32 // block-map texture is MAP_SIZE² cells centered on the city
export const MAX_BUILDINGS = 1600

export const KIND_HOUSE = 0
export const KIND_MIDRISE = 1
export const KIND_TOWER = 2
export const KIND_INDUSTRY = 3

// Block types, stored in the block map's G channel for the ground shader
export const BLOCK_NONE = 0
export const BLOCK_HOMES = 1
export const BLOCK_MIDRISE = 2
export const BLOCK_DOWNTOWN = 3
export const BLOCK_PARK = 4
export const BLOCK_INDUSTRY = 5

export const TIER_NONE = 0
export const TIER_LIGHT = 1 // windows out, sway
export const TIER_MODERATE = 2 // roof sheared off, slight lean
export const TIER_HEAVY = 3 // upper floors and the leeward facade torn away
export const TIER_DESTROYED = 4 // blown apart down to a rubble mound

export interface Building {
  x: number
  z: number
  w: number // along the grid's local x
  d: number // along the grid's local z
  h: number
  kind: number
  strength: number // overpressure (psi) at which it is destroyed outright
  color: [number, number, number] // sRGB 0..1
  roofColor: [number, number, number]
  roofH: number // pitched roof height, 0 for flat roofs
  floorH: number
  seed: number
  lit: number // fraction of windows lit at night
  // planned by planBlast
  dist: number // ground distance to ground zero
  psi: number
  tier: number
  hitTime: number
  burn: number // thermal exposure 0..1
  dirX: number // unit ground direction away from ground zero
  dirZ: number
  cut: number // fraction of height torn away (heavy tier)
  pieces: number // debris budget
  // response
  hit: boolean
  curH: number
  roofOn: boolean
  lean: number // permanent lean (rad)
  sway: number // sway amplitude (rad)
  swayPeriod: number
}

export interface City {
  radius: number
  angle: number // grid rotation about +Y (three.js convention)
  cos: number
  sin: number
  downtownX: number
  downtownZ: number
  maxHeight: number
  buildings: Building[]
  map: Uint8Array // MAP_SIZE² RGBA: R city mask, G block type, B shade, A 255
}

const HOUSE_COLORS: [number, number, number][] = [
  [0.86, 0.82, 0.74], [0.78, 0.7, 0.6], [0.72, 0.62, 0.52], [0.9, 0.88, 0.84], [0.62, 0.66, 0.68],
  [0.74, 0.56, 0.44], [0.8, 0.76, 0.62], [0.66, 0.72, 0.64],
]
const ROOF_COLORS: [number, number, number][] = [
  [0.42, 0.2, 0.16], [0.3, 0.3, 0.32], [0.48, 0.3, 0.22], [0.24, 0.24, 0.26], [0.36, 0.24, 0.2],
]
const MIDRISE_COLORS: [number, number, number][] = [
  [0.72, 0.7, 0.66], [0.62, 0.6, 0.58], [0.7, 0.6, 0.5], [0.56, 0.42, 0.36], [0.78, 0.74, 0.66],
  [0.5, 0.52, 0.55],
]
const TOWER_COLORS: [number, number, number][] = [
  [0.42, 0.5, 0.58], [0.55, 0.58, 0.62], [0.36, 0.4, 0.44], [0.62, 0.62, 0.6], [0.32, 0.38, 0.46],
]
const INDUSTRY_COLORS: [number, number, number][] = [
  [0.58, 0.6, 0.6], [0.5, 0.56, 0.62], [0.6, 0.48, 0.38], [0.66, 0.66, 0.62],
]

const pick = <T>(rng: Rng, list: T[]): T => list[Math.floor(rng() * list.length) % list.length]
const jitterColor = (rng: Rng, c: [number, number, number]): [number, number, number] => {
  const k = range(rng, 0.88, 1.08)
  return [clamp(c[0] * k, 0, 1), clamp(c[1] * k, 0, 1), clamp(c[2] * k, 0, 1)]
}

export function generateCity(rng: Rng): City {
  const radius = range(rng, 430, 560)
  const angle = rng() * Math.PI * 0.5
  const cos = Math.cos(angle)
  const sin = Math.sin(angle)
  const dtA = rng() * Math.PI * 2
  const dtR = Math.sqrt(rng()) * 0.25 * radius
  const downtownX = Math.cos(dtA) * dtR
  const downtownZ = Math.sin(dtA) * dtR
  const ph1 = rng() * Math.PI * 2
  const ph2 = rng() * Math.PI * 2
  const map = new Uint8Array(MAP_SIZE * MAP_SIZE * 4)
  const buildings: Building[] = []
  const half = MAP_SIZE / 2
  const inner = (BLOCK_PITCH - ROAD_WIDTH) / 2 - SIDEWALK // usable half-size of a block
  let maxHeight = 0

  // Local grid frame -> world, matching a three.js rotation of `angle` about +Y
  const add = (
    lx: number, lz: number, w: number, d: number, h: number, kind: number,
    color: [number, number, number], strength: number, roofH = 0,
  ) => {
    if (buildings.length >= MAX_BUILDINGS) return
    maxHeight = Math.max(maxHeight, h + roofH)
    buildings.push({
      x: lx * cos + lz * sin,
      z: -lx * sin + lz * cos,
      w, d, h, kind, strength,
      color: jitterColor(rng, color),
      roofColor: jitterColor(rng, pick(rng, ROOF_COLORS)),
      roofH,
      floorH: kind === KIND_HOUSE ? 2.9 : kind === KIND_INDUSTRY ? 5 : range(rng, 3.2, 3.8),
      seed: rng(),
      lit: 0,
      dist: 0, psi: 0, tier: 0, hitTime: Infinity, burn: 0, dirX: 1, dirZ: 0, cut: 0, pieces: 0,
      hit: false, curH: h, roofOn: roofH > 0, lean: 0, sway: 0, swayPeriod: 1,
    })
  }

  for (let j = -half; j < half; j++) {
    for (let i = -half; i < half; i++) {
      const lx = (i + 0.5) * BLOCK_PITCH
      const lz = (j + 0.5) * BLOCK_PITCH
      const wx = lx * cos + lz * sin
      const wz = -lx * sin + lz * cos
      const r = Math.hypot(wx, wz)
      const th = Math.atan2(wz, wx)
      const edge = radius * (1 + 0.12 * Math.sin(3 * th + ph1) + 0.07 * Math.sin(5 * th + ph2))
      if (r > edge) continue
      const u = Math.exp(-((wx - downtownX) ** 2 + (wz - downtownZ) ** 2) / (0.3 * radius) ** 2)
      let type: number
      if (rng() < 0.05 && u < 0.8) type = BLOCK_PARK
      else if (u > 0.55) type = BLOCK_DOWNTOWN
      else if (u > 0.17 || (r < 0.35 * radius && rng() < 0.5)) type = BLOCK_MIDRISE
      else if (r > 0.55 * radius && rng() < 0.16) type = BLOCK_INDUSTRY
      else type = BLOCK_HOMES
      const mi = ((j + half) * MAP_SIZE + (i + half)) * 4
      map[mi] = 255
      map[mi + 1] = type
      map[mi + 2] = Math.floor(rng() * 255)
      map[mi + 3] = 255

      if (type === BLOCK_HOMES) {
        // 3 x 2 lots, houses set back toward their street
        const lotW = (2 * inner) / 3
        for (let ci = 0; ci < 3; ci++) {
          for (let cj = 0; cj < 2; cj++) {
            if (rng() < 0.08) continue
            const w = range(rng, 7, 10.5)
            const d = range(rng, 7, 10.5)
            const h = rng() < 0.35 ? range(rng, 6.2, 7.6) : range(rng, 3.6, 4.8)
            const cx = lx - inner + lotW * (ci + 0.5) + range(rng, -1, 1)
            const cz = lz + (cj === 0 ? -1 : 1) * (inner / 2 + 1.5)
            add(cx, cz, w, d, h, KIND_HOUSE, pick(rng, HOUSE_COLORS), range(rng, 3.5, 5),
              rng() < 0.85 ? range(rng, 2, 3.4) : 0)
          }
        }
      } else if (type === BLOCK_MIDRISE) {
        const k = 0.7 + 0.6 * u
        const col = pick(rng, MIDRISE_COLORS)
        const layout = rng()
        if (layout < 0.4) {
          // two slabs facing their streets
          const alongX = rng() < 0.5
          for (const s of [-1, 1]) {
            const depth = range(rng, 10, 15)
            const L = 2 * inner * range(rng, 0.78, 0.95)
            const off = s * (inner - depth / 2 - 1)
            const h = range(rng, 12, 34) * k
            if (alongX) add(lx, lz + off, L, depth, h, KIND_MIDRISE, col, range(rng, 7, 10))
            else add(lx + off, lz, depth, L, h, KIND_MIDRISE, col, range(rng, 7, 10))
          }
        } else if (layout < 0.8) {
          for (const sx of [-1, 1]) {
            for (const sz of [-1, 1]) {
              if (rng() < 0.12) continue
              const w = range(rng, 12, 16.5)
              const d = range(rng, 12, 16.5)
              add(lx + sx * inner * 0.5, lz + sz * inner * 0.5, w, d, range(rng, 10, 30) * k,
                KIND_MIDRISE, rng() < 0.5 ? col : pick(rng, MIDRISE_COLORS), range(rng, 7, 10))
            }
          }
        } else {
          add(lx, lz, range(rng, 26, 34), range(rng, 26, 34), range(rng, 12, 24) * k, KIND_MIDRISE, col,
            range(rng, 7, 10))
        }
      } else if (type === BLOCK_DOWNTOWN) {
        const col = pick(rng, TOWER_COLORS)
        if (rng() < 0.55) {
          // one tower on a corner, low podiums on the rest
          const cx = rng() < 0.5 ? -1 : 1
          const cz = rng() < 0.5 ? -1 : 1
          const tw = range(rng, 16, 22)
          add(lx + cx * (inner - tw / 2 - 1), lz + cz * (inner - tw / 2 - 1), tw, range(rng, 16, 22),
            range(rng, 60, 150) * (0.6 + 0.4 * u), KIND_TOWER, col, range(rng, 11, 16))
          add(lx - cx * inner * 0.45, lz, range(rng, 14, 18), 2 * inner * 0.85, range(rng, 8, 18),
            KIND_MIDRISE, pick(rng, MIDRISE_COLORS), range(rng, 7, 10))
          add(lx + cx * inner * 0.45, lz - cz * inner * 0.5, range(rng, 12, 16), range(rng, 12, 16),
            range(rng, 10, 26), KIND_MIDRISE, pick(rng, MIDRISE_COLORS), range(rng, 7, 10))
        } else {
          // twin towers
          for (const s of [-1, 1]) {
            const tw = range(rng, 12, 16)
            add(lx + s * inner * 0.5, lz + range(rng, -6, 6), tw, range(rng, 12, 16),
              range(rng, 45, 115) * (0.6 + 0.4 * u), KIND_TOWER, rng() < 0.6 ? col : pick(rng, TOWER_COLORS),
              range(rng, 11, 16))
          }
        }
      } else if (type === BLOCK_INDUSTRY) {
        const n = rng() < 0.5 ? 1 : 2
        for (let s = 0; s < n; s++) {
          const off = n === 1 ? 0 : (s === 0 ? -1 : 1) * inner * 0.5
          add(lx + off, lz + range(rng, -3, 3), n === 1 ? range(rng, 26, 36) : range(rng, 14, 17),
            range(rng, 18, 32), range(rng, 6, 11), KIND_INDUSTRY, pick(rng, INDUSTRY_COLORS), range(rng, 4.5, 6.5))
        }
      }
    }
  }
  return { radius, angle, cos, sin, downtownX, downtownZ, maxHeight, buildings, map }
}

// ---------------------------------------------------------------------------------------------
// Blast

export const DEBRIS_POOL = 3600
export const T5 = 2.0 // seconds for the shock to reach the 5 psi radius (stylized time scale)
const PSI_MAX = 120
const OVERPRESSURE_EXPONENT = 1.84 // power-law fit through 5 psi at r5 and 1 psi at 2.4 r5
const AMBIENT_PSI = 14.7
const SOUND_SPEED_MS = 340
const WIND_SCALE = 0.5 // m/s of real Rankine–Hugoniot wind -> units/s on screen
const MAX_LAUNCH = 110
export const GRAVITY = 9.8
const DRAG = 0.03 // quadratic drag coefficient per unit of piece size
const RESTITUTION = 0.22
const FRICTION = 0.55

export interface Blast {
  gx: number
  gz: number
  hb: number // burst height
  yieldKt: number
  r5: number // 5 psi slant radius
  rf: number // maximum fireball radius
  capH: number // final height of the cap's center
  capRmin: number // final tube radius of the toroidal cap
  capRmaj: number // final ring radius of the cap
  stemR: number
  groundBurst: boolean
  humidity: number // strength of the Wilson condensation cloud
  thermalR: number // radius out to which surfaces scorch
  // shock law: Sedov–Taylor R = k t^0.4 until its speed falls to c, then a sound wave
  k: number
  c: number
  tc: number
  rc: number
}

export function createBlast(rng: Rng, city: City): Blast {
  const yieldKt = 5 * Math.pow(100 / 5, rng()) // log-uniform 5..100 kt
  const s = Math.cbrt(yieldKt / 5)
  const r5 = city.radius * 0.27 * s // damage radii scale with the cube root of yield
  const rf = 0.21 * r5 * Math.pow(yieldKt / 5, 0.4 - 1 / 3) // fireball radius ∝ W^0.4
  const a = rng() * Math.PI * 2
  const d = Math.sqrt(rng()) * 0.2 * city.radius
  const groundBurst = rng() < 0.3
  const hb = groundBurst ? rf * range(rng, 0.3, 0.75) : rf * range(rng, 1.3, 2.6)
  const k = r5 / Math.pow(T5, 0.4)
  // The front moves at Mach ~1.14 when it reaches the 5 psi radius
  // (M² = 1 + (γ+1)/(2γ)·Δp/p0), which fixes the ambient sound speed in these units
  const c = (0.4 * r5) / T5 / Math.sqrt(1 + ((6 / 7) * 5) / AMBIENT_PSI)
  const tc = Math.pow((0.4 * k) / c, 1 / 0.6)
  const rc = k * Math.pow(tc, 0.4)
  return {
    gx: city.downtownX * 0.6 + Math.cos(a) * d,
    gz: city.downtownZ * 0.6 + Math.sin(a) * d,
    hb,
    yieldKt,
    r5,
    rf,
    capH: r5 * range(rng, 1.9, 2.2),
    capRmin: 0.42 * r5,
    capRmaj: 0.5 * r5,
    stemR: r5 * range(rng, 0.08, 0.11) * (groundBurst ? 1.3 : 1),
    groundBurst,
    humidity: rng() < 0.75 ? range(rng, 0.4, 1) : 0,
    thermalR: r5 * 1.6,
    k, c, tc, rc,
  }
}

/** Shock radius (slant, from the burst point) at time t after detonation. */
export const shockRadius = (b: Blast, t: number) =>
  t <= 0 ? 0 : t < b.tc ? b.k * Math.pow(t, 0.4) : b.rc + b.c * (t - b.tc)

/** Speed of the shock front at time t. */
export const shockSpeed = (b: Blast, t: number) =>
  t <= 0 ? Infinity : t < b.tc ? 0.4 * b.k * Math.pow(t, -0.6) : b.c

/** Radius of the shock's footprint on the ground (regular reflection, no Mach stem). */
export const groundShockRadius = (b: Blast, t: number) => {
  const R = shockRadius(b, t)
  return R > b.hb ? Math.sqrt(R * R - b.hb * b.hb) : 0
}

/** Arrival time of the shock at a slant distance from the burst point. */
export const shockArrival = (b: Blast, slant: number) =>
  slant <= b.rc ? Math.pow(slant / b.k, 2.5) : b.tc + (slant - b.rc) / b.c

export const slantDistance = (b: Blast, x: number, y: number, z: number) =>
  Math.hypot(x - b.gx, y - b.hb, z - b.gz)

/** Peak overpressure (psi) at a slant distance. */
export const overpressure = (b: Blast, slant: number) =>
  Math.min(PSI_MAX, 5 * Math.pow(b.r5 / Math.max(slant, 1e-3), OVERPRESSURE_EXPONENT))

/** Peak wind speed behind a shock of the given overpressure (Rankine–Hugoniot), units/s. */
export const windSpeed = (psi: number) => {
  const x = psi / AMBIENT_PSI
  return ((SOUND_SPEED_MS * (5 * x)) / 7 / Math.sqrt(1 + (6 * x) / 7)) * WIND_SCALE
}

/** Duration of the positive (outward) wind phase, growing with distance. */
export const positivePhase = (b: Blast, slant: number) => 0.35 + 0.35 * Math.min(slant / b.r5, 4)

/** Friedlander-like wind profile: outward positive phase, then a weak reverse afterwind. */
export const windProfile = (b: Blast, slant: number, tau: number) => {
  if (tau <= 0) return 0
  const tp = positivePhase(b, slant)
  if (tau < tp) return (1 - tau / tp) * Math.exp((-1.2 * tau) / tp)
  return -0.08 * Math.min(1, (tau - tp) / 0.5) * Math.exp(-(tau - tp) / 5)
}

export const damageTier = (psi: number, strength: number) => {
  if (psi >= strength) return TIER_DESTROYED
  if (psi >= strength * 0.6) return TIER_HEAVY
  if (psi >= strength * 0.3) return TIER_MODERATE
  if (psi >= 0.6) return TIER_LIGHT // ordinary window glass fails around half a psi
  return TIER_NONE
}

const PIECE_VOLUME = 170

const desiredPieces = (b: Building) => {
  const vol = b.w * b.d * b.h
  const roof = b.roofH > 0 ? 2 : 0
  switch (b.tier) {
    case TIER_DESTROYED:
      return clamp(Math.round(vol / PIECE_VOLUME), 3, 56) + roof
    case TIER_HEAVY:
      return clamp(Math.round((vol * b.cut) / PIECE_VOLUME), 2, 36) +
        clamp(Math.round((b.h * (1 - b.cut) * Math.max(b.w, b.d)) / 45), 2, 14) + roof
    case TIER_MODERATE:
      return roof > 0 ? roof : b.w * b.d > 300 ? 4 : 2
    default:
      return 0
  }
}

export interface BlastPlan {
  order: number[] // building indices by shock arrival
  desired: number
  planned: number
  tierCounts: number[]
}

/** Decide every building's fate up front so the debris pool is never exceeded. */
export function planBlast(city: City, b: Blast, rng: Rng, pool = DEBRIS_POOL): BlastPlan {
  const tierCounts = [0, 0, 0, 0, 0]
  let desired = 0
  const want: number[] = []
  for (const bd of city.buildings) {
    const dx = bd.x - b.gx
    const dz = bd.z - b.gz
    bd.dist = Math.hypot(dx, dz)
    bd.dirX = bd.dist > 1e-3 ? dx / bd.dist : 1
    bd.dirZ = bd.dist > 1e-3 ? dz / bd.dist : 0
    const slant = Math.hypot(bd.dist, Math.max(0, b.hb - bd.h * 0.5))
    bd.psi = overpressure(b, slant)
    bd.tier = damageTier(bd.psi, bd.strength)
    bd.hitTime = shockArrival(b, slant)
    bd.burn = clamp((b.thermalR / Math.max(slant, 1)) ** 2 - 0.35, 0, 1)
    bd.cut = range(rng, 0.3, 0.65)
    bd.hit = false
    bd.curH = bd.h
    bd.roofOn = bd.roofH > 0
    bd.lean = 0
    bd.sway = 0
    bd.swayPeriod = 0.5 + bd.h / 70
    tierCounts[bd.tier]++
    const n = desiredPieces(bd)
    want.push(n)
    desired += n
  }
  const order = city.buildings.map((_, i) => i).sort((p, q) => city.buildings[p].hitTime - city.buildings[q].hitTime)
  const scale = desired > pool ? pool / desired : 1
  let remaining = pool
  for (const i of order) {
    const n = want[i] > 0 ? Math.min(Math.max(1, Math.floor(want[i] * scale)), remaining) : 0
    city.buildings[i].pieces = n
    remaining -= n
  }
  return { order, desired, planned: pool - remaining, tierCounts }
}

// ---------------------------------------------------------------------------------------------
// Debris: a fixed pool of rigid boxes

export interface Debris {
  capacity: number
  count: number // high-water mark of allocated slots
  px: Float32Array; py: Float32Array; pz: Float32Array
  vx: Float32Array; vy: Float32Array; vz: Float32Array
  qx: Float32Array; qy: Float32Array; qz: Float32Array; qw: Float32Array
  wx: Float32Array; wy: Float32Array; wz: Float32Array
  sx: Float32Array; sy: Float32Array; sz: Float32Array
  size: Float32Array // characteristic size for drag
  cr: Float32Array; cg: Float32Array; cb: Float32Array
  contact: Float32Array // seconds spent touching the ground
  state: Uint8Array // 0 free, 1 flying, 2 resting
}

export const PIECE_FREE = 0
export const PIECE_FLYING = 1
export const PIECE_RESTING = 2

export function createDebris(capacity = DEBRIS_POOL): Debris {
  const f = () => new Float32Array(capacity)
  return {
    capacity, count: 0,
    px: f(), py: f(), pz: f(), vx: f(), vy: f(), vz: f(),
    qx: f(), qy: f(), qz: f(), qw: f(), wx: f(), wy: f(), wz: f(),
    sx: f(), sy: f(), sz: f(), size: f(), cr: f(), cg: f(), cb: f(), contact: f(),
    state: new Uint8Array(capacity),
  }
}

export function resetDebris(d: Debris) {
  d.count = 0
  d.state.fill(0)
}

function addPiece(
  d: Debris, x: number, y: number, z: number, sx: number, sy: number, sz: number, yaw: number,
  vx: number, vy: number, vz: number, spin: number, rng: Rng, c: [number, number, number], shade: number,
) {
  if (d.count >= d.capacity) return false
  const i = d.count++
  d.px[i] = x; d.py[i] = y; d.pz[i] = z
  d.vx[i] = vx; d.vy[i] = vy; d.vz[i] = vz
  d.qx[i] = 0; d.qy[i] = Math.sin(yaw / 2); d.qz[i] = 0; d.qw[i] = Math.cos(yaw / 2)
  // random spin axis, faster for small, fast pieces
  const ax = rng() - 0.5
  const ay = rng() - 0.5
  const az = rng() - 0.5
  const al = Math.hypot(ax, ay, az) || 1
  d.wx[i] = (ax / al) * spin; d.wy[i] = (ay / al) * spin; d.wz[i] = (az / al) * spin
  d.sx[i] = sx; d.sy[i] = sy; d.sz[i] = sz
  d.size[i] = Math.cbrt(sx * sy * sz)
  d.cr[i] = c[0] * shade; d.cg[i] = c[1] * shade; d.cb[i] = c[2] * shade
  d.contact[i] = 0
  d.state[i] = PIECE_FLYING
  return true
}

// Split a box into at most n similar cells
const gridFor = (W: number, H: number, D: number, n: number) => {
  const s = Math.cbrt((W * H * D) / Math.max(1, n))
  const g = [Math.max(1, Math.round(W / s)), Math.max(1, Math.round(H / s)), Math.max(1, Math.round(D / s))]
  while (g[0] * g[1] * g[2] > n) {
    const k = g[0] >= g[1] && g[0] >= g[2] ? 0 : g[1] >= g[2] ? 1 : 2
    g[k]--
  }
  return g
}

/** Launch velocity for a piece at world position (x, y, z). */
function launch(b: Blast, bd: Building, x: number, y: number, z: number, rng: Rng, out: number[]) {
  // Away from the burst point in 3D: under a high air burst the push is mostly downward
  const dx = x - b.gx
  const dy = y - b.hb
  const dz = z - b.gz
  const l = Math.hypot(dx, dy, dz) || 1
  const exposure = 0.6 + 0.4 * clamp(y / Math.max(bd.h, 1), 0, 1) // upper floors catch more wind
  const speed = Math.min(MAX_LAUNCH, windSpeed(bd.psi) * range(rng, 0.3, 0.75) * exposure)
  out[0] = (dx / l) * speed + range(rng, -0.12, 0.12) * speed
  out[1] = (dy / l) * speed + speed * range(rng, 0.08, 0.4)
  out[2] = (dz / l) * speed + range(rng, -0.12, 0.12) * speed
  return speed
}

const tmpV = [0, 0, 0]

function spawnChunks(
  d: Debris, b: Blast, bd: Building, y0: number, y1: number, n: number, angle: number, rng: Rng,
  color: [number, number, number],
) {
  if (n <= 0 || y1 - y0 < 0.05) return 0
  const [nx, ny, nz] = gridFor(bd.w, y1 - y0, bd.d, n)
  const cw = bd.w / nx
  const ch = (y1 - y0) / ny
  const cd = bd.d / nz
  const c = Math.cos(angle)
  const s = Math.sin(angle)
  let made = 0
  for (let ix = 0; ix < nx; ix++) {
    for (let iy = 0; iy < ny; iy++) {
      for (let iz = 0; iz < nz; iz++) {
        const lx = -bd.w / 2 + cw * (ix + 0.5)
        const lz = -bd.d / 2 + cd * (iz + 0.5)
        const x = bd.x + lx * c + lz * s
        const z = bd.z - lx * s + lz * c
        const y = y0 + ch * (iy + 0.5)
        const speed = launch(b, bd, x, y, z, rng, tmpV)
        const sz = Math.min(cw, ch, cd)
        if (addPiece(d, x, y, z, cw * range(rng, 0.7, 0.95), ch * range(rng, 0.7, 0.95), cd * range(rng, 0.7, 0.95),
          angle, tmpV[0], tmpV[1], tmpV[2], (speed / Math.max(sz, 0.5)) * range(rng, 0.15, 0.5), rng, color,
          range(rng, 0.55, 0.95))) made++
      }
    }
  }
  return made
}

// Thin plates peeled off the leeward face (the windward face is pushed in; the back blows out)
function spawnFacade(
  d: Debris, b: Blast, bd: Building, height: number, n: number, angle: number, rng: Rng,
) {
  if (n <= 0 || height < 1) return 0
  const c = Math.cos(angle)
  const s = Math.sin(angle)
  // local axes in world: ax = (c, -s), az = (s, c); pick the face whose normal is most along dir
  const dx = bd.dirX
  const dz = bd.dirZ
  const alongX = Math.abs(dx * c - dz * s) > Math.abs(dx * s + dz * c)
  const sign = alongX ? Math.sign(dx * c - dz * s) || 1 : Math.sign(dx * s + dz * c) || 1
  const faceW = alongX ? bd.d : bd.w
  const half = alongX ? bd.w / 2 : bd.d / 2
  const cell = Math.sqrt((faceW * height) / n)
  let cols = Math.max(1, Math.round(faceW / cell))
  let rows = Math.max(1, Math.round(height / cell))
  while (cols * rows > n) {
    if (cols >= rows) cols--
    else rows--
  }
  const pw = faceW / cols
  const ph = height / rows
  let made = 0
  for (let ci = 0; ci < cols; ci++) {
    for (let ri = 0; ri < rows; ri++) {
      const u = -faceW / 2 + pw * (ci + 0.5)
      const lx = alongX ? sign * (half + 0.4) : u
      const lz = alongX ? u : sign * (half + 0.4)
      const x = bd.x + lx * c + lz * s
      const z = bd.z - lx * s + lz * c
      const y = ph * (ri + 0.5)
      const speed = launch(b, bd, x, y, z, rng, tmpV)
      const sx = alongX ? 0.7 : pw * 0.92
      const sz = alongX ? pw * 0.92 : 0.7
      if (addPiece(d, x, y, z, sx, ph * 0.92, sz, angle, tmpV[0] * 1.15, tmpV[1], tmpV[2] * 1.15,
        (speed / 1.5) * range(rng, 0.2, 0.6), rng, bd.color, range(rng, 0.75, 1))) made++
    }
  }
  return made
}

function spawnRoof(d: Debris, b: Blast, bd: Building, n: number, angle: number, rng: Rng) {
  let made = 0
  for (let k = 0; k < n; k++) {
    const side = n === 1 ? 0 : (k % 2 === 0 ? -1 : 1) * 0.25
    const lx = side * bd.w
    const lz = (Math.floor(k / 2) - (Math.ceil(n / 2) - 1) / 2) * (bd.d / Math.max(1, Math.ceil(n / 2)))
    const c = Math.cos(angle)
    const s = Math.sin(angle)
    const x = bd.x + lx * c + lz * s
    const z = bd.z - lx * s + lz * c
    const y = bd.curH + Math.max(0.3, bd.roofH * 0.5)
    const speed = launch(b, bd, x, y, z, rng, tmpV)
    const pw = bd.roofH > 0 ? bd.w * 0.55 : bd.w / 2
    const pd = bd.roofH > 0 ? bd.d * 1.05 / Math.max(1, Math.ceil(n / 2)) : bd.d / 2
    if (addPiece(d, x, y, z, pw, 0.45, pd, angle, tmpV[0] * 1.2, tmpV[1] + 3, tmpV[2] * 1.2,
      (speed / 1.2) * range(rng, 0.2, 0.6), rng, bd.roofH > 0 ? bd.roofColor : bd.color, range(rng, 0.7, 1))) made++
  }
  return made
}

/** The shock has reached this building: tear it apart according to its planned tier. */
export function applyHit(bd: Building, b: Blast, city: City, d: Debris, rng: Rng) {
  bd.hit = true
  const angle = city.angle
  let budget = bd.pieces
  const before = d.count
  const roofPieces = bd.roofH > 0 && bd.tier >= TIER_MODERATE ? Math.min(2, budget) : 0
  switch (bd.tier) {
    case TIER_DESTROYED: {
      const rubble = Math.min(bd.h, 1.2 + 0.1 * bd.h)
      if (roofPieces > 0) {
        spawnRoof(d, b, bd, roofPieces, angle, rng)
        budget -= roofPieces
      }
      bd.roofOn = false
      spawnChunks(d, b, bd, rubble, bd.h, budget, angle, rng, bd.color)
      bd.curH = rubble
      break
    }
    case TIER_HEAVY: {
      const keep = bd.h * (1 - bd.cut)
      if (roofPieces > 0) {
        spawnRoof(d, b, bd, roofPieces, angle, rng)
        budget -= roofPieces
      }
      bd.roofOn = false
      const facade = Math.min(budget, clamp(Math.round((keep * Math.max(bd.w, bd.d)) / 45), 1, 14))
      spawnChunks(d, b, bd, keep, bd.h, budget - facade, angle, rng, bd.color)
      spawnFacade(d, b, bd, keep, facade, angle, rng)
      bd.curH = keep
      bd.lean = range(rng, 0.03, 0.12)
      bd.sway = range(rng, 0.01, 0.03)
      break
    }
    case TIER_MODERATE: {
      if (bd.roofH > 0) {
        spawnRoof(d, b, bd, roofPieces, angle, rng)
        bd.roofOn = false
      } else if (budget > 0) {
        const t = Math.min(1, 0.15 * bd.h)
        spawnChunks(d, b, bd, bd.h - t, bd.h, budget, angle, rng, bd.color)
        bd.curH = bd.h - t
      }
      bd.lean = range(rng, 0.008, 0.04)
      bd.sway = range(rng, 0.015, 0.035)
      break
    }
    case TIER_LIGHT:
      bd.sway = range(rng, 0.006, 0.02)
      break
  }
  return d.count - before
}

/** Lean angle (rad, away from ground zero) of a building at time t. */
export const buildingLean = (bd: Building, t: number) => {
  if (!bd.hit) return 0
  const tau = Math.max(0, t - bd.hitTime)
  return bd.lean * smoothstep(0, 0.7, tau) +
    bd.sway * Math.exp(-tau / 1.8) * Math.sin((2 * Math.PI * tau) / bd.swayPeriod)
}

/** Advance every flying piece by dt (call with small fixed steps). */
export function stepDebris(d: Debris, b: Blast, t: number, dt: number) {
  for (let i = 0; i < d.count; i++) {
    if (d.state[i] !== PIECE_FLYING) continue
    let px = d.px[i], py = d.py[i], pz = d.pz[i]
    let vx = d.vx[i], vy = d.vy[i], vz = d.vz[i]

    // Blast wind: outward positive phase then a weak reverse afterwind
    const rx = px - b.gx
    const rz = pz - b.gz
    const r = Math.hypot(rx, rz) || 1e-3
    const slant = Math.hypot(r, py - b.hb)
    const tau = t - shockArrival(b, slant)
    let ux = 0
    let uz = 0
    if (tau > 0) {
      const u = windSpeed(overpressure(b, slant)) * windProfile(b, slant, tau)
      ux = (rx / r) * u
      uz = (rz / r) * u
    }
    const relx = ux - vx
    const rely = -vy
    const relz = uz - vz
    const rel = Math.hypot(relx, rely, relz)
    const k = Math.min(0.9, ((DRAG * rel) / Math.max(d.size[i], 0.3)) * dt)
    vx += relx * k
    vy += rely * k
    vz += relz * k
    vy -= GRAVITY * dt
    px += vx * dt
    py += vy * dt
    pz += vz * dt

    // Orientation: q += 0.5 * (w, 0) * q
    let qx = d.qx[i], qy = d.qy[i], qz = d.qz[i], qw = d.qw[i]
    let wx = d.wx[i], wy = d.wy[i], wz = d.wz[i]
    const h = 0.5 * dt
    const nqx = qx + h * (wx * qw + wy * qz - wz * qy)
    const nqy = qy + h * (wy * qw + wz * qx - wx * qz)
    const nqz = qz + h * (wz * qw + wx * qy - wy * qx)
    const nqw = qw - h * (wx * qx + wy * qy + wz * qz)
    const ql = Math.hypot(nqx, nqy, nqz, nqw) || 1
    qx = nqx / ql; qy = nqy / ql; qz = nqz / ql; qw = nqw / ql

    // Ground contact against the box's true vertical half-extent
    const r10 = 2 * (qx * qy + qw * qz)
    const r11 = 1 - 2 * (qx * qx + qz * qz)
    const r12 = 2 * (qy * qz - qw * qx)
    const hy = 0.5 * (Math.abs(r10) * d.sx[i] + Math.abs(r11) * d.sy[i] + Math.abs(r12) * d.sz[i])
    if (py < hy) {
      py = hy
      const impact = vy < 0 ? -vy : 0
      vy = impact * RESTITUTION
      if (vy < 1) vy = 0
      // Coulomb friction: impulse from the impact plus resting normal force
      const hs = Math.hypot(vx, vz)
      if (hs > 1e-6) {
        const nhs = Math.max(0, hs - FRICTION * (impact * (1 + RESTITUTION) + GRAVITY * dt))
        vx *= nhs / hs
        vz *= nhs / hs
      }
      // Gravity's torque about the contact tips the piece onto its nearest face
      let ax: number, ay: number, az: number
      const a0 = Math.abs(r10), a1 = Math.abs(r11), a2 = Math.abs(r12)
      if (a0 >= a1 && a0 >= a2) {
        ax = 1 - 2 * (qy * qy + qz * qz); ay = r10; az = 2 * (qx * qz - qw * qy)
      } else if (a1 >= a2) {
        ax = 2 * (qx * qy - qw * qz); ay = r11; az = 2 * (qy * qz + qw * qx)
      } else {
        ax = 2 * (qx * qz + qw * qy); ay = r12; az = 1 - 2 * (qx * qx + qy * qy)
      }
      const sg = ay < 0 ? -1 : 1
      ax *= sg; az *= sg
      // axis (a × up) = (-az, 0, ax) rotates a toward up
      wx += -az * 30 * dt
      wz += ax * 30 * dt
      const damp = Math.exp(-6 * dt)
      wx *= damp; wy *= damp; wz *= damp
      d.contact[i] += dt
      const v2 = vx * vx + vy * vy + vz * vz
      const w2 = wx * wx + wy * wy + wz * wz
      if ((v2 < 0.25 && w2 < 0.05) || d.contact[i] > 6) {
        vx = vy = vz = 0
        d.state[i] = PIECE_RESTING
      }
    }
    if (r > 9000 || !Number.isFinite(px + py + pz)) d.state[i] = PIECE_FREE

    d.px[i] = px; d.py[i] = py; d.pz[i] = pz
    d.vx[i] = vx; d.vy[i] = vy; d.vz[i] = vz
    d.qx[i] = qx; d.qy[i] = qy; d.qz[i] = qz; d.qw[i] = qw
    d.wx[i] = wx; d.wy[i] = wy; d.wz[i] = wz
  }
}

// ---------------------------------------------------------------------------------------------
// Dust: a ring buffer of soft billows (the newest overwrite the oldest)

export interface Dust {
  capacity: number
  head: number
  px: Float32Array; py: Float32Array; pz: Float32Array
  vx: Float32Array; vy: Float32Array; vz: Float32Array
  age: Float32Array
  life: Float32Array // 0 = dead slot
  size0: Float32Array
  size1: Float32Array
  shade: Float32Array
}

export function createDust(capacity: number): Dust {
  const f = () => new Float32Array(capacity)
  return {
    capacity, head: 0, px: f(), py: f(), pz: f(), vx: f(), vy: f(), vz: f(), age: f(), life: f(),
    size0: f(), size1: f(), shade: f(),
  }
}

export function resetDust(d: Dust) {
  d.head = 0
  d.life.fill(0)
}

export function spawnDust(
  d: Dust, x: number, y: number, z: number, vx: number, vy: number, vz: number, life: number,
  s0: number, s1: number, shade: number,
) {
  const i = d.head
  d.head = (d.head + 1) % d.capacity
  d.px[i] = x; d.py[i] = y; d.pz[i] = z
  d.vx[i] = vx; d.vy[i] = vy; d.vz[i] = vz
  d.age[i] = 0; d.life[i] = life
  d.size0[i] = s0; d.size1[i] = s1; d.shade[i] = shade
}

export function stepDust(d: Dust, b: Blast, cloud: CloudState, t: number, dt: number) {
  const drag = Math.exp(-0.7 * dt)
  const relax = 1 - Math.exp(-0.6 * dt)
  for (let i = 0; i < d.capacity; i++) {
    if (d.life[i] <= 0) continue
    d.age[i] += dt
    if (d.age[i] >= d.life[i]) {
      d.life[i] = 0
      continue
    }
    let vx = d.vx[i] * drag
    let vy = d.vy[i] * drag + 0.6 * dt // warm dust drifts up
    let vz = d.vz[i] * drag
    // Afterwind: once the shock has passed, air rushes back in toward the rising fireball and
    // up the stem, dragging low dust with it
    const rx = d.px[i] - b.gx
    const rz = d.pz[i] - b.gz
    const r = Math.hypot(rx, rz) || 1e-3
    const passed = t - shockArrival(b, Math.hypot(r, d.py[i] - b.hb))
    if (passed > positivePhase(b, r)) {
      const inflow = (0.05 * b.r5 * clamp((passed - 1) / 3, 0, 1)) / (1 + r / b.r5)
      const lift = cloud.stemR > 0 ? Math.exp(-((r / (cloud.stemR * 2.5)) ** 2)) * 0.06 * b.capH : 0
      vx += (-(rx / r) * inflow - vx) * relax * 0.5
      vz += (-(rz / r) * inflow - vz) * relax * 0.5
      vy += (lift - vy) * relax * (lift > 0.5 ? 1 : 0)
    }
    d.vx[i] = vx; d.vy[i] = vy; d.vz[i] = vz
    d.px[i] += vx * dt
    d.py[i] = Math.max(0, d.py[i] + vy * dt)
    d.pz[i] += vz * dt
  }
}

// ---------------------------------------------------------------------------------------------
// Fireball and mushroom cloud

export interface CloudState {
  active: number
  capH: number
  capRmin: number
  capRmaj: number
  squash: number
  temp: number // core temperature, K
  fireball: number // 1 while it is still an incandescent ball
  stemTop: number
  stemR: number
  stemDen: number
  skirtR: number
  skirtH: number
  skirtDen: number
  nox: number // reddish-brown nitrogen dioxide tint of the young cap
  roll: number // accumulated toroidal roll (rad)
  boundR: number
  boundH: number
}

export const createCloudState = (): CloudState => ({
  active: 0, capH: 0, capRmin: 0, capRmaj: 0, squash: 1, temp: 0, fireball: 0, stemTop: 0, stemR: 0,
  stemDen: 0, skirtR: 0, skirtH: 0, skirtDen: 0, nox: 0, roll: 0, boundR: 0, boundH: 0,
})

/** Luminous fireball radius: it grows with the shock, then hangs at its maximum. */
export const fireballRadius = (b: Blast, t: number) => (t <= 0 ? 0 : b.rf * Math.min(1, Math.pow(t / 0.8, 0.4)))

export function cloudState(b: Blast, t: number, out: CloudState) {
  if (t <= 0) {
    Object.assign(out, createCloudState())
    return out
  }
  const rfire = fireballRadius(b, t)
  const morph = 1 - Math.exp(-t / 10) // ball -> rolling torus
  const rise = 1 - Math.exp(-Math.pow(t / 15, 1.7)) // buoyant climb that stalls at the tropopause-ish
  out.active = 1
  out.capH = b.hb + (b.capH - b.hb) * rise
  out.capRmin = rfire + (b.capRmin - rfire) * morph
  out.capRmaj = b.capRmaj * (1 - Math.exp(-Math.max(0, t - 1) / 9))
  out.squash = 1 - 0.32 * morph
  out.temp = 900 + 9000 * Math.exp(-t / 1.1) + 1700 * Math.exp(-t / 9)
  out.fireball = Math.exp(-t / 2.5)
  const stemStart = b.groundBurst ? 0.8 : 1.4 + (0.4 * b.hb) / b.rf
  const capBottom = out.capH - out.capRmin * out.squash * 0.4
  out.stemTop = Math.min(capBottom, Math.max(0, t - stemStart) * (b.capH / 11))
  out.stemR = b.stemR * (0.65 + 0.35 * morph)
  out.stemDen = 0.55 * smoothstep(stemStart, stemStart + 4, t)
  out.skirtR = Math.min(groundShockRadius(b, t) * 0.85, b.r5 * (0.55 + 0.25 * morph))
  out.skirtH = b.r5 * (0.035 + 0.1 * morph)
  out.skirtDen = (b.groundBurst ? 0.5 : 0.35) * smoothstep(0.2, 2.5, t)
  out.nox = smoothstep(1.2, 5, t) * (1 - smoothstep(14, 32, t))
  out.roll = 6 * (1 - Math.exp(-t / 10)) + 0.05 * t
  out.boundR = Math.max(out.capRmaj + out.capRmin * 1.5, out.skirtR * 1.3, out.stemR * 4.5)
  out.boundH = out.capH + out.capRmin * out.squash * 1.6
  return out
}

/** Double-humped thermal pulse: a split-second spike, then a broader second maximum. */
export function thermalPulse(t: number) {
  if (t < 0) return { white: 0, light: 0 }
  const first = Math.exp(-t / 0.05)
  const x = t / 0.4
  const second = x * x * Math.exp(2 * (1 - x))
  // the whiteout lingers a little longer than the light itself: the eye (camera) is saturated
  return { white: Math.min(1, 1.3 * Math.exp(-t / 0.12) + 0.15 * second), light: first + 0.6 * second }
}

/** Normalized blackbody color from Planck's law sampled at R, G, B wavelengths. */
export function blackbody(T: number, out: number[]) {
  const lam = [0.61, 0.55, 0.465] // µm
  let m = 0
  for (let i = 0; i < 3; i++) {
    const l = lam[i]
    out[i] = 1 / (l ** 5 * (Math.exp(14388 / (l * Math.max(T, 300))) - 1))
    m = Math.max(m, out[i])
  }
  for (let i = 0; i < 3; i++) out[i] /= m || 1
  return out
}

// ---------------------------------------------------------------------------------------------
// Camera framing

export interface Frame {
  distance: number
  lift: number // camera offset along its own up axis
  fx: number; fy: number; fz: number // forward
  ux: number; uy: number; uz: number // up
}

/**
 * Smallest camera distance (along its forward axis) and vertical shift such that every point
 * projects inside the frustum. Camera looks toward the center at the given azimuth, pitched
 * down by `pitch`. points is a flat xyz list.
 */
export function fitFrame(
  points: number[], cx: number, cy: number, cz: number, azimuth: number, pitch: number,
  tanV: number, tanH: number, out: Frame,
) {
  const cp = Math.cos(pitch)
  const fx = -cp * Math.cos(azimuth), fy = -Math.sin(pitch), fz = -cp * Math.sin(azimuth)
  // right = f × up(0,1,0) normalized; up' = right × f
  let rx = -fz, rz = fx
  const rl = Math.hypot(rx, rz) || 1
  rx /= rl; rz /= rl
  const ux = -rz * fy, uy = rz * fx - rx * fz, uz = rx * fy
  const n = points.length / 3
  const X: number[] = [], Y: number[] = [], Z: number[] = []
  for (let i = 0; i < n; i++) {
    const px = points[3 * i] - cx, py = points[3 * i + 1] - cy, pz = points[3 * i + 2] - cz
    X.push(px * rx + pz * rz)
    Y.push(px * ux + py * uy + pz * uz)
    Z.push(px * fx + py * fy + pz * fz)
  }
  const feasible = (D: number) => {
    let lo = -Infinity, hi = Infinity
    for (let i = 0; i < n; i++) {
      const depth = Z[i] + D
      if (depth < 10) return null
      if (Math.abs(X[i]) > tanH * depth) return null
      lo = Math.max(lo, Y[i] - tanV * depth)
      hi = Math.min(hi, Y[i] + tanV * depth)
    }
    return lo <= hi ? [lo, hi] : null
  }
  let a = 10, bnd = 100
  while (!feasible(bnd) && bnd < 1e6) bnd *= 2
  for (let it = 0; it < 40; it++) {
    const m = (a + bnd) / 2
    if (feasible(m)) bnd = m
    else a = m
  }
  const iv = feasible(bnd) ?? [0, 0]
  out.distance = bnd
  out.lift = (iv[0] + iv[1]) / 2
  out.fx = fx; out.fy = fy; out.fz = fz
  out.ux = ux; out.uy = uy; out.uz = uz
  return out
}
