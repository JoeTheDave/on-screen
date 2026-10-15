// Wave field and sphere physics for WaveGrid, kept free of three.js so it can be exercised
// headlessly. Everything is plain arrays and numbers; WaveGrid only reads the results.
//
// The field is a pure function of (x, z, t) given the current pool of wave sources. Sources
// are only ever added with zero amplitude (fading in from "now") and removed long after they
// have faded out, so changing the pool never changes the surface: it morphs, it never pops.
// That purity also lets the physics take the surface's vertical velocity as a finite
// difference of the field in time.

// Lattice: a square rows/columns grid, keeping only the cubes inside a disc
export const GRID_SIZE = 50 // lattice cells across the disc's diameter
export const CUBE_SIZE = 1
export const SPACING = 1.2 // center-to-center distance between cubes
export const DISC_RADIUS = ((GRID_SIZE - 1) * SPACING) / 2 + SPACING / 2 // ~30, the old half-width

// Wave field
export const HEIGHT_LIMIT = 5 // soft ceiling on |height| of a cube center (tanh limiter)
const WAVE_SCALE = 3.1 // two equal waves peak near 4.4, like the original pair of 2.2
const MIN_WAVELENGTH = 14
const MAX_WAVELENGTH = 30
const MIN_WAVE_SPEED = 4 // units per second
const MAX_WAVE_SPEED = 9
const RADIAL_CHANCE = 0.5 // fraction of new sources that are circular
const RADIAL_CENTER_REACH = 1.3 // radial centers land within this many disc radii
const RADIAL_SOFTENING = 1 // smooths the cusp at a radial wave's center
const MIN_FALLOFF_LENGTH = 30 // radial falloff e-folding distance, when it has one
const MAX_FALLOFF_LENGTH = 60
export const WAVE_FADE_SECONDS = 6 // fade in / fade out duration for a source
export const MIN_ACTIVE_WAVES = 2
export const MAX_ACTIVE_WAVES = 4
export const MIN_WAVE_EVENT_SECONDS = 8
export const MAX_WAVE_EVENT_SECONDS = 20

// Sphere physics
export const SPHERE_COUNT = 20
const MIN_SPHERE_RADIUS = 0.4 // 0.8 - 1.2 cube widths across
const MAX_SPHERE_RADIUS = 0.6
const GRAVITY = 22
export const PHYSICS_STEP = 1 / 120
const MAX_SPEED = 30 // keeps per-step travel well under a sphere radius
const RESTITUTION = 0.45
const BOUNCE_THRESHOLD = 1.2 // normal impact speeds below this don't bounce (no resting jitter)
const FRICTION = 0.3 // Coulomb coefficient against the cube tops
const ROLLING_DAMPING = 3 // per second, tangential speed lost in contact; lower and they race off the disc
const SPHERE_RESTITUTION = 0.7
const FALL_LIMIT = -20 // below this a sphere has left the disc for good: respawn
const SPAWN_REACH = 0.8 // respawn within this many disc radii of the center
const MIN_SPAWN_HEIGHT = 12
const MAX_SPAWN_HEIGHT = 22

type Random = () => number

const between = (random: Random, min: number, max: number) => min + random() * (max - min)

const smoothstep = (x: number) => (x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x))

// ---------------------------------------------------------------------------------------------
// Lattice

export interface Lattice {
  count: number
  baseX: Float32Array
  baseZ: Float32Array
  half: number
  index: Int32Array // row * GRID_SIZE + col -> instance index, or -1 outside the disc
}

export function createLattice(): Lattice {
  const half = ((GRID_SIZE - 1) * SPACING) / 2
  const index = new Int32Array(GRID_SIZE * GRID_SIZE).fill(-1)
  const xs: number[] = []
  const zs: number[] = []
  for (let row = 0; row < GRID_SIZE; row++) {
    for (let col = 0; col < GRID_SIZE; col++) {
      const x = col * SPACING - half
      const z = row * SPACING - half
      if (Math.hypot(x, z) > DISC_RADIUS) continue
      index[row * GRID_SIZE + col] = xs.length
      xs.push(x)
      zs.push(z)
    }
  }
  return { count: xs.length, baseX: Float32Array.from(xs), baseZ: Float32Array.from(zs), half, index }
}

// ---------------------------------------------------------------------------------------------
// Wave field

export interface WaveSource {
  radial: boolean
  // Plane: travel direction. Radial: center.
  ax: number
  az: number
  k: number // wavenumber, 2π / wavelength
  omega: number // angular frequency; for radial waves negative means traveling inward
  phase: number
  invFalloff: number // radial only; 0 = no falloff
  weight: number // relative strength
  bornAt: number
  retireAt: number // Infinity until retired
}

export interface WaveField {
  sources: WaveSource[]
  nextEventAt: number
}

export function createSource(random: Random, t: number, fadeIn: boolean): WaveSource {
  const k = (Math.PI * 2) / between(random, MIN_WAVELENGTH, MAX_WAVELENGTH)
  const omega = k * between(random, MIN_WAVE_SPEED, MAX_WAVE_SPEED)
  const radial = random() < RADIAL_CHANCE
  let ax: number
  let az: number
  if (radial) {
    // Uniform over a disc slightly larger than the cube disc, so some ripple in from offstage
    const r = DISC_RADIUS * RADIAL_CENTER_REACH * Math.sqrt(random())
    const a = random() * Math.PI * 2
    ax = r * Math.cos(a)
    az = r * Math.sin(a)
  } else {
    const a = random() * Math.PI * 2
    ax = Math.cos(a)
    az = Math.sin(a)
  }
  return {
    radial,
    ax,
    az,
    k,
    omega: radial && random() < 0.5 ? -omega : omega,
    phase: random() * Math.PI * 2,
    invFalloff: radial && random() < 0.5 ? 1 / between(random, MIN_FALLOFF_LENGTH, MAX_FALLOFF_LENGTH) : 0,
    weight: between(random, 0.75, 1),
    // Initial sources start at full strength; later ones fade in from now
    bornAt: fadeIn ? t : -Infinity,
    retireAt: Infinity,
  }
}

export function createField(random: Random = Math.random, t = 0): WaveField {
  const sources: WaveSource[] = []
  const initial = MIN_ACTIVE_WAVES + Math.floor(random() * 2)
  for (let i = 0; i < initial; i++) sources.push(createSource(random, t, false))
  return { sources, nextEventAt: t + between(random, MIN_WAVE_EVENT_SECONDS, MAX_WAVE_EVENT_SECONDS) }
}

const envelope = (s: WaveSource, t: number) =>
  smoothstep((t - s.bornAt) / WAVE_FADE_SECONDS) * (1 - smoothstep((t - s.retireAt) / WAVE_FADE_SECONDS))

// Cube-center height at (x, z) and time t
export function sampleField(field: WaveField, x: number, z: number, t: number): number {
  let raw = 0
  let power = 0
  for (const s of field.sources) {
    const a = envelope(s, t) * s.weight
    if (a === 0) continue
    power += a * a
    if (s.radial) {
      const dx = x - s.ax
      const dz = z - s.az
      const r = Math.sqrt(dx * dx + dz * dz + RADIAL_SOFTENING)
      const falloff = s.invFalloff === 0 ? 1 : Math.exp(-r * s.invFalloff)
      // Positive omega: crests move outward (sin(kr - ωt)); negative: inward
      raw += a * falloff * Math.sin(s.k * r - s.omega * t + s.phase)
    } else {
      raw += a * Math.sin(s.k * (s.ax * x + s.az * z) - s.omega * t + s.phase)
    }
  }
  // Normalize by the RMS strength so 2 or 4 sources look equally lively, then soft-limit so
  // rare constructive pileups round off instead of spiking past the camera framing
  const h = (WAVE_SCALE * raw) / Math.sqrt(Math.max(power, 1))
  return HEIGHT_LIMIT * Math.tanh(h / HEIGHT_LIMIT)
}

// Retire/introduce sources on schedule and drop fully faded ones. Call once per frame.
export function updateField(field: WaveField, t: number, random: Random = Math.random) {
  // Keep a fully-faded source a little past its fade so finite differences at t - dt still see it
  field.sources = field.sources.filter(s => t < s.retireAt + WAVE_FADE_SECONDS + 1)
  if (t < field.nextEventAt) return
  field.nextEventAt = t + between(random, MIN_WAVE_EVENT_SECONDS, MAX_WAVE_EVENT_SECONDS)

  const live = field.sources.filter(s => s.retireAt === Infinity)
  // Mostly swap one for one; sometimes just add or just retire so the count drifts in range
  const roll = random()
  const retire = live.length >= MAX_ACTIVE_WAVES || (live.length > MIN_ACTIVE_WAVES && roll < 0.85)
  const add = live.length <= MIN_ACTIVE_WAVES || roll < 0.7 || roll >= 0.85
  if (retire) {
    // Retire the oldest, so no source lingers for the whole session
    let oldest = live[0]
    for (const s of live) if (s.bornAt < oldest.bornAt) oldest = s
    oldest.retireAt = t
  }
  if (add) field.sources.push(createSource(random, t, true))
}

// ---------------------------------------------------------------------------------------------
// Spheres

export interface Spheres {
  count: number
  radius: Float64Array
  x: Float64Array
  y: Float64Array
  z: Float64Array
  vx: Float64Array
  vy: Float64Array
  vz: Float64Array
  respawns: number // running total, for diagnostics
}

export function respawnSphere(s: Spheres, i: number, random: Random = Math.random, height?: number) {
  // A few tries to avoid appearing inside another sphere; overlap would be pushed apart anyway
  const reach = DISC_RADIUS * SPAWN_REACH
  for (let attempt = 0; attempt < 8; attempt++) {
    const r = reach * Math.sqrt(random())
    const a = random() * Math.PI * 2
    s.x[i] = r * Math.cos(a)
    s.z[i] = r * Math.sin(a)
    s.y[i] = height ?? between(random, MIN_SPAWN_HEIGHT, MAX_SPAWN_HEIGHT)
    let clear = true
    for (let j = 0; j < s.count && clear; j++) {
      if (j === i) continue
      const gap = s.radius[i] + s.radius[j] + 0.2
      const dx = s.x[j] - s.x[i]
      const dy = s.y[j] - s.y[i]
      const dz = s.z[j] - s.z[i]
      if (dx * dx + dy * dy + dz * dz < gap * gap) clear = false
    }
    if (clear) break
  }
  s.vx[i] = 0
  s.vy[i] = 0
  s.vz[i] = 0
}

export function createSpheres(random: Random = Math.random, count = SPHERE_COUNT): Spheres {
  const s: Spheres = {
    count,
    radius: new Float64Array(count),
    x: new Float64Array(count),
    y: new Float64Array(count),
    z: new Float64Array(count),
    vx: new Float64Array(count),
    vy: new Float64Array(count),
    vz: new Float64Array(count),
    respawns: 0,
  }
  for (let i = 0; i < count; i++) {
    s.radius[i] = between(random, MIN_SPHERE_RADIUS, MAX_SPHERE_RADIUS)
    // Stagger the opening drop so they land one by one rather than as a single sheet
    s.y[i] = Infinity
  }
  for (let i = 0; i < count; i++) respawnSphere(s, i, random, between(random, 8, 30))
  return s
}

const HALF_CUBE = CUBE_SIZE / 2

// Range of lattice columns (or rows) whose cubes can reach within r of coordinate c
const cellRange = (c: number, r: number, half: number): [number, number] => [
  Math.max(0, Math.ceil((c - r - HALF_CUBE + half) / SPACING)),
  Math.min(GRID_SIZE - 1, Math.floor((c + r + HALF_CUBE + half) / SPACING)),
]

// Resolve one sphere against the cube columns under/around it. Cubes are treated as columns
// with nothing below their top face mattering, except their sides when the sphere's center is
// below a neighbor's top (it is then pushed sideways, not teleported up a wall).
function collideGround(
  s: Spheres,
  i: number,
  lattice: Lattice,
  field: WaveField,
  t: number,
  dt: number,
) {
  const r = s.radius[i]
  const [c0, c1] = cellRange(s.x[i], r, lattice.half)
  const [r0, r1] = cellRange(s.z[i], r, lattice.half)

  let support = -Infinity // center height needed to clear every cube top in the footprint
  let supportCube = -1
  let supportTop = 0
  let supportPx = 0
  let supportPz = 0

  for (let row = r0; row <= r1; row++) {
    for (let col = c0; col <= c1; col++) {
      const idx = lattice.index[row * GRID_SIZE + col]
      if (idx < 0) continue
      const cx = lattice.baseX[idx]
      const cz = lattice.baseZ[idx]
      // Closest point of the cube's top square to the sphere center, horizontally
      const px = Math.min(Math.max(s.x[i], cx - HALF_CUBE), cx + HALF_CUBE)
      const pz = Math.min(Math.max(s.z[i], cz - HALF_CUBE), cz + HALF_CUBE)
      const dx = s.x[i] - px
      const dz = s.z[i] - pz
      const d2 = dx * dx + dz * dz
      if (d2 >= r * r) continue
      const top = sampleField(field, cx, cz, t) + HALF_CUBE

      if (s.y[i] < top && d2 > 1e-8) {
        // Beside a taller column: push out horizontally, bounce off its face
        const d = Math.sqrt(d2)
        const nx = dx / d
        const nz = dz / d
        s.x[i] += nx * (r - d)
        s.z[i] += nz * (r - d)
        const vn = s.vx[i] * nx + s.vz[i] * nz
        if (vn < 0) {
          const e = -vn > BOUNCE_THRESHOLD ? RESTITUTION : 0
          s.vx[i] -= (1 + e) * vn * nx
          s.vz[i] -= (1 + e) * vn * nz
        }
        continue
      }
      // Resting on the top face (d = 0) or its edge/corner (0 < d < r): a rounded footprint,
      // so sliding over an edge raises/lowers the sphere continuously
      const need = top + Math.sqrt(r * r - d2)
      if (need > support) {
        support = need
        supportCube = idx
        supportTop = top
        supportPx = px
        supportPz = pz
      }
    }
  }

  if (supportCube < 0 || s.y[i] >= support) return

  s.y[i] = support

  // Normal: the actual contact direction (straight up on a face, tilted on an edge) blended
  // with the wave surface's slope estimated from heights one cube away, so spheres roll down
  // the waves rather than sitting on individual flat tops
  const e = SPACING
  const gx = (sampleField(field, s.x[i] + e, s.z[i], t) - sampleField(field, s.x[i] - e, s.z[i], t)) / (2 * e)
  const gz = (sampleField(field, s.x[i], s.z[i] + e, t) - sampleField(field, s.x[i], s.z[i] - e, t)) / (2 * e)
  const fieldLen = Math.sqrt(gx * gx + 1 + gz * gz)
  let nx = (s.x[i] - supportPx) / r - gx / fieldLen
  let ny = (support - supportTop) / r + 1 / fieldLen
  let nz = (s.z[i] - supportPz) / r - gz / fieldLen
  const nLen = Math.sqrt(nx * nx + ny * ny + nz * nz)
  nx /= nLen
  ny /= nLen
  nz /= nLen

  // The supporting cube's vertical velocity: rising waves carry (and launch) the sphere
  const cx = lattice.baseX[supportCube]
  const cz = lattice.baseZ[supportCube]
  const surfaceVy = (sampleField(field, cx, cz, t) - sampleField(field, cx, cz, t - dt)) / dt

  const rvx = s.vx[i]
  const rvy = s.vy[i] - surfaceVy
  const rvz = s.vz[i]
  const vn = rvx * nx + rvy * ny + rvz * nz
  if (vn >= 0) return // already separating

  const bounce = -vn > BOUNCE_THRESHOLD ? RESTITUTION : 0
  let tx = rvx - vn * nx
  let ty = rvy - vn * ny
  let tz = rvz - vn * nz
  const tLen = Math.sqrt(tx * tx + ty * ty + tz * tz)
  if (tLen > 1e-9) {
    // Coulomb friction from the normal impulse, plus a little rolling resistance
    const keep = Math.max(0, 1 - (FRICTION * -vn) / tLen) * (1 - ROLLING_DAMPING * dt)
    tx *= keep
    ty *= keep
    tz *= keep
  }
  s.vx[i] = tx - bounce * vn * nx
  s.vy[i] = ty - bounce * vn * ny + surfaceVy
  s.vz[i] = tz - bounce * vn * nz
}

function collideSpheres(s: Spheres) {
  for (let i = 0; i < s.count; i++) {
    for (let j = i + 1; j < s.count; j++) {
      const reach = s.radius[i] + s.radius[j]
      let dx = s.x[j] - s.x[i]
      let dy = s.y[j] - s.y[i]
      let dz = s.z[j] - s.z[i]
      const d2 = dx * dx + dy * dy + dz * dz
      if (d2 >= reach * reach) continue
      const d = Math.sqrt(d2)
      if (d > 1e-9) {
        dx /= d
        dy /= d
        dz /= d
      } else {
        dx = 0
        dy = 1
        dz = 0
      }
      // Mass ~ volume; split the push-apart and the impulse accordingly
      const mi = s.radius[i] ** 3
      const mj = s.radius[j] ** 3
      const pen = reach - d
      const wi = mj / (mi + mj)
      const wj = mi / (mi + mj)
      s.x[i] -= dx * pen * wi
      s.y[i] -= dy * pen * wi
      s.z[i] -= dz * pen * wi
      s.x[j] += dx * pen * wj
      s.y[j] += dy * pen * wj
      s.z[j] += dz * pen * wj
      const vrel = (s.vx[j] - s.vx[i]) * dx + (s.vy[j] - s.vy[i]) * dy + (s.vz[j] - s.vz[i]) * dz
      if (vrel >= 0) continue
      const impulse = (-(1 + SPHERE_RESTITUTION) * vrel) / (1 / mi + 1 / mj)
      s.vx[i] -= (impulse / mi) * dx
      s.vy[i] -= (impulse / mi) * dy
      s.vz[i] -= (impulse / mi) * dz
      s.vx[j] += (impulse / mj) * dx
      s.vy[j] += (impulse / mj) * dy
      s.vz[j] += (impulse / mj) * dz
    }
  }
}

// One fixed physics step ending at field time t
export function stepSpheres(
  s: Spheres,
  lattice: Lattice,
  field: WaveField,
  t: number,
  dt: number = PHYSICS_STEP,
  random: Random = Math.random,
) {
  for (let i = 0; i < s.count; i++) {
    s.vy[i] -= GRAVITY * dt
    const speed = Math.hypot(s.vx[i], s.vy[i], s.vz[i])
    if (speed > MAX_SPEED) {
      const k = MAX_SPEED / speed
      s.vx[i] *= k
      s.vy[i] *= k
      s.vz[i] *= k
    }
    s.x[i] += s.vx[i] * dt
    s.y[i] += s.vy[i] * dt
    s.z[i] += s.vz[i] * dt
  }

  collideSpheres(s)

  for (let i = 0; i < s.count; i++) {
    // Two passes: a sideways push off one column can land the sphere on another's edge
    collideGround(s, i, lattice, field, t, dt)
    collideGround(s, i, lattice, field, t, dt)

    const finite =
      Number.isFinite(s.x[i]) &&
      Number.isFinite(s.y[i]) &&
      Number.isFinite(s.z[i]) &&
      Number.isFinite(s.vx[i]) &&
      Number.isFinite(s.vy[i]) &&
      Number.isFinite(s.vz[i])
    if (!finite || s.y[i] < FALL_LIMIT) {
      respawnSphere(s, i, random)
      s.respawns++
    }
  }
}

// Deepest overlap of sphere i with any cube (as a solid column under its top), for diagnostics
export function groundPenetration(s: Spheres, i: number, lattice: Lattice, field: WaveField, t: number): number {
  const r = s.radius[i]
  const [c0, c1] = cellRange(s.x[i], r, lattice.half)
  const [r0, r1] = cellRange(s.z[i], r, lattice.half)
  let worst = 0
  for (let row = r0; row <= r1; row++) {
    for (let col = c0; col <= c1; col++) {
      const idx = lattice.index[row * GRID_SIZE + col]
      if (idx < 0) continue
      const cx = lattice.baseX[idx]
      const cz = lattice.baseZ[idx]
      const top = sampleField(field, cx, cz, t) + HALF_CUBE
      const px = Math.min(Math.max(s.x[i], cx - HALF_CUBE), cx + HALF_CUBE)
      const pz = Math.min(Math.max(s.z[i], cz - HALF_CUBE), cz + HALF_CUBE)
      const py = Math.min(s.y[i], top)
      const dx = s.x[i] - px
      const dy = s.y[i] - py
      const dz = s.z[i] - pz
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz)
      // Center inside the column: depth to the nearest way out (up) plus the radius
      const pen = d < 1e-9 ? r + top - s.y[i] : r - d
      worst = Math.max(worst, pen)
    }
  }
  return worst
}
