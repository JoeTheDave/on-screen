// Restricted N-body galaxy encounter (Toomre & Toomre 1972), pure math, no Three.js, so it
// can be sanity-checked in node.
//
// Two massive cores attract each other through a softened (Plummer) potential; every star is
// massless and only feels the two cores. That's what makes tens of thousands of stars cheap:
// O(N) per step instead of O(N^2). Toomre's cores fly past each other forever, so to get a
// merger we add a dynamical-friction drag on the cores' relative velocity that switches on as
// their dark halos overlap; it's an internal force, so the centre of mass never moves.
//
// Units: G = 1, the larger galaxy has mass 1 and a disk radius of ~1.4.

export type Rand = () => number
export type Vec3 = [number, number, number]

export interface GalaxySpec {
  mass: number
  eps: number // core softening length: makes the core a Plummer sphere, not a point
  radius: number // disk outer radius
  normal: Vec3 // disk spin axis
  e1: Vec3 // disk plane basis; stars orbit from e1 towards e2
  e2: Vec3
}

export interface Encounter {
  galaxies: [GalaxySpec, GalaxySpec]
  // Per core: x y z vx vy vz ax ay az
  cores: Float64Array
  coreCoreEps: number
  innerScale: number // the stiff final-sinking friction acts inside roughly this separation
  haloScale: number // friction switches on inside roughly this separation
  friction: number
  // Per core: the friction-only part of its acceleration (x y z). Stars near a core get the
  // same drag, otherwise a decelerating satellite would leave its own disk behind
  drag: Float64Array
  pericenterTime: number // sim time of the first close passage
  mergeTime: number // sim time after which the cores stay together
}

export const CORE_STRIDE = 9
const MERGE_SEPARATION = 0.25
const INNER_FRICTION = 3
const PRECOMPUTE_DT = 0.01
const PRECOMPUTE_MAX_TIME = 160

// ---- small vector helpers ----
const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
]
const normalize = (a: Vec3): Vec3 => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1
  return [a[0] / l, a[1] / l, a[2] / l]
}

// Rodrigues rotation of v about unit axis k by angle t
const rotate = (v: Vec3, k: Vec3, t: number): Vec3 => {
  const c = Math.cos(t)
  const s = Math.sin(t)
  const kxv = cross(k, v)
  const kdv = k[0] * v[0] + k[1] * v[1] + k[2] * v[2]
  return [
    v[0] * c + kxv[0] * s + k[0] * kdv * (1 - c),
    v[1] * c + kxv[1] * s + k[1] * kdv * (1 - c),
    v[2] * c + kxv[2] * s + k[2] * kdv * (1 - c),
  ]
}

const gaussian = (rand: Rand) => {
  const u = Math.max(rand(), 1e-12)
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand())
}

/** Disk tilted away from the orbital angular momentum `orbitAxis` by `inclination`. */
function makeDisk(mass: number, orbitAxis: Vec3, inclination: number, rand: Rand): GalaxySpec {
  // Tilt about a random horizontal axis (perpendicular to the orbit axis)
  const phi = rand() * Math.PI * 2
  const tiltAxis: Vec3 = [Math.cos(phi), 0, Math.sin(phi)]
  const normal = normalize(rotate(orbitAxis, tiltAxis, inclination))
  const helper: Vec3 = Math.abs(normal[0]) < 0.9 ? [1, 0, 0] : [0, 0, 1]
  const e1 = normalize(cross(helper, normal))
  const e2 = cross(normal, e1)
  return {
    mass,
    eps: 0.22 * Math.sqrt(mass),
    radius: 1.4 * Math.sqrt(mass),
    normal,
    e1,
    e2,
  }
}

/** Fill the cores' accelerations from their current positions and velocities. */
export function coreAccelerations(enc: Encounter) {
  const c = enc.cores
  const m1 = enc.galaxies[0].mass
  const m2 = enc.galaxies[1].mass
  const o = CORE_STRIDE
  const dx = c[o] - c[0]
  const dy = c[o + 1] - c[1]
  const dz = c[o + 2] - c[2]
  const r2 = dx * dx + dy * dy + dz * dz
  const s2 = r2 + enc.coreCoreEps * enc.coreCoreEps
  const inv3 = 1 / (s2 * Math.sqrt(s2))

  // Dynamical friction on the relative velocity, weighted by halo overlap (1 at contact,
  // ~ (h/r)^3 far away so the approach is untouched)
  const h2 = enc.haloScale * enc.haloScale
  const overlap = Math.pow(h2 / (r2 + h2), 1.5)
  // Plus a stiff term once the cores are inside each other: without it they'd slosh through
  // one another for many fast orbits, and that whirling binary slingshots stars out of the
  // remnant (a restricted-N-body artifact; real cores are extended and sink quickly)
  const q2 = enc.innerScale * enc.innerScale
  const inner = q2 / (r2 + q2)
  const k = enc.friction * overlap + INNER_FRICTION * inner * inner
  const fx = -k * (c[o + 3] - c[3])
  const fy = -k * (c[o + 4] - c[4])
  const fz = -k * (c[o + 5] - c[5])
  const m = m1 + m2

  const d = enc.drag
  d[0] = -(m2 / m) * fx
  d[1] = -(m2 / m) * fy
  d[2] = -(m2 / m) * fz
  d[3] = (m1 / m) * fx
  d[4] = (m1 / m) * fy
  d[5] = (m1 / m) * fz
  c[6] = m2 * dx * inv3 + d[0]
  c[7] = m2 * dy * inv3 + d[1]
  c[8] = m2 * dz * inv3 + d[2]
  c[o + 6] = -m1 * dx * inv3 + d[3]
  c[o + 7] = -m1 * dy * inv3 + d[4]
  c[o + 8] = -m1 * dz * inv3 + d[5]
}

/** Accelerations of massless particles [start, end) from both softened cores. */
export function particleAccelerations(
  enc: Encounter,
  pos: Float32Array,
  acc: Float32Array,
  start: number,
  end: number,
) {
  const c = enc.cores
  const o = CORE_STRIDE
  const [g1, g2] = enc.galaxies
  const x1 = c[0], y1 = c[1], z1 = c[2]
  const x2 = c[o], y2 = c[o + 1], z2 = c[o + 2]
  const m1 = g1.mass, m2 = g2.mass
  const e1 = g1.eps * g1.eps, e2 = g2.eps * g2.eps
  // Drag shares fall off over each galaxy's disk radius
  const r1 = g1.radius * g1.radius, r2 = g2.radius * g2.radius
  const d = enc.drag
  for (let i = start * 3; i < end * 3; i += 3) {
    const px = pos[i], py = pos[i + 1], pz = pos[i + 2]
    let dx = x1 - px, dy = y1 - py, dz = z1 - pz
    let s2 = dx * dx + dy * dy + dz * dz + e1
    let f = m1 / (s2 * Math.sqrt(s2))
    let w = r1 / (s2 - e1 + r1)
    w *= w
    let ax = dx * f + d[0] * w, ay = dy * f + d[1] * w, az = dz * f + d[2] * w
    dx = x2 - px; dy = y2 - py; dz = z2 - pz
    s2 = dx * dx + dy * dy + dz * dz + e2
    f = m2 / (s2 * Math.sqrt(s2))
    w = r2 / (s2 - e2 + r2)
    w *= w
    ax += dx * f + d[3] * w; ay += dy * f + d[4] * w; az += dz * f + d[5] * w
    acc[i] = ax
    acc[i + 1] = ay
    acc[i + 2] = az
  }
}

/**
 * One kick-drift-kick leapfrog step of cores and particles [start, end). Requires `acc` and
 * the cores' accelerations to be current on entry (see `prime`); leaves them current on exit.
 */
export function step(
  enc: Encounter,
  pos: Float32Array,
  vel: Float32Array,
  acc: Float32Array,
  start: number,
  end: number,
  dt: number,
) {
  const c = enc.cores
  const h = dt * 0.5
  for (let k = 0; k < 2; k++) {
    const o = k * CORE_STRIDE
    c[o + 3] += c[o + 6] * h; c[o + 4] += c[o + 7] * h; c[o + 5] += c[o + 8] * h
    c[o] += c[o + 3] * dt; c[o + 1] += c[o + 4] * dt; c[o + 2] += c[o + 5] * dt
  }
  for (let i = start * 3; i < end * 3; i++) {
    vel[i] += acc[i] * h
    pos[i] += vel[i] * dt
  }
  coreAccelerations(enc)
  particleAccelerations(enc, pos, acc, start, end)
  for (let k = 0; k < 2; k++) {
    const o = k * CORE_STRIDE
    c[o + 3] += c[o + 6] * h; c[o + 4] += c[o + 7] * h; c[o + 5] += c[o + 8] * h
  }
  for (let i = start * 3; i < end * 3; i++) vel[i] += acc[i] * h
}

/** Compute the starting accelerations so the first `step` is a proper leapfrog step. */
export function prime(enc: Encounter, pos: Float32Array, acc: Float32Array, start: number, end: number) {
  coreAccelerations(enc)
  particleAccelerations(enc, pos, acc, start, end)
}

export const coreSeparation = (c: Float64Array) =>
  Math.hypot(c[CORE_STRIDE] - c[0], c[CORE_STRIDE + 1] - c[1], c[CORE_STRIDE + 2] - c[2])

/** Total core energy (kinetic + softened potential); friction should only ever lower it. */
export function coreEnergy(enc: Encounter) {
  const c = enc.cores
  const o = CORE_STRIDE
  const [g1, g2] = enc.galaxies
  const ke =
    0.5 * g1.mass * (c[3] ** 2 + c[4] ** 2 + c[5] ** 2) +
    0.5 * g2.mass * (c[o + 3] ** 2 + c[o + 4] ** 2 + c[o + 5] ** 2)
  const r = coreSeparation(c)
  return ke - (g1.mass * g2.mass) / Math.sqrt(r * r + enc.coreCoreEps * enc.coreCoreEps)
}

/** Fly the two cores alone (stars don't affect them) to find pericentre and merger times. */
function precomputeTimes(enc: Encounter) {
  const saved = enc.cores.slice()
  const empty = new Float32Array(0)
  prime(enc, empty, empty, 0, 0)
  let t = 0
  let prev = coreSeparation(enc.cores)
  let falling = true
  let pericenterTime = -1
  let lastApart = 0
  while (t < PRECOMPUTE_MAX_TIME) {
    step(enc, empty, empty, empty, 0, 0, PRECOMPUTE_DT)
    t += PRECOMPUTE_DT
    const sep = coreSeparation(enc.cores)
    if (falling && sep > prev && pericenterTime < 0) pericenterTime = t
    falling = sep <= prev
    if (sep > MERGE_SEPARATION) lastApart = t
    prev = sep
  }
  enc.cores.set(saved)
  enc.pericenterTime = pericenterTime
  enc.mergeTime = lastApart
}

/**
 * A fresh encounter: random mass ratio, disk tilts, pericentre and orbit energy. The relative
 * orbit lies in the horizontal (xz) plane, centre of mass at the origin, at rest. Rerolls
 * until the cores actually merge within a sensible time window.
 */
export function createEncounter(
  rand: Rand,
  mergeWindow: [number, number] = [28, 70],
): Encounter {
  let enc: Encounter | null = null
  for (let attempt = 0; attempt < 30; attempt++) {
    const minor = 0.35 + 0.65 * rand()
    const masses: [number, number] = rand() < 0.5 ? [1, minor] : [minor, 1]
    const sense = rand() < 0.5 ? 1 : -1
    const orbitAxis: Vec3 = [0, sense, 0]
    // Mostly prograde, gently tilted disks (those grow the long tails); sometimes anything
    const tilt = () => (rand() < 0.7 ? rand() * 0.9 : rand() * Math.PI)
    const g1 = makeDisk(masses[0], orbitAxis, tilt(), rand)
    const g2 = makeDisk(masses[1], orbitAxis, tilt(), rand)
    const mu = g1.mass + g2.mass

    const rp = (0.8 + 0.8 * rand()) * Math.max(g1.radius, g2.radius)
    const ecc = 0.75 + 0.25 * rand()
    const d0 = 6 + rand() * 1.5
    // Vis-viva for the (unsoftened, fine at this range) Kepler orbit with pericentre rp
    const invA = (1 - ecc) / rp
    const v2 = mu * (2 / d0 - invA)
    const vt = Math.sqrt(mu * rp * (1 + ecc)) / d0
    const vr = -Math.sqrt(Math.max(v2 - vt * vt, 0))

    // Relative position along a random horizontal direction, tangential velocity sets L
    const phi = rand() * Math.PI * 2
    const rx = Math.cos(phi), rz = Math.sin(phi)
    // Tangent t = axis x r gives angular momentum along +axis
    const tx = sense * rz, tz = -sense * rx
    const rel = [rx * d0, 0, rz * d0]
    const relV = [rx * vr + tx * vt, 0, rz * vr + tz * vt]

    const cores = new Float64Array(CORE_STRIDE * 2)
    const f1 = g2.mass / mu
    const f2 = g1.mass / mu
    for (let a = 0; a < 3; a++) {
      cores[a] = -f1 * rel[a]
      cores[3 + a] = -f1 * relV[a]
      cores[CORE_STRIDE + a] = f2 * rel[a]
      cores[CORE_STRIDE + 3 + a] = f2 * relV[a]
    }

    enc = {
      galaxies: [g1, g2],
      cores,
      coreCoreEps: Math.hypot(g1.eps, g2.eps),
      haloScale: 0.7 * (g1.radius + g2.radius),
      friction: 0.1 + 0.06 * rand(),
      innerScale: 0.3 * Math.max(g1.radius, g2.radius),
      drag: new Float64Array(6),
      pericenterTime: 0,
      mergeTime: 0,
    }
    precomputeTimes(enc)
    if (
      enc.pericenterTime > 0 &&
      enc.mergeTime >= mergeWindow[0] &&
      enc.mergeTime <= mergeWindow[1]
    ) {
      return enc
    }
  }
  return enc as Encounter
}

export type ParticleKind = 'disk' | 'bulge' | 'dust'

/**
 * Place particle i in galaxy gi on a circular orbit in that galaxy's potential (the other
 * galaxy is far away at the start). Returns the radius as a fraction of the disk radius.
 */
export function seedParticle(
  enc: Encounter,
  gi: 0 | 1,
  kind: ParticleKind,
  rand: Rand,
  pos: Float32Array,
  vel: Float32Array,
  i: number,
): number {
  const g = enc.galaxies[gi]
  const c = enc.cores
  const o = gi * CORE_STRIDE
  const R = g.radius
  let r: number
  let u: Vec3 // unit radial direction
  let w: Vec3 // unit direction of travel
  let height = 0

  if (kind === 'bulge') {
    // Compact spherical swarm; circular orbits in random planes are stable in a spherical
    // potential and read as a puffy, unrotating bulge
    r = g.eps * 0.9 * Math.pow(rand(), 0.75) + 0.02
    const cz = rand() * 2 - 1
    const az = rand() * Math.PI * 2
    const sz = Math.sqrt(1 - cz * cz)
    u = [sz * Math.cos(az), sz * Math.sin(az), cz]
    const helper: Vec3 = Math.abs(u[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]
    const p = normalize(cross(u, helper))
    const q = cross(u, p)
    const t = rand() * Math.PI * 2
    w = [
      p[0] * Math.cos(t) + q[0] * Math.sin(t),
      p[1] * Math.cos(t) + q[1] * Math.sin(t),
      p[2] * Math.cos(t) + q[2] * Math.sin(t),
    ]
  } else {
    // Exponential disk (gamma-2 radius draw), truncated; dust sits in a thinner inner layer
    const scale = kind === 'dust' ? R * 0.3 : R * 0.27
    const inner = kind === 'dust' ? 0.2 * R : 0.1 * R
    const outer = kind === 'dust' ? 0.95 * R : 1.1 * R
    do {
      r = -scale * Math.log(Math.max(rand() * rand(), 1e-12))
    } while (r < inner || r > outer)
    const t = rand() * Math.PI * 2
    const ct = Math.cos(t), st = Math.sin(t)
    u = [g.e1[0] * ct + g.e2[0] * st, g.e1[1] * ct + g.e2[1] * st, g.e1[2] * ct + g.e2[2] * st]
    w = [-g.e1[0] * st + g.e2[0] * ct, -g.e1[1] * st + g.e2[1] * ct, -g.e1[2] * st + g.e2[2] * ct]
    height = gaussian(rand) * R * (kind === 'dust' ? 0.012 : 0.025)
  }

  // Circular speed in the Plummer potential: v^2 = M r^2 / (r^2 + eps^2)^1.5
  const s2 = r * r + g.eps * g.eps
  const vc = Math.sqrt((g.mass * r * r) / (s2 * Math.sqrt(s2)))
  const jitter = kind === 'bulge' ? 0.08 : 0.04
  const j = i * 3
  for (let a = 0; a < 3; a++) {
    pos[j + a] = c[o + a] + u[a] * r + g.normal[a] * height
    vel[j + a] = c[o + 3 + a] + w[a] * vc + gaussian(rand) * jitter * vc
  }
  return r / R
}
