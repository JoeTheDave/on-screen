// Pure simulation for Tornado.tsx: town generation, the tornado's life cycle and wind field,
// progressive structural damage, rigid-body debris, kicked-up dust, the ground scour map, and
// the storm-chaser camera. No three.js in here so it can be run headlessly.
//
// Units are meters, seconds, kilograms. y is up. Every body (wall panel, roof strip, tree crown,
// car, loose shingle...) is one box-ish rigid body in a fixed-size structure-of-arrays pool, so
// the renderer can map it 1:1 onto an InstancedMesh slot and nothing is allocated per cycle.

export type Rand = () => number

export function mulberry32(seed: number): Rand {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// ---------------------------------------------------------------------------------------------
// Configuration

export const PHYSICS_STEP = 1 / 60
export const GRAVITY = 9.81
export const MAX_STRUCT_BODIES = 2600 // buildings, trees, cars, fences, roads
export const FRAG_POOL = 1400 // recycled small debris: shingles, boards, glass, leaves
export const MAX_BODIES = MAX_STRUCT_BODIES + FRAG_POOL
export const MAX_DUST = 1600 // CPU-advected dust puffs near the ground
export const SCOUR_SIZE = 256 // texels per side of the ground-scour map
export const SCOUR_EXTENT = 1000 // meters covered by the scour map, centered on the town
export const TOWN_RADIUS = 165

const START_DIST = 330 // the tornado forms this far from the town center
const FORM_SECONDS = 10
const MATURE_MAX_SECONDS = 75
const ROPE_SECONDS = 13
const AFTER_SECONDS = 9
const DAMAGE_RATE = 4 // damage per second at 25% over a piece's threshold is 1 (= gone)
const GROUND_FRICTION = 0.55
const RESTITUTION = 0.3
const MAX_SPEED = 110
const AIR = 0.66 // 0.5 * air density * drag coefficient

// Shapes (one InstancedMesh each; all unit-sized, centered, fitting in a unit cube)
export const SHAPE_BOX = 0
export const SHAPE_WEDGE = 1 // triangular prism, apex up, extruded along local z
export const SHAPE_CYL = 2
export const SHAPE_CROWN = 3 // faceted sphere
export const SHAPE_CONE = 4
export const SHAPE_COUNT = 5

// Body states
export const ST_EMPTY = 0
export const ST_STATIC = 1 // roads, foundations: never move
export const ST_ATTACHED = 2 // part of an intact structure, taking damage
export const ST_FLYING = 3
export const ST_SLEEPING = 4 // loose but at rest on the ground

// Body kinds (drive thresholds, fragments and the collapse rule)
const K_STATIC = 0
const K_WALL = 1
const K_ROOF = 2
const K_TRIM = 3 // gables, chimneys, steeples, window frames
const K_TRUNK = 4
const K_CROWN = 5
const K_FENCE = 6
const K_CAR = 7
const K_POLE = 8
const K_FRAG = 9
const K_HEAVY = 10 // silos, elevators, tanks
const K_GLASS = 11

// Storm phases
export const PH_FORMING = 0
export const PH_MATURE = 1
export const PH_ROPEOUT = 2
export const PH_AFTER = 3
export const PH_DONE = 4

// ---------------------------------------------------------------------------------------------
// Small math helpers

const smoothstep = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)))
  return t * t * (3 - 2 * t)
}
const pick = <T>(rand: Rand, arr: readonly T[]): T => arr[Math.floor(rand() * arr.length) % arr.length]
const range = (rand: Rand, a: number, b: number) => a + (b - a) * rand()
const TAU = Math.PI * 2

// Nudge each sRGB channel of a hex color by up to +-amt (0..1)
function vary(hex: number, rand: Rand, amt: number): number {
  const f = 1 + (rand() * 2 - 1) * amt
  const r = Math.min(255, Math.max(0, Math.round(((hex >> 16) & 255) * f)))
  const g = Math.min(255, Math.max(0, Math.round(((hex >> 8) & 255) * f)))
  const b = Math.min(255, Math.max(0, Math.round((hex & 255) * f)))
  return (r << 16) | (g << 8) | b
}

// ---------------------------------------------------------------------------------------------
// World: the body pool plus everything regenerated per cycle

export interface Structure {
  x: number
  z: number
  radius: number
  top: number
  first: number // body index range
  last: number
  walls: number
  wallsLeft: number
  collapsed: boolean
}

export interface World {
  count: number // structural bodies in use; fragments live at [MAX_STRUCT_BODIES, MAX_BODIES)
  px: Float32Array
  py: Float32Array
  pz: Float32Array
  vx: Float32Array
  vy: Float32Array
  vz: Float32Array
  qx: Float32Array
  qy: Float32Array
  qz: Float32Array
  qw: Float32Array
  wx: Float32Array // angular velocity (world space)
  wy: Float32Array
  wz: Float32Array
  sax: Float32Array // preferred tumble axis
  say: Float32Array
  saz: Float32Array
  hx: Float32Array // half extents
  hy: Float32Array
  hz: Float32Array
  k: Float32Array // drag acceleration per (m/s)^2
  wake: Float32Array // wind speed that lifts a resting body again
  threshold: Float32Array // wind speed where damage starts
  damage: Float32Array
  shake: Float32Array // 0..1 visual vibration of a loaded, still-attached piece
  rest: Float32Array
  color: Uint32Array // sRGB hex
  parent: Int32Array
  structure: Int32Array
  shape: Uint8Array
  kind: Uint8Array
  state: Uint8Array
  shed: Uint8Array // remaining shingles/siding this piece can shed before failing
  inst: Int32Array // instance slot in its shape's mesh
  dirty: Uint8Array // matrix needs re-upload
  colorDirty: Uint8Array
  shapeCount: Int32Array // instances per shape (box includes the fragment pool)
  structures: Structure[]
  obstacles: { x: number; z: number; r: number; top: number }[] // for camera clearance
  fragCursor: number
  roadAngle: number
  stepIndex: number
  // Dust (rendered as points straight from these arrays)
  dustPos: Float32Array
  dustVel: Float32Array
  dustAge: Float32Array
  dustLife: Float32Array
  dustSize: Float32Array // rendered size (0 = dead)
  dustAlpha: Float32Array
  dustCursor: number
  scour: Uint8Array
  // Stats (for tests and tuning)
  detached: number
  fragmentsSpawned: number
}

export function createWorld(): World {
  const n = MAX_BODIES
  const f = () => new Float32Array(n)
  return {
    count: 0,
    px: f(), py: f(), pz: f(), vx: f(), vy: f(), vz: f(),
    qx: f(), qy: f(), qz: f(), qw: f(), wx: f(), wy: f(), wz: f(),
    sax: f(), say: f(), saz: f(), hx: f(), hy: f(), hz: f(),
    k: f(), wake: f(), threshold: f(), damage: f(), shake: f(), rest: f(),
    color: new Uint32Array(n),
    parent: new Int32Array(n).fill(-1),
    structure: new Int32Array(n).fill(-1),
    shape: new Uint8Array(n),
    kind: new Uint8Array(n),
    state: new Uint8Array(n),
    shed: new Uint8Array(n),
    inst: new Int32Array(n),
    dirty: new Uint8Array(n),
    colorDirty: new Uint8Array(n),
    shapeCount: new Int32Array(SHAPE_COUNT),
    structures: [],
    obstacles: [],
    fragCursor: 0,
    roadAngle: 0,
    stepIndex: 0,
    dustPos: new Float32Array(MAX_DUST * 3),
    dustVel: new Float32Array(MAX_DUST * 3),
    dustAge: new Float32Array(MAX_DUST),
    dustLife: new Float32Array(MAX_DUST),
    dustSize: new Float32Array(MAX_DUST),
    dustAlpha: new Float32Array(MAX_DUST),
    dustCursor: 0,
    scour: new Uint8Array(SCOUR_SIZE * SCOUR_SIZE),
    detached: 0,
    fragmentsSpawned: 0,
  }
}

// ---------------------------------------------------------------------------------------------
// Palettes (sRGB hex)

const FOUNDATION = 0x8d8a84
const ASPHALT = 0x2b2c2e
const GRAVEL = 0x7b705f
const CENTER_LINE = 0xc9a227
const HOUSE_WALLS = [0xe8e4da, 0xd9d2c0, 0xb8c4cc, 0xc9b99a, 0xe0d4a8, 0x9aa8a0, 0xc4a68a, 0xf2efe8, 0x8c9aa6, 0xa9b59a]
const HOUSE_ROOFS = [0x3b3b3f, 0x4a3f38, 0x2c2e33, 0x5a4a42, 0x6b3a2e, 0x3f4a52, 0x51463c]
const BARN_WALLS = [0x8e2b22, 0x7a2a20, 0xa33b2a, 0x7b7266, 0x8a8f8c]
const BARN_ROOFS = [0x55595e, 0x3a3d40, 0x8a8d90, 0x6e7276]
const BRICK = [0x8a4b3a, 0x9c5a45, 0xa88a6a, 0x7d4636]
const METAL = 0x9aa0a4
const CONCRETE = 0xb5b0a6
const WINDOW = 0x1b2229
const DOORS = [0x5a2e22, 0x2f3a4a, 0x6b6b5a, 0x7a1f1f, 0x2e4030]
const TRUNK = 0x4a3a2c
const CROWNS = [0x2e4a22, 0x3a5527, 0x24401f, 0x4a5a2a, 0x355226]
const CONIFERS = [0x1f3820, 0x26402a, 0x203a28]
const CARS = [0xb02a2a, 0x2a4f8a, 0xd8d8d8, 0x262626, 0x6a6e72, 0xc9b27a, 0x2f5a3a, 0x8a1f1f, 0xe8e8e8, 0x1f3550]
const FENCES = [0xe8e6e0, 0x8a7a66, 0x6e604f]
const POLE = 0x5e4a38
const GLASS = 0x8fa3ad

// ---------------------------------------------------------------------------------------------
// Town generation

interface Frame {
  ox: number
  oz: number
  c: number
  s: number
  qy: number // yaw quaternion (0, qy, 0, qw)
  qw: number
}

// Frame whose local +z (the building's front) faces world direction (fx, fz)
function frameFacing(ox: number, oz: number, fx: number, fz: number): Frame {
  // Ry(yaw) maps local +z to (sin yaw, cos yaw)
  const yaw = Math.atan2(fx, fz)
  return { ox, oz, c: Math.cos(yaw), s: Math.sin(yaw), qy: Math.sin(yaw / 2), qw: Math.cos(yaw / 2) }
}

interface Road {
  x: number
  z: number
  ux: number
  uz: number
  len: number
  width: number
}

class Builder {
  w: World
  rand: Rand
  roads: Road[] = []
  foot: { x: number; z: number; r: number }[] = []
  structure = -1
  quality = 1

  constructor(w: World, rand: Rand) {
    this.w = w
    this.rand = rand
  }

  room(n: number) {
    return this.w.count + n <= MAX_STRUCT_BODIES
  }

  // Add one body in frame f at local center (lx, ly, lz) with local rotation lq (x,y,z,w).
  add(
    f: Frame, shape: number, kind: number,
    lx: number, ly: number, lz: number,
    sx: number, sy: number, sz: number,
    color: number, threshold: number, mass: number,
    parent = -1, lq: readonly number[] | null = null,
  ): number {
    const w = this.w
    if (w.count >= MAX_STRUCT_BODIES) return -1
    const i = w.count++
    w.px[i] = f.ox + f.c * lx + f.s * lz
    w.py[i] = ly
    w.pz[i] = f.oz - f.s * lx + f.c * lz
    // world = yaw * local
    const ax = 0, ay = f.qy, az = 0, aw = f.qw
    const bx = lq ? lq[0] : 0, by = lq ? lq[1] : 0, bz = lq ? lq[2] : 0, bw = lq ? lq[3] : 1
    w.qx[i] = aw * bx + ax * bw + ay * bz - az * by
    w.qy[i] = aw * by - ax * bz + ay * bw + az * bx
    w.qz[i] = aw * bz + ax * by - ay * bx + az * bw
    w.qw[i] = aw * bw - ax * bx - ay * by - az * bz
    w.vx[i] = w.vy[i] = w.vz[i] = 0
    w.wx[i] = w.wy[i] = w.wz[i] = 0
    w.hx[i] = sx / 2
    w.hy[i] = sy / 2
    w.hz[i] = sz / 2
    const area = (sx * sy + sy * sz + sx * sz) / 2 // mean projected area of a box
    w.k[i] = (AIR * area) / Math.max(mass, 0.1)
    w.wake[i] = Math.max(16, Math.sqrt((GROUND_FRICTION * GRAVITY) / w.k[i]) * 1.15)
    w.threshold[i] = threshold * this.quality
    w.damage[i] = 0
    w.shake[i] = 0
    w.rest[i] = 0
    w.color[i] = vary(color, this.rand, 0.05)
    w.parent[i] = parent
    w.structure[i] = this.structure
    w.shape[i] = shape
    w.kind[i] = kind
    w.state[i] = kind === K_STATIC ? ST_STATIC : ST_ATTACHED
    w.shed[i] = kind === K_ROOF ? 10 : kind === K_WALL ? 3 : 0
    this.randomSpinAxis(i)
    w.inst[i] = w.shapeCount[shape]++
    w.dirty[i] = 1
    w.colorDirty[i] = 1
    return i
  }

  randomSpinAxis(i: number) {
    const w = this.w
    const u = this.rand() * 2 - 1
    const a = this.rand() * TAU
    const r = Math.sqrt(1 - u * u)
    w.sax[i] = r * Math.cos(a)
    w.say[i] = u
    w.saz[i] = r * Math.sin(a)
  }

  begin(x: number, z: number, radius: number, top: number) {
    this.structure = this.w.structures.length
    this.w.structures.push({ x, z, radius, top, first: this.w.count, last: this.w.count, walls: 0, wallsLeft: 0, collapsed: false })
    this.w.obstacles.push({ x, z, r: radius, top })
    this.foot.push({ x, z, r: radius })
    this.quality = range(this.rand, 0.85, 1.2)
  }

  end() {
    const s = this.w.structures[this.structure]
    s.last = this.w.count
    for (let i = s.first; i < s.last; i++) if (this.w.kind[i] === K_WALL) s.walls++
    s.wallsLeft = s.walls
    this.structure = -1
    this.quality = 1
  }

  free(x: number, z: number, r: number, offRoad = true) {
    for (const f of this.foot) {
      const dx = f.x - x
      const dz = f.z - z
      if (dx * dx + dz * dz < (f.r + r) * (f.r + r)) return false
    }
    if (!offRoad) return true
    for (const road of this.roads) {
      const dx = x - road.x
      const dz = z - road.z
      const along = dx * road.ux + dz * road.uz
      if (Math.abs(along) > road.len / 2 + r) continue
      const across = Math.abs(-dx * road.uz + dz * road.ux)
      if (across < road.width / 2 + r + 1) return false
    }
    return true
  }

  road(x: number, z: number, ux: number, uz: number, len: number, width: number, color: number, centerLine: boolean) {
    this.roads.push({ x, z, ux, uz, len, width })
    const f = frameFacing(x, z, -uz, ux) // local x runs along the road
    this.add(f, SHAPE_BOX, K_STATIC, 0, 0.03, 0, len, 0.06, width, color, Infinity, 1)
    if (centerLine) {
      for (let t = -len / 2 + 6; t < len / 2 - 6; t += 12) {
        if (!this.room(80)) break
        if (Math.abs(t) > 500) continue // fog hides it beyond this anyway
        this.add(f, SHAPE_BOX, K_STATIC, t, 0.065, 0, 4, 0.02, 0.18, CENTER_LINE, Infinity, 1)
      }
    }
  }

  // Gabled building: ridge along local x, front wall at local +z. Returns false if out of room.
  gabled(
    f: Frame, w: number, d: number, stories: number, storyH: number, pitch: number,
    wallColor: number, roofColor: number, wallT: number, roofT: number,
    opts: { windows: boolean; door: number; chimney: boolean; wallArea: number; brick?: boolean },
  ) {
    const rand = this.rand
    const t = 0.22
    const base = 0.35
    const wallTop = base + stories * storyH
    const roofH = (d / 2) * Math.tan(pitch)
    // Foundation slab stays behind when everything else is gone
    this.add(f, SHAPE_BOX, K_STATIC, 0, base / 2, 0, w + 0.5, base, d + 0.5, FOUNDATION, Infinity, 1)
    const nw = Math.max(1, Math.round(w / 3.4))
    const nd = Math.max(1, Math.round(d / 3.4))
    const pw = w / nw
    const pd = (d - 2 * t) / nd
    const below: number[] = []
    for (let s = 0; s < stories; s++) {
      const y = base + storyH * (s + 0.5)
      const panels: number[] = []
      const T = wallT * (s > 0 ? 0.93 : 1)
      // front and back
      for (const side of [1, -1]) {
        for (let i = 0; i < nw; i++) {
          const x = -w / 2 + pw * (i + 0.5)
          const z = side * (d / 2 - t / 2)
          const par = s > 0 ? below[panels.length] : -1
          const p = this.add(f, SHAPE_BOX, K_WALL, x, y, z, pw, storyH, t, wallColor, T, pw * storyH * opts.wallArea, par)
          panels.push(p)
          if (p < 0) return false
          if (!opts.windows) continue
          const isDoor = s === 0 && side === 1 && i === Math.floor(nw / 2)
          if (isDoor) {
            this.add(f, SHAPE_BOX, K_TRIM, x, base + 1.05, z + side * 0.13, 1.0, 2.1, 0.08, opts.door, T, 40, p)
          } else if (rand() < (side === 1 ? 0.85 : 0.55)) {
            this.add(f, SHAPE_BOX, K_GLASS, x, y + 0.2, z + side * 0.13, Math.min(1.3, pw * 0.5), 1.25, 0.08, WINDOW, T, 25, p)
          }
        }
      }
      // sides
      for (const side of [1, -1]) {
        for (let i = 0; i < nd; i++) {
          const z = -d / 2 + t + pd * (i + 0.5)
          const x = side * (w / 2 - t / 2)
          const par = s > 0 ? below[panels.length] : -1
          const p = this.add(f, SHAPE_BOX, K_WALL, x, y, z, t, storyH, pd, wallColor, T, pd * storyH * opts.wallArea, par)
          panels.push(p)
          if (p < 0) return false
          if (opts.windows && nd > 1 && rand() < 0.5) {
            this.add(f, SHAPE_BOX, K_GLASS, x + side * 0.13, y + 0.2, z, 0.08, 1.2, Math.min(1.1, pd * 0.45), WINDOW, T, 25, p)
          }
        }
      }
      below.length = 0
      below.push(...panels)
    }
    // Gable ends: wedge's triangle lies in its local x-y plane, so turn it to span local z
    const gq = [0, Math.SQRT1_2, 0, Math.SQRT1_2]
    for (const side of [1, -1]) {
      this.add(f, SHAPE_WEDGE, K_TRIM, side * (w / 2 - t / 2), wallTop + roofH / 2, 0, d, roofH, t, wallColor, roofT + 5, d * roofH * 0.5 * opts.wallArea, -1, gq)
    }
    // Roof: two slopes, each split into strips along the ridge
    const o = 0.5
    const run = d / 2 + o
    const slope = run / Math.cos(pitch)
    const nr = Math.max(1, Math.round((w + 2 * o) / 3.6))
    const sw = (w + 2 * o) / nr
    const top = wallTop + roofH
    for (const side of [1, -1]) {
      const a = side * pitch
      const q = [Math.sin(a / 2), 0, 0, Math.cos(a / 2)]
      // midpoint of the slope from ridge to eave, lifted half a thickness off it
      const zm = side * (run * 0.5 + Math.sin(pitch) * 0.12)
      const ym = top - run * Math.tan(pitch) * 0.5 + Math.cos(pitch) * 0.12
      for (let i = 0; i < nr; i++) {
        const x = -w / 2 - o + sw * (i + 0.5)
        if (this.add(f, SHAPE_BOX, K_ROOF, x, ym, zm, sw, 0.2, slope, roofColor, roofT, sw * slope * 22, -1, q) < 0) return false
      }
    }
    if (opts.chimney) {
      const cx = (rand() < 0.5 ? -1 : 1) * w * 0.28
      const ch = roofH + 1.3
      this.add(f, SHAPE_BOX, K_TRIM, cx, wallTop + ch / 2, -d * 0.18, 0.8, ch, 0.8, pick(rand, BRICK), roofT + 6, 900, -1)
    }
    return true
  }

  house(x: number, z: number, fx: number, fz: number) {
    const rand = this.rand
    const stories = rand() < 0.35 ? 2 : 1
    const w = range(rand, 9, 13.5)
    const d = range(rand, 7, 9.5)
    const r = Math.hypot(w, d) / 2 + 1.5
    if (!this.free(x, z, r) || !this.room(48)) return false
    const top = 0.35 + stories * 2.8 + (d / 2) * Math.tan(0.5)
    this.begin(x, z, r, top)
    const f = frameFacing(x, z, fx, fz)
    this.gabled(f, w, d, stories, 2.8, range(rand, 0.38, 0.62), pick(rand, HOUSE_WALLS), pick(rand, HOUSE_ROOFS), 52, 35, {
      windows: true, door: pick(rand, DOORS), chimney: rand() < 0.45, wallArea: 42,
    })
    this.end()
    // Yard: a tree, a car in the drive, sometimes a shed out back
    const side = rand() < 0.5 ? -1 : 1
    if (rand() < 0.45) this.car(f.ox + f.c * side * (w / 2 + 3) + f.s * (d / 2 + 2), f.oz - f.s * side * (w / 2 + 3) + f.c * (d / 2 + 2), fz, -fx)
    if (rand() < 0.7) this.tree(x - fx * (d / 2 + range(rand, 3, 7)) + (rand() - 0.5) * 8, z - fz * (d / 2 + range(rand, 3, 7)) + (rand() - 0.5) * 8)
    if (rand() < 0.3) this.shed(x - fx * (d / 2 + 9) - side * fz * 4, z - fz * (d / 2 + 9) + side * fx * 4, fx, fz)
    return true
  }

  shed(x: number, z: number, fx: number, fz: number) {
    const rand = this.rand
    const w = range(rand, 3, 5)
    const d = range(rand, 2.5, 4)
    const r = Math.hypot(w, d) / 2 + 0.5
    if (!this.free(x, z, r) || !this.room(16)) return false
    this.begin(x, z, r, 3.5)
    this.gabled(frameFacing(x, z, fx, fz), w, d, 1, 2.3, 0.35, rand() < 0.5 ? METAL : pick(rand, HOUSE_WALLS), pick(rand, BARN_ROOFS), 38, 32, {
      windows: false, door: 0, chimney: false, wallArea: 14,
    })
    this.end()
    return true
  }

  barn(x: number, z: number, fx: number, fz: number) {
    const rand = this.rand
    const w = range(rand, 16, 24)
    const d = range(rand, 11, 14)
    const r = Math.hypot(w, d) / 2 + 1
    if (!this.free(x, z, r) || !this.room(60)) return false
    const wall = pick(rand, BARN_WALLS)
    this.begin(x, z, r, 4.5 + d * 0.5)
    const f = frameFacing(x, z, fx, fz)
    this.gabled(f, w, d, 1, 4.6, range(rand, 0.65, 0.8), wall, pick(rand, BARN_ROOFS), 44, 33, {
      windows: false, door: 0, chimney: false, wallArea: 26,
    })
    // big sliding door, white-trimmed
    const p = this.w.structures[this.structure].first + 1
    this.add(f, SHAPE_BOX, K_TRIM, 0, 2.5, d / 2 + 0.05, 4.2, 4.2, 0.1, 0xe6e2d8, 44, 300, p)
    this.add(f, SHAPE_BOX, K_TRIM, 0, 2.5, d / 2 + 0.1, 3.6, 3.6, 0.1, wall, 44, 250, p)
    this.end()
    return true
  }

  // Flat-roofed brick/metal block: school, store, warehouse
  block(x: number, z: number, fx: number, fz: number) {
    const rand = this.rand
    const w = range(rand, 22, 34)
    const d = range(rand, 14, 20)
    const r = Math.hypot(w, d) / 2 + 2
    if (!this.free(x, z, r) || !this.room(130)) return false
    const brick = rand() < 0.65
    const wallColor = brick ? pick(rand, BRICK) : METAL
    const storyH = 4.2
    const stories = rand() < 0.4 ? 2 : 1
    this.begin(x, z, r, stories * storyH + 1)
    const f = frameFacing(x, z, fx, fz)
    const t = 0.3
    const base = 0.35
    this.add(f, SHAPE_BOX, K_STATIC, 0, base / 2, 0, w + 0.6, base, d + 0.6, FOUNDATION, Infinity, 1)
    const T = brick ? 66 : 50
    const areal = brick ? 160 : 30
    const nw = Math.round(w / 4.5)
    const nd = Math.round(d / 4.5)
    const below: number[] = []
    for (let s = 0; s < stories; s++) {
      const y = base + storyH * (s + 0.5)
      const panels: number[] = []
      for (const side of [1, -1]) {
        for (let i = 0; i < nw; i++) {
          const px = -w / 2 + (w / nw) * (i + 0.5)
          const p = this.add(f, SHAPE_BOX, K_WALL, px, y, side * (d / 2 - t / 2), w / nw, storyH, t, wallColor, T, (w / nw) * storyH * areal, s > 0 ? below[panels.length] : -1)
          panels.push(p)
          if (p >= 0 && brick) this.add(f, SHAPE_BOX, K_GLASS, px, y + 0.3, side * (d / 2 + 0.02), (w / nw) * 0.7, 1.4, 0.08, WINDOW, T, 40, p)
        }
        for (let i = 0; i < nd; i++) {
          const pz = -d / 2 + t + ((d - 2 * t) / nd) * (i + 0.5)
          const p = this.add(f, SHAPE_BOX, K_WALL, side * (w / 2 - t / 2), y, pz, t, storyH, (d - 2 * t) / nd, wallColor, T, ((d - 2 * t) / nd) * storyH * areal, s > 0 ? below[panels.length] : -1)
          panels.push(p)
        }
      }
      below.length = 0
      below.push(...panels)
    }
    // Flat roof tiles (light membrane/deck: the first thing to go) plus a parapet cap
    const top = base + stories * storyH
    const tx = Math.round(w / 6)
    const tz = Math.round(d / 6)
    for (let i = 0; i < tx; i++) {
      for (let j = 0; j < tz; j++) {
        this.add(f, SHAPE_BOX, K_ROOF, -w / 2 + (w / tx) * (i + 0.5), top + 0.15, -d / 2 + (d / tz) * (j + 0.5), w / tx, 0.3, d / tz, 0x55524e, 40, (w / tx) * (d / tz) * 30)
      }
    }
    if (rand() < 0.6) {
      // rooftop HVAC unit
      this.add(f, SHAPE_BOX, K_TRIM, w * 0.2, top + 1.0, 0, 3, 1.4, 2, METAL, 42, 900)
    }
    this.end()
    return true
  }

  church(x: number, z: number, fx: number, fz: number) {
    const rand = this.rand
    const w = range(rand, 16, 20)
    const d = range(rand, 9, 11)
    const r = Math.hypot(w + 4, d) / 2 + 1
    if (!this.free(x, z, r) || !this.room(90)) return false
    const wall = rand() < 0.5 ? 0xece8de : pick(rand, BRICK)
    this.begin(x, z, r, 22)
    // The ridge runs toward the road, so build in a frame whose local +x points at it
    const side = frameFacing(x, z, -fz, fx)
    this.gabled(side, w, d, 1, 5.5, 0.75, wall, pick(rand, HOUSE_ROOFS), 55, 37, {
      windows: true, door: 0x5a2e22, chimney: false, wallArea: 60,
    })
    // Steeple: tower + spire at one end of the ridge
    const tx = w / 2 + 1.6
    const tower = this.add(side, SHAPE_BOX, K_WALL, tx, 0.35 + 6, 0, 3.2, 12, 3.2, wall, 58, 6000)
    this.add(side, SHAPE_CONE, K_TRIM, tx, 12.35 + 4, 0, 3.4, 8, 3.4, pick(rand, HOUSE_ROOFS), 40, 1500, tower)
    this.end()
    return true
  }

  silo(x: number, z: number) {
    const rand = this.rand
    const rr = range(rand, 2.5, 3.4)
    if (!this.free(x, z, rr + 0.5) || !this.room(6)) return false
    const h = range(rand, 12, 18)
    this.begin(x, z, rr + 0.5, h + rr)
    const f = frameFacing(x, z, 0, 1)
    const col = rand() < 0.5 ? CONCRETE : 0x6d8aa0
    const rings = 3
    let p = -1
    for (let i = 0; i < rings; i++) {
      p = this.add(f, SHAPE_CYL, K_HEAVY, 0, (h / rings) * (i + 0.5), 0, rr * 2, h / rings, rr * 2, col, 62 - i * 4, 9000, p)
    }
    this.add(f, SHAPE_CROWN, K_TRIM, 0, h, 0, rr * 2, rr * 1.4, rr * 2, METAL, 45, 600, p)
    this.end()
    return true
  }

  // Corrugated grain bins: light, they roll away like cans
  bin(x: number, z: number) {
    const rand = this.rand
    const rr = range(rand, 3, 4.5)
    if (!this.free(x, z, rr + 0.5) || !this.room(3)) return false
    const h = range(rand, 5, 7)
    this.begin(x, z, rr + 0.5, h + 2)
    const f = frameFacing(x, z, 0, 1)
    const body = this.add(f, SHAPE_CYL, K_HEAVY, 0, h / 2, 0, rr * 2, h, rr * 2, 0xb8bcbe, 46, 2600)
    this.add(f, SHAPE_CONE, K_TRIM, 0, h + rr * 0.3, 0, rr * 2.05, rr * 0.6, rr * 2.05, 0xc2c6c8, 40, 500, body)
    this.end()
    return true
  }

  waterTower(x: number, z: number) {
    const rand = this.rand
    if (!this.free(x, z, 7) || !this.room(8)) return false
    const h = range(rand, 22, 28)
    this.begin(x, z, 7, h + 8)
    const f = frameFacing(x, z, 0, 1)
    let leg = -1
    for (const [lx, lz] of [[-3.5, -3.5], [3.5, -3.5], [-3.5, 3.5], [3.5, 3.5]]) {
      leg = this.add(f, SHAPE_CYL, K_WALL, lx, h / 2, lz, 0.6, h, 0.6, 0x9ca3a8, 74, 3000)
    }
    // the tank comes down when a leg buckles
    const tank = this.add(f, SHAPE_CROWN, K_HEAVY, 0, h + 3.5, 0, 11, 8, 11, 0xd8dcdf, 80, 20000, leg)
    this.add(f, SHAPE_CONE, K_TRIM, 0, h + 8.5, 0, 6, 2.5, 6, 0xc6cacd, 50, 400, tank)
    this.end()
    return true
  }

  tree(x: number, z: number) {
    const rand = this.rand
    if (!this.free(x, z, 2) || !this.room(2)) return false
    const conifer = rand() < 0.3
    const h = range(rand, 3, 5.5)
    const cr = range(rand, 4, 7.5)
    const f = frameFacing(x, z, rand() - 0.5, rand() - 0.5)
    const trunk = this.add(f, SHAPE_CYL, K_TRUNK, 0, h / 2, 0, 0.6, h, 0.6, TRUNK, 52, 500)
    if (conifer) {
      this.add(f, SHAPE_CONE, K_CROWN, 0, h * 0.6 + cr, 0, cr * 0.9, cr * 2.2, cr * 0.9, pick(rand, CONIFERS), 34, 280, trunk)
    } else {
      this.add(f, SHAPE_CROWN, K_CROWN, 0, h + cr * 0.35, 0, cr, cr * range(rand, 0.8, 1.05), cr, pick(rand, CROWNS), 30, 260, trunk)
    }
    this.foot.push({ x, z, r: 2 })
    this.w.obstacles.push({ x, z, r: 2, top: h + cr })
    return true
  }

  car(x: number, z: number, fx: number, fz: number, offRoad = true) {
    if (!this.free(x, z, 2.5, offRoad) || !this.room(2)) return false
    const rand = this.rand
    const f = frameFacing(x, z, -fz, fx) // cars parked along the frame's direction
    const col = pick(rand, CARS)
    const body = this.add(f, SHAPE_BOX, K_CAR, 0, 0.75, 0, 1.8, 0.9, 4.4, col, 44, 1250)
    this.add(f, SHAPE_BOX, K_CAR, 0, 1.5, -0.25, 1.6, 0.65, 2.4, vary(col, rand, 0.15), 44, 200, body)
    this.foot.push({ x, z, r: 2.5 })
    return true
  }

  // Fence along a straight line: posts with one rail per bay
  fence(x0: number, z0: number, x1: number, z1: number) {
    const rand = this.rand
    const len = Math.hypot(x1 - x0, z1 - z0)
    const bays = Math.floor(len / 3)
    if (bays < 1 || !this.room(bays * 2 + 2)) return
    const ux = (x1 - x0) / len
    const uz = (z1 - z0) / len
    const col = pick(rand, FENCES)
    const f = frameFacing(x0, z0, -uz, ux) // local x along the fence
    for (let i = 0; i <= bays; i++) {
      const post = this.add(f, SHAPE_BOX, K_FENCE, i * 3, 0.65, 0, 0.16, 1.3, 0.16, col, 30, 18)
      if (i < bays) this.add(f, SHAPE_BOX, K_FENCE, i * 3 + 1.5, 0.95, 0.1, 3, 0.14, 0.05, col, 26, 8, post)
    }
  }

  pole(x: number, z: number, ux: number, uz: number) {
    if (!this.room(2)) return
    const f = frameFacing(x, z, ux, uz)
    const p = this.add(f, SHAPE_CYL, K_POLE, 0, 5.5, 0, 0.32, 11, 0.32, POLE, 48, 450)
    this.add(f, SHAPE_BOX, K_POLE, 0, 10.3, 0, 2.6, 0.14, 0.14, POLE, 48, 30, p)
    this.w.obstacles.push({ x, z, r: 1, top: 11 })
  }

  // A spot beside a road, facing it: returns [x, z, fx, fz]
  besideRoad(road: Road, t: number, side: number, setback: number): [number, number, number, number] {
    const nx = -road.uz * side
    const nz = road.ux * side
    return [road.x + road.ux * t + nx * (road.width / 2 + setback), road.z + road.uz * t + nz * (road.width / 2 + setback), -nx, -nz]
  }
}

export function generateTown(w: World, rand: Rand) {
  // Reset the pool (fragments included)
  w.count = 0
  w.state.fill(ST_EMPTY)
  w.parent.fill(-1)
  w.structure.fill(-1)
  w.dirty.fill(1)
  w.shapeCount.fill(0)
  w.structures = []
  w.obstacles = []
  w.fragCursor = 0
  w.stepIndex = 0
  w.dustSize.fill(0)
  w.dustAlpha.fill(0)
  w.dustLife.fill(0)
  w.scour.fill(0)
  w.detached = 0
  w.fragmentsSpawned = 0

  const b = new Builder(w, rand)
  const angle = rand() * Math.PI
  w.roadAngle = angle
  const ux = Math.cos(angle)
  const uz = Math.sin(angle)
  const nx = -uz
  const nz = ux

  // Roads: a main highway through town, a grid of streets, gravel section roads out in the country
  b.road(0, 0, ux, uz, 2600, 9, ASPHALT, true)
  const blockLen = range(rand, 62, 80)
  const nStreets = rand() < 0.5 ? 1 : 2
  const streets: Road[] = [b.roads[0]]
  for (let k = -nStreets; k <= nStreets; k++) {
    if (k !== 0) {
      b.road(nx * k * blockLen, nz * k * blockLen, ux, uz, 2 * TOWN_RADIUS + 30, 6.5, ASPHALT, false)
      streets.push(b.roads[b.roads.length - 1])
    }
    const crossLong = k === 0
    b.road(ux * k * blockLen, uz * k * blockLen, nx, nz, crossLong ? 2600 : 2 * TOWN_RADIUS + 30, crossLong ? 7 : 6.5, crossLong ? GRAVEL : ASPHALT, false)
    streets.push(b.roads[b.roads.length - 1])
  }
  const rural: Road[] = []
  for (const side of [-1, 1]) {
    const off = side * range(rand, 330, 420)
    b.road(ux * off, uz * off, nx, nz, 2200, 6, GRAVEL, false)
    rural.push(b.roads[b.roads.length - 1])
  }
  rural.push(b.roads[0], streets[streets.length > 2 ? 2 : 1])

  // Landmarks first so they get good spots: a church, a school/store block, grain bins, a water tower
  const landmark = (fn: (x: number, z: number, fx: number, fz: number) => boolean, setback: number, tries = 30) => {
    for (let i = 0; i < tries; i++) {
      const road = pick(rand, streets)
      const t = range(rand, -TOWN_RADIUS * 0.8, TOWN_RADIUS * 0.8)
      const [x, z, fx, fz] = b.besideRoad(road, t, rand() < 0.5 ? -1 : 1, setback)
      if (Math.hypot(x, z) > TOWN_RADIUS) continue
      if (fn.call(b, x, z, fx, fz)) return true
    }
    return false
  }
  landmark(b.church, 14)
  landmark(b.block, 16)
  if (rand() < 0.6) landmark(b.block, 16)
  if (rand() < 0.75) landmark((x, z) => b.waterTower(x, z), 14)
  // Grain elevator cluster by the highway at the edge of town
  {
    const t = (rand() < 0.5 ? -1 : 1) * range(rand, TOWN_RADIUS * 0.7, TOWN_RADIUS + 40)
    const [x, z, fx, fz] = b.besideRoad(b.roads[0], t, rand() < 0.5 ? -1 : 1, 16)
    for (let i = 0; i < 5; i++) {
      const bx = x - fz * (i - 2) * 8.5
      const bz = z + fx * (i - 2) * 8.5
      if (i % 2 === 0) b.silo(bx, bz)
      else b.bin(bx, bz)
    }
  }

  // Houses on lots along the town streets, nearest the center first (with jitter) so the
  // budget makes a compact town rather than a sparse one
  const lots: { x: number; z: number; fx: number; fz: number; pri: number }[] = []
  for (const road of streets) {
    const lotW = range(rand, 19, 25)
    const half = Math.min(road.len / 2, TOWN_RADIUS + 20)
    for (let t = -half + lotW / 2; t < half; t += lotW) {
      for (const side of [-1, 1]) {
        const [x, z, fx, fz] = b.besideRoad(road, t + (rand() - 0.5) * 3, side, range(rand, 8, 11))
        const dist = Math.hypot(x, z)
        if (dist > TOWN_RADIUS) continue
        lots.push({ x, z, fx, fz, pri: dist + rand() * 70 })
      }
    }
  }
  lots.sort((a, c) => a.pri - c.pri)
  const houseBudget = MAX_STRUCT_BODIES - 650 // keep room for farms, trees, fences and poles
  for (const lot of lots) {
    if (w.count > houseBudget) break
    if (rand() < 0.82) b.house(lot.x, lot.z, lot.fx, lot.fz)
    else if (rand() < 0.5) b.tree(lot.x, lot.z)
  }

  // Cars parked along the streets
  for (let i = 0; i < 14; i++) {
    const road = pick(rand, streets)
    const t = range(rand, -TOWN_RADIUS, TOWN_RADIUS)
    const side = rand() < 0.5 ? -1 : 1
    const x = road.x + road.ux * t - road.uz * side * (road.width / 2 + 1.3)
    const z = road.z + road.uz * t + road.ux * side * (road.width / 2 + 1.3)
    if (Math.hypot(x, z) < TOWN_RADIUS) {
      b.car(x, z, -road.uz, road.ux, false) // parked on the shoulder
    }
  }

  // Power poles along the highway
  for (let t = -520; t <= 520; t += 46) {
    const [x, z] = b.besideRoad(b.roads[0], t, 1, 3.5)
    b.pole(x, z, ux, uz)
  }

  // Farmsteads out in the country
  const farms = 3 + Math.floor(rand() * 4)
  for (let i = 0; i < farms; i++) {
    for (let tries = 0; tries < 12; tries++) {
      const road = pick(rand, rural)
      const t = (rand() < 0.5 ? -1 : 1) * range(rand, 60, 560)
      const side = rand() < 0.5 ? -1 : 1
      const [x, z, fx, fz] = b.besideRoad(road, t, side, 22)
      const dist = Math.hypot(x, z)
      if (dist < TOWN_RADIUS + 40 || dist > 640) continue
      if (!b.free(x, z, 40)) continue
      // lay the yard out in the farm's own frame: along-road (ax, az), toward the road (fx, fz)
      const ax = -fz
      const az = fx
      b.house(x + ax * 10, z + az * 10, fx, fz)
      b.barn(x - ax * 14 - fx * 18, z - az * 14 - fz * 18, ax, az)
      b.silo(x - ax * 32 - fx * 12, z - az * 32 - fz * 12)
      if (rand() < 0.6) b.bin(x - ax * 32 - fx * 22, z - az * 32 - fz * 22)
      b.shed(x + ax * 8 - fx * 22, z + az * 8 - fz * 22, fx, fz)
      // windbreak row of trees behind the yard
      for (let j = 0; j < 10; j++) b.tree(x + ax * (j * 6 - 30) - fx * 40 + (rand() - 0.5) * 2, z + az * (j * 6 - 30) - fz * 40 + (rand() - 0.5) * 2)
      // pasture fence along the road beyond the yard
      const fx0 = x + ax * 22 + fx * 12
      const fz0 = z + az * 22 + fz * 12
      b.fence(fx0, fz0, fx0 + ax * 45, fz0 + az * 45)
      if (rand() < 0.6) b.fence(fx0 + ax * 45, fz0 + az * 45, fx0 + ax * 45 - fx * 36, fz0 + az * 45 - fz * 36)
      break
    }
  }

  // Scattered trees: tree lines along field edges and lone trees
  for (let i = 0; i < 400 && b.room(4); i++) {
    const a = rand() * TAU
    const rr = range(rand, 40, 560)
    b.tree(Math.cos(a) * rr, Math.sin(a) * rr)
  }

  // Fragment pool: box instances after the structural boxes
  const fragBase = w.shapeCount[SHAPE_BOX]
  for (let j = 0; j < FRAG_POOL; j++) {
    const i = MAX_STRUCT_BODIES + j
    w.state[i] = ST_EMPTY
    w.shape[i] = SHAPE_BOX
    w.kind[i] = K_FRAG
    w.inst[i] = fragBase + j
    w.dirty[i] = 1
    w.colorDirty[i] = 1
    w.color[i] = 0x777777
  }
  w.shapeCount[SHAPE_BOX] += FRAG_POOL
}

// ---------------------------------------------------------------------------------------------
// The storm

export interface Storm {
  time: number
  phase: number
  phaseTime: number
  x: number
  z: number
  heading: number
  heading0: number
  turnAmp: number
  turnFreq: number
  turnPhase: number
  speed: number
  speedMul: number
  vx: number
  vz: number
  traveled: number
  ropeDistance: number
  RcBase: number
  Rc: number
  Vmax: number
  H: number // height of the funnel's top (the wall cloud's base)
  wallDepth: number // how far the wall cloud hangs below the main cloud base
  bottomFrac: number
  topFrac: number
  rB: number // visual funnel radius at the ground
  rT: number // and at the top
  intensity: number // wind strength 0..1
  debris: number // debris cloud strength 0..1
  funnelBottom: number // lowest visible condensation, y
  funnelVis: number
  rope: number
  leanBaseX: number
  leanBaseZ: number
  leanX: number
  leanZ: number
  wobbleBase: number
  wobble: number
  ph1: number
  ph2: number
  subCount: number
  svPhase: number
  svOmega: number
  nearestX: number
  nearestZ: number
  nearestD: number
}

export function newStorm(rand: Rand): Storm {
  const styleRoll = rand()
  // rope / cone / wedge
  const style = styleRoll < 0.25 ? 0 : styleRoll < 0.72 ? 1 : 2
  const RcBase = style === 0 ? range(rand, 18, 26) : style === 1 ? range(rand, 26, 38) : range(rand, 40, 55)
  const Vmax = range(rand, 62, 94) + (style === 2 ? 6 : 0)
  const H = range(rand, 165, 225)
  const theta = rand() * TAU
  const perp = (rand() - 0.5) * 150
  const heading0 = theta + (rand() - 0.5) * 0.35
  const x = -Math.cos(theta) * START_DIST - Math.sin(theta) * perp
  const z = -Math.sin(theta) * START_DIST + Math.cos(theta) * perp
  const lean = range(rand, 0.12, 0.42) * H
  const lat = (rand() - 0.5) * 0.3 * H
  const s: Storm = {
    time: 0,
    phase: PH_FORMING,
    phaseTime: 0,
    x, z,
    heading: heading0,
    heading0,
    turnAmp: range(rand, 0.1, 0.35),
    turnFreq: range(rand, 0.03, 0.07),
    turnPhase: rand() * TAU,
    speed: range(rand, 9, 15),
    speedMul: 0.6,
    vx: 0,
    vz: 0,
    traveled: 0,
    ropeDistance: START_DIST + range(rand, 150, 280),
    RcBase,
    Rc: RcBase,
    Vmax,
    H,
    wallDepth: range(rand, 25, 45),
    bottomFrac: style === 0 ? 0.35 : style === 1 ? 0.55 : 1.0,
    topFrac: style === 0 ? 2.2 : style === 1 ? 2.6 : 2.4,
    rB: 0,
    rT: 0,
    intensity: 0,
    debris: 0,
    funnelBottom: H,
    funnelVis: 1,
    rope: 0,
    // the top trails behind the ground circulation
    leanBaseX: -Math.cos(heading0) * lean - Math.sin(heading0) * lat,
    leanBaseZ: -Math.sin(heading0) * lean + Math.cos(heading0) * lat,
    leanX: 0,
    leanZ: 0,
    wobbleBase: range(rand, 3, 9),
    wobble: 0,
    ph1: rand() * TAU,
    ph2: rand() * TAU,
    subCount: style === 0 ? 0 : Math.floor(range(rand, style === 2 ? 3 : 0, style === 2 ? 6 : 4)),
    svPhase: rand() * TAU,
    svOmega: 0,
    nearestX: x,
    nearestZ: z,
    nearestD: Infinity,
  }
  s.svOmega = (0.45 * Vmax) / (0.8 * RcBase)
  stepStorm(s, 0)
  return s
}

export function stepStorm(s: Storm, dt: number) {
  s.time += dt
  s.phaseTime += dt
  const pt = s.phaseTime
  if (s.phase === PH_FORMING) {
    s.intensity = smoothstep(0, FORM_SECONDS, pt)
    s.funnelBottom = s.H * 0.85 * (1 - smoothstep(1.5, FORM_SECONDS * 0.95, pt))
    s.debris = smoothstep(2, FORM_SECONDS * 0.7, pt)
    s.funnelVis = 1
    s.rope = 0
    s.speedMul = 0.6
    if (pt >= FORM_SECONDS) {
      s.phase = PH_MATURE
      s.phaseTime = 0
    }
  } else if (s.phase === PH_MATURE) {
    s.intensity = 0.92 + 0.08 * Math.sin(s.time * 0.35)
    s.funnelBottom = 0
    s.debris = 1
    s.speedMul = 1
    if (s.traveled >= s.ropeDistance || pt > MATURE_MAX_SECONDS) {
      s.phase = PH_ROPEOUT
      s.phaseTime = 0
    }
  } else if (s.phase === PH_ROPEOUT) {
    s.rope = smoothstep(0, ROPE_SECONDS * 0.7, pt)
    s.intensity = 1 - smoothstep(0, ROPE_SECONDS * 0.9, pt)
    s.funnelBottom = s.H * smoothstep(ROPE_SECONDS * 0.55, ROPE_SECONDS, pt)
    s.funnelVis = 1 - smoothstep(ROPE_SECONDS * 0.75, ROPE_SECONDS, pt)
    s.debris = 1 - smoothstep(0, ROPE_SECONDS * 0.6, pt)
    s.speedMul = 0.8
    if (pt >= ROPE_SECONDS) {
      s.phase = PH_AFTER
      s.phaseTime = 0
    }
  } else if (s.phase === PH_AFTER) {
    s.intensity = 0
    s.debris = 0
    s.funnelVis = 0
    s.funnelBottom = s.H
    s.speedMul = 0
    if (pt >= AFTER_SECONDS) {
      s.phase = PH_DONE
      s.phaseTime = 0
    }
  }
  s.heading = s.heading0 + s.turnAmp * Math.sin(s.time * s.turnFreq + s.turnPhase)
  const sp = s.speed * s.speedMul
  s.vx = Math.cos(s.heading) * sp
  s.vz = Math.sin(s.heading) * sp
  s.x += s.vx * dt
  s.z += s.vz * dt
  s.traveled += sp * dt
  s.Rc = s.RcBase * (1 - 0.55 * s.rope)
  s.rB = s.RcBase * s.bottomFrac * (1 - 0.75 * s.rope)
  s.rT = s.RcBase * s.topFrac * (1 - 0.45 * s.rope)
  s.wobble = s.wobbleBase * (1 + 2.5 * s.rope)
  s.leanX = s.leanBaseX * (1 + 0.8 * s.rope)
  s.leanZ = s.leanBaseZ * (1 + 0.8 * s.rope)
  s.svPhase = (((s.svPhase - s.svOmega * dt) % TAU) + TAU) % TAU
  const d = Math.hypot(s.x, s.z)
  if (s.phase <= PH_ROPEOUT && d < s.nearestD) {
    s.nearestD = d
    s.nearestX = s.x
    s.nearestZ = s.z
  }
}

// Funnel axis at height y. Mirrored exactly by centerAt() in Tornado.tsx's particle shader.
export function centerAt(s: Storm, y: number, out: { x: number; z: number }) {
  const h = Math.min(1.2, Math.max(0, y / s.H))
  const p = Math.pow(h, 1.4)
  out.x = s.x + s.leanX * p + s.wobble * h * Math.sin(y * 0.021 + s.time * 0.9 + s.ph1)
  out.z = s.z + s.leanZ * p + s.wobble * h * Math.sin(y * 0.017 - s.time * 0.7 + s.ph2)
}

const _c = { x: 0, z: 0 }

// Modified Rankine vortex with surface inflow, core updraft and outflow aloft, carried along
// by the storm's motion. Rotation is cyclonic (counterclockwise seen from above).
export function windAt(s: Storm, x: number, y: number, z: number, out: Float32Array) {
  const I = s.intensity
  if (I <= 0.001) {
    out[0] = out[1] = out[2] = 0
    return
  }
  centerAt(s, y, _c)
  const dx = x - _c.x
  const dz = z - _c.z
  const r = Math.sqrt(dx * dx + dz * dz) + 1e-3
  const Rc = s.Rc
  const rr = r / Rc
  let vt = rr < 1 ? s.Vmax * rr : s.Vmax * Math.pow(rr, -0.62)
  const taper = Math.exp(-(rr * rr) / 36) // far field dies out within ~6 core radii
  vt *= taper * I * (0.8 + 0.2 * smoothstep(0, 12, y))
  const inflow = 0.55 * Math.exp(-Math.max(0, y) / 22) * smoothstep(0.4, 1.3, rr)
  const outflow = 0.6 * smoothstep(0.55 * s.H, 0.95 * s.H, y)
  const vr = vt * (outflow - inflow)
  const updraft = 0.72 * s.Vmax * I * Math.exp(-(rr * rr) / 1.96) * (1 - smoothstep(0.7 * s.H, 1.05 * s.H, y)) * (0.45 + 0.55 * smoothstep(0, 15, y))
  const carry = I * Math.exp(-(rr * rr) / 9)
  const ix = 1 / r
  out[0] = (vt * dz + vr * dx) * ix + s.vx * carry
  out[1] = updraft
  out[2] = (-vt * dx + vr * dz) * ix + s.vz * carry
}

// Ground-level horizontal reach of damaging wind, for cheap culling
const influenceRadius = (s: Storm) => s.Rc * 6.5 + 30

// ---------------------------------------------------------------------------------------------
// Damage and debris

const _w = new Float32Array(3)
const _sv = new Float64Array(16)

function spawnDust(w: World, x: number, y: number, z: number, vx: number, vy: number, vz: number, size: number, life: number) {
  const j = w.dustCursor
  w.dustCursor = (j + 1) % MAX_DUST
  w.dustPos[j * 3] = x
  w.dustPos[j * 3 + 1] = y
  w.dustPos[j * 3 + 2] = z
  w.dustVel[j * 3] = vx
  w.dustVel[j * 3 + 1] = vy
  w.dustVel[j * 3 + 2] = vz
  w.dustAge[j] = 0
  w.dustLife[j] = life
  w.dustSize[j] = size
  w.dustAlpha[j] = 0
}

// Fragment types: [sx, sy, sz, mass]
// (masses give terminal velocities of roughly 10-25 m/s, so the core's updraft can loft them)
const FRAG_SHINGLE = [0.9, 0.05, 0.4, 1.5]
const FRAG_PLANK = [2.4, 0.1, 0.22, 5]
const FRAG_SHEET = [1.3, 0.05, 1.0, 7]
const FRAG_CHUNK = [0.6, 0.4, 0.45, 18]
const FRAG_LEAVES = [0.9, 0.6, 0.8, 0.8]
const FRAG_GLASS = [0.5, 0.03, 0.4, 1.5]

function spawnFragment(w: World, rand: Rand, x: number, y: number, z: number, vx: number, vy: number, vz: number, color: number, type: readonly number[]) {
  // Prefer a slot that is free or at rest; fall back to the oldest flier
  let slot = -1
  for (let n = 0; n < 48; n++) {
    const j = (w.fragCursor + n) % FRAG_POOL
    if (w.state[MAX_STRUCT_BODIES + j] !== ST_FLYING) {
      slot = j
      break
    }
  }
  if (slot < 0) slot = w.fragCursor
  w.fragCursor = (slot + 1) % FRAG_POOL
  const i = MAX_STRUCT_BODIES + slot
  const sc = range(rand, 0.75, 1.25)
  const sx = type[0] * sc
  const sy = type[1] * sc
  const sz = type[2] * sc
  w.px[i] = x
  w.py[i] = Math.max(y, 0.3)
  w.pz[i] = z
  w.vx[i] = vx
  w.vy[i] = vy
  w.vz[i] = vz
  // random orientation
  const u1 = rand()
  const u2 = rand() * TAU
  const u3 = rand() * TAU
  const a = Math.sqrt(1 - u1)
  const b = Math.sqrt(u1)
  w.qx[i] = a * Math.sin(u2)
  w.qy[i] = a * Math.cos(u2)
  w.qz[i] = b * Math.sin(u3)
  w.qw[i] = b * Math.cos(u3)
  w.wx[i] = (rand() - 0.5) * 12
  w.wy[i] = (rand() - 0.5) * 12
  w.wz[i] = (rand() - 0.5) * 12
  w.hx[i] = sx / 2
  w.hy[i] = sy / 2
  w.hz[i] = sz / 2
  const area = (sx * sy + sy * sz + sx * sz) / 2
  w.k[i] = (AIR * area) / (type[3] * sc * sc)
  w.wake[i] = Math.max(16, Math.sqrt((GROUND_FRICTION * GRAVITY) / w.k[i]) * 1.15)
  const u = rand() * 2 - 1
  const ang = rand() * TAU
  const rr = Math.sqrt(1 - u * u)
  w.sax[i] = rr * Math.cos(ang)
  w.say[i] = u
  w.saz[i] = rr * Math.sin(ang)
  w.rest[i] = 0
  w.state[i] = ST_FLYING
  w.color[i] = vary(color, rand, 0.12)
  w.dirty[i] = 1
  w.colorDirty[i] = 1
  w.fragmentsSpawned++
}

function fragmentTypeFor(kind: number, rand: Rand): readonly number[] {
  switch (kind) {
    case K_ROOF: return rand() < 0.6 ? FRAG_SHINGLE : FRAG_SHEET
    case K_WALL: return rand() < 0.5 ? FRAG_PLANK : rand() < 0.5 ? FRAG_SHEET : FRAG_CHUNK
    case K_CROWN: return rand() < 0.8 ? FRAG_LEAVES : FRAG_PLANK
    case K_GLASS: return FRAG_GLASS
    case K_HEAVY: return rand() < 0.5 ? FRAG_SHEET : FRAG_CHUNK
    case K_CAR: return rand() < 0.5 ? FRAG_GLASS : FRAG_CHUNK
    default: return rand() < 0.5 ? FRAG_PLANK : FRAG_CHUNK
  }
}

function detach(w: World, s: Storm, rand: Rand, i: number, depth = 0) {
  if (w.state[i] !== ST_ATTACHED) return
  w.state[i] = ST_FLYING
  w.detached++
  windAt(s, w.px[i], w.py[i], w.pz[i], _w)
  const kick = 0.25 + 0.15 * rand()
  w.vx[i] = _w[0] * kick + (rand() - 0.5) * 2
  w.vy[i] = _w[1] * kick + 1.5 + rand() * 3
  w.vz[i] = _w[2] * kick + (rand() - 0.5) * 2
  w.wx[i] = (rand() - 0.5) * 3
  w.wy[i] = (rand() - 0.5) * 3
  w.wz[i] = (rand() - 0.5) * 3
  w.rest[i] = 0
  w.shake[i] = 0
  w.dirty[i] = 1
  const kind = w.kind[i]
  // Splinters and shards burst off as it goes
  const face = Math.max(w.hx[i] * w.hy[i], w.hy[i] * w.hz[i], w.hx[i] * w.hz[i]) * 4
  const nFrag = kind === K_FENCE || kind === K_POLE || kind === K_TRUNK ? 1 : Math.min(6, 1 + Math.floor(face / 3))
  for (let n = 0; n < nFrag; n++) {
    spawnFragment(
      w, rand,
      w.px[i] + (rand() - 0.5) * w.hx[i] * 2, w.py[i] + (rand() - 0.5) * w.hy[i] * 2, w.pz[i] + (rand() - 0.5) * w.hz[i] * 2,
      _w[0] * 0.35 + (rand() - 0.5) * 6, _w[1] * 0.35 + rand() * 4, _w[2] * 0.35 + (rand() - 0.5) * 6,
      kind === K_GLASS ? GLASS : w.color[i], fragmentTypeFor(kind, rand),
    )
  }
  for (let n = 0; n < 2; n++) {
    spawnDust(w, w.px[i], Math.max(1, w.py[i] - w.hy[i]), w.pz[i], _w[0] * 0.3, 2, _w[2] * 0.3, range(rand, 5, 10), range(rand, 2, 4))
  }
  // Anything this piece was holding goes with it
  if (depth < 6) {
    const lo = Math.max(0, i - 64)
    const hi = Math.min(w.count, i + 64)
    for (let j = lo; j < hi; j++) if (w.parent[j] === i) detach(w, s, rand, j, depth + 1)
  }
  // Losing half its walls drops whatever roof is left
  const si = w.structure[i]
  if (si >= 0 && kind === K_WALL) {
    const st = w.structures[si]
    st.wallsLeft--
    if (!st.collapsed && st.wallsLeft < st.walls * 0.55) {
      st.collapsed = true
      for (let j = st.first; j < st.last; j++) {
        const kj = w.kind[j]
        if (kj === K_ROOF || kj === K_TRIM) detach(w, s, rand, j, depth + 1)
      }
    }
  }
}

// Bodies resting on the ground tip over onto their broadest face
function settleRotation(w: World, i: number, rate: number) {
  const x = w.qx[i], y = w.qy[i], z = w.qz[i], q = w.qw[i]
  // World-y component of each local axis (row 1 of the rotation matrix)
  const r0 = 2 * (x * y + z * q)
  const r1 = 1 - 2 * (x * x + z * z)
  const r2 = 2 * (y * z - x * q)
  // Prefer the axis of the thinnest dimension pointing up (lying flat)
  const ex = Math.abs(r0) / (w.hx[i] + 0.05)
  const ey = Math.abs(r1) / (w.hy[i] + 0.05)
  const ez = Math.abs(r2) / (w.hz[i] + 0.05)
  let ax: number, ay: number, az: number
  if (ex >= ey && ex >= ez) {
    ax = 1 - 2 * (y * y + z * z); ay = r0; az = 2 * (x * z - y * q)
  } else if (ey >= ez) {
    ax = 2 * (x * y - z * q); ay = r1; az = 2 * (y * z + x * q)
  } else {
    ax = 2 * (x * z + y * q); ay = r2; az = 1 - 2 * (x * x + y * y)
  }
  const sgn = ay >= 0 ? 1 : -1
  // rotation taking (ax, ay, az) toward (0, sgn, 0): axis = a x up
  let cx = -az * sgn
  let cz = ax * sgn
  const len = Math.sqrt(cx * cx + cz * cz)
  if (len < 1e-5) return
  cx /= len
  cz /= len
  const angle = Math.acos(Math.min(1, Math.abs(ay))) * Math.min(1, rate)
  const sh = Math.sin(angle / 2)
  const dx = cx * sh, dz = cz * sh, dw = Math.cos(angle / 2)
  // q = d * q
  const nx = dw * x + dx * q + 0 * z - dz * y
  const ny = dw * y - dx * z + 0 * q + dz * x
  const nz = dw * z + dx * y - 0 * x + dz * q
  const nw = dw * q - dx * x - 0 * y - dz * z
  const n = 1 / Math.sqrt(nx * nx + ny * ny + nz * nz + nw * nw)
  w.qx[i] = nx * n
  w.qy[i] = ny * n
  w.qz[i] = nz * n
  w.qw[i] = nw * n
}

function integrate(w: World, s: Storm, rand: Rand, i: number, dt: number) {
  windAt(s, w.px[i], w.py[i], w.pz[i], _w)
  let vx = w.vx[i], vy = w.vy[i], vz = w.vz[i]
  const rx = vx - _w[0], ry = vy - _w[1], rz = vz - _w[2]
  const rel = Math.sqrt(rx * rx + ry * ry + rz * rz)
  const k = w.k[i]
  // Quadratic drag, solved semi-implicitly so light pieces in 90 m/s wind stay stable
  const f = 1 / (1 + k * rel * dt)
  vx = _w[0] + rx * f
  vy = _w[1] + ry * f - GRAVITY * dt
  vz = _w[2] + rz * f
  const sp = Math.sqrt(vx * vx + vy * vy + vz * vz)
  if (sp > MAX_SPEED) {
    const c = MAX_SPEED / sp
    vx *= c; vy *= c; vz *= c
  }
  let px = w.px[i] + vx * dt
  let py = w.py[i] + vy * dt
  const pz = w.pz[i] + vz * dt
  // Tumble: the airflow spins pieces up about their own preferred axis
  const size = Math.max(w.hx[i], w.hy[i], w.hz[i])
  const spin = Math.min(rel, 70) * 0.35 / (size + 0.3)
  const blend = Math.min(1, k * rel * dt * 2 + 0.002)
  let wx = w.wx[i] + (w.sax[i] * spin - w.wx[i]) * blend
  let wy = w.wy[i] + (w.say[i] * spin - w.wy[i]) * blend
  let wz = w.wz[i] + (w.saz[i] * spin - w.wz[i]) * blend
  // Integrate orientation: q += 0.5 * (w, 0) * q * dt
  let qx = w.qx[i], qy = w.qy[i], qz = w.qz[i], qw = w.qw[i]
  const h = 0.5 * dt
  const dqx = (wx * qw + wy * qz - wz * qy) * h
  const dqy = (-wx * qz + wy * qw + wz * qx) * h
  const dqz = (wx * qy - wy * qx + wz * qw) * h
  const dqw = (-wx * qx - wy * qy - wz * qz) * h
  qx += dqx; qy += dqy; qz += dqz; qw += dqw
  const qn = 1 / Math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
  w.qx[i] = qx * qn
  w.qy[i] = qy * qn
  w.qz[i] = qz * qn
  w.qw[i] = qw * qn
  // Ground contact against the oriented box's lowest corner
  const x = w.qx[i], y = w.qy[i], z = w.qz[i], q = w.qw[i]
  const ext = Math.abs(2 * (x * y + z * q)) * w.hx[i] + Math.abs(1 - 2 * (x * x + z * z)) * w.hy[i] + Math.abs(2 * (y * z - x * q)) * w.hz[i]
  let grounded = false
  if (py - ext <= 0) {
    grounded = true
    py = ext
    if (vy < 0) {
      if (vy < -7 && k < 0.02) {
        // heavy impact: a puff of dirt
        spawnDust(w, px, 0.5, pz, vx * 0.2, 1.5, vz * 0.2, range(rand, 4, 8), range(rand, 1.5, 3))
      }
      vy = vy < -2.5 ? -vy * RESTITUTION : 0
    }
    const hv = Math.sqrt(vx * vx + vz * vz)
    const dec = GROUND_FRICTION * GRAVITY * dt
    if (hv <= dec) {
      vx = 0
      vz = 0
    } else {
      vx *= 1 - dec / hv
      vz *= 1 - dec / hv
    }
    const damp = hv < 3 ? 0.8 : 0.93
    wx *= damp
    wy *= damp
    wz *= damp
    settleRotation(w, i, (hv < 3 ? 6 : 1.5) * dt)
  }
  if (py > s.H * 1.3) {
    py = s.H * 1.3
    if (vy > 0) vy = 0
  }
  px = Math.max(-3000, Math.min(3000, px))
  w.px[i] = px
  w.py[i] = py
  w.pz[i] = Math.max(-3000, Math.min(3000, pz))
  w.vx[i] = vx
  w.vy[i] = vy
  w.vz[i] = vz
  w.wx[i] = wx
  w.wy[i] = wy
  w.wz[i] = wz
  w.dirty[i] = 1
  if (grounded && vx * vx + vy * vy + vz * vz < 0.3 && wx * wx + wy * wy + wz * wz < 0.4) {
    w.rest[i] += dt
    if (w.rest[i] > 0.7) {
      w.state[i] = ST_SLEEPING
      w.vx[i] = w.vy[i] = w.vz[i] = 0
      w.wx[i] = w.wy[i] = w.wz[i] = 0
    }
  } else {
    w.rest[i] = 0
  }
  // Safety net: a numerical blow-up parks the body rather than poisoning the frame
  if (!Number.isFinite(w.px[i] + w.py[i] + w.pz[i] + w.qx[i] + w.qy[i] + w.qz[i] + w.qw[i])) {
    w.px[i] = s.x
    w.py[i] = 1
    w.pz[i] = s.z
    w.qx[i] = w.qy[i] = w.qz[i] = 0
    w.qw[i] = 1
    w.vx[i] = w.vy[i] = w.vz[i] = 0
    w.wx[i] = w.wy[i] = w.wz[i] = 0
    w.state[i] = ST_SLEEPING
  }
}

export function stepWorld(w: World, s: Storm, dt: number, rand: Rand) {
  w.stepIndex++
  const reach = influenceRadius(s)
  const reach2 = reach * reach
  const active = s.intensity > 0.001
  // Ground center is a fine cull point; the lean only matters for lofted debris
  const cx = s.x
  const cz = s.z
  for (let i = 0; i < MAX_BODIES; i++) {
    if (i === w.count) i = MAX_STRUCT_BODIES
    const st = w.state[i]
    if (st === ST_EMPTY || st === ST_STATIC) continue
    if (st === ST_FLYING) {
      integrate(w, s, rand, i, dt)
      continue
    }
    if (!active) {
      if (w.shake[i] > 0) {
        w.shake[i] = 0
        w.dirty[i] = 1
      }
      continue
    }
    const dx = w.px[i] - cx
    const dz = w.pz[i] - cz
    const d2 = dx * dx + dz * dz
    if (d2 > reach2) {
      if (w.shake[i] > 0) {
        w.shake[i] = 0
        w.dirty[i] = 1
      }
      continue
    }
    if (st === ST_SLEEPING) {
      // Resting debris only needs an occasional look
      if ((w.stepIndex + i) % 4 !== 0) continue
      windAt(s, w.px[i], w.py[i] + w.hy[i], w.pz[i], _w)
      const sp = Math.hypot(_w[0], _w[1], _w[2])
      if (sp > w.wake[i]) {
        w.state[i] = ST_FLYING
        w.rest[i] = 0
        w.vy[i] = 1
      }
      continue
    }
    // Attached: accumulate damage once the wind exceeds what the piece can take
    windAt(s, w.px[i], w.py[i], w.pz[i], _w)
    const sp = Math.hypot(_w[0], _w[1], _w[2])
    const T = w.threshold[i]
    const load = sp / T
    const shake = load > 0.75 ? Math.min(1, (load - 0.75) * 3) : 0
    if (shake > 0 || w.shake[i] > 0) w.dirty[i] = 1
    w.shake[i] = shake
    if (load > 0.8 && w.shed[i] > 0 && rand() < (load - 0.8) * 5 * dt) {
      // Shingles and siding peel off before the piece itself lets go
      w.shed[i]--
      spawnFragment(
        w, rand,
        w.px[i] + (rand() - 0.5) * w.hx[i] * 1.6, w.py[i] + w.hy[i], w.pz[i] + (rand() - 0.5) * w.hz[i] * 1.6,
        _w[0] * 0.4, _w[1] * 0.4 + 3, _w[2] * 0.4,
        w.color[i], w.kind[i] === K_ROOF ? FRAG_SHINGLE : FRAG_PLANK,
      )
    }
    if (load > 1) {
      w.damage[i] += (load - 1) * DAMAGE_RATE * dt
      if (w.damage[i] >= 1) detach(w, s, rand, i)
    }
  }
}

// Dust: point masses that follow the wind closely and fade
export function stepDust(w: World, s: Storm, dt: number, rand: Rand) {
  // Ground dust and grass torn up in the inflow ring, plus the suction vortices
  if (s.debris > 0.05 && s.intensity > 0.1) {
    const n = Math.floor(70 * s.debris * dt + rand())
    for (let i = 0; i < n * 2; i++) {
      const a = rand() * TAU
      const r = s.Rc * range(rand, 0.5, 2.6)
      spawnDust(w, s.x + Math.cos(a) * r, range(rand, 0.3, 3), s.z + Math.sin(a) * r, 0, 2, 0, range(rand, 4, 9), range(rand, 2.5, 5))
    }
  }
  for (let j = 0; j < MAX_DUST; j++) {
    if (w.dustSize[j] <= 0) continue
    const age = (w.dustAge[j] += dt)
    const life = w.dustLife[j]
    if (age >= life) {
      w.dustSize[j] = 0
      w.dustAlpha[j] = 0
      continue
    }
    const o = j * 3
    windAt(s, w.dustPos[o], w.dustPos[o + 1], w.dustPos[o + 2], _w)
    const blend = Math.min(1, 1.6 * dt)
    w.dustVel[o] += (_w[0] - w.dustVel[o]) * blend
    w.dustVel[o + 1] += (_w[1] * 0.8 - w.dustVel[o + 1]) * blend
    w.dustVel[o + 2] += (_w[2] - w.dustVel[o + 2]) * blend
    w.dustPos[o] += w.dustVel[o] * dt
    w.dustPos[o + 1] = Math.max(0.5, w.dustPos[o + 1] + w.dustVel[o + 1] * dt)
    w.dustPos[o + 2] += w.dustVel[o + 2] * dt
    const a = age / life
    w.dustAlpha[j] = Math.min(1, age / 0.5) * (1 - a) * (1 - a)
    w.dustSize[j] += dt * 2.2 // puffs spread as they age
  }
}

// Paint where the ground has been scoured: the core's swath plus the suction vortices'
// cycloidal marks
export function paintScour(w: World, s: Storm, dt: number) {
  if (s.intensity < 0.15) return
  const reach = s.Rc * 2.4
  const texel = SCOUR_EXTENT / SCOUR_SIZE
  const toTex = (v: number) => (v + SCOUR_EXTENT / 2) / texel
  const x0 = Math.max(1, Math.floor(toTex(s.x - reach)))
  const x1 = Math.min(SCOUR_SIZE - 2, Math.ceil(toTex(s.x + reach)))
  const z0 = Math.max(1, Math.floor(toTex(s.z - reach)))
  const z1 = Math.min(SCOUR_SIZE - 2, Math.ceil(toTex(s.z + reach)))
  if (x0 > x1 || z0 > z1) return
  const sv = _sv
  const nsv = Math.min(s.subCount, 8)
  for (let k = 0; k < nsv; k++) {
    const a = s.svPhase + (k * TAU) / s.subCount
    sv[k * 2] = s.x + Math.cos(a) * s.Rc * 0.8
    sv[k * 2 + 1] = s.z + Math.sin(a) * s.Rc * 0.8
  }
  for (let tz = z0; tz <= z1; tz++) {
    const z = (tz + 0.5) * texel - SCOUR_EXTENT / 2
    for (let tx = x0; tx <= x1; tx++) {
      const x = (tx + 0.5) * texel - SCOUR_EXTENT / 2
      windAt(s, x, 0.5, z, _w)
      let amt = (Math.hypot(_w[0], _w[2]) - 38) / 45
      for (let k = 0; k < nsv * 2; k += 2) {
        const d = Math.hypot(x - sv[k], z - sv[k + 1])
        if (d < 8) amt += (1 - d / 8) * 1.2 * s.intensity
      }
      if (amt <= 0) continue
      const idx = tz * SCOUR_SIZE + tx
      w.scour[idx] = Math.min(255, w.scour[idx] + Math.ceil(amt * dt * 255 * 1.5))
    }
  }
}

// Column-major 4x4 instance matrix for body i (attached pieces under load shiver a little)
export function writeInstanceMatrix(w: World, i: number, out: Float32Array, o: number, time: number) {
  if (w.state[i] === ST_EMPTY) {
    out.fill(0, o, o + 16)
    return
  }
  let x = w.qx[i], y = w.qy[i], z = w.qz[i], q = w.qw[i]
  const sh = w.shake[i]
  if (sh > 0 && w.state[i] === ST_ATTACHED) {
    const a = sh * 0.035 * Math.sin(time * 31 + i * 1.7)
    const s = Math.sin(a / 2)
    const c = Math.cos(a / 2)
    // local rotation about the piece's own spin axis (random per piece)
    const bx = w.sax[i] * s, by = w.say[i] * s, bz = w.saz[i] * s
    const nx = q * bx + x * c + y * bz - z * by
    const ny = q * by - x * bz + y * c + z * bx
    const nz = q * bz + x * by - y * bx + z * c
    const nw = q * c - x * bx - y * by - z * bz
    x = nx; y = ny; z = nz; q = nw
  }
  const sx = w.hx[i] * 2, sy = w.hy[i] * 2, sz = w.hz[i] * 2
  const x2 = x + x, y2 = y + y, z2 = z + z
  const xx = x * x2, xy = x * y2, xz = x * z2
  const yy = y * y2, yz = y * z2, zz = z * z2
  const wx = q * x2, wy = q * y2, wz = q * z2
  out[o] = (1 - (yy + zz)) * sx
  out[o + 1] = (xy + wz) * sx
  out[o + 2] = (xz - wy) * sx
  out[o + 3] = 0
  out[o + 4] = (xy - wz) * sy
  out[o + 5] = (1 - (xx + zz)) * sy
  out[o + 6] = (yz + wx) * sy
  out[o + 7] = 0
  out[o + 8] = (xz + wy) * sz
  out[o + 9] = (yz - wx) * sz
  out[o + 10] = (1 - (xx + yy)) * sz
  out[o + 11] = 0
  out[o + 12] = w.px[i]
  out[o + 13] = w.py[i]
  out[o + 14] = w.pz[i]
  out[o + 15] = 1
}

// ---------------------------------------------------------------------------------------------
// Storm-chaser camera: holds a safe distance off the tornado's flank, drifting around it, and
// swings back over the damage path once it has roped out

export interface ChaseCam {
  px: number
  py: number
  pz: number
  tx: number
  ty: number
  tz: number
  side: number
  az: number
  azAmp: number
  azFreq: number
  azPhase: number
  dist: number
  height: number
  drone: boolean
}

export function createChaseCam(s: Storm, w: World, rand: Rand): ChaseCam {
  const drone = rand() < 0.25
  const cam: ChaseCam = {
    px: 0, py: 0, pz: 0, tx: 0, ty: 0, tz: 0,
    side: rand() < 0.5 ? -1 : 1,
    // 60deg = ahead of the storm watching it come, 120deg = following behind it
    az: range(rand, 60, 120) * (Math.PI / 180),
    azAmp: range(rand, 0.15, 0.35),
    azFreq: range(rand, 0.025, 0.05),
    azPhase: rand() * TAU,
    dist: Math.max(125 + 3.4 * s.RcBase, 1.12 * s.H) * (drone ? 1.15 : 1),
    height: drone ? range(rand, 45, 75) : range(rand, 3, 9),
    drone,
  }
  updateChaseCam(cam, s, w, 0, true)
  return cam
}

export function cameraClearance(w: World, x: number, z: number) {
  let top = 0
  for (const o of w.obstacles) {
    const dx = o.x - x
    const dz = o.z - z
    const r = o.r + 9
    if (dx * dx + dz * dz < r * r) top = Math.max(top, o.top + 2.5)
  }
  return top
}

export function updateChaseCam(cam: ChaseCam, s: Storm, w: World, dt: number, snap = false) {
  const after = s.phase >= PH_AFTER ? smoothstep(0, 6, s.phaseTime + (s.phase === PH_DONE ? 9 : 0)) : 0
  const az = cam.az + cam.azAmp * Math.sin(s.time * cam.azFreq + cam.azPhase)
  const ang = s.heading + cam.side * az
  const dist = cam.dist
  const dx = s.x + Math.cos(ang) * dist
  const dz = s.z + Math.sin(ang) * dist
  const dy = Math.max(cam.height + after * 30, cameraClearance(w, dx, dz))
  // Aim at the lower third of the funnel, nudged back toward the damage path
  const back = 0.12 * dist
  let tx = s.x - Math.cos(s.heading) * back
  let tz = s.z - Math.sin(s.heading) * back
  let ty = (cam.drone ? 0.27 : 0.36) * s.H
  if (after > 0) {
    // the funnel is gone: look over the wreckage instead
    tx += (s.nearestX - tx) * after
    tz += (s.nearestZ - tz) * after
    ty += (4 - ty) * after
  }
  const kp = snap ? 1 : 1 - Math.exp(-dt / 2.2)
  const kt = snap ? 1 : 1 - Math.exp(-dt / 1.2)
  cam.px += (dx - cam.px) * kp
  cam.py += (dy - cam.py) * kp
  cam.pz += (dz - cam.pz) * kp
  cam.tx += (tx - cam.tx) * kt
  cam.ty += (ty - cam.ty) * kt
  cam.tz += (tz - cam.tz) * kt
}
