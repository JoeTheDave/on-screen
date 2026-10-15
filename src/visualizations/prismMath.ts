// Pure optics for the Prism visualization: real glass dispersion (Sellmeier), wavelength ->
// display color (CIE 1931 fit), 2D ray tracing with Snell + Fresnel + total internal reflection
// through extruded optical elements, and the scripted "experiments" that pose them over time.
//
// Everything is planar: every element is a vertical extrusion of a 2D outline and every beam is
// a vertical sheet of light from a slit (like a classroom ray-optics kit), so tracing the sheet's
// 2D cross-section on the table is exact, not an approximation.

// ---------- Glass ----------

export interface Glass {
  name: string
  // Sellmeier: n^2 = 1 + sum B_i L^2 / (L^2 - C_i), L in micrometres
  b: readonly number[]
  c: readonly number[]
}

export const BK7: Glass = {
  name: 'N-BK7',
  b: [1.03961212, 0.231792344, 1.01046945],
  c: [0.00600069867, 0.0200179144, 103.560653],
}
// Dense flint: Abbe number ~25.7, about 2.5x the dispersion of BK7
export const SF11: Glass = {
  name: 'SF11',
  b: [1.73759695, 0.313747346, 1.89878101],
  c: [0.013188707, 0.0623068142, 155.23629],
}
// Daimon & Masumura (2007), 20 C
export const WATER: Glass = {
  name: 'water',
  b: [5.684027565e-1, 1.726177391e-1, 2.086189578e-2, 1.130748688e-1],
  c: [5.101829712e-3, 1.821153936e-2, 2.620722293e-2, 1.069792721e1],
}
// Peter (1923)
export const DIAMOND: Glass = {
  name: 'diamond',
  b: [0.3306, 4.3356],
  c: [0.175 * 0.175, 0.106 * 0.106],
}

export function refractiveIndex(glass: Glass, nm: number): number {
  const l2 = (nm / 1000) * (nm / 1000)
  let n2 = 1
  for (let i = 0; i < glass.b.length; i++) n2 += (glass.b[i] * l2) / (l2 - glass.c[i])
  return Math.sqrt(n2)
}

export const SODIUM_D_NM = 589.3

// ---------- Color ----------

const lobe = (x: number, mu: number, s1: number, s2: number) => {
  const t = (x - mu) / (x < mu ? s1 : s2)
  return Math.exp(-0.5 * t * t)
}

// CIE 1931 2-degree colour matching functions, multi-lobe Gaussian fit (Wyman, Sloan & Shirley 2013)
export function cieXYZ(nm: number): [number, number, number] {
  const x = 1.056 * lobe(nm, 599.8, 37.9, 31.0) + 0.362 * lobe(nm, 442.0, 16.0, 26.7) - 0.065 * lobe(nm, 501.1, 20.4, 26.2)
  const y = 0.821 * lobe(nm, 568.8, 46.9, 40.5) + 0.286 * lobe(nm, 530.9, 16.3, 31.1)
  const z = 1.217 * lobe(nm, 437.0, 11.8, 36.0) + 0.681 * lobe(nm, 459.0, 26.0, 13.8)
  return [x, y, z]
}

export function xyzToLinearSRGB(x: number, y: number, z: number): [number, number, number] {
  return [
    3.2406 * x - 1.5372 * y - 0.4986 * z,
    -0.9689 * x + 1.8758 * y + 0.0415 * z,
    0.0557 * x - 0.204 * y + 1.057 * z,
  ]
}

export interface Spectrum {
  count: number
  nm: Float64Array
  // Linear sRGB per sample; each channel sums to exactly 1 over the samples, so the samples
  // added back together are white and a white ray can be drawn as (1,1,1)
  colors: Float32Array
}

export const SPECTRUM_MIN_NM = 405
export const SPECTRUM_MAX_NM = 695
// Spectral colors fall in brightness steeply toward both ends; lift them partway (on top of the
// CIE colour) so violet and deep red still read on a display, then white-balance the set
const END_LIFT = 0.12

export function makeSpectrum(count: number): Spectrum {
  const nm = new Float64Array(count)
  const colors = new Float32Array(count * 3)
  const sum = [0, 0, 0]
  for (let i = 0; i < count; i++) {
    const l = SPECTRUM_MIN_NM + ((i + 0.5) / count) * (SPECTRUM_MAX_NM - SPECTRUM_MIN_NM)
    nm[i] = l
    const [x, y, z] = cieXYZ(l)
    // Out-of-gamut spectral colors: clip the negative lobes (keeps the hue fully saturated)
    const rgb = xyzToLinearSRGB(x, y, z).map(v => Math.max(0, v))
    const lift = 1 / Math.sqrt(y + END_LIFT)
    for (let k = 0; k < 3; k++) {
      colors[i * 3 + k] = rgb[k] * lift
      sum[k] += rgb[k] * lift
    }
  }
  for (let i = 0; i < count; i++) for (let k = 0; k < 3; k++) colors[i * 3 + k] /= sum[k]
  return { count, nm, colors }
}

// ---------- Shapes ----------

export type Edge =
  | { kind: 'line'; ax: number; ay: number; ex: number; ey: number; nx: number; ny: number }
  // Arc of a circle: the part within half-aperture `a` of the axis u (vertex = c + r u).
  // sign +1: convex (outward normal points away from c); -1: concave
  | { kind: 'arc'; cx: number; cy: number; r: number; ux: number; uy: number; a: number; sign: number }
  | { kind: 'circle'; cx: number; cy: number; r: number }

export interface Shape {
  edges: Edge[]
  outline: number[] // closed CCW outline [x0, y0, x1, y1, ...] for meshing
  boundR: number // bounding radius about the local origin
}

const lineEdge = (ax: number, ay: number, bx: number, by: number): Edge => {
  const ex = bx - ax
  const ey = by - ay
  const len = Math.hypot(ex, ey)
  // CCW outline: outward normal is the right-hand perpendicular of the edge direction
  return { kind: 'line', ax, ay, ex, ey, nx: ey / len, ny: -ex / len }
}

const boundOf = (outline: number[]) => {
  let r = 0
  for (let i = 0; i < outline.length; i += 2) r = Math.max(r, Math.hypot(outline[i], outline[i + 1]))
  return r * 1.001 + 1e-6
}

// Convex or concave polygon from CCW points
export function polygonShape(points: number[]): Shape {
  const edges: Edge[] = []
  const n = points.length / 2
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n
    edges.push(lineEdge(points[i * 2], points[i * 2 + 1], points[j * 2], points[j * 2 + 1]))
  }
  return { edges, outline: points.slice(), boundR: boundOf(points) }
}

// Equilateral (apex angle 60 deg by default) prism, centroid at the origin, apex toward +y
export function prismShape(side: number): Shape {
  const h = (side * Math.sqrt(3)) / 2
  return polygonShape([-side / 2, -h / 3, side / 2, -h / 3, 0, (2 * h) / 3])
}

export function circleShape(r: number, segments = 96): Shape {
  const outline: number[] = []
  for (let i = 0; i < segments; i++) {
    const a = (i / segments) * Math.PI * 2
    outline.push(r * Math.cos(a), r * Math.sin(a))
  }
  return { edges: [{ kind: 'circle', cx: 0, cy: 0, r }], outline, boundR: r * 1.001 + 1e-6 }
}

// Symmetric cylindrical lens, optical axis along x, half-aperture a (in y).
// Convex: edge thickness `thickness`; concave: centre thickness `thickness`.
export function lensShape(radius: number, a: number, thickness: number, convex: boolean, arcSegments = 40): Shape {
  const sag = radius - Math.sqrt(radius * radius - a * a)
  const half = convex ? thickness / 2 : thickness / 2 + sag // |x| of the flat rims
  const vertex = convex ? thickness / 2 + sag : thickness / 2 // |x| of the surface on axis
  const edges: Edge[] = []
  // Right surface (vertex at +x) and left surface (vertex at -x)
  for (const side of [1, -1]) {
    const cx = convex ? side * (vertex - radius) : side * (vertex + radius)
    edges.push({ kind: 'arc', cx, cy: 0, r: radius, ux: convex ? side : -side, uy: 0, a, sign: convex ? 1 : -1 })
  }
  // Flat rims
  edges.push(lineEdge(half, a, -half, a))
  edges.push(lineEdge(-half, -a, half, -a))

  // CCW outline: bottom-right -> up the right surface -> top rim -> down the left surface
  const outline: number[] = []
  const surfaceX = (y: number, side: number) => {
    const s = radius - Math.sqrt(radius * radius - y * y)
    return side * (convex ? vertex - s : vertex + s)
  }
  for (let i = 0; i <= arcSegments; i++) {
    const y = -a + (2 * a * i) / arcSegments
    outline.push(surfaceX(y, 1), y)
  }
  for (let i = 0; i <= arcSegments; i++) {
    const y = a - (2 * a * i) / arcSegments
    outline.push(surfaceX(y, -1), y)
  }
  return { edges, outline, boundR: boundOf(outline) }
}

// Thin-ish lens focal length (thick-lens lensmaker's equation) for a symmetric biconvex lens
export function lensFocalLength(radius: number, a: number, edgeThickness: number, n: number): number {
  const sag = radius - Math.sqrt(radius * radius - a * a)
  const t = edgeThickness + 2 * sag
  const power = (n - 1) * (2 / radius) - ((n - 1) * (n - 1) * t) / (n * radius * radius)
  return 1 / power
}

// 2D cross-section of a round brilliant cut (table up), girdle width 1 before scaling
export function brilliantShape(scale: number): Shape {
  const p = [
    0, -0.46, // culet
    0.27, -0.215, // pavilion break
    0.5, -0.025, // girdle bottom
    0.5, 0.0, // girdle top
    0.41, 0.085, // crown break
    0.28, 0.16, // table edge
    -0.28, 0.16,
    -0.41, 0.085,
    -0.5, 0.0,
    -0.5, -0.025,
    -0.27, -0.215,
  ]
  // Re-centre vertically so the gem spins about its middle
  return polygonShape(p.map((v, i) => (i % 2 ? v + 0.15 : v) * scale))
}

// Random convex crystal: points on an ellipse-ish ring at sorted random angles
export function crystalShape(rng: () => number, radius: number): Shape {
  const count = 6 + Math.floor(rng() * 4)
  const angles: number[] = []
  for (let i = 0; i < count; i++) angles.push(((i + 0.2 + rng() * 0.6) / count) * Math.PI * 2)
  const stretch = 0.75 + rng() * 0.5
  const pts: number[] = []
  for (const a of angles) pts.push(radius * Math.cos(a) * stretch, (radius * Math.sin(a)) / stretch)
  return polygonShape(pts) // sorted angles on an ellipse: always convex, CCW
}

// ---------- Fresnel / Snell ----------

export interface Interface {
  reflectance: number // unpolarized power reflectance (1 = total internal reflection)
  cosT: number // cosine of the refraction angle (0 if TIR)
}

export function fresnel(cosI: number, n1: number, n2: number, out: Interface): Interface {
  const eta = n1 / n2
  const k = 1 - eta * eta * (1 - cosI * cosI)
  if (k <= 0) {
    out.reflectance = 1
    out.cosT = 0
    return out
  }
  const cosT = Math.sqrt(k)
  const rs = (n1 * cosI - n2 * cosT) / (n1 * cosI + n2 * cosT)
  const rp = (n1 * cosT - n2 * cosI) / (n1 * cosT + n2 * cosI)
  out.reflectance = 0.5 * (rs * rs + rp * rp)
  out.cosT = cosT
  return out
}

// ---------- Tracing ----------

export interface TraceElement {
  shape: Shape
  x: number
  y: number
  rotation: number
  scale: number
  n: Float64Array // index per spectrum sample
  nRef: number // index used for white (undispersed) light
}

interface Hit {
  t: number
  nx: number
  ny: number
  element: number
}

const T_EPS = 1e-7
const MIN_TRACE_SCALE = 0.05

// Nearest hit along a ray among `elements` (or only element `only` when >= 0). Returns false
// if nothing is hit; otherwise `hit` holds the distance, the element and its outward normal.
function intersect(elements: TraceElement[], ox: number, oy: number, dx: number, dy: number, only: number, hit: Hit): boolean {
  hit.t = Infinity
  hit.element = -1
  const first = only >= 0 ? only : 0
  const last = only >= 0 ? only + 1 : elements.length
  for (let e = first; e < last; e++) {
    const el = elements[e]
    // Elements shrinking away (or not yet grown) are skipped: the beam is off then anyway, and a
    // near-zero scale would blow the local-frame epsilons up
    if (el.scale < MIN_TRACE_SCALE) continue
    const c = Math.cos(el.rotation)
    const s = Math.sin(el.rotation)
    const inv = 1 / el.scale
    const rx = ox - el.x
    const ry = oy - el.y
    const lx = (rx * c + ry * s) * inv
    const ly = (-rx * s + ry * c) * inv
    const ldx = dx * c + dy * s
    const ldy = -dx * s + dy * c
    // Bounding circle reject
    const B = lx * ldx + ly * ldy
    const C = lx * lx + ly * ly - el.shape.boundR * el.shape.boundR
    if (C > 0 && (B > 0 || B * B - C < 0)) continue
    const limit = hit.t * inv // compare in local units
    let bestT = limit
    let bnx = 0
    let bny = 0
    for (const edge of el.shape.edges) {
      if (edge.kind === 'line') {
        const denom = ldx * edge.ey - ldy * edge.ex
        if (Math.abs(denom) < 1e-14) continue
        const qx = edge.ax - lx
        const qy = edge.ay - ly
        const t = (qx * edge.ey - qy * edge.ex) / denom
        if (t <= T_EPS || t >= bestT) continue
        const u = (qx * ldy - qy * ldx) / denom
        if (u < 0 || u > 1) continue
        bestT = t
        bnx = edge.nx
        bny = edge.ny
      } else {
        const fx = lx - edge.cx
        const fy = ly - edge.cy
        const b = fx * ldx + fy * ldy
        const cc = fx * fx + fy * fy - edge.r * edge.r
        const disc = b * b - cc
        if (disc < 0) continue
        const sq = Math.sqrt(disc)
        for (let k = 0; k < 2; k++) {
          const t = k === 0 ? -b - sq : -b + sq
          if (t <= T_EPS || t >= bestT) continue
          const qx = fx + t * ldx
          const qy = fy + t * ldy
          let sign = 1
          if (edge.kind === 'arc') {
            if (qx * edge.ux + qy * edge.uy <= 0) continue
            if (Math.abs(edge.ux * qy - edge.uy * qx) > edge.a) continue
            sign = edge.sign
          }
          bestT = t
          bnx = (sign * qx) / edge.r
          bny = (sign * qy) / edge.r
        }
      }
    }
    if (bestT < limit) {
      hit.t = bestT * el.scale
      hit.element = e
      hit.nx = bnx * c - bny * s
      hit.ny = bnx * s + bny * c
    }
  }
  return hit.element >= 0
}

export interface TraceBuffers {
  maxSegments: number
  maxSpots: number
  segPos: Float32Array // x0, y0, x1, y1
  segCol: Float32Array // r, g, b
  segCount: number
  spotPos: Float32Array // x, y, tangent x, tangent y
  spotCol: Float32Array // r, g, b, half width
  spotCount: number
}

export function createTraceBuffers(maxSegments: number, maxSpots: number): TraceBuffers {
  return {
    maxSegments,
    maxSpots,
    segPos: new Float32Array(maxSegments * 4),
    segCol: new Float32Array(maxSegments * 3),
    segCount: 0,
    spotPos: new Float32Array(maxSpots * 4),
    spotCol: new Float32Array(maxSpots * 4),
    spotCount: 0,
  }
}

export interface TraceSettings {
  wallRadius: number
  maxDepth: number // surface interactions per ray path
  minWhiteEnergy: number
  minSpectralEnergy: number
  glint: number // fraction of light hitting a glass face that shows as a glow line on the face
  glintHalfWidth: number
  wallHalfWidth: number
}

export interface WhiteRay {
  x: number
  y: number
  dx: number
  dy: number
  energy: number
}

export interface TraceStats {
  emitted: number // total white energy launched
  toWall: number // energy (spectrally averaged) that reached the wall
  pruned: number // energy dropped by thresholds/depth/buffer limits
}

// Stack entry layout: ox, oy, dx, dy, energy, lambda index (-1 white), depth, inside element (-1 air)
const STRIDE = 8
const PUSH_OFFSET = 1e-6

export function createTracer(stackCapacity = 8192) {
  const stack = new Float64Array(stackCapacity * STRIDE)
  const hit: Hit = { t: 0, nx: 0, ny: 0, element: -1 }
  const fr: Interface = { reflectance: 0, cosT: 0 }

  return function trace(
    elements: TraceElement[],
    rays: WhiteRay[],
    spectrum: Spectrum,
    settings: TraceSettings,
    out: TraceBuffers,
    stats?: TraceStats,
  ) {
    out.segCount = 0
    out.spotCount = 0
    if (stats) {
      stats.emitted = 0
      stats.toWall = 0
      stats.pruned = 0
    }
    const N = spectrum.count
    const col = spectrum.colors
    let sp = 0

    const push = (ox: number, oy: number, dx: number, dy: number, energy: number, li: number, depth: number, inside: number) => {
      const minE = li < 0 ? settings.minWhiteEnergy : settings.minSpectralEnergy
      if (energy < minE || depth > settings.maxDepth || sp >= stackCapacity) {
        if (stats) stats.pruned += li < 0 ? energy : energy / N
        return
      }
      const o = sp * STRIDE
      stack[o] = ox + dx * PUSH_OFFSET
      stack[o + 1] = oy + dy * PUSH_OFFSET
      stack[o + 2] = dx
      stack[o + 3] = dy
      stack[o + 4] = energy
      stack[o + 5] = li
      stack[o + 6] = depth
      stack[o + 7] = inside
      sp++
    }

    const addSegment = (x0: number, y0: number, x1: number, y1: number, li: number, energy: number) => {
      if (out.segCount >= out.maxSegments) return
      const s = out.segCount++
      out.segPos[s * 4] = x0
      out.segPos[s * 4 + 1] = y0
      out.segPos[s * 4 + 2] = x1
      out.segPos[s * 4 + 3] = y1
      for (let k = 0; k < 3; k++) out.segCol[s * 3 + k] = li < 0 ? energy : col[li * 3 + k] * energy
    }

    const addSpot = (x: number, y: number, tx: number, ty: number, li: number, energy: number, halfWidth: number) => {
      if (out.spotCount >= out.maxSpots) return
      const s = out.spotCount++
      out.spotPos[s * 4] = x
      out.spotPos[s * 4 + 1] = y
      out.spotPos[s * 4 + 2] = tx
      out.spotPos[s * 4 + 3] = ty
      for (let k = 0; k < 3; k++) out.spotCol[s * 4 + k] = li < 0 ? energy : col[li * 3 + k] * energy
      out.spotCol[s * 4 + 3] = halfWidth
    }

    for (const ray of rays) {
      if (stats) stats.emitted += ray.energy
      push(ray.x, ray.y, ray.dx, ray.dy, ray.energy, -1, 0, -1)
      while (sp > 0) {
        sp--
        const o = sp * STRIDE
        const ox = stack[o]
        const oy = stack[o + 1]
        const dx = stack[o + 2]
        const dy = stack[o + 3]
        const energy = stack[o + 4]
        const li = stack[o + 5]
        const depth = stack[o + 6]
        const inside = stack[o + 7]
        const share = li < 0 ? 1 : 1 / N // fraction of white this ray's energy unit represents

        if (!intersect(elements, ox, oy, dx, dy, inside, hit)) {
          if (inside >= 0) {
            // Numerically escaped a closed shape: drop it rather than draw a bogus ray
            if (stats) stats.pruned += energy * share
            continue
          }
          // Out to the screen wall around the bench
          const b = ox * dx + oy * dy
          const c = ox * ox + oy * oy - settings.wallRadius * settings.wallRadius
          const t = -b + Math.sqrt(Math.max(0, b * b - c))
          const wx = ox + dx * t
          const wy = oy + dy * t
          addSegment(ox, oy, wx, wy, li, energy)
          const r = Math.hypot(wx, wy) || 1
          addSpot(wx, wy, -wy / r, wx / r, li, energy, settings.wallHalfWidth)
          if (stats) stats.toWall += energy * share
          continue
        }

        const hx = ox + dx * hit.t
        const hy = oy + dy * hit.t
        addSegment(ox, oy, hx, hy, li, energy)
        addSpot(hx, hy, -hit.ny, hit.nx, li, energy * settings.glint, settings.glintHalfWidth)

        const el = elements[hit.element]
        const dn = dx * hit.nx + dy * hit.ny
        const entering = dn < 0
        // Normal facing against the incoming ray
        const fnx = entering ? hit.nx : -hit.nx
        const fny = entering ? hit.ny : -hit.ny
        const cosI = Math.abs(dn)
        const rx = dx + 2 * cosI * fnx
        const ry = dy + 2 * cosI * fny
        const nextDepth = depth + 1
        const reflectedInside = entering ? -1 : hit.element
        const transmittedInside = entering ? hit.element : -1

        if (li < 0) {
          // White light from air: reflection keeps it white (angle is wavelength-independent and
          // the reflectance varies by under 1% across the visible), transmission disperses it
          fresnel(cosI, 1, el.nRef, fr)
          push(hx, hy, rx, ry, energy * fr.reflectance, -1, nextDepth, reflectedInside)
          for (let l = 0; l < N; l++) {
            const n2 = el.n[l]
            fresnel(cosI, 1, n2, fr)
            const eta = 1 / n2
            const k = eta * cosI - fr.cosT
            push(hx, hy, eta * dx + k * fnx, eta * dy + k * fny, energy * (1 - fr.reflectance), l, nextDepth, transmittedInside)
          }
          continue
        }

        const n = el.n[li]
        const n1 = entering ? 1 : n
        const n2 = entering ? n : 1
        fresnel(cosI, n1, n2, fr)
        push(hx, hy, rx, ry, energy * fr.reflectance, li, nextDepth, reflectedInside)
        if (fr.reflectance < 1) {
          const eta = n1 / n2
          const k = eta * cosI - fr.cosT
          push(hx, hy, eta * dx + k * fnx, eta * dy + k * fny, energy * (1 - fr.reflectance), li, nextDepth, transmittedInside)
        }
      }
    }
  }
}

// Follow ONE path through the elements for a single wavelength: at each surface take the
// transmitted ('T') or reflected ('R') branch per `path`. For verification.
export interface PathResult {
  ok: boolean // false if a requested T hit total internal reflection or the path ran out of surfaces
  x: number
  y: number
  dx: number
  dy: number
  energy: number
}

export function followPath(elements: TraceElement[], x: number, y: number, dx: number, dy: number, li: number, path: string): PathResult {
  const hit: Hit = { t: 0, nx: 0, ny: 0, element: -1 }
  const fr: Interface = { reflectance: 0, cosT: 0 }
  let inside = -1
  let energy = 1
  for (const step of path) {
    if (!intersect(elements, x, y, dx, dy, inside, hit)) return { ok: false, x, y, dx, dy, energy }
    x += dx * hit.t
    y += dy * hit.t
    const n = elements[hit.element].n[li]
    const dn = dx * hit.nx + dy * hit.ny
    const entering = dn < 0
    const fnx = entering ? hit.nx : -hit.nx
    const fny = entering ? hit.ny : -hit.ny
    const cosI = Math.abs(dn)
    const n1 = entering ? 1 : n
    const n2 = entering ? n : 1
    fresnel(cosI, n1, n2, fr)
    if (step === 'R') {
      dx += 2 * cosI * fnx
      dy += 2 * cosI * fny
      energy *= fr.reflectance
      inside = entering ? -1 : hit.element
    } else {
      if (fr.reflectance >= 1) return { ok: false, x, y, dx, dy, energy }
      const eta = n1 / n2
      const k = eta * cosI - fr.cosT
      dx = eta * dx + k * fnx
      dy = eta * dy + k * fny
      energy *= 1 - fr.reflectance
      inside = entering ? hit.element : -1
    }
    const len = Math.hypot(dx, dy)
    dx /= len
    dy /= len
    x += dx * PUSH_OFFSET
    y += dy * PUSH_OFFSET
  }
  return { ok: true, x, y, dx, dy, energy }
}

export function makeTraceElement(shape: Shape, glass: Glass, spectrum: Spectrum): TraceElement {
  const n = new Float64Array(spectrum.count)
  for (let i = 0; i < spectrum.count; i++) n[i] = refractiveIndex(glass, spectrum.nm[i])
  return { shape, x: 0, y: 0, rotation: 0, scale: 1, n, nRef: refractiveIndex(glass, 550) }
}

// ---------- Experiments ----------

export const WALL_RADIUS = 9

export type ExperimentKind = 'newton' | 'recombine' | 'lens' | 'raindrop' | 'crystal'
export const EXPERIMENT_KINDS: ExperimentKind[] = ['newton', 'recombine', 'lens', 'raindrop', 'crystal']

export interface Pose {
  x: number
  y: number
  rotation: number
}

// A ray box: a slit lamp at (x, y) shining along `angle`, emitting `rays` parallel white rays
// spread across `width`; the middle `gap` of the aperture can be masked off
export interface Source {
  x: number
  y: number
  angle: number
  width: number
  gap: number
  rays: number
  power: number
}

export interface ElementSpec {
  shape: Shape
  glass: Glass
}

export interface Experiment {
  kind: ExperimentKind
  elements: ElementSpec[]
  sources: Source[]
  poses: Pose[]
  spectrumCount: number
  maxDepth: number
  duration: number
  viewAzimuth: number // camera azimuth offset this layout looks best from
  update(t: number): void // pose elements and sources for experiment-local time t (seconds)
}

const TAU = Math.PI * 2
const DEG = Math.PI / 180

const range = (rng: () => number, a: number, b: number) => a + rng() * (b - a)

// White rays for a source, appended to `out` from index `start`; returns the new count
export function emitRays(src: Source, out: WhiteRay[], start: number): number {
  const dx = Math.cos(src.angle)
  const dy = Math.sin(src.angle)
  const px = -dy
  const py = dx
  let count = start
  const e = src.power / src.rays
  for (let i = 0; i < src.rays; i++) {
    let off: number
    if (src.gap > 0) {
      // Two half-apertures either side of the mask
      const perSide = Math.ceil(src.rays / 2)
      const j = i < perSide ? i : i - perSide
      const along = perSide > 1 ? j / (perSide - 1) : 0.5
      const mag = src.gap / 2 + along * (src.width / 2 - src.gap / 2)
      off = i < perSide ? -mag : mag
    } else {
      off = src.rays > 1 ? (i / (src.rays - 1) - 0.5) * src.width : 0
    }
    const ray = out[count] ?? (out[count] = { x: 0, y: 0, dx: 0, dy: 0, energy: 0 })
    ray.x = src.x + px * off
    ray.y = src.y + py * off
    ray.dx = dx
    ray.dy = dy
    ray.energy = e
    count++
  }
  return count
}

// Rotation that puts a 60-degree prism (apex +y) at minimum deviation for a beam travelling +x
export function prismMinDeviationRotation(n: number): number {
  return Math.PI / 6 - Math.asin(n * 0.5)
}

const pose = (x = 0, y = 0, rotation = 0): Pose => ({ x, y, rotation })

// Point on the left (entry) face of a prism shape, `f` from the base vertex toward the apex
const prismLeftFacePoint = (side: number, f: number, p: Pose) => {
  const h = (side * Math.sqrt(3)) / 2
  const lx = -side / 2 + f * (side / 2)
  const ly = -h / 3 + f * h
  const c = Math.cos(p.rotation)
  const s = Math.sin(p.rotation)
  return { x: p.x + lx * c - ly * s, y: p.y + lx * s + ly * c }
}

function newton(rng: () => number): Experiment {
  const glass = rng() < 0.7 ? SF11 : BK7
  const side = range(rng, 1.5, 1.8)
  const nD = refractiveIndex(glass, SODIUM_D_NM)
  const thetaMin = prismMinDeviationRotation(nD)
  const amp = range(rng, 16, 24) * DEG // big enough to tip into total internal reflection
  const period = range(rng, 24, 34)
  const phase = rng() * TAU
  const center = pose(range(rng, -1.0, -0.5), range(rng, 0.4, 0.9), thetaMin)
  const aim = prismLeftFacePoint(side, 0.45, center)
  const src: Source = { x: -4.6, y: aim.y, angle: 0, width: 0.06, gap: 0, rays: 2, power: 1 }
  return {
    kind: 'newton',
    elements: [{ shape: prismShape(side), glass }],
    sources: [src],
    poses: [center],
    spectrumCount: 40,
    maxDepth: 8,
    duration: range(rng, 34, 42),
    viewAzimuth: range(rng, -0.35, 0.1),
    update(t) {
      this.poses[0].rotation = thetaMin + amp * Math.sin((TAU * t) / period + phase)
    },
  }
}

function recombine(rng: () => number): Experiment {
  const glass = rng() < 0.75 ? SF11 : BK7
  const side = range(rng, 1.3, 1.5)
  const shape = prismShape(side)
  const nD = refractiveIndex(glass, SODIUM_D_NM)
  const thetaMin = prismMinDeviationRotation(nD)
  const a = pose(range(rng, -1.9, -1.5), range(rng, 1.0, 1.5), thetaMin - 5 * DEG)
  const b = pose()
  const aim = prismLeftFacePoint(side, 0.5, a)
  const src: Source = { x: -4.7, y: aim.y, angle: 0, width: 0.24, gap: 0, rays: 6, power: 1.2 }
  // Tilted a little past minimum deviation toward steeper incidence: dense flint's violet exits
  // close to grazing at minimum deviation and would fan too wide to land on the second prism
  const bias = -5 * DEG
  const swayAmp = range(rng, 2.5, 4) * DEG
  const swayPeriod = range(rng, 30, 40)
  const gapPeriod = range(rng, 20, 28)
  const maxGap = glass === SF11 ? 0.25 : 0.5
  const h = (side * Math.sqrt(3)) / 2
  return {
    kind: 'recombine',
    elements: [
      { shape, glass },
      { shape, glass },
    ],
    sources: [src],
    poses: [a, b],
    spectrumCount: 28,
    maxDepth: 8,
    duration: range(rng, 34, 42),
    viewAzimuth: range(rng, -0.25, 0.25),
    update(t) {
      a.rotation = thetaMin + bias + swayAmp * Math.sin((TAU * t) / swayPeriod)
      // The second prism is the first turned 180 degrees about a point C past its exit face, so
      // every face pair is parallel and every color leaves parallel to the input: recombined.
      // That holds for ANY C, so C is chosen to centre the second prism on the dispersed fan.
      // Breathing the gap fans the colors apart and folds them back together.
      const gap = 0.03 + (maxGap - 0.03) * (0.5 - 0.5 * Math.cos((TAU * t) / gapPeriod))
      const c = Math.cos(a.rotation)
      const s = Math.sin(a.rotation)
      const mx = side / 4
      const my = h / 6
      const exitX = a.x + mx * c - my * s
      const exitY = a.y + mx * s + my * c
      // Exit-face outward normal, and the face direction pointing toward the base
      const nx = Math.cos(Math.PI / 6 + a.rotation)
      const ny = Math.sin(Math.PI / 6 + a.rotation)
      const bx = ny
      const by = -nx
      // Exit angle (from the normal) of sodium-D light, by Snell at both faces
      const i1 = Math.acos(Math.min(1, -Math.cos((5 * Math.PI) / 6 + a.rotation)))
      const r2 = Math.PI / 3 - Math.asin(Math.sin(i1) / nD)
      const exitAngle = Math.asin(Math.min(0.97, nD * Math.sin(r2)))
      const slide = gap * Math.tan(exitAngle)
      const cx = exitX + nx * gap + bx * slide
      const cy = exitY + ny * gap + by * slide
      b.x = 2 * cx - a.x
      b.y = 2 * cy - a.y
      b.rotation = a.rotation + Math.PI
    },
  }
}

function lens(rng: () => number): Experiment {
  const glass = rng() < 0.7 ? SF11 : BK7
  const nD = refractiveIndex(glass, SODIUM_D_NM)
  const kepler = rng() < 0.45
  const y = range(rng, 0.0, 0.5)
  const swayPeriod = range(rng, 26, 36)
  const swayAmp = range(rng, 8, 14) * DEG
  const phase = rng() * TAU
  if (!kepler) {
    const radius = range(rng, 1.8, 2.4)
    const a = 1.05
    const edge = 0.12
    const lensPose = pose(range(rng, -1.6, -0.9), y, 0)
    const src: Source = { x: -4.7, y, angle: 0, width: 1.7, gap: 0, rays: 11, power: 1.6 }
    return {
      kind: 'lens',
      elements: [{ shape: lensShape(radius, a, edge, true), glass }],
      sources: [src],
      poses: [lensPose],
      spectrumCount: 18,
      maxDepth: 6,
      duration: range(rng, 32, 40),
      viewAzimuth: range(rng, -0.3, 0.3),
      update(t) {
        lensPose.rotation = swayAmp * Math.sin((TAU * t) / swayPeriod + phase)
        src.width = 1.7 - 0.5 * (0.5 - 0.5 * Math.cos((TAU * t) / (swayPeriod * 0.71)))
      },
    }
  }
  // Keplerian beam expander/reducer: lenses one combined focal length apart
  const r1 = range(rng, 1.7, 2.1)
  const r2 = range(rng, 1.1, 1.4)
  const a1 = 1.0
  const a2 = 0.85
  const f1 = lensFocalLength(r1, a1, 0.1, nD)
  const f2 = lensFocalLength(r2, a2, 0.1, nD)
  const p1 = pose(-2.4, y, 0)
  const p2 = pose(-2.4 + f1 + f2, y, 0)
  const src: Source = { x: -4.7, y, angle: 0, width: 1.5, gap: 0, rays: 9, power: 1.5 }
  return {
    kind: 'lens',
    elements: [
      { shape: lensShape(r1, a1, 0.1, true), glass },
      { shape: lensShape(r2, a2, 0.1, true), glass },
    ],
    sources: [src],
    poses: [p1, p2],
    spectrumCount: 16,
    maxDepth: 6,
    duration: range(rng, 32, 40),
    viewAzimuth: range(rng, -0.3, 0.3),
    update(t) {
      // Rock the eyepiece: the recollimated beam steers and picks up off-axis color
      p2.rotation = swayAmp * 0.8 * Math.sin((TAU * t) / swayPeriod + phase)
    },
  }
}

function raindrop(rng: () => number): Experiment {
  const radius = range(rng, 0.95, 1.15)
  const drop = pose(range(rng, -0.4, 0.4), range(rng, 0.2, 0.6), 0)
  const dist = 4.3
  const swayAmp = range(rng, 0.25, 0.4)
  const swayPeriod = range(rng, 32, 44)
  const phase = rng() * TAU
  // The lamp sits on the far side shining toward the viewer, so both bows (thrown back toward
  // the lamp at ~42 and ~51 degrees) land on the far wall, left and right of it. A mask blocks
  // the beam's middle: the bows come from rays near the drop's edge (b/R ~ 0.86 and ~0.95).
  const src: Source = { x: 0, y: 0, angle: Math.PI / 2, width: radius * 2 * 0.998, gap: radius * 2 * 0.74, rays: 48, power: 14 }
  return {
    kind: 'raindrop',
    elements: [{ shape: circleShape(radius), glass: WATER }],
    sources: [src],
    poses: [drop],
    spectrumCount: 12,
    maxDepth: 5,
    duration: range(rng, 34, 44),
    viewAzimuth: range(rng, -0.15, 0.15),
    update(t) {
      const angle = Math.PI / 2 + swayAmp * Math.sin((TAU * t) / swayPeriod + phase)
      src.angle = angle
      src.x = drop.x - Math.cos(angle) * dist
      src.y = drop.y - Math.sin(angle) * dist
    },
  }
}

function crystal(rng: () => number): Experiment {
  const isDiamond = rng() < 0.6
  const shape = isDiamond ? brilliantShape(range(rng, 1.7, 2.0)) : crystalShape(rng, range(rng, 0.8, 0.95))
  const glass = isDiamond ? DIAMOND : SF11
  const gem = pose(range(rng, -0.3, 0.3), range(rng, 0.0, 0.5), rng() * TAU)
  const spin = (rng() < 0.5 ? -1 : 1) * range(rng, 0.09, 0.15)
  const wobble = range(rng, 0.2, 0.5)
  const start = gem.rotation
  const dist = 4.4
  const lampAngles = [Math.PI + range(rng, -0.35, 0.2), -Math.PI / 2 + range(rng, -0.5, 0.5)]
  const sources: Source[] = lampAngles.map(a => ({
    x: gem.x + Math.cos(a) * dist,
    y: gem.y + Math.sin(a) * dist,
    angle: a + Math.PI + range(rng, -0.05, 0.05),
    width: 0.1,
    gap: 0,
    rays: 2,
    power: 0.9,
  }))
  return {
    kind: 'crystal',
    elements: [{ shape, glass }],
    sources,
    poses: [gem],
    spectrumCount: 16,
    maxDepth: 11,
    duration: range(rng, 34, 42),
    viewAzimuth: range(rng, -0.3, 0.3),
    update(t) {
      gem.rotation = start + spin * t + wobble * Math.sin((TAU * t) / 23)
    },
  }
}

export function createExperiment(kind: ExperimentKind, rng: () => number): Experiment {
  switch (kind) {
    case 'newton':
      return newton(rng)
    case 'recombine':
      return recombine(rng)
    case 'lens':
      return lens(rng)
    case 'raindrop':
      return raindrop(rng)
    case 'crystal':
      return crystal(rng)
  }
}

// Fade envelope: elements grow, then the lamp comes on; at the end the lamp goes off, then
// the elements shrink away. Returns [element scale, beam level], both 0..1.
export const GROW_SECONDS = 2.2
export const BEAM_RAMP_SECONDS = 1.4

const smooth = (x: number) => {
  const c = Math.min(1, Math.max(0, x))
  return c * c * (3 - 2 * c)
}

export function envelope(t: number, duration: number): [number, number] {
  const grow = smooth(t / GROW_SECONDS) * (1 - smooth((t - (duration - GROW_SECONDS)) / GROW_SECONDS))
  const beam =
    smooth((t - GROW_SECONDS) / BEAM_RAMP_SECONDS) *
    (1 - smooth((t - (duration - GROW_SECONDS - BEAM_RAMP_SECONDS)) / BEAM_RAMP_SECONDS))
  return [grow, beam]
}
