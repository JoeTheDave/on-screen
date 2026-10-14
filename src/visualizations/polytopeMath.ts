// Pure 4D geometry for the Hypercube visualization: the regular 4-polytopes, built
// programmatically, plus 4D rotation and projection. No three.js here so it can be
// sanity-checked headlessly.

export interface Polytope {
  name: string
  /** xyzw per vertex, normalized to unit circumradius */
  vertices: Float64Array
  /** vertex index pairs, one per edge */
  edges: Uint16Array
  vertexCount: number
  edgeCount: number
}

const PHI = (1 + Math.sqrt(5)) / 2

type Vec4 = [number, number, number, number]

// All sign combinations of the nonzero entries of v
function signVariants(v: Vec4): Vec4[] {
  let out: Vec4[] = [[...v]]
  for (let axis = 0; axis < 4; axis++) {
    if (v[axis] === 0) continue
    out = out.flatMap(p => {
      const flipped = [...p] as Vec4
      flipped[axis] = -flipped[axis]
      return [p, flipped]
    })
  }
  return out
}

// All 24 permutations of [0,1,2,3], tagged with parity
function permutations4(): { perm: number[]; even: boolean }[] {
  const out: { perm: number[]; even: boolean }[] = []
  const rec = (prefix: number[], rest: number[]) => {
    if (rest.length === 0) {
      let inversions = 0
      for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) if (prefix[i] > prefix[j]) inversions++
      out.push({ perm: prefix, even: inversions % 2 === 0 })
      return
    }
    rest.forEach((r, i) => rec([...prefix, r], [...rest.slice(0, i), ...rest.slice(i + 1)]))
  }
  rec([], [0, 1, 2, 3])
  return out
}

// Sign variants of every (optionally only even) coordinate permutation of v, deduplicated
function expand(v: Vec4, evenOnly: boolean): Vec4[] {
  const seen = new Map<string, Vec4>()
  for (const { perm, even } of permutations4()) {
    if (evenOnly && !even) continue
    const permuted: Vec4 = [v[perm[0]], v[perm[1]], v[perm[2]], v[perm[3]]]
    for (const s of signVariants(permuted)) {
      // +0 normalizes -0 so the dedupe key is stable
      seen.set(s.map(c => (c + 0).toFixed(6)).join(','), s)
    }
  }
  return [...seen.values()]
}

function normalize(points: Vec4[]): Vec4[] {
  return points.map(p => {
    const r = Math.hypot(...p)
    return p.map(c => c / r) as Vec4
  })
}

function distSq(a: Vec4, b: Vec4): number {
  const dx = a[0] - b[0]
  const dy = a[1] - b[1]
  const dz = a[2] - b[2]
  const dw = a[3] - b[3]
  return dx * dx + dy * dy + dz * dz + dw * dw
}

// In a regular polytope the edges are exactly the vertex pairs at the minimal distance
function minimalDistanceEdges(points: Vec4[]): [number, number][] {
  let min = Infinity
  for (let i = 0; i < points.length; i++)
    for (let j = i + 1; j < points.length; j++) min = Math.min(min, distSq(points[i], points[j]))
  const edges: [number, number][] = []
  const limit = min * (1 + 1e-6)
  for (let i = 0; i < points.length; i++)
    for (let j = i + 1; j < points.length; j++) if (distSq(points[i], points[j]) <= limit) edges.push([i, j])
  return edges
}

function pack(name: string, points: Vec4[], edges: [number, number][]): Polytope {
  return {
    name,
    vertices: new Float64Array(points.flat()),
    edges: new Uint16Array(edges.flat()),
    vertexCount: points.length,
    edgeCount: edges.length,
  }
}

function build(name: string, points: Vec4[]): Polytope {
  const unit = normalize(points)
  return pack(name, unit, minimalDistanceEdges(unit))
}

function cells600(): Vec4[] {
  return [
    ...expand([1, 0, 0, 0], false),
    ...expand([0.5, 0.5, 0.5, 0.5], false),
    ...expand([PHI / 2, 0.5, 1 / (2 * PHI), 0], true),
  ]
}

// The 120-cell is the dual of the 600-cell: one vertex at the center of each of the
// 600-cell's 600 tetrahedral cells (4-cliques of its edge graph).
function cells120(): Vec4[] {
  const v = normalize(cells600())
  const edges = minimalDistanceEdges(v)
  const adjacent = v.map(() => new Set<number>())
  for (const [a, b] of edges) {
    adjacent[a].add(b)
    adjacent[b].add(a)
  }
  const centers: Vec4[] = []
  for (let a = 0; a < v.length; a++) {
    for (const b of adjacent[a]) {
      if (b <= a) continue
      for (const c of adjacent[b]) {
        if (c <= b || !adjacent[a].has(c)) continue
        for (const d of adjacent[c]) {
          if (d <= c || !adjacent[a].has(d) || !adjacent[b].has(d)) continue
          centers.push([0, 1, 2, 3].map(k => (v[a][k] + v[b][k] + v[c][k] + v[d][k]) / 4) as Vec4)
        }
      }
    }
  }
  return centers
}

/** The five polytopes, ordered small to large for the show's cycle */
export function buildPolytopes(): Polytope[] {
  return [
    build('8-cell', expand([1, 1, 1, 1], false)),
    build('16-cell', expand([1, 0, 0, 0], false)),
    build('24-cell', expand([1, 1, 0, 0], false)),
    build('600-cell', cells600()),
    build('120-cell', cells120()),
  ]
}

/**
 * Row-major 4x4 rotation composed from the six plane angles
 * [xy, xz, xw, yz, yw, zw]. Written into `out` to avoid per-frame allocation.
 */
export function rotation4(angles: ArrayLike<number>, out: Float64Array): Float64Array {
  out.fill(0)
  out[0] = out[5] = out[10] = out[15] = 1
  const planes: [number, number][] = [
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 2],
    [1, 3],
    [2, 3],
  ]
  for (let p = 0; p < 6; p++) {
    const [i, j] = planes[p]
    const c = Math.cos(angles[p])
    const s = Math.sin(angles[p])
    // out = R_p * out: only rows i and j change
    for (let k = 0; k < 4; k++) {
      const a = out[i * 4 + k]
      const b = out[j * 4 + k]
      out[i * 4 + k] = c * a - s * b
      out[j * 4 + k] = s * a + c * b
    }
  }
  return out
}

/**
 * Rotate every vertex by `m`, then perspective-project 4D -> 3D from an eye on the +w axis
 * at `eye`. Scale is normalized so w = 0 keeps its size; vertices nearer the eye (w -> 1)
 * swell, which is what makes inner cells bloom outward and swallow the outer ones.
 * Writes xyz into out3 and the rotated w into outW.
 */
export function projectVertices(
  src: Float64Array,
  count: number,
  m: Float64Array,
  eye: number,
  out3: Float32Array,
  outW: Float32Array,
): void {
  for (let v = 0; v < count; v++) {
    const i = v * 4
    const x = src[i]
    const y = src[i + 1]
    const z = src[i + 2]
    const w = src[i + 3]
    const rx = m[0] * x + m[1] * y + m[2] * z + m[3] * w
    const ry = m[4] * x + m[5] * y + m[6] * z + m[7] * w
    const rz = m[8] * x + m[9] * y + m[10] * z + m[11] * w
    const rw = m[12] * x + m[13] * y + m[14] * z + m[15] * w
    const k = eye / (eye - rw)
    out3[v * 3] = rx * k
    out3[v * 3 + 1] = ry * k
    out3[v * 3 + 2] = rz * k
    outW[v] = rw
  }
}
