// CPU twin of the GLSL terrain() in TerrainFlight.tsx. The camera uses it to look ahead and
// keep its altitude above the ground. It has to match the shader closely (it does, apart
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
