// Chladni-plate math for the Cymatics visualization: vibration modes, the sampled field, and the
// sand-grain update. Pure math, no Three.js, so it can be sanity-checked outside the browser.
//
// Plate coordinates span [-1, 1] on both axes. A square plate is that whole square; a circular
// plate is the unit disk. The plate outline is a rounded box whose corner radius morphs 0 -> 1,
// so a square smoothly becomes a circle (and back) when the plate shape changes.

export type PlateShape = 'square' | 'circle'

export interface Mode {
  shape: PlateShape
  n: number
  m: number
  sign: number // square only: +1 or -1 combination of the two degenerate modes
  rotation: number // circle only: angular offset of the nodal diameters
}

export interface Field {
  size: number // samples per side over [-1, 1]
  u: Float32Array // displacement amplitude, normalized so max |u| on the plate is 1
  gx: Float32Array // du/dx
  gy: Float32Array // du/dy
}

export interface StepParams {
  dt: number // seconds
  drift: number // pull toward nodal lines: velocity = -drift * u * grad(u)
  jitter: number // random hopping, rms distance per sqrt(second) at an antinode
  floor: number // tiny hop everywhere so settled lines still breathe
  kick: number // extra uniform hopping right after the plate is struck with a new mode
  roundness: number // plate outline corner radius: 0 square, 1 circle
}

export const createField = (size: number): Field => ({
  size,
  u: new Float32Array(size * size),
  gx: new Float32Array(size * size),
  gy: new Float32Array(size * size),
})

// Bessel function of the first kind via its integral form,
// J_n(x) = (1/pi) * integral_0^pi cos(n t - x sin t) dt. The midpoint rule is spectrally accurate
// here (the integrand is smooth and periodic), so 128 samples are exact to float precision for
// the x < ~60 this file ever needs.
export function besselJ(n: number, x: number): number {
  const samples = 128
  let sum = 0
  for (let k = 0; k < samples; k++) {
    const t = ((k + 0.5) / samples) * Math.PI
    sum += Math.cos(n * t - x * Math.sin(t))
  }
  return sum / samples
}

// The m-th positive zero of J_n (m >= 1): scan for a sign change, then bisect
export function besselZero(n: number, m: number): number {
  let count = 0
  let a = 0.05
  let fa = besselJ(n, a)
  for (let b = a + 0.05; b < 200; b += 0.05) {
    const fb = besselJ(n, b)
    if (fa * fb < 0) {
      count++
      if (count === m) {
        let lo = a
        let hi = b
        let flo = fa
        for (let i = 0; i < 50; i++) {
          const mid = (lo + hi) / 2
          const fm = besselJ(n, mid)
          if (flo * fm <= 0) hi = mid
          else {
            lo = mid
            flo = fm
          }
        }
        return (lo + hi) / 2
      }
    }
    a = b
    fa = fb
  }
  return NaN
}

const RADIAL_TABLE = 2048
const RADIAL_MAX = 1.5 // past the corners of the [-1,1] square (sqrt 2), for mid-morph grains
const radialTable = new Float64Array(RADIAL_TABLE + 1)

// Write the mode's displacement into field.u (normalized) and its gradient into gx/gy.
export function fillField(field: Field, mode: Mode): void {
  const { size, u, gx, gy } = field
  const step = 2 / (size - 1)

  if (mode.shape === 'circle') {
    // Clamped edge: the rim is the m-th nodal circle, so sand also gathers in a ring at the edge
    const k = besselZero(mode.n, mode.m)
    for (let i = 0; i <= RADIAL_TABLE; i++) {
      radialTable[i] = besselJ(mode.n, (k * i * RADIAL_MAX) / RADIAL_TABLE)
    }
  }

  let maxAbs = 0
  for (let j = 0; j < size; j++) {
    const y = -1 + j * step
    for (let i = 0; i < size; i++) {
      const x = -1 + i * step
      let value: number
      let onPlate = true
      if (mode.shape === 'square') {
        const { n, m, sign } = mode
        value =
          Math.cos(n * Math.PI * x) * Math.cos(m * Math.PI * y) +
          sign * Math.cos(m * Math.PI * x) * Math.cos(n * Math.PI * y)
      } else {
        const r = Math.hypot(x, y)
        const theta = Math.atan2(y, x) + mode.rotation
        const f = (Math.min(r, RADIAL_MAX) / RADIAL_MAX) * RADIAL_TABLE
        const i0 = Math.min(Math.floor(f), RADIAL_TABLE - 1)
        const t = f - i0
        const radial = radialTable[i0] * (1 - t) + radialTable[i0 + 1] * t
        value = radial * Math.cos(mode.n * theta)
        onPlate = r <= 1
      }
      u[j * size + i] = value
      if (onPlate) maxAbs = Math.max(maxAbs, Math.abs(value))
    }
  }

  const scale = maxAbs > 0 ? 1 / maxAbs : 1
  for (let k = 0; k < u.length; k++) u[k] *= scale

  // Central differences (one-sided at the border)
  for (let j = 0; j < size; j++) {
    for (let i = 0; i < size; i++) {
      const k = j * size + i
      const il = Math.max(i - 1, 0)
      const ir = Math.min(i + 1, size - 1)
      const jd = Math.max(j - 1, 0)
      const ju = Math.min(j + 1, size - 1)
      gx[k] = (u[j * size + ir] - u[j * size + il]) / ((ir - il) * step)
      gy[k] = (u[ju * size + i] - u[jd * size + i]) / ((ju - jd) * step)
    }
  }
}

// Pick a random mode that makes a rich, legible figure, never repeating the previous one
export function randomMode(previous: Mode | null, random: () => number = Math.random): Mode {
  for (;;) {
    let mode: Mode
    if (random() < 0.6) {
      const n = 1 + Math.floor(random() * 8)
      const m = 1 + Math.floor(random() * 8)
      // Too simple (a few lines) or too busy (lines closer than the sand can resolve)
      if (n === m || n + m < 5 || n + m > 13) continue
      mode = { shape: 'square', n, m, sign: random() < 0.5 ? 1 : -1, rotation: 0 }
    } else {
      const n = Math.floor(random() * 7)
      const m = 1 + Math.floor(random() * 5)
      if (n + m < 3 || n + m > 9 || (n === 0 && m < 3)) continue
      mode = { shape: 'circle', n, m, sign: 1, rotation: random() * Math.PI * 2 }
    }
    if (
      previous &&
      previous.shape === mode.shape &&
      ((previous.n === mode.n && previous.m === mode.m) ||
        (mode.shape === 'square' && previous.n === mode.m && previous.m === mode.n))
    ) {
      continue
    }
    return mode
  }
}

// Bilinear sample of u, du/dx, du/dy at plate point (x, y) into out[0..2]
export function sampleField(field: Field, x: number, y: number, out: Float32Array | Float64Array): void {
  const { size, u, gx, gy } = field
  const last = size - 1
  let fx = (x + 1) * 0.5 * last
  let fy = (y + 1) * 0.5 * last
  fx = fx < 0 ? 0 : fx > last ? last : fx
  fy = fy < 0 ? 0 : fy > last ? last : fy
  let i0 = Math.floor(fx)
  let j0 = Math.floor(fy)
  if (i0 >= last) i0 = last - 1
  if (j0 >= last) j0 = last - 1
  const tx = fx - i0
  const ty = fy - j0
  const k00 = j0 * size + i0
  const k10 = k00 + 1
  const k01 = k00 + size
  const k11 = k01 + 1
  const w00 = (1 - tx) * (1 - ty)
  const w10 = tx * (1 - ty)
  const w01 = (1 - tx) * ty
  const w11 = tx * ty
  out[0] = u[k00] * w00 + u[k10] * w10 + u[k01] * w01 + u[k11] * w11
  out[1] = gx[k00] * w00 + gx[k10] * w10 + gx[k01] * w01 + gx[k11] * w11
  out[2] = gy[k00] * w00 + gy[k10] * w10 + gy[k01] * w01 + gy[k11] * w11
}

// Keep (x, y) inside the rounded-box plate outline (corner radius r, half-size 1).
// Writes the corrected point into out[0..1].
export function constrainToPlate(x: number, y: number, r: number, out: Float32Array | Float64Array): void {
  const inner = 1 - r
  const cx = x < -inner ? -inner : x > inner ? inner : x
  const cy = y < -inner ? -inner : y > inner ? inner : y
  const dx = x - cx
  const dy = y - cy
  const d = Math.hypot(dx, dy)
  // Stay a hair inside so the grain does not sit exactly on the rim
  const limit = r * 0.998
  if (d > limit) {
    if (d > 0 && r > 0) {
      out[0] = cx + (dx / d) * limit
      out[1] = cy + (dy / d) * limit
    } else {
      out[0] = cx * 0.998
      out[1] = cy * 0.998
    }
  } else {
    out[0] = x
    out[1] = y
  }
}

const scratchSample = new Float64Array(3)
const scratchFixed = new Float64Array(2)

const MAX_MOVE = 0.03 // per step, so a coarse frame can never fling a grain across a nodal line

// Approximately standard normal: sum of three uniforms has variance 1/4
const gauss = (random: () => number) => (random() + random() + random() - 1.5) * 2

// Advance every grain one step. Grains hop randomly with an amplitude proportional to the local
// vibration and drift down the gradient of u^2, so they leave the antinodes and collect on the
// nodal lines, where the plate is still. Writes each grain's |u| into amplitude.
export function stepGrains(
  positions: Float32Array, // x, y pairs in plate coordinates
  amplitude: Float32Array,
  count: number,
  field: Field,
  params: StepParams,
  random: () => number = Math.random,
): void {
  const { dt, drift, jitter, floor, kick, roundness } = params
  const sqrtDt = Math.sqrt(dt)
  const sample = scratchSample
  const fixed = scratchFixed

  for (let p = 0; p < count; p++) {
    const ix = p * 2
    const x = positions[ix]
    const y = positions[ix + 1]
    sampleField(field, x, y, sample)
    const value = sample[0]
    const a = Math.abs(value)

    const hop = (jitter * a + floor + kick) * sqrtDt
    let mx = -drift * value * sample[1] * dt + hop * gauss(random)
    let my = -drift * value * sample[2] * dt + hop * gauss(random)
    const move = Math.hypot(mx, my)
    if (move > MAX_MOVE) {
      mx *= MAX_MOVE / move
      my *= MAX_MOVE / move
    }

    constrainToPlate(x + mx, y + my, roundness, fixed)
    positions[ix] = fixed[0]
    positions[ix + 1] = fixed[1]
    amplitude[p] = a
  }
}

// Scatter grains uniformly over the plate (rejection sampling against the outline)
export function scatterGrains(
  positions: Float32Array,
  count: number,
  roundness: number,
  random: () => number = Math.random,
): void {
  const inner = 1 - roundness
  for (let p = 0; p < count; p++) {
    let x: number
    let y: number
    do {
      x = random() * 2 - 1
      y = random() * 2 - 1
    } while (Math.hypot(Math.max(Math.abs(x) - inner, 0), Math.max(Math.abs(y) - inner, 0)) > roundness)
    positions[p * 2] = x
    positions[p * 2 + 1] = y
  }
}
