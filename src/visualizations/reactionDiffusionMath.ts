// Slowly evolving fields that keep the Reaction Diffusion visualization moving: a divergence-free
// flow that stirs the chemicals, traveling fronts that sweep feed/kill between two regimes, and
// an occasional gentle stirring vortex. Pure math, no Three.js, so it can be exercised in node.
//
// Every wave uses integer wave numbers in sim uv space, so all fields tile the wrapped (toroidal)
// sim seamlessly. Waves are born, swell, fade and get replaced with fresh random ones, so the
// motion never repeats; phases are reported modulo 2*pi so the shader never sees a large number.

export type Rand = () => number

const TAU = Math.PI * 2

interface WaveBand {
  minK: number // |k| range, in waves per sim width/height
  maxK: number
  minLife: number // seconds from birth to fade-out
  maxLife: number
  strength: number // flow: speed (texels/step) at the crest; fronts: relative weight
  minCross: number // seconds for a crest to travel one full wavelength
  maxCross: number
}

interface Wave {
  kx: number
  ky: number
  phase: number
  omega: number // rad/s
  strength: number
  born: number
  life: number
  band: WaveBand
}

export interface WaveField {
  waves: Wave[]
  rand: Rand
}

// A few large swirls plus finer eddies
export const FLOW_BANDS: WaveBand[] = [
  { minK: 1, maxK: 2.3, minLife: 24, maxLife: 40, strength: 0.014, minCross: 18, maxCross: 40 },
  { minK: 1, maxK: 2.3, minLife: 24, maxLife: 40, strength: 0.014, minCross: 18, maxCross: 40 },
  { minK: 1, maxK: 2.3, minLife: 24, maxLife: 40, strength: 0.014, minCross: 18, maxCross: 40 },
  { minK: 1, maxK: 2.3, minLife: 24, maxLife: 40, strength: 0.014, minCross: 18, maxCross: 40 },
  { minK: 2.8, maxK: 5, minLife: 16, maxLife: 28, strength: 0.006, minCross: 10, maxCross: 24 },
  { minK: 2.8, maxK: 5, minLife: 16, maxLife: 28, strength: 0.006, minCross: 10, maxCross: 24 },
  { minK: 2.8, maxK: 5, minLife: 16, maxLife: 28, strength: 0.006, minCross: 10, maxCross: 24 },
  { minK: 2.8, maxK: 5, minLife: 16, maxLife: 28, strength: 0.006, minCross: 10, maxCross: 24 },
]

// Broad bands that roll across the screen, converting one regime into the other as they pass
export const FRONT_BANDS: WaveBand[] = [
  { minK: 1, maxK: 1.5, minLife: 70, maxLife: 130, strength: 1, minCross: 40, maxCross: 75 },
  { minK: 1, maxK: 2.3, minLife: 70, maxLife: 130, strength: 0.8, minCross: 40, maxCross: 75 },
  { minK: 1.4, maxK: 2.3, minLife: 70, maxLife: 130, strength: 0.6, minCross: 40, maxCross: 75 },
]

const between = (rand: Rand, lo: number, hi: number) => lo + (hi - lo) * rand()

// Random integer wave vector with lo <= |k| <= hi (ky >= 0; the sign of omega picks direction)
const pickK = (rand: Rand, lo: number, hi: number): [number, number] => {
  const r = Math.ceil(hi)
  for (;;) {
    const kx = Math.floor(rand() * (2 * r + 1)) - r
    const ky = Math.floor(rand() * (r + 1))
    const len = Math.hypot(kx, ky)
    if (len >= lo && len <= hi && len > 0) return [kx, ky]
  }
}

const spawn = (rand: Rand, band: WaveBand, born: number): Wave => {
  const [kx, ky] = pickK(rand, band.minK, band.maxK)
  return {
    kx,
    ky,
    phase: rand() * TAU,
    omega: (rand() < 0.5 ? -1 : 1) * (TAU / between(rand, band.minCross, band.maxCross)),
    strength: band.strength * between(rand, 0.75, 1.15),
    born,
    life: between(rand, band.minLife, band.maxLife),
    band,
  }
}

// Births are staggered so the waves never all fade at once
export const createWaveField = (bands: WaveBand[], t: number, rand: Rand): WaveField => ({
  rand,
  waves: bands.map((band, i) => {
    const w = spawn(rand, band, t)
    w.born = t - ((i + rand() * 0.5) / bands.length) * w.life
    return w
  }),
})

// sin^2 swell and fade over the wave's life
const envelope = (w: Wave, t: number) => {
  const s = Math.sin((Math.PI * (t - w.born)) / w.life)
  return s * s
}

const advance = (field: WaveField, t: number) => {
  field.waves.forEach((w, i) => {
    if (t - w.born >= w.life) field.waves[i] = spawn(field.rand, w.band, t)
  })
}

// Current phase in [0, 2pi): t - born never exceeds a wave's life, so this stays precise forever
const phaseAt = (w: Wave, t: number) => {
  const p = (w.phase + w.omega * (t - w.born)) % TAU
  return p < 0 ? p + TAU : p
}

// Flow field: the stream function is a sum of plane waves, so each wave's velocity is a constant
// vector perpendicular to its (texel-space) wave vector times cos(phase): divergence-free.
// waves gets (kx, ky, phase, 0) per wave; vel gets the crest velocity in texels/step.
export function updateFlow(
  field: WaveField,
  t: number,
  simW: number,
  simH: number,
  speedScale: number,
  waves: Float32Array,
  vel: Float32Array,
) {
  advance(field, t)
  field.waves.forEach((w, i) => {
    const px = w.ky / simH
    const py = -w.kx / simW
    const s = (w.strength * envelope(w, t) * speedScale) / Math.hypot(px, py)
    waves[i * 4] = w.kx
    waves[i * 4 + 1] = w.ky
    waves[i * 4 + 2] = phaseAt(w, t)
    waves[i * 4 + 3] = 0
    vel[i * 2] = px * s
    vel[i * 2 + 1] = py * s
  })
}

// Fronts: (kx, ky, phase, weight) per wave, weights normalized to sum to 1 so |sum| <= 1
export function updateFronts(field: WaveField, t: number, out: Float32Array) {
  advance(field, t)
  let total = 0
  for (const w of field.waves) total += w.strength * envelope(w, t)
  total = Math.max(total, 1e-3)
  field.waves.forEach((w, i) => {
    out[i * 4] = w.kx
    out[i * 4 + 1] = w.ky
    out[i * 4 + 2] = phaseAt(w, t)
    out[i * 4 + 3] = (w.strength * envelope(w, t)) / total
  })
}

// The same sums the shaders evaluate, for CPU checks
export const flowAt = (waves: Float32Array, vel: Float32Array, u: number, v: number): [number, number] => {
  let x = 0
  let y = 0
  for (let i = 0; i < waves.length / 4; i++) {
    const c = Math.cos(TAU * (waves[i * 4] * u + waves[i * 4 + 1] * v) + waves[i * 4 + 2])
    x += vel[i * 2] * c
    y += vel[i * 2 + 1] * c
  }
  return [x, y]
}

export const frontAt = (fronts: Float32Array, u: number, v: number) => {
  let s = 0
  for (let i = 0; i < fronts.length / 4; i++) {
    s += fronts[i * 4 + 3] * Math.sin(TAU * (fronts[i * 4] * u + fronts[i * 4 + 1] * v) + fronts[i * 4 + 2])
  }
  return s
}

// Occasional stirring vortex: a Gaussian swirl, v = spin * exp(-r^2/R^2) * perp(r), which is
// divergence-free. It swells and fades over its life, then rests a while before the next.
export interface Vortex {
  x: number // uv
  y: number
  radius: number // fraction of the sim's long side
  peak: number // peak tangential speed, texels/step (signed: direction of spin)
  born: number
  life: number
  next: number // time the following vortex may start
}

export const VORTEX = {
  minRadius: 0.06,
  maxRadius: 0.12,
  minPeak: 0.02,
  maxPeak: 0.035,
  minLife: 14,
  maxLife: 22,
  minRest: 15,
  maxRest: 35,
}

export const createVortex = (t: number, rand: Rand): Vortex => ({
  x: 0.5,
  y: 0.5,
  radius: VORTEX.minRadius,
  peak: 0,
  born: t,
  life: 1,
  next: t + between(rand, 6, 20),
})

// Writes (x, y, radius in texels, spin) where spin * |r| * exp(-r^2/R^2) peaks at `peak`
export function updateVortex(v: Vortex, t: number, simW: number, simH: number, rand: Rand, out: Float32Array) {
  if (t >= v.next) {
    v.x = rand()
    v.y = rand()
    v.radius = between(rand, VORTEX.minRadius, VORTEX.maxRadius)
    v.peak = (rand() < 0.5 ? -1 : 1) * between(rand, VORTEX.minPeak, VORTEX.maxPeak)
    v.born = t
    v.life = between(rand, VORTEX.minLife, VORTEX.maxLife)
    v.next = t + v.life + between(rand, VORTEX.minRest, VORTEX.maxRest)
  }
  const age = t - v.born
  const s = age < v.life ? Math.sin((Math.PI * age) / v.life) : 0
  const radius = v.radius * Math.max(simW, simH)
  // max over r of r * exp(-r^2/R^2) is R * exp(-1/2) / sqrt(2)
  const spin = (v.peak * s * s) / (radius * Math.exp(-0.5) * Math.SQRT1_2)
  out[0] = v.x
  out[1] = v.y
  out[2] = radius
  out[3] = spin
}

// Coverage probe: share of samples (red = B * 2 in 0..255) where chemical B is clearly present
export function coverage(pixels: Uint8Array): number {
  let alive = 0
  const n = pixels.length / 4
  for (let i = 0; i < n; i++) if (pixels[i * 4] > 51) alive++
  return alive / n
}
