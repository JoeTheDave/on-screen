// Strange attractor systems and the warm-up that normalizes them for display.
// Pure math, no Three.js, so it can be sanity-checked outside the browser.

export type Derivative = (x: number, y: number, z: number, out: Float64Array) => void

export interface Attractor {
  name: string
  derivative: Derivative
  maxStep: number // largest stable integration step (native time units)
  seed: () => [number, number, number] // a start point inside the basin
}

const jitter = (scale: number) => (Math.random() - 0.5) * scale

export const ATTRACTORS: Attractor[] = [
  {
    name: 'Lorenz',
    derivative: (x, y, z, out) => {
      out[0] = 10 * (y - x)
      out[1] = x * (28 - z) - y
      out[2] = x * y - (8 / 3) * z
    },
    maxStep: 0.004,
    seed: () => [1 + jitter(2), 1 + jitter(2), 20 + jitter(2)],
  },
  {
    name: 'Thomas',
    derivative: (x, y, z, out) => {
      const b = 0.208186
      out[0] = Math.sin(y) - b * x
      out[1] = Math.sin(z) - b * y
      out[2] = Math.sin(x) - b * z
    },
    maxStep: 0.05,
    seed: () => [1 + jitter(1), jitter(1), jitter(1)],
  },
  {
    name: 'Aizawa',
    derivative: (x, y, z, out) => {
      const a = 0.95
      const b = 0.7
      const c = 0.6
      const d = 3.5
      const e = 0.25
      const f = 0.1
      out[0] = (z - b) * x - d * y
      out[1] = d * x + (z - b) * y
      out[2] = c + a * z - (z * z * z) / 3 - (x * x + y * y) * (1 + e * z) + f * z * x * x * x
    },
    maxStep: 0.008,
    seed: () => [0.1 + jitter(0.1), jitter(0.1), jitter(0.1)],
  },
  {
    name: 'Halvorsen',
    derivative: (x, y, z, out) => {
      const a = 1.89
      out[0] = -a * x - 4 * y - 4 * z - y * y
      out[1] = -a * y - 4 * z - 4 * x - z * z
      out[2] = -a * z - 4 * x - 4 * y - x * x
    },
    maxStep: 0.004,
    seed: () => [-1.5 + jitter(0.5), jitter(0.5), jitter(0.5)],
  },
]

const scratch = new Float64Array(3)
const mid = new Float64Array(3)

// One midpoint (RK2) step, in place on p[i..i+2]
export const step = (derivative: Derivative, p: Float64Array, i: number, dt: number) => {
  const x = p[i]
  const y = p[i + 1]
  const z = p[i + 2]
  derivative(x, y, z, scratch)
  derivative(x + scratch[0] * dt * 0.5, y + scratch[1] * dt * 0.5, z + scratch[2] * dt * 0.5, mid)
  p[i] = x + mid[0] * dt
  p[i + 1] = y + mid[1] * dt
  p[i + 2] = z + mid[2] * dt
}

export interface AttractorStats {
  center: [number, number, number]
  radius: number // 95th-percentile distance from center
  meanSpeed: number // native units per native time unit
  pool: Float64Array // points on the attractor, for respawning
}

// Run a handful of trajectories past their transient, then measure where and how fast they go
export const warmUp = (attractor: Attractor, trajectories = 60, settleSteps = 3000, sampleSteps = 3000) => {
  const dt = attractor.maxStep
  const every = 10
  const samplesPer = Math.floor(sampleSteps / every)
  const pool = new Float64Array(trajectories * samplesPer * 3)
  const p = new Float64Array(3)
  let speedSum = 0
  let speedCount = 0
  let n = 0

  for (let t = 0; t < trajectories; t++) {
    p.set(attractor.seed())
    for (let s = 0; s < settleSteps; s++) step(attractor.derivative, p, 0, dt)
    for (let s = 0; s < sampleSteps; s++) {
      step(attractor.derivative, p, 0, dt)
      if (s % every === 0) {
        pool.set(p, n * 3)
        n++
        attractor.derivative(p[0], p[1], p[2], scratch)
        speedSum += Math.hypot(scratch[0], scratch[1], scratch[2])
        speedCount++
      }
    }
  }

  const center: [number, number, number] = [0, 0, 0]
  for (let i = 0; i < n; i++) {
    center[0] += pool[i * 3] / n
    center[1] += pool[i * 3 + 1] / n
    center[2] += pool[i * 3 + 2] / n
  }
  const distances = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    distances[i] = Math.hypot(pool[i * 3] - center[0], pool[i * 3 + 1] - center[1], pool[i * 3 + 2] - center[2])
  }
  distances.sort()
  const radius = distances[Math.floor(n * 0.95)]

  return { center, radius, meanSpeed: speedSum / speedCount, pool } satisfies AttractorStats
}
