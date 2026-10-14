import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { Line2 } from 'three/examples/jsm/lines/Line2.js'
import { LineGeometry } from 'three/examples/jsm/lines/LineGeometry.js'
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js'

// Configuration constants
const ANIMATION_DURATION_SECONDS = 120
const PAUSE_DURATION_SECONDS = 30
const SAMPLES = 40000 // trace points per full cycle
const MIN_ROTATIONS = 30 // per arm, per full cycle
const MAX_ROTATIONS = 240
const ARM_COUNT = 3
const MIN_AXIS_ANGLE_DEG = 35 // keep the spin axes apart so the curve is never flat
const MIN_HUE_SEPARATION = 1 / 3 // inner vs outer trace hue, as a fraction of the wheel
const TRACE_WIDTH = 2.5
const ARM_WIDTH = 2
const CAMERA_ORBIT_SECONDS = 12 // one full turn around the shape
const CAMERA_BOB_SECONDS = 17 // one full elevation swing
const CAMERA_BOB_RADIANS = 0.5

const gcd = (a: number, b: number): number => (b === 0 ? a : gcd(b, a % b))

const randomInt = (min: number, max: number) => Math.floor(Math.random() * (max - min + 1)) + min

const randomUnitVector = () => {
  // Uniform on the sphere
  const z = Math.random() * 2 - 1
  const theta = Math.random() * Math.PI * 2
  const r = Math.sqrt(1 - z * z)
  return new THREE.Vector3(r * Math.cos(theta), r * Math.sin(theta), z)
}

const randomPerpendicular = (axis: THREE.Vector3) => {
  const helper = Math.abs(axis.x) < 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0)
  const perp = new THREE.Vector3().crossVectors(axis, helper).normalize()
  return perp.applyAxisAngle(axis, Math.random() * Math.PI * 2)
}

const randomHsl = (): [number, number, number] => [
  Math.random(),
  0.7 + Math.random() * 0.3,
  0.5 + Math.random() * 0.2,
]

interface Arm {
  axis: THREE.Vector3
  start: THREE.Vector3 // arm vector at t=0, perpendicular to axis
  turns: number // signed rotations per cycle
}

type Shape = Arm[]

const createShape = (): Shape => {
  // Spin axes pairwise at least MIN_AXIS_ANGLE_DEG apart
  const maxDot = Math.cos((MIN_AXIS_ANGLE_DEG * Math.PI) / 180)
  const axes: THREE.Vector3[] = []
  while (axes.length < ARM_COUNT) {
    const axis = randomUnitVector()
    if (axes.every(other => Math.abs(other.dot(axis)) <= maxDot)) axes.push(axis)
  }

  // Random length shares (each 0.4-1.0 before normalizing) summing to 1
  const weights = axes.map(() => 0.4 + Math.random() * 0.6)
  const total = weights.reduce((sum, w) => sum + w, 0)

  // Distinct rotation counts with no common divisor, so the curve closes exactly once per cycle
  let turns: number[]
  do {
    turns = axes.map(() => randomInt(MIN_ROTATIONS, MAX_ROTATIONS))
  } while (new Set(turns).size < turns.length || turns.reduce(gcd) !== 1)

  return axes.map((axis, i) => ({
    axis,
    start: randomPerpendicular(axis).multiplyScalar(weights[i] / total),
    turns: turns[i] * (Math.random() < 0.5 ? 1 : -1),
  }))
}

// Joint positions at cycle progress p (0..1); joints[i] is the end of arm i, the last is the tip
const previous = new THREE.Vector3()
const armPositions = (shape: Shape, p: number, joints: THREE.Vector3[]) => {
  previous.set(0, 0, 0)
  shape.forEach((arm, i) => {
    joints[i].copy(arm.start).applyAxisAngle(arm.axis, arm.turns * p * Math.PI * 2).add(previous)
    previous.copy(joints[i])
  })
}

export default function Epicycles3D() {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!containerRef.current) return
    const container = containerRef.current

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x000000)
    scene.fog = new THREE.Fog(0x000000, 1, 10)

    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.01, 100)

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(window.innerWidth, window.innerHeight)
    container.appendChild(renderer.domElement)

    // Trace: the whole closed curve is precomputed per shape; instanceCount reveals it over time
    let traceGeometry = new LineGeometry()
    const traceMaterial = new LineMaterial({ vertexColors: true, linewidth: TRACE_WIDTH, fog: true })
    const trace = new Line2(traceGeometry, traceMaterial)
    scene.add(trace)

    // Arms: center -> joints -> tip
    const armGeometry = new LineGeometry()
    armGeometry.setPositions(new Array((ARM_COUNT + 1) * 3).fill(0))
    // Interleaved [start, end] per segment; updated in place each frame to avoid reallocating
    const armBuffer = (armGeometry.getAttribute('instanceStart') as THREE.InterleavedBufferAttribute).data
    const armMaterial = new LineMaterial({ color: 0xffffff, linewidth: ARM_WIDTH, fog: true })
    const arms = new Line2(armGeometry, armMaterial)
    scene.add(arms)

    const dotGeometry = new THREE.SphereGeometry(0.012, 12, 12)
    const dotMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff, fog: true })
    const dots = Array.from({ length: ARM_COUNT + 1 }, () => {
      const dot = new THREE.Mesh(dotGeometry, dotMaterial)
      scene.add(dot)
      return dot
    })

    let shape = createShape()

    const buildTrace = () => {
      shape = createShape()
      const positions = new Float32Array((SAMPLES + 1) * 3)
      const joints = Array.from({ length: ARM_COUNT }, () => new THREE.Vector3())
      for (let i = 0; i <= SAMPLES; i++) {
        armPositions(shape, i / SAMPLES, joints)
        joints[ARM_COUNT - 1].toArray(positions, i * 3)
      }

      // Color each point by its distance from the center: the curve's nearest point gets the
      // inner color, its farthest point the outer color, everything else in between
      const distances = new Float32Array(SAMPLES + 1)
      let minDistance = Infinity
      let maxDistance = 0
      for (let i = 0; i <= SAMPLES; i++) {
        const d = Math.hypot(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2])
        distances[i] = d
        minDistance = Math.min(minDistance, d)
        maxDistance = Math.max(maxDistance, d)
      }

      // Inner and outer hues at least MIN_HUE_SEPARATION apart; blend along the hue wheel
      // (not through RGB, which greys out between distant hues)
      const [innerHue, innerSat, innerLight] = randomHsl()
      const [, outerSat, outerLight] = randomHsl()
      const hueShift = MIN_HUE_SEPARATION + Math.random() * (0.5 - MIN_HUE_SEPARATION)
      const outerHue = innerHue + (Math.random() < 0.5 ? hueShift : -hueShift)
      const color = new THREE.Color()
      const colors = new Float32Array((SAMPLES + 1) * 3)
      for (let i = 0; i <= SAMPLES; i++) {
        const ratio = maxDistance > minDistance ? (distances[i] - minDistance) / (maxDistance - minDistance) : 0
        const hue = (((innerHue + (outerHue - innerHue) * ratio) % 1) + 1) % 1
        color
          .setHSL(
            hue,
            innerSat + (outerSat - innerSat) * ratio,
            innerLight + (outerLight - innerLight) * ratio,
          )
          .toArray(colors, i * 3)
      }

      // A fresh geometry per shape; disposing the old one frees its GPU buffers
      traceGeometry.dispose()
      traceGeometry = new LineGeometry()
      trace.geometry = traceGeometry
      traceGeometry.setPositions(positions)
      traceGeometry.setColors(colors)
      traceGeometry.instanceCount = 0
    }

    const fitCamera = () => {
      // Whole shape lies within radius 1; keep it in frame for any aspect ratio
      const vFov = THREE.MathUtils.degToRad(camera.fov)
      const hFov = 2 * Math.atan(Math.tan(vFov / 2) * camera.aspect)
      const distance = (1 / Math.sin(Math.min(vFov, hFov) / 2)) * 1.08
      if (scene.fog instanceof THREE.Fog) {
        scene.fog.near = distance - 0.6
        scene.fog.far = distance + 2.2
      }
      return distance
    }

    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight
      camera.updateProjectionMatrix()
      renderer.setSize(window.innerWidth, window.innerHeight)
      traceMaterial.resolution.set(window.innerWidth, window.innerHeight)
      armMaterial.resolution.set(window.innerWidth, window.innerHeight)
      cameraDistance = fitCamera()
    }

    let cameraDistance = fitCamera()
    handleResize()
    buildTrace()

    const cycleSeconds = ANIMATION_DURATION_SECONDS + PAUSE_DURATION_SECONDS
    const startTime = performance.now()
    let cycleStart = startTime
    const joints = Array.from({ length: ARM_COUNT }, () => new THREE.Vector3())
    let animationId = 0

    const animate = () => {
      animationId = requestAnimationFrame(animate)
      const now = performance.now()

      if ((now - cycleStart) / 1000 >= cycleSeconds) {
        cycleStart = now
        buildTrace()
      }

      const progress = Math.min((now - cycleStart) / 1000 / ANIMATION_DURATION_SECONDS, 1)
      const drawing = progress < 1
      traceGeometry.instanceCount = Math.floor(progress * SAMPLES)

      arms.visible = drawing
      dots.forEach(d => (d.visible = drawing))
      if (drawing) {
        armPositions(shape, progress, joints)
        // Segment i runs from point i to point i+1: [start, end] pairs of 6 floats
        const a = armBuffer.array as Float32Array
        joints.forEach((point, i) => {
          point.toArray(a, i * 6 + 3)
          if (i < ARM_COUNT - 1) point.toArray(a, (i + 1) * 6)
          dots[i + 1].position.copy(point)
        })
        armBuffer.needsUpdate = true
      }

      // Camera orbits continuously (independent of the drawing cycle) and swings in elevation
      const t = (now - startTime) / 1000
      const azimuth = (t / CAMERA_ORBIT_SECONDS) * Math.PI * 2
      const elevation = Math.sin((t / CAMERA_BOB_SECONDS) * Math.PI * 2) * CAMERA_BOB_RADIANS
      camera.position.set(
        cameraDistance * Math.cos(elevation) * Math.cos(azimuth),
        cameraDistance * Math.sin(elevation),
        cameraDistance * Math.cos(elevation) * Math.sin(azimuth),
      )
      camera.lookAt(0, 0, 0)

      renderer.render(scene, camera)
    }

    animate()
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      cancelAnimationFrame(animationId)
      traceGeometry.dispose()
      traceMaterial.dispose()
      armGeometry.dispose()
      armMaterial.dispose()
      dotGeometry.dispose()
      dotMaterial.dispose()
      renderer.dispose()
      container.removeChild(renderer.domElement)
    }
  }, [])

  return <div ref={containerRef} className="w-full h-full" />
}
