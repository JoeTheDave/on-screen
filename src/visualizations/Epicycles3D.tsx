import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { Line2 } from 'three/examples/jsm/lines/Line2.js'
import { LineGeometry } from 'three/examples/jsm/lines/LineGeometry.js'
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js'

// Configuration constants
const ANIMATION_DURATION_SECONDS = 60
const PAUSE_DURATION_SECONDS = 30
const SAMPLES = 4000 // trace points per full cycle
const MIN_ROTATIONS = 3
const MAX_ROTATIONS = 24
const MIN_AXIS_ANGLE_DEG = 35 // keep the two spin axes apart so the curve is never flat
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

interface Shape {
  axis1: THREE.Vector3
  axis2: THREE.Vector3
  arm1: THREE.Vector3 // arm 1 at t=0, perpendicular to axis1
  arm2: THREE.Vector3 // arm 2 at t=0, perpendicular to axis2
  turns1: number // signed rotations per cycle
  turns2: number
}

const createShape = (): Shape => {
  const axis1 = randomUnitVector()
  let axis2 = randomUnitVector()
  const maxDot = Math.cos((MIN_AXIS_ANGLE_DEG * Math.PI) / 180)
  while (Math.abs(axis1.dot(axis2)) > maxDot) axis2 = randomUnitVector()

  const split = 0.2 + Math.random() * 0.3 // arm 1: 20-50%, arm 2: 50-80%

  // Coprime rotation counts so the curve closes exactly once per cycle
  const r1 = randomInt(MIN_ROTATIONS, MAX_ROTATIONS)
  let r2 = randomInt(MIN_ROTATIONS, MAX_ROTATIONS)
  while (r2 === r1 || gcd(r1, r2) !== 1) r2 = randomInt(MIN_ROTATIONS, MAX_ROTATIONS)

  return {
    axis1,
    axis2,
    arm1: randomPerpendicular(axis1).multiplyScalar(split),
    arm2: randomPerpendicular(axis2).multiplyScalar(1 - split),
    turns1: r1 * (Math.random() < 0.5 ? 1 : -1),
    turns2: r2 * (Math.random() < 0.5 ? 1 : -1),
  }
}

// Joint and tip positions at cycle progress p (0..1)
const armPositions = (shape: Shape, p: number, joint: THREE.Vector3, tip: THREE.Vector3) => {
  joint.copy(shape.arm1).applyAxisAngle(shape.axis1, shape.turns1 * p * Math.PI * 2)
  tip.copy(shape.arm2).applyAxisAngle(shape.axis2, shape.turns2 * p * Math.PI * 2).add(joint)
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

    // Arms: center -> joint -> tip
    const armGeometry = new LineGeometry()
    armGeometry.setPositions(new Array(9).fill(0))
    // Interleaved [start, end] per segment; updated in place each frame to avoid reallocating
    const armBuffer = (armGeometry.getAttribute('instanceStart') as THREE.InterleavedBufferAttribute).data
    const armMaterial = new LineMaterial({ color: 0xffffff, linewidth: ARM_WIDTH, fog: true })
    const arms = new Line2(armGeometry, armMaterial)
    scene.add(arms)

    const dotGeometry = new THREE.SphereGeometry(0.012, 12, 12)
    const dotMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff, fog: true })
    const dots = [0, 1, 2].map(() => {
      const dot = new THREE.Mesh(dotGeometry, dotMaterial)
      scene.add(dot)
      return dot
    })

    let shape = createShape()

    const buildTrace = () => {
      shape = createShape()
      const positions = new Float32Array((SAMPLES + 1) * 3)
      const joint = new THREE.Vector3()
      const tip = new THREE.Vector3()
      for (let i = 0; i <= SAMPLES; i++) {
        armPositions(shape, i / SAMPLES, joint, tip)
        tip.toArray(positions, i * 3)
      }

      // Color each point by local tip speed, slow hue -> fast hue (as in the 2D version)
      const speeds = new Float32Array(SAMPLES + 1)
      for (let i = 0; i < SAMPLES; i++) {
        const dx = positions[i * 3 + 3] - positions[i * 3]
        const dy = positions[i * 3 + 4] - positions[i * 3 + 1]
        const dz = positions[i * 3 + 5] - positions[i * 3 + 2]
        speeds[i + 1] = Math.sqrt(dx * dx + dy * dy + dz * dz)
      }
      speeds[0] = speeds[SAMPLES]
      let minSpeed = Infinity
      let maxSpeed = 0
      for (const s of speeds) {
        minSpeed = Math.min(minSpeed, s)
        maxSpeed = Math.max(maxSpeed, s)
      }

      const slow = new THREE.Color().setHSL(...randomHsl())
      const fast = new THREE.Color().setHSL(...randomHsl())
      const color = new THREE.Color()
      const colors = new Float32Array((SAMPLES + 1) * 3)
      for (let i = 0; i <= SAMPLES; i++) {
        const ratio = maxSpeed > minSpeed ? (speeds[i] - minSpeed) / (maxSpeed - minSpeed) : 0
        color.lerpColors(slow, fast, ratio).toArray(colors, i * 3)
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
    const joint = new THREE.Vector3()
    const tip = new THREE.Vector3()
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
        armPositions(shape, progress, joint, tip)
        const a = armBuffer.array as Float32Array
        joint.toArray(a, 3)
        joint.toArray(a, 6)
        tip.toArray(a, 9)
        armBuffer.needsUpdate = true
        dots[1].position.copy(joint)
        dots[2].position.copy(tip)
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
