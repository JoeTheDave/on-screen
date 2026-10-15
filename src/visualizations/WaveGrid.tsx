import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js'
import {
  CUBE_SIZE,
  DISC_RADIUS,
  HEIGHT_LIMIT,
  PHYSICS_STEP,
  createField,
  createLattice,
  createSpheres,
  sampleField,
  stepSpheres,
  updateField,
} from './waveGridMath'

// Configuration constants (wave and physics tuning lives in waveGridMath.ts)
const MIN_HUE_RANGE = 0.2 // hue shift from trough to peak, as a fraction of the wheel
const MAX_HUE_RANGE = 0.45
const MIN_PALETTE_SECONDS = 20 // time between palette changes
const MAX_PALETTE_SECONDS = 40
const PALETTE_FADE_SECONDS = 8
const SPHERE_HUE_SPREAD = 0.3 // spheres sit around the palette's complement, this wide
const MAX_FRAME_SECONDS = 1 / 20 // a backgrounded tab resumes with one short step, not a leap
const CAMERA_ORBIT_SECONDS = 40 // one full turn around the grid
const CAMERA_ELEVATION_DEG = 35 // angle looking down onto the plane
const FIT_RADIUS = DISC_RADIUS + CUBE_SIZE * 0.7 // disc plus a cube's half-diagonal

interface Palette {
  baseHue: number
  hueRange: number // signed: the gradient can run either way around the wheel
}

const randomPalette = (): Palette => ({
  baseHue: Math.random(),
  hueRange: (Math.random() < 0.5 ? -1 : 1) * (MIN_HUE_RANGE + Math.random() * (MAX_HUE_RANGE - MIN_HUE_RANGE)),
})

const wrap01 = (x: number) => ((x % 1) + 1) % 1

// Blend two palettes, taking the short way around the hue wheel
const mixPalette = (a: Palette, b: Palette, f: number): Palette => {
  let dh = wrap01(b.baseHue - a.baseHue)
  if (dh > 0.5) dh -= 1
  return { baseHue: wrap01(a.baseHue + dh * f), hueRange: a.hueRange + (b.hueRange - a.hueRange) * f }
}

export default function WaveGrid() {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!containerRef.current) return
    const container = containerRef.current

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x000000)

    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 500)

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(window.innerWidth, window.innerHeight)
    container.appendChild(renderer.domElement)

    scene.add(new THREE.HemisphereLight(0xffffff, 0x202030, 1.2))
    const sun = new THREE.DirectionalLight(0xffffff, 1.6)
    sun.position.set(30, 60, 20)
    scene.add(sun)

    // Cubes: the square lattice, trimmed to a disc, one instance per surviving cube
    const lattice = createLattice()
    const count = lattice.count
    const geometry = new THREE.BoxGeometry(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
    const material = new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0.1 })
    const cubes = new THREE.InstancedMesh(geometry, material, count)
    cubes.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
    scene.add(cubes)

    // Spheres: glossy, with a room environment map on their material only so they have
    // something to reflect, while the cubes keep the original flat lighting
    const pmrem = new THREE.PMREMGenerator(renderer)
    const room = new RoomEnvironment()
    const envMap = pmrem.fromScene(room, 0.04).texture
    room.dispose()
    pmrem.dispose()
    const spheres = createSpheres()
    const sphereGeometry = new THREE.SphereGeometry(1, 32, 16)
    const sphereMaterial = new THREE.MeshPhysicalMaterial({
      roughness: 0.12,
      metalness: 0.15,
      clearcoat: 1,
      clearcoatRoughness: 0.05,
      envMap,
      envMapIntensity: 0.7,
    })
    const sphereMesh = new THREE.InstancedMesh(sphereGeometry, sphereMaterial, spheres.count)
    sphereMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
    // Instanced bounds are computed once and would go stale as the spheres fly about
    sphereMesh.frustumCulled = false
    scene.add(sphereMesh)
    const sphereHueOffset = Array.from({ length: spheres.count }, () => (Math.random() - 0.5) * SPHERE_HUE_SPREAD)
    const sphereLightness = Array.from({ length: spheres.count }, () => 0.45 + Math.random() * 0.2)

    // Waves and palette evolve on simulation time, which (like the physics) only advances by
    // clamped frame steps, so nothing lurches after the tab has been in the background
    let simTime = 0
    const field = createField(Math.random, simTime)
    let paletteFrom = randomPalette()
    let paletteTo = paletteFrom
    let paletteFadeStart = -Infinity
    let nextPaletteAt = MIN_PALETTE_SECONDS + Math.random() * (MAX_PALETTE_SECONDS - MIN_PALETTE_SECONDS)

    let cameraDistance = 0
    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight
      camera.updateProjectionMatrix()
      renderer.setSize(window.innerWidth, window.innerHeight)
      const vFov = THREE.MathUtils.degToRad(camera.fov)
      const hFov = 2 * Math.atan(Math.tan(vFov / 2) * camera.aspect)
      // Smallest distance that keeps the disc (at peak and trough height) in frame, with
      // perspective: nearer points need more room. The disc is rotation-invariant, so one fit
      // holds for the whole orbit. Camera frame at azimuth 0: toward-camera u, screen-up v.
      const el = THREE.MathUtils.degToRad(CAMERA_ELEVATION_DEG)
      const tanV = Math.tan(vFov / 2)
      const tanH = Math.tan(hFov / 2)
      const reach = HEIGHT_LIMIT + CUBE_SIZE / 2
      cameraDistance = 0
      for (let i = 0; i < 64; i++) {
        const a = (i / 64) * Math.PI * 2
        const px = FIT_RADIUS * Math.cos(a)
        const pz = FIT_RADIUS * Math.sin(a)
        for (const py of [-reach, reach]) {
          const along = px * Math.cos(el) + py * Math.sin(el)
          const up = -px * Math.sin(el) + py * Math.cos(el)
          cameraDistance = Math.max(cameraDistance, along + Math.abs(up) / tanV, along + Math.abs(pz) / tanH)
        }
      }
    }
    handleResize()

    const matrix = new THREE.Matrix4()
    const position = new THREE.Vector3()
    const scale = new THREE.Vector3()
    const identity = new THREE.Quaternion()
    const color = new THREE.Color()
    const elevation = THREE.MathUtils.degToRad(CAMERA_ELEVATION_DEG)
    let lastTime = performance.now()
    let physicsDebt = 0 // real seconds not yet simulated
    let animationId = 0

    const animate = () => {
      animationId = requestAnimationFrame(animate)
      const now = performance.now()
      const frameSeconds = Math.min((now - lastTime) / 1000, MAX_FRAME_SECONDS)
      lastTime = now

      // Fixed 120 Hz substeps; the field is advanced in lockstep so the frame shows the cubes
      // at exactly the time the spheres were last resolved against
      physicsDebt += frameSeconds
      updateField(field, simTime)
      while (physicsDebt >= PHYSICS_STEP) {
        physicsDebt -= PHYSICS_STEP
        simTime += PHYSICS_STEP
        stepSpheres(spheres, lattice, field, simTime)
      }
      const t = simTime

      const paletteMix = () => THREE.MathUtils.smoothstep((t - paletteFadeStart) / PALETTE_FADE_SECONDS, 0, 1)
      if (t >= nextPaletteAt) {
        // Start the next crossfade from whatever is on screen right now
        paletteFrom = mixPalette(paletteFrom, paletteTo, paletteMix())
        paletteTo = randomPalette()
        paletteFadeStart = t
        nextPaletteAt = t + MIN_PALETTE_SECONDS + Math.random() * (MAX_PALETTE_SECONDS - MIN_PALETTE_SECONDS)
      }
      const { baseHue, hueRange } = mixPalette(paletteFrom, paletteTo, paletteMix())

      for (let i = 0; i < count; i++) {
        const x = lattice.baseX[i]
        const z = lattice.baseZ[i]
        const height = sampleField(field, x, z, t)
        matrix.makeTranslation(x, height, z)
        cubes.setMatrixAt(i, matrix)

        // Trough -> peak maps across the palette's hue range, brightening toward the peaks
        const level = (height + HEIGHT_LIMIT) / (2 * HEIGHT_LIMIT)
        color.setHSL(wrap01(baseHue + level * hueRange), 0.75, 0.35 + level * 0.25)
        cubes.setColorAt(i, color)
      }
      cubes.instanceMatrix.needsUpdate = true
      if (cubes.instanceColor) cubes.instanceColor.needsUpdate = true

      // Spheres take the complementary side of the wheel, each a little apart from the rest
      const sphereBase = baseHue + hueRange / 2 + 0.5
      for (let i = 0; i < spheres.count; i++) {
        const r = spheres.radius[i]
        position.set(spheres.x[i], spheres.y[i], spheres.z[i])
        scale.set(r, r, r)
        matrix.compose(position, identity, scale)
        sphereMesh.setMatrixAt(i, matrix)
        color.setHSL(wrap01(sphereBase + sphereHueOffset[i]), 0.8, sphereLightness[i])
        sphereMesh.setColorAt(i, color)
      }
      sphereMesh.instanceMatrix.needsUpdate = true
      if (sphereMesh.instanceColor) sphereMesh.instanceColor.needsUpdate = true

      const azimuth = (t / CAMERA_ORBIT_SECONDS) * Math.PI * 2
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
      cubes.dispose()
      geometry.dispose()
      material.dispose()
      sphereMesh.dispose()
      sphereGeometry.dispose()
      sphereMaterial.dispose()
      envMap.dispose()
      renderer.dispose()
      container.removeChild(renderer.domElement)
    }
  }, [])

  return <div ref={containerRef} className="w-full h-full" />
}
