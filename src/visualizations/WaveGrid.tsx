import { useEffect, useRef } from 'react'
import * as THREE from 'three'

// Configuration constants
const GRID_SIZE = 50 // cubes per side
const CUBE_SIZE = 1
const SPACING = 1.2 // center-to-center distance between cubes
const WAVE_AMPLITUDE = 2.2 // per wave; the two add, so peaks reach twice this
const MIN_WAVELENGTH = 14
const MAX_WAVELENGTH = 30
const MIN_WAVE_SPEED = 4 // units per second
const MAX_WAVE_SPEED = 9
const MIN_WAVE_ANGLE_DEG = 45 // keep the two wave directions apart
const HUE_RANGE = 0.35 // hue shift from trough to peak, as a fraction of the wheel
const CAMERA_ORBIT_SECONDS = 40 // one full turn around the grid
const CAMERA_ELEVATION_DEG = 35 // angle looking down onto the plane
const FIT_RADIUS = 34 // grid radius framed (half-width 30, corners ~42 may graze the edges)

interface Wave {
  dirX: number
  dirZ: number
  k: number // wavenumber, 2π / wavelength
  omega: number // angular frequency
  phase: number
}

const randomBetween = (min: number, max: number) => min + Math.random() * (max - min)

const createWaves = (): [Wave, Wave] => {
  const angle1 = Math.random() * Math.PI * 2
  const minGap = (MIN_WAVE_ANGLE_DEG * Math.PI) / 180
  // Second direction at least minGap away from the first (and from its opposite)
  const angle2 = angle1 + randomBetween(minGap, Math.PI - minGap)
  return [angle1, angle2].map(angle => {
    const wavelength = randomBetween(MIN_WAVELENGTH, MAX_WAVELENGTH)
    const k = (Math.PI * 2) / wavelength
    return {
      dirX: Math.cos(angle),
      dirZ: Math.sin(angle),
      k,
      omega: k * randomBetween(MIN_WAVE_SPEED, MAX_WAVE_SPEED),
      phase: Math.random() * Math.PI * 2,
    }
  }) as [Wave, Wave]
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

    const count = GRID_SIZE * GRID_SIZE
    const geometry = new THREE.BoxGeometry(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
    const material = new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0.1 })
    const cubes = new THREE.InstancedMesh(geometry, material, count)
    cubes.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
    scene.add(cubes)

    // Base x/z of each cube, centered on the origin
    const half = ((GRID_SIZE - 1) * SPACING) / 2
    const baseX = new Float32Array(count)
    const baseZ = new Float32Array(count)
    for (let row = 0; row < GRID_SIZE; row++) {
      for (let col = 0; col < GRID_SIZE; col++) {
        baseX[row * GRID_SIZE + col] = col * SPACING - half
        baseZ[row * GRID_SIZE + col] = row * SPACING - half
      }
    }

    const waves = createWaves()
    const baseHue = Math.random()

    let cameraDistance = 0
    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight
      camera.updateProjectionMatrix()
      renderer.setSize(window.innerWidth, window.innerHeight)
      const vFov = THREE.MathUtils.degToRad(camera.fov)
      const hFov = 2 * Math.atan(Math.tan(vFov / 2) * camera.aspect)
      // Smallest distance that keeps a circle of FIT_RADIUS (at peak and trough height) in frame,
      // with perspective: nearer points need more room. The circle is rotation-invariant, so
      // one fit holds for the whole orbit. Camera frame at azimuth 0: toward-camera u, screen-up v.
      const el = THREE.MathUtils.degToRad(CAMERA_ELEVATION_DEG)
      const tanV = Math.tan(vFov / 2)
      const tanH = Math.tan(hFov / 2)
      cameraDistance = 0
      for (let i = 0; i < 64; i++) {
        const a = (i / 64) * Math.PI * 2
        const px = FIT_RADIUS * Math.cos(a)
        const pz = FIT_RADIUS * Math.sin(a)
        for (const py of [-2 * WAVE_AMPLITUDE, 2 * WAVE_AMPLITUDE]) {
          const along = px * Math.cos(el) + py * Math.sin(el)
          const up = -px * Math.sin(el) + py * Math.cos(el)
          cameraDistance = Math.max(cameraDistance, along + Math.abs(up) / tanV, along + Math.abs(pz) / tanH)
        }
      }
    }
    handleResize()

    const matrix = new THREE.Matrix4()
    const color = new THREE.Color()
    const elevation = THREE.MathUtils.degToRad(CAMERA_ELEVATION_DEG)
    const startTime = performance.now()
    let animationId = 0

    const animate = () => {
      animationId = requestAnimationFrame(animate)
      const t = (performance.now() - startTime) / 1000

      for (let i = 0; i < count; i++) {
        const x = baseX[i]
        const z = baseZ[i]
        let height = 0
        for (const w of waves) {
          height += WAVE_AMPLITUDE * Math.sin(w.k * (w.dirX * x + w.dirZ * z) - w.omega * t + w.phase)
        }
        matrix.makeTranslation(x, height, z)
        cubes.setMatrixAt(i, matrix)

        // Trough -> peak maps across HUE_RANGE of the wheel, brightening toward the peaks
        const level = (height + 2 * WAVE_AMPLITUDE) / (4 * WAVE_AMPLITUDE)
        color.setHSL((baseHue + level * HUE_RANGE) % 1, 0.75, 0.35 + level * 0.25)
        cubes.setColorAt(i, color)
      }
      cubes.instanceMatrix.needsUpdate = true
      if (cubes.instanceColor) cubes.instanceColor.needsUpdate = true

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
      renderer.dispose()
      container.removeChild(renderer.domElement)
    }
  }, [])

  return <div ref={containerRef} className="w-full h-full" />
}
