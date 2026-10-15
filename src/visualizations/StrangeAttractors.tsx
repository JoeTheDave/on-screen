import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'
import { ATTRACTORS, step, warmUp, type AttractorStats } from './attractors'

// Configuration constants
const PARTICLE_COUNT = 20000
// A real halo around every streak, but over normal-blended particles: a clump's brightness is
// capped at its own color, so the glow can't snowball into a white blob
const BLOOM_STRENGTH = 0.75
const BLOOM_RADIUS = 0.55
const BLOOM_THRESHOLD = 0.25
const ATTRACTOR_SECONDS = 45 // time on each attractor before the equations switch
const WORLD_SPEED = 0.35 // mean particle speed, in attractor radii per second
const TRAIL_SECONDS = 0.15 // streak length, as seconds of travel
const ESCAPE_RADIUS = 4 // particles beyond this (in attractor radii) respawn on the attractor
const FIT_RADIUS = 1.2
const CAMERA_ORBIT_SECONDS = 20
const CAMERA_BOB_SECONDS = 23
const CAMERA_BOB_RADIANS = 0.4

interface Active {
  index: number
  stats: AttractorStats
  timeScale: number // native time per real second
  hueSlow: number
  hueFast: number
}

export default function StrangeAttractors() {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!containerRef.current) return
    const container = containerRef.current

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x000000)

    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.01, 100)

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(window.innerWidth, window.innerHeight)
    container.appendChild(renderer.domElement)

    const composer = new EffectComposer(renderer)
    composer.addPass(new RenderPass(scene, camera))
    composer.addPass(
      new UnrealBloomPass(
        new THREE.Vector2(window.innerWidth, window.innerHeight),
        BLOOM_STRENGTH,
        BLOOM_RADIUS,
        BLOOM_THRESHOLD,
      ),
    )

    // Each particle is one segment: head at its position, tail trailing along its velocity
    const segmentPositions = new Float32Array(PARTICLE_COUNT * 6)
    // RGBA per vertex: the tail fades out via alpha (a dark tail color would smudge under normal blending)
    const segmentColors = new Float32Array(PARTICLE_COUNT * 8)
    const geometry = new THREE.BufferGeometry()
    const positionAttr = new THREE.BufferAttribute(segmentPositions, 3).setUsage(THREE.DynamicDrawUsage)
    const colorAttr = new THREE.BufferAttribute(segmentColors, 4).setUsage(THREE.DynamicDrawUsage)
    geometry.setAttribute('position', positionAttr)
    geometry.setAttribute('color', colorAttr)
    const material = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      // Normal (not additive) blending: a lone streak shows its full color, but thousands
      // overlapping saturate to that color instead of summing to a white blob
      opacity: 0.55,
      blending: THREE.NormalBlending,
      depthWrite: false,
    })
    const streaks = new THREE.LineSegments(geometry, material)
    streaks.frustumCulled = false
    scene.add(streaks)

    // Measure every attractor once up front (~100 ms) so each is framed to unit radius
    const allStats = ATTRACTORS.map(a => warmUp(a))

    // Particle state lives in the active attractor's native coordinates
    const native = new Float64Array(PARTICLE_COUNT * 3)
    const velocity = new Float64Array(3)

    // World <-> native: center, scale to unit radius, and map native z to world up (y)
    const toWorld = (stats: AttractorStats, i: number, out: THREE.Vector3) => {
      const r = stats.radius
      const [cx, cy, cz] = stats.center
      out.set((native[i] - cx) / r, (native[i + 2] - cz) / r, (native[i + 1] - cy) / r)
    }
    const fromWorld = (stats: AttractorStats, i: number, w: THREE.Vector3) => {
      const r = stats.radius
      const [cx, cy, cz] = stats.center
      native[i] = w.x * r + cx
      native[i + 1] = w.z * r + cy
      native[i + 2] = w.y * r + cz
    }

    const respawn = (stats: AttractorStats, i: number) => {
      const poolCount = stats.pool.length / 3
      const j = Math.floor(Math.random() * poolCount) * 3
      const jitter = stats.radius * 0.01
      native[i] = stats.pool[j] + (Math.random() - 0.5) * jitter
      native[i + 1] = stats.pool[j + 1] + (Math.random() - 0.5) * jitter
      native[i + 2] = stats.pool[j + 2] + (Math.random() - 0.5) * jitter
    }

    const activate = (index: number): Active => {
      const stats = allStats[index]
      const hueSlow = Math.random()
      return {
        index,
        stats,
        timeScale: (WORLD_SPEED * stats.radius) / stats.meanSpeed,
        hueSlow,
        hueFast: hueSlow + (Math.random() < 0.5 ? 1 : -1) * (0.3 + Math.random() * 0.2),
      }
    }

    // First attractor: start everyone on it. Later switches keep the cloud where it is.
    let active = activate(Math.floor(Math.random() * ATTRACTORS.length))
    for (let i = 0; i < PARTICLE_COUNT * 3; i += 3) respawn(active.stats, i)

    const switchAttractor = () => {
      let next = Math.floor(Math.random() * (ATTRACTORS.length - 1))
      if (next >= active.index) next++
      const previous = active.stats
      active = activate(next)
      // Re-express each particle's on-screen position in the new system's coordinates;
      // the new equations then pull the old shape onto the new attractor
      const w = new THREE.Vector3()
      for (let i = 0; i < PARTICLE_COUNT * 3; i += 3) {
        toWorld(previous, i, w)
        fromWorld(active.stats, i, w)
      }
    }

    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight
      camera.updateProjectionMatrix()
      renderer.setSize(window.innerWidth, window.innerHeight)
      composer.setSize(window.innerWidth, window.innerHeight)
    }
    handleResize()

    const head = new THREE.Vector3()
    const color = new THREE.Color()
    const startTime = performance.now()
    let lastTime = startTime
    let attractorStart = startTime
    let animationId = 0

    const animate = () => {
      animationId = requestAnimationFrame(animate)
      const now = performance.now()
      // Clamp so a backgrounded tab doesn't take one giant (unstable) step on return
      const frameSeconds = Math.min((now - lastTime) / 1000, 1 / 20)
      lastTime = now

      if ((now - attractorStart) / 1000 >= ATTRACTOR_SECONDS) {
        attractorStart = now
        switchAttractor()
      }

      const { stats, timeScale, hueSlow, hueFast } = active
      const attractor = ATTRACTORS[active.index]
      const advance = frameSeconds * timeScale
      const substeps = Math.max(1, Math.ceil(advance / attractor.maxStep))
      const dt = advance / substeps
      const trailScale = (timeScale * TRAIL_SECONDS) / stats.radius
      const escapeSq = ESCAPE_RADIUS * ESCAPE_RADIUS

      for (let p = 0; p < PARTICLE_COUNT; p++) {
        const i = p * 3
        for (let s = 0; s < substeps; s++) step(attractor.derivative, native, i, dt)

        toWorld(stats, i, head)
        if (!(head.lengthSq() < escapeSq)) {
          // Escaped or NaN: put it back on the attractor
          respawn(stats, i)
          toWorld(stats, i, head)
        }

        attractor.derivative(native[i], native[i + 1], native[i + 2], velocity)
        // Velocity in world axes (same y/z swap as toWorld), scaled to the trail length
        const vx = velocity[0] * trailScale
        const vy = velocity[2] * trailScale
        const vz = velocity[1] * trailScale

        const o = p * 6
        segmentPositions[o] = head.x
        segmentPositions[o + 1] = head.y
        segmentPositions[o + 2] = head.z
        segmentPositions[o + 3] = head.x - vx
        segmentPositions[o + 4] = head.y - vy
        segmentPositions[o + 5] = head.z - vz

        // Slow -> fast across the palette; the head glows, the tail fades out
        const speed = Math.hypot(velocity[0], velocity[1], velocity[2]) / stats.meanSpeed
        const ratio = Math.min(speed / 1.8, 1)
        const hue = (((hueSlow + (hueFast - hueSlow) * ratio) % 1) + 1) % 1
        color.setHSL(hue, 0.85, 0.45 + ratio * 0.25)
        const c = p * 8
        segmentColors[c] = segmentColors[c + 4] = color.r
        segmentColors[c + 1] = segmentColors[c + 5] = color.g
        segmentColors[c + 2] = segmentColors[c + 6] = color.b
        segmentColors[c + 3] = 1
        segmentColors[c + 7] = 0
      }
      positionAttr.needsUpdate = true
      colorAttr.needsUpdate = true

      const t = (now - startTime) / 1000
      const vFov = THREE.MathUtils.degToRad(camera.fov)
      const hFov = 2 * Math.atan(Math.tan(vFov / 2) * camera.aspect)
      const distance = FIT_RADIUS / Math.sin(Math.min(vFov, hFov) / 2)
      const azimuth = (t / CAMERA_ORBIT_SECONDS) * Math.PI * 2
      const elevation = 0.25 + Math.sin((t / CAMERA_BOB_SECONDS) * Math.PI * 2) * CAMERA_BOB_RADIANS
      camera.position.set(
        distance * Math.cos(elevation) * Math.cos(azimuth),
        distance * Math.sin(elevation),
        distance * Math.cos(elevation) * Math.sin(azimuth),
      )
      camera.lookAt(0, 0, 0)

      composer.render()
    }

    animate()
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      cancelAnimationFrame(animationId)
      geometry.dispose()
      material.dispose()
      composer.dispose()
      renderer.dispose()
      container.removeChild(renderer.domElement)
    }
  }, [])

  return <div ref={containerRef} className="w-full h-full" />
}
