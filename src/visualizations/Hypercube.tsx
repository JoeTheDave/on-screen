import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js'
import { LineSegments2 } from 'three/examples/jsm/lines/LineSegments2.js'
import { LineSegmentsGeometry } from 'three/examples/jsm/lines/LineSegmentsGeometry.js'
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js'
import { buildPolytopes, projectVertices, rotation4, type Polytope } from './polytopeMath'

// Configuration constants
const POLYTOPE_SECONDS = 40 // time on each polytope before the next one takes over
const TRANSITION_SECONDS = 5 // crossfade: old one swells and fades while the new one grows in
const FIT_RADIUS = 1.2 // projected shapes stay within ~1.3; a little overflow at the extremes is fine
const CAMERA_FOV = 45
const CAMERA_ORBIT_SECONDS = 70
const CAMERA_BOB_SECONDS = 29
const CAMERA_BOB_RADIANS = 0.35
const EYE_MIN = 1.7 // 4D eye distance on the w axis; closer = more dramatic inside-out swelling
const EYE_MAX = 2.4
const EYE_BREATH = 0.2 // the eye drifts in and out by this much...
const EYE_BREATH_SECONDS = 19 // ...over this period, so the distortion itself breathes
const NODE_WORLD_SIZE = 0.16 // sprite diameter (core + halo) at w = 0
const REFERENCE_HEIGHT = 1080 // line widths are tuned for a 1080p screen and scale with it

// Per-polytope look: denser polytopes get thinner, dimmer lines so additive overlap doesn't blow out
const STYLES: Record<string, { lineWidth: number; lineOpacity: number; nodeSize: number }> = {
  '8-cell': { lineWidth: 3.2, lineOpacity: 0.95, nodeSize: 1.0 },
  '16-cell': { lineWidth: 3.2, lineOpacity: 0.95, nodeSize: 1.0 },
  '24-cell': { lineWidth: 2.6, lineOpacity: 0.85, nodeSize: 0.8 },
  '600-cell': { lineWidth: 1.6, lineOpacity: 0.55, nodeSize: 0.45 },
  '120-cell': { lineWidth: 1.4, lineOpacity: 0.5, nodeSize: 0.35 },
}

// Three-stop gradients along 4D depth: far side of w (dim) -> middle -> near side (bright)
const PALETTES: [string, string, string][] = [
  ['#2a1b8e', '#d63aff', '#7ff8ff'], // ultraviolet -> magenta -> cyan
  ['#4a0b2e', '#ff4d2e', '#ffd88a'], // ember
  ['#0b2a6a', '#19e3a0', '#eaffa0'], // aurora
  ['#1b2a7b', '#4aa8ff', '#f0f6ff'], // ice
  ['#3a0a6c', '#ff2e88', '#ffb36b'], // synthwave
  ['#083d3a', '#2fd6b5', '#ffe08a'], // jade and gold
  ['#300a50', '#7a5cff', '#ff9ef5'], // lavender
]

// Glowing round node: hot white-tinted core with a soft colored halo
const NODE_VERTEX_SHADER = /* glsl */ `
attribute vec3 aColor;
attribute float aGlow;
uniform float uSize;
uniform float uOpacity;
varying vec3 vColor;
varying float vAlpha;
void main() {
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mvPosition;
  gl_PointSize = max(uSize * aGlow / max(-mvPosition.z, 0.001), 1.0);
  vColor = aColor;
  vAlpha = uOpacity;
}
`

const NODE_FRAGMENT_SHADER = /* glsl */ `
varying vec3 vColor;
varying float vAlpha;
void main() {
  vec2 p = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(p, p);
  if (r2 > 1.0) discard;
  float core = exp(-r2 * 28.0);
  float halo = exp(-r2 * 5.0) * 0.45;
  vec3 c = vColor * (core * 1.8 + halo) + vec3(core * 0.5);
  gl_FragColor = vec4(c * vAlpha, 1.0);
}
`

interface Slot {
  group: THREE.Group
  lineGeometry: LineSegmentsGeometry
  linePositions: Float32Array
  lineColors: Float32Array
  lineBuffers: THREE.InterleavedBuffer[]
  lineMaterial: LineMaterial
  pointGeometry: THREE.BufferGeometry
  pointPositions: Float32Array
  pointColors: Float32Array
  pointGlow: Float32Array
  pointAttrs: THREE.BufferAttribute[]
  pointMaterial: THREE.ShaderMaterial
  // Per-cycle state
  polytope: Polytope | null
  palette: THREE.Color[]
  angles: Float64Array
  rates: Float64Array
  eyeBase: number
  eyePhase: number
  start: number // ms
  fadeOutStart: number // ms, Infinity while this is the live polytope
}

export default function Hypercube() {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!containerRef.current) return
    const container = containerRef.current

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x000000)

    const camera = new THREE.PerspectiveCamera(CAMERA_FOV, window.innerWidth / window.innerHeight, 0.01, 100)

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(window.innerWidth, window.innerHeight)
    container.appendChild(renderer.domElement)

    const composer = new EffectComposer(renderer)
    const renderPass = new RenderPass(scene, camera)
    const bloomPass = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 1.1, 0.55, 0.05)
    // OutputPass does the linear -> sRGB conversion, so the palette hexes read as written
    const outputPass = new OutputPass()
    composer.addPass(renderPass)
    composer.addPass(bloomPass)
    composer.addPass(outputPass)

    // ~15 ms: all five polytopes, including the 120-cell derived from the 600-cell's cells
    const polytopes = buildPolytopes()
    const maxVertices = Math.max(...polytopes.map(p => p.vertexCount))
    const maxEdges = Math.max(...polytopes.map(p => p.edgeCount))

    // Projection scratch, sized for the largest polytope and reused every frame
    const projected = new Float32Array(maxVertices * 3)
    const depthW = new Float32Array(maxVertices)
    const vertexColors = new Float32Array(maxVertices * 3)
    const matrix4d = new Float64Array(16)

    let lineScale = 1 // CSS pixels per reference pixel
    let nodePixelScale = 1 // device pixels per world unit at distance 1

    // Two slots, each sized for the largest polytope once, so cycling never allocates GPU
    // buffers: one is live, the other holds the outgoing polytope during a crossfade.
    const makeSlot = (): Slot => {
      const group = new THREE.Group()

      const linePositions = new Float32Array(maxEdges * 6)
      const lineColors = new Float32Array(maxEdges * 6)
      const lineGeometry = new LineSegmentsGeometry()
      lineGeometry.setPositions(linePositions)
      lineGeometry.setColors(lineColors)
      const lineBuffers = [
        (lineGeometry.getAttribute('instanceStart') as THREE.InterleavedBufferAttribute).data,
        (lineGeometry.getAttribute('instanceColorStart') as THREE.InterleavedBufferAttribute).data,
      ]
      lineBuffers.forEach(b => b.setUsage(THREE.DynamicDrawUsage))
      const lineMaterial = new LineMaterial({
        vertexColors: true,
        worldUnits: false,
        linewidth: 2,
        transparent: true,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      })
      const lines = new LineSegments2(lineGeometry, lineMaterial)
      lines.frustumCulled = false
      group.add(lines)

      const pointPositions = new Float32Array(maxVertices * 3)
      const pointColors = new Float32Array(maxVertices * 3)
      const pointGlow = new Float32Array(maxVertices)
      const pointGeometry = new THREE.BufferGeometry()
      const pointAttrs = [
        new THREE.BufferAttribute(pointPositions, 3),
        new THREE.BufferAttribute(pointColors, 3),
        new THREE.BufferAttribute(pointGlow, 1),
      ]
      pointAttrs.forEach(a => a.setUsage(THREE.DynamicDrawUsage))
      pointGeometry.setAttribute('position', pointAttrs[0])
      pointGeometry.setAttribute('aColor', pointAttrs[1])
      pointGeometry.setAttribute('aGlow', pointAttrs[2])
      const pointMaterial = new THREE.ShaderMaterial({
        vertexShader: NODE_VERTEX_SHADER,
        fragmentShader: NODE_FRAGMENT_SHADER,
        uniforms: { uSize: { value: 1 }, uOpacity: { value: 1 } },
        transparent: true,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      })
      const points = new THREE.Points(pointGeometry, pointMaterial)
      points.frustumCulled = false
      group.add(points)

      group.visible = false
      scene.add(group)

      return {
        group,
        lineGeometry,
        linePositions,
        lineColors,
        lineBuffers,
        lineMaterial,
        pointGeometry,
        pointPositions,
        pointColors,
        pointGlow,
        pointAttrs,
        pointMaterial,
        polytope: null,
        palette: [],
        angles: new Float64Array(6),
        rates: new Float64Array(6),
        eyeBase: 2,
        eyePhase: 0,
        start: 0,
        fadeOutStart: Infinity,
      }
    }
    const slots = [makeSlot(), makeSlot()]

    const randomSign = () => (Math.random() < 0.5 ? -1 : 1)
    let paletteIndex = Math.floor(Math.random() * PALETTES.length)
    let polytopeIndex = Math.floor(Math.random() * polytopes.length)
    let liveSlot = 0

    const startPolytope = (slot: Slot, polytope: Polytope, now: number) => {
      // A different palette each time, never the same one twice in a row
      paletteIndex = (paletteIndex + 1 + Math.floor(Math.random() * (PALETTES.length - 1))) % PALETTES.length
      slot.palette = PALETTES[paletteIndex].map(hex => new THREE.Color(hex))
      slot.polytope = polytope
      // Planes are [xy, xz, xw, yz, yw, zw]. The headline motion is a double rotation in
      // the orthogonal XW and YZ planes (cells pass through each other and turn inside out);
      // the other four planes drift slowly so the pattern never exactly repeats.
      const primary = 0.16 + Math.random() * 0.14
      slot.rates[2] = primary * randomSign()
      slot.rates[3] = primary * (0.5 + Math.random() * 0.9) * randomSign()
      for (const p of [0, 1, 4, 5]) slot.rates[p] = (0.015 + Math.random() * 0.045) * randomSign()
      for (let p = 0; p < 6; p++) slot.angles[p] = Math.random() * Math.PI * 2
      slot.eyeBase = EYE_MIN + Math.random() * (EYE_MAX - EYE_MIN)
      slot.eyePhase = Math.random() * Math.PI * 2
      slot.start = now
      slot.fadeOutStart = Infinity
      slot.group.visible = true

      const style = STYLES[polytope.name]
      slot.lineGeometry.instanceCount = polytope.edgeCount
      slot.pointGeometry.setDrawRange(0, polytope.vertexCount)
      slot.lineMaterial.linewidth = style.lineWidth * lineScale
    }

    const smoothstep = (x: number) => {
      const t = Math.min(Math.max(x, 0), 1)
      return t * t * (3 - 2 * t)
    }

    const color = new THREE.Color()
    const gradient = (palette: THREE.Color[], t: number, out: THREE.Color) => {
      if (t < 0.5) out.lerpColors(palette[0], palette[1], t * 2)
      else out.lerpColors(palette[1], palette[2], (t - 0.5) * 2)
      return out
    }

    const updateSlot = (slot: Slot, now: number, dt: number) => {
      const polytope = slot.polytope
      if (!polytope || !slot.group.visible) return
      const style = STYLES[polytope.name]

      for (let p = 0; p < 6; p++) slot.angles[p] += slot.rates[p] * dt
      rotation4(slot.angles, matrix4d)
      const seconds = (now - slot.start) / 1000
      const eye = slot.eyeBase + EYE_BREATH * Math.sin((seconds / EYE_BREATH_SECONDS) * Math.PI * 2 + slot.eyePhase)
      const n = polytope.vertexCount
      projectVertices(polytope.vertices, n, matrix4d, eye, projected, depthW)

      // Color and size by 4D depth: the far side of w is dim and deep, the near side hot
      for (let v = 0; v < n; v++) {
        const t = (depthW[v] + 1) / 2
        const brightness = 0.22 + 0.78 * t * Math.sqrt(t)
        gradient(slot.palette, t, color)
        vertexColors[v * 3] = color.r * brightness
        vertexColors[v * 3 + 1] = color.g * brightness
        vertexColors[v * 3 + 2] = color.b * brightness
        slot.pointGlow[v] = style.nodeSize * (0.45 + 0.75 * t) * (eye / (eye - depthW[v]))
      }
      slot.pointPositions.set(projected.subarray(0, n * 3))
      slot.pointColors.set(vertexColors.subarray(0, n * 3))

      const edges = polytope.edges
      for (let e = 0; e < polytope.edgeCount; e++) {
        const a = edges[e * 2] * 3
        const b = edges[e * 2 + 1] * 3
        const o = e * 6
        for (let k = 0; k < 3; k++) {
          slot.linePositions[o + k] = projected[a + k]
          slot.linePositions[o + 3 + k] = projected[b + k]
          slot.lineColors[o + k] = vertexColors[a + k]
          slot.lineColors[o + 3 + k] = vertexColors[b + k]
        }
      }
      slot.lineBuffers.forEach(b => (b.needsUpdate = true))
      slot.pointAttrs.forEach(a => (a.needsUpdate = true))

      // Crossfade: grow in from the center; on the way out, swell past the viewer and fade
      const fadeIn = smoothstep(seconds / TRANSITION_SECONDS)
      const fadeOut = smoothstep((now - slot.fadeOutStart) / 1000 / TRANSITION_SECONDS)
      if (fadeOut >= 1) {
        slot.group.visible = false
        return
      }
      const opacity = fadeIn * (1 - fadeOut)
      slot.group.scale.setScalar((0.35 + 0.65 * fadeIn) * (1 + 0.8 * fadeOut))
      slot.lineMaterial.opacity = style.lineOpacity * opacity
      slot.lineMaterial.linewidth = style.lineWidth * lineScale
      slot.pointMaterial.uniforms.uOpacity.value = opacity
      slot.pointMaterial.uniforms.uSize.value = NODE_WORLD_SIZE * nodePixelScale
    }

    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight
      camera.updateProjectionMatrix()
      renderer.setSize(window.innerWidth, window.innerHeight)
      composer.setSize(window.innerWidth, window.innerHeight)
      const pixelHeight = window.innerHeight * renderer.getPixelRatio()
      // LineMaterial (worldUnits: false) measures width against its resolution uniform, which
      // LineSegments2 fills from renderer.getViewport(): CSS pixels, not drawing-buffer pixels
      lineScale = window.innerHeight / REFERENCE_HEIGHT
      nodePixelScale = pixelHeight / (2 * Math.tan(THREE.MathUtils.degToRad(CAMERA_FOV) / 2))
    }
    handleResize()

    const startTime = performance.now()
    let lastTime = startTime
    let cycleStart = startTime
    startPolytope(slots[liveSlot], polytopes[polytopeIndex], startTime)
    let animationId = 0

    const animate = () => {
      animationId = requestAnimationFrame(animate)
      const now = performance.now()
      // Clamp so a backgrounded tab doesn't lurch on return
      const dt = Math.min((now - lastTime) / 1000, 1 / 20)
      lastTime = now

      if ((now - cycleStart) / 1000 >= POLYTOPE_SECONDS) {
        cycleStart = now
        slots[liveSlot].fadeOutStart = now
        liveSlot = 1 - liveSlot
        polytopeIndex = (polytopeIndex + 1) % polytopes.length
        startPolytope(slots[liveSlot], polytopes[polytopeIndex], now)
      }

      for (const slot of slots) updateSlot(slot, now, dt)

      const t = (now - startTime) / 1000
      const vFov = THREE.MathUtils.degToRad(camera.fov)
      const hFov = 2 * Math.atan(Math.tan(vFov / 2) * camera.aspect)
      const distance = FIT_RADIUS / Math.sin(Math.min(vFov, hFov) / 2)
      const azimuth = (t / CAMERA_ORBIT_SECONDS) * Math.PI * 2
      const elevation = 0.3 + Math.sin((t / CAMERA_BOB_SECONDS) * Math.PI * 2) * CAMERA_BOB_RADIANS
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
      for (const slot of slots) {
        slot.lineGeometry.dispose()
        slot.lineMaterial.dispose()
        slot.pointGeometry.dispose()
        slot.pointMaterial.dispose()
      }
      renderPass.dispose()
      bloomPass.dispose()
      outputPass.dispose()
      composer.dispose()
      renderer.dispose()
      container.removeChild(renderer.domElement)
    }
  }, [])

  return <div ref={containerRef} className="w-full h-full" />
}
