import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js'
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js'
import {
  BK7,
  DIAMOND,
  EXPERIMENT_KINDS,
  SF11,
  SODIUM_D_NM,
  WALL_RADIUS,
  WATER,
  createExperiment,
  createTraceBuffers,
  createTracer,
  emitRays,
  envelope,
  makeSpectrum,
  makeTraceElement,
  refractiveIndex,
  type Experiment,
  type ExperimentKind,
  type Glass,
  type Spectrum,
  type TraceElement,
  type TraceSettings,
  type WhiteRay,
} from './prismMath'

// Configuration constants (optics and the experiment scripts live in prismMath.ts)
const MAX_RENDER_PIXELS = 2_000_000 // cap on 4K screens: transmission + bloom + additive overdraw
const MAX_SEGMENTS = 6000 // beam segments per frame (one per straight run of one wavelength)
const MAX_SPOTS = 6000 // wall spots + glints on glass faces
const MAX_FRAME_SECONDS = 1 / 20
const BEAM_GAIN = 4.2 // brightness of a full white beam; dispersed colors carry their share
const FLOOR_HALF_WIDTH = 0.055 // where the light sheet grazes the table: core + halo
const FLOOR_GAIN = 1
const SHEET_HEIGHT = 0.3 // the slit lamp's sheet of light, scattering in the hazy air
const SHEET_GAIN = 0.11
const SPOT_HEIGHT = 0.36
const SPOT_GAIN = 1.5
const WALL_SPOT_HALF_WIDTH = 0.1
const GLINT = 0.1 // light scattered by a glass face where a beam crosses it
const GLINT_HALF_WIDTH = 0.025
const ELEMENT_HEIGHT = 0.7
const DROP_HEIGHT = 0.95 // the water cylinder stands a bit taller, like a beaker
const WALL_HEIGHT = 1.6
const HOLE_PITCH = 0.25 // optical breadboard: tapped holes on a square grid
const BENCH_FIT_RADIUS = 4.9
const CAMERA_ELEVATION_DEG = 31
const CAMERA_ELEVATION_SWAY_DEG = 4
const CAMERA_AZIMUTH_EASE = 0.25 // per second, toward each experiment's preferred view
const LAMP_LENGTH = 0.62
const LAMP_HEIGHT = 0.44
const APERTURE_GLOW = 7
const BLOOM_STRENGTH = 0.85
const BLOOM_RADIUS = 0.45
const BLOOM_THRESHOLD = 0.7
const LOOP_SECONDS = 3600 // camera sway clock wraps hourly so it never loses precision

// Abbe-number-based dispersion strength for three's MeshPhysicalMaterial (20 / V)
const GLASS_LOOK: { glass: Glass; dispersion: number; attenuation: number; distance: number }[] = [
  { glass: BK7, dispersion: 0.31, attenuation: 0xeef6ff, distance: 12 },
  { glass: SF11, dispersion: 0.78, attenuation: 0xfff6e4, distance: 8 },
  { glass: WATER, dispersion: 0.36, attenuation: 0xcfe9ff, distance: 5 },
  { glass: DIAMOND, dispersion: 0.36, attenuation: 0xffffff, distance: 50 },
]

const TRACE_SETTINGS: TraceSettings = {
  wallRadius: WALL_RADIUS - 0.02, // spots sit just in front of the wall
  maxDepth: 8,
  minWhiteEnergy: 0.003,
  minSpectralEnergy: 0.012,
  glint: GLINT,
  glintHalfWidth: GLINT_HALF_WIDTH,
  wallHalfWidth: WALL_SPOT_HALF_WIDTH,
}

// Beam segments: each instance is one straight run of light, drawn as two quads — a thin
// streak where the vertical light sheet grazes the table, and the sheet itself in the haze
const segmentVertexShader = /* glsl */ `
attribute vec4 aSeg;
attribute vec3 aColor;
uniform float uHalfWidth;
uniform float uSheetHeight;
varying vec3 vColor;
varying float vCoord;
varying float vPart;
void main() {
  vec2 a = aSeg.xy;
  vec2 b = aSeg.zw;
  vec2 d = b - a;
  float len = length(d);
  vec2 t = len > 1e-6 ? d / len : vec2(1.0, 0.0);
  vec2 n = vec2(-t.y, t.x);
  vec2 p = mix(a, b, position.x);
  vec3 world;
  if (position.z < 0.5) {
    p += n * position.y * uHalfWidth;
    world = vec3(p.x, 0.004, p.y);
  } else {
    world = vec3(p.x, position.y * uSheetHeight, p.y);
  }
  vColor = aColor;
  vCoord = position.y;
  vPart = position.z;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(world, 1.0);
}
`

const segmentFragmentShader = /* glsl */ `
uniform float uLevel;
uniform float uFloorGain;
uniform float uSheetGain;
varying vec3 vColor;
varying float vCoord;
varying float vPart;
void main() {
  float k;
  if (vPart < 0.5) {
    float a = vCoord * vCoord;
    k = uFloorGain * (exp(-a * 16.0) + 0.2 * exp(-a * 3.0));
  } else {
    float h = vCoord;
    k = uSheetGain * (1.0 - smoothstep(0.5, 1.0, h)) * (0.65 + 0.35 * exp(-h * 4.0));
  }
  gl_FragColor = vec4(vColor * (k * uLevel), 1.0);
}
`

// Spots: vertical bands of light standing on a surface (the screen wall, or a glass face the
// sheet passes through), aligned with that surface
const spotVertexShader = /* glsl */ `
attribute vec4 aSpot;
attribute vec4 aSpotColor;
uniform float uSpotHeight;
varying vec3 vColor;
varying vec2 vCoord;
void main() {
  vec2 p = aSpot.xy + aSpot.zw * (position.x * aSpotColor.w);
  vec3 world = vec3(p.x, position.y * uSpotHeight, p.y);
  vColor = aSpotColor.rgb;
  vCoord = position.xy;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(world, 1.0);
}
`

const spotFragmentShader = /* glsl */ `
uniform float uLevel;
uniform float uSpotGain;
varying vec3 vColor;
varying vec2 vCoord;
void main() {
  float across = exp(-vCoord.x * vCoord.x * 5.0);
  float up = (1.0 - smoothstep(0.55, 1.0, vCoord.y)) * smoothstep(0.0, 0.04, vCoord.y);
  gl_FragColor = vec4(vColor * (across * up * uSpotGain * uLevel), 1.0);
}
`

// Black anodized optical breadboard: a square grid of tapped holes under a dim overhead pool
const tableVertexShader = /* glsl */ `
varying vec2 vXZ;
void main() {
  vec4 w = modelMatrix * vec4(position, 1.0);
  vXZ = w.xz;
  gl_Position = projectionMatrix * viewMatrix * w;
}
`

const tableFragmentShader = /* glsl */ `
uniform float uRadius;
uniform float uPitch;
varying vec2 vXZ;
void main() {
  vec2 g = vXZ / uPitch;
  vec2 f = fract(g) - 0.5;
  float d = length(f);
  float aa = fwidth(d) + 1e-4;
  float hole = 1.0 - smoothstep(0.11 - aa, 0.11 + aa, d);
  float rim = smoothstep(0.11, 0.13 + aa, d) * (1.0 - smoothstep(0.13, 0.18 + aa, d));
  // Hide the grid where holes shrink below a few pixels instead of letting it moire
  vec2 gw = fwidth(g);
  float detail = 1.0 - smoothstep(0.08, 0.25, max(gw.x, gw.y));
  float r = length(vXZ) / uRadius;
  vec3 base = vec3(0.014, 0.0155, 0.019) * (1.0 - 0.6 * smoothstep(0.15, 1.0, r));
  vec3 col = base * (1.0 + detail * (0.7 * rim - 0.85 * hole));
  gl_FragColor = vec4(col, 1.0);
}
`

// The projection screen: a dark matte ring around the bench, seen from the inside
const wallVertexShader = /* glsl */ `
uniform float uHeight;
varying float vH;
void main() {
  vH = position.y / uHeight + 0.5;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`

const wallFragmentShader = /* glsl */ `
varying float vH;
void main() {
  vec3 col = vec3(0.012, 0.0125, 0.014) * (1.0 - 0.75 * smoothstep(0.1, 1.0, vH));
  gl_FragColor = vec4(col, 1.0);
}
`

interface ElementView {
  mesh: THREE.Mesh
  edges: THREE.LineSegments
  geometry: THREE.ExtrudeGeometry
  edgeGeometry: THREE.EdgesGeometry
}

interface LampView {
  group: THREE.Group
  housing: THREE.Mesh
  apertures: [THREE.Mesh, THREE.Mesh]
}

export default function Prism() {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!containerRef.current) return
    const container = containerRef.current

    const renderer = new THREE.WebGLRenderer({ antialias: false })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(window.innerWidth, window.innerHeight)
    // Neutral tone mapping keeps the spectral colors saturated where ACES would skew their hues;
    // OutputPass applies it at the end of the HDR chain
    renderer.toneMapping = THREE.NeutralToneMapping
    renderer.toneMappingExposure = 1
    // The glass refracts a half-resolution copy of the table behind it: plenty for a blur of holes
    renderer.transmissionResolutionScale = 0.5
    container.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x000000)
    const camera = new THREE.PerspectiveCamera(42, window.innerWidth / window.innerHeight, 0.1, 60)

    // MSAA HalfFloat target: thin glass edges stay clean and the beams keep their HDR for bloom
    const composerTarget = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 4 })
    const composer = new EffectComposer(renderer, composerTarget)
    composer.setPixelRatio(1)
    const renderPass = new RenderPass(scene, camera)
    const bloomPass = new UnrealBloomPass(new THREE.Vector2(256, 256), BLOOM_STRENGTH, BLOOM_RADIUS, BLOOM_THRESHOLD)
    const outputPass = new OutputPass()
    composer.addPass(renderPass)
    composer.addPass(bloomPass)
    composer.addPass(outputPass)

    // Studio-softbox reflections for the glass and lamp housings (the room itself stays dark)
    const pmrem = new THREE.PMREMGenerator(renderer)
    const room = new RoomEnvironment()
    const envMap = pmrem.fromScene(room, 0.04).texture
    room.dispose()
    pmrem.dispose()

    // ---------- Bench ----------
    const tableGeometry = new THREE.CircleGeometry(WALL_RADIUS, 160)
    tableGeometry.rotateX(-Math.PI / 2)
    const tableMaterial = new THREE.ShaderMaterial({
      vertexShader: tableVertexShader,
      fragmentShader: tableFragmentShader,
      uniforms: { uRadius: { value: WALL_RADIUS }, uPitch: { value: HOLE_PITCH } },
    })
    scene.add(new THREE.Mesh(tableGeometry, tableMaterial))

    const wallGeometry = new THREE.CylinderGeometry(WALL_RADIUS, WALL_RADIUS, WALL_HEIGHT, 192, 1, true)
    const wallMaterial = new THREE.ShaderMaterial({
      vertexShader: wallVertexShader,
      fragmentShader: wallFragmentShader,
      uniforms: { uHeight: { value: WALL_HEIGHT } },
      side: THREE.BackSide, // the camera is inside the ring: draw only its inner face
    })
    const wall = new THREE.Mesh(wallGeometry, wallMaterial)
    wall.position.y = WALL_HEIGHT / 2
    scene.add(wall)

    // ---------- Light ----------
    const buffers = createTraceBuffers(MAX_SEGMENTS, MAX_SPOTS)

    const segmentGeometry = new THREE.InstancedBufferGeometry()
    // x: along the segment, y: across (floor) or up (sheet), z: 0 floor quad / 1 sheet quad
    segmentGeometry.setAttribute(
      'position',
      new THREE.Float32BufferAttribute([0, -1, 0, 1, -1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1], 3),
    )
    segmentGeometry.setIndex([0, 1, 2, 2, 1, 3, 4, 5, 6, 6, 5, 7])
    const segAttr = new THREE.InstancedBufferAttribute(buffers.segPos, 4).setUsage(THREE.DynamicDrawUsage)
    const segColorAttr = new THREE.InstancedBufferAttribute(buffers.segCol, 3).setUsage(THREE.DynamicDrawUsage)
    segmentGeometry.setAttribute('aSeg', segAttr)
    segmentGeometry.setAttribute('aColor', segColorAttr)
    segmentGeometry.instanceCount = 0
    const segmentMaterial = new THREE.ShaderMaterial({
      vertexShader: segmentVertexShader,
      fragmentShader: segmentFragmentShader,
      uniforms: {
        uHalfWidth: { value: FLOOR_HALF_WIDTH },
        uSheetHeight: { value: SHEET_HEIGHT },
        uLevel: { value: 0 },
        uFloorGain: { value: FLOOR_GAIN },
        uSheetGain: { value: SHEET_GAIN },
      },
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      side: THREE.DoubleSide,
    })
    const segments = new THREE.Mesh(segmentGeometry, segmentMaterial)
    segments.frustumCulled = false
    scene.add(segments)

    const spotGeometry = new THREE.InstancedBufferGeometry()
    spotGeometry.setAttribute('position', new THREE.Float32BufferAttribute([-1, 0, 0, 1, 0, 0, -1, 1, 0, 1, 1, 0], 3))
    spotGeometry.setIndex([0, 1, 2, 2, 1, 3])
    const spotAttr = new THREE.InstancedBufferAttribute(buffers.spotPos, 4).setUsage(THREE.DynamicDrawUsage)
    const spotColorAttr = new THREE.InstancedBufferAttribute(buffers.spotCol, 4).setUsage(THREE.DynamicDrawUsage)
    spotGeometry.setAttribute('aSpot', spotAttr)
    spotGeometry.setAttribute('aSpotColor', spotColorAttr)
    spotGeometry.instanceCount = 0
    const spotMaterial = new THREE.ShaderMaterial({
      vertexShader: spotVertexShader,
      fragmentShader: spotFragmentShader,
      uniforms: {
        uSpotHeight: { value: SPOT_HEIGHT },
        uLevel: { value: 0 },
        uSpotGain: { value: SPOT_GAIN },
      },
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      side: THREE.DoubleSide,
    })
    const spots = new THREE.Mesh(spotGeometry, spotMaterial)
    spots.frustumCulled = false
    scene.add(spots)

    // ---------- Glass ----------
    const glassMaterials = new Map<Glass, THREE.MeshPhysicalMaterial>()
    for (const look of GLASS_LOOK) {
      glassMaterials.set(
        look.glass,
        new THREE.MeshPhysicalMaterial({
          color: 0xffffff,
          metalness: 0,
          roughness: 0.03,
          transmission: 1,
          thickness: 0.8,
          ior: Math.min(2.333, refractiveIndex(look.glass, SODIUM_D_NM)),
          dispersion: look.dispersion,
          attenuationColor: new THREE.Color(look.attenuation),
          attenuationDistance: look.distance,
          envMap,
          envMapIntensity: 0.55,
          // The beams are traced and drawn separately, through the glass: it must not hide them
          depthWrite: false,
        }),
      )
    }
    // Polished edges catch the light: faint additive outlines so the shapes read in the dark
    const edgeMaterial = new THREE.LineBasicMaterial({
      color: new THREE.Color(0.32, 0.38, 0.46),
      transparent: true,
      opacity: 0.55,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    })

    let elementViews: ElementView[] = []
    const disposeElements = () => {
      for (const view of elementViews) {
        scene.remove(view.mesh)
        scene.remove(view.edges)
        view.geometry.dispose()
        view.edgeGeometry.dispose()
      }
      elementViews = []
    }

    // ---------- Lamps (slit ray boxes) ----------
    const housingGeometry = new THREE.BoxGeometry(1, 1, 1)
    const housingMaterial = new THREE.MeshStandardMaterial({
      color: 0x1a1c21,
      metalness: 0.75,
      roughness: 0.38,
      envMap,
      envMapIntensity: 0.35,
    })
    const apertureGeometry = new THREE.PlaneGeometry(1, 1)
    const apertureMaterial = new THREE.MeshBasicMaterial({ color: 0x000000, side: THREE.DoubleSide })
    const lamps: LampView[] = []
    for (let i = 0; i < 2; i++) {
      const group = new THREE.Group()
      const housing = new THREE.Mesh(housingGeometry, housingMaterial)
      group.add(housing)
      const apertures: [THREE.Mesh, THREE.Mesh] = [
        new THREE.Mesh(apertureGeometry, apertureMaterial),
        new THREE.Mesh(apertureGeometry, apertureMaterial),
      ]
      for (const a of apertures) {
        a.rotation.y = Math.PI / 2 // face along the group's +x, the beam direction
        group.add(a)
      }
      group.visible = false
      scene.add(group)
      lamps.push({ group, housing, apertures })
    }

    // ---------- Experiments ----------
    const spectra = new Map<number, Spectrum>()
    const spectrumFor = (count: number) => {
      let s = spectra.get(count)
      if (!s) {
        s = makeSpectrum(count)
        spectra.set(count, s)
      }
      return s
    }

    let bag: ExperimentKind[] = []
    let lastKind: ExperimentKind | null = null
    const nextKind = (): ExperimentKind => {
      if (bag.length === 0) {
        bag = EXPERIMENT_KINDS.slice()
        for (let i = bag.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1))
          ;[bag[i], bag[j]] = [bag[j], bag[i]]
        }
        // Never the same experiment twice in a row across a reshuffle
        if (bag[bag.length - 1] === lastKind) [bag[0], bag[bag.length - 1]] = [bag[bag.length - 1], bag[0]]
      }
      const kind = bag.pop() as ExperimentKind
      lastKind = kind
      return kind
    }

    let experiment: Experiment
    let spectrum: Spectrum
    let traceElements: TraceElement[] = []
    let experimentTime = 0
    let settings: TraceSettings = TRACE_SETTINGS

    const startExperiment = () => {
      disposeElements()
      experiment = createExperiment(nextKind(), Math.random)
      experimentTime = 0
      spectrum = spectrumFor(experiment.spectrumCount)
      settings = { ...TRACE_SETTINGS, maxDepth: experiment.maxDepth }
      traceElements = experiment.elements.map(e => makeTraceElement(e.shape, e.glass, spectrum))
      const height = experiment.kind === 'raindrop' ? DROP_HEIGHT : ELEMENT_HEIGHT
      for (const spec of experiment.elements) {
        const outline = spec.shape.outline
        const shape = new THREE.Shape()
        shape.moveTo(outline[0], outline[1])
        for (let i = 2; i < outline.length; i += 2) shape.lineTo(outline[i], outline[i + 1])
        shape.closePath()
        const geometry = new THREE.ExtrudeGeometry(shape, { depth: height, bevelEnabled: false, curveSegments: 1 })
        // Outline (x, y) -> world (x, z), extruded upward from the table
        geometry.rotateX(Math.PI / 2)
        geometry.translate(0, height, 0)
        const edgeGeometry = new THREE.EdgesGeometry(geometry, 20)
        const material = glassMaterials.get(spec.glass) ?? (glassMaterials.get(BK7) as THREE.MeshPhysicalMaterial)
        const mesh = new THREE.Mesh(geometry, material)
        const edges = new THREE.LineSegments(edgeGeometry, edgeMaterial)
        mesh.scale.setScalar(0)
        edges.scale.setScalar(0)
        scene.add(mesh)
        scene.add(edges)
        elementViews.push({ mesh, edges, geometry, edgeGeometry })
      }
    }
    startExperiment()

    const rays: WhiteRay[] = []
    const trace = createTracer()

    const uploadLight = (level: number) => {
      const segCount = level > 0 ? buffers.segCount : 0
      const spotCount = level > 0 ? buffers.spotCount : 0
      segmentGeometry.instanceCount = segCount
      spotGeometry.instanceCount = spotCount
      segments.visible = segCount > 0
      spots.visible = spotCount > 0
      if (segCount > 0) {
        for (const [attr, size] of [
          [segAttr, 4],
          [segColorAttr, 3],
        ] as const) {
          attr.clearUpdateRanges()
          attr.addUpdateRange(0, segCount * size)
          attr.needsUpdate = true
        }
      }
      if (spotCount > 0) {
        for (const attr of [spotAttr, spotColorAttr]) {
          attr.clearUpdateRanges()
          attr.addUpdateRange(0, spotCount * 4)
          attr.needsUpdate = true
        }
      }
      segmentMaterial.uniforms.uLevel.value = level * BEAM_GAIN
      spotMaterial.uniforms.uLevel.value = level * BEAM_GAIN
    }

    const placeLamps = (grow: number, beam: number) => {
      apertureMaterial.color.setScalar(APERTURE_GLOW * beam + 0.02)
      for (let i = 0; i < lamps.length; i++) {
        const lamp = lamps[i]
        const src = experiment.sources[i]
        lamp.group.visible = !!src && grow > 0.001
        if (!lamp.group.visible) continue
        lamp.group.position.set(src.x, 0, src.y)
        lamp.group.rotation.y = -src.angle
        lamp.group.scale.setScalar(grow)
        const width = Math.max(src.width, 0.05)
        lamp.housing.position.set(-LAMP_LENGTH / 2, LAMP_HEIGHT / 2, 0)
        lamp.housing.scale.set(LAMP_LENGTH, LAMP_HEIGHT, width + 0.22)
        const [a, b] = lamp.apertures
        if (src.gap > 0) {
          const part = (width - src.gap) / 2
          const offset = src.gap / 2 + part / 2
          a.visible = b.visible = true
          a.scale.set(part, SHEET_HEIGHT, 1)
          b.scale.set(part, SHEET_HEIGHT, 1)
          a.position.set(0.003, SHEET_HEIGHT / 2, -offset)
          b.position.set(0.003, SHEET_HEIGHT / 2, offset)
        } else {
          a.visible = true
          b.visible = false
          a.scale.set(width, SHEET_HEIGHT, 1)
          a.position.set(0.003, SHEET_HEIGHT / 2, 0)
        }
      }
    }

    let cameraDistance = 8
    const handleResize = () => {
      const w = window.innerWidth
      const h = window.innerHeight
      renderer.setSize(w, h)
      const scale = Math.min(Math.min(window.devicePixelRatio, 2), Math.sqrt(MAX_RENDER_PIXELS / (w * h)))
      composer.setSize(Math.max(1, Math.round(w * scale)), Math.max(1, Math.round(h * scale)))
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      // Fit the bench across the narrower field of view, but always stay inside the screen ring
      const vFov = THREE.MathUtils.degToRad(camera.fov)
      const hFov = 2 * Math.atan(Math.tan(vFov / 2) * camera.aspect)
      const el = THREE.MathUtils.degToRad(CAMERA_ELEVATION_DEG)
      const fit = Math.max(BENCH_FIT_RADIUS / Math.tan(hFov / 2), (BENCH_FIT_RADIUS * Math.sin(el)) / Math.tan(vFov / 2))
      cameraDistance = Math.min(Math.max(fit, 6), (WALL_RADIUS - 0.8) / Math.cos(el))
    }
    handleResize()

    const target = new THREE.Vector3(0, 0.2, 0.25)
    let azimuth = experiment!.viewAzimuth
    let clock = 0
    let lastTime = performance.now()
    let animationId = 0

    const animate = () => {
      animationId = requestAnimationFrame(animate)
      const now = performance.now()
      const dt = Math.min((now - lastTime) / 1000, MAX_FRAME_SECONDS)
      lastTime = now
      clock = (clock + dt) % LOOP_SECONDS

      experimentTime += dt
      if (experimentTime >= experiment.duration) startExperiment()
      experiment.update(experimentTime)
      const [grow, beam] = envelope(experimentTime, experiment.duration)

      for (let i = 0; i < traceElements.length; i++) {
        const pose = experiment.poses[i]
        const el = traceElements[i]
        el.x = pose.x
        el.y = pose.y
        el.rotation = pose.rotation
        el.scale = grow
        const view = elementViews[i]
        for (const obj of [view.mesh, view.edges]) {
          obj.position.set(pose.x, 0, pose.y)
          // 2D counter-clockwise rotation about +y in the (x, z) table plane is -rotation in three
          obj.rotation.y = -pose.rotation
          obj.scale.setScalar(Math.max(grow, 1e-4))
        }
      }
      placeLamps(grow, beam)

      if (beam > 0.001) {
        let n = 0
        for (const src of experiment.sources) n = emitRays(src, rays, n)
        rays.length = n
        trace(traceElements, rays, spectrum, settings, buffers)
      }
      uploadLight(beam)

      // Camera: drift toward this experiment's best side, with slow sways so it never sits still
      const desired = experiment.viewAzimuth + 0.14 * Math.sin((clock / 47) * Math.PI * 2)
      azimuth += (desired - azimuth) * Math.min(1, CAMERA_AZIMUTH_EASE * dt)
      const elevation = THREE.MathUtils.degToRad(
        CAMERA_ELEVATION_DEG + CAMERA_ELEVATION_SWAY_DEG * Math.sin((clock / 53) * Math.PI * 2),
      )
      camera.position.set(
        target.x + cameraDistance * Math.cos(elevation) * Math.sin(azimuth),
        target.y + cameraDistance * Math.sin(elevation),
        target.z + cameraDistance * Math.cos(elevation) * Math.cos(azimuth),
      )
      camera.lookAt(target)

      composer.render()
    }

    animate()
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      cancelAnimationFrame(animationId)
      disposeElements()
      for (const material of glassMaterials.values()) material.dispose()
      edgeMaterial.dispose()
      housingGeometry.dispose()
      housingMaterial.dispose()
      apertureGeometry.dispose()
      apertureMaterial.dispose()
      segmentGeometry.dispose()
      segmentMaterial.dispose()
      spotGeometry.dispose()
      spotMaterial.dispose()
      tableGeometry.dispose()
      tableMaterial.dispose()
      wallGeometry.dispose()
      wallMaterial.dispose()
      envMap.dispose()
      bloomPass.dispose()
      outputPass.dispose()
      renderPass.dispose()
      composer.dispose()
      renderer.dispose()
      container.removeChild(renderer.domElement)
    }
  }, [])

  return <div ref={containerRef} className="w-full h-full" />
}
