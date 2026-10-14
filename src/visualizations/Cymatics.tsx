import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js'
import {
  createField,
  fillField,
  randomMode,
  scatterGrains,
  stepGrains,
  type Mode,
} from './cymaticsMath'

// Configuration constants
const GRAIN_COUNT = 50000
const FIELD_SIZE = 256 // field samples per side; also the plate-glow texture resolution
const MODE_SECONDS = 18 // time on each vibration mode before the plate is struck with the next
const MODES_PER_PALETTE = 2 // new sand colour every other mode (~36 s)
const PALETTE_FADE_SECONDS = 4
const SHAPE_MORPH_SECONDS = 1.6 // square <-> circle plate outline morph
const MAX_STEP_SECONDS = 1 / 60 // physics substep; the field gradient is steep for busy modes

// Grain physics (plate units, seconds); tuned headlessly so ~95% of grains sit on nodal lines
// within ~5 s of a new mode, leaving ~13 s of crisp figure before the next strike
const DRIFT = 0.04
const JITTER = 0.35
const FLOOR_JITTER = 0.007 // gives the settled lines a little width and life
const STRIKE_KICK = 0.5 // hopping right after a mode change: the sand visibly scatters
const STRIKE_DECAY_SECONDS = 0.5

// Look
const HOP_HEIGHT = 0.03 // world units a grain at an antinode bounces off the plate
const STRIKE_HOP_HEIGHT = 0.12
const GRAIN_PX_PER_1000 = 2.4 // grain sprite size per 1000 px of drawing-buffer height
const GRAIN_INTENSITY = 0.25 // additive; dense lines reach ~1-2 before tone mapping
const FIT_RADIUS = 1.3 // plate half-size is 1; leave room for the rim and the rotating corners
const CAMERA_ELEVATION = 1.08 // radians above the plate (~62 deg): tilted, but the figure reads
const CAMERA_SWAY_RADIANS = 0.12
const CAMERA_SWAY_SECONDS = 41
const CAMERA_ROTATE_SECONDS = 300 // one slow revolution of the view around the plate

interface Palette {
  shadow: number // darker grains
  lit: number // brighter grains
  glint: number // facet sparkle
  plate: number // plate base colour
}

// Linear-space colours; gold is listed twice so warm sand is the most common look
const PALETTES: Palette[] = [
  { shadow: 0x8c4f1c, lit: 0xffc06a, glint: 0xfff0c8, plate: 0x060504 },
  { shadow: 0x8c4f1c, lit: 0xffc06a, glint: 0xfff0c8, plate: 0x060504 },
  { shadow: 0x9a2c0c, lit: 0xff8238, glint: 0xffd0a0, plate: 0x070403 }, // copper embers
  { shadow: 0x56688c, lit: 0xd8e6ff, glint: 0xf0f6ff, plate: 0x040507 }, // moonlit silver
  { shadow: 0x147064, lit: 0x8cffd8, glint: 0xe0fff4, plate: 0x030606 }, // sea glass
  { shadow: 0x8a3038, lit: 0xffa898, glint: 0xffe8e0, plate: 0x060404 }, // rose gold
]

const grainVertexShader = /* glsl */ `
attribute float seed;
attribute float energy;
uniform float uScale;
varying float vSeed;
varying float vEnergy;

void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  // Perspective-correct size with a little per-grain variation
  float size = uScale * (0.7 + 0.6 * fract(seed * 7.31)) / -mv.z;
  gl_PointSize = max(size, 1.0);
  vSeed = seed;
  vEnergy = energy;
}
`

const grainFragmentShader = /* glsl */ `
uniform vec3 uShadow;
uniform vec3 uLit;
uniform vec3 uGlint;
uniform float uTime;
uniform float uIntensity;
varying float vSeed;
varying float vEnergy;

void main() {
  vec2 c = gl_PointCoord * 2.0 - 1.0;
  float d2 = dot(c, c);
  if (d2 > 1.0) discard;
  float disc = exp(-d2 * 3.0);

  vec3 col = mix(uShadow, uLit, fract(vSeed * 13.7));

  // Each grain has a facet that catches the light now and then
  float phase = vSeed * 61.0 + uTime * (0.4 + 1.2 * fract(vSeed * 3.1));
  float glint = pow(max(sin(phase), 0.0), 60.0) * 2.5;
  // Bouncing grains tumble, so they flicker far more often
  glint += vEnergy * pow(max(sin(phase * 9.0 + vSeed * 17.0), 0.0), 12.0) * 1.5;

  col = col * (0.75 + 0.5 * vEnergy) + uGlint * glint;
  gl_FragColor = vec4(col * disc * uIntensity, 1.0);
}
`

const plateVertexShader = /* glsl */ `
varying vec3 vWorld;

void main() {
  vec4 w = modelMatrix * vec4(position, 1.0);
  vWorld = w.xyz;
  gl_Position = projectionMatrix * viewMatrix * w;
}
`

const plateFragmentShader = /* glsl */ `
uniform float uRoundness;
uniform sampler2D uField;
uniform float uFieldGain;
uniform vec3 uPlate;
uniform vec3 uSand;
uniform vec3 uLightDir;
varying vec3 vWorld;

// Rounded box, half-size 1, corner radius r: r = 0 is the square plate, r = 1 the disk
float plateSdf(vec2 p, float r) {
  vec2 q = abs(p) - vec2(1.0 - r);
  return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

float hash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float valueNoise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  float a = hash(i);
  float b = hash(i + vec2(1.0, 0.0));
  float c = hash(i + vec2(0.0, 1.0));
  float d = hash(i + vec2(1.0, 1.0));
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

void main() {
  vec2 p = vWorld.xz;
  float r = uRoundness;
  float d = plateSdf(p, r);
  const float rim = 0.035; // bevelled lip beyond the sand's reach
  float aa = fwidth(d) * 1.5;
  float onPlate = 1.0 - smoothstep(rim - aa, rim, d);

  // Bevel: the normal tips outward across the lip
  vec2 e = vec2(0.002, 0.0);
  vec2 grad = vec2(plateSdf(p + e.xy, r) - plateSdf(p - e.xy, r),
                   plateSdf(p + e.yx, r) - plateSdf(p - e.yx, r)) / (2.0 * e.x);
  float bevel = smoothstep(0.0, rim, d);
  vec3 n = normalize(vec3(grad.x * bevel * 1.2, 1.0, grad.y * bevel * 1.2));

  vec3 v = normalize(cameraPosition - vWorld);
  vec3 h = normalize(uLightDir + v);
  float nh = max(dot(n, h), 0.0);
  // Soft smudges vary the polish so the sheen is not perfectly uniform
  float smudge = valueNoise(p * 5.0) * 0.6 + valueNoise(p * 13.0) * 0.4;
  float spec = pow(nh, 18.0) * (0.05 + 0.05 * smudge) + pow(nh, 160.0) * 0.35;
  float fresnel = pow(1.0 - max(dot(n, v), 0.0), 4.0);

  vec3 col = uPlate * (0.8 + 0.4 * smudge);
  col *= 0.75 + 0.5 * exp(-dot(p, p) * 0.9); // light pools toward the centre
  col += vec3(0.9, 0.85, 0.8) * spec;
  col += uSand * fresnel * 0.04;
  // A fine bright line where the lip turns over
  col += uSand * 0.10 * (1.0 - smoothstep(0.0, 0.008, abs(d - rim * 0.45)));

  // Faint warm glow where the plate is ringing hardest
  float field = texture2D(uField, p * 0.5 + 0.5).r;
  col += uSand * field * field * uFieldGain * (1.0 - smoothstep(-0.01, 0.0, d));

  // A dim halo of light spilling onto the darkness around the plate
  float halo = exp(-max(d - rim, 0.0) * 7.0) * 0.012;
  gl_FragColor = vec4(col * onPlate + uSand * halo * (1.0 - onPlate), 1.0);
}
`

export default function Cymatics() {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!containerRef.current) return
    const container = containerRef.current

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x000000)

    const camera = new THREE.PerspectiveCamera(35, window.innerWidth / window.innerHeight, 0.05, 50)

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(window.innerWidth, window.innerHeight)
    // Dense nodal lines stack many additive grains; ACES rolls them off to pale gold, not clipped white
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1
    container.appendChild(renderer.domElement)

    const composer = new EffectComposer(renderer)
    composer.addPass(new RenderPass(scene, camera))
    const bloomPass = new UnrealBloomPass(
      new THREE.Vector2(window.innerWidth, window.innerHeight),
      0.55, // strength
      0.45, // radius
      0.75, // threshold: only packed lines and glints bloom, not lone grains
    )
    composer.addPass(bloomPass)
    const outputPass = new OutputPass()
    composer.addPass(outputPass)

    // --- Vibration field (CPU copy drives the grains, a texture copy lights the plate) ---
    const field = createField(FIELD_SIZE)
    const fieldPixels = new Uint8Array(FIELD_SIZE * FIELD_SIZE * 4)
    const fieldTexture = new THREE.DataTexture(fieldPixels, FIELD_SIZE, FIELD_SIZE, THREE.RGBAFormat)
    fieldTexture.magFilter = THREE.LinearFilter
    fieldTexture.minFilter = THREE.LinearFilter
    fieldTexture.wrapS = THREE.ClampToEdgeWrapping
    fieldTexture.wrapT = THREE.ClampToEdgeWrapping

    // --- Palette (uniform colours fade between palettes) ---
    const shadowColor = new THREE.Color()
    const litColor = new THREE.Color()
    const glintColor = new THREE.Color()
    const plateColor = new THREE.Color()
    let paletteIndex = Math.floor(Math.random() * PALETTES.length)
    let paletteFrom = PALETTES[paletteIndex]
    let paletteTo = PALETTES[paletteIndex]
    let paletteStart = performance.now()
    const scratchA = new THREE.Color()
    const scratchB = new THREE.Color()
    const blend = (from: number, to: number, t: number, out: THREE.Color) =>
      out.copy(scratchA.setHex(from, THREE.LinearSRGBColorSpace)).lerp(scratchB.setHex(to, THREE.LinearSRGBColorSpace), t)

    // --- Plate ---
    const plateGeometry = new THREE.PlaneGeometry(3.6, 3.6)
    plateGeometry.rotateX(-Math.PI / 2) // lie in the xz plane: plate (x, y) -> world (x, z)
    const plateUniforms = {
      uRoundness: { value: 0 },
      uField: { value: fieldTexture },
      uFieldGain: { value: 0 },
      uPlate: { value: plateColor },
      uSand: { value: litColor },
      uLightDir: { value: new THREE.Vector3(-0.45, 0.8, -0.4).normalize() },
    }
    const plateMaterial = new THREE.ShaderMaterial({
      vertexShader: plateVertexShader,
      fragmentShader: plateFragmentShader,
      uniforms: plateUniforms,
    })
    const plate = new THREE.Mesh(plateGeometry, plateMaterial)
    plate.renderOrder = 0
    scene.add(plate)

    // --- Sand grains ---
    const grains = new Float32Array(GRAIN_COUNT * 2) // plate coordinates, owned by the physics
    const amplitude = new Float32Array(GRAIN_COUNT)
    const positions = new Float32Array(GRAIN_COUNT * 3)
    const energies = new Float32Array(GRAIN_COUNT)
    const seeds = new Float32Array(GRAIN_COUNT)
    for (let p = 0; p < GRAIN_COUNT; p++) seeds[p] = Math.random()
    const grainGeometry = new THREE.BufferGeometry()
    const positionAttr = new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage)
    const energyAttr = new THREE.BufferAttribute(energies, 1).setUsage(THREE.DynamicDrawUsage)
    grainGeometry.setAttribute('position', positionAttr)
    grainGeometry.setAttribute('energy', energyAttr)
    grainGeometry.setAttribute('seed', new THREE.BufferAttribute(seeds, 1))
    const grainUniforms = {
      uScale: { value: 1 },
      uShadow: { value: shadowColor },
      uLit: { value: litColor },
      uGlint: { value: glintColor },
      uTime: { value: 0 },
      uIntensity: { value: GRAIN_INTENSITY },
    }
    const grainMaterial = new THREE.ShaderMaterial({
      vertexShader: grainVertexShader,
      fragmentShader: grainFragmentShader,
      uniforms: grainUniforms,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      depthTest: false, // grains always sit on top of the plate
    })
    const sand = new THREE.Points(grainGeometry, grainMaterial)
    sand.frustumCulled = false
    sand.renderOrder = 1
    scene.add(sand)

    // --- Modes ---
    let mode: Mode = randomMode(null)
    let roundness = mode.shape === 'circle' ? 1 : 0
    const writeFieldTexture = () => {
      for (let k = 0; k < field.u.length; k++) {
        const v = Math.min(255, Math.abs(field.u[k]) * 255)
        const o = k * 4
        fieldPixels[o] = v
        fieldPixels[o + 1] = v
        fieldPixels[o + 2] = v
        fieldPixels[o + 3] = 255
      }
      fieldTexture.needsUpdate = true
    }
    fillField(field, mode)
    writeFieldTexture()
    // Opening: sand poured evenly over the plate, then it finds the first figure
    scatterGrains(grains, GRAIN_COUNT, roundness)

    let modeCount = 0
    let modeStart = performance.now()
    let strikeTime = -Infinity // no strike at startup; the even scatter already needs to settle

    const nextMode = (now: number) => {
      mode = randomMode(mode)
      fillField(field, mode) // reuses the same arrays: no allocation per cycle
      writeFieldTexture()
      modeStart = now
      strikeTime = now
      modeCount++
      if (modeCount % MODES_PER_PALETTE === 0) {
        let next = Math.floor(Math.random() * (PALETTES.length - 1))
        if (next >= paletteIndex) next++
        // Fades take seconds and palettes change every ~36 s, so the last fade is complete
        paletteFrom = paletteTo
        paletteIndex = next
        paletteTo = PALETTES[next]
        paletteStart = now
      }
    }

    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight
      camera.updateProjectionMatrix()
      renderer.setSize(window.innerWidth, window.innerHeight)
      composer.setSize(window.innerWidth, window.innerHeight)
    }
    handleResize()

    const drawSize = new THREE.Vector2()
    const startTime = performance.now()
    let lastTime = startTime
    let animationId = 0

    const animate = () => {
      animationId = requestAnimationFrame(animate)
      const now = performance.now()
      // Clamp so a backgrounded tab doesn't take one giant step on return
      const frameSeconds = Math.min((now - lastTime) / 1000, 1 / 20)
      lastTime = now
      const t = (now - startTime) / 1000

      if ((now - modeStart) / 1000 >= MODE_SECONDS) nextMode(now)

      // Plate outline eases toward the current mode's shape; grains are swept along with it
      const targetRoundness = mode.shape === 'circle' ? 1 : 0
      const morphStep = frameSeconds / SHAPE_MORPH_SECONDS
      roundness += Math.max(-morphStep, Math.min(morphStep, targetRoundness - roundness))
      const morph = roundness * roundness * (3 - 2 * roundness) // smoothstep: eased outline

      const sinceStrike = (now - strikeTime) / 1000
      const kick = STRIKE_KICK * Math.exp(-sinceStrike / STRIKE_DECAY_SECONDS)

      const substeps = Math.max(1, Math.ceil(frameSeconds / MAX_STEP_SECONDS))
      const dt = frameSeconds / substeps
      for (let s = 0; s < substeps; s++) {
        stepGrains(grains, amplitude, GRAIN_COUNT, field, {
          dt,
          drift: DRIFT,
          jitter: JITTER,
          floor: FLOOR_JITTER,
          kick,
          roundness: morph,
        })
      }

      // Lift each grain by a fresh random hop each frame: antinodes buzz, nodal lines lie still
      for (let p = 0; p < GRAIN_COUNT; p++) {
        const a = amplitude[p]
        const o = p * 3
        positions[o] = grains[p * 2]
        positions[o + 1] = 0.003 + (HOP_HEIGHT * a + STRIKE_HOP_HEIGHT * kick) * Math.random()
        positions[o + 2] = grains[p * 2 + 1]
        energies[p] = Math.min(1, a * 1.5 + kick * 2)
      }
      positionAttr.needsUpdate = true
      energyAttr.needsUpdate = true

      // Palette fade
      const fade = Math.min((now - paletteStart) / 1000 / PALETTE_FADE_SECONDS, 1)
      const eased = fade * fade * (3 - 2 * fade)
      blend(paletteFrom.shadow, paletteTo.shadow, eased, shadowColor)
      blend(paletteFrom.lit, paletteTo.lit, eased, litColor)
      blend(paletteFrom.glint, paletteTo.glint, eased, glintColor)
      blend(paletteFrom.plate, paletteTo.plate, eased, plateColor)
      if (fade >= 1) paletteFrom = paletteTo

      plateUniforms.uRoundness.value = morph
      // The plate glows where it rings: a flare at each strike, settling to a faint shimmer
      plateUniforms.uFieldGain.value = 0.025 + 0.1 * Math.exp(-sinceStrike / 1.5)
      // Only drives per-grain glint phases (random, so the wrap is invisible); unbounded time
      // would quantize sin(phase * 9) in float32 after a day or so
      grainUniforms.uTime.value = t % 4096

      // Camera: slow revolution around a tilted plate, with a gentle sway in elevation
      const vFov = THREE.MathUtils.degToRad(camera.fov)
      const hFov = 2 * Math.atan(Math.tan(vFov / 2) * camera.aspect)
      const distance = FIT_RADIUS / Math.sin(Math.min(vFov, hFov) / 2)
      const azimuth = (t / CAMERA_ROTATE_SECONDS) * Math.PI * 2
      const elevation = CAMERA_ELEVATION + Math.sin((t / CAMERA_SWAY_SECONDS) * Math.PI * 2) * CAMERA_SWAY_RADIANS
      camera.position.set(
        distance * Math.cos(elevation) * Math.cos(azimuth),
        distance * Math.sin(elevation),
        distance * Math.cos(elevation) * Math.sin(azimuth),
      )
      camera.lookAt(0, 0, 0)

      // Grain sprites scale with the drawing buffer so density looks the same on any screen
      renderer.getDrawingBufferSize(drawSize)
      grainUniforms.uScale.value = (GRAIN_PX_PER_1000 * drawSize.y * distance) / 1000

      composer.render()
    }

    animate()
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      cancelAnimationFrame(animationId)
      grainGeometry.dispose()
      grainMaterial.dispose()
      plateGeometry.dispose()
      plateMaterial.dispose()
      fieldTexture.dispose()
      bloomPass.dispose()
      outputPass.dispose()
      composer.dispose()
      renderer.dispose()
      container.removeChild(renderer.domElement)
    }
  }, [])

  return <div ref={containerRef} className="w-full h-full" />
}
