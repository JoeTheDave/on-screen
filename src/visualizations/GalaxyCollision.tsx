import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'
import {
  CORE_STRIDE,
  createEncounter,
  prime,
  seedParticle,
  step,
  type Encounter,
} from './galaxyMath'

// Configuration constants
const STAR_COUNT = 40000 // massless stars shared between the two galaxies by mass
const BULGE_FRACTION = 0.18
const DUST_COUNT = 8000
const SKY_STAR_COUNT = 1200 // static far-away field stars, for depth while the camera orbits
const SIM_DT = 0.01 // fixed leapfrog step (sim time units); energy-checked in node
const MAX_STEPS_PER_FRAME = 6 // after a long stall, drop sim time rather than stutter
const SETTLE_TIME = 16 // sim time after the cores merge for the tails to rain back in
const ENCOUNTER_SECONDS = 72 // real seconds an encounter is paced to (~60-90 s with clamping)
const MAX_ENCOUNTER_SECONDS = 95
const FADE_SECONDS = 3
const STAR_SIZE = 0.016 // world units; the shader converts to pixels with perspective
const CAMERA_ORBIT_SECONDS = 150
const CAMERA_BOB_SECONDS = 41
const FIT_SECONDS = 6 // time constant for the camera's zoom following the system's extent
const FIT_SAMPLES = 600

// The core sprites live at the start of the star buffer, so they're drawn with the same shader
const CORE_SPRITES = 2
const STAR_BUFFER = CORE_SPRITES + STAR_COUNT
const SIM_START = CORE_SPRITES
const SIM_END = STAR_BUFFER + DUST_COUNT

// Shared by stars and dust. Each point knows its galaxy (aGal) and how strongly it follows the
// spiral density wave (aArm). The arms are a rigidly rotating logarithmic-spiral brightness
// pattern in that galaxy's disk frame: stars light up as they pass through it, the way young
// stars flare in real arms, so the arms never wind up. The pattern fades as the encounter
// wrecks the disks, leaving the true tidal arms and tails the simulation makes.
const vertexShader = /* glsl */ `
attribute vec3 aColor;
attribute float aSize;
attribute float aGal;
attribute float aArm;
uniform vec3 uCore[2];
uniform vec3 uE1[2];
uniform vec3 uE2[2];
uniform float uRadius[2];
uniform float uWind[2];
uniform float uArms[2];
uniform float uPattern[2];
uniform float uArmW[2];
uniform float uArmOffset;
uniform float uScale;
uniform float uFade;
varying vec3 vColor;

void main() {
  int g = int(aGal + 0.5);
  vec3 rel = position - uCore[g];
  float a = dot(rel, uE1[g]);
  float b = dot(rel, uE2[g]);
  float r = max(length(vec2(a, b)), 1e-4) / uRadius[g];
  // Trailing spiral: arm angle falls with log radius
  float phase = atan(b, a) + log(r) * uWind[g] - uPattern[g] + uArmOffset;
  float wave = pow(0.5 + 0.5 * cos(uArms[g] * phase), 3.0);
  float radial = smoothstep(0.08, 0.3, r) * (1.0 - smoothstep(1.0, 1.6, r));
  float w = aArm * uArmW[g] * radial;
  float bright = mix(1.0, mix(0.2, 2.4, wave), w);

  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  float px = aSize * uScale / max(-mv.z, 0.01);
  // Below ~1.5 px, keep the sprite at 1.5 px and dim it instead: no sub-pixel shimmer
  float coverage = min(px / 1.5, 1.0);
  vColor = aColor * bright * coverage * coverage * uFade;
  gl_PointSize = clamp(px, 1.5, 96.0);
  gl_Position = projectionMatrix * mv;
}
`

const fragmentShader = /* glsl */ `
varying vec3 vColor;

void main() {
  vec2 c = gl_PointCoord * 2.0 - 1.0;
  float d2 = dot(c, c);
  if (d2 > 1.0) discard;
  // Gaussian, shifted so it reaches exactly zero at the sprite edge
  float glow = (exp(-5.0 * d2) - 0.0067) / 0.9933;
  gl_FragColor = vec4(vColor * glow, 1.0);
}
`

// Rendered as a filter: dst * (1 - src). Dust "color" is how much of each channel it absorbs,
// more blue than red, so lanes read as warm brown shadows across the bright disks.
const dustBlending = {
  blending: THREE.CustomBlending,
  blendEquation: THREE.AddEquation,
  blendSrc: THREE.ZeroFactor,
  blendDst: THREE.OneMinusSrcColorFactor,
} as const

const rand = Math.random

export default function GalaxyCollision() {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!containerRef.current) return
    const container = containerRef.current

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x000000)

    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.05, 400)

    const renderer = new THREE.WebGLRenderer({ antialias: false })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(window.innerWidth, window.innerHeight)
    container.appendChild(renderer.domElement)

    const composer = new EffectComposer(renderer)
    composer.addPass(new RenderPass(scene, camera))
    const bloomPass = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 0.85, 0.6, 0.12)
    composer.addPass(bloomPass)

    // ---- Simulation state: one position buffer, viewed by the star and dust geometries ----
    const pos = new Float32Array(SIM_END * 3)
    const vel = new Float32Array(SIM_END * 3)
    const acc = new Float32Array(SIM_END * 3)

    const uniforms = {
      uCore: { value: [new THREE.Vector3(), new THREE.Vector3()] },
      uE1: { value: [new THREE.Vector3(), new THREE.Vector3()] },
      uE2: { value: [new THREE.Vector3(), new THREE.Vector3()] },
      uRadius: { value: [1, 1] },
      uWind: { value: [3, 3] },
      uArms: { value: [2, 2] },
      uPattern: { value: [0, 0] },
      uArmW: { value: [1, 1] },
      uScale: { value: 1 },
      uFade: { value: 0 },
    }
    // Spreading shares the same {value} objects, so one update drives both materials
    const starMaterial = new THREE.ShaderMaterial({
      uniforms: { ...uniforms, uArmOffset: { value: 0 } },
      vertexShader,
      fragmentShader,
      blending: THREE.AdditiveBlending,
      transparent: true,
      depthTest: false,
      depthWrite: false,
    })
    const dustMaterial = new THREE.ShaderMaterial({
      // Lanes hug the inner edge of each arm
      uniforms: { ...uniforms, uArmOffset: { value: 0.3 } },
      vertexShader,
      fragmentShader,
      ...dustBlending,
      transparent: true,
      depthTest: false,
      depthWrite: false,
    })

    const makeGeometry = (positions: Float32Array, count: number) => {
      const geometry = new THREE.BufferGeometry()
      const position = new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage)
      const color = new THREE.BufferAttribute(new Float32Array(count * 3), 3)
      const size = new THREE.BufferAttribute(new Float32Array(count), 1)
      const gal = new THREE.BufferAttribute(new Float32Array(count), 1)
      const arm = new THREE.BufferAttribute(new Float32Array(count), 1)
      geometry.setAttribute('position', position)
      geometry.setAttribute('aColor', color)
      geometry.setAttribute('aSize', size)
      geometry.setAttribute('aGal', gal)
      geometry.setAttribute('aArm', arm)
      return { geometry, position, color, size, gal, arm }
    }

    const stars = makeGeometry(pos.subarray(0, STAR_BUFFER * 3), STAR_BUFFER)
    const dust = makeGeometry(pos.subarray(STAR_BUFFER * 3), DUST_COUNT)
    const sky = makeGeometry(new Float32Array(SKY_STAR_COUNT * 3), SKY_STAR_COUNT)

    const starPoints = new THREE.Points(stars.geometry, starMaterial)
    const dustPoints = new THREE.Points(dust.geometry, dustMaterial)
    const skyPoints = new THREE.Points(sky.geometry, starMaterial)
    for (const p of [starPoints, dustPoints, skyPoints]) p.frustumCulled = false
    skyPoints.renderOrder = 0
    starPoints.renderOrder = 1
    dustPoints.renderOrder = 2 // filters whatever light is already there
    scene.add(skyPoints, starPoints, dustPoints)

    // Far field stars on a big shell: faint, mostly white with a few tinted ones
    {
      const color = new THREE.Color()
      const p = sky.position.array as Float32Array
      for (let i = 0; i < SKY_STAR_COUNT; i++) {
        const z = rand() * 2 - 1
        const t = rand() * Math.PI * 2
        const s = Math.sqrt(1 - z * z)
        const r = 120 + rand() * 60
        p.set([r * s * Math.cos(t), r * z, r * s * Math.sin(t)], i * 3)
        color.setHSL(rand() < 0.5 ? 0.6 : 0.1, 0.4 * rand(), 0.5)
        const brightness = 0.08 + 0.35 * Math.pow(rand(), 4)
        sky.color.setXYZ(i, color.r * brightness, color.g * brightness, color.b * brightness)
        sky.size.setX(i, 0.12 + 0.25 * Math.pow(rand(), 3))
      }
    }

    const color = new THREE.Color()

    // ---- Encounter setup: new masses, tilts and orbit, then stars on circular orbits ----
    let enc: Encounter = createEncounter(rand)
    let encounterStart = 0 // real seconds
    let timeScale = 1
    let simTime = 0
    let endTime = 0
    const patternSpeed = [0, 0]
    const patternPhase = [0, 0]

    // Seeds everything for the current `enc`
    const startEncounter = () => {
      simTime = 0
      endTime = enc.mergeTime + SETTLE_TIME
      timeScale = THREE.MathUtils.clamp(endTime / ENCOUNTER_SECONDS, 0.5, 1.2)

      const [g0, g1] = enc.galaxies
      const share = Math.pow(g0.mass, 0.8) / (Math.pow(g0.mass, 0.8) + Math.pow(g1.mass, 0.8))

      // Per-galaxy look: arm count, pitch, palette nudges
      const youngHue = [0, 0]
      const oldHue = [0, 0]
      enc.galaxies.forEach((g, k) => {
        uniforms.uE1.value[k].fromArray(g.e1)
        uniforms.uE2.value[k].fromArray(g.e2)
        uniforms.uRadius.value[k] = g.radius
        uniforms.uArms.value[k] = rand() < 0.8 ? 2 : 3
        const pitch = THREE.MathUtils.degToRad(13 + rand() * 13)
        uniforms.uWind.value[k] = 1 / Math.tan(pitch)
        // Pattern turns at the disk's angular speed around 0.7 R (corotation)
        const r = 0.7 * g.radius
        const s2 = r * r + g.eps * g.eps
        patternSpeed[k] = Math.sqrt(g.mass / (s2 * Math.sqrt(s2)))
        patternPhase[k] = rand() * Math.PI * 2
        youngHue[k] = 0.58 + (rand() - 0.5) * 0.06
        oldHue[k] = 0.09 + (rand() - 0.5) * 0.04
      })

      // Core sprites: a big warm glow on each nucleus
      for (let k = 0; k < CORE_SPRITES; k++) {
        const m = enc.galaxies[k].mass
        stars.color.setXYZ(k, 0.55, 0.38, 0.2)
        stars.size.setX(k, 0.3 * Math.sqrt(m))
        stars.gal.setX(k, k)
        stars.arm.setX(k, 0)
      }

      const bulgeCount = Math.round(STAR_COUNT * BULGE_FRACTION)
      for (let s = 0; s < STAR_COUNT; s++) {
        const i = SIM_START + s
        const gi: 0 | 1 = rand() < share ? 0 : 1
        const isBulge = s < bulgeCount
        const rr = seedParticle(enc, gi, isBulge ? 'bulge' : 'disk', rand, pos, vel, i)
        let brightness: number
        let size = STAR_SIZE * (0.8 + 0.4 * rand())
        let arm: number
        if (isBulge) {
          // Old, warm, dense
          color.setHSL(oldHue[gi] + (rand() - 0.5) * 0.03, 0.75, 0.62)
          brightness = 0.09
          arm = 0
        } else {
          const roll = rand()
          if (roll < 0.04) {
            // Pink HII knots strung along the arms
            color.setHSL(0.94 + rand() * 0.03, 0.8, 0.65)
            brightness = 0.55
            size *= 1.8
            arm = 1
          } else if (roll < 0.52) {
            // Young blue stars: these carry the arms
            color.setHSL(youngHue[gi] + (rand() - 0.5) * 0.05, 0.75, 0.72)
            brightness = 0.42
            arm = 1
            if (rand() < 0.03) size *= 2.6 // the occasional blazing giant
          } else {
            // Old disk population, warmer toward the middle
            color.setHSL(oldHue[gi] + 0.04 * rr, 0.45, 0.72)
            brightness = 0.3
            arm = 0.3
          }
          // Outskirts fade a little, so disks have soft edges
          brightness *= 1 - 0.35 * Math.min(rr, 1)
        }
        stars.color.setXYZ(i, color.r * brightness, color.g * brightness, color.b * brightness)
        stars.size.setX(i, size)
        stars.gal.setX(i, gi)
        stars.arm.setX(i, arm)
      }

      for (let d = 0; d < DUST_COUNT; d++) {
        const i = STAR_BUFFER + d
        const gi: 0 | 1 = rand() < share ? 0 : 1
        seedParticle(enc, gi, 'dust', rand, pos, vel, i)
        const a = 0.08 + rand() * 0.06
        dust.color.setXYZ(d, a * 0.45, a * 0.7, a)
        dust.size.setX(d, 0.07 + rand() * 0.06)
        dust.gal.setX(d, gi)
        dust.arm.setX(d, 1)
      }

      for (const g of [stars, dust]) {
        g.color.needsUpdate = true
        g.size.needsUpdate = true
        g.gal.needsUpdate = true
        g.arm.needsUpdate = true
      }
      prime(enc, pos, acc, SIM_START, SIM_END)
    }

    // ---- Camera framing: follow the bulk of the stars, not the farthest tail ----
    const sampleIndices = Array.from({ length: FIT_SAMPLES }, () => SIM_START + Math.floor(rand() * STAR_COUNT))
    const sampleDistances = new Float64Array(FIT_SAMPLES)
    const measureExtent = () => {
      for (let k = 0; k < FIT_SAMPLES; k++) {
        const j = sampleIndices[k] * 3
        sampleDistances[k] = Math.hypot(pos[j], pos[j + 1], pos[j + 2])
      }
      sampleDistances.sort()
      return THREE.MathUtils.clamp(sampleDistances[Math.floor(FIT_SAMPLES * 0.75)] * 1.15, 2.4, 6)
    }

    startEncounter()
    let fitRadius = measureExtent()
    let fitTarget = fitRadius

    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight
      camera.updateProjectionMatrix()
      renderer.setSize(window.innerWidth, window.innerHeight)
      composer.setSize(window.innerWidth, window.innerHeight)
      // World size -> pixels at distance 1, in drawing-buffer pixels
      const bufferHeight = window.innerHeight * renderer.getPixelRatio()
      uniforms.uScale.value = bufferHeight / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2))
    }
    handleResize()

    // Arm visibility: crisp spirals on approach, fading after the first passage as the disks
    // are torn up, gone once the cores have merged
    const armWeight = (t: number) => {
      const peri = enc.pericenterTime
      if (t < peri - 2) return 1
      if (t < peri + 6) return THREE.MathUtils.lerp(1, 0.35, (t - (peri - 2)) / 8)
      const span = Math.max(enc.mergeTime - (peri + 6), 1)
      return 0.35 * Math.max(0, 1 - (t - (peri + 6)) / span)
    }

    const startTime = performance.now()
    let lastTime = startTime
    let fadeOutStart = -1 // real seconds; -1 while the encounter is still playing
    let fadeInStart = 0
    let lastMeasure = 0
    let simDebt = 0
    let animationId = 0

    const animate = () => {
      animationId = requestAnimationFrame(animate)
      const now = performance.now()
      // Clamp so a backgrounded tab doesn't take one giant catch-up leap on return
      const frameSeconds = Math.min((now - lastTime) / 1000, 1 / 20)
      lastTime = now
      const t = (now - startTime) / 1000

      // Fixed-step physics, paced so the whole encounter lasts about ENCOUNTER_SECONDS
      simDebt += frameSeconds * timeScale
      let steps = Math.floor(simDebt / SIM_DT)
      if (steps > MAX_STEPS_PER_FRAME) {
        steps = MAX_STEPS_PER_FRAME
        simDebt = 0
      } else {
        simDebt -= steps * SIM_DT
      }
      for (let s = 0; s < steps; s++) step(enc, pos, vel, acc, SIM_START, SIM_END, SIM_DT)
      simTime += steps * SIM_DT

      // Merger has settled: fade to black, start a new encounter, fade back in
      // (the real-time cap only matters if createEncounter ever had to settle for a slow merger)
      if (fadeOutStart < 0 && (simTime >= endTime || t - encounterStart > MAX_ENCOUNTER_SECONDS)) {
        fadeOutStart = t
      }
      let fade = Math.min((t - fadeInStart) / FADE_SECONDS, 1)
      if (fadeOutStart >= 0) {
        const out = (t - fadeOutStart) / FADE_SECONDS
        if (out >= 1) {
          enc = createEncounter(rand)
          startEncounter()
          encounterStart = t
          fitRadius = fitTarget = measureExtent()
          fadeOutStart = -1
          fadeInStart = t
          fade = 0
        } else {
          fade = Math.min(fade, 1 - out)
        }
      }
      uniforms.uFade.value = fade * fade * (3 - 2 * fade)

      // Core sprites ride on the cores; arm patterns turn with their disks
      const c = enc.cores
      const w = armWeight(simTime)
      for (let k = 0; k < 2; k++) {
        const o = k * CORE_STRIDE
        pos[k * 3] = c[o]
        pos[k * 3 + 1] = c[o + 1]
        pos[k * 3 + 2] = c[o + 2]
        uniforms.uCore.value[k].set(c[o], c[o + 1], c[o + 2])
        uniforms.uPattern.value[k] = patternPhase[k] + patternSpeed[k] * simTime
        uniforms.uArmW.value[k] = w
      }
      stars.position.needsUpdate = true
      dust.position.needsUpdate = true

      // Zoom to keep the bulk of the system framed as tails grow and the remnant settles
      if (t - lastMeasure > 0.5) {
        lastMeasure = t
        fitTarget = measureExtent()
      }
      fitRadius += (fitTarget - fitRadius) * (1 - Math.exp(-frameSeconds / FIT_SECONDS))

      const vFov = THREE.MathUtils.degToRad(camera.fov)
      const hFov = 2 * Math.atan(Math.tan(vFov / 2) * camera.aspect)
      const distance = fitRadius / Math.sin(Math.min(vFov, hFov) / 2)
      const azimuth = (t / CAMERA_ORBIT_SECONDS) * Math.PI * 2
      const elevation = 0.45 + 0.3 * Math.sin((t / CAMERA_BOB_SECONDS) * Math.PI * 2)
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
      stars.geometry.dispose()
      dust.geometry.dispose()
      sky.geometry.dispose()
      starMaterial.dispose()
      dustMaterial.dispose()
      bloomPass.dispose() // EffectComposer.dispose() does not dispose its passes
      composer.dispose()
      renderer.dispose()
      container.removeChild(renderer.domElement)
    }
  }, [])

  return <div ref={containerRef} className="w-full h-full" />
}
