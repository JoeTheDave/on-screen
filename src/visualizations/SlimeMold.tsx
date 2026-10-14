import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'

// Configuration constants
const AGENT_TEXTURE_SIZE = 512 // agents live in a 512x512 state texture -> 262,144 agents
const AGENT_COUNT = AGENT_TEXTURE_SIZE * AGENT_TEXTURE_SIZE
const TRAIL_MAX_LONG_SIDE = 1400 // trail map resolution cap (pixels on the long side)
const REFERENCE_SHORT_SIDE = 720 // sensor/step distances are authored for a trail this tall
const SIM_HZ = 60 // fixed simulation rate, independent of display refresh
const MAX_STEPS_PER_FRAME = 3
const CYCLE_MIN_SECONDS = 45 // time on one species before its parameters change
const CYCLE_MAX_SECONDS = 60
const TRANSITION_SECONDS = 8 // parameter/palette morph between species
const FADE_OUT_SECONDS = 1.8 // dip to black before re-scattering the colony
const FADE_IN_SECONDS = 3
const EXPOSURE = 0.22 // trail density (in multiples of the mean) -> brightness
const BREATH_SECONDS = 13 // slow sensor-distance oscillation makes the veins pulse and re-route
const BREATH_AMOUNT = 0.12
const GLOW_PULSE_SECONDS = 7.3
const GLOW_PULSE_AMOUNT = 0.06

// Jones-model species. Angles in radians, distances in reference pixels per simulation step.
interface Species {
  sensorAngle: number
  sensorDistance: number
  turnSpeed: number
  stepSize: number
  decay: number // trail multiplier per step
  diffuse: number // 0 = no blur, 1 = full 3x3 mean
  wander: number // random heading jitter per step
}

const SPECIES: Species[] = [
  // Classic Physarum veins (Jones 2010 defaults)
  { sensorAngle: 0.39, sensorDistance: 9, turnSpeed: 0.785, stepSize: 1.0, decay: 0.9, diffuse: 1, wander: 0.05 },
  // Fine lace: wide eyes, long reach, persistent trails
  { sensorAngle: 0.75, sensorDistance: 16, turnSpeed: 0.45, stepSize: 1.3, decay: 0.94, diffuse: 0.8, wander: 0.04 },
  // Labyrinth cells: broad sensing, slow turning
  { sensorAngle: 1.05, sensorDistance: 22, turnSpeed: 0.5, stepSize: 1.5, decay: 0.95, diffuse: 0.9, wander: 0.06 },
  // Long rivers: narrow sensors far ahead, gentle steering
  { sensorAngle: 0.28, sensorDistance: 28, turnSpeed: 0.18, stepSize: 1.6, decay: 0.97, diffuse: 0.6, wander: 0.02 },
  // Pulsing spots: twitchy agents that knot into nodes
  { sensorAngle: 1.3, sensorDistance: 7, turnSpeed: 1.1, stepSize: 0.9, decay: 0.86, diffuse: 1, wander: 0.1 },
  // Tight mesh: short sensors, sharp turns, many small loops
  { sensorAngle: 0.6, sensorDistance: 5, turnSpeed: 0.9, stepSize: 0.8, decay: 0.9, diffuse: 1, wander: 0.05 },
]

// Each palette ramps black -> deep -> mid -> hot as trail density rises
interface Palette {
  deep: THREE.Color
  mid: THREE.Color
  hot: THREE.Color
}

const PALETTES: Palette[] = [
  { deep: new THREE.Color(0.11, 0.04, 0.42), mid: new THREE.Color(0.0, 0.82, 1.0), hot: new THREE.Color(0.92, 1.0, 1.0) }, // indigo / electric cyan
  { deep: new THREE.Color(0.32, 0.02, 0.26), mid: new THREE.Color(1.0, 0.42, 0.14), hot: new THREE.Color(1.0, 0.95, 0.8) }, // magenta / amber
  { deep: new THREE.Color(0.0, 0.14, 0.2), mid: new THREE.Color(0.15, 1.0, 0.62), hot: new THREE.Color(0.9, 1.0, 0.92) }, // abyss / bioluminescent green
  { deep: new THREE.Color(0.2, 0.03, 0.4), mid: new THREE.Color(0.95, 0.32, 0.85), hot: new THREE.Color(1.0, 0.9, 1.0) }, // violet / rose
  { deep: new THREE.Color(0.16, 0.05, 0.0), mid: new THREE.Color(1.0, 0.66, 0.18), hot: new THREE.Color(1.0, 1.0, 0.9) }, // ember / gold
]

const FORMATION_COUNT = 5 // random, inward ring, outward disc, clusters, counter-rotating rings

const HASH_GLSL = /* glsl */ `
uint hashU(uint x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}
float rand(uint s) {
  return float(hashU(s)) / 4294967295.0;
}
`

const QUAD_VERTEX = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`

// Sense left/forward/right on the trail map, steer toward the strongest, step forward
const AGENT_FRAGMENT = /* glsl */ `
uniform sampler2D uAgents;
uniform sampler2D uTrail;
uniform vec2 uTrailSize;
uniform float uSensorAngle;
uniform float uSensorDistance;
uniform float uTurnSpeed;
uniform float uStepSize;
uniform float uWander;
uniform float uStep;
varying vec2 vUv;
${HASH_GLSL}
float sense(vec2 pos, float angle) {
  vec2 p = pos + vec2(cos(angle), sin(angle)) * uSensorDistance / uTrailSize;
  return texture2D(uTrail, p).r;
}
void main() {
  vec4 agent = texture2D(uAgents, vUv);
  vec2 pos = agent.xy;
  float heading = agent.z;
  float trait = agent.w;
  uvec2 id = uvec2(gl_FragCoord.xy);
  uint seed = hashU(id.x + id.y * 4096u + hashU(uint(uStep)));

  float f = sense(pos, heading);
  float l = sense(pos, heading + uSensorAngle);
  float r = sense(pos, heading - uSensorAngle);
  float turn = uTurnSpeed * (0.7 + 0.6 * rand(seed + 1u));
  if (f > l && f > r) {
    // already on course
  } else if (f < l && f < r) {
    heading += (rand(seed) < 0.5 ? 1.0 : -1.0) * turn;
  } else if (l > r) {
    heading += turn;
  } else if (r > l) {
    heading -= turn;
  }
  heading += (rand(seed + 2u) - 0.5) * uWander;

  float speed = uStepSize * (0.85 + 0.3 * trait);
  pos = fract(pos + vec2(cos(heading), sin(heading)) * speed / uTrailSize);
  gl_FragColor = vec4(pos, mod(heading, 6.2831853), trait);
}
`

// 3x3 diffusion and exponential decay of the trail map (texture wraps, so the world is a torus)
const DIFFUSE_FRAGMENT = /* glsl */ `
uniform sampler2D uTrail;
uniform vec2 uTexel;
uniform float uDecay;
uniform float uDiffuse;
varying vec2 vUv;
void main() {
  float center = texture2D(uTrail, vUv).r;
  float sum = 0.0;
  for (int y = -1; y <= 1; y++) {
    for (int x = -1; x <= 1; x++) {
      sum += texture2D(uTrail, vUv + vec2(float(x), float(y)) * uTexel).r;
    }
  }
  float v = mix(center, sum / 9.0, uDiffuse) * uDecay;
  gl_FragColor = vec4(v, 0.0, 0.0, 1.0);
}
`

// One 1-pixel point per agent, positioned by fetching the agent's state texel
const DEPOSIT_VERTEX = /* glsl */ `
uniform sampler2D uAgents;
void main() {
  vec4 agent = texture2D(uAgents, position.xy);
  gl_Position = vec4(agent.xy * 2.0 - 1.0, 0.0, 1.0);
  gl_PointSize = 1.0;
}
`

const DEPOSIT_FRAGMENT = /* glsl */ `
uniform float uDeposit;
void main() {
  gl_FragColor = vec4(uDeposit, 0.0, 0.0, 1.0);
}
`

// Writes a fresh colony in one of several starting formations
const SCATTER_FRAGMENT = /* glsl */ `
uniform float uFormation;
uniform float uSeed;
uniform vec2 uAspectScale;
${HASH_GLSL}
void main() {
  const float TAU = 6.2831853;
  const float PI = 3.14159265;
  uvec2 id = uvec2(gl_FragCoord.xy);
  uint base = hashU(uint(uSeed));
  uint s = hashU(id.x + id.y * 4096u + base);
  float r0 = rand(s);
  float r1 = rand(s + 1u);
  float r2 = rand(s + 2u);
  float r3 = rand(s + 3u);
  int formation = int(uFormation + 0.5);

  if (formation == 0) {
    // Uniform random soup
    gl_FragColor = vec4(r0, r1, r2 * TAU, r3);
    return;
  }

  vec2 p;
  float heading;
  if (formation == 1) {
    // Thin ring, everyone marching inward
    float a = r0 * TAU;
    float radius = 0.38 + (r1 - 0.5) * 0.015;
    p = vec2(cos(a), sin(a)) * radius;
    heading = a + PI + (r2 - 0.5) * 0.3;
  } else if (formation == 2) {
    // Filled disc bursting outward
    float a = r0 * TAU;
    float radius = sqrt(r1) * 0.22;
    p = vec2(cos(a), sin(a)) * radius;
    heading = a + (r2 - 0.5) * 0.2;
  } else if (formation == 3) {
    // Seven spore clusters scattered across the screen, each bursting outward
    uint k = uint(floor(r0 * 7.0));
    vec2 c = vec2(rand(base + k * 7u + 11u), rand(base + k * 7u + 12u)) - 0.5;
    c *= vec2(1.6, 0.8);
    float a = r1 * TAU;
    float radius = sqrt(r2) * 0.035;
    p = c + vec2(cos(a), sin(a)) * radius;
    heading = a;
  } else {
    // Two concentric rings orbiting in opposite directions
    float a = r0 * TAU;
    bool inner = r1 < 0.4;
    float radius = (inner ? 0.18 : 0.42) + (r2 - 0.5) * 0.01;
    p = vec2(cos(a), sin(a)) * radius;
    heading = a + (inner ? -0.5 : 0.5) * PI;
  }
  vec2 pos = fract(0.5 + p * uAspectScale);
  gl_FragColor = vec4(pos, mod(heading, TAU), r3);
}
`

// Trail density -> color ramp, with vignette, scene fade and dithering against banding
const DISPLAY_FRAGMENT = /* glsl */ `
uniform sampler2D uTrail;
uniform float uInvMean;
uniform float uExposure;
uniform float uFade;
uniform vec3 uDeep;
uniform vec3 uMid;
uniform vec3 uHot;
varying vec2 vUv;
${HASH_GLSL}
void main() {
  float density = texture2D(uTrail, vUv).r * uInvMean;
  float v = 1.0 - exp(-max(density, 0.0) * uExposure);
  vec3 col = uDeep * smoothstep(0.0, 0.42, v);
  col = mix(col, uMid, smoothstep(0.28, 0.8, v));
  col = mix(col, uHot, smoothstep(0.8, 0.995, v));
  vec2 q = vUv - 0.5;
  col *= (1.0 - dot(q, q) * 0.7) * uFade;
  uvec2 id = uvec2(gl_FragCoord.xy);
  col += (rand(id.x * 7919u + id.y * 104729u) - 0.5) / 255.0;
  gl_FragColor = vec4(max(col, 0.0), 1.0);
}
`

const randomBetween = (a: number, b: number) => a + Math.random() * (b - a)

const jitterSpecies = (s: Species): Species => {
  const j = () => randomBetween(0.85, 1.15)
  return {
    sensorAngle: s.sensorAngle * j(),
    sensorDistance: s.sensorDistance * j(),
    turnSpeed: s.turnSpeed * j(),
    stepSize: s.stepSize * j(),
    decay: Math.min(0.985, s.decay * randomBetween(0.99, 1.01)),
    diffuse: s.diffuse,
    wander: s.wander,
  }
}

const lerpSpecies = (a: Species, b: Species, t: number): Species => ({
  sensorAngle: a.sensorAngle + (b.sensorAngle - a.sensorAngle) * t,
  sensorDistance: a.sensorDistance + (b.sensorDistance - a.sensorDistance) * t,
  turnSpeed: a.turnSpeed + (b.turnSpeed - a.turnSpeed) * t,
  stepSize: a.stepSize + (b.stepSize - a.stepSize) * t,
  decay: a.decay + (b.decay - a.decay) * t,
  diffuse: a.diffuse + (b.diffuse - a.diffuse) * t,
  wander: a.wander + (b.wander - a.wander) * t,
})

const pickOther = (count: number, current: number) => {
  let next = Math.floor(Math.random() * (count - 1))
  if (next >= current) next++
  return next
}

const smooth = (t: number) => {
  const c = Math.min(Math.max(t, 0), 1)
  return c * c * (3 - 2 * c)
}

export default function SlimeMold() {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!containerRef.current) return
    const container = containerRef.current

    const renderer = new THREE.WebGLRenderer({ antialias: false })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(window.innerWidth, window.innerHeight)
    renderer.autoClear = false
    container.appendChild(renderer.domElement)

    // Agent positions need full float precision: half floats would quantize a 1-pixel step
    // on a ~1400-pixel map. Fall back to half floats (coarser, jitterier motion) if needed.
    const agentType = renderer.extensions.has('EXT_color_buffer_float') ? THREE.FloatType : THREE.HalfFloatType

    const makeTarget = (w: number, h: number, type: THREE.TextureDataType, filter: THREE.MagnificationTextureFilter) =>
      new THREE.WebGLRenderTarget(w, h, {
        type,
        format: THREE.RGBAFormat,
        minFilter: filter,
        magFilter: filter,
        wrapS: THREE.RepeatWrapping,
        wrapT: THREE.RepeatWrapping,
        depthBuffer: false,
        generateMipmaps: false,
      })

    const trailSize = new THREE.Vector2()
    const computeTrailSize = () => {
      const w = window.innerWidth
      const h = window.innerHeight
      const scale = Math.min(1, TRAIL_MAX_LONG_SIDE / Math.max(w, h))
      trailSize.set(Math.max(64, Math.round(w * scale)), Math.max(64, Math.round(h * scale)))
    }
    computeTrailSize()

    const agentTargets = [
      makeTarget(AGENT_TEXTURE_SIZE, AGENT_TEXTURE_SIZE, agentType, THREE.NearestFilter),
      makeTarget(AGENT_TEXTURE_SIZE, AGENT_TEXTURE_SIZE, agentType, THREE.NearestFilter),
    ]
    // Half floats are filterable and blendable everywhere WebGL2 runs, which deposition needs
    const trailTargets = [
      makeTarget(trailSize.x, trailSize.y, THREE.HalfFloatType, THREE.LinearFilter),
      makeTarget(trailSize.x, trailSize.y, THREE.HalfFloatType, THREE.LinearFilter),
    ]
    let agentIndex = 0
    let trailIndex = 0

    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1)
    const quadGeometry = new THREE.PlaneGeometry(2, 2)

    const agentMaterial = new THREE.ShaderMaterial({
      vertexShader: QUAD_VERTEX,
      fragmentShader: AGENT_FRAGMENT,
      uniforms: {
        uAgents: { value: null },
        uTrail: { value: null },
        uTrailSize: { value: trailSize },
        uSensorAngle: { value: 0 },
        uSensorDistance: { value: 0 },
        uTurnSpeed: { value: 0 },
        uStepSize: { value: 0 },
        uWander: { value: 0 },
        uStep: { value: 0 },
      },
      depthTest: false,
      depthWrite: false,
    })
    const diffuseMaterial = new THREE.ShaderMaterial({
      vertexShader: QUAD_VERTEX,
      fragmentShader: DIFFUSE_FRAGMENT,
      uniforms: {
        uTrail: { value: null },
        uTexel: { value: new THREE.Vector2() },
        uDecay: { value: 0.9 },
        uDiffuse: { value: 1 },
      },
      depthTest: false,
      depthWrite: false,
    })
    const scatterMaterial = new THREE.ShaderMaterial({
      vertexShader: QUAD_VERTEX,
      fragmentShader: SCATTER_FRAGMENT,
      uniforms: {
        uFormation: { value: 0 },
        uSeed: { value: 0 },
        uAspectScale: { value: new THREE.Vector2(1, 1) },
      },
      depthTest: false,
      depthWrite: false,
    })
    const depositMaterial = new THREE.ShaderMaterial({
      vertexShader: DEPOSIT_VERTEX,
      fragmentShader: DEPOSIT_FRAGMENT,
      uniforms: {
        uAgents: { value: null },
        uDeposit: { value: 0.1 },
      },
      blending: THREE.AdditiveBlending,
      transparent: true,
      depthTest: false,
      depthWrite: false,
    })
    const displayMaterial = new THREE.ShaderMaterial({
      vertexShader: QUAD_VERTEX,
      fragmentShader: DISPLAY_FRAGMENT,
      uniforms: {
        uTrail: { value: null },
        uInvMean: { value: 1 },
        uExposure: { value: EXPOSURE },
        uFade: { value: 0 },
        uDeep: { value: new THREE.Color() },
        uMid: { value: new THREE.Color() },
        uHot: { value: new THREE.Color() },
      },
      depthTest: false,
      depthWrite: false,
    })

    // One fullscreen quad shared by the simulation passes (material swapped per pass)
    const simScene = new THREE.Scene()
    const simQuad = new THREE.Mesh(quadGeometry, agentMaterial)
    simQuad.frustumCulled = false
    simScene.add(simQuad)

    // Deposition: one point per agent; its "position" is the agent's texel coordinate
    const pointGeometry = new THREE.BufferGeometry()
    const refs = new Float32Array(AGENT_COUNT * 3)
    for (let i = 0; i < AGENT_COUNT; i++) {
      refs[i * 3] = ((i % AGENT_TEXTURE_SIZE) + 0.5) / AGENT_TEXTURE_SIZE
      refs[i * 3 + 1] = (Math.floor(i / AGENT_TEXTURE_SIZE) + 0.5) / AGENT_TEXTURE_SIZE
    }
    pointGeometry.setAttribute('position', new THREE.BufferAttribute(refs, 3))
    const points = new THREE.Points(pointGeometry, depositMaterial)
    points.frustumCulled = false
    const depositScene = new THREE.Scene()
    depositScene.add(points)

    const displayScene = new THREE.Scene()
    const displayQuad = new THREE.Mesh(quadGeometry, displayMaterial)
    displayQuad.frustumCulled = false
    displayScene.add(displayQuad)

    const composer = new EffectComposer(renderer)
    const renderPass = new RenderPass(displayScene, camera)
    composer.addPass(renderPass)
    const bloomPass = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 0.55, 0.45, 0.55)
    composer.addPass(bloomPass)

    const runPass = (material: THREE.ShaderMaterial, target: THREE.WebGLRenderTarget) => {
      simQuad.material = material
      renderer.setRenderTarget(target)
      renderer.render(simScene, camera)
    }

    const clearTrails = () => {
      renderer.setClearColor(0x000000, 0)
      for (const t of trailTargets) {
        renderer.setRenderTarget(t)
        renderer.clear()
      }
      renderer.setRenderTarget(null)
      renderer.setClearColor(0x000000, 1)
    }

    const syncSizeUniforms = () => {
      diffuseMaterial.uniforms.uTexel.value.set(1 / trailSize.x, 1 / trailSize.y)
      const minSide = Math.min(trailSize.x, trailSize.y)
      scatterMaterial.uniforms.uAspectScale.value.set(minSide / trailSize.x, minSide / trailSize.y)
      // With deposit = (1 - decay) the mean trail value equals the agent density, so this
      // normalization keeps brightness stable across species and screen sizes
      displayMaterial.uniforms.uInvMean.value = (trailSize.x * trailSize.y) / AGENT_COUNT
    }
    syncSizeUniforms()

    let lastFormation = -1
    const scatter = () => {
      const formation = pickOther(FORMATION_COUNT, Math.max(lastFormation, 0))
      lastFormation = formation
      scatterMaterial.uniforms.uFormation.value = formation
      scatterMaterial.uniforms.uSeed.value = Math.floor(Math.random() * 1e6)
      runPass(scatterMaterial, agentTargets[agentIndex])
      renderer.setRenderTarget(null)
      clearTrails()
    }

    // Species/palette state: `current` morphs from `from` to `to` during a transition
    let speciesIndex = Math.floor(Math.random() * SPECIES.length)
    let paletteIndex = 0 // open on the signature indigo/cyan
    let from = jitterSpecies(SPECIES[speciesIndex])
    let to = from
    const paletteFrom = { deep: new THREE.Color(), mid: new THREE.Color(), hot: new THREE.Color() }
    const paletteTo = { deep: new THREE.Color(), mid: new THREE.Color(), hot: new THREE.Color() }
    const setPaletteEndpoints = (fromP: Palette, toP: Palette) => {
      paletteFrom.deep.copy(fromP.deep)
      paletteFrom.mid.copy(fromP.mid)
      paletteFrom.hot.copy(fromP.hot)
      paletteTo.deep.copy(toP.deep)
      paletteTo.mid.copy(toP.mid)
      paletteTo.hot.copy(toP.hot)
    }
    setPaletteEndpoints(PALETTES[paletteIndex], PALETTES[paletteIndex])
    let current = from

    const handleResize = () => {
      renderer.setSize(window.innerWidth, window.innerHeight)
      composer.setSize(window.innerWidth, window.innerHeight)
      computeTrailSize()
      // Agents keep their normalized positions; only the trail map is reallocated
      for (const t of trailTargets) t.setSize(trailSize.x, trailSize.y)
      syncSizeUniforms()
      clearTrails()
    }

    const startTime = performance.now()
    let lastTime = startTime
    let accumulator = 0
    let stepCounter = 0
    let cycleStart = startTime
    let cycleLength = randomBetween(CYCLE_MIN_SECONDS, CYCLE_MAX_SECONDS)
    let cyclesSinceScatter = 0
    let transitionStart = -Infinity
    // Fade state: the colony opens by growing out of black from a fresh formation
    let fadeMode: 'in' | 'out' | 'none' = 'in'
    let fadeStart = startTime
    let animationId = 0

    scatter()

    const beginCycle = (now: number) => {
      cycleStart = now
      cycleLength = randomBetween(CYCLE_MIN_SECONDS, CYCLE_MAX_SECONDS)
      cyclesSinceScatter++
      // Re-scatter every 2-4 cycles; otherwise morph the living network into a new species
      const rescatter = cyclesSinceScatter >= 4 || (cyclesSinceScatter >= 2 && Math.random() < 0.5)
      if (rescatter) {
        fadeMode = 'out'
        fadeStart = now
        return
      }
      speciesIndex = pickOther(SPECIES.length, speciesIndex)
      const nextPalette = Math.random() < 0.5 ? paletteIndex : pickOther(PALETTES.length, paletteIndex)
      from = current
      to = jitterSpecies(SPECIES[speciesIndex])
      setPaletteEndpoints(PALETTES[paletteIndex], PALETTES[nextPalette])
      paletteIndex = nextPalette
      transitionStart = now
    }

    const rescatterNow = (now: number) => {
      cyclesSinceScatter = 0
      speciesIndex = pickOther(SPECIES.length, speciesIndex)
      paletteIndex = pickOther(PALETTES.length, paletteIndex)
      from = to = current = jitterSpecies(SPECIES[speciesIndex])
      setPaletteEndpoints(PALETTES[paletteIndex], PALETTES[paletteIndex])
      transitionStart = -Infinity
      scatter()
      fadeMode = 'in'
      fadeStart = now
    }

    const simulate = (params: Species, t: number) => {
      const distanceScale = Math.min(trailSize.x, trailSize.y) / REFERENCE_SHORT_SIDE
      const breath = 1 + Math.sin((t / BREATH_SECONDS) * Math.PI * 2) * BREATH_AMOUNT

      const au = agentMaterial.uniforms
      au.uAgents.value = agentTargets[agentIndex].texture
      au.uTrail.value = trailTargets[trailIndex].texture
      au.uSensorAngle.value = params.sensorAngle
      au.uSensorDistance.value = params.sensorDistance * distanceScale * breath
      au.uTurnSpeed.value = params.turnSpeed
      au.uStepSize.value = params.stepSize * distanceScale
      au.uWander.value = params.wander
      au.uStep.value = stepCounter
      stepCounter = (stepCounter + 1) % 65536
      runPass(agentMaterial, agentTargets[1 - agentIndex])
      agentIndex = 1 - agentIndex

      const du = diffuseMaterial.uniforms
      du.uTrail.value = trailTargets[trailIndex].texture
      du.uDecay.value = params.decay
      du.uDiffuse.value = params.diffuse
      runPass(diffuseMaterial, trailTargets[1 - trailIndex])
      trailIndex = 1 - trailIndex

      // Deposit additively on top of the freshly diffused map
      depositMaterial.uniforms.uAgents.value = agentTargets[agentIndex].texture
      depositMaterial.uniforms.uDeposit.value = 1 - params.decay
      renderer.setRenderTarget(trailTargets[trailIndex])
      renderer.render(depositScene, camera)
    }

    const animate = () => {
      animationId = requestAnimationFrame(animate)
      const now = performance.now()
      // Clamp so a backgrounded tab doesn't trigger a burst of catch-up steps
      const frameSeconds = Math.min((now - lastTime) / 1000, 0.1)
      lastTime = now
      const t = (now - startTime) / 1000

      if (fadeMode === 'none' && (now - cycleStart) / 1000 >= cycleLength) beginCycle(now)

      let fade = 1
      if (fadeMode === 'out') {
        const p = (now - fadeStart) / 1000 / FADE_OUT_SECONDS
        fade = 1 - smooth(p)
        if (p >= 1) {
          rescatterNow(now)
          fade = 0
        }
      } else if (fadeMode === 'in') {
        const p = (now - fadeStart) / 1000 / FADE_IN_SECONDS
        fade = smooth(p)
        if (p >= 1) {
          fadeMode = 'none'
          cycleStart = now
        }
      }

      const morph = smooth((now - transitionStart) / 1000 / TRANSITION_SECONDS)
      current = lerpSpecies(from, to, morph)

      accumulator += frameSeconds
      let steps = 0
      while (accumulator >= 1 / SIM_HZ && steps < MAX_STEPS_PER_FRAME) {
        simulate(current, t)
        accumulator -= 1 / SIM_HZ
        steps++
      }
      if (steps === MAX_STEPS_PER_FRAME) accumulator = 0
      renderer.setRenderTarget(null)

      const dispU = displayMaterial.uniforms
      dispU.uTrail.value = trailTargets[trailIndex].texture
      dispU.uFade.value = fade
      dispU.uExposure.value = EXPOSURE * (1 + Math.sin((t / GLOW_PULSE_SECONDS) * Math.PI * 2) * GLOW_PULSE_AMOUNT)
      dispU.uDeep.value.lerpColors(paletteFrom.deep, paletteTo.deep, morph)
      dispU.uMid.value.lerpColors(paletteFrom.mid, paletteTo.mid, morph)
      dispU.uHot.value.lerpColors(paletteFrom.hot, paletteTo.hot, morph)

      composer.render()
    }

    animate()
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      cancelAnimationFrame(animationId)
      for (const t of agentTargets) t.dispose()
      for (const t of trailTargets) t.dispose()
      agentMaterial.dispose()
      diffuseMaterial.dispose()
      scatterMaterial.dispose()
      depositMaterial.dispose()
      displayMaterial.dispose()
      quadGeometry.dispose()
      pointGeometry.dispose()
      bloomPass.dispose()
      renderPass.dispose()
      composer.dispose()
      renderer.dispose()
      container.removeChild(renderer.domElement)
    }
  }, [])

  return <div ref={containerRef} className="w-full h-full" />
}
