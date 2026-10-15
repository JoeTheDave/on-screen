import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import {
  FLOW_BANDS,
  FRONT_BANDS,
  coverage,
  createVortex,
  createWaveField,
  updateFlow,
  updateFronts,
  updateVortex,
} from './reactionDiffusionMath'

// Configuration constants
const SIM_LONG_SIDE = 512 // sim grid texels along the screen's long axis
const STEPS_PER_SECOND = 960 // Gray-Scott steps (dt = 1) per real second
const MAX_STEPS_PER_FRAME = 24 // cap so a slow frame can't snowball into slower frames
const CYCLE_SECONDS = 60 // new regime + palette + fresh drops
const SPRINKLE_SECONDS = 20 // a few extra drops in between, so nothing ever settles
const REGIME_BLEND_SECONDS = 25 // feed/kill glide from one regime to the next
const PALETTE_BLEND_SECONDS = 10
const MAX_DROPS = 24 // must match the size of uDrops (passed as a #define)
const INITIAL_DROPS = 22
const CYCLE_DROPS = 10
const SPRINKLE_DROPS = 4
const FLOW_SPEED = 1 // scales the stirring flow (crest speeds live in reactionDiffusionMath)
const FLOW_WAVES = FLOW_BANDS.length // sizes of the shader's uniform arrays (#defines)
const FRONT_WAVES = FRONT_BANDS.length
const PROBE_SIZE = 32 // coverage probe grid, read back now and then to catch a dead/solid field
const PROBE_SECONDS = 3
const DEAD_COVERAGE = 0.03 // below this share of B, re-seed
const SOLID_COVERAGE = 0.9 // above it, punch holes
const BUMP = 3.2 // height-gradient -> normal steepness
const LIGHT_SECONDS_X = 57
const LIGHT_SECONDS_Y = 83

// Well-known Gray-Scott regimes (feed, kill) — Pearson / Sims / Munafo naming
const REGIMES: [number, number][] = [
  [0.0545, 0.062], // coral
  [0.0367, 0.0649], // mitosis: spots that grow and split
  [0.029, 0.057], // labyrinth / fingerprints
  [0.078, 0.061], // worms
  [0.03, 0.062], // solitons: stable spots
  [0.039, 0.058], // holes / negative spots
]

// Regime pairs whose in-between (F, k) fills solid with B, so they never share the screen or
// glide straight into each other
const CLASHING_PAIRS = ['2-3', '3-5'] // labyrinth/worms, worms/holes
const clashes = (i: number, j: number) =>
  i === j || CLASHING_PAIRS.includes(`${Math.min(i, j)}-${Math.max(i, j)}`)

// [deep valley, mid body, raised peak, specular tint] — sRGB hex
const PALETTES: [number, number, number, number][] = [
  [0x0b0420, 0xc2384a, 0xffc9a3, 0xfff1e6], // coral reef
  [0x02130f, 0x138a72, 0xd8f5e0, 0xeafff6], // jade enamel
  [0x050a24, 0x1d3fa8, 0xf0c25a, 0xfff4d6], // lapis and gold
  [0x0d0216, 0x7b2fa8, 0xf5d9ff, 0xffffff], // amethyst pearl
  [0x120300, 0xb33a06, 0xffd27a, 0xfff0d0], // ember glaze
  [0x01101c, 0x0e6e9e, 0xc9f3ff, 0xf0fbff], // deep ocean ice
  [0x061418, 0x1a7f86, 0xff9fb2, 0xffeef2], // teal and rose
  [0x100a02, 0x6b5a12, 0xe8f0a0, 0xffffe8], // olive lacquer
]

const vertexShader = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`

// One advected Gray-Scott step. The red channel stores (1 - A) rather than A: A sits near 1
// almost everywhere, where half-float spacing is ~0.0005 — coarser than the per-step changes.
// Near 0 half floats are very precise, so storing the complement keeps the reaction from stalling.
//
// First the field is carried along a slow, divergence-free flow (semi-Lagrangian: read the state
// at uv - vel). The displacement is under a texel, so the bilinear lookup is done by hand from the
// 3x3 neighbourhood already fetched for the Laplacian — full float precision, no extra fetches,
// and none of the 8-bit weight quantization of hardware filtering that would freeze slow regions.
// Then feed/kill come from two regimes blended by broad traveling fronts, so as a front rolls past,
// one kind of pattern converts into the other.
const simFragmentShader = /* glsl */ `
uniform sampler2D uState;
uniform vec2 uTexel;
uniform vec2 uSimSize;
uniform vec2 uFeedKillA;
uniform vec2 uFeedKillB;
uniform vec4 uFlowWaves[FLOW_WAVES]; // kx, ky (integer, per uv), phase, unused
uniform vec2 uFlowVel[FLOW_WAVES]; // velocity at the crest, texels/step
uniform vec4 uFronts[FRONT_WAVES]; // kx, ky, phase, weight (weights sum to 1)
uniform vec4 uVortex; // center (uv), radius (texels), spin
varying vec2 vUv;

const float TAU = 6.2831853;

void main() {
  vec2 c = texture2D(uState, vUv).rg;
  vec2 n = texture2D(uState, vUv + vec2(0.0, uTexel.y)).rg;
  vec2 s = texture2D(uState, vUv - vec2(0.0, uTexel.y)).rg;
  vec2 e = texture2D(uState, vUv + vec2(uTexel.x, 0.0)).rg;
  vec2 w = texture2D(uState, vUv - vec2(uTexel.x, 0.0)).rg;
  vec2 ne = texture2D(uState, vUv + uTexel).rg;
  vec2 sw = texture2D(uState, vUv - uTexel).rg;
  vec2 nw = texture2D(uState, vUv + vec2(-uTexel.x, uTexel.y)).rg;
  vec2 se = texture2D(uState, vUv + vec2(uTexel.x, -uTexel.y)).rg;
  vec2 lap = 0.2 * (n + s + e + w) + 0.05 * (ne + nw + se + sw) - c;

  // Flow: curl of a sum of plane-wave stream functions, plus an occasional Gaussian swirl
  vec2 vel = vec2(0.0);
  for (int i = 0; i < FLOW_WAVES; i++) {
    vec4 fw = uFlowWaves[i];
    vel += uFlowVel[i] * cos(TAU * dot(fw.xy, vUv) + fw.z);
  }
  vec2 d = vUv - uVortex.xy;
  d -= floor(d + 0.5); // shortest wrapped offset
  d *= uSimSize;
  vel += uVortex.w * exp(-dot(d, d) / (uVortex.z * uVortex.z)) * vec2(-d.y, d.x);
  vec2 o = -clamp(vel, -0.9, 0.9); // source offset in texels, kept inside the 3x3 neighbourhood
  vec2 f = abs(o);
  vec2 hx = o.x > 0.0 ? e : w;
  vec2 vy = o.y > 0.0 ? n : s;
  vec2 cr = o.x > 0.0 ? (o.y > 0.0 ? ne : se) : (o.y > 0.0 ? nw : sw);
  vec2 adv = mix(mix(c, hx, f.x), mix(vy, cr, f.x), f.y);

  // Traveling fronts between the two regimes
  float band = 0.0;
  for (int i = 0; i < FRONT_WAVES; i++) {
    vec4 fr = uFronts[i];
    band += fr.w * sin(TAU * dot(fr.xy, vUv) + fr.z);
  }
  vec2 fk = mix(uFeedKillA, uFeedKillB, smoothstep(-0.3, 0.3, band));
  float feed = fk.x;
  float kill = fk.y;

  float a = 1.0 - adv.r;
  float b = adv.g;
  float lapA = -lap.r;
  float lapB = lap.g;

  float reaction = a * b * b;
  float na = a + lapA - reaction + feed * (1.0 - a);
  float nb = b + 0.5 * lapB + reaction - (feed + kill) * b;

  gl_FragColor = vec4(clamp(1.0 - na, 0.0, 1.0), clamp(nb, 0.0, 1.0), 0.0, 1.0);
}
`

// Paints drops into the field (w > 0 seeds chemical B, w < 0 erases back to bare A), or
// clears everything first. With no drops and no clear it's a plain resampling copy, which
// is how the state survives a resize onto a differently sized grid.
const seedFragmentShader = /* glsl */ `
uniform sampler2D uState;
uniform float uClear;
uniform vec2 uSimSize;
uniform float uSeed;
uniform vec4 uDrops[MAX_DROPS];
varying vec2 vUv;

float hash(vec2 q) {
  return fract(sin(dot(q, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
  vec2 state = uClear > 0.5 ? vec2(0.0) : texture2D(uState, vUv).rg;
  float n = hash(floor(vUv * uSimSize) + uSeed);
  for (int i = 0; i < MAX_DROPS; i++) {
    vec4 d = uDrops[i];
    if (d.z <= 0.0) continue;
    vec2 diff = vUv - d.xy;
    diff -= floor(diff + 0.5); // shortest wrapped offset
    float dist = length(diff * uSimSize);
    // Ragged, noisy edges break the symmetry so drops don't grow as perfect rings
    if (dist < d.z * (0.7 + 0.3 * n)) {
      state = d.w > 0.0 ? vec2(0.5, 0.25 + 0.1 * n) : vec2(0.0);
    }
  }
  gl_FragColor = vec4(state, 0.0, 1.0);
}
`

// Coverage probe: B (x2, so its usual 0..0.5 range fills a byte) on a tiny 8-bit grid
const probeFragmentShader = /* glsl */ `
uniform sampler2D uState;
varying vec2 vUv;

void main() {
  gl_FragColor = vec4(clamp(texture2D(uState, vUv).g * 2.0, 0.0, 1.0), 0.0, 0.0, 1.0);
}
`

// Treats chemical B as a height field and lights it as glossy enamel under a wandering
// point light: gradient normals, crevice occlusion, sharp + broad specular, fresnel sheen.
const displayFragmentShader = /* glsl */ `
uniform sampler2D uState;
uniform vec2 uTexel;
uniform float uAspect;
uniform vec3 uLight;
uniform float uBump;
uniform vec3 uDeep;
uniform vec3 uMid;
uniform vec3 uPeak;
uniform vec3 uSpec;
varying vec2 vUv;

float height(vec2 uv) {
  return smoothstep(0.05, 0.32, texture2D(uState, uv).g);
}

void main() {
  vec2 dx = vec2(uTexel.x, 0.0);
  vec2 dy = vec2(0.0, uTexel.y);
  float h = height(vUv);

  // Sobel gradient over one sim texel: smoother than plain central differences on an
  // upscaled grid, so the relief doesn't look faceted
  float tl = height(vUv - dx + dy);
  float tc = height(vUv + dy);
  float tr = height(vUv + dx + dy);
  float ml = height(vUv - dx);
  float mr = height(vUv + dx);
  float bl = height(vUv - dx - dy);
  float bc = height(vUv - dy);
  float br = height(vUv + dx - dy);
  float gx = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl);
  float gy = (tl + 2.0 * tc + tr) - (bl + 2.0 * bc + br);
  vec3 N = normalize(vec3(-gx * uBump * 0.25, -gy * uBump * 0.25, 1.0));

  // Crevice occlusion: darker where the surroundings stand higher than this point
  float ring = height(vUv + 3.0 * dx) + height(vUv - 3.0 * dx)
             + height(vUv + 3.0 * dy) + height(vUv - 3.0 * dy)
             + height(vUv + 2.1 * (dx + dy)) + height(vUv - 2.1 * (dx + dy))
             + height(vUv + 2.1 * (dx - dy)) + height(vUv - 2.1 * (dx - dy));
  float ao = clamp(1.0 - (ring / 8.0 - h) * 1.4, 0.35, 1.0);

  vec3 P = vec3((vUv - 0.5) * vec2(uAspect, 1.0), h * 0.025);
  vec3 toLight = uLight - P;
  float dist2 = dot(toLight, toLight);
  vec3 L = toLight * inversesqrt(dist2);
  vec3 V = vec3(0.0, 0.0, 1.0);
  vec3 H = normalize(L + V);
  float falloff = 1.0 / (1.0 + dist2 * 1.6);

  float diffuse = max(dot(N, L), 0.0);
  float nh = max(dot(N, H), 0.0);
  float glint = pow(nh, 90.0);
  float sheen = pow(nh, 14.0);
  float fresnel = pow(1.0 - N.z, 2.0);

  vec3 albedo = mix(uDeep, uMid, smoothstep(0.0, 0.55, h));
  albedo = mix(albedo, uPeak, smoothstep(0.55, 1.0, h));

  // Raised enamel is glossier than the matte valleys between it
  float gloss = 0.25 + 0.75 * h;
  vec3 col = albedo * (0.12 + 1.5 * diffuse * falloff) * ao;
  col += uSpec * (glint * 1.6 + sheen * 0.18) * falloff * gloss;
  col += mix(uMid, uPeak, N.y * 0.5 + 0.5) * fresnel * 0.35 * ao;

  // Gentle vignette, soft shoulder, then linear -> sRGB by hand (ShaderMaterial won't)
  vec2 q = vUv - 0.5;
  col *= 1.0 - 0.45 * dot(q, q) * 2.0;
  col = 1.0 - exp(-col * 1.25);
  gl_FragColor = vec4(pow(col, vec3(1.0 / 2.2)), 1.0);
}
`

interface Palette {
  deep: THREE.Color
  mid: THREE.Color
  peak: THREE.Color
  spec: THREE.Color
}

const toPalette = (hexes: [number, number, number, number]): Palette => ({
  deep: new THREE.Color(hexes[0]),
  mid: new THREE.Color(hexes[1]),
  peak: new THREE.Color(hexes[2]),
  spec: new THREE.Color(hexes[3]),
})

const pickOther = (count: number, current: number) => {
  let next = Math.floor(Math.random() * (count - 1))
  if (next >= current) next++
  return next
}

// Random regime that neither equals nor clashes with any of `avoid` (always solvable: at most
// two avoided regimes, each ruling out at most three of the six)
const pickRegime = (...avoid: number[]) => {
  const options = REGIMES.map((_, i) => i).filter((i) => avoid.every((a) => !clashes(i, a)))
  return options[Math.floor(Math.random() * options.length)]
}

const smooth = (x: number) => {
  const t = Math.min(Math.max(x, 0), 1)
  return t * t * (3 - 2 * t)
}

export default function ReactionDiffusion() {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!containerRef.current) return
    const container = containerRef.current

    const renderer = new THREE.WebGLRenderer({ antialias: false })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(window.innerWidth, window.innerHeight)
    renderer.setClearColor(0x000000, 1)
    container.appendChild(renderer.domElement)

    // Half-float targets are renderable via either extension on WebGL2; without them the
    // sim can't run correctly, but rendering anyway beats a blank screen
    if (
      !renderer.extensions.has('EXT_color_buffer_half_float') &&
      !renderer.extensions.has('EXT_color_buffer_float')
    ) {
      console.warn('ReactionDiffusion: half-float render targets unsupported; output may be wrong')
    }

    const scene = new THREE.Scene()
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1)
    const geometry = new THREE.PlaneGeometry(2, 2)

    const simSize = new THREE.Vector2()
    const simTexel = new THREE.Vector2()

    // Flow, fronts and vortex are computed on the CPU each frame (as small Float32Arrays, then
    // copied into these preallocated uniform vectors)
    const flowWaves = Array.from({ length: FLOW_WAVES }, () => new THREE.Vector4())
    const flowVel = Array.from({ length: FLOW_WAVES }, () => new THREE.Vector2())
    const fronts = Array.from({ length: FRONT_WAVES }, () => new THREE.Vector4())
    const vortexUniform = new THREE.Vector4(0.5, 0.5, 1, 0)
    const flowWaveData = new Float32Array(FLOW_WAVES * 4)
    const flowVelData = new Float32Array(FLOW_WAVES * 2)
    const frontData = new Float32Array(FRONT_WAVES * 4)
    const vortexData = new Float32Array(4)

    const simMaterial = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader: simFragmentShader,
      defines: { FLOW_WAVES, FRONT_WAVES },
      uniforms: {
        uState: { value: null },
        uTexel: { value: simTexel },
        uSimSize: { value: simSize },
        uFeedKillA: { value: new THREE.Vector2(...REGIMES[0]) },
        uFeedKillB: { value: new THREE.Vector2(...REGIMES[1]) },
        uFlowWaves: { value: flowWaves },
        uFlowVel: { value: flowVel },
        uFronts: { value: fronts },
        uVortex: { value: vortexUniform },
      },
      depthTest: false,
      depthWrite: false,
    })

    const drops = Array.from({ length: MAX_DROPS }, () => new THREE.Vector4(0, 0, 0, 0))
    const seedMaterial = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader: seedFragmentShader,
      defines: { MAX_DROPS },
      uniforms: {
        uState: { value: null },
        uClear: { value: 0 },
        uSimSize: { value: simSize },
        uSeed: { value: 0 },
        uDrops: { value: drops },
      },
      depthTest: false,
      depthWrite: false,
    })

    const probeMaterial = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader: probeFragmentShader,
      uniforms: { uState: { value: null } },
      depthTest: false,
      depthWrite: false,
    })
    const probeTarget = new THREE.WebGLRenderTarget(PROBE_SIZE, PROBE_SIZE, {
      type: THREE.UnsignedByteType,
      format: THREE.RGBAFormat,
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      depthBuffer: false,
      stencilBuffer: false,
    })
    const probePixels = new Uint8Array(PROBE_SIZE * PROBE_SIZE * 4)

    const shown = toPalette(PALETTES[0])
    const displayMaterial = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader: displayFragmentShader,
      uniforms: {
        uState: { value: null },
        uTexel: { value: simTexel },
        uAspect: { value: 1 },
        uLight: { value: new THREE.Vector3(0, 0, 0.6) },
        uBump: { value: BUMP },
        uDeep: { value: shown.deep },
        uMid: { value: shown.mid },
        uPeak: { value: shown.peak },
        uSpec: { value: shown.spec },
      },
      depthTest: false,
      depthWrite: false,
    })

    const quad = new THREE.Mesh(geometry, simMaterial)
    quad.frustumCulled = false
    scene.add(quad)

    const makeTarget = (w: number, h: number) =>
      new THREE.WebGLRenderTarget(w, h, {
        type: THREE.HalfFloatType,
        format: THREE.RGBAFormat,
        // Sim passes sample exactly at texel centers, so linear filtering is harmless there
        // and gives the display pass a smooth upscale
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        wrapS: THREE.RepeatWrapping,
        wrapT: THREE.RepeatWrapping,
        depthBuffer: false,
        stencilBuffer: false,
      })

    let read: THREE.WebGLRenderTarget | null = null
    let write: THREE.WebGLRenderTarget | null = null

    const runPass = (material: THREE.ShaderMaterial, target: THREE.WebGLRenderTarget | null) => {
      quad.material = material
      renderer.setRenderTarget(target)
      renderer.render(scene, camera)
    }

    const swap = () => {
      const t = read
      read = write
      write = t
    }

    const clearDrops = () => {
      for (const d of drops) d.set(0, 0, 0, 0)
    }

    // Applies whatever is in `drops` (and optionally a full clear) to the field
    const applySeed = (clear: boolean) => {
      if (!read || !write) return
      seedMaterial.uniforms.uState.value = read.texture
      seedMaterial.uniforms.uClear.value = clear ? 1 : 0
      seedMaterial.uniforms.uSeed.value = Math.random() * 1000
      runPass(seedMaterial, write)
      swap()
      clearDrops()
    }

    // count seeds + a share of erasers; erased holes visibly regrow from their rims
    const scatterDrops = (count: number, eraseShare: number) => {
      clearDrops()
      for (let i = 0; i < Math.min(count, MAX_DROPS); i++) {
        const erase = Math.random() < eraseShare
        drops[i].set(
          Math.random(),
          Math.random(),
          erase ? 14 + Math.random() * 26 : 3 + Math.random() * 9,
          erase ? -1 : 1,
        )
      }
    }

    const handleResize = () => {
      const w = window.innerWidth
      const h = window.innerHeight
      renderer.setSize(w, h)
      displayMaterial.uniforms.uAspect.value = w / h

      const long = Math.max(w, h)
      const simW = Math.max(16, Math.round((SIM_LONG_SIDE * w) / long))
      const simH = Math.max(16, Math.round((SIM_LONG_SIDE * h) / long))
      if (read && simW === simSize.x && simH === simSize.y) return

      simSize.set(simW, simH)
      simTexel.set(1 / simW, 1 / simH)
      const oldRead = read
      const oldWrite = write
      read = makeTarget(simW, simH)
      write = makeTarget(simW, simH)
      if (oldRead) {
        // Resample the living pattern onto the new grid rather than starting over
        seedMaterial.uniforms.uState.value = oldRead.texture
        seedMaterial.uniforms.uClear.value = 0
        runPass(seedMaterial, read)
        oldRead.dispose()
        oldWrite?.dispose()
      } else {
        scatterDrops(INITIAL_DROPS, 0)
        applySeed(true)
      }
    }
    handleResize()

    // Regime and palette state; both glide from where they are toward a new target. Two regimes
    // are live at once (A and B), and the traveling fronts decide which holds where.
    let regimeIndex = Math.floor(Math.random() * REGIMES.length)
    let regimeIndexB = pickRegime(regimeIndex)
    const feedKillFrom = new THREE.Vector2(...REGIMES[regimeIndex])
    const feedKillTo = feedKillFrom.clone()
    const feedKill = simMaterial.uniforms.uFeedKillA.value as THREE.Vector2
    feedKill.copy(feedKillFrom)
    const feedKillFromB = new THREE.Vector2(...REGIMES[regimeIndexB])
    const feedKillToB = feedKillFromB.clone()
    const feedKillB = simMaterial.uniforms.uFeedKillB.value as THREE.Vector2
    feedKillB.copy(feedKillFromB)

    let paletteIndex = Math.floor(Math.random() * PALETTES.length)
    let paletteFrom = toPalette(PALETTES[paletteIndex])
    let paletteTo = toPalette(PALETTES[paletteIndex])
    shown.deep.copy(paletteTo.deep)
    shown.mid.copy(paletteTo.mid)
    shown.peak.copy(paletteTo.peak)
    shown.spec.copy(paletteTo.spec)

    const startTime = performance.now()
    let lastTime = startTime
    let cycleStart = startTime
    let lastSprinkle = startTime
    let lastProbe = startTime
    let probePending = false
    let disposed = false
    let stepDebt = 0
    let animationId = 0

    const flowField = createWaveField(FLOW_BANDS, 0, Math.random)
    const frontField = createWaveField(FRONT_BANDS, 0, Math.random)
    const vortex = createVortex(0, Math.random)

    // Now and then, read back a tiny map of B; if the field has died out or filled solid,
    // re-seed or punch holes rather than waiting for the next scheduled drops
    const probeCoverage = () => {
      if (probePending || !read) return
      probeMaterial.uniforms.uState.value = read.texture
      runPass(probeMaterial, probeTarget)
      probePending = true
      renderer
        .readRenderTargetPixelsAsync(probeTarget, 0, 0, PROBE_SIZE, PROBE_SIZE, probePixels)
        .then(() => {
          if (disposed) return
          const share = coverage(probePixels)
          if (share < DEAD_COVERAGE) {
            scatterDrops(INITIAL_DROPS, 0)
            applySeed(false)
          } else if (share > SOLID_COVERAGE) {
            scatterDrops(CYCLE_DROPS, 1)
            applySeed(false)
          }
        })
        .catch(() => {
          // Readback unsupported or context lost: the scheduled drops still keep things going
        })
        .finally(() => {
          probePending = false
        })
    }

    const startCycle = (now: number) => {
      cycleStart = now
      lastSprinkle = now
      feedKillFrom.copy(feedKill)
      regimeIndex = pickRegime(regimeIndex)
      feedKillTo.set(...REGIMES[regimeIndex])
      feedKillFromB.copy(feedKillB)
      regimeIndexB = pickRegime(regimeIndex, regimeIndexB)
      feedKillToB.set(...REGIMES[regimeIndexB])
      paletteFrom = {
        deep: shown.deep.clone(),
        mid: shown.mid.clone(),
        peak: shown.peak.clone(),
        spec: shown.spec.clone(),
      }
      paletteIndex = pickOther(PALETTES.length, paletteIndex)
      paletteTo = toPalette(PALETTES[paletteIndex])
      scatterDrops(CYCLE_DROPS, 0.35)
      applySeed(false)
    }

    const animate = () => {
      animationId = requestAnimationFrame(animate)
      const now = performance.now()
      // Clamp so a backgrounded tab doesn't come back owing thousands of steps
      const frameSeconds = Math.min((now - lastTime) / 1000, 1 / 20)
      lastTime = now
      const t = (now - startTime) / 1000

      if ((now - cycleStart) / 1000 >= CYCLE_SECONDS) {
        startCycle(now)
      } else if ((now - lastSprinkle) / 1000 >= SPRINKLE_SECONDS) {
        lastSprinkle = now
        scatterDrops(SPRINKLE_DROPS, 0.5)
        applySeed(false)
      }

      const cycleT = (now - cycleStart) / 1000
      const rb = smooth(cycleT / REGIME_BLEND_SECONDS)
      feedKill.lerpVectors(feedKillFrom, feedKillTo, rb)
      feedKillB.lerpVectors(feedKillFromB, feedKillToB, rb)
      const pb = smooth(cycleT / PALETTE_BLEND_SECONDS)
      shown.deep.lerpColors(paletteFrom.deep, paletteTo.deep, pb)
      shown.mid.lerpColors(paletteFrom.mid, paletteTo.mid, pb)
      shown.peak.lerpColors(paletteFrom.peak, paletteTo.peak, pb)
      shown.spec.lerpColors(paletteFrom.spec, paletteTo.spec, pb)

      stepDebt += frameSeconds * STEPS_PER_SECOND
      const steps = Math.min(Math.floor(stepDebt), MAX_STEPS_PER_FRAME)
      stepDebt = Math.min(stepDebt - steps, 1)
      updateFlow(flowField, t, simSize.x, simSize.y, FLOW_SPEED, flowWaveData, flowVelData)
      updateFronts(frontField, t, frontData)
      updateVortex(vortex, t, simSize.x, simSize.y, Math.random, vortexData)
      flowWaves.forEach((v, i) => v.fromArray(flowWaveData, i * 4))
      flowVel.forEach((v, i) => v.fromArray(flowVelData, i * 2))
      fronts.forEach((v, i) => v.fromArray(frontData, i * 4))
      vortexUniform.fromArray(vortexData)
      for (let i = 0; i < steps; i++) {
        simMaterial.uniforms.uState.value = read!.texture
        runPass(simMaterial, write)
        swap()
      }

      if ((now - lastProbe) / 1000 >= PROBE_SECONDS) {
        lastProbe = now
        probeCoverage()
      }

      // Light wanders a Lissajous path just above the surface, sweeping highlights across it
      const aspect = displayMaterial.uniforms.uAspect.value as number
      ;(displayMaterial.uniforms.uLight.value as THREE.Vector3).set(
        Math.sin((t / LIGHT_SECONDS_X) * Math.PI * 2) * 0.42 * aspect,
        Math.sin((t / LIGHT_SECONDS_Y) * Math.PI * 2 + 1.1) * 0.38,
        0.55 + 0.1 * Math.sin(t * 0.05),
      )
      displayMaterial.uniforms.uState.value = read!.texture
      runPass(displayMaterial, null)
    }

    animate()
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      cancelAnimationFrame(animationId)
      disposed = true
      probeTarget.dispose()
      probeMaterial.dispose()
      read?.dispose()
      write?.dispose()
      geometry.dispose()
      simMaterial.dispose()
      seedMaterial.dispose()
      displayMaterial.dispose()
      renderer.dispose()
      container.removeChild(renderer.domElement)
    }
  }, [])

  return <div ref={containerRef} className="w-full h-full" />
}
