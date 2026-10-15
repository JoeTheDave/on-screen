import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import {
  BLEACH_KILL,
  ORIGINS,
  REGIMES,
  SEED_BAND,
  SEED_CELL,
  SEED_GAP,
  SEED_RADIUS,
  WOBBLES,
  coverage,
  frontRadius,
  measureFront,
  phaseSeconds,
  planCycle,
  startPhase,
  type Phase,
} from './reactionDiffusionMath'

// Configuration constants (phase timing, regimes and transitions live in reactionDiffusionMath)
const SIM_LONG_SIDE = 512 // sim grid texels along the screen's long axis
const STEPS_PER_SECOND = 960 // Gray-Scott steps (dt = 1) per sim-second; phases run on sim time
const MAX_STEPS_PER_FRAME = 24 // cap so a slow frame can't snowball into slower frames
const MAX_DROPS = 24 // must match the size of uDrops (passed as a #define)
const INITIAL_DROPS = 22
const RESCUE_DROPS = 10
const PROBE_SIZE = 32 // coverage probe grid, read back now and then to catch a dead/solid field
const PROBE_SECONDS = 3
const GUARD_GRACE_SECONDS = 10 // a fresh spread starts below DEAD_COVERAGE; let it grow first
const DEAD_COVERAGE = 0.03 // below this share of B, re-seed
const SOLID_COVERAGE = 0.985 // above it, punch holes (the holes regime itself reads up to ~0.92)
const GLOW_FADE_SECONDS = 1.5 // the front's bright edge fades in as it appears
const BUMP = 3.2 // height-gradient -> normal steepness
const LIGHT_SECONDS_X = 57
const LIGHT_SECONDS_Y = 83

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

const glslFloat = (x: number) => x.toFixed(4)

const vertexShader = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`

// The advancing front, shared by the sim and display passes (and mirrored by frontDistance in
// reactionDiffusionMath): texel distance to the nearest origin, wrapped because the sim tiles,
// plus a few integer-wavenumber ripples so the edge is organic rather than a perfect circle.
// An origin with weights (0, 1) or (1, 0) is a line, and its front is a pair of parallel bands.
const frontChunk = /* glsl */ `
uniform vec2 uSimSize;
uniform vec4 uOrigins[ORIGINS]; // u, v, weight x, weight y
uniform vec4 uWobble[WOBBLES]; // kx, ky (integer, per uv), phase, amplitude (texels)

float frontDistance(vec2 uv) {
  float d = 1e6;
  for (int i = 0; i < ORIGINS; i++) {
    vec2 o = uv - uOrigins[i].xy;
    o -= floor(o + 0.5);
    d = min(d, length(o * uSimSize * uOrigins[i].zw));
  }
  for (int i = 0; i < WOBBLES; i++) {
    vec4 w = uWobble[i];
    d += w.w * sin(6.2831853 * dot(w.xy, uv) + w.z);
  }
  return d;
}
`

// One Gray-Scott step. The red channel stores (1 - A) rather than A: A sits near 1 almost
// everywhere, where half-float spacing is ~0.0005 — coarser than the per-step changes. Near 0
// half floats are very precise, so storing the complement keeps the reaction from stalling.
//
// Feed/kill come from the new regime behind the front and the old one ahead of it. Optionally a
// band just behind the front bleaches B away, and behind that nuclei (fixed, hashed discs) are
// planted so the new regime has something to grow from where the old pattern dissolves.
const simFragmentShader = /* glsl */ `
uniform sampler2D uState;
uniform vec2 uTexel;
uniform vec2 uFeedKillOld;
uniform vec2 uFeedKillNew;
uniform vec4 uFront; // radius (texels), bleach width (texels), seed density, seed salt
varying vec2 vUv;
${frontChunk}
uint hash3(uvec3 v) {
  uint h = (v.x * 0x9E3779B1u) ^ (v.y * 0x85EBCA77u) ^ (v.z * 0xC2B2AE3Du);
  h ^= h >> 15u;
  h *= 0x2C1B3C6Du;
  h ^= h >> 12u;
  h *= 0x297A2D39u;
  h ^= h >> 15u;
  return h;
}

// At most one nucleus per seed cell, jittered inside it; same test as seedNucleus on the CPU.
// Returns whether this texel is inside one, and its center (texels).
bool nucleus(float density, uint salt, out vec2 center) {
  vec2 p = floor(vUv * uSimSize);
  uvec2 cell = uvec2(p) / uint(SEED_CELL);
  uint h = hash3(uvec3(cell, salt));
  vec2 jitter = vec2(float((h >> 16u) & 0xFFu), float(h >> 24u)) / 255.0;
  center = (vec2(cell) + 0.25 + 0.5 * jitter) * float(SEED_CELL);
  return float(h & 0xFFFFu) / 65536.0 < density && distance(p + 0.5, center) < SEED_RADIUS;
}

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

  // Signed texels behind the front (negative: not reached yet)
  float behind = uFront.x - frontDistance(vUv);
  vec2 fk = mix(uFeedKillOld, uFeedKillNew, smoothstep(-1.5, 1.5, behind));
  float bleach = behind > 0.0 && behind < uFront.y ? BLEACH_KILL : 0.0;

  float a = 1.0 - c.r;
  float b = c.g;
  float reaction = a * b * b;
  float na = a - lap.r - reaction + fk.x * (1.0 - a);
  float nb = b + 0.5 * lap.g + reaction - (fk.x + fk.y + bleach) * b;

  // Nuclei are held whole, keyed to their center's distance behind the front: a band sweeping
  // across a disc would leave only thin arcs, which fizzle in the coral and worm regimes. The
  // cheap pre-check skips the hash away from the seeding ring (wide enough for the wobble).
  float seedStart = max(uFront.y, SEED_GAP) + SEED_RADIUS;
  vec2 center;
  if (uFront.z > 0.0 && abs(behind - seedStart - 0.5 * SEED_BAND) < 0.5 * SEED_BAND + 4.0 * SEED_RADIUS &&
      nucleus(uFront.z, uint(uFront.w), center)) {
    float centerBehind = uFront.x - frontDistance(center / uSimSize);
    if (centerBehind > seedStart && centerBehind < seedStart + SEED_BAND) {
      na += (0.5 - na) * 0.25;
      nb += (0.25 - nb) * 0.25;
    }
  }

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
// The palette changes a few texels behind the advancing front, whose leading edge glows faintly,
// so the viewer reads "something new is spreading through".
const displayFragmentShader = /* glsl */ `
uniform sampler2D uState;
uniform vec2 uTexel;
uniform float uAspect;
uniform vec3 uLight;
uniform float uBump;
uniform float uFrontRadius;
uniform float uFrontGlow;
uniform vec3 uDeepOld;
uniform vec3 uMidOld;
uniform vec3 uPeakOld;
uniform vec3 uSpecOld;
uniform vec3 uDeepNew;
uniform vec3 uMidNew;
uniform vec3 uPeakNew;
uniform vec3 uSpecNew;
varying vec2 vUv;
${frontChunk}
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

  // Palette: new behind the front, old ahead, changing over a soft band just behind the edge
  float behind = uFrontRadius - frontDistance(vUv);
  float pal = smoothstep(-6.0, 18.0, behind);
  vec3 deep = mix(uDeepOld, uDeepNew, pal);
  vec3 mid = mix(uMidOld, uMidNew, pal);
  vec3 peak = mix(uPeakOld, uPeakNew, pal);
  vec3 spec = mix(uSpecOld, uSpecNew, pal);

  vec3 albedo = mix(deep, mid, smoothstep(0.0, 0.55, h));
  albedo = mix(albedo, peak, smoothstep(0.55, 1.0, h));

  // Raised enamel is glossier than the matte valleys between it
  float gloss = 0.25 + 0.75 * h;
  vec3 col = albedo * (0.12 + 1.5 * diffuse * falloff) * ao;
  col += spec * (glint * 1.6 + sheen * 0.18) * falloff * gloss;
  col += mix(mid, peak, N.y * 0.5 + 0.5) * fresnel * 0.35 * ao;

  // The front: a thin bright leading edge with a faint wash trailing behind it
  float edge = exp(-behind * behind / 12.0);
  float wash = behind > 0.0 ? exp(-behind / 28.0) : 0.0;
  col += (mix(uPeakNew, uSpecNew, 0.5) * edge * 0.32 + uMidNew * wash * 0.07) * uFrontGlow;

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

const newPalette = (): Palette => ({
  deep: new THREE.Color(),
  mid: new THREE.Color(),
  peak: new THREE.Color(),
  spec: new THREE.Color(),
})

const setPalette = (p: Palette, hexes: [number, number, number, number]) => {
  p.deep.setHex(hexes[0])
  p.mid.setHex(hexes[1])
  p.peak.setHex(hexes[2])
  p.spec.setHex(hexes[3])
}

const copyPalette = (to: Palette, from: Palette) => {
  to.deep.copy(from.deep)
  to.mid.copy(from.mid)
  to.peak.copy(from.peak)
  to.spec.copy(from.spec)
}

const frontDefines = { ORIGINS, WOBBLES }

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

    // Front geometry for the current phase, shared by the sim and display passes
    const origins = Array.from({ length: ORIGINS }, () => new THREE.Vector4())
    const wobble = Array.from({ length: WOBBLES }, () => new THREE.Vector4())
    const front = new THREE.Vector4() // radius, bleach width, seed density, seed salt

    const simMaterial = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader: simFragmentShader,
      defines: {
        ...frontDefines,
        SEED_CELL,
        SEED_RADIUS: glslFloat(SEED_RADIUS),
        SEED_GAP: glslFloat(SEED_GAP),
        SEED_BAND: glslFloat(SEED_BAND),
        BLEACH_KILL: glslFloat(BLEACH_KILL),
      },
      uniforms: {
        uState: { value: null },
        uTexel: { value: simTexel },
        uSimSize: { value: simSize },
        uFeedKillOld: { value: new THREE.Vector2() },
        uFeedKillNew: { value: new THREE.Vector2() },
        uOrigins: { value: origins },
        uWobble: { value: wobble },
        uFront: { value: front },
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

    // Palette ahead of the front (old) and behind it (new)
    const paletteOld = newPalette()
    const paletteNew = newPalette()
    const displayMaterial = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader: displayFragmentShader,
      defines: frontDefines,
      uniforms: {
        uState: { value: null },
        uTexel: { value: simTexel },
        uSimSize: { value: simSize },
        uAspect: { value: 1 },
        uLight: { value: new THREE.Vector3(0, 0, 0.6) },
        uBump: { value: BUMP },
        uOrigins: { value: origins },
        uWobble: { value: wobble },
        uFrontRadius: { value: 0 },
        uFrontGlow: { value: 0 },
        uDeepOld: { value: paletteOld.deep },
        uMidOld: { value: paletteOld.mid },
        uPeakOld: { value: paletteOld.peak },
        uSpecOld: { value: paletteOld.spec },
        uDeepNew: { value: paletteNew.deep },
        uMidNew: { value: paletteNew.mid },
        uPeakNew: { value: paletteNew.peak },
        uSpecNew: { value: paletteNew.spec },
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

    // count seeds + a share of erasers
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

    // Phase state: the cycle's plan, which phase is running, and sim-seconds into it
    let cycle: Phase[] = []
    let phaseIndex = 0
    let phaseT = 0

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
      }
      // The front's reach is measured in texels, so re-measure it for the new grid
      if (cycle[phaseIndex]) measureFront(cycle[phaseIndex], simW, simH)
    }
    handleResize()

    const enterPhase = (index: number) => {
      if (index >= cycle.length) {
        const last = cycle[cycle.length - 1]
        cycle = planCycle(Math.random, PALETTES.length, last?.to ?? -1, last?.palette ?? -1)
        index = 0
      }
      phaseIndex = index
      phaseT = 0
      const phase = cycle[index]
      startPhase(phase, simSize.x, simSize.y)

      ;(simMaterial.uniforms.uFeedKillOld.value as THREE.Vector2).set(REGIMES[phase.from].feed, REGIMES[phase.from].kill)
      ;(simMaterial.uniforms.uFeedKillNew.value as THREE.Vector2).set(REGIMES[phase.to].feed, REGIMES[phase.to].kill)
      origins.forEach((v, i) => v.fromArray(phase.origins, i * 4))
      wobble.forEach((v, i) => v.fromArray(phase.wobble, i * 4))
      front.set(frontRadius(phase, 0), phase.bleach, phase.seedDensity, phase.seedSalt)

      // What was behind the last front is now ahead of this one
      copyPalette(paletteOld, paletteNew)
      setPalette(paletteNew, PALETTES[phase.palette])
      if (phase.kind === 'spread') {
        // A fresh board (the reset just erased it): drops grow across the screen
        copyPalette(paletteOld, paletteNew)
        scatterDrops(INITIAL_DROPS, 0)
        applySeed(true)
      }
    }
    enterPhase(0)

    const startTime = performance.now()
    let lastTime = startTime
    let lastProbe = startTime
    let probePending = false
    let disposed = false
    let stepDebt = 0
    let animationId = 0

    // Now and then, read back a tiny map of B; if the field has died out or filled solid,
    // re-seed or punch holes. Off while the reset deliberately erases the board and while a
    // fresh spread is still small.
    const guardActive = () => {
      const phase = cycle[phaseIndex]
      return phase.kind === 'transform' || (phase.kind === 'spread' && phaseT > GUARD_GRACE_SECONDS)
    }
    const probeCoverage = () => {
      if (probePending || !read || !guardActive()) return
      probeMaterial.uniforms.uState.value = read.texture
      runPass(probeMaterial, probeTarget)
      probePending = true
      renderer
        .readRenderTargetPixelsAsync(probeTarget, 0, 0, PROBE_SIZE, PROBE_SIZE, probePixels)
        .then(() => {
          if (disposed || !guardActive()) return
          const share = coverage(probePixels)
          if (share < DEAD_COVERAGE) {
            scatterDrops(INITIAL_DROPS, 0)
            applySeed(false)
          } else if (share > SOLID_COVERAGE) {
            scatterDrops(RESCUE_DROPS, 1)
            applySeed(false)
          }
        })
        .catch(() => {
          // Readback unsupported or context lost: the phases still move things along
        })
        .finally(() => {
          probePending = false
        })
    }

    const animate = () => {
      animationId = requestAnimationFrame(animate)
      const now = performance.now()
      // Clamp so a backgrounded tab doesn't come back owing thousands of steps
      const frameSeconds = Math.min((now - lastTime) / 1000, 1 / 20)
      lastTime = now
      const t = (now - startTime) / 1000

      stepDebt += frameSeconds * STEPS_PER_SECOND
      const steps = Math.min(Math.floor(stepDebt), MAX_STEPS_PER_FRAME)
      stepDebt = Math.min(stepDebt - steps, 1)

      // Phases run on sim time, so a slow GPU sees the same front speed relative to the pattern
      phaseT += steps / STEPS_PER_SECOND
      if (phaseT >= phaseSeconds(cycle[phaseIndex])) enterPhase(phaseIndex + 1)
      const phase = cycle[phaseIndex]
      front.x = frontRadius(phase, phaseT)
      displayMaterial.uniforms.uFrontRadius.value = front.x
      displayMaterial.uniforms.uFrontGlow.value = phase.kind === 'spread' ? 0 : Math.min(phaseT / GLOW_FADE_SECONDS, 1)

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
