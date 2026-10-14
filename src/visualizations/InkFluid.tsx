import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'

// GPU stable fluids (Stam, after Pavel Dobryakov's WebGL fluid): every frame the velocity
// field is advected, vorticity is re-injected, and a pressure solve makes it divergence-free;
// the dye field rides along on that velocity. Invisible jets wandering along smooth paths
// stir the water and bleed coloured ink into it.

// Configuration constants
const SIM_RESOLUTION = 256 // velocity/pressure grid, short side
const DYE_RESOLUTION = 1024 // ink texture, short side
const PRESSURE_ITERATIONS = 20
const PRESSURE_DECAY = 0.8 // carry over last frame's pressure as a warm start for Jacobi
const VELOCITY_DISSIPATION = 0.25 // per second
const MAX_JETS = 6 // must match the array size in the splat shader
const CYCLE_SECONDS = 45 // new palette + jet behaviour this often
const BLEND_SECONDS = 3 // time constant for easing into a new cycle's parameters
const PALETTE_DRIFT = 0.012 // palette cycles per second each jet slides along
const JET_MARGIN = 0.1 // keep nozzles this far (in uv) from the edges
const EXPOSURE = 1.6 // dye -> display tone-map gain
const MAX_FRAME_SECONDS = 1 / 30 // clamp so a background tab doesn't take one giant step

// Curated palettes (linear-ish RGB, roughly unit intensity). A cycle picks one and may rotate it.
const PALETTES: [number, number, number][][] = [
  [[0.1, 0.25, 1.0], [0.55, 0.1, 0.9], [1.0, 0.6, 0.1], [0.05, 0.7, 0.9]], // indigo & gold
  [[1.0, 0.35, 0.3], [0.0, 0.75, 0.7], [1.0, 0.8, 0.25], [0.9, 0.15, 0.45]], // coral reef
  [[0.1, 0.9, 0.45], [0.0, 0.6, 1.0], [0.85, 0.15, 0.85], [0.3, 1.0, 0.8]], // aurora
  [[0.9, 0.08, 0.06], [1.0, 0.45, 0.05], [1.0, 0.8, 0.3], [0.6, 0.02, 0.2]], // ember
  [[0.0, 0.35, 0.9], [0.0, 0.85, 0.85], [0.6, 0.85, 1.0], [0.1, 0.15, 0.6]], // deep ocean
  [[0.95, 0.15, 0.6], [0.5, 0.1, 1.0], [1.0, 0.55, 0.75], [0.25, 0.3, 1.0]], // orchid
  [[0.95, 0.75, 0.4], [0.2, 0.55, 0.5], [0.8, 0.25, 0.15], [0.9, 0.9, 0.75]], // sumi & saffron
]

interface Behavior {
  count: number // active jets
  speed: number // jet velocity, sim texels per second
  velRadius: number // gaussian width (screen heights^2) of the velocity nozzle
  dyeRadius: number
  curl: number // vorticity confinement strength
  dyeDissipation: number // per second
  puffSeconds: number // period of each jet's on/off breathing
  duty: number // fraction of the period a jet is on
  wander: number // how fast nozzles drift along their paths
  aim: number // how far (radians) jets swing away from aiming at the centre
}

interface Jet {
  phase: Float64Array // path phase accumulators: x1, x2, y1, y2, aim, puff
  freq: Float64Array // matching angular frequencies (rad/s at wander = 1)
  weight: number // eases to 1 when the jet is part of the cycle, 0 when not
  color: THREE.Vector3 // eased toward the palette target
  slot: number // where along the palette this jet sits
}

const randomBehavior = (): Behavior => ({
  count: 2 + Math.floor(Math.random() * (MAX_JETS - 1)),
  speed: 140 + Math.random() * 200,
  velRadius: 0.0008 + Math.random() * 0.0012,
  dyeRadius: 0.0004 + Math.random() * 0.0008,
  curl: 8 + Math.random() * 30,
  dyeDissipation: 0.07 + Math.random() * 0.1,
  puffSeconds: 3 + Math.random() * 7,
  duty: 0.3 + Math.random() * 0.5,
  wander: 0.5 + Math.random() * 1.2,
  aim: 0.4 + Math.random() * 1.2,
})

// Pick a palette, optionally rotate its hues, and shuffle so jets get fresh colours
const randomPalette = (): THREE.Vector3[] => {
  const base = PALETTES[Math.floor(Math.random() * PALETTES.length)]
  const rotate = Math.random() < 0.4 ? (Math.random() - 0.5) * 0.25 : 0
  const c = new THREE.Color()
  const hsl = { h: 0, s: 0, l: 0 }
  const colors = base.map(([r, g, b]) => {
    c.setRGB(r, g, b)
    if (rotate !== 0) {
      c.getHSL(hsl)
      c.setHSL((hsl.h + rotate + 1) % 1, hsl.s, hsl.l)
    }
    return new THREE.Vector3(c.r, c.g, c.b)
  })
  for (let i = colors.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[colors[i], colors[j]] = [colors[j], colors[i]]
  }
  return colors
}

// Smooth cyclic interpolation through the palette
const samplePalette = (palette: THREE.Vector3[], t: number, out: THREE.Vector3) => {
  const n = palette.length
  const x = (((t % 1) + 1) % 1) * n
  const i = Math.floor(x)
  const f = x - i
  const s = f * f * (3 - 2 * f)
  return out.copy(palette[i % n]).lerp(palette[(i + 1) % n], s)
}

const smoothstep = (a: number, b: number, x: number) => {
  const t = Math.min(Math.max((x - a) / (b - a), 0), 1)
  return t * t * (3 - 2 * t)
}

// ---------------------------------------------------------------------------------------------
// Shaders (THREE.ShaderMaterial, so three prepends precision / gl_FragColor / texture2D shims)

const BASE_VERT = /* glsl */ `
uniform vec2 texelSize;
varying vec2 vUv;
varying vec2 vL;
varying vec2 vR;
varying vec2 vT;
varying vec2 vB;
void main() {
  vUv = uv;
  vL = uv - vec2(texelSize.x, 0.0);
  vR = uv + vec2(texelSize.x, 0.0);
  vT = uv + vec2(0.0, texelSize.y);
  vB = uv - vec2(0.0, texelSize.y);
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`

const COPY_FRAG = /* glsl */ `
uniform sampler2D uTexture;
uniform float value;
varying vec2 vUv;
void main() {
  gl_FragColor = value * texture2D(uTexture, vUv);
}
`

// Each jet pulls the field toward its own value inside a gaussian nozzle. Pulling (rather than
// adding) caps the dye at the palette colour and the velocity at the jet speed, so a jet that
// sits still for a while can never blow the field out.
const SPLAT_FRAG = /* glsl */ `
uniform sampler2D uTarget;
uniform float aspectRatio;
uniform vec2 uPoints[6];
uniform vec3 uValues[6];
uniform float uAmounts[6];
uniform float uRadius;
varying vec2 vUv;
void main() {
  vec4 base = texture2D(uTarget, vUv);
  vec3 result = base.rgb;
  for (int i = 0; i < 6; i++) {
    vec2 p = vUv - uPoints[i];
    p.x *= aspectRatio;
    float g = exp(-dot(p, p) / uRadius) * uAmounts[i];
    result = mix(result, uValues[i], clamp(g, 0.0, 1.0));
  }
  gl_FragColor = vec4(result, 1.0);
}
`

// Semi-Lagrangian advection; velocity is in sim texels per second
const ADVECTION_FRAG = /* glsl */ `
uniform sampler2D uVelocity;
uniform sampler2D uSource;
uniform vec2 velocityTexelSize;
uniform float dt;
uniform float dissipation;
varying vec2 vUv;
void main() {
  vec2 coord = vUv - dt * texture2D(uVelocity, vUv).xy * velocityTexelSize;
  vec4 result = texture2D(uSource, coord);
  gl_FragColor = result / (1.0 + dissipation * dt);
}
`

const DIVERGENCE_FRAG = /* glsl */ `
uniform sampler2D uVelocity;
varying vec2 vUv;
varying vec2 vL;
varying vec2 vR;
varying vec2 vT;
varying vec2 vB;
void main() {
  float L = texture2D(uVelocity, vL).x;
  float R = texture2D(uVelocity, vR).x;
  float T = texture2D(uVelocity, vT).y;
  float B = texture2D(uVelocity, vB).y;
  vec2 C = texture2D(uVelocity, vUv).xy;
  // Reflect at the walls so ink doesn't leak out of the edges
  if (vL.x < 0.0) { L = -C.x; }
  if (vR.x > 1.0) { R = -C.x; }
  if (vT.y > 1.0) { T = -C.y; }
  if (vB.y < 0.0) { B = -C.y; }
  gl_FragColor = vec4(0.5 * (R - L + T - B), 0.0, 0.0, 1.0);
}
`

const CURL_FRAG = /* glsl */ `
uniform sampler2D uVelocity;
varying vec2 vUv;
varying vec2 vL;
varying vec2 vR;
varying vec2 vT;
varying vec2 vB;
void main() {
  float L = texture2D(uVelocity, vL).y;
  float R = texture2D(uVelocity, vR).y;
  float T = texture2D(uVelocity, vT).x;
  float B = texture2D(uVelocity, vB).x;
  gl_FragColor = vec4(0.5 * (R - L - T + B), 0.0, 0.0, 1.0);
}
`

// Vorticity confinement: push velocity along the curl gradient to restore the small swirls
// numerical diffusion keeps smearing away. This is what makes the ink curl.
const VORTICITY_FRAG = /* glsl */ `
uniform sampler2D uVelocity;
uniform sampler2D uCurl;
uniform float curl;
uniform float dt;
varying vec2 vUv;
varying vec2 vL;
varying vec2 vR;
varying vec2 vT;
varying vec2 vB;
void main() {
  float L = texture2D(uCurl, vL).x;
  float R = texture2D(uCurl, vR).x;
  float T = texture2D(uCurl, vT).x;
  float B = texture2D(uCurl, vB).x;
  float C = texture2D(uCurl, vUv).x;
  vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
  force /= length(force) + 0.0001;
  force *= curl * C;
  force.y *= -1.0;
  vec2 velocity = texture2D(uVelocity, vUv).xy + force * dt;
  velocity = clamp(velocity, vec2(-1000.0), vec2(1000.0));
  gl_FragColor = vec4(velocity, 0.0, 1.0);
}
`

const PRESSURE_FRAG = /* glsl */ `
uniform sampler2D uPressure;
uniform sampler2D uDivergence;
varying vec2 vUv;
varying vec2 vL;
varying vec2 vR;
varying vec2 vT;
varying vec2 vB;
void main() {
  float L = texture2D(uPressure, vL).x;
  float R = texture2D(uPressure, vR).x;
  float T = texture2D(uPressure, vT).x;
  float B = texture2D(uPressure, vB).x;
  float divergence = texture2D(uDivergence, vUv).x;
  gl_FragColor = vec4((L + R + B + T - divergence) * 0.25, 0.0, 0.0, 1.0);
}
`

const GRADIENT_SUBTRACT_FRAG = /* glsl */ `
uniform sampler2D uPressure;
uniform sampler2D uVelocity;
varying vec2 vUv;
varying vec2 vL;
varying vec2 vR;
varying vec2 vT;
varying vec2 vB;
void main() {
  float L = texture2D(uPressure, vL).x;
  float R = texture2D(uPressure, vR).x;
  float T = texture2D(uPressure, vT).x;
  float B = texture2D(uPressure, vB).x;
  vec2 velocity = texture2D(uVelocity, vUv).xy - vec2(R - L, T - B);
  gl_FragColor = vec4(velocity, 0.0, 1.0);
}
`

// Treat dye density as a height field: its gradient gives a normal, so thick ink gets soft
// relief shading and a faint sheen along its folds. Then tone-map so overlapping plumes
// saturate gracefully instead of clipping to white.
const DISPLAY_FRAG = /* glsl */ `
uniform sampler2D uDye;
uniform vec2 texelSize;
uniform float exposure;
uniform vec3 background;
varying vec2 vUv;
varying vec2 vL;
varying vec2 vR;
varying vec2 vT;
varying vec2 vB;
void main() {
  vec3 c = texture2D(uDye, vUv).rgb;
  float lc = length(texture2D(uDye, vL).rgb);
  float rc = length(texture2D(uDye, vR).rgb);
  float tc = length(texture2D(uDye, vT).rgb);
  float bc = length(texture2D(uDye, vB).rgb);
  vec3 n = normalize(vec3(rc - lc, tc - bc, length(texelSize) * 6.0));
  float diffuse = clamp(dot(n, vec3(0.0, 0.0, 1.0)) + 0.7, 0.7, 1.0);
  vec3 lightDir = normalize(vec3(-0.35, 0.45, 1.0));
  float spec = pow(clamp(dot(reflect(-lightDir, n), vec3(0.0, 0.0, 1.0)), 0.0, 1.0), 28.0);

  vec3 ink = 1.0 - exp(-c * exposure);
  float density = clamp(length(ink), 0.0, 1.0);
  ink *= diffuse;
  ink += spec * 0.18 * density * (0.6 + 0.4 * normalize(ink + 0.001));

  vec2 d = vUv - 0.5;
  float vignette = 1.0 - 0.45 * dot(d, d) * 2.0;
  gl_FragColor = vec4((background + ink) * vignette, 1.0);
}
`

// ---------------------------------------------------------------------------------------------

interface DoubleTarget {
  read: THREE.WebGLRenderTarget
  write: THREE.WebGLRenderTarget
  swap: () => void
}

export default function InkFluid() {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!containerRef.current) return
    const container = containerRef.current

    const renderer = new THREE.WebGLRenderer({ antialias: false })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(window.innerWidth, window.innerHeight)
    container.appendChild(renderer.domElement)

    // Half floats are needed for signed velocity/pressure. WebGL2 renders to them with either
    // extension; without one we fall back to bytes (degraded, but it still draws something).
    const halfFloatOk =
      renderer.extensions.has('EXT_color_buffer_float') ||
      renderer.extensions.has('EXT_color_buffer_half_float')
    if (!halfFloatOk) console.warn('InkFluid: half-float render targets unsupported; quality degraded')
    const targetType = halfFloatOk ? THREE.HalfFloatType : THREE.UnsignedByteType

    // One full-screen quad; each pass swaps its material and renders into a target
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x000000)
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1)

    const makeMaterial = (fragmentShader: string, uniforms: Record<string, THREE.IUniform>) =>
      new THREE.ShaderMaterial({
        vertexShader: BASE_VERT,
        fragmentShader,
        uniforms: { texelSize: { value: new THREE.Vector2() }, ...uniforms },
        depthTest: false,
        depthWrite: false,
      })

    const copyMat = makeMaterial(COPY_FRAG, { uTexture: { value: null }, value: { value: 1 } })
    const splatMat = makeMaterial(SPLAT_FRAG, {
      uTarget: { value: null },
      aspectRatio: { value: 1 },
      uPoints: { value: Array.from({ length: MAX_JETS }, () => new THREE.Vector2()) },
      uValues: { value: Array.from({ length: MAX_JETS }, () => new THREE.Vector3()) },
      uAmounts: { value: new Array<number>(MAX_JETS).fill(0) },
      uRadius: { value: 0.001 },
    })
    const advectionMat = makeMaterial(ADVECTION_FRAG, {
      uVelocity: { value: null },
      uSource: { value: null },
      velocityTexelSize: { value: new THREE.Vector2() },
      dt: { value: 0 },
      dissipation: { value: 0 },
    })
    const divergenceMat = makeMaterial(DIVERGENCE_FRAG, { uVelocity: { value: null } })
    const curlMat = makeMaterial(CURL_FRAG, { uVelocity: { value: null } })
    const vorticityMat = makeMaterial(VORTICITY_FRAG, {
      uVelocity: { value: null },
      uCurl: { value: null },
      curl: { value: 0 },
      dt: { value: 0 },
    })
    const pressureMat = makeMaterial(PRESSURE_FRAG, {
      uPressure: { value: null },
      uDivergence: { value: null },
    })
    const gradientMat = makeMaterial(GRADIENT_SUBTRACT_FRAG, {
      uPressure: { value: null },
      uVelocity: { value: null },
    })
    const displayMat = makeMaterial(DISPLAY_FRAG, {
      uDye: { value: null },
      exposure: { value: EXPOSURE },
      background: { value: new THREE.Vector3(0.004, 0.006, 0.012) },
    })
    const materials = [
      copyMat, splatMat, advectionMat, divergenceMat, curlMat,
      vorticityMat, pressureMat, gradientMat, displayMat,
    ]

    const quadGeometry = new THREE.PlaneGeometry(2, 2)
    const quad = new THREE.Mesh(quadGeometry, displayMat)
    quad.frustumCulled = false
    scene.add(quad)

    // --- Render targets ---------------------------------------------------------------------
    const createTarget = (w: number, h: number) =>
      new THREE.WebGLRenderTarget(w, h, {
        type: targetType,
        format: THREE.RGBAFormat,
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        wrapS: THREE.ClampToEdgeWrapping,
        wrapT: THREE.ClampToEdgeWrapping,
        depthBuffer: false,
        stencilBuffer: false,
      })
    const createDouble = (w: number, h: number): DoubleTarget => {
      const d: DoubleTarget = {
        read: createTarget(w, h),
        write: createTarget(w, h),
        swap: () => {
          const t = d.read
          d.read = d.write
          d.write = t
        },
      }
      return d
    }

    const blit = (material: THREE.ShaderMaterial, target: THREE.WebGLRenderTarget | null) => {
      quad.material = material
      renderer.setRenderTarget(target)
      renderer.render(scene, camera)
    }

    // Short side gets the nominal resolution; long side scales with the aspect
    const maxTextureSize = renderer.capabilities.maxTextureSize
    const resolutionFor = (base: number) => {
      // Guard a 0-sized window (aspect 0 / Infinity) and extreme aspects past the GPU limit
      const aspect = Math.max(window.innerWidth, 1) / Math.max(window.innerHeight, 1)
      const lo = base
      const hi = Math.min(Math.round(base * Math.max(aspect, 1 / aspect)), maxTextureSize)
      return aspect >= 1 ? { w: hi, h: lo } : { w: lo, h: hi }
    }

    let simSize = resolutionFor(SIM_RESOLUTION)
    let dyeSize = resolutionFor(DYE_RESOLUTION)
    let velocity = createDouble(simSize.w, simSize.h)
    let pressure = createDouble(simSize.w, simSize.h)
    let divergence = createTarget(simSize.w, simSize.h)
    let curlTarget = createTarget(simSize.w, simSize.h)
    let dye = createDouble(dyeSize.w, dyeSize.h)
    const simTexel = new THREE.Vector2(1 / simSize.w, 1 / simSize.h)
    const dyeTexel = new THREE.Vector2(1 / dyeSize.w, 1 / dyeSize.h)

    // Resample a field into a new-size target so a resize keeps the ink on screen
    const resizeDouble = (old: DoubleTarget, w: number, h: number) => {
      const next = createDouble(w, h)
      copyMat.uniforms.uTexture.value = old.read.texture
      copyMat.uniforms.value.value = 1
      blit(copyMat, next.read)
      old.read.dispose()
      old.write.dispose()
      return next
    }

    const composer = new EffectComposer(renderer)
    composer.addPass(new RenderPass(scene, camera))
    const bloomPass = new UnrealBloomPass(
      new THREE.Vector2(window.innerWidth, window.innerHeight),
      0.55, // strength
      0.7, // radius
      0.45, // threshold: only the dense, bright cores of plumes glow
    )
    composer.addPass(bloomPass)

    const handleResize = () => {
      renderer.setSize(window.innerWidth, window.innerHeight)
      composer.setSize(window.innerWidth, window.innerHeight)
      const nextSim = resolutionFor(SIM_RESOLUTION)
      const nextDye = resolutionFor(DYE_RESOLUTION)
      if (nextSim.w !== simSize.w || nextSim.h !== simSize.h) {
        simSize = nextSim
        velocity = resizeDouble(velocity, simSize.w, simSize.h)
        pressure = resizeDouble(pressure, simSize.w, simSize.h)
        divergence.dispose()
        curlTarget.dispose()
        divergence = createTarget(simSize.w, simSize.h)
        curlTarget = createTarget(simSize.w, simSize.h)
        simTexel.set(1 / simSize.w, 1 / simSize.h)
      }
      if (nextDye.w !== dyeSize.w || nextDye.h !== dyeSize.h) {
        dyeSize = nextDye
        dye = resizeDouble(dye, dyeSize.w, dyeSize.h)
        dyeTexel.set(1 / dyeSize.w, 1 / dyeSize.h)
      }
    }

    // --- Jets & cycles ------------------------------------------------------------------------
    const jets: Jet[] = Array.from({ length: MAX_JETS }, (_, i) => ({
      phase: Float64Array.from({ length: 6 }, () => Math.random() * Math.PI * 2),
      freq: Float64Array.from([
        0.05 + Math.random() * 0.08,
        0.11 + Math.random() * 0.12,
        0.05 + Math.random() * 0.08,
        0.11 + Math.random() * 0.12,
        0.15 + Math.random() * 0.2,
        1, // puff phase advances at 2π / puffSeconds, handled separately
      ]),
      weight: 0,
      color: new THREE.Vector3(),
      slot: i / MAX_JETS,
    }))

    let target = randomBehavior()
    const current: Behavior = { ...target }
    let palette = randomPalette()
    for (const jet of jets) samplePalette(palette, jet.slot, jet.color)

    const newCycle = () => {
      target = randomBehavior()
      palette = randomPalette()
      // Reshuffle which nozzles are active and where each sits on the palette
      for (let i = jets.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1))
        ;[jets[i], jets[j]] = [jets[j], jets[i]]
      }
    }

    const jetPos = new THREE.Vector2()
    const jetVelocities = Array.from({ length: MAX_JETS }, () => new THREE.Vector3())
    const envelopes = new Float64Array(MAX_JETS)
    const paletteColor = new THREE.Vector3()
    const startTime = performance.now()
    let lastTime = startTime
    let cycleStart = startTime
    let animationId = 0

    const step = (dt: number, t: number) => {
      const ease = 1 - Math.exp(-dt / BLEND_SECONDS)
      for (const key of Object.keys(target) as (keyof Behavior)[]) {
        current[key] += (target[key] - current[key]) * ease
      }

      const aspect = simSize.w / simSize.h
      const points = splatMat.uniforms.uPoints.value as THREE.Vector2[]
      const values = splatMat.uniforms.uValues.value as THREE.Vector3[]
      const amounts = splatMat.uniforms.uAmounts.value as number[]

      jets.forEach((jet, i) => {
        const p = jet.phase
        for (let k = 0; k < 5; k++) p[k] += jet.freq[k] * current.wander * dt
        p[5] += (dt / current.puffSeconds) * Math.PI * 2

        // Two incommensurate sines per axis give a smooth path that never quite repeats
        const span = 0.5 - JET_MARGIN
        jetPos.set(
          0.5 + span * (0.6 * Math.sin(p[0]) + 0.4 * Math.sin(p[1])),
          0.5 + span * (0.6 * Math.sin(p[2]) + 0.4 * Math.sin(p[3])),
        )
        points[i].copy(jetPos)

        // Aim roughly at the centre so plumes stay on screen, swinging off to either side
        const toCentre = Math.atan2(0.5 - jetPos.y, (0.5 - jetPos.x) * aspect)
        const angle = toCentre + current.aim * Math.sin(p[4])

        jet.weight += ((i < Math.round(current.count) ? 1 : 0) - jet.weight) * ease
        const breathe = 0.5 + 0.5 * Math.sin(p[5])
        const envelope = jet.weight * smoothstep(1 - current.duty - 0.15, 1 - current.duty + 0.15, breathe)

        jetVelocities[i].set(Math.cos(angle) * current.speed, Math.sin(angle) * current.speed, 0)
        envelopes[i] = envelope

        samplePalette(palette, jet.slot + t * PALETTE_DRIFT, paletteColor)
        jet.color.lerp(paletteColor, ease)
      })

      // 1. Inject velocity
      splatMat.uniforms.aspectRatio.value = aspect
      splatMat.uniforms.uRadius.value = current.velRadius
      splatMat.uniforms.texelSize.value.copy(simTexel)
      splatMat.uniforms.uTarget.value = velocity.read.texture
      for (let i = 0; i < MAX_JETS; i++) {
        values[i].copy(jetVelocities[i])
        amounts[i] = Math.min(1, 10 * dt) * envelopes[i]
      }
      blit(splatMat, velocity.write)
      velocity.swap()

      // 2. Inject dye (pull toward the jet's colour, scaled by how open the nozzle is)
      splatMat.uniforms.uRadius.value = current.dyeRadius
      splatMat.uniforms.texelSize.value.copy(dyeTexel)
      splatMat.uniforms.uTarget.value = dye.read.texture
      for (let i = 0; i < MAX_JETS; i++) {
        values[i].copy(jets[i].color)
        amounts[i] = Math.min(1, 6 * dt) * envelopes[i]
      }
      blit(splatMat, dye.write)
      dye.swap()

      // 3. Curl + vorticity confinement
      curlMat.uniforms.texelSize.value.copy(simTexel)
      curlMat.uniforms.uVelocity.value = velocity.read.texture
      blit(curlMat, curlTarget)

      vorticityMat.uniforms.texelSize.value.copy(simTexel)
      vorticityMat.uniforms.uVelocity.value = velocity.read.texture
      vorticityMat.uniforms.uCurl.value = curlTarget.texture
      vorticityMat.uniforms.curl.value = current.curl
      vorticityMat.uniforms.dt.value = dt
      blit(vorticityMat, velocity.write)
      velocity.swap()

      // 4. Divergence, then Jacobi pressure solve
      divergenceMat.uniforms.texelSize.value.copy(simTexel)
      divergenceMat.uniforms.uVelocity.value = velocity.read.texture
      blit(divergenceMat, divergence)

      copyMat.uniforms.texelSize.value.copy(simTexel)
      copyMat.uniforms.uTexture.value = pressure.read.texture
      copyMat.uniforms.value.value = PRESSURE_DECAY
      blit(copyMat, pressure.write)
      pressure.swap()

      pressureMat.uniforms.texelSize.value.copy(simTexel)
      pressureMat.uniforms.uDivergence.value = divergence.texture
      for (let i = 0; i < PRESSURE_ITERATIONS; i++) {
        pressureMat.uniforms.uPressure.value = pressure.read.texture
        blit(pressureMat, pressure.write)
        pressure.swap()
      }

      // 5. Make velocity divergence-free
      gradientMat.uniforms.texelSize.value.copy(simTexel)
      gradientMat.uniforms.uPressure.value = pressure.read.texture
      gradientMat.uniforms.uVelocity.value = velocity.read.texture
      blit(gradientMat, velocity.write)
      velocity.swap()

      // 6. Advect velocity through itself, then the dye through the velocity
      advectionMat.uniforms.texelSize.value.copy(simTexel)
      advectionMat.uniforms.velocityTexelSize.value.copy(simTexel)
      advectionMat.uniforms.dt.value = dt
      advectionMat.uniforms.uVelocity.value = velocity.read.texture
      advectionMat.uniforms.uSource.value = velocity.read.texture
      advectionMat.uniforms.dissipation.value = VELOCITY_DISSIPATION
      blit(advectionMat, velocity.write)
      velocity.swap()

      advectionMat.uniforms.texelSize.value.copy(dyeTexel)
      advectionMat.uniforms.uVelocity.value = velocity.read.texture
      advectionMat.uniforms.uSource.value = dye.read.texture
      advectionMat.uniforms.dissipation.value = current.dyeDissipation
      blit(advectionMat, dye.write)
      dye.swap()
    }

    const animate = () => {
      animationId = requestAnimationFrame(animate)
      const now = performance.now()
      const dt = Math.min((now - lastTime) / 1000, MAX_FRAME_SECONDS)
      lastTime = now

      if ((now - cycleStart) / 1000 >= CYCLE_SECONDS) {
        cycleStart = now
        newCycle()
      }

      step(dt, (now - startTime) / 1000)

      displayMat.uniforms.texelSize.value.copy(dyeTexel)
      displayMat.uniforms.uDye.value = dye.read.texture
      quad.material = displayMat
      renderer.setRenderTarget(null)
      composer.render()
    }

    animate()
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      cancelAnimationFrame(animationId)
      for (const d of [velocity, pressure, dye]) {
        d.read.dispose()
        d.write.dispose()
      }
      divergence.dispose()
      curlTarget.dispose()
      for (const m of materials) m.dispose()
      quadGeometry.dispose()
      bloomPass.dispose()
      composer.dispose()
      renderer.dispose()
      container.removeChild(renderer.domElement)
    }
  }, [])

  return <div ref={containerRef} className="w-full h-full" />
}
