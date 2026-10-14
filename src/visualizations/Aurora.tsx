import { useEffect, useRef } from 'react'
import * as THREE from 'three'

// Configuration constants
const RENDER_SCALE = 0.5 // the scene shader runs at this fraction of device resolution...
const MAX_TARGET_HEIGHT = 720 // ...and never taller than this, so 4K screens stay cheap
const MIN_CYCLE_SECONDS = 50 // one calm -> surge -> calm cycle; new palette/shape each cycle
const MAX_CYCLE_SECONDS = 70
const TRANSITION_SECONDS = 12 // parameter crossfade, done during the calm opening of a cycle
const CALM_ACTIVITY = 0.12
const FLOW_SPEED = 0.55 // curtain evolution rate (noise units per second at activity 0)
const RAY_SPEED = 0.18 // ray drift rate along the curtain
const FLOW_WRAP = 20000 // reset accumulated phases (at a calm moment) before fp32 precision degrades
const SKY_ROTATION = 0.0006 // radians per second of the star field around an off-screen pole
const TWINKLE_WRAP = 4096 // seconds; star twinkle phases are random, so wrapping is invisible

// Aurora palettes: [lower edge, body, upper tips]. Real aurora are oxygen green low down with
// red/violet (high oxygen, nitrogen) up top; the stranger ones are for variety.
const PALETTES: [number, number, number][][] = [
  [[0.15, 1.0, 0.45], [0.1, 0.75, 0.8], [0.55, 0.2, 0.95]],
  [[0.2, 1.0, 0.35], [0.25, 0.9, 0.65], [0.95, 0.2, 0.65]],
  [[0.1, 0.95, 0.7], [0.12, 0.6, 1.0], [0.45, 0.25, 1.0]],
  [[0.3, 1.0, 0.4], [0.45, 0.85, 0.45], [0.95, 0.15, 0.3]],
  [[0.55, 1.0, 0.3], [0.2, 0.9, 0.6], [1.0, 0.3, 0.8]],
  [[0.1, 0.9, 0.55], [0.35, 0.5, 1.0], [0.85, 0.25, 0.9]],
]

// Shaders. Everything is in "horizon coordinates": x across the screen (aspect-corrected,
// 0 at center), y = height above the lake's horizon line, in screen heights.
const VERT = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`

const COMMON_GLSL = /* glsl */ `
uniform float uAspect;
uniform float uPixel;
uniform float uTime;
uniform float uSkyAngle;
uniform vec3 uWave;
varying vec2 vUv;

const vec2 POLE = vec2(-0.35, 1.25);

float hash11(float p) {
  p = fract(p * 0.1031);
  p *= p + 33.33;
  p *= p + p;
  return fract(p);
}

float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

float noise1(float x) {
  float i = floor(x);
  float f = fract(x);
  return mix(hash11(i), hash11(i + 1.0), f * f * (3.0 - 2.0 * f));
}

float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash12(i), hash12(i + vec2(1.0, 0.0)), u.x),
             mix(hash12(i + vec2(0.0, 1.0)), hash12(i + vec2(1.0, 1.0)), u.x), u.y);
}

vec2 screenCoord() {
  return vec2((vUv.x - 0.5) * uAspect, vUv.y - 0.3);
}

// Where a lake pixel's reflected ray lands in the sky. Mirror about the horizon, then
// displace by perspective-compressed ripples that grow toward the viewer.
vec2 reflectCoord(vec2 p) {
  float below = max(-p.y, 0.0);
  float z = 1.0 / (below + 0.04);
  float wx = p.x * z;
  float wave = sin(z * 2.7 - uWave.x + 1.3 * sin(wx * 0.6 + uWave.y))
             + 0.5 * sin(z * 4.9 + wx * 0.35 - uWave.z);
  float wob = sin(wx * 1.7 + z * 0.9 - uWave.y);
  return vec2(p.x + wob * below * 0.015, max(below + wave * below * 0.05, 0.0));
}

// Sky-fixed coordinates: the celestial sphere turns slowly about an off-screen pole
vec2 skyCoord(vec2 q) {
  float c = cos(uSkyAngle);
  float s = sin(uSkyAngle);
  vec2 d = q - POLE;
  return POLE + vec2(c * d.x - s * d.y, s * d.x + c * d.y);
}
`

const SCENE_FRAG =
  COMMON_GLSL +
  /* glsl */ `
uniform float uFlow;
uniform float uRay;
uniform float uActivity;
uniform float uFold;
uniform float uBase;
uniform float uHeight;
uniform float uBandAngle;
uniform float uRidgeSeed;
uniform vec3 uColLow;
uniform vec3 uColMid;
uniform vec3 uColHigh;

float fbm(vec2 p) {
  float s = 0.0;
  float a = 0.5;
  for (int i = 0; i < 4; i++) {
    s += a * noise(p);
    p = p * 2.03 + vec2(17.1, 9.3);
    a *= 0.5;
  }
  return s;
}

// Horizontal domain warp of a curtain's path: this is what makes it fold and ripple
float warpAt(float x, float t, float layer) {
  return ((noise(vec2(x * 0.42, t * 0.06 + layer * 3.1)) - 0.5) * 2.0
        + (noise(vec2(x * 1.25 + 4.0, t * 0.11 + layer * 7.7)) - 0.5) * 0.6) * uFold;
}

// Five curtains at increasing distance. A curtain at depth D has its lower edge at
// altitude uBase / D on screen and spans uHeight / D, so far curtains sit low and small.
vec3 aurora(vec2 p, out float halo) {
  vec3 acc = vec3(0.0);
  halo = 0.0;
  for (int i = 0; i < 5; i++) {
    float fi = float(i);
    float depth = pow(1.55, fi);
    float t = uFlow + fi * 11.0;
    float x = p.x * depth + fi * 17.3;

    float w0 = warpAt(x, t, fi);
    float w1 = warpAt(x + 0.02, t, fi);
    float xw = x + w0;
    // Where the warp compresses the curtain we see it edge-on: more glowing column per pixel
    float stretch = abs(1.0 + (w1 - w0) / 0.02);
    float fold = min(1.15 / (stretch + 0.15), 3.5);

    float base = (uBase + 0.12 * (noise(vec2(xw * 0.7 + 9.0, t * 0.08)) - 0.5)) / depth;
    float h = uHeight * (0.6 + 0.8 * noise(vec2(xw * 0.45 + 2.0, t * 0.05))) / depth;
    float d = (p.y - base) / h; // 0 at the lower edge, 1 one curtain-height up

    float atten = 1.0 / (0.5 + 0.5 * depth);
    float patchy = smoothstep(0.2, 0.75, noise(vec2(xw * 0.35 - t * 0.05, fi * 5.0 + t * 0.02)));
    halo += exp(-abs(d - 0.3) * 1.3) * atten * patchy;
    if (d < -0.5 || d > 5.0) continue;

    // Sharp lower edge, long exponential fade upward
    float profile = exp(min(d, 0.0) * 16.0) * exp(-max(d, 0.0) * 1.7);

    // Vertical rays: high frequency along the curtain, slow variation up it. Fade each
    // octave out before it aliases (far and folded curtains pack rays densely).
    float px = depth * uPixel / max(stretch, 0.05);
    float r1 = noise(vec2(xw * 34.0 + uRay * 3.0, d * 0.5 - t * 0.12));
    float r2 = noise(vec2(xw * 83.0 - uRay * 5.0, d * 0.9 + fi * 3.0));
    r1 = mix(0.5, r1, smoothstep(0.5, 0.2, 34.0 * px));
    r2 = mix(0.5, r2, smoothstep(0.5, 0.2, 83.0 * px));
    float k = 1.5 + 3.0 * uActivity; // surges sharpen the rays
    float rays = 0.5 * pow(2.0 * (r1 * 0.6 + r2 * 0.4), k);
    float str = mix(1.0, rays, 0.65);

    // During surges bright pulses run along the curtains
    float pulse = 0.5 + 0.5 * sin(xw * 2.3 - t * 1.1 + fi * 1.7);
    float surge = mix(1.0, 0.35 + 1.3 * pulse * pulse, uActivity);
    float bright = profile * str * fold * patchy * surge * atten;

    vec3 col = mix(uColLow, uColMid, smoothstep(0.1, 0.9, d));
    col = mix(col, uColHigh, smoothstep(0.6, 2.4, d));
    // Strong storms show a magenta lower fringe
    float fringe = exp(-abs(d + 0.04) * 20.0) * uActivity * uActivity;
    col = mix(col, uColHigh * 1.3, clamp(fringe * 0.8, 0.0, 1.0));
    // Tight folds burn toward white
    col += vec3(0.14) * max(fold - 1.6, 0.0) * exp(-abs(d) * 3.0);
    acc += col * bright;
  }
  return acc;
}

float mountainH(float x) {
  float m = 0.0;
  float a = 0.5;
  float f = 1.1;
  for (int i = 0; i < 5; i++) {
    float n = 1.0 - abs(noise1(x * f + uRidgeSeed + float(i) * 13.0) * 2.0 - 1.0);
    m += a * n * n;
    a *= 0.5;
    f *= 2.1;
  }
  float envelope = 0.35 + 0.65 * noise1(x * 0.35 + uRidgeSeed * 0.5);
  return 0.012 + 0.12 * m * envelope;
}

// Near-shore conifers: one triangular tip per cell over a lumpy canopy, in stretches
float treeH(float x) {
  float g = x * 110.0;
  float c = floor(g);
  float tri = 1.0 - abs(fract(g) - 0.5) * 2.0;
  float tip = (0.004 + 0.02 * hash11(c + uRidgeSeed)) * tri;
  float canopy = 0.004 + 0.008 * noise1(x * 5.0 + uRidgeSeed);
  float stretchMask = smoothstep(0.35, 0.6, noise1(x * 0.9 + 40.0 + uRidgeSeed));
  return (canopy + tip) * stretchMask;
}

// Color of everything above the waterline at q, and how much of it is solid silhouette
vec3 skyScene(vec2 q, out float sil) {
  float aa = uPixel * 1.2;
  float m = mountainH(q.x);
  float tr = treeH(q.x);
  float mMask = smoothstep(m + aa, m - aa, q.y);
  float tMask = smoothstep(tr + aa, tr - aa, q.y);
  sil = max(mMask, tMask);

  float gain = 0.18 + 0.9 * uActivity;
  vec3 c = vec3(0.0);
  float halo = 0.0;
  if (sil < 0.999) {
    float e = q.y;
    c = mix(vec3(0.02, 0.042, 0.07), vec3(0.002, 0.004, 0.014), pow(smoothstep(0.0, 0.8, e), 0.55));

    // Faint milky band with dust lanes, fixed to the rotating sky
    vec2 s = skyCoord(q);
    vec2 n = vec2(-sin(uBandAngle), cos(uBandAngle));
    float dist = dot(s - vec2(0.1, 0.45), n) + 0.05 * sin(dot(s, n.yx) * 1.7);
    float band = exp(-dist * dist / 0.02);
    float cloud = fbm(s * 3.5 + 11.0);
    float dust = smoothstep(0.5, 0.72, fbm(s * 7.0 + 3.0)) * exp(-dist * dist / 0.005);
    float grain = noise(s * 60.0);
    c += vec3(0.04, 0.042, 0.055) * band * (0.3 + cloud) * (0.7 + 0.6 * grain) * (1.0 - 0.8 * dust);

    vec3 a = aurora(q, halo);
    float extinction = smoothstep(0.0, 0.05, q.y);
    c += a * gain * extinction;
    // Aurora light scattered into the air near the horizon
    c += mix(uColLow, uColMid, 0.4) * halo * gain * 0.018;
  } else {
    halo = 1.5;
  }

  // Far mountains: hazy at their base, darker toward the ridgeline, faintly aurora-lit
  vec3 mCol = mix(vec3(0.016, 0.026, 0.04), vec3(0.006, 0.009, 0.016), smoothstep(0.0, 0.05, q.y));
  mCol += uColLow * gain * 0.006 * min(halo, 2.0);
  c = mix(c, mCol, mMask);
  c = mix(c, vec3(0.002, 0.003, 0.005), tMask);
  return c;
}

void main() {
  vec2 p = screenCoord();
  vec3 col;
  float sil;
  float starVis;
  if (p.y >= 0.0) {
    col = skyScene(p, sil);
    starVis = 1.0 - sil;
  } else {
    // Still lake: reflection darkens away from grazing angles
    float below = -p.y;
    float fres = mix(0.75, 0.28, smoothstep(0.0, 0.35, below));
    col = skyScene(reflectCoord(p), sil) * fres * vec3(0.9, 0.97, 1.0) + vec3(0.001, 0.003, 0.006);
    starVis = (1.0 - sil) * fres * 0.8;
  }
  // Soft tone map so surges saturate gracefully instead of clipping
  col = 1.0 - exp(-col);
  // Alpha carries star visibility to the full-resolution composite pass
  gl_FragColor = vec4(col, starVis);
}
`

const COMPOSITE_FRAG =
  COMMON_GLSL +
  /* glsl */ `
uniform sampler2D uScene;

// One candidate star per grid cell, kept away from the cell edges so its glow isn't clipped
vec3 starLayer(vec2 s, float cells, float density, float seed, float gain, float twinkleAmp) {
  vec2 g = s * cells;
  vec2 id = floor(g);
  vec2 f = fract(g);
  if (hash12(id + seed) > density) return vec3(0.0);
  vec2 pos = 0.2 + 0.6 * vec2(hash12(id + seed + 3.7), hash12(id + seed + 9.1));
  vec2 d = (f - pos) / cells;
  float sz = uPixel * (0.55 + 0.6 * hash12(id + seed + 5.3));
  float mag = hash12(id + seed + 7.9);
  float b = gain * (0.12 + 1.6 * pow(mag, 7.0));
  float h = hash12(id + seed + 4.4);
  float tw = 1.0 - twinkleAmp * (0.5 + 0.5 * sin(uTime * (1.5 + 3.5 * h) + h * 40.0));
  vec3 tint = mix(vec3(1.0, 0.82, 0.66), vec3(0.72, 0.84, 1.0), hash12(id + seed + 2.2));
  return tint * b * tw * exp(-dot(d, d) / (sz * sz));
}

void main() {
  vec4 sc = texture2D(uScene, vUv);
  vec2 p = screenCoord();
  vec2 q = p.y >= 0.0 ? p : reflectCoord(p);
  vec2 s = skyCoord(q);
  // Low stars twinkle harder (more air) and fade into the horizon haze
  float twinkleAmp = mix(0.7, 0.3, smoothstep(0.0, 0.4, q.y));
  vec3 stars = starLayer(s, 110.0, 0.4, 1.0, 1.0, twinkleAmp)
             + starLayer(s, 38.0, 0.55, 7.0, 1.3, twinkleAmp);
  stars *= smoothstep(0.0, 0.12, q.y) * sc.a;
  vec3 col = sc.rgb + stars;
  // Dither: the night sky is all dark gradients, which band badly in 8 bits
  col += (hash12(gl_FragCoord.xy + fract(uTime * 7.31) * 91.0) - 0.5) / 255.0;
  gl_FragColor = vec4(col, 1.0);
}
`

interface AuroraParams {
  low: THREE.Color
  mid: THREE.Color
  high: THREE.Color
  fold: number // how hard the curtains fold back on themselves
  base: number // lower-edge altitude of the nearest curtain
  height: number // vertical extent of the nearest curtain
  drift: number // signed ray drift speed along the curtains
}

const randomBetween = (min: number, max: number) => min + Math.random() * (max - min)
const smoothstep = (a: number, b: number, x: number) => {
  const t = Math.min(Math.max((x - a) / (b - a), 0), 1)
  return t * t * (3 - 2 * t)
}

const jitteredColor = ([r, g, b]: [number, number, number]) =>
  new THREE.Color(
    Math.min(Math.max(r + randomBetween(-0.06, 0.06), 0), 1),
    Math.min(Math.max(g + randomBetween(-0.06, 0.06), 0), 1),
    Math.min(Math.max(b + randomBetween(-0.06, 0.06), 0), 1),
  )

const randomParams = (): AuroraParams => {
  const [low, mid, high] = PALETTES[Math.floor(Math.random() * PALETTES.length)]
  return {
    low: jitteredColor(low),
    mid: jitteredColor(mid),
    high: jitteredColor(high),
    fold: randomBetween(0.25, 0.75),
    base: randomBetween(0.26, 0.4),
    height: randomBetween(0.45, 0.8),
    drift: (Math.random() < 0.5 ? -1 : 1) * randomBetween(0.6, 1.4),
  }
}

const lerpParams = (from: AuroraParams, to: AuroraParams, t: number, out: AuroraParams) => {
  out.low.lerpColors(from.low, to.low, t)
  out.mid.lerpColors(from.mid, to.mid, t)
  out.high.lerpColors(from.high, to.high, t)
  out.fold = from.fold + (to.fold - from.fold) * t
  out.base = from.base + (to.base - from.base) * t
  out.height = from.height + (to.height - from.height) * t
  out.drift = from.drift + (to.drift - from.drift) * t
}

const cloneParams = (p: AuroraParams): AuroraParams => ({
  ...p,
  low: p.low.clone(),
  mid: p.mid.clone(),
  high: p.high.clone(),
})

export default function Aurora() {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!containerRef.current) return
    const container = containerRef.current

    const renderer = new THREE.WebGLRenderer({ antialias: false })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(window.innerWidth, window.innerHeight)
    container.appendChild(renderer.domElement)

    // Half float keeps the dark sky gradients from banding in the low-res target
    const halfFloatOk =
      renderer.extensions.has('EXT_color_buffer_half_float') || renderer.extensions.has('EXT_color_buffer_float')
    const target = new THREE.WebGLRenderTarget(1, 1, {
      type: halfFloatOk ? THREE.HalfFloatType : THREE.UnsignedByteType,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      depthBuffer: false,
    })

    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1)
    const quad = new THREE.PlaneGeometry(2, 2)

    const params = randomParams()
    let from = cloneParams(params)
    let to = cloneParams(params)

    const commonUniforms = {
      uAspect: { value: 1 },
      uPixel: { value: 1 },
      uTime: { value: 0 },
      uSkyAngle: { value: Math.random() * 0.5 },
      uWave: { value: new THREE.Vector3() },
    }
    const sceneMaterial = new THREE.ShaderMaterial({
      vertexShader: VERT,
      fragmentShader: SCENE_FRAG,
      depthTest: false,
      depthWrite: false,
      blending: THREE.NoBlending,
      uniforms: {
        ...THREE.UniformsUtils.clone(commonUniforms),
        uFlow: { value: Math.random() * 100 },
        uRay: { value: 0 },
        uActivity: { value: CALM_ACTIVITY },
        uFold: { value: params.fold },
        uBase: { value: params.base },
        uHeight: { value: params.height },
        uBandAngle: { value: (Math.random() < 0.5 ? -1 : 1) * randomBetween(0.4, 1.0) },
        uRidgeSeed: { value: Math.random() * 100 },
        uColLow: { value: params.low },
        uColMid: { value: params.mid },
        uColHigh: { value: params.high },
      },
    })
    const compositeMaterial = new THREE.ShaderMaterial({
      vertexShader: VERT,
      fragmentShader: COMPOSITE_FRAG,
      depthTest: false,
      depthWrite: false,
      uniforms: {
        ...THREE.UniformsUtils.clone(commonUniforms),
        uScene: { value: target.texture },
      },
    })
    const sceneScene = new THREE.Scene()
    sceneScene.add(new THREE.Mesh(quad, sceneMaterial))
    const compositeScene = new THREE.Scene()
    compositeScene.add(new THREE.Mesh(quad, compositeMaterial))
    const su = sceneMaterial.uniforms
    const cu = compositeMaterial.uniforms

    const handleResize = () => {
      const w = window.innerWidth
      const h = window.innerHeight
      renderer.setSize(w, h)
      const dpr = renderer.getPixelRatio()
      const scale = Math.min(RENDER_SCALE, MAX_TARGET_HEIGHT / (h * dpr))
      const tw = Math.max(1, Math.floor(w * dpr * scale))
      const th = Math.max(1, Math.floor(h * dpr * scale))
      target.setSize(tw, th)
      su.uAspect.value = w / h
      cu.uAspect.value = w / h
      su.uPixel.value = 1 / th
      // Stars are sized in CSS pixels so they look the same on any display density
      cu.uPixel.value = 1 / h
    }
    handleResize()

    // Phases accumulate in float64 here so the shader only ever sees small numbers
    let flow = su.uFlow.value as number
    let ray = 0
    let wave1 = 0
    let wave2 = 0
    let wave3 = 0
    let skyAngle = commonUniforms.uSkyAngle.value
    let twinkleTime = 0
    let cycleTime = 0
    let cycleLength = randomBetween(MIN_CYCLE_SECONDS, MAX_CYCLE_SECONDS)
    let peakActivity = randomBetween(0.7, 1.0)
    let elapsed = 0
    let lastTime = performance.now()
    let animationId = 0

    const startCycle = () => {
      cycleTime = 0
      cycleLength = randomBetween(MIN_CYCLE_SECONDS, MAX_CYCLE_SECONDS)
      peakActivity = randomBetween(0.7, 1.0)
      from = cloneParams(params)
      to = randomParams()
      // Calm opening of the cycle is the least visible moment to reset the phases
      if (flow > FLOW_WRAP) flow = Math.random() * 100
      if (Math.abs(ray) > FLOW_WRAP) ray = 0
    }

    const animate = () => {
      animationId = requestAnimationFrame(animate)
      const now = performance.now()
      // Clamp so a backgrounded tab resumes smoothly instead of leaping ahead
      const dt = Math.min((now - lastTime) / 1000, 0.1)
      lastTime = now
      elapsed += dt

      cycleTime += dt
      if (cycleTime >= cycleLength) startCycle()
      lerpParams(from, to, smoothstep(0, TRANSITION_SECONDS, cycleTime), params)

      // Calm glow, then a surge that peaks past the middle of the cycle, then a long fade
      const phase = cycleTime / cycleLength
      const envelope = smoothstep(0.15, 0.5, phase) * (1 - smoothstep(0.62, 0.95, phase))
      const flicker = 0.88 + 0.12 * Math.sin(elapsed * 1.3) * Math.sin(elapsed * 0.47 + 1)
      const activity = CALM_ACTIVITY + (peakActivity - CALM_ACTIVITY) * envelope * flicker

      flow += dt * FLOW_SPEED * (0.5 + 1.5 * activity)
      ray += dt * RAY_SPEED * params.drift * (0.4 + 1.6 * activity)
      const tau = Math.PI * 2
      wave1 = (wave1 + dt * 0.9) % tau
      wave2 = (wave2 + dt * 0.35) % tau
      wave3 = (wave3 + dt * 1.3) % tau
      skyAngle = (skyAngle + dt * SKY_ROTATION) % tau
      twinkleTime = (twinkleTime + dt) % TWINKLE_WRAP

      for (const u of [su, cu]) {
        u.uTime.value = twinkleTime
        u.uSkyAngle.value = skyAngle
        ;(u.uWave.value as THREE.Vector3).set(wave1, wave2, wave3)
      }
      su.uFlow.value = flow
      su.uRay.value = ray
      su.uActivity.value = activity
      su.uFold.value = params.fold
      su.uBase.value = params.base
      su.uHeight.value = params.height

      renderer.setRenderTarget(target)
      renderer.render(sceneScene, camera)
      renderer.setRenderTarget(null)
      renderer.render(compositeScene, camera)
    }

    animate()
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      cancelAnimationFrame(animationId)
      quad.dispose()
      sceneMaterial.dispose()
      compositeMaterial.dispose()
      target.dispose()
      renderer.dispose()
      container.removeChild(renderer.domElement)
    }
  }, [])

  return <div ref={containerRef} className="w-full h-full" />
}
