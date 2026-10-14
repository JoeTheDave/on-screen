import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js'

// Configuration constants
// Units: the Schwarzschild radius is 1, so the photon sphere sits at r = 1.5 and the
// innermost stable circular orbit (where a real disk ends) at r = 3.
const RENDER_SCALE = 0.6 // fraction of CSS resolution the ray tracer runs at
const MAX_RENDER_PIXELS = 1_200_000 // hard cap so 4K screens don't melt the GPU
const LOOK_SECONDS = 60 // time on each look (palette, disk shape, camera path) before re-rolling
const TRANSITION_SECONDS = 10 // cross-fade between looks
const FLOW_PERIOD = 7 // seconds per flow-map layer; see the disk noise in the shader
const ORBIT_SECONDS = 150 // one full camera revolution around the hole
const BOB_SECONDS = 47 // camera rising above / sinking toward the disk plane
const DOLLY_SECONDS = 71 // camera distance breathing

// Color ramps for the disk, coolest to hottest (sRGB hex; converted to linear below).
// Doppler shift slides the approaching side up the ramp and the receding side down it.
const PALETTES: number[][] = [
  [0x5a1200, 0xc84a0a, 0xffb347, 0xfff1d6], // Interstellar
  [0x5a1200, 0xc84a0a, 0xffb347, 0xfff1d6], // (weighted: it's the classic)
  [0x3d1a00, 0xb86a00, 0xffd27a, 0xfffbe8], // gold
  [0x4a0018, 0xd02a3a, 0xff9a6a, 0xfff0e0], // ember rose
  [0x1a0a5a, 0x2f5bd8, 0x8fd0ff, 0xf2fbff], // blue giant
  [0x2a0040, 0x9b2fd0, 0xff7ad9, 0xfff0ff], // violet plasma
  [0x00262a, 0x00a39a, 0x9cffd8, 0xf0fff8], // teal aurora
]

const vertexShader = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`

const fragmentShader = /* glsl */ `
varying vec2 vUv;

uniform float uAspect;
uniform float uTanHalfFov;
uniform vec3 uCamPos;
uniform vec3 uCamRight;
uniform vec3 uCamUp;
uniform vec3 uCamFwd;

uniform float uDiskIn;
uniform float uDiskOut;
uniform float uDiskGain;
uniform float uBeam;
uniform float uNoiseScale;
uniform float uFlowPeriod;
uniform vec4 uFlow; // phase A, phase B, weight A, weight B
uniform vec3 uOffA;
uniform vec3 uOffB;
uniform vec3 uPal0;
uniform vec3 uPal1;
uniform vec3 uPal2;
uniform vec3 uPal3;
uniform float uRingGain;

uniform vec3 uNebA;
uniform vec3 uNebB;
uniform float uNebGain;
uniform vec3 uBandNormal;
uniform vec3 uNebSeed;

#define MAX_STEPS 180
#define SPIN_K 4.0

// PCG-style integer hash: decorrelated and exact on every GPU, unlike float-trick hashes
// (which line stars up in rows when fed integer cell ids)
uvec3 pcg3d(uvec3 v) {
  v = v * 1664525u + 1013904223u;
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  v ^= v >> 16u;
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  return v;
}

// p must hold integer values (cell ids)
vec3 hash33(vec3 p) {
  return vec3(pcg3d(uvec3(ivec3(p)))) * (1.0 / 4294967295.0);
}

float hash13(vec3 p) {
  return hash33(p).x;
}

float vnoise(vec3 p) {
  vec3 i = floor(p);
  vec3 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  float a = hash13(i);
  float b = hash13(i + vec3(1.0, 0.0, 0.0));
  float c = hash13(i + vec3(0.0, 1.0, 0.0));
  float d = hash13(i + vec3(1.0, 1.0, 0.0));
  float e = hash13(i + vec3(0.0, 0.0, 1.0));
  float g = hash13(i + vec3(1.0, 0.0, 1.0));
  float h = hash13(i + vec3(0.0, 1.0, 1.0));
  float k = hash13(i + vec3(1.0, 1.0, 1.0));
  return mix(mix(mix(a, b, f.x), mix(c, d, f.x), f.y), mix(mix(e, g, f.x), mix(h, k, f.x), f.y), f.z);
}

float fbm4(vec3 p) {
  float s = 0.0;
  float amp = 0.5;
  for (int i = 0; i < 4; i++) {
    s += amp * vnoise(p);
    p = p * 2.03 + vec3(1.7, 9.2, 3.1);
    amp *= 0.5;
  }
  return s / 0.9375;
}

float fbm5(vec3 p) {
  float s = 0.0;
  float amp = 0.5;
  for (int i = 0; i < 5; i++) {
    s += amp * vnoise(p);
    p = p * 2.03 + vec3(1.7, 9.2, 3.1);
    amp *= 0.5;
  }
  return s / 0.96875;
}

// Temperature -> color along the current palette. t ~ 1 is the hottest part of the disk
// at rest; Doppler boosting can push it past 1 toward the white-hot end.
vec3 ramp(float t) {
  t = max(t, 0.0);
  vec3 c = mix(vec3(0.0), uPal0, smoothstep(0.0, 0.3, t));
  c = mix(c, uPal1, smoothstep(0.25, 0.6, t));
  c = mix(c, uPal2, smoothstep(0.55, 0.95, t));
  c = mix(c, uPal3, smoothstep(0.9, 1.4, t));
  return c;
}

// Disk turbulence in log-polar coordinates: a circle in noise space (tangential) times a
// log-radius axis, so features come out as streaks stretched along the orbit.
float diskNoise(float phi, float logr, vec3 off) {
  vec3 q = vec3(cos(phi) * 3.0, sin(phi) * 3.0, logr * 8.0) * uNoiseScale + off;
  return fbm4(q);
}

// Emission (rgb) and opacity (a) where a ray crosses the disk plane
vec4 diskSample(vec3 hit, vec3 rayDir) {
  float r = length(hit.xz);
  if (r < uDiskIn * 0.9 || r > uDiskOut) return vec4(0.0);

  float phi = atan(hit.z, hit.x);
  float logr = log(r);

  // Differential (Keplerian) rotation shears any texture into infinitely thin threads
  // over time. Flow-map trick: two copies of the noise, each only rotated through half a
  // period before it fades out and restarts (with a fresh seed) while the other takes over.
  float omega = SPIN_K * pow(r, -1.5);
  float na = diskNoise(phi - omega * (uFlow.x - 0.5) * uFlowPeriod, logr, uOffA);
  float nb = diskNoise(phi - omega * (uFlow.y - 0.5) * uFlowPeriod, logr, uOffB);
  float n = na * uFlow.z + nb * uFlow.w;
  // Blending two noises flattens contrast mid-fade; renormalize the variance
  n = (n - 0.5) / sqrt(uFlow.z * uFlow.z + uFlow.w * uFlow.w) + 0.5;

  float dens = smoothstep(0.2, 0.85, n);
  // Fine ring structure, warped by the turbulence
  dens *= 0.6 + 0.4 * sin(logr * 26.0 + n * 7.0);

  float edge = smoothstep(uDiskIn * 0.9, uDiskIn * 1.15, r) * (1.0 - smoothstep(uDiskOut * 0.5, uDiskOut, r));

  // Thin-disk temperature profile, normalized so its peak is ~1
  float x = uDiskIn / r;
  float temp = pow(x, 0.75) * pow(max(1.0 - 0.97 * sqrt(x), 0.0), 0.25) / 0.51;

  // Relativistic Doppler + gravitational redshift. Orbital speed of a circular orbit as
  // seen by a static observer: beta = sqrt(M / (r - 2M)) with M = 0.5.
  float beta = min(sqrt(0.5 / max(r - 1.0, 0.05)), 0.7);
  float gamma = 1.0 / sqrt(1.0 - beta * beta);
  vec3 orbitDir = normalize(vec3(-hit.z, 0.0, hit.x));
  // The photon travels toward the camera, i.e. opposite to our backwards-traced ray
  float cosTheta = dot(orbitDir, -rayDir);
  float g = sqrt(max(1.0 - 1.0 / r, 0.0)) / (gamma * (1.0 - beta * cosTheta));

  vec3 col = ramp(temp * g * 1.1);
  float intensity = uDiskGain * pow(g, uBeam) * (0.2 + 1.6 * temp * temp);
  float alpha = clamp(dens * edge * 1.15, 0.0, 0.96);
  return vec4(col * intensity * (0.4 + 0.6 * dens), alpha);
}

vec3 starLayer(vec3 d, float scale, float density, float seed, float gain) {
  vec3 p = d * scale;
  vec3 id = floor(p);
  vec3 f = fract(p) - 0.5;
  vec3 h = hash33(id + seed);
  if (h.z > density) return vec3(0.0);
  vec3 off = (hash33(id + seed + 1717.0) - 0.5) * 0.6;
  vec3 dv = f - off;
  float s = exp(-dot(dv, dv) * 180.0);
  float b = pow(h.x, 6.0) * 3.0 + 0.04;
  vec3 tint = mix(vec3(1.0, 0.72, 0.5), vec3(0.62, 0.78, 1.0), h.y);
  return tint * (b * s * gain);
}

vec3 background(vec3 d) {
  // A faint galactic band: denser stars and brighter nebula near one great circle
  float bd = dot(d, uBandNormal);
  float band = exp(-bd * bd * 7.0);

  vec3 q = d * 2.2 + uNebSeed;
  float n1 = fbm5(q);
  float n2 = fbm4(q * 2.3 + 7.1);
  float neb = smoothstep(0.38, 0.9, n1) * (0.25 + 0.75 * band);
  vec3 c = mix(uNebA, uNebB, n2) * (neb * uNebGain);

  float starBoost = 0.55 + 0.9 * band;
  c += starLayer(d, 45.0, 0.45, 0.0, 1.0) * starBoost;
  c += starLayer(d, 95.0, 0.5, 41.0, 0.6) * starBoost;
  c += starLayer(d, 190.0, 0.55, 83.0, 0.35) * starBoost;
  return c;
}

void main() {
  vec2 ndc = vUv * 2.0 - 1.0;
  vec3 dir = normalize(uCamFwd + (ndc.x * uAspect * uTanHalfFov) * uCamRight + (ndc.y * uTanHalfFov) * uCamUp);

  vec3 pos = uCamPos;
  vec3 vel = dir;
  // Specific angular momentum; conserved, and it sets the strength of the bending
  vec3 hv = cross(pos, vel);
  float h2 = dot(hv, hv);

  float r = length(pos);
  vec3 acc = -1.5 * h2 * pos / (r * r * r * r * r);
  float escapeR = max(length(uCamPos) * 1.3, 30.0);

  vec3 col = vec3(0.0);
  float trans = 1.0;
  float glow = 0.0;
  bool captured = false;

  for (int i = 0; i < MAX_STEPS; i++) {
    r = length(pos);
    if (r < 1.0) { captured = true; break; }
    if (r > escapeR && dot(pos, vel) > 0.0) break;

    // Step size shrinks near the hole, where the bending is strong
    float dt = clamp(0.06 * r * min(1.0, r - 0.9), 0.03, 3.0);

    // Leapfrog (kick-drift-kick) keeps orbits around the photon sphere stable
    vel += acc * (0.5 * dt);
    vec3 prev = pos;
    pos += vel * dt;
    float rn = length(pos);
    acc = -1.5 * h2 * pos / (rn * rn * rn * rn * rn);
    vel += acc * (0.5 * dt);

    if (prev.y * pos.y < 0.0) {
      vec3 hit = mix(prev, pos, prev.y / (prev.y - pos.y));
      vec4 d = diskSample(hit, normalize(vel));
      col += trans * d.rgb * d.a;
      trans *= 1.0 - d.a;
      if (trans < 0.02) break;
    }

    // Light lingering near the photon sphere: a soft halo hugging the shadow
    float ring = (rn - 1.5) * 5.0;
    glow += exp(-ring * ring) * dt;
  }

  // The threshold keeps rays that merely cross r = 1.5 on their way in from tinting the shadow
  col += trans * max(glow - 0.4, 0.0) * uRingGain * ramp(1.0);
  if (!captured) col += trans * background(normalize(vel));

  gl_FragColor = vec4(col, 1.0);
}
`

interface Look {
  pal0: number[]
  pal1: number[]
  pal2: number[]
  pal3: number[]
  diskIn: number[]
  diskOut: number[]
  diskGain: number[]
  beam: number[]
  noiseScale: number[]
  ringGain: number[]
  nebA: number[]
  nebB: number[]
  nebGain: number[]
  band: number[]
  elevBase: number[]
  elevAmp: number[]
  distBase: number[]
  roll: number[]
}

const rand = (lo: number, hi: number) => lo + Math.random() * (hi - lo)

const linear = (hex: number) => {
  const c = new THREE.Color(hex) // ColorManagement converts sRGB hex to linear
  return [c.r, c.g, c.b]
}

const randomLook = (): Look => {
  const pal = PALETTES[Math.floor(Math.random() * PALETTES.length)]
  const nebA = new THREE.Color().setHSL(Math.random(), rand(0.4, 0.8), rand(0.25, 0.45))
  const nebB = new THREE.Color().setHSL(Math.random(), rand(0.4, 0.8), rand(0.2, 0.4))
  const band = new THREE.Vector3(rand(-1, 1), rand(-1, 1), rand(-1, 1))
  if (band.lengthSq() < 1e-4) band.set(0.3, 1, 0.2)
  band.normalize()
  return {
    pal0: linear(pal[0]),
    pal1: linear(pal[1]),
    pal2: linear(pal[2]),
    pal3: linear(pal[3]),
    diskIn: [rand(2.7, 3.2)],
    diskOut: [rand(9, 14)],
    diskGain: [rand(2.2, 3.4)],
    beam: [rand(2.4, 3.4)], // Doppler beaming exponent (physical thin-disk value is ~3-4)
    noiseScale: [rand(0.8, 1.4)],
    ringGain: [rand(0.03, 0.07)],
    nebA: [nebA.r, nebA.g, nebA.b],
    nebB: [nebB.r, nebB.g, nebB.b],
    nebGain: [rand(0.06, 0.16)],
    band: [band.x, band.y, band.z],
    elevBase: [rand(0.05, 0.3)], // radians above the disk plane; low is the cinematic look
    elevAmp: [rand(0.06, 0.18)],
    distBase: [rand(19, 25)],
    roll: [rand(-0.22, 0.22)],
  }
}

const lerpLook = (a: Look, b: Look, k: number, out: Look) => {
  for (const key of Object.keys(out) as (keyof Look)[]) {
    const o = out[key]
    for (let i = 0; i < o.length; i++) o[i] = a[key][i] + (b[key][i] - a[key][i]) * k
  }
}

const cloneLook = (l: Look): Look =>
  Object.fromEntries(Object.entries(l).map(([k, v]) => [k, [...(v as number[])]])) as unknown as Look

// A stable pseudo-random noise offset per flow-map cycle (re-seeds a layer when it is invisible)
const cycleOffset = (cycle: number, out: THREE.Vector3) => {
  const f = (s: number) => (((Math.sin(cycle * s) * 43758.5453) % 1) + 1) % 1
  out.set(f(12.9898) * 200, f(78.233) * 200, f(37.719) * 200)
}

export default function BlackHole() {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!containerRef.current) return
    const container = containerRef.current

    const renderer = new THREE.WebGLRenderer({ antialias: false })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(window.innerWidth, window.innerHeight)
    // The shader works in linear HDR; OutputPass tone-maps and converts to sRGB at the end
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1
    container.appendChild(renderer.domElement)

    // One full-screen quad running the ray tracer
    const scene = new THREE.Scene()
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1)
    const uniforms = {
      uAspect: { value: 1 },
      uTanHalfFov: { value: 0.35 },
      uCamPos: { value: new THREE.Vector3() },
      uCamRight: { value: new THREE.Vector3() },
      uCamUp: { value: new THREE.Vector3() },
      uCamFwd: { value: new THREE.Vector3() },
      uDiskIn: { value: 3 },
      uDiskOut: { value: 12 },
      uDiskGain: { value: 3 },
      uBeam: { value: 3 },
      uNoiseScale: { value: 1 },
      uFlowPeriod: { value: FLOW_PERIOD },
      uFlow: { value: new THREE.Vector4() },
      uOffA: { value: new THREE.Vector3() },
      uOffB: { value: new THREE.Vector3() },
      uPal0: { value: new THREE.Vector3() },
      uPal1: { value: new THREE.Vector3() },
      uPal2: { value: new THREE.Vector3() },
      uPal3: { value: new THREE.Vector3() },
      uRingGain: { value: 0.05 },
      uNebA: { value: new THREE.Vector3() },
      uNebB: { value: new THREE.Vector3() },
      uNebGain: { value: 0.1 },
      uBandNormal: { value: new THREE.Vector3(0, 1, 0) },
      uNebSeed: { value: new THREE.Vector3(rand(0, 100), rand(0, 100), rand(0, 100)) },
    }
    const geometry = new THREE.PlaneGeometry(2, 2)
    const material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms,
      depthTest: false,
      depthWrite: false,
    })
    const quad = new THREE.Mesh(geometry, material)
    quad.frustumCulled = false
    scene.add(quad)

    // The composer runs at reduced resolution (HalfFloat targets keep the HDR for bloom);
    // the final OutputPass draws to the full-size canvas, upscaling with linear filtering
    const composer = new EffectComposer(renderer)
    composer.setPixelRatio(1)
    const renderPass = new RenderPass(scene, camera)
    const bloomPass = new UnrealBloomPass(new THREE.Vector2(256, 256), 0.7, 0.55, 0.7)
    const outputPass = new OutputPass()
    composer.addPass(renderPass)
    composer.addPass(bloomPass)
    composer.addPass(outputPass)

    const handleResize = () => {
      const w = window.innerWidth
      const h = window.innerHeight
      renderer.setSize(w, h)
      const scale = Math.min(RENDER_SCALE, Math.sqrt(MAX_RENDER_PIXELS / (w * h)))
      composer.setSize(Math.max(1, Math.round(w * scale)), Math.max(1, Math.round(h * scale)))
      const aspect = w / h
      uniforms.uAspect.value = aspect
      // ~38 degree vertical FOV on landscape; widen on portrait so the disk still fits across
      uniforms.uTanHalfFov.value = Math.max(Math.tan(THREE.MathUtils.degToRad(19)), Math.tan(THREE.MathUtils.degToRad(29)) / aspect)
    }
    handleResize()

    let fromLook = randomLook()
    let toLook = fromLook
    const look = cloneLook(fromLook)
    let lookAge = 0 // seconds since the current transition began

    const applyLook = () => {
      uniforms.uPal0.value.fromArray(look.pal0)
      uniforms.uPal1.value.fromArray(look.pal1)
      uniforms.uPal2.value.fromArray(look.pal2)
      uniforms.uPal3.value.fromArray(look.pal3)
      uniforms.uDiskIn.value = look.diskIn[0]
      uniforms.uDiskOut.value = look.diskOut[0]
      uniforms.uDiskGain.value = look.diskGain[0]
      uniforms.uBeam.value = look.beam[0]
      uniforms.uNoiseScale.value = look.noiseScale[0]
      uniforms.uRingGain.value = look.ringGain[0]
      uniforms.uNebA.value.fromArray(look.nebA)
      uniforms.uNebB.value.fromArray(look.nebB)
      uniforms.uNebGain.value = look.nebGain[0]
      uniforms.uBandNormal.value.fromArray(look.band).normalize()
    }

    const camPos = new THREE.Vector3()
    const fwd = new THREE.Vector3()
    const right = new THREE.Vector3()
    const up = new THREE.Vector3()
    const worldUp = new THREE.Vector3(0, 1, 0)

    // Animation clock built from clamped frame deltas, so a backgrounded tab resumes smoothly
    let time = 0
    let azimuth = Math.random() * Math.PI * 2
    let lastTime = performance.now()
    let animationId = 0

    const animate = () => {
      animationId = requestAnimationFrame(animate)
      const now = performance.now()
      const dt = Math.min((now - lastTime) / 1000, 1 / 20)
      lastTime = now
      time += dt
      lookAge += dt

      if (lookAge >= LOOK_SECONDS) {
        fromLook = cloneLook(look)
        toLook = randomLook()
        lookAge = 0
      }
      const k = THREE.MathUtils.smoothstep(lookAge / TRANSITION_SECONDS, 0, 1)
      lerpLook(fromLook, toLook, k, look)
      applyLook()

      // Disk flow-map phases (computed here in double precision so the GPU never sees a
      // large time value). Each layer's weight hits 0 exactly when its phase wraps.
      const tf = time / FLOW_PERIOD
      const pA = tf - Math.floor(tf)
      const pB = tf + 0.5 - Math.floor(tf + 0.5)
      const wA = 1 - Math.abs(2 * pA - 1)
      uniforms.uFlow.value.set(pA, pB, wA, 1 - wA)
      cycleOffset(Math.floor(tf), uniforms.uOffA.value)
      cycleOffset(Math.floor(tf + 0.5) + 0.5, uniforms.uOffB.value)

      // Camera: slow orbit around the hole, bobbing above the disk plane, gentle dolly
      azimuth += (dt / ORBIT_SECONDS) * Math.PI * 2
      const elevation = look.elevBase[0] + look.elevAmp[0] * Math.sin((time / BOB_SECONDS) * Math.PI * 2)
      const distance = look.distBase[0] + 1.5 * Math.sin((time / DOLLY_SECONDS) * Math.PI * 2)
      camPos.set(
        distance * Math.cos(elevation) * Math.cos(azimuth),
        distance * Math.sin(elevation),
        distance * Math.cos(elevation) * Math.sin(azimuth),
      )
      fwd.copy(camPos).negate().normalize()
      right.crossVectors(fwd, worldUp).normalize()
      up.crossVectors(right, fwd)
      const roll = look.roll[0]
      const cr = Math.cos(roll)
      const sr = Math.sin(roll)
      uniforms.uCamPos.value.copy(camPos)
      uniforms.uCamFwd.value.copy(fwd)
      uniforms.uCamRight.value.copy(right).multiplyScalar(cr).addScaledVector(up, sr)
      uniforms.uCamUp.value.copy(up).multiplyScalar(cr).addScaledVector(right, -sr)

      composer.render()
    }

    animate()
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      cancelAnimationFrame(animationId)
      geometry.dispose()
      material.dispose()
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
