import { useEffect, useRef } from 'react'
import * as THREE from 'three'

// Configuration constants
// The fractal is raymarched at a fraction of the screen resolution and bilinearly upscaled:
// the DE is expensive (8 trig-heavy iterations per sample, ~100+ samples per pixel), and the
// soft, glowing look survives the upscale far better than the frame rate would survive 1:1.
const RENDER_SCALE = 0.5
const MAX_RENDER_PIXELS = 700_000 // hard cap so 4K / retina screens don't blow the budget
const LOOK_MIN_SECONDS = 40 // time on one palette/light "look" before re-randomizing
const LOOK_MAX_SECONDS = 75
const TRANSITION_SECONDS = 12 // crossfade between looks
const FADE_IN_SECONDS = 3
const POWER_PERIOD_SECONDS = 70 // one full breath of the power parameter
const ZOOM_PERIOD_SECONDS = 53 // camera drift in/out (co-prime-ish with the others)
const ELEVATION_PERIOD_SECONDS = 37
const CAMERA_NEAR = 1.75
const CAMERA_FAR = 3.0
const CAMERA_ELEVATION = 0.5 // radians of up/down swing
const PALETTE_DRIFT = 0.012 // palette cycles per second, so colors creep even within a look

// Curated cosine palettes (a + b*cos(2pi(c*x + d))) that read as iridescent / oil-slick.
// Each look picks one and rotates its phase, so the same family never repeats exactly.
const PALETTES: [number, number, number][][] = [
  [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [0.0, 0.33, 0.67]], // full spectrum
  [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [1.0, 1.0, 0.5], [0.8, 0.9, 0.3]], // oil slick
  [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [1.0, 0.7, 0.4], [0.0, 0.15, 0.2]], // teal / magenta
  [[0.8, 0.5, 0.4], [0.2, 0.4, 0.2], [2.0, 1.0, 1.0], [0.0, 0.25, 0.25]], // gold / violet
  [[0.6, 0.55, 0.62], [0.4, 0.35, 0.4], [1.0, 1.0, 1.0], [0.0, 0.1, 0.2]], // pearl
  [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [2.0, 1.0, 0.0], [0.5, 0.2, 0.25]], // fire / ice
  [[0.45, 0.5, 0.55], [0.45, 0.45, 0.45], [1.0, 1.0, 1.0], [0.6, 0.7, 0.8]], // abalone
]

interface Look {
  palA: THREE.Vector3
  palB: THREE.Vector3
  palC: THREE.Vector3
  palD: THREE.Vector3
  light: THREE.Vector3
  powerCenter: number
  powerAmp: number
  orbitSpeed: number // radians per second; sign picks the orbit direction
}

const randomLook = (): Look => {
  const [a, b, c, d] = PALETTES[Math.floor(Math.random() * PALETTES.length)]
  const hueShift = Math.random() // same offset on all channels = a hue rotation
  const lightAz = Math.random() * Math.PI * 2
  const lightEl = 0.3 + Math.random() * 0.7
  return {
    palA: new THREE.Vector3(...a),
    palB: new THREE.Vector3(...b),
    palC: new THREE.Vector3(...c),
    palD: new THREE.Vector3(d[0] + hueShift, d[1] + hueShift, d[2] + hueShift),
    light: new THREE.Vector3(
      Math.cos(lightEl) * Math.cos(lightAz),
      Math.sin(lightEl),
      Math.cos(lightEl) * Math.sin(lightAz),
    ),
    // Center 7.5-8.5, amplitude 1.5-2 => power sweeps roughly 6 -> 10
    powerCenter: 7.5 + Math.random(),
    powerAmp: 1.5 + Math.random() * 0.5,
    orbitSpeed: (Math.random() < 0.5 ? 1 : -1) * (0.035 + Math.random() * 0.03),
  }
}

const VERTEX_SHADER = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`

const MARCH_SHADER = /* glsl */ `
uniform vec2 uResolution;
uniform float uPower;
uniform vec3 uCamPos;
uniform vec3 uCamTarget;
uniform vec3 uLightDir;
uniform vec3 uPalA;
uniform vec3 uPalB;
uniform vec3 uPalC;
uniform vec3 uPalD;
uniform float uPalShift;
uniform float uFade;

#define MAX_STEPS 90
#define ITERATIONS 8
#define SHADOW_STEPS 16
#define BOUND 1.45
#define FOCAL 2.0

vec3 palette(float x) {
  return uPalA + uPalB * cos(6.28318 * (uPalC * x + uPalD));
}

// Mandelbulb distance estimator. trap collects the orbit's closest approach to the
// coordinate planes (xyz) and to the origin (w) for coloring.
float mandelbulb(vec3 p, out vec4 trap) {
  vec3 z = p;
  float dr = 1.0;
  float r = length(z);
  trap = vec4(abs(z), r * r);
  for (int i = 0; i < ITERATIONS; i++) {
    if (r > 2.0) break;
    float rr = max(r, 1e-6);
    float theta = acos(clamp(z.y / rr, -1.0, 1.0)) * uPower;
    float phi = atan(z.z, z.x) * uPower;
    float rp = pow(rr, uPower - 1.0);
    dr = rp * uPower * dr + 1.0;
    float zr = rp * rr;
    z = p + zr * vec3(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi));
    r = length(z);
    trap = min(trap, vec4(abs(z), r * r));
  }
  return 0.5 * log(max(r, 1e-6)) * r / dr;
}

float map(vec3 p) {
  vec4 t;
  return mandelbulb(p, t);
}

vec2 sphereHit(vec3 ro, vec3 rd, float rad) {
  float b = dot(ro, rd);
  float c = dot(ro, ro) - rad * rad;
  float h = b * b - c;
  if (h < 0.0) return vec2(-1.0);
  h = sqrt(h);
  return vec2(-b - h, -b + h);
}

vec3 calcNormal(vec3 p, float h) {
  const vec2 k = vec2(1.0, -1.0);
  return normalize(
    k.xyy * map(p + k.xyy * h) +
    k.yyx * map(p + k.yyx * h) +
    k.yxy * map(p + k.yxy * h) +
    k.xxx * map(p + k.xxx * h));
}

float softShadow(vec3 ro, vec3 rd) {
  vec2 bs = sphereHit(ro, rd, BOUND);
  float tmax = max(bs.y, 0.0);
  float res = 1.0;
  float t = 0.01;
  for (int i = 0; i < SHADOW_STEPS; i++) {
    float h = map(ro + rd * t);
    res = min(res, 8.0 * h / t);
    t += clamp(h, 0.01, 0.2);
    if (res < 0.01 || t > tmax) break;
  }
  res = clamp(res, 0.0, 1.0);
  return res * res * (3.0 - 2.0 * res);
}

float calcAO(vec3 p, vec3 n) {
  float occ = 0.0;
  float sca = 1.0;
  for (int i = 0; i < 4; i++) {
    float h = 0.01 + 0.09 * float(i) / 3.0;
    occ += (h - map(p + n * h)) * sca;
    sca *= 0.8;
  }
  return clamp(1.0 - 2.5 * occ, 0.0, 1.0);
}

void main() {
  vec2 uv = (2.0 * gl_FragCoord.xy - uResolution) / uResolution.y;
  vec3 ro = uCamPos;
  vec3 ww = normalize(uCamTarget - ro);
  vec3 uu = normalize(cross(ww, vec3(0.0, 1.0, 0.0)));
  vec3 vv = cross(uu, ww);
  vec3 rd = normalize(uv.x * uu + uv.y * vv + FOCAL * ww);
  // Angular size of one (low-res) pixel: hit threshold grows with distance, so detail
  // stops at what the pixel can show instead of burning steps on sub-pixel filigree
  float pixel = 1.5 / (uResolution.y * FOCAL);

  vec3 col = vec3(0.0);
  float glow = 0.0;
  float hitT = -1.0;
  vec2 bs = sphereHit(ro, rd, BOUND);

  if (bs.y > 0.0) {
    float t = max(bs.x, 0.0);
    bool exited = false;
    for (int i = 0; i < MAX_STEPS; i++) {
      float d = map(ro + rd * t);
      // Near-misses accumulate a halo; this is what lights the black around the silhouette
      glow += 1.0 / (1.0 + 2500.0 * d * d);
      if (d < pixel * t) { hitT = t; break; }
      t += d * 0.85; // understep: the DE is not exact for non-integer powers
      if (t > bs.y) { exited = true; break; }
    }
    // Out of steps while still deep in a crevice: call it a hit rather than leave a hole
    if (hitT < 0.0 && !exited) hitT = t;
  }

  vec3 glowCol = palette(uPalShift + 0.35);

  if (hitT > 0.0) {
    vec3 pos = ro + rd * hitT;
    float eps = pixel * hitT;
    vec4 trap;
    mandelbulb(pos, trap);
    vec3 n = calcNormal(pos, max(eps * 0.5, 1e-4));
    vec3 l = normalize(uLightDir);
    vec3 v = -rd;

    float ndv = clamp(dot(n, v), 0.0, 1.0);
    float fres = pow(1.0 - ndv, 3.0);
    float dif = clamp(dot(n, l), 0.0, 1.0);
    float sha = dif > 0.001 ? softShadow(pos + n * eps * 2.0, l) : 0.0;
    float ao = calcAO(pos, n);
    // Orbit traps also encode crevice depth: points whose orbit dives near the origin
    // sit deep in folds, which deepens the DE-based AO cheaply
    ao *= clamp(0.4 + 1.6 * sqrt(trap.w), 0.0, 1.0);

    // Orbit-trap coloring; the view-angle term makes the hue slide across the surface as
    // the camera moves, which is what reads as iridescence rather than paint
    float x = 0.55 * sqrt(trap.w) + 0.6 * trap.y + 0.25 * trap.x + uPalShift;
    vec3 albedo = palette(x);
    vec3 sheen = palette(x + 0.45 * (1.0 - ndv) + 0.2);

    vec3 h = normalize(l + v);
    float spe = pow(clamp(dot(n, h), 0.0, 1.0), 40.0) * dif * sha;
    float bac = clamp(dot(n, normalize(vec3(-l.x, 0.2, -l.z))), 0.0, 1.0);
    float sky = 0.5 + 0.5 * n.y;

    vec3 lin = vec3(0.0);
    lin += 2.2 * dif * sha * vec3(1.0, 0.95, 0.88);
    lin += 0.35 * sky * ao * mix(glowCol, vec3(0.6, 0.7, 1.0), 0.5);
    lin += 0.25 * bac * ao * glowCol;
    col = albedo * albedo * lin; // squared albedo: richer, more saturated mids
    col += 1.4 * fres * ao * sheen; // rim light
    col += 0.8 * spe * mix(vec3(1.0), sheen, 0.4);

    // Depth fade: the far side of the bulb sinks into the dark
    float depth = hitT - (length(ro) - 1.1);
    col *= exp(-0.9 * max(depth, 0.0));
    glow *= 0.35;
  }

  col += glowCol * glow * 0.012;

  // Filmic-ish tone map then gamma; output is display-ready
  col = col / (1.0 + col);
  col = pow(col, vec3(0.4545));
  gl_FragColor = vec4(col * uFade, 1.0);
}
`

// Upscale pass: bilinear sample of the low-res frame plus vignette and dither (the dither
// hides 8-bit banding in the dark glow gradients once they're stretched 2x)
const BLIT_SHADER = /* glsl */ `
uniform sampler2D tScene;
uniform float uTime;
varying vec2 vUv;

float hash(vec2 p) {
  return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
  vec3 c = texture2D(tScene, vUv).rgb;
  vec2 q = vUv - 0.5;
  c *= 1.0 - 0.9 * dot(q, q);
  c += (hash(gl_FragCoord.xy + fract(uTime) * 61.0) - 0.5) / 255.0;
  gl_FragColor = vec4(max(c, 0.0), 1.0);
}
`

export default function Mandelbulb() {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!containerRef.current) return
    const container = containerRef.current

    // A full-screen shader needs no MSAA; antialiasing comes from the bilinear upscale
    const renderer = new THREE.WebGLRenderer({ antialias: false })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(window.innerWidth, window.innerHeight)
    container.appendChild(renderer.domElement)

    const renderTarget = new THREE.WebGLRenderTarget(1, 1, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      depthBuffer: false,
    })

    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1)
    const quad = new THREE.PlaneGeometry(2, 2)

    const initial = randomLook()
    const marchUniforms = {
      uResolution: { value: new THREE.Vector2(1, 1) },
      uPower: { value: 8 },
      uCamPos: { value: new THREE.Vector3(0, 0, 2.5) },
      uCamTarget: { value: new THREE.Vector3() },
      uLightDir: { value: initial.light.clone() },
      uPalA: { value: initial.palA.clone() },
      uPalB: { value: initial.palB.clone() },
      uPalC: { value: initial.palC.clone() },
      uPalD: { value: initial.palD.clone() },
      uPalShift: { value: 0 },
      uFade: { value: 0 },
    }
    const marchMaterial = new THREE.ShaderMaterial({
      vertexShader: VERTEX_SHADER,
      fragmentShader: MARCH_SHADER,
      uniforms: marchUniforms,
      depthTest: false,
      depthWrite: false,
    })
    const marchScene = new THREE.Scene()
    marchScene.add(new THREE.Mesh(quad, marchMaterial))

    const blitUniforms = {
      tScene: { value: renderTarget.texture },
      uTime: { value: 0 },
    }
    const blitMaterial = new THREE.ShaderMaterial({
      vertexShader: VERTEX_SHADER,
      fragmentShader: BLIT_SHADER,
      uniforms: blitUniforms,
      depthTest: false,
      depthWrite: false,
    })
    const blitScene = new THREE.Scene()
    blitScene.add(new THREE.Mesh(quad, blitMaterial))

    const handleResize = () => {
      renderer.setSize(window.innerWidth, window.innerHeight)
      const pr = renderer.getPixelRatio()
      let w = window.innerWidth * pr * RENDER_SCALE
      let h = window.innerHeight * pr * RENDER_SCALE
      const over = Math.sqrt((w * h) / MAX_RENDER_PIXELS)
      if (over > 1) {
        w /= over
        h /= over
      }
      w = Math.max(1, Math.floor(w))
      h = Math.max(1, Math.floor(h))
      renderTarget.setSize(w, h)
      marchUniforms.uResolution.value.set(w, h)
    }
    handleResize()

    // Looks crossfade: `from` is a snapshot of whatever was on screen when the switch began
    let from = initial
    let to = initial
    let lookStart = performance.now()
    let lookSeconds = LOOK_MIN_SECONDS + Math.random() * (LOOK_MAX_SECONDS - LOOK_MIN_SECONDS)
    const current: Look = randomLook()
    const blendLook = (k: number) => {
      current.palA.lerpVectors(from.palA, to.palA, k)
      current.palB.lerpVectors(from.palB, to.palB, k)
      current.palC.lerpVectors(from.palC, to.palC, k)
      current.palD.lerpVectors(from.palD, to.palD, k)
      current.light.lerpVectors(from.light, to.light, k).normalize()
      current.powerCenter = from.powerCenter + (to.powerCenter - from.powerCenter) * k
      current.powerAmp = from.powerAmp + (to.powerAmp - from.powerAmp) * k
      current.orbitSpeed = from.orbitSpeed + (to.orbitSpeed - from.orbitSpeed) * k
    }
    const snapshot = (look: Look): Look => ({
      palA: look.palA.clone(),
      palB: look.palB.clone(),
      palC: look.palC.clone(),
      palD: look.palD.clone(),
      light: look.light.clone(),
      powerCenter: look.powerCenter,
      powerAmp: look.powerAmp,
      orbitSpeed: look.orbitSpeed,
    })

    // Motion is integrated phase, not a function of absolute time, so changing speeds
    // between looks never makes the camera jump
    const startTime = performance.now()
    let lastTime = startTime
    let azimuth = Math.random() * Math.PI * 2
    let powerPhase = Math.random() * Math.PI * 2
    let zoomPhase = Math.random() * Math.PI * 2
    let elevationPhase = Math.random() * Math.PI * 2
    let palShift = Math.random()
    let animationId = 0

    const animate = () => {
      animationId = requestAnimationFrame(animate)
      const now = performance.now()
      // Clamp so a backgrounded tab doesn't lurch forward on return
      const dt = Math.min((now - lastTime) / 1000, 1 / 20)
      lastTime = now

      if ((now - lookStart) / 1000 >= lookSeconds) {
        from = snapshot(current)
        to = randomLook()
        lookStart = now
        lookSeconds = LOOK_MIN_SECONDS + Math.random() * (LOOK_MAX_SECONDS - LOOK_MIN_SECONDS)
      }
      const raw = Math.min((now - lookStart) / 1000 / TRANSITION_SECONDS, 1)
      blendLook(raw * raw * (3 - 2 * raw))

      azimuth += current.orbitSpeed * dt
      powerPhase += (dt / POWER_PERIOD_SECONDS) * Math.PI * 2
      zoomPhase += (dt / ZOOM_PERIOD_SECONDS) * Math.PI * 2
      elevationPhase += (dt / ELEVATION_PERIOD_SECONDS) * Math.PI * 2
      palShift = (palShift + dt * PALETTE_DRIFT) % 1

      const distance =
        CAMERA_NEAR + (CAMERA_FAR - CAMERA_NEAR) * (0.5 + 0.5 * Math.sin(zoomPhase))
      const elevation = CAMERA_ELEVATION * Math.sin(elevationPhase)
      marchUniforms.uCamPos.value.set(
        distance * Math.cos(elevation) * Math.cos(azimuth),
        distance * Math.sin(elevation),
        distance * Math.cos(elevation) * Math.sin(azimuth),
      )
      // Aim slightly off-center when close so the close-ups frame lobes, not the core
      const closeness = 1 - (distance - CAMERA_NEAR) / (CAMERA_FAR - CAMERA_NEAR)
      marchUniforms.uCamTarget.value.set(
        0.18 * closeness * Math.sin(azimuth * 1.7),
        0.12 * closeness * Math.cos(elevationPhase * 1.3),
        0.18 * closeness * Math.cos(azimuth * 1.7),
      )

      marchUniforms.uPower.value = current.powerCenter + current.powerAmp * Math.sin(powerPhase)
      marchUniforms.uLightDir.value.copy(current.light)
      marchUniforms.uPalA.value.copy(current.palA)
      marchUniforms.uPalB.value.copy(current.palB)
      marchUniforms.uPalC.value.copy(current.palC)
      marchUniforms.uPalD.value.copy(current.palD)
      marchUniforms.uPalShift.value = palShift
      marchUniforms.uFade.value = Math.min((now - startTime) / 1000 / FADE_IN_SECONDS, 1)
      // Shader only uses fract(uTime): wrapping on an integer is seamless and keeps fp32 precise
      blitUniforms.uTime.value = ((now - startTime) / 1000) % 1000

      renderer.setRenderTarget(renderTarget)
      renderer.render(marchScene, camera)
      renderer.setRenderTarget(null)
      renderer.render(blitScene, camera)
    }

    animate()
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      cancelAnimationFrame(animationId)
      quad.dispose()
      marchMaterial.dispose()
      blitMaterial.dispose()
      renderTarget.dispose()
      renderer.dispose()
      container.removeChild(renderer.domElement)
    }
  }, [])

  return <div ref={containerRef} className="w-full h-full" />
}
