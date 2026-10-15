import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js'
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js'
import {
  BLOCK_PITCH,
  DEBRIS_POOL,
  MAP_SIZE,
  MAX_BUILDINGS,
  PIECE_FREE,
  ROAD_WIDTH,
  SIDEWALK,
  TIER_DESTROYED,
  TIER_HEAVY,
  TIER_LIGHT,
  TIER_MODERATE,
  applyHit,
  blackbody,
  buildingLean,
  cloudState,
  createBlast,
  createCloudState,
  createDebris,
  createDust,
  fireballRadius,
  fitFrame,
  generateCity,
  groundShockRadius,
  overpressure,
  planBlast,
  resetDebris,
  resetDust,
  shockArrival,
  shockRadius,
  shockSpeed,
  smoothstep,
  spawnDust,
  stepDebris,
  stepDust,
  thermalPulse,
  windSpeed,
  type Blast,
  type BlastPlan,
  type City,
  type Frame,
} from './atomicBombMath'

// Render budget. The scene itself is cheap per pixel, the raymarched cloud is not, so each
// stage has its own pixel cap (Joe's 4K screen would otherwise ask for 8+ MP per stage).
const MAX_SCENE_PIXELS = 2_400_000
const MAX_CLOUD_PIXELS = 320_000
const MAX_TRANS_PIXELS = 900_000
const MSAA_SAMPLES = 4
const BLOOM_STRENGTH = 0.55
const BLOOM_RADIUS = 0.55
const BLOOM_THRESHOLD = 1.0

// Timeline (seconds)
const FADE_IN_SECONDS = 2.5
const MIN_CALM_SECONDS = 4 // establishing shot of the untouched city
const MAX_CALM_SECONDS = 7
const EVENT_SECONDS = 52 // detonation -> start of the fade out
const FADE_OUT_SECONDS = 3
const PHYSICS_STEP = 1 / 90
const MAX_FRAME_SECONDS = 1 / 20

// Pools
const DUST_COUNT = 3000
const MAX_POINT_PX = 200 // largest dust billow, in render pixels (bounds overdraw)

// Camera
const FOV = 40
const FRAME_MARGIN = 0.88
const MIN_PITCH_DEG = 6.5
const MAX_PITCH_DEG = 10
const ORBIT_DEG_PER_SECOND = 0.45
const LOOKAHEAD_SECONDS = 6 // frame for the cloud as it will be a few seconds from now
const SHAKE_RADIANS = 0.0035

// Fireball light: illuminance at one 5-psi radius, relative to a sun of intensity ~2.5
const FLASH_ILLUMINANCE = 45
const GLOW_ILLUMINANCE = 8

interface Preset {
  zenith: number
  horizon: number
  sunColor: number
  sunIntensity: number
  sunElevation: [number, number] // degrees
  hemiSky: number
  hemiGround: number
  hemiIntensity: number
  fogDensity: number
  cover: number // sky cloud cover
  windowsLit: number // fraction of windows lit
  windowGlow: number
  streetLights: number
  exposure: number
  dust: number
  smoke: number
}

const PRESETS: Preset[] = [
  {
    // dawn
    zenith: 0x2d4a7c, horizon: 0xf0a07c, sunColor: 0xffb47e, sunIntensity: 2.4, sunElevation: [3, 8],
    hemiSky: 0x8aa0c8, hemiGround: 0x3a2e28, hemiIntensity: 0.9, fogDensity: 0.00026, cover: 0.3,
    windowsLit: 0.2, windowGlow: 1.2, streetLights: 0.6, exposure: 1.0, dust: 0x9a8670, smoke: 0xa8a09c,
  },
  {
    // dusk
    zenith: 0x1a2346, horizon: 0xd87a5c, sunColor: 0xff8e58, sunIntensity: 1.7, sunElevation: [1.5, 5],
    hemiSky: 0x5a6a9a, hemiGround: 0x2a2020, hemiIntensity: 0.75, fogDensity: 0.00024, cover: 0.35,
    windowsLit: 0.55, windowGlow: 1.6, streetLights: 1.0, exposure: 1.2, dust: 0x8a7866, smoke: 0x96908e,
  },
  {
    // overcast daylight
    zenith: 0x8a9098, horizon: 0xc2c5c8, sunColor: 0xe6e2da, sunIntensity: 0.8, sunElevation: [28, 42],
    hemiSky: 0xc6ccd4, hemiGround: 0x4a4a46, hemiIntensity: 1.7, fogDensity: 0.0003, cover: 0.9,
    windowsLit: 0.06, windowGlow: 0.8, streetLights: 0, exposure: 1.0, dust: 0x9a9080, smoke: 0xb4b2b0,
  },
  {
    // golden afternoon
    zenith: 0x3f6fb5, horizon: 0xe6cfb0, sunColor: 0xffd9a8, sunIntensity: 2.8, sunElevation: [11, 18],
    hemiSky: 0x9ab4d8, hemiGround: 0x4a3c30, hemiIntensity: 1.0, fogDensity: 0.0002, cover: 0.18,
    windowsLit: 0.04, windowGlow: 0.6, streetLights: 0, exposure: 0.95, dust: 0xa69078, smoke: 0xc0b8b0,
  },
]

// ---------------------------------------------------------------------------------------------
// GLSL

const NOISE_GLSL = /* glsl */ `
float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
float hash13(vec3 p3) {
  p3 = fract(p3 * 0.1031);
  p3 += dot(p3, p3.zyx + 31.32);
  return fract((p3.x + p3.y) * p3.z);
}
float vnoise2(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash12(i), hash12(i + vec2(1.0, 0.0)), f.x),
             mix(hash12(i + vec2(0.0, 1.0)), hash12(i + vec2(1.0, 1.0)), f.x), f.y);
}
float vnoise3(vec3 p) {
  vec3 i = floor(p);
  vec3 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(hash13(i), hash13(i + vec3(1.0, 0.0, 0.0)), f.x),
                 mix(hash13(i + vec3(0.0, 1.0, 0.0)), hash13(i + vec3(1.0, 1.0, 0.0)), f.x), f.y),
             mix(mix(hash13(i + vec3(0.0, 0.0, 1.0)), hash13(i + vec3(1.0, 0.0, 1.0)), f.x),
                 mix(hash13(i + vec3(0.0, 1.0, 1.0)), hash13(i + vec3(1.0, 1.0, 1.0)), f.x), f.y), f.z);
}
float fbm2(vec2 p) {
  float s = 0.0;
  float a = 0.5;
  for (int i = 0; i < 5; i++) {
    s += a * vnoise2(p);
    p = p * 2.03 + vec2(17.1, 3.7);
    a *= 0.5;
  }
  return s / 0.96875;
}
float fbm3(vec3 p, int oct) {
  float s = 0.0;
  float a = 0.5;
  float w = 0.0;
  for (int i = 0; i < 4; i++) {
    if (i >= oct) break;
    s += a * vnoise3(p);
    w += a;
    p = p * 2.02 + vec3(5.3, 1.7, 9.1);
    a *= 0.5;
  }
  return s / w;
}
`

const FULLSCREEN_VERT = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`

// Volumetric fireball + mushroom cloud, raymarched at reduced resolution inside a bounding
// cylinder around ground zero, stopped by the opaque scene's depth. Output is premultiplied
// color with coverage in alpha.
const CLOUD_FRAG = /* glsl */ `
uniform sampler2D uDepth;
uniform mat4 uInvProj;
uniform mat4 uCamWorld;
uniform vec3 uCamPos;
uniform float uNear;
uniform float uFar;
uniform vec3 uGZ;
uniform float uCapH;
uniform float uCapRmin;
uniform float uCapRmaj;
uniform float uSquash;
uniform float uStemTop;
uniform float uStemR;
uniform float uStemDen;
uniform float uSkirtR;
uniform float uSkirtH;
uniform float uSkirtDen;
uniform float uBoundR;
uniform float uBoundH;
uniform float uTime;
uniform float uRoll;
uniform float uTemp;
uniform float uFireball;
uniform float uNox;
uniform vec3 uSunDir;
uniform vec3 uSunCol;
uniform vec3 uSkyCol;
uniform vec3 uGroundCol;
uniform vec3 uFogCol;
uniform float uFogDen;
uniform vec3 uDustCol;
uniform vec3 uSmokeCol;
varying vec2 vUv;
${NOISE_GLSL}
// Planck's law sampled at R, G, B wavelengths (µm), normalized to the brightest channel
vec3 blackbody(float t) {
  vec3 lam = vec3(0.61, 0.55, 0.465);
  vec3 l5 = lam * lam * lam * lam * lam;
  vec3 e = exp(min(vec3(14388.0) / (lam * max(t, 300.0)), vec3(80.0))) - 1.0;
  vec3 b = 1.0 / (l5 * e);
  return b / max(max(b.r, b.g), b.b);
}

float gHeat;
float gDust;

float cloudDensity(vec3 p, int oct) {
  vec3 q = p - uGZ;
  float rho = length(q.xz);
  float capD = 0.0;
  float dustD = 0.0;
  gHeat = 0.0;

  // Cap: a torus whose ring radius grows from zero, so the fireball (a sphere) rolls open into
  // the vortex ring. A dome fills the ring's hole over the stem.
  vec2 tp = vec2(rho - uCapRmaj, (q.y - uCapH) / uSquash);
  float tl = length(tp);
  vec2 dp = vec2(rho / max(uCapRmaj * 0.9 + uCapRmin * 0.55, 1.0),
                 (q.y - uCapH - uCapRmin * 0.2) / max(uCapRmin * 0.75 * uSquash, 1.0));
  float capEdge = max(1.0 - tl / uCapRmin, (1.0 - length(dp)) * 0.8);
  if (capEdge > -0.6) {
    // The noise is carried around the tube's minor circle: out over the top, down the outside
    float c = cos(uRoll);
    float s = sin(uRoll);
    vec2 rp = vec2(c * tp.x - s * tp.y, s * tp.x + c * tp.y);
    vec2 dir = rho > 0.001 ? q.xz / rho : vec2(1.0, 0.0);
    vec3 np = vec3(dir.x * (uCapRmaj + rp.x), rp.y, dir.y * (uCapRmaj + rp.x)) / uCapRmin;
    float n = fbm3(np * 1.7 + vec3(0.0, uTime * 0.03, 0.0), oct);
    float cd = capEdge + (n - 0.5) * mix(1.0, 0.4, uFireball);
    capD = smoothstep(0.0, 0.22, cd);
    gHeat = smoothstep(0.1, 0.85, cd) * (0.7 + 0.6 * n);
  }

  // Stem: dust drawn up from the ground by the afterwind, flared at the base
  if (uStemDen > 0.0 && q.y < uStemTop + uStemR * 2.0) {
    float flare = 1.0 + 2.2 * exp(-q.y / (uStemR * 1.5));
    float widen = 1.0 + 0.9 * smoothstep(uStemTop * 0.6, uStemTop, q.y);
    float r = uStemR * flare * widen;
    if (rho < r * 1.6) {
      vec3 sp = vec3(q.x, q.y - uTime * uStemR * 0.6, q.z) / (uStemR * 1.2);
      float n = fbm3(sp, oct);
      float sd = 1.0 - rho / r + (n - 0.5) * 0.9;
      sd = smoothstep(0.0, 0.4, sd) * uStemDen;
      sd *= 1.0 - smoothstep(uStemTop - uStemR, uStemTop + uStemR * 2.0, q.y);
      dustD = max(dustD, sd);
    }
  }

  // Skirt: the dust raised around ground zero by the shock
  if (uSkirtDen > 0.0 && q.y < uSkirtH * 2.0 && rho < uSkirtR * 1.3) {
    vec3 kp = q / (uSkirtH * 0.9) + vec3(0.0, -uTime * 0.05, 0.0);
    float n = fbm3(kp, oct);
    float hf = 1.0 - q.y / uSkirtH;
    float ef = (1.0 - rho / max(uSkirtR, 1.0)) * 3.0;
    float sd = min(hf, ef) + (n - 0.5);
    dustD = max(dustD, smoothstep(0.0, 0.5, sd) * uSkirtDen);
  }

  float d = max(capD, dustD);
  gDust = dustD / max(capD + dustD, 0.0001);
  return d;
}

float hgPhase(float mu, float g) {
  float g2 = g * g;
  return (1.0 - g2) / pow(1.0 + g2 - 2.0 * g * mu, 1.5);
}

void main() {
  vec2 ndc = vUv * 2.0 - 1.0;
  vec4 vp = uInvProj * vec4(ndc, 1.0, 1.0);
  vec3 dirV = normalize(vp.xyz / vp.w);
  vec3 rd = normalize((uCamWorld * vec4(dirV, 0.0)).xyz);
  vec3 ro = uCamPos;
  float depth = texture2D(uDepth, vUv).x;
  float viewZ = (uNear * uFar) / ((uFar - uNear) * depth - uFar);
  float sceneT = depth >= 0.99999 ? 1e9 : viewZ / dirV.z;

  // Bounding cylinder around ground zero
  vec2 oc = ro.xz - uGZ.xz;
  float a = max(dot(rd.xz, rd.xz), 1e-6);
  float b = dot(oc, rd.xz);
  float c = dot(oc, oc) - uBoundR * uBoundR;
  float h = b * b - a * c;
  if (uBoundR <= 0.0 || h <= 0.0) {
    gl_FragColor = vec4(0.0);
    return;
  }
  h = sqrt(h);
  float t0 = (-b - h) / a;
  float t1 = (-b + h) / a;
  if (abs(rd.y) > 1e-5) {
    float ya = (uGZ.y - ro.y) / rd.y;
    float yb = (uGZ.y + uBoundH - ro.y) / rd.y;
    t0 = max(t0, min(ya, yb));
    t1 = min(t1, max(ya, yb));
  }
  t0 = max(t0, 0.0);
  t1 = min(t1, sceneT);
  if (t1 <= t0) {
    gl_FragColor = vec4(0.0);
    return;
  }

  const int STEPS = 56;
  float dt = (t1 - t0) / float(STEPS);
  // interleaved gradient noise: a fixed per-pixel offset that hides banding without shimmer
  float ign = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
  float t = t0 + dt * ign;
  float T = 1.0;
  vec3 col = vec3(0.0);
  float firstT = -1.0;
  float sigmaK = 5.0 / max(uCapRmin, 1.0);
  vec3 capCenter = uGZ + vec3(0.0, uCapH, 0.0);
  vec3 smoke = mix(uSmokeCol, vec3(0.42, 0.2, 0.1), uNox * 0.7);
  vec3 glowCol = blackbody(uTemp);
  float glowI = 6.0 * pow(uTemp / 6000.0, 3.0);
  float mu = dot(rd, uSunDir);
  float phase = 0.65 * hgPhase(mu, 0.45) + 0.35;
  float lightStep = uCapRmin * 0.3;

  for (int i = 0; i < STEPS; i++) {
    vec3 p = ro + rd * t;
    float dens = cloudDensity(p, 4);
    if (dens > 0.003) {
      if (firstT < 0.0) firstT = t;
      float heat = gHeat;
      float dust = gDust;
      // sunlight reaching this sample, three coarse steps toward the sun
      float od = cloudDensity(p + uSunDir * lightStep, 2) * lightStep;
      od += cloudDensity(p + uSunDir * lightStep * 2.5, 2) * lightStep * 1.5;
      od += cloudDensity(p + uSunDir * lightStep * 5.0, 2) * lightStep * 2.5;
      float sunT = exp(-sigmaK * od);
      vec3 albedo = mix(smoke, uDustCol, dust);
      float up = clamp(0.5 + 0.5 * (p.y - capCenter.y) / max(uCapRmin, 1.0), 0.2, 1.0);
      vec3 amb = mix(uGroundCol, uSkyCol, up) * (1.0 - 0.45 * dens);
      // the hot core lights the smoke around it and the top of the stem from inside
      float dc = length(p - capCenter) / max(uCapRmin, 1.0);
      vec3 fireLit = glowCol * glowI * 0.3 / (1.0 + dc * dc * 1.5);
      vec3 S = albedo * (uSunCol * sunT * phase + amb + fireLit) + glowCol * glowI * heat * heat * (1.0 - dust);
      float Ts = exp(-dens * sigmaK * dt);
      col += T * (1.0 - Ts) * S;
      T *= Ts;
      if (T < 0.01) break;
    }
    t += dt;
  }
  if (firstT > 0.0) {
    float k = uFogDen * firstT;
    col = mix(col, uFogCol * (1.0 - T), 1.0 - exp(-k * k));
  }
  gl_FragColor = vec4(col, 1.0 - T);
}
`

// Opaque scene + cloud + transparent dust, then exposure, the flash whiteout and the fade
const COMPOSITE_FRAG = /* glsl */ `
uniform sampler2D tScene;
uniform sampler2D tCloud;
uniform sampler2D tTrans;
uniform float uExposure;
uniform float uWhite;
uniform vec3 uWhiteCol;
uniform float uFade;
varying vec2 vUv;
void main() {
  vec3 c = texture2D(tScene, vUv).rgb;
  vec4 cl = texture2D(tCloud, vUv);
  c = c * (1.0 - cl.a) + cl.rgb;
  vec4 tr = texture2D(tTrans, vUv);
  c = c * (1.0 - tr.a) + tr.rgb;
  c *= uExposure;
  c = mix(c, uWhiteCol, uWhite);
  vec2 v = vUv - 0.5;
  c *= 1.0 - 0.3 * pow(clamp(dot(v, v) * 1.6, 0.0, 1.0), 1.5);
  gl_FragColor = vec4(c * uFade, 1.0);
}
`

const SKY_VERT = /* glsl */ `
varying vec3 vDir;
void main() {
  vDir = position;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  gl_Position.z = gl_Position.w;
}
`

const SKY_FRAG = /* glsl */ `
uniform vec3 uZenith;
uniform vec3 uHorizon;
uniform vec3 uSunDir;
uniform vec3 uSunCol;
uniform vec3 uFireDir;
uniform vec3 uFireCol;
uniform float uFireI;
uniform float uCover;
uniform float uTime;
uniform float uDrift;
varying vec3 vDir;
${NOISE_GLSL}
void main() {
  vec3 d = normalize(vDir);
  float h = d.y;
  vec3 col = mix(uHorizon, uZenith, pow(clamp(h, 0.0, 1.0), 0.5));
  float sd = max(dot(d, uSunDir), 0.0);
  float clearSky = 1.0 - uCover * 0.8;
  col += uSunCol * (pow(sd, 6.0) * 0.12 + pow(sd, 48.0) * 0.35) * clearSky;
  col += uSunCol * smoothstep(0.99985, 0.99993, sd) * 30.0 * (1.0 - uCover);
  float fd = max(dot(d, uFireDir), 0.0);
  vec3 fireGlow = uFireCol * uFireI;
  if (h > 0.0) {
    // a high cloud deck, lit by the sun and, after the detonation, by the fireball
    vec2 cp = d.xz / (h + 0.08) * 1.6 + vec2(uDrift, uDrift * 0.4) + uTime * vec2(0.004, 0.002);
    float n = fbm2(cp);
    float cov = smoothstep(1.0 - uCover - 0.15, 1.0 - uCover + 0.35, n) * smoothstep(0.0, 0.12, h);
    vec3 base = mix(uHorizon, vec3(dot(uHorizon, vec3(0.3333))), 0.35) * (0.75 + 0.35 * n);
    vec3 cc = base + uSunCol * pow(sd, 4.0) * 0.35 + fireGlow * (0.04 + 0.25 * pow(fd, 3.0));
    col = mix(col, cc, cov * 0.85);
  }
  col = mix(col, uHorizon, smoothstep(0.02, -0.02, h));
  col += fireGlow * (pow(fd, 4.0) * 0.05 + pow(fd, 40.0) * 0.3 + 0.01);
  gl_FragColor = vec4(col, 1.0);
}
`

// Shared by every transparent (dust) shader: soft intersection with the opaque scene, hidden
// behind the cloud when farther than ground zero, and the same exponential fog as the scene
const TRANSPARENT_GLSL = /* glsl */ `
uniform sampler2D uDepth;
uniform sampler2D uCloud;
uniform vec2 uRes;
uniform float uNear;
uniform float uFar;
uniform float uGZDepth;
uniform vec3 uFogCol;
uniform float uFogDen;
uniform vec3 uDustCol;
uniform vec3 uAmb;
uniform vec3 uSunCol;
uniform vec3 uFirePos;
uniform vec3 uFireCol;
uniform float uFireI;
uniform float uFireR2;
uniform float uFireSoft2;
uniform float uTime;
float softFade(float viewZ, float soft) {
  vec2 suv = gl_FragCoord.xy / uRes;
  float d = texture2D(uDepth, suv).x;
  float sceneZ = (uNear * uFar) / (uFar - (uFar - uNear) * d);
  float f = clamp((sceneZ - viewZ) / soft, 0.0, 1.0);
  f *= 1.0 - smoothstep(uGZDepth - 60.0, uGZDepth + 60.0, viewZ) * texture2D(uCloud, suv).a;
  return f;
}
vec3 applyFog(vec3 c, float viewZ) {
  float k = uFogDen * viewZ;
  return mix(c, uFogCol, 1.0 - exp(-k * k));
}
vec3 fireLight(vec3 w) {
  vec3 d = w - uFirePos;
  return uFireCol * uFireI * uFireR2 / (dot(d, d) + uFireSoft2);
}
`

const DUST_VERT = /* glsl */ `
attribute float aSize;
attribute float aAlpha;
attribute float aShade;
uniform float uPxScale;
uniform float uMaxPx;
varying float vAlpha;
varying float vShade;
varying vec3 vWorld;
varying float vViewZ;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  gl_PointSize = clamp(aSize * uPxScale / max(-mv.z, 1.0), 1.0, uMaxPx);
  vAlpha = aAlpha;
  vShade = aShade;
  vWorld = position;
  vViewZ = -mv.z;
}
`

const DUST_FRAG = /* glsl */ `
${TRANSPARENT_GLSL}
${NOISE_GLSL}
varying float vAlpha;
varying float vShade;
varying vec3 vWorld;
varying float vViewZ;
void main() {
  vec2 c = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(c, c);
  if (r2 > 1.0 || vAlpha <= 0.0) discard;
  float n = vnoise2(c * 2.5 + vShade * 37.0);
  float a = (1.0 - r2) * (1.0 - r2) * (0.55 + 0.6 * n) * vAlpha;
  a *= softFade(vViewZ, 15.0);
  float top = 0.6 - 0.4 * c.y; // gl_PointCoord runs downward: lighter on the upper side
  vec3 col = uDustCol * vShade * (uAmb + uSunCol * top * 0.7 + fireLight(vWorld));
  col = applyFog(col, vViewZ);
  gl_FragColor = vec4(col * a, a);
}
`

// Ground-hugging dust wall riding the shock front: a ring whose profile trails behind the front
const WALL_VERT = /* glsl */ `
attribute vec2 aProf;
attribute float aAng;
uniform vec3 uGZ;
uniform float uR;
uniform float uBand;
uniform float uH;
varying vec3 vWorld;
varying float vViewZ;
varying vec2 vProf;
void main() {
  float r = max(uR - aProf.x * uBand, 0.0);
  vec3 w = uGZ + vec3(cos(aAng) * r, aProf.y * uH, sin(aAng) * r);
  vec4 mv = viewMatrix * vec4(w, 1.0);
  gl_Position = projectionMatrix * mv;
  vWorld = w;
  vViewZ = -mv.z;
  vProf = aProf;
}
`

const WALL_FRAG = /* glsl */ `
${TRANSPARENT_GLSL}
${NOISE_GLSL}
uniform float uAlpha;
uniform float uH;
varying vec3 vWorld;
varying float vViewZ;
varying vec2 vProf;
void main() {
  float n = fbm3(vec3(vWorld.xz * 0.03, vWorld.y * 0.05 - uTime * 0.7), 3);
  float body = smoothstep(0.3, 0.75, n + 0.3 * (1.0 - vProf.y) - 0.15 * vProf.x);
  float a = uAlpha * body * (1.0 - smoothstep(0.35, 1.0, vProf.x)) * (1.0 - 0.5 * smoothstep(0.7, 1.0, vProf.y));
  a *= softFade(vViewZ, 8.0);
  float lit = 0.55 + 0.45 * vProf.y;
  vec3 col = uDustCol * (0.65 + 0.5 * n) * (uAmb + uSunCol * lit * 0.6 + fireLight(vWorld));
  col = applyFog(col, vViewZ);
  gl_FragColor = vec4(col * a, a);
}
`

// Wilson condensation cloud: a thin shell of fog briefly marking the shock in humid air
const SHELL_VERT = /* glsl */ `
uniform vec3 uCenter;
uniform float uR;
varying vec3 vWorld;
varying vec3 vN;
varying float vViewZ;
void main() {
  vec3 w = uCenter + position * uR;
  vec4 mv = viewMatrix * vec4(w, 1.0);
  gl_Position = projectionMatrix * mv;
  vWorld = w;
  vN = normalize(position);
  vViewZ = -mv.z;
}
`

const SHELL_FRAG = /* glsl */ `
${TRANSPARENT_GLSL}
${NOISE_GLSL}
uniform float uAlpha;
varying vec3 vWorld;
varying vec3 vN;
varying float vViewZ;
void main() {
  if (vWorld.y < 0.0) discard;
  vec3 v = normalize(cameraPosition - vWorld);
  float rim = 1.0 - abs(dot(vN, v));
  float n = fbm3(vN * 5.0 + vec3(0.0, uTime * 0.15, 0.0), 3);
  float a = uAlpha * pow(rim, 2.2) * smoothstep(0.35, 0.7, n) * smoothstep(0.0, 8.0, vWorld.y);
  a *= softFade(vViewZ, 20.0);
  vec3 col = vec3(0.92, 0.94, 0.96) * (uAmb * 1.3 + uSunCol * 0.5) + fireLight(vWorld) * 0.8;
  col = applyFog(col, vViewZ);
  gl_FragColor = vec4(col * a, a);
}
`

// ---- ground: MeshStandardMaterial patched with roads, blocks, fields, scorch and dust

const GROUND_VERT_DECL = /* glsl */ `
varying vec3 vGround;
`
const GROUND_VERT_BODY = /* glsl */ `
vGround = (modelMatrix * vec4(transformed, 1.0)).xyz;
`
const GROUND_FRAG_DECL = /* glsl */ `
uniform sampler2D uBlockMap;
uniform float uCityCos;
uniform float uCitySin;
uniform vec3 uGZ;
uniform float uScorch;
uniform float uScorchR;
uniform float uDustCover;
uniform float uShockR;
uniform float uDustR;
uniform vec3 uDustTint;
uniform float uGroundGlow;
uniform float uGlowR;
uniform float uStreetLights;
uniform float uFieldAngle;
uniform float uTime;
varying vec3 vGround;
${NOISE_GLSL}
float blockAt(vec2 cell, out float kind) {
  vec2 idx = cell + ${(MAP_SIZE / 2).toFixed(1)};
  if (idx.x < 0.0 || idx.y < 0.0 || idx.x >= ${MAP_SIZE.toFixed(1)} || idx.y >= ${MAP_SIZE.toFixed(1)}) {
    kind = 0.0;
    return 0.0;
  }
  vec4 b = texture2D(uBlockMap, (idx + 0.5) / ${MAP_SIZE.toFixed(1)});
  kind = floor(b.g * 255.0 + 0.5);
  return b.r;
}
`
const GROUND_FRAG_COLOR = /* glsl */ `
vec3 gEmit = vec3(0.0);
{
  const float PITCH = ${BLOCK_PITCH.toFixed(1)};
  const float HW = ${(ROAD_WIDTH / 2).toFixed(1)};
  const float WALK = ${SIDEWALK.toFixed(1)};
  vec2 wp = vGround.xz;
  vec2 lp = vec2(wp.x * uCityCos - wp.y * uCitySin, wp.x * uCitySin + wp.y * uCityCos);
  vec2 cell = floor(lp / PITCH);
  vec2 f = lp - cell * PITCH;
  float kind;
  float isCity = blockAt(cell, kind);
  float dummy;
  vec2 side = vec2(f.x < PITCH * 0.5 ? -1.0 : 1.0, f.y < PITCH * 0.5 ? -1.0 : 1.0);
  vec2 ed = min(f, PITCH - f);
  float nx = blockAt(cell + vec2(side.x, 0.0), dummy);
  float nz = blockAt(cell + vec2(0.0, side.y), dummy);
  float nd = blockAt(cell + side, dummy);
  vec2 aa = max(fwidth(lp), vec2(0.001));
  float roadX = (1.0 - smoothstep(HW - aa.x, HW + aa.x, ed.x)) * max(isCity, nx);
  float roadZ = (1.0 - smoothstep(HW - aa.y, HW + aa.y, ed.y)) * max(isCity, nz);
  float road = max(max(roadX, roadZ), (1.0 - smoothstep(HW - aa.x, HW + aa.x, ed.x)) *
    (1.0 - smoothstep(HW - aa.y, HW + aa.y, ed.y)) * nd);
  float walk = isCity * (1.0 - smoothstep(HW + WALK - aa.x, HW + WALK + aa.x, min(ed.x, ed.y)));

  // Farmland around the town: patchwork fields with furrows
  vec2 fr = vec2(wp.x * cos(uFieldAngle) - wp.y * sin(uFieldAngle), wp.x * sin(uFieldAngle) + wp.y * cos(uFieldAngle));
  vec2 fc = floor(fr / vec2(150.0, 110.0));
  float fh = hash12(fc);
  vec3 field = mix(vec3(0.06, 0.1, 0.03), vec3(0.16, 0.13, 0.06), fh);
  field = mix(field, vec3(0.1, 0.075, 0.045), step(0.78, fh));
  float furrow = 0.88 + 0.12 * sin((fh > 0.5 ? fr.x : fr.y) * 1.3);
  float fn = fbm2(wp * 0.004);
  field *= furrow * (0.75 + 0.5 * fn);
  vec2 fe = abs(fract(fr / vec2(150.0, 110.0)) - 0.5);
  field = mix(field, vec3(0.03, 0.05, 0.02), smoothstep(0.485, 0.5, max(fe.x, fe.y)));

  // City lots by block type
  float ln = vnoise2(wp * 0.15);
  vec3 lot = vec3(0.16, 0.155, 0.15);
  if (kind < 1.5) lot = mix(vec3(0.07, 0.11, 0.035), vec3(0.12, 0.1, 0.07), ln * 0.6);
  else if (kind < 2.5) lot = vec3(0.15, 0.145, 0.14) * (0.9 + 0.2 * ln);
  else if (kind < 3.5) lot = vec3(0.2, 0.19, 0.18) * (0.92 + 0.08 * step(0.5, fract((f.x + f.y) * 0.25)));
  else if (kind < 4.5) lot = mix(vec3(0.05, 0.11, 0.03), vec3(0.08, 0.13, 0.04), ln);
  else lot = vec3(0.13, 0.125, 0.115) * (0.85 + 0.3 * ln);
  vec3 col = mix(field, lot, isCity);
  col = mix(col, vec3(0.2, 0.2, 0.19), walk);
  // asphalt with a dashed center line
  vec3 asphalt = vec3(0.04, 0.04, 0.045) * (0.85 + 0.3 * ln);
  float along = roadX > roadZ ? lp.y : lp.x;
  float lineD = roadX > roadZ ? ed.x : ed.y;
  float dash = (1.0 - smoothstep(0.15, 0.35, lineD)) * step(0.5, fract(along / 9.0));
  asphalt = mix(asphalt, vec3(0.3, 0.27, 0.2), dash * 0.6);
  col = mix(col, asphalt, road);

  // Thermal scorch and the dust left behind the shock
  float gd = length(wp - uGZ.xz);
  float scorch = uScorch * (1.0 - smoothstep(uScorchR * 0.2, uScorchR, gd)) * (0.75 + 0.25 * fn);
  col = mix(col, vec3(0.022, 0.018, 0.015), scorch * 0.9);
  float dusted = uDustCover * (1.0 - smoothstep(uShockR - 40.0, uShockR, gd)) * (1.0 - smoothstep(uDustR * 0.5, uDustR, gd));
  col = mix(col, uDustTint, dusted * 0.45 * (0.6 + 0.4 * fn));
  diffuseColor.rgb = col;

  // Streetlights until the shock cuts the power, and the glowing ground under the fireball
  float lampX = length(vec2(ed.x - (HW - 1.0), (fract(lp.y / 18.0) - 0.5) * 18.0));
  float lampZ = length(vec2(ed.y - (HW - 1.0), (fract(lp.x / 18.0) - 0.5) * 18.0));
  float lamp = (1.0 - smoothstep(0.0, 2.6, min(lampX, lampZ))) * max(isCity, max(nx, nz));
  float power = smoothstep(uShockR - 4.0, uShockR + 4.0, gd);
  gEmit += vec3(1.0, 0.62, 0.3) * lamp * uStreetLights * power * 2.5;
  gEmit += vec3(1.0, 0.32, 0.07) * uGroundGlow * exp(-gd * gd / (uGlowR * uGlowR)) *
    (0.6 + 0.4 * vnoise2(wp * 0.08 + uTime * 0.6));
}
`

const patchGround = (shader: { vertexShader: string; fragmentShader: string }) => {
  shader.vertexShader = shader.vertexShader
    .replace('#include <common>', `#include <common>\n${GROUND_VERT_DECL}`)
    .replace('#include <begin_vertex>', `#include <begin_vertex>\n${GROUND_VERT_BODY}`)
  shader.fragmentShader = shader.fragmentShader
    .replace('#include <common>', `#include <common>\n${GROUND_FRAG_DECL}`)
    .replace('#include <color_fragment>', `#include <color_fragment>\n${GROUND_FRAG_COLOR}`)
    .replace('#include <emissivemap_fragment>', '#include <emissivemap_fragment>\ntotalEmissiveRadiance += gEmit;')
}

// ---- buildings: windows, glass, rubble, thermal char, flash heat, fires

const BUILDING_VERT_DECL = /* glsl */ `
attribute vec4 aInfo;
attribute vec4 aState;
uniform vec3 uFirePos;
varying vec3 vLocal;
varying vec3 vLocalN;
varying vec4 vInfo;
varying vec4 vState;
varying float vFacing;
`
const BUILDING_VERT_BODY = /* glsl */ `
#ifdef USE_INSTANCING
  vec3 iScale = vec3(length(instanceMatrix[0].xyz), length(instanceMatrix[1].xyz), length(instanceMatrix[2].xyz));
  vLocal = position * iScale;
  vec3 bWorld = (modelMatrix * instanceMatrix * vec4(position, 1.0)).xyz;
  vec3 bN = normalize(mat3(modelMatrix) * mat3(instanceMatrix) * normal);
#else
  vLocal = position;
  vec3 bWorld = (modelMatrix * vec4(position, 1.0)).xyz;
  vec3 bN = normalize(mat3(modelMatrix) * normal);
#endif
  vLocalN = normal;
  vInfo = aInfo;
  vState = aState;
  vFacing = dot(bN, normalize(uFirePos - bWorld));
`
const BUILDING_FRAG_DECL = /* glsl */ `
uniform float uTime;
uniform float uWindowGlow;
uniform float uHeat;
uniform float uChar;
uniform float uFires;
uniform vec3 uFireCol;
varying vec3 vLocal;
varying vec3 vLocalN;
varying vec4 vInfo;
varying vec4 vState;
varying float vFacing;
float bhash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}
`
const BUILDING_FRAG_COLOR = /* glsl */ `
float bBroken = vState.x;
float bRubble = vState.y;
float bPower = 1.0 - vState.z;
float bTower = step(1.5, vState.w) * step(vState.w, 2.5);
float roofFace = step(0.5, vLocalN.y);
float faceU = abs(vLocalN.x) > 0.5 ? vLocal.z : vLocal.x;
float colW = mix(3.4, 2.6, bTower);
vec2 wcell = vec2(faceU / colW, vLocal.y / vInfo.w);
vec2 wf = fract(wcell);
vec2 wid = floor(wcell);
float glazing = mix(0.3, 0.42, bTower);
float win = step(abs(wf.x - 0.5), glazing) * step(0.28, wf.y) * step(wf.y, 0.82);
win *= (1.0 - roofFace) * (1.0 - bRubble);
float wHash = bhash(wid + vec2(vInfo.x * 113.0, vLocalN.x * 7.0 + vLocalN.z * 3.0));
vec3 glass = mix(vec3(0.045, 0.055, 0.075), vec3(0.012, 0.01, 0.01), bBroken);
diffuseColor.rgb = mix(diffuseColor.rgb, glass, win);
diffuseColor.rgb *= mix(1.0, 0.72, roofFace);
diffuseColor.rgb *= 0.7 + 0.3 * smoothstep(0.0, 5.0, vLocal.y);
float lump = bhash(floor(vLocal.xz * 0.6) + vInfo.x * 31.0);
diffuseColor.rgb = mix(diffuseColor.rgb, mix(vec3(0.15, 0.13, 0.11), diffuseColor.rgb * 0.45, lump), bRubble);
float facing = clamp(vFacing, 0.0, 1.0);
float charred = vInfo.z * uChar * (0.35 + 0.65 * facing);
diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.03, 0.024, 0.02), charred * 0.85);
`
const BUILDING_FRAG_ROUGH = /* glsl */ `
roughnessFactor = mix(roughnessFactor, 0.12, win * (1.0 - bBroken));
`
const BUILDING_FRAG_EMIT = /* glsl */ `
float litWin = step(wHash, vInfo.y) * bPower * (1.0 - bBroken) * win;
vec3 warm = mix(vec3(1.0, 0.72, 0.42), vec3(0.75, 0.85, 1.0), step(0.85, fract(wHash * 7.3)));
totalEmissiveRadiance += warm * litWin * uWindowGlow;
totalEmissiveRadiance += uFireCol * uHeat * vInfo.z * facing * facing;
float burning = step(fract(wHash * 13.7), vInfo.z * 0.55) * uFires * max(bBroken, bRubble) * max(win, bRubble * 0.6);
float flicker = 0.55 + 0.45 * sin(uTime * (6.0 + 5.0 * wHash) + wHash * 40.0) * sin(uTime * 2.3 + wHash * 17.0);
totalEmissiveRadiance += vec3(3.2, 1.0, 0.24) * burning * flicker;
`

const patchBuilding = (shader: { vertexShader: string; fragmentShader: string }) => {
  shader.vertexShader = shader.vertexShader
    .replace('#include <common>', `#include <common>\n${BUILDING_VERT_DECL}`)
    .replace('#include <begin_vertex>', `#include <begin_vertex>\n${BUILDING_VERT_BODY}`)
  shader.fragmentShader = shader.fragmentShader
    .replace('#include <common>', `#include <common>\n${BUILDING_FRAG_DECL}`)
    .replace('#include <color_fragment>', `#include <color_fragment>\n${BUILDING_FRAG_COLOR}`)
    .replace('#include <roughnessmap_fragment>', `#include <roughnessmap_fragment>\n${BUILDING_FRAG_ROUGH}`)
    .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>\n${BUILDING_FRAG_EMIT}`)
}

// Dust wall cross-section: (distance behind the front in band widths, height fraction)
const WALL_PROFILE: [number, number][] = [
  [0, 0], [0, 0.45], [0.04, 0.8], [0.12, 1], [0.25, 0.92], [0.45, 0.7], [0.7, 0.42], [1, 0.2],
]
const WALL_SEGMENTS = 200

const createWallGeometry = () => {
  const P = WALL_PROFILE.length
  const n = (WALL_SEGMENTS + 1) * P
  const prof = new Float32Array(n * 2)
  const ang = new Float32Array(n)
  for (let s = 0; s <= WALL_SEGMENTS; s++) {
    for (let p = 0; p < P; p++) {
      const i = s * P + p
      prof[2 * i] = WALL_PROFILE[p][0]
      prof[2 * i + 1] = WALL_PROFILE[p][1]
      ang[i] = (s / WALL_SEGMENTS) * Math.PI * 2
    }
  }
  const index: number[] = []
  for (let s = 0; s < WALL_SEGMENTS; s++) {
    for (let p = 0; p < P - 1; p++) {
      const a = s * P + p
      const b = (s + 1) * P + p
      index.push(a, b, a + 1, b, b + 1, a + 1)
    }
  }
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3))
  g.setAttribute('aProf', new THREE.BufferAttribute(prof, 2))
  g.setAttribute('aAng', new THREE.BufferAttribute(ang, 1))
  g.setIndex(index)
  return g
}

// Gable roof: a triangular prism, ridge along x, base on y = 0, apex at y = 1
const createRoofGeometry = () => {
  const v = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z)
  const A = v(-0.5, 0, -0.5), B = v(0.5, 0, -0.5), C = v(0.5, 0, 0.5), D = v(-0.5, 0, 0.5)
  const E = v(-0.5, 1, 0), F = v(0.5, 1, 0)
  const tris = [[A, B, F], [A, F, E], [C, D, E], [C, E, F], [B, C, F], [D, A, E]]
  const pos: number[] = []
  const center = v(0, 0.3, 0)
  const ab = new THREE.Vector3()
  const ac = new THREE.Vector3()
  const mid = new THREE.Vector3()
  for (const [p, q, r] of tris) {
    // wind every face outward
    ab.subVectors(q, p)
    ac.subVectors(r, p)
    mid.copy(p).add(q).add(r).divideScalar(3).sub(center)
    const out = ab.cross(ac).dot(mid) > 0
    for (const w of out ? [p, q, r] : [p, r, q]) pos.push(w.x, w.y, w.z)
  }
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
  g.computeVertexNormals()
  return g
}

const randRange = (a: number, b: number) => a + (b - a) * Math.random()

export default function AtomicBomb() {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!containerRef.current) return
    const container = containerRef.current

    const renderer = new THREE.WebGLRenderer({ antialias: false })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(window.innerWidth, window.innerHeight)
    renderer.setClearColor(0x000000, 0)
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFShadowMap
    container.appendChild(renderer.domElement)

    const camera = new THREE.PerspectiveCamera(FOV, window.innerWidth / window.innerHeight, 2, 40000)

    // ---- render targets: opaque scene (MSAA + depth), cloud and dust at reduced resolution
    const sceneTarget = new THREE.WebGLRenderTarget(1, 1, {
      type: THREE.HalfFloatType,
      samples: MSAA_SAMPLES,
      depthTexture: new THREE.DepthTexture(1, 1),
    })
    const cloudTarget = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, depthBuffer: false })
    const transTarget = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, depthBuffer: false })

    const compositeMaterial = new THREE.ShaderMaterial({
      vertexShader: FULLSCREEN_VERT,
      fragmentShader: COMPOSITE_FRAG,
      uniforms: {
        tScene: { value: sceneTarget.texture },
        tCloud: { value: cloudTarget.texture },
        tTrans: { value: transTarget.texture },
        uExposure: { value: 1 },
        uWhite: { value: 0 },
        uWhiteCol: { value: new THREE.Color(6, 5.6, 5.2) },
        uFade: { value: 0 },
      },
      depthTest: false,
      depthWrite: false,
    })
    const composer = new EffectComposer(renderer)
    const compositePass = new ShaderPass(compositeMaterial)
    const bloomPass = new UnrealBloomPass(new THREE.Vector2(1, 1), BLOOM_STRENGTH, BLOOM_RADIUS, BLOOM_THRESHOLD)
    const outputPass = new OutputPass()
    composer.addPass(compositePass)
    composer.addPass(bloomPass)
    composer.addPass(outputPass)

    // ---- opaque scene
    const scene = new THREE.Scene()
    scene.fog = new THREE.FogExp2(0x888888, 0.00025)

    const skyUniforms = {
      uZenith: { value: new THREE.Color() },
      uHorizon: { value: new THREE.Color() },
      uSunDir: { value: new THREE.Vector3(0, 1, 0) },
      uSunCol: { value: new THREE.Color() },
      uFireDir: { value: new THREE.Vector3(0, 1, 0) },
      uFireCol: { value: new THREE.Color() },
      uFireI: { value: 0 },
      uCover: { value: 0.3 },
      uTime: { value: 0 },
      uDrift: { value: 0 },
    }
    const skyGeometry = new THREE.SphereGeometry(1, 48, 24)
    const skyMaterial = new THREE.ShaderMaterial({
      vertexShader: SKY_VERT,
      fragmentShader: SKY_FRAG,
      uniforms: skyUniforms,
      side: THREE.BackSide,
      depthWrite: false,
      depthTest: false,
    })
    const sky = new THREE.Mesh(skyGeometry, skyMaterial)
    sky.scale.setScalar(15000)
    sky.renderOrder = -1
    sky.frustumCulled = false
    scene.add(sky)

    const hemi = new THREE.HemisphereLight(0xffffff, 0x444444, 1)
    scene.add(hemi)
    const sun = new THREE.DirectionalLight(0xffffff, 2)
    sun.castShadow = true
    sun.shadow.mapSize.set(2048, 2048)
    sun.shadow.bias = -0.0004
    sun.shadow.normalBias = 0.6
    scene.add(sun)
    scene.add(sun.target)
    // The fireball as a light: harsh radiating shadows during the flash, warm glow after
    const fireLight = new THREE.PointLight(0xffffff, 0, 0, 2)
    fireLight.castShadow = true
    fireLight.shadow.mapSize.set(1024, 1024)
    fireLight.shadow.camera.near = 2
    fireLight.shadow.camera.far = 6000
    fireLight.shadow.bias = -0.002
    fireLight.shadow.normalBias = 0.4
    // Render the cube shadow map once up front: autoUpdate is off until detonation, and with no
    // map three binds a plain cube texture to the samplerCubeShadow (draws fail in WebGL2)
    fireLight.shadow.needsUpdate = true
    scene.add(fireLight)

    // Ground
    const blockData = new Uint8Array(MAP_SIZE * MAP_SIZE * 4)
    const blockMap = new THREE.DataTexture(blockData, MAP_SIZE, MAP_SIZE, THREE.RGBAFormat)
    blockMap.magFilter = THREE.NearestFilter
    blockMap.minFilter = THREE.NearestFilter
    blockMap.needsUpdate = true
    const groundUniforms = {
      uBlockMap: { value: blockMap },
      uCityCos: { value: 1 },
      uCitySin: { value: 0 },
      uGZ: { value: new THREE.Vector3() },
      uScorch: { value: 0 },
      uScorchR: { value: 1 },
      uDustCover: { value: 0 },
      uShockR: { value: 0 },
      uDustR: { value: 1 },
      uDustTint: { value: new THREE.Color() },
      uGroundGlow: { value: 0 },
      uGlowR: { value: 1 },
      uStreetLights: { value: 0 },
      uFieldAngle: { value: 0 },
      uTime: { value: 0 },
    }
    const groundGeometry = new THREE.PlaneGeometry(30000, 30000).rotateX(-Math.PI / 2)
    const groundMaterial = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.95, metalness: 0 })
    groundMaterial.onBeforeCompile = (shader) => {
      Object.assign(shader.uniforms, groundUniforms)
      patchGround(shader)
    }
    groundMaterial.customProgramCacheKey = () => 'atomic-ground'
    const ground = new THREE.Mesh(groundGeometry, groundMaterial)
    ground.receiveShadow = true
    scene.add(ground)

    // Buildings: one instance each, unit box with its base on y = 0
    const buildingGeometry = new THREE.BoxGeometry(1, 1, 1).translate(0, 0.5, 0)
    const infoAttr = new THREE.InstancedBufferAttribute(new Float32Array(MAX_BUILDINGS * 4), 4)
    const stateAttr = new THREE.InstancedBufferAttribute(new Float32Array(MAX_BUILDINGS * 4), 4)
    stateAttr.setUsage(THREE.DynamicDrawUsage)
    buildingGeometry.setAttribute('aInfo', infoAttr)
    buildingGeometry.setAttribute('aState', stateAttr)
    const buildingUniforms = {
      uFirePos: { value: new THREE.Vector3(0, -1e4, 0) },
      uTime: { value: 0 },
      uWindowGlow: { value: 1 },
      uHeat: { value: 0 },
      uChar: { value: 0 },
      uFires: { value: 0 },
      uFireCol: { value: new THREE.Color() },
    }
    const buildingMaterial = new THREE.MeshStandardMaterial({ roughness: 0.82, metalness: 0.05 })
    buildingMaterial.onBeforeCompile = (shader) => {
      Object.assign(shader.uniforms, buildingUniforms)
      patchBuilding(shader)
    }
    buildingMaterial.customProgramCacheKey = () => 'atomic-building'
    const buildings = new THREE.InstancedMesh(buildingGeometry, buildingMaterial, MAX_BUILDINGS)
    buildings.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
    buildings.castShadow = true
    buildings.receiveShadow = true
    buildings.frustumCulled = false
    scene.add(buildings)

    const roofGeometry = createRoofGeometry()
    const roofMaterial = new THREE.MeshStandardMaterial({ roughness: 0.85, metalness: 0, flatShading: true })
    const roofs = new THREE.InstancedMesh(roofGeometry, roofMaterial, MAX_BUILDINGS)
    roofs.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
    roofs.castShadow = true
    roofs.receiveShadow = true
    roofs.frustumCulled = false
    scene.add(roofs)
    const roofOf = new Int32Array(MAX_BUILDINGS) // building -> roof instance, -1 if flat

    const debrisGeometry = new THREE.BoxGeometry(1, 1, 1)
    const debrisMaterial = new THREE.MeshStandardMaterial({ roughness: 0.9, metalness: 0 })
    const debrisMesh = new THREE.InstancedMesh(debrisGeometry, debrisMaterial, DEBRIS_POOL)
    debrisMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
    debrisMesh.castShadow = true
    debrisMesh.receiveShadow = true
    debrisMesh.frustumCulled = false
    debrisMesh.count = 0
    scene.add(debrisMesh)
    // allocate the instance color buffer up front
    debrisMesh.setColorAt(0, new THREE.Color(1, 1, 1))

    // ---- cloud pass
    const quadGeometry = new THREE.PlaneGeometry(2, 2)
    const quadCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1)
    const cloudUniforms = {
      uDepth: { value: sceneTarget.depthTexture },
      uInvProj: { value: new THREE.Matrix4() },
      uCamWorld: { value: new THREE.Matrix4() },
      uCamPos: { value: new THREE.Vector3() },
      uNear: { value: camera.near },
      uFar: { value: camera.far },
      uGZ: { value: new THREE.Vector3() },
      uCapH: { value: 0 },
      uCapRmin: { value: 1 },
      uCapRmaj: { value: 0 },
      uSquash: { value: 1 },
      uStemTop: { value: 0 },
      uStemR: { value: 1 },
      uStemDen: { value: 0 },
      uSkirtR: { value: 0 },
      uSkirtH: { value: 1 },
      uSkirtDen: { value: 0 },
      uBoundR: { value: 0 },
      uBoundH: { value: 0 },
      uTime: { value: 0 },
      uRoll: { value: 0 },
      uTemp: { value: 1000 },
      uFireball: { value: 0 },
      uNox: { value: 0 },
      uSunDir: { value: new THREE.Vector3(0, 1, 0) },
      uSunCol: { value: new THREE.Color() },
      uSkyCol: { value: new THREE.Color() },
      uGroundCol: { value: new THREE.Color() },
      uFogCol: { value: new THREE.Color() },
      uFogDen: { value: 0 },
      uDustCol: { value: new THREE.Color() },
      uSmokeCol: { value: new THREE.Color() },
    }
    const cloudMaterial = new THREE.ShaderMaterial({
      vertexShader: FULLSCREEN_VERT,
      fragmentShader: CLOUD_FRAG,
      uniforms: cloudUniforms,
      depthTest: false,
      depthWrite: false,
    })
    const cloudScene = new THREE.Scene()
    const cloudQuad = new THREE.Mesh(quadGeometry, cloudMaterial)
    cloudQuad.frustumCulled = false
    cloudScene.add(cloudQuad)

    // ---- transparent dust (own target, depth-tested by hand against the scene depth)
    const transUniforms = {
      uDepth: { value: sceneTarget.depthTexture },
      uCloud: { value: cloudTarget.texture },
      uRes: { value: new THREE.Vector2(1, 1) },
      uNear: { value: camera.near },
      uFar: { value: camera.far },
      uGZDepth: { value: 1e5 },
      uFogCol: { value: new THREE.Color() },
      uFogDen: { value: 0 },
      uDustCol: { value: new THREE.Color() },
      uAmb: { value: new THREE.Color() },
      uSunCol: { value: new THREE.Color() },
      uFirePos: { value: new THREE.Vector3() },
      uFireCol: { value: new THREE.Color() },
      uFireI: { value: 0 },
      uFireR2: { value: 1 },
      uFireSoft2: { value: 1 },
      uTime: { value: 0 },
    }
    const transScene = new THREE.Scene()
    const transparentMaterial = (vertexShader: string, fragmentShader: string, extra: Record<string, THREE.IUniform>) =>
      new THREE.ShaderMaterial({
        vertexShader,
        fragmentShader,
        uniforms: { ...transUniforms, ...extra },
        transparent: true,
        premultipliedAlpha: true,
        blending: THREE.NormalBlending,
        depthTest: false,
        depthWrite: false,
        side: THREE.DoubleSide,
      })

    const dust = createDust(DUST_COUNT)
    const dustGeometry = new THREE.BufferGeometry()
    const dustPos = new THREE.BufferAttribute(new Float32Array(DUST_COUNT * 3), 3)
    const dustSize = new THREE.BufferAttribute(new Float32Array(DUST_COUNT), 1)
    const dustAlpha = new THREE.BufferAttribute(new Float32Array(DUST_COUNT), 1)
    const dustShade = new THREE.BufferAttribute(new Float32Array(DUST_COUNT), 1)
    for (const a of [dustPos, dustSize, dustAlpha, dustShade]) a.setUsage(THREE.DynamicDrawUsage)
    dustGeometry.setAttribute('position', dustPos)
    dustGeometry.setAttribute('aSize', dustSize)
    dustGeometry.setAttribute('aAlpha', dustAlpha)
    dustGeometry.setAttribute('aShade', dustShade)
    const dustMaterial = transparentMaterial(DUST_VERT, DUST_FRAG, {
      uPxScale: { value: 1 },
      uMaxPx: { value: MAX_POINT_PX },
    })
    const dustPoints = new THREE.Points(dustGeometry, dustMaterial)
    dustPoints.frustumCulled = false
    transScene.add(dustPoints)

    const wallGeometry = createWallGeometry()
    const wallMaterial = transparentMaterial(WALL_VERT, WALL_FRAG, {
      uGZ: { value: new THREE.Vector3() },
      uR: { value: 0 },
      uBand: { value: 1 },
      uH: { value: 1 },
      uAlpha: { value: 0 },
    })
    const wall = new THREE.Mesh(wallGeometry, wallMaterial)
    wall.frustumCulled = false
    transScene.add(wall)

    const shellGeometry = new THREE.SphereGeometry(1, 64, 32)
    const shellMaterial = transparentMaterial(SHELL_VERT, SHELL_FRAG, {
      uCenter: { value: new THREE.Vector3() },
      uR: { value: 0 },
      uAlpha: { value: 0 },
    })
    const shell = new THREE.Mesh(shellGeometry, shellMaterial)
    shell.frustumCulled = false
    transScene.add(shell)

    // ---- sizing
    let sceneW = 1
    let sceneH = 1
    let transH = 1
    const handleResize = () => {
      const dpr = Math.min(window.devicePixelRatio, 2)
      const W = window.innerWidth * dpr
      const H = window.innerHeight * dpr
      const s = Math.min(1, Math.sqrt(MAX_SCENE_PIXELS / Math.max(1, W * H)))
      sceneW = Math.max(1, Math.floor(W * s))
      sceneH = Math.max(1, Math.floor(H * s))
      const cs = Math.min(0.5, Math.sqrt(MAX_CLOUD_PIXELS / (sceneW * sceneH)))
      const ts = Math.min(0.75, Math.sqrt(MAX_TRANS_PIXELS / (sceneW * sceneH)))
      const transW = Math.max(1, Math.floor(sceneW * ts))
      transH = Math.max(1, Math.floor(sceneH * ts))
      renderer.setSize(window.innerWidth, window.innerHeight)
      sceneTarget.setSize(sceneW, sceneH)
      cloudTarget.setSize(Math.max(1, Math.floor(sceneW * cs)), Math.max(1, Math.floor(sceneH * cs)))
      transTarget.setSize(transW, transH)
      transUniforms.uRes.value.set(transW, transH)
      composer.setPixelRatio(1)
      composer.setSize(sceneW, sceneH)
      camera.aspect = window.innerWidth / window.innerHeight
      camera.updateProjectionMatrix()
      dustMaterial.uniforms.uPxScale.value = transH / (2 * Math.tan(THREE.MathUtils.degToRad(FOV / 2)))
    }
    handleResize()

    // ---- per-cycle state
    let city: City = generateCity(Math.random)
    let blast: Blast = createBlast(Math.random, city)
    let plan: BlastPlan = planBlast(city, blast, Math.random)
    let preset = PRESETS[0]
    const debris = createDebris(DEBRIS_POOL)
    const cloud = createCloudState()
    const futureCloud = createCloudState()
    let cycleTime = 0
    let detonateAt = 0
    let nextHit = 0
    let azimuth = 0
    let orbitRate = 0
    let pitch = 0
    let frameDistance = 0
    let frameLift = 0
    let framed = false
    let exposure = 1
    let frontDustDebt = 0
    let stemDustDebt = 0
    let physicsDebt = 0
    let simTime = -1 // time since detonation, advanced in fixed physics steps
    const sunDir = new THREE.Vector3()
    const sunColor = new THREE.Color()
    const hemiSkyColor = new THREE.Color()
    const hemiGroundColor = new THREE.Color()
    const horizonColor = new THREE.Color()
    const dustColor = new THREE.Color()

    const color = new THREE.Color()
    const matrix = new THREE.Matrix4()
    const local = new THREE.Matrix4()
    const pos = new THREE.Vector3()
    const scl = new THREE.Vector3()
    const quat = new THREE.Quaternion()
    const yawQuat = new THREE.Quaternion()
    const leanQuat = new THREE.Quaternion()
    const roofQuat = new THREE.Quaternion()
    const axis = new THREE.Vector3()
    const up = new THREE.Vector3(0, 1, 0)
    const zero = new THREE.Matrix4().makeScale(0, 0, 0)
    const bb = [0, 0, 0]
    const frame: Frame = { distance: 0, lift: 0, fx: 0, fy: 0, fz: 0, ux: 0, uy: 0, uz: 0 }
    const framePoints: number[] = []
    const firePos = new THREE.Vector3()
    const gzView = new THREE.Vector3()

    const setBuildingState = (i: number) => {
      const bd = city.buildings[i]
      const tier = bd.hit ? bd.tier : 0
      stateAttr.setXYZW(
        i,
        tier >= TIER_LIGHT ? 1 : 0,
        tier === TIER_DESTROYED ? 1 : 0,
        bd.hit && (tier >= TIER_LIGHT || bd.seed < 0.7) ? 1 : 0, // the shock takes the power lines down
        bd.kind,
      )
    }

    const startCycle = () => {
      city = generateCity(Math.random)
      blast = createBlast(Math.random, city)
      plan = planBlast(city, blast, Math.random)
      preset = PRESETS[Math.floor(Math.random() * PRESETS.length)]

      // Sky and light for this time of day
      const elev = THREE.MathUtils.degToRad(randRange(preset.sunElevation[0], preset.sunElevation[1]))
      const sunAz = Math.random() * Math.PI * 2
      sunDir.set(Math.cos(elev) * Math.cos(sunAz), Math.sin(elev), Math.cos(elev) * Math.sin(sunAz))
      sunColor.setHex(preset.sunColor)
      hemiSkyColor.setHex(preset.hemiSky)
      hemiGroundColor.setHex(preset.hemiGround)
      horizonColor.setHex(preset.horizon)
      dustColor.setHex(preset.dust)
      skyUniforms.uZenith.value.setHex(preset.zenith)
      skyUniforms.uHorizon.value.copy(horizonColor)
      skyUniforms.uSunDir.value.copy(sunDir)
      skyUniforms.uSunCol.value.copy(sunColor)
      skyUniforms.uCover.value = preset.cover
      skyUniforms.uDrift.value = Math.random() * 100
      ;(scene.fog as THREE.FogExp2).color.copy(horizonColor)
      ;(scene.fog as THREE.FogExp2).density = preset.fogDensity
      hemi.color.copy(hemiSkyColor)
      hemi.groundColor.copy(hemiGroundColor)
      hemi.intensity = preset.hemiIntensity
      sun.color.copy(sunColor)
      sun.intensity = preset.sunIntensity
      const reach = city.radius + 150
      sun.position.copy(sunDir).multiplyScalar(3000)
      sun.target.position.set(0, 0, 0)
      const sc = sun.shadow.camera
      sc.left = -reach
      sc.right = reach
      sc.top = reach
      sc.bottom = -reach
      sc.near = 100
      sc.far = 6000
      sc.updateProjectionMatrix()

      cloudUniforms.uSunDir.value.copy(sunDir)
      cloudUniforms.uSunCol.value.copy(sunColor).multiplyScalar(preset.sunIntensity / Math.PI)
      cloudUniforms.uSkyCol.value.copy(hemiSkyColor).multiplyScalar(preset.hemiIntensity / Math.PI)
      cloudUniforms.uGroundCol.value.copy(hemiGroundColor).multiplyScalar(preset.hemiIntensity / Math.PI)
      cloudUniforms.uFogCol.value.copy(horizonColor)
      cloudUniforms.uFogDen.value = preset.fogDensity
      cloudUniforms.uDustCol.value.copy(dustColor)
      cloudUniforms.uSmokeCol.value.setHex(preset.smoke)
      cloudUniforms.uGZ.value.set(blast.gx, 0, blast.gz)
      transUniforms.uFogCol.value.copy(horizonColor)
      transUniforms.uFogDen.value = preset.fogDensity
      transUniforms.uDustCol.value.copy(dustColor)
      transUniforms.uAmb.value.copy(hemiSkyColor).lerp(hemiGroundColor, 0.35).multiplyScalar(preset.hemiIntensity / Math.PI)
      transUniforms.uSunCol.value.copy(sunColor).multiplyScalar(preset.sunIntensity / Math.PI)
      transUniforms.uFireR2.value = blast.r5 * blast.r5
      transUniforms.uFireSoft2.value = blast.rf * blast.rf
      wallMaterial.uniforms.uGZ.value.set(blast.gx, 0, blast.gz)
      wallMaterial.uniforms.uBand.value = blast.r5 * 0.25
      shellMaterial.uniforms.uCenter.value.set(blast.gx, blast.hb, blast.gz)

      groundUniforms.uCityCos.value = city.cos
      groundUniforms.uCitySin.value = city.sin
      groundUniforms.uGZ.value.set(blast.gx, 0, blast.gz)
      groundUniforms.uScorchR.value = blast.thermalR * 0.6
      groundUniforms.uDustR.value = blast.r5 * 2.4
      groundUniforms.uDustTint.value.copy(dustColor).multiplyScalar(0.5)
      groundUniforms.uGlowR.value = blast.rf * 1.3
      groundUniforms.uStreetLights.value = preset.streetLights
      groundUniforms.uFieldAngle.value = Math.random() * Math.PI
      blockData.set(city.map)
      blockMap.needsUpdate = true
      buildingUniforms.uWindowGlow.value = preset.windowGlow

      // Instances
      let roofCount = 0
      for (let i = 0; i < city.buildings.length; i++) {
        const bd = city.buildings[i]
        color.setRGB(bd.color[0], bd.color[1], bd.color[2], THREE.SRGBColorSpace)
        buildings.setColorAt(i, color)
        infoAttr.setXYZW(i, bd.seed, Math.min(1, preset.windowsLit * randRange(0.4, 1.5)), bd.burn, bd.floorH)
        setBuildingState(i)
        if (bd.roofH > 0) {
          roofOf[i] = roofCount
          color.setRGB(bd.roofColor[0], bd.roofColor[1], bd.roofColor[2], THREE.SRGBColorSpace)
          roofs.setColorAt(roofCount, color)
          roofCount++
        } else roofOf[i] = -1
      }
      buildings.count = city.buildings.length
      roofs.count = roofCount
      infoAttr.needsUpdate = true
      stateAttr.needsUpdate = true
      if (buildings.instanceColor) buildings.instanceColor.needsUpdate = true
      if (roofs.instanceColor) roofs.instanceColor.needsUpdate = true
      yawQuat.setFromAxisAngle(up, city.angle)

      resetDebris(debris)
      debrisMesh.count = 0
      resetDust(dust)
      cycleTime = 0
      detonateAt = FADE_IN_SECONDS + randRange(MIN_CALM_SECONDS, MAX_CALM_SECONDS)
      simTime = -detonateAt
      physicsDebt = 0
      nextHit = 0
      frontDustDebt = 0
      stemDustDebt = 0
      azimuth = Math.random() * Math.PI * 2
      orbitRate = THREE.MathUtils.degToRad(ORBIT_DEG_PER_SECOND) * (Math.random() < 0.5 ? -1 : 1)
      pitch = THREE.MathUtils.degToRad(randRange(MIN_PITCH_DEG, MAX_PITCH_DEG))
      framed = false
      exposure = 1
    }
    startCycle()

    const spawnHitDust = (i: number) => {
      const bd = city.buildings[i]
      if (bd.tier < TIER_MODERATE) return
      const n = bd.tier === TIER_DESTROYED ? 3 : bd.tier === TIER_HEAVY ? 2 : 1
      const size = Math.max(bd.w, bd.d)
      const push = windSpeed(bd.psi) * 0.12
      for (let k = 0; k < n; k++) {
        spawnDust(
          dust,
          bd.x + randRange(-0.3, 0.3) * bd.w,
          bd.h * randRange(0.15, 0.6),
          bd.z + randRange(-0.3, 0.3) * bd.d,
          bd.dirX * push * randRange(0.5, 1.2),
          randRange(1, 4),
          bd.dirZ * push * randRange(0.5, 1.2),
          randRange(7, 12),
          size * 1.2,
          size * randRange(2.5, 4) + bd.h * 0.4,
          randRange(0.75, 1.1),
        )
      }
    }

    let lastTime = performance.now()
    let animationId = 0

    const animate = () => {
      animationId = requestAnimationFrame(animate)
      const now = performance.now()
      const frameSeconds = Math.min((now - lastTime) / 1000, MAX_FRAME_SECONDS)
      lastTime = now
      cycleTime += frameSeconds
      const cycleEnd = detonateAt + EVENT_SECONDS + FADE_OUT_SECONDS
      if (cycleTime >= cycleEnd) startCycle()

      // ---- physics in fixed steps: shock arrivals, debris
      physicsDebt += frameSeconds
      let structural = false
      while (physicsDebt >= PHYSICS_STEP) {
        physicsDebt -= PHYSICS_STEP
        simTime += PHYSICS_STEP
        while (nextHit < plan.order.length && city.buildings[plan.order[nextHit]].hitTime <= simTime) {
          const i = plan.order[nextHit++]
          applyHit(city.buildings[i], blast, city, debris, Math.random)
          setBuildingState(i)
          spawnHitDust(i)
          structural = true
        }
        if (simTime > 0) stepDebris(debris, blast, simTime, PHYSICS_STEP)
      }
      if (structural) stateAttr.needsUpdate = true
      const t = simTime

      // ---- blast state
      cloudState(blast, t, cloud)
      const pulse = thermalPulse(t)
      const R = shockRadius(blast, t)
      const groundR = groundShockRadius(blast, t)
      const frontPsi = t > 0 ? overpressure(blast, Math.max(R, 1)) : 0
      const fireR = fireballRadius(blast, t)
      firePos.set(blast.gx, t > 0 ? cloud.capH : blast.hb, blast.gz)
      blackbody(t > 0 ? cloud.temp : 6000, bb)
      const glowFade = 1 - smoothstep(25, 40, t)
      // grows with the fireball's surface as it blooms, cools as T² (stylized, not T⁴, so the
      // dull red glow still lights the town for a while)
      const fireIllum = t > 0
        ? (FLASH_ILLUMINANCE * pulse.light + GLOW_ILLUMINANCE * (cloud.temp / 6000) ** 2 * glowFade) *
          (0.3 + 0.7 * fireR / blast.rf)
        : 0
      fireLight.position.copy(firePos)
      fireLight.color.setRGB(bb[0], bb[1], bb[2])
      fireLight.intensity = fireIllum * blast.r5 * blast.r5
      fireLight.shadow.autoUpdate = fireIllum > 0.05

      // ---- dust sources
      if (t > 0 && groundR > 0 && frontPsi > 0.5) {
        frontDustDebt += frameSeconds * Math.min(1.5, Math.max(0.25, frontPsi / 5)) * 260
        const speed = Math.min(shockSpeed(blast, t), 80)
        while (frontDustDebt >= 1) {
          frontDustDebt -= 1
          const a = Math.random() * Math.PI * 2
          const r = Math.max(0, groundR - Math.random() * 12)
          const size = blast.r5 * 0.04 * Math.min(2, Math.sqrt(frontPsi / 3) + 0.4)
          spawnDust(dust, blast.gx + Math.cos(a) * r, randRange(0, 4), blast.gz + Math.sin(a) * r,
            Math.cos(a) * speed * 0.25, randRange(2, 6), Math.sin(a) * speed * 0.25,
            randRange(5, 9), size, size * randRange(2, 3.2), randRange(0.8, 1.15))
        }
      }
      if (t > 2 && t < 40) {
        stemDustDebt += frameSeconds * 28
        while (stemDustDebt >= 1) {
          stemDustDebt -= 1
          const a = Math.random() * Math.PI * 2
          const r = blast.r5 * randRange(0.2, 0.9)
          const size = blast.r5 * 0.05
          spawnDust(dust, blast.gx + Math.cos(a) * r, randRange(0, 6), blast.gz + Math.sin(a) * r,
            0, randRange(0.5, 2), 0, randRange(8, 12), size, size * 2.4, randRange(0.7, 1))
        }
      }
      stepDust(dust, blast, cloud, t, frameSeconds)

      // ---- camera: frame the city and (a few seconds ahead) the cloud, slow orbit, shake
      azimuth += orbitRate * frameSeconds
      const cx = blast.gx * 0.5
      const cz = blast.gz * 0.5
      framePoints.length = 0
      for (let k = 0; k < 24; k++) {
        const a = (k / 24) * Math.PI * 2
        framePoints.push(Math.cos(a) * city.radius, 0, Math.sin(a) * city.radius)
      }
      framePoints.push(city.downtownX, city.maxHeight, city.downtownZ)
      if (t > -1) {
        cloudState(blast, Math.max(0.5, t + LOOKAHEAD_SECONDS), futureCloud)
        const ext = futureCloud.capRmaj + futureCloud.capRmin
        const top = futureCloud.capH + futureCloud.capRmin * futureCloud.squash
        for (let k = 0; k < 8; k++) {
          const a = (k / 8) * Math.PI * 2
          framePoints.push(blast.gx + Math.cos(a) * ext, futureCloud.capH, blast.gz + Math.sin(a) * ext)
        }
        framePoints.push(blast.gx, top * 1.05, blast.gz)
      }
      const tanV = Math.tan(THREE.MathUtils.degToRad(FOV / 2)) * FRAME_MARGIN
      fitFrame(framePoints, cx, 0, cz, azimuth, pitch, tanV, tanV * camera.aspect, frame)
      if (!framed) {
        frameDistance = frame.distance
        frameLift = frame.lift
        framed = true
      } else {
        // never push back in during an event; ease out toward what the cloud will need
        const k = 1 - Math.exp(-frameSeconds / 2.5)
        frameDistance += (Math.max(frame.distance, t > 0 ? frameDistance : 0) - frameDistance) * k
        frameLift += (frame.lift - frameLift) * k
      }
      camera.position.set(
        cx - frame.fx * frameDistance + frame.ux * frameLift,
        -frame.fy * frameDistance + frame.uy * frameLift,
        cz - frame.fz * frameDistance + frame.uz * frameLift,
      )
      camera.lookAt(camera.position.x + frame.fx, camera.position.y + frame.fy, camera.position.z + frame.fz)
      // The shock reaches the camera long after the flash: a short, subtle jolt
      const camGround = Math.hypot(camera.position.x - blast.gx, camera.position.z - blast.gz)
      const camSlant = Math.hypot(camGround, camera.position.y - blast.hb)
      const tauCam = t - shockArrival(blast, camSlant)
      if (tauCam > 0 && tauCam < 4) {
        const amp = SHAKE_RADIANS * Math.min(2, Math.sqrt(overpressure(blast, camSlant) / 0.15)) *
          (1 - Math.exp(-tauCam / 0.04)) * Math.exp(-tauCam / 0.9)
        camera.rotateX(amp * (Math.sin(tauCam * 53) + 0.6 * Math.sin(tauCam * 97 + 1.3)))
        camera.rotateY(amp * 0.7 * (Math.sin(tauCam * 61 + 2.1) + 0.5 * Math.sin(tauCam * 113)))
      }
      camera.updateMatrixWorld()
      sky.position.copy(camera.position)

      // ---- exposure: the camera stops down for the fireball, then slowly recovers
      const targetExposure = preset.exposure / (1 + 0.06 * fireIllum)
      const adapt = targetExposure < exposure ? 1 - Math.exp(-frameSeconds / 0.25) : 1 - Math.exp(-frameSeconds / 2.5)
      exposure += (targetExposure - exposure) * adapt
      compositeMaterial.uniforms.uExposure.value = exposure
      compositeMaterial.uniforms.uWhite.value = pulse.white
      compositeMaterial.uniforms.uFade.value =
        smoothstep(0, FADE_IN_SECONDS, cycleTime) * (1 - smoothstep(cycleEnd - FADE_OUT_SECONDS, cycleEnd, cycleTime))

      // ---- uniforms
      const ct = cycleTime
      skyUniforms.uTime.value = ct
      skyUniforms.uFireDir.value.copy(firePos).sub(camera.position).normalize()
      skyUniforms.uFireCol.value.setRGB(bb[0], bb[1], bb[2])
      skyUniforms.uFireI.value = fireIllum * 0.25
      groundUniforms.uTime.value = ct
      groundUniforms.uScorch.value = smoothstep(0, 1.5, t)
      groundUniforms.uDustCover.value = smoothstep(0.3, 4, t)
      groundUniforms.uShockR.value = t > 0 ? groundR : -100
      groundUniforms.uGroundGlow.value = t > 0
        ? smoothstep(0, 0.3, t) * (blast.groundBurst ? 3 * Math.exp(-t / 12) : 1.2 * Math.exp(-t / 5))
        : 0
      buildingUniforms.uTime.value = ct
      buildingUniforms.uFirePos.value.copy(firePos)
      buildingUniforms.uFireCol.value.setRGB(bb[0], bb[1], bb[2])
      buildingUniforms.uHeat.value = t > 0 ? pulse.light * 2 + 1.2 * Math.exp(-t / 2) : 0
      buildingUniforms.uChar.value = smoothstep(0, 1.5, t)
      buildingUniforms.uFires.value = smoothstep(3, 10, t)

      cloudUniforms.uInvProj.value.copy(camera.projectionMatrixInverse)
      cloudUniforms.uCamWorld.value.copy(camera.matrixWorld)
      cloudUniforms.uCamPos.value.copy(camera.position)
      cloudUniforms.uCapH.value = cloud.capH
      cloudUniforms.uCapRmin.value = Math.max(cloud.capRmin, 1)
      cloudUniforms.uCapRmaj.value = cloud.capRmaj
      cloudUniforms.uSquash.value = cloud.squash
      cloudUniforms.uStemTop.value = cloud.stemTop
      cloudUniforms.uStemR.value = Math.max(cloud.stemR, 1)
      cloudUniforms.uStemDen.value = cloud.stemDen
      cloudUniforms.uSkirtR.value = cloud.skirtR
      cloudUniforms.uSkirtH.value = Math.max(cloud.skirtH, 1)
      cloudUniforms.uSkirtDen.value = cloud.skirtDen
      cloudUniforms.uBoundR.value = cloud.active ? cloud.boundR : 0
      cloudUniforms.uBoundH.value = cloud.boundH
      cloudUniforms.uTime.value = Math.max(0, t)
      cloudUniforms.uRoll.value = cloud.roll
      cloudUniforms.uTemp.value = cloud.active ? cloud.temp : 1000
      cloudUniforms.uFireball.value = cloud.fireball
      cloudUniforms.uNox.value = cloud.nox

      gzView.set(blast.gx, cloud.stemTop * 0.5, blast.gz).applyMatrix4(camera.matrixWorldInverse)
      transUniforms.uGZDepth.value = cloud.active ? -gzView.z : 1e5
      transUniforms.uTime.value = ct
      transUniforms.uFirePos.value.copy(firePos)
      transUniforms.uFireCol.value.setRGB(bb[0], bb[1], bb[2])
      transUniforms.uFireI.value = (fireIllum / Math.PI) * 0.6
      wallMaterial.uniforms.uR.value = groundR
      wallMaterial.uniforms.uH.value = THREE.MathUtils.clamp(blast.r5 * 0.06 * Math.sqrt(frontPsi / 5), 6, blast.r5 * 0.12)
      wallMaterial.uniforms.uAlpha.value = t > 0 ? smoothstep(0.35, 1.5, frontPsi) * smoothstep(0.05, 0.5, t) * 0.85 : 0
      shellMaterial.uniforms.uR.value = R
      shellMaterial.uniforms.uAlpha.value = t > 0
        ? blast.humidity * smoothstep(0.15, 0.5, t) * (1 - smoothstep(1.2, 3.2, t)) * 0.6
        : 0

      // ---- instances
      for (let i = 0; i < city.buildings.length; i++) {
        const bd = city.buildings[i]
        const lean = buildingLean(bd, t)
        axis.set(bd.dirZ, 0, -bd.dirX) // tilts the top away from ground zero
        leanQuat.setFromAxisAngle(axis, lean)
        quat.multiplyQuaternions(leanQuat, yawQuat)
        pos.set(bd.x, 0, bd.z)
        scl.set(bd.w, Math.max(bd.curH, 0.05), bd.d)
        matrix.compose(pos, quat, scl)
        buildings.setMatrixAt(i, matrix)
        const ri = roofOf[i]
        if (ri >= 0) {
          if (bd.roofOn) {
            matrix.compose(pos, quat, scl.set(1, 1, 1))
            const ridgeAlongX = bd.w >= bd.d
            roofQuat.setFromAxisAngle(up, ridgeAlongX ? 0 : Math.PI / 2)
            local.compose(
              pos.set(0, bd.curH, 0),
              roofQuat,
              scl.set((ridgeAlongX ? bd.w : bd.d) * 1.08, bd.roofH, (ridgeAlongX ? bd.d : bd.w) * 1.08),
            )
            matrix.multiply(local)
            roofs.setMatrixAt(ri, matrix)
          } else roofs.setMatrixAt(ri, zero)
        }
      }
      buildings.instanceMatrix.needsUpdate = true
      roofs.instanceMatrix.needsUpdate = true

      for (let i = 0; i < debris.count; i++) {
        if (debris.state[i] === PIECE_FREE) {
          debrisMesh.setMatrixAt(i, zero)
          continue
        }
        pos.set(debris.px[i], debris.py[i], debris.pz[i])
        quat.set(debris.qx[i], debris.qy[i], debris.qz[i], debris.qw[i])
        scl.set(debris.sx[i], debris.sy[i], debris.sz[i])
        matrix.compose(pos, quat, scl)
        debrisMesh.setMatrixAt(i, matrix)
        if (i >= debrisMesh.count) {
          color.setRGB(debris.cr[i], debris.cg[i], debris.cb[i], THREE.SRGBColorSpace)
          debrisMesh.setColorAt(i, color)
          if (debrisMesh.instanceColor) debrisMesh.instanceColor.needsUpdate = true
        }
      }
      debrisMesh.count = debris.count
      debrisMesh.instanceMatrix.needsUpdate = true

      const dp = dustPos.array as Float32Array
      const ds = dustSize.array as Float32Array
      const da = dustAlpha.array as Float32Array
      const dsh = dustShade.array as Float32Array
      for (let i = 0; i < DUST_COUNT; i++) {
        const life = dust.life[i]
        if (life <= 0) {
          da[i] = 0
          ds[i] = 0
          continue
        }
        const u = dust.age[i] / life
        dp[3 * i] = dust.px[i]
        dp[3 * i + 1] = dust.py[i]
        dp[3 * i + 2] = dust.pz[i]
        ds[i] = dust.size0[i] + (dust.size1[i] - dust.size0[i]) * Math.sqrt(u)
        da[i] = smoothstep(0, 0.06, u) * (1 - smoothstep(0.5, 1, u)) * 0.5
        dsh[i] = dust.shade[i]
      }
      dustPos.needsUpdate = true
      dustSize.needsUpdate = true
      dustAlpha.needsUpdate = true
      dustShade.needsUpdate = true

      // ---- render: opaque scene, cloud, dust, then composite/bloom/output
      renderer.setRenderTarget(sceneTarget)
      renderer.render(scene, camera)
      renderer.setRenderTarget(cloudTarget)
      renderer.render(cloudScene, quadCamera)
      renderer.setRenderTarget(transTarget)
      renderer.render(transScene, camera)
      renderer.setRenderTarget(null)
      composer.render(frameSeconds)
    }

    animate()
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      cancelAnimationFrame(animationId)
      buildings.dispose()
      roofs.dispose()
      debrisMesh.dispose()
      for (const g of [skyGeometry, groundGeometry, buildingGeometry, roofGeometry, debrisGeometry, quadGeometry,
        dustGeometry, wallGeometry, shellGeometry]) g.dispose()
      for (const m of [skyMaterial, groundMaterial, buildingMaterial, roofMaterial, debrisMaterial, cloudMaterial,
        dustMaterial, wallMaterial, shellMaterial, compositeMaterial]) m.dispose()
      blockMap.dispose()
      sun.shadow.dispose()
      fireLight.shadow.dispose()
      sceneTarget.depthTexture?.dispose()
      sceneTarget.dispose()
      cloudTarget.dispose()
      transTarget.dispose()
      compositePass.dispose()
      bloomPass.dispose()
      outputPass.dispose()
      composer.dispose()
      renderer.dispose()
      container.removeChild(renderer.domElement)
    }
  }, [])

  return <div ref={containerRef} className="w-full h-full" />
}
