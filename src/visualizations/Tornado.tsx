import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import {
  MAX_BODIES,
  MAX_SHARDS,
  PHYSICS_STEP,
  PH_AFTER,
  PH_DONE,
  SCOUR_EXTENT,
  SCOUR_SIZE,
  SHAPE_BOX,
  SHAPE_COUNT,
  TILE_POOL,
  TOWN_RADIUS,
  TRI_POOL,
  centerAt,
  chaseCamPose,
  createChaseCam,
  createWorld,
  generateTown,
  newStorm,
  paintScour,
  stepDust,
  stepStorm,
  stepWorld,
  updateChaseCam,
  writeInstanceMatrix,
  writeShardMatrix,
  type ChaseCam,
  type Storm,
} from './tornadoMath'

// Configuration constants (physics, damage and town tuning live in tornadoMath.ts)
const MAX_RENDER_PIXELS = 4_200_000 // drawing-buffer budget: a 4K screen renders at ~0.7x
const MIN_PIXEL_RATIO = 0.45 // floor for the automatic quality ratchet on slow GPUs
const SLOW_FRAME_SECONDS = 1 / 45 // average frame time that triggers a resolution drop
const MAX_FRAME_SECONDS = 1 / 20 // a backgrounded tab resumes with one short step, not a leap
const MAX_SUBSTEPS = 3
const FADE_SECONDS = 2.5 // fade to black between towns (applied to the final image)
const BASE_EXPOSURE = 1.25
const MSAA_SAMPLES = 4
const FOV = 55
const FOG_DENSITY = 0.0016 // FogExp2: ~20% at 300 m, gone by ~1.2 km
const HEMI_INTENSITY = 2.6
const SUN_INTENSITY = 2.4
const SUN_ELEVATION_DEG = 11 // low light raking in under the storm's edge
const FUNNEL_PARTICLES = 2600
const DEBRIS_PARTICLES = 1200
const CLOUD_RADIUS = 3600
const CLOUD_RINGS = 72
const CLOUD_SEGMENTS = 128
const GROUND_SIZE = 9000
const LIGHTNING_MIN_GAP = 3 // seconds between flashes
const LIGHTNING_MAX_GAP = 10
const BOLT_CHANCE = 0.45 // otherwise the flash stays inside the cloud
const BOLT_POINTS = 34
const BOLT_BRANCH_POINTS = 12
const BOLT_WIDTH = 2.6

// Sky moods, sRGB. Each cycle picks one.
interface Mood {
  zenith: number
  horizon: number
  glow: number
  funnel: number
  debris: number
  hemiSky: number
  hemiGround: number
  sun: number
  cloudDark: number
  cloudLight: number
}

const MOODS: Mood[] = [
  // Green supercell: the bruised green-grey light
  { zenith: 0x1f2a26, horizon: 0x6c7a68, glow: 0xd9c48a, funnel: 0x9aa29c, debris: 0x6b5a48,
    hemiSky: 0xa9b8ad, hemiGround: 0x4a4636, sun: 0xffdcaa, cloudDark: 0x161d1b, cloudLight: 0x46534b },
  // Steel blue
  { zenith: 0x1d222b, horizon: 0x6f7984, glow: 0xe6d6b0, funnel: 0xa3a8b0, debris: 0x6e5f50,
    hemiSky: 0xaab4c2, hemiGround: 0x45443c, sun: 0xfff0d0, cloudDark: 0x15191f, cloudLight: 0x454d5a },
  // Amber evening
  { zenith: 0x29211f, horizon: 0x856b55, glow: 0xffb060, funnel: 0x968a80, debris: 0x74563c,
    hemiSky: 0xb8a898, hemiGround: 0x4a4030, sun: 0xffc488, cloudDark: 0x1f1917, cloudLight: 0x55463d },
  // Bruised violet-grey
  { zenith: 0x231f28, horizon: 0x726d76, glow: 0xe8c89a, funnel: 0x9d97a0, debris: 0x6a5848,
    hemiSky: 0xb0a8b4, hemiGround: 0x474038, sun: 0xffe0b8, cloudDark: 0x1a161d, cloudLight: 0x4a4350 },
]

// ---------------------------------------------------------------------------------------------
// Shaders. Shared sky/fog code so the ground, cloud base and funnel all fade into the same sky.

const SKY_GLSL = /* glsl */ `
uniform vec3 uZenith;
uniform vec3 uHorizon;
uniform vec3 uGlow;
uniform vec3 uGlowDir;
uniform float uFogDensity;
uniform float uFlash;
uniform vec3 uFlashColor;

vec3 skyColor(vec3 d) {
  float el = d.y;
  vec3 col = mix(uHorizon, uZenith, smoothstep(-0.02, 0.4, el));
  float hl = length(d.xz);
  float g = hl > 1e-4 ? max(dot(d.xz / hl, uGlowDir.xz), 0.0) : 0.0;
  // the bright slot of clear air under the storm's edge, hugging the horizon
  col += uGlow * (0.22 + 0.78 * pow(g, 3.0)) * exp(-abs(el - 0.015) * 28.0);
  col += uFlashColor * uFlash * 0.1;
  return col;
}

float fogFactor(float dist) {
  float f = dist * uFogDensity;
  return 1.0 - exp(-f * f);
}

vec3 applyFog(vec3 col, vec3 worldPos) {
  vec3 v = worldPos - cameraPosition;
  float dist = length(v);
  return mix(col, skyColor(v / max(dist, 1e-3)), fogFactor(dist));
}
`

const NOISE_GLSL = /* glsl */ `
float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

float vnoise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash12(i), hash12(i + vec2(1.0, 0.0)), u.x),
             mix(hash12(i + vec2(0.0, 1.0)), hash12(i + vec2(1.0, 1.0)), u.x), u.y);
}

float fbm(vec2 p) {
  float s = 0.0;
  float a = 0.5;
  for (int i = 0; i < 5; i++) {
    s += a * vnoise(p);
    p = mat2(1.6, 1.2, -1.2, 1.6) * p;
    a *= 0.5;
  }
  return s;
}
`

const SKY_VERT = /* glsl */ `
varying vec3 vDir;
void main() {
  vDir = position;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`

const SKY_FRAG = /* glsl */ `
uniform vec3 uFlashDir;
varying vec3 vDir;
${SKY_GLSL}
void main() {
  vec3 d = normalize(vDir);
  vec3 col = skyColor(d);
  // a distant flash lights its patch of sky
  col += uFlashColor * uFlash * 0.5 * pow(max(dot(d, uFlashDir), 0.0), 6.0);
  gl_FragColor = vec4(col, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`

const WORLD_VERT = /* glsl */ `
varying vec3 vWorld;
void main() {
  vec4 w = modelMatrix * vec4(position, 1.0);
  vWorld = w.xyz;
  gl_Position = projectionMatrix * viewMatrix * w;
}
`

const GROUND_FRAG = /* glsl */ `
uniform sampler2D uScour;
uniform float uScourExtent;
#define SCOUR_STRENGTH 0.2
uniform vec3 uAmbient;
uniform vec3 uSunColor;
uniform vec3 uSunDir;
uniform float uRoadAngle;
uniform float uFieldSeed;
uniform float uTownRadius;
uniform float uTime;
varying vec3 vWorld;
${SKY_GLSL}
${NOISE_GLSL}

// Furrows/crop rows that fade out before they can alias into moire
float rows(float coord, float freq) {
  float phase = coord * freq;
  float fade = 1.0 - smoothstep(0.4, 1.4, fwidth(phase));
  return 0.5 + 0.5 * sin(phase) * fade;
}

void main() {
  vec2 p = vWorld.xz;
  float ca = cos(uRoadAngle);
  float sa = sin(uRoadAngle);
  vec2 q = vec2(ca * p.x + sa * p.y, -sa * p.x + ca * p.y); // road-aligned
  float big = fbm(p * 0.006 + uFieldSeed);
  float fine = fbm(p * 0.11);
  vec3 grass = mix(vec3(0.040, 0.068, 0.020), vec3(0.075, 0.098, 0.030), big) * (0.8 + 0.4 * fine);

  // Patchwork of fields aligned with the section roads
  vec2 cell = floor(q / 160.0);
  vec2 fq = fract(q / 160.0);
  float h = hash12(cell + uFieldSeed);
  float along = hash12(cell + 7.13 + uFieldSeed) < 0.5 ? q.x : q.y;
  vec3 crop;
  if (h < 0.3) {
    crop = grass * (0.9 + 0.25 * fbm(p * 0.03));
  } else if (h < 0.48) {
    crop = mix(vec3(0.20, 0.15, 0.06), vec3(0.26, 0.20, 0.09), fine) * (0.9 + 0.1 * rows(along, 3.0));
  } else if (h < 0.66) {
    crop = vec3(0.075, 0.052, 0.034) * (0.75 + 0.45 * rows(along, 4.0)) * (0.85 + 0.3 * fine);
  } else if (h < 0.86) {
    crop = mix(vec3(0.07, 0.05, 0.035), vec3(0.05, 0.10, 0.025), rows(along, 3.5)) * (0.85 + 0.3 * fine);
  } else {
    crop = vec3(0.15, 0.12, 0.07) * (0.85 + 0.3 * fine);
  }
  float edge = min(min(fq.x, 1.0 - fq.x), min(fq.y, 1.0 - fq.y)) * 160.0;
  crop = mix(grass, crop, smoothstep(2.0, 4.0, edge));
  // Mowed lawns in town
  float town = 1.0 - smoothstep(uTownRadius + 10.0, uTownRadius + 45.0, length(p));
  vec3 albedo = mix(crop, grass * 1.15, town);

  // Scoured ground: grass stripped to wet dirt, streaked
  float s = texture2D(uScour, p / uScourExtent + 0.5).r;
  vec3 soil = vec3(0.07, 0.05, 0.036) * (0.65 + 0.7 * fbm(p * 0.25));
  albedo = mix(albedo, soil, SCOUR_STRENGTH * smoothstep(0.06, 0.65, s)); // a faint track, not a brown stripe

  // Slow shadows of the lower cloud layer
  float shade = 0.72 + 0.4 * fbm(p * 0.0022 + uTime * vec2(0.012, 0.005));
  vec3 light = uAmbient * shade + uSunColor * max(uSunDir.y, 0.0) * shade;
  vec3 col = albedo * (light + uFlashColor * uFlash * 0.35);
  gl_FragColor = vec4(applyFog(col, vWorld), 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`

const CLOUD_VERT = /* glsl */ `
uniform float uBaseH;
uniform float uWallDepth;
uniform float uWallRadius;
uniform vec2 uCenter;
uniform float uTime;
varying vec3 vWorld;
varying float vWall;
void main() {
  vec4 w = modelMatrix * vec4(position, 1.0);
  float r = length(w.xz - uCenter);
  float wall = exp(-pow(r / uWallRadius, 2.0));
  // a broad lowering around the wall cloud plus a gentle undulation
  float lower = exp(-pow(r / (uWallRadius * 3.0), 2.0)) * 0.35;
  w.y = uBaseH - uWallDepth * (wall + lower) + 9.0 * sin(w.x * 0.0035 + uTime * 0.03) * sin(w.z * 0.0045);
  vWorld = w.xyz;
  vWall = wall;
  gl_Position = projectionMatrix * viewMatrix * w;
}
`

const CLOUD_FRAG = /* glsl */ `
uniform vec2 uCenter;
uniform float uTime;
uniform float uSwirl;
uniform vec3 uCloudDark;
uniform vec3 uCloudLight;
uniform vec2 uFlashPos;
varying vec3 vWorld;
varying float vWall;
${SKY_GLSL}
${NOISE_GLSL}
void main() {
  vec2 d = vWorld.xz - uCenter;
  float r = length(d);
  // The mesocyclone: the cloud base turns counterclockwise, fastest near the funnel
  float ang = uSwirl * exp(-r / 650.0);
  float c = cos(ang);
  float s = sin(ang);
  vec2 q = uCenter + vec2(c * d.x - s * d.y, s * d.x + c * d.y);
  vec2 warp = vec2(fbm(q * 0.003 + 3.1), fbm(q * 0.003 + 7.7));
  float n = fbm(q * 0.0045 + warp * 1.8 + vec2(uTime * 0.008, 0.0));
  float scud = fbm(q * 0.018 + vec2(uTime * 0.025, uTime * 0.01));
  vec3 col = mix(uCloudDark, uCloudLight, smoothstep(0.25, 0.8, n));
  col *= 0.8 + 0.4 * scud;
  col *= 1.0 - 0.45 * vWall;
  // clear-slot light catching the underside toward the horizon
  vec3 v = vWorld - cameraPosition;
  float dist = length(v);
  vec2 hd = v.xz / max(length(v.xz), 1e-3);
  col += uGlow * 0.2 * pow(max(dot(hd, uGlowDir.xz), 0.0), 2.0) * smoothstep(400.0, 2600.0, dist);
  // lightning lighting the cloud from inside
  col += uFlashColor * uFlash * exp(-length(vWorld.xz - uFlashPos) / 650.0) * (0.35 + 0.9 * n);
  gl_FragColor = vec4(applyFog(col, vWorld), 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`

// Every funnel particle's position is a pure function of time, so the whole funnel and its
// debris cloud cost the CPU nothing per frame. Always one funnel: no satellite vortices.
const FUNNEL_VERT = /* glsl */ `
uniform float uTime;
uniform float uH;
uniform vec2 uBase;
uniform vec2 uLean;
uniform float uWobble;
uniform vec2 uWobblePh;
uniform float uRB;
uniform float uRT;
uniform float uRc;
uniform float uBottom;
uniform float uVis;
uniform float uDebris;
uniform float uPointScale;
uniform float uMaxPoint;
uniform vec3 uAmbient;
uniform vec3 uSunColor;
uniform vec3 uSunDir;
uniform vec3 uFunnelTint;
uniform vec3 uDebrisTint;
attribute vec4 aSeed;
attribute float aType;
varying vec4 vColor;
varying vec4 vFog;
varying vec2 vRot;
${SKY_GLSL}

#define FUNNEL_SLIM 0.7

// Mirrors centerAt() in tornadoMath.ts
vec2 centerAt(float y) {
  float h = clamp(y / uH, 0.0, 1.2);
  return uBase + uLean * pow(h, 1.4)
    + uWobble * h * vec2(sin(y * 0.021 + uTime * 0.9 + uWobblePh.x), sin(y * 0.017 - uTime * 0.7 + uWobblePh.y));
}

void main() {
  vec3 wp;
  vec2 radial;
  float size;
  float alpha;
  vec3 tint;
  if (aType < 0.5) {
    // condensation funnel: a shell of puffs spiraling upward
    float h01 = fract(aSeed.x + uTime * (0.035 + 0.04 * aSeed.y));
    float y = h01 * uH;
    // Slimmer than the wind core's visual radius and tightly banded, so the funnel reads crisp
    float r = (uRB + (uRT - uRB) * pow(h01, 2.5)) * FUNNEL_SLIM * (0.85 + 0.2 * aSeed.w);
    float omega = (16.0 + 10.0 * aSeed.y) / max(r, 3.0);
    float a = aSeed.z * 6.2832 - uTime * omega;
    radial = vec2(cos(a), sin(a));
    vec2 c = centerAt(y);
    wp = vec3(c.x + radial.x * r, y, c.y + radial.y * r);
    size = min(r * 0.5 + 3.0, 30.0); // smaller puffs = sharper edge; capped against overdraw
    alpha = 0.42 * smoothstep(uBottom, uBottom + 30.0, y) * smoothstep(0.0, 0.05, h01) * (1.0 - smoothstep(0.92, 1.0, h01)) * uVis;
    float dirt = (1.0 - smoothstep(0.0, 55.0, y)) * uDebris;
    tint = mix(uFunnelTint, uDebrisTint, dirt * 0.75);
  } else {
    // debris cloud: dirt and splinters thrown up and out around the base
    float ph = fract(aSeed.x + uTime * (0.07 + 0.06 * aSeed.y));
    float r = uRc * (0.8 + 1.7 * aSeed.w) * (0.85 + 0.5 * ph);
    float y = pow(ph, 0.8) * (14.0 + 50.0 * aSeed.y * aSeed.y);
    float omega = (22.0 + 10.0 * aSeed.y) / max(r, 4.0);
    float a = aSeed.z * 6.2832 - uTime * omega;
    radial = vec2(cos(a), sin(a));
    vec2 c = centerAt(y);
    wp = vec3(c.x + radial.x * r, y, c.y + radial.y * r);
    size = 7.0 + 13.0 * ph + 0.2 * r;
    alpha = 0.26 * uDebris * sin(ph * 3.1416);
    tint = uDebrisTint;
  }
  // side facing the low light is brighter
  float lit = 0.5 + 0.5 * dot(radial, normalize(uSunDir.xz + vec2(1e-4)));
  vec3 col = tint * (uAmbient * 0.9 + uSunColor * lit * 0.8 + uFlashColor * uFlash * 0.4);
  vec4 mv = viewMatrix * vec4(wp, 1.0);
  gl_Position = projectionMatrix * mv;
  gl_PointSize = alpha > 0.002 ? clamp(size * uPointScale / max(-mv.z, 1.0), 1.0, uMaxPoint) : 0.0;
  vec3 v = wp - cameraPosition;
  float dist = length(v);
  vFog = vec4(skyColor(v / max(dist, 1e-3)), fogFactor(dist));
  vColor = vec4(col, alpha);
  float rot = aSeed.z * 40.0 + uTime * (aSeed.y - 0.5);
  vRot = vec2(cos(rot), sin(rot));
}
`

// Dust puffs advected on the CPU (tornadoMath stepDust)
const DUST_VERT = /* glsl */ `
uniform float uPointScale;
uniform float uMaxPoint;
uniform vec3 uAmbient;
uniform vec3 uSunColor;
uniform vec3 uDebrisTint;
attribute float aSize;
attribute float aAlpha;
varying vec4 vColor;
varying vec4 vFog;
varying vec2 vRot;
${SKY_GLSL}
void main() {
  vec4 mv = viewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  gl_PointSize = aSize > 0.0 ? clamp(aSize * uPointScale / max(-mv.z, 1.0), 1.0, uMaxPoint) : 0.0;
  vec3 col = uDebrisTint * 1.1 * (uAmbient + uSunColor * 0.45 + uFlashColor * uFlash * 0.4);
  vColor = vec4(col, aAlpha * 0.4);
  vec3 v = position - cameraPosition;
  float dist = length(v);
  vFog = vec4(skyColor(v / max(dist, 1e-3)), fogFactor(dist));
  float rot = position.x * 0.07 + position.z * 0.05 + position.y * 0.03;
  vRot = vec2(cos(rot), sin(rot));
}
`

const PUFF_FRAG = /* glsl */ `
uniform sampler2D uPuff;
varying vec4 vColor;
varying vec4 vFog;
varying vec2 vRot;
void main() {
  vec2 uv = gl_PointCoord - 0.5;
  uv = vec2(vRot.x * uv.x - vRot.y * uv.y, vRot.y * uv.x + vRot.x * uv.y) + 0.5;
  float a = texture2D(uPuff, uv).r * vColor.a;
  if (a < 0.004) discard;
  gl_FragColor = vec4(mix(vColor.rgb, vFog.rgb, vFog.a), a);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`

// Final pass: the whole scene is rendered linear/HDR into a render target, then tone mapped,
// encoded and faded here, so the fade scales every output pixel by the same factor (built-in
// materials add fog after tone mapping, so fading by exposure left fogged roads glowing)
const POST_VERT = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = position.xy * 0.5 + 0.5;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`

const POST_FRAG = /* glsl */ `
uniform sampler2D uScene;
uniform float uFade;
varying vec2 vUv;
void main() {
  gl_FragColor = vec4(texture2D(uScene, vUv).rgb, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
  gl_FragColor.rgb *= uFade;
}
`

// ---------------------------------------------------------------------------------------------
// Geometry helpers

// One leaf: a small triangle lying in the x-z plane
function makeLeafGeometry() {
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.Float32BufferAttribute([-0.5, 0, -0.29, 0.5, 0, -0.29, 0, 0, 0.58], 3))
  geometry.computeVertexNormals()
  return geometry
}

// Unit triangular prism: triangle in x-y (apex up), extruded along z
function makeWedgeGeometry() {
  const A0 = [-0.5, -0.5, -0.5], B0 = [0.5, -0.5, -0.5], C0 = [0, 0.5, -0.5]
  const A1 = [-0.5, -0.5, 0.5], B1 = [0.5, -0.5, 0.5], C1 = [0, 0.5, 0.5]
  const tris = [
    A1, B1, C1, // front
    A0, C0, B0, // back
    A0, B0, B1, A0, B1, A1, // bottom
    B0, C0, C1, B0, C1, B1, // right slope
    C0, A0, A1, C0, A1, C1, // left slope
  ]
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(tris.flat(), 3))
  geometry.computeVertexNormals()
  return geometry
}

// Polar grid, dense near the center where the wall cloud hangs
function makeCloudGeometry() {
  const positions: number[] = [0, 0, 0]
  const index: number[] = []
  for (let i = 1; i <= CLOUD_RINGS; i++) {
    const r = CLOUD_RADIUS * Math.pow(i / CLOUD_RINGS, 2)
    for (let j = 0; j < CLOUD_SEGMENTS; j++) {
      const a = (j / CLOUD_SEGMENTS) * Math.PI * 2
      positions.push(Math.cos(a) * r, 0, Math.sin(a) * r)
    }
  }
  const ring = (i: number, j: number) => 1 + (i - 1) * CLOUD_SEGMENTS + (j % CLOUD_SEGMENTS)
  for (let j = 0; j < CLOUD_SEGMENTS; j++) index.push(0, ring(1, j + 1), ring(1, j))
  for (let i = 1; i < CLOUD_RINGS; i++) {
    for (let j = 0; j < CLOUD_SEGMENTS; j++) {
      index.push(ring(i, j), ring(i, j + 1), ring(i + 1, j + 1), ring(i, j), ring(i + 1, j + 1), ring(i + 1, j))
    }
  }
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  geometry.setIndex(index)
  return geometry
}

// Soft, lumpy cloud puff (a cluster of gaussian blobs), zero at the edges
function makePuffTexture() {
  const N = 64
  const data = new Uint8Array(N * N)
  const blobs = Array.from({ length: 9 }, () => ({
    x: (Math.random() - 0.5) * 0.34,
    y: (Math.random() - 0.5) * 0.34,
    r: 0.13 + Math.random() * 0.13,
  }))
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const u = (x + 0.5) / N - 0.5
      const v = (y + 0.5) / N - 0.5
      let a = 0
      for (const b of blobs) a += Math.exp((-((u - b.x) ** 2 + (v - b.y) ** 2) / (b.r * b.r)) * 2)
      const edge = 1 - THREE.MathUtils.smoothstep(Math.hypot(u, v), 0.3, 0.5)
      data[y * N + x] = Math.round(Math.min(1, a * 0.5) * edge * 255)
    }
  }
  const texture = new THREE.DataTexture(data, N, N, THREE.RedFormat, THREE.UnsignedByteType)
  texture.minFilter = THREE.LinearMipmapLinearFilter
  texture.magFilter = THREE.LinearFilter
  texture.generateMipmaps = true
  texture.needsUpdate = true
  return texture
}

export default function Tornado() {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!containerRef.current) return
    const container = containerRef.current

    // Antialiasing happens in the multisampled scene target; the canvas only gets the final pass
    const renderer = new THREE.WebGLRenderer({ antialias: false })
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = BASE_EXPOSURE
    container.appendChild(renderer.domElement)
    const sceneTarget = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: MSAA_SAMPLES })
    const postGeometry = new THREE.BufferGeometry()
    postGeometry.setAttribute('position', new THREE.Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3))
    const postMaterial = new THREE.ShaderMaterial({
      vertexShader: POST_VERT,
      fragmentShader: POST_FRAG,
      uniforms: { uScene: { value: sceneTarget.texture }, uFade: { value: 0 } },
      depthTest: false,
      depthWrite: false,
    })
    const postQuad = new THREE.Mesh(postGeometry, postMaterial)
    postQuad.frustumCulled = false
    const postScene = new THREE.Scene()
    postScene.add(postQuad)
    const postCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1)

    const scene = new THREE.Scene()
    scene.fog = new THREE.FogExp2(0x6c7a68, FOG_DENSITY)
    const camera = new THREE.PerspectiveCamera(FOV, window.innerWidth / window.innerHeight, 1, 9000)

    const hemi = new THREE.HemisphereLight(0xffffff, 0x444444, HEMI_INTENSITY)
    scene.add(hemi)
    const sun = new THREE.DirectionalLight(0xffffff, SUN_INTENSITY)
    scene.add(sun)
    scene.add(sun.target)

    // Uniforms shared by every custom shader (same objects, so one update reaches all)
    const shared = {
      uZenith: { value: new THREE.Color() },
      uHorizon: { value: new THREE.Color() },
      uGlow: { value: new THREE.Color() },
      uGlowDir: { value: new THREE.Vector3(1, 0, 0) },
      uFogDensity: { value: FOG_DENSITY },
      uFlash: { value: 0 },
      uFlashColor: { value: new THREE.Color(0.75, 0.8, 1.0) },
      uAmbient: { value: new THREE.Color() },
      uSunColor: { value: new THREE.Color() },
      uSunDir: { value: new THREE.Vector3(1, 0.2, 0) },
      uTime: { value: 0 },
      uPointScale: { value: 1 },
      uMaxPoint: { value: 256 },
      uFunnelTint: { value: new THREE.Color() },
      uDebrisTint: { value: new THREE.Color() },
    }

    // Sky dome (follows the camera)
    const skyGeometry = new THREE.SphereGeometry(6000, 48, 24)
    const skyMaterial = new THREE.ShaderMaterial({
      vertexShader: SKY_VERT,
      fragmentShader: SKY_FRAG,
      uniforms: { ...shared, uFlashDir: { value: new THREE.Vector3(0, 1, 0) } },
      side: THREE.BackSide,
      depthWrite: false,
    })
    const sky = new THREE.Mesh(skyGeometry, skyMaterial)
    sky.renderOrder = -2
    sky.frustumCulled = false
    scene.add(sky)

    // Ground: fields, lawns and the scour track painted by the tornado
    const world = createWorld()
    const scourTexture = new THREE.DataTexture(world.scour, SCOUR_SIZE, SCOUR_SIZE, THREE.RedFormat, THREE.UnsignedByteType)
    scourTexture.minFilter = THREE.LinearFilter
    scourTexture.magFilter = THREE.LinearFilter
    scourTexture.generateMipmaps = false
    scourTexture.needsUpdate = true
    const groundGeometry = new THREE.PlaneGeometry(GROUND_SIZE, GROUND_SIZE)
    groundGeometry.rotateX(-Math.PI / 2)
    const groundMaterial = new THREE.ShaderMaterial({
      vertexShader: WORLD_VERT,
      fragmentShader: GROUND_FRAG,
      uniforms: {
        ...shared,
        uScour: { value: scourTexture },
        uScourExtent: { value: SCOUR_EXTENT },
        uRoadAngle: { value: 0 },
        uFieldSeed: { value: 0 },
        uTownRadius: { value: TOWN_RADIUS },
      },
    })
    const ground = new THREE.Mesh(groundGeometry, groundMaterial)
    ground.frustumCulled = false
    scene.add(ground)

    // Cloud base with the lowered, rotating wall cloud
    const cloudGeometry = makeCloudGeometry()
    const cloudMaterial = new THREE.ShaderMaterial({
      vertexShader: CLOUD_VERT,
      fragmentShader: CLOUD_FRAG,
      uniforms: {
        ...shared,
        uBaseH: { value: 220 },
        uWallDepth: { value: 30 },
        uWallRadius: { value: 150 },
        uCenter: { value: new THREE.Vector2() },
        uSwirl: { value: 0 },
        uCloudDark: { value: new THREE.Color() },
        uCloudLight: { value: new THREE.Color() },
        uFlashPos: { value: new THREE.Vector2() },
      },
      side: THREE.DoubleSide,
    })
    const cloud = new THREE.Mesh(cloudGeometry, cloudMaterial)
    cloud.frustumCulled = false
    scene.add(cloud)

    // Rigid bodies: one InstancedMesh per shape, every slot owned by one body in the pool
    const shapeGeometries: THREE.BufferGeometry[] = []
    shapeGeometries[SHAPE_BOX] = new THREE.BoxGeometry(1, 1, 1)
    shapeGeometries[1] = makeWedgeGeometry()
    shapeGeometries[2] = new THREE.CylinderGeometry(0.5, 0.5, 1, 14)
    shapeGeometries[3] = new THREE.IcosahedronGeometry(0.5, 1)
    shapeGeometries[4] = new THREE.ConeGeometry(0.5, 1, 12)
    const bodyMaterial = new THREE.MeshStandardMaterial({ roughness: 0.82, metalness: 0.04, flatShading: true })
    const meshes: THREE.InstancedMesh[] = []
    const black = new THREE.Color(0, 0, 0)
    for (let s = 0; s < SHAPE_COUNT; s++) {
      const mesh = new THREE.InstancedMesh(shapeGeometries[s], bodyMaterial, MAX_BODIES)
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
      mesh.setColorAt(0, black) // allocates the color buffer
      mesh.instanceColor!.setUsage(THREE.DynamicDrawUsage)
      mesh.frustumCulled = false // bounds would go stale as debris flies
      mesh.count = 0
      scene.add(mesh)
      meshes.push(mesh)
    }

    // Shards: leaf triangles and small tiles (shingles, decking, splinters, glass), one slot each
    const leafGeometry = makeLeafGeometry()
    const tileGeometry = new THREE.BoxGeometry(1, 1, 1)
    const leafMaterial = new THREE.MeshStandardMaterial({ roughness: 0.75, metalness: 0, flatShading: true, side: THREE.DoubleSide })
    const shardMeshes = [
      new THREE.InstancedMesh(leafGeometry, leafMaterial, TRI_POOL),
      new THREE.InstancedMesh(tileGeometry, bodyMaterial, TILE_POOL),
    ]
    for (const mesh of shardMeshes) {
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
      mesh.setColorAt(0, black)
      mesh.instanceColor!.setUsage(THREE.DynamicDrawUsage)
      mesh.frustumCulled = false
      scene.add(mesh)
    }

    // Funnel / debris cloud particles (positions computed in the vertex shader)
    const puffTexture = makePuffTexture()
    const funnelCount = FUNNEL_PARTICLES + DEBRIS_PARTICLES
    const funnelGeometry = new THREE.BufferGeometry()
    const seeds = new Float32Array(funnelCount * 4)
    const types = new Float32Array(funnelCount)
    for (let i = 0; i < funnelCount; i++) {
      for (let k = 0; k < 4; k++) seeds[i * 4 + k] = Math.random()
      types[i] = i < FUNNEL_PARTICLES ? 0 : 1
    }
    funnelGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(funnelCount * 3), 3))
    funnelGeometry.setAttribute('aSeed', new THREE.BufferAttribute(seeds, 4))
    funnelGeometry.setAttribute('aType', new THREE.BufferAttribute(types, 1))
    const funnelMaterial = new THREE.ShaderMaterial({
      vertexShader: FUNNEL_VERT,
      fragmentShader: PUFF_FRAG,
      uniforms: {
        ...shared,
        uPuff: { value: puffTexture },
        uH: { value: 200 },
        uBase: { value: new THREE.Vector2() },
        uLean: { value: new THREE.Vector2() },
        uWobble: { value: 0 },
        uWobblePh: { value: new THREE.Vector2() },
        uRB: { value: 10 },
        uRT: { value: 60 },
        uRc: { value: 30 },
        uBottom: { value: 0 },
        uVis: { value: 1 },
        uDebris: { value: 0 },
      },
      transparent: true,
      depthWrite: false,
    })
    const funnel = new THREE.Points(funnelGeometry, funnelMaterial)
    funnel.frustumCulled = false
    funnel.renderOrder = 2
    scene.add(funnel)

    // CPU dust puffs: attributes are views straight onto the simulation's arrays
    const dustGeometry = new THREE.BufferGeometry()
    const dustPosAttr = new THREE.BufferAttribute(world.dustPos, 3).setUsage(THREE.DynamicDrawUsage)
    const dustSizeAttr = new THREE.BufferAttribute(world.dustSize, 1).setUsage(THREE.DynamicDrawUsage)
    const dustAlphaAttr = new THREE.BufferAttribute(world.dustAlpha, 1).setUsage(THREE.DynamicDrawUsage)
    dustGeometry.setAttribute('position', dustPosAttr)
    dustGeometry.setAttribute('aSize', dustSizeAttr)
    dustGeometry.setAttribute('aAlpha', dustAlphaAttr)
    const dustMaterial = new THREE.ShaderMaterial({
      vertexShader: DUST_VERT,
      fragmentShader: PUFF_FRAG,
      uniforms: { ...shared, uPuff: { value: puffTexture } },
      transparent: true,
      depthWrite: false,
    })
    const dust = new THREE.Points(dustGeometry, dustMaterial)
    dust.frustumCulled = false
    dust.renderOrder = 1
    scene.add(dust)

    // Lightning bolt: a camera-facing ribbon rebuilt in place on each strike
    const boltMaxSegments = BOLT_POINTS - 1 + BOLT_BRANCH_POINTS - 1
    const boltPositions = new Float32Array(boltMaxSegments * 6 * 3)
    const boltGeometry = new THREE.BufferGeometry()
    const boltAttr = new THREE.BufferAttribute(boltPositions, 3).setUsage(THREE.DynamicDrawUsage)
    boltGeometry.setAttribute('position', boltAttr)
    const boltMaterial = new THREE.MeshBasicMaterial({
      color: new THREE.Color(0.85, 0.9, 1.0),
      transparent: true,
      opacity: 0,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      fog: false,
      toneMapped: false,
      side: THREE.DoubleSide,
    })
    const bolt = new THREE.Mesh(boltGeometry, boltMaterial)
    bolt.frustumCulled = false
    bolt.visible = false
    scene.add(bolt)

    // ------------------------------------------------------------------------------------------
    // Sizing, with a pixel budget and an automatic ratchet for slow GPUs

    let pixelRatioCap = 2
    const handleResize = () => {
      const w = window.innerWidth
      const h = window.innerHeight
      const budget = Math.sqrt(MAX_RENDER_PIXELS / Math.max(1, w * h))
      renderer.setPixelRatio(Math.max(MIN_PIXEL_RATIO, Math.min(window.devicePixelRatio, 2, budget, pixelRatioCap)))
      renderer.setSize(w, h)
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      const buffer = renderer.getDrawingBufferSize(new THREE.Vector2())
      sceneTarget.setSize(buffer.x, buffer.y)
      const bufferHeight = buffer.y
      shared.uPointScale.value = bufferHeight / (2 * Math.tan(THREE.MathUtils.degToRad(FOV) / 2))
      const gl = renderer.getContext()
      const range = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE) as Float32Array | null
      shared.uMaxPoint.value = Math.min(bufferHeight * 0.3, range ? range[1] : 64)
    }
    handleResize()

    // ------------------------------------------------------------------------------------------
    // Cycle setup

    const rand = Math.random
    let storm: Storm = newStorm(rand)
    let cam: ChaseCam = createChaseCam(storm, world, rand)
    const color = new THREE.Color()

    const writeAllBodies = () => {
      for (let s = 0; s < SHAPE_COUNT; s++) meshes[s].count = world.shapeCount[s]
      world.dirty.fill(1)
      world.colorDirty.fill(1)
    }

    const startCycle = () => {
      generateTown(world, rand)
      storm = newStorm(rand)
      cam = createChaseCam(storm, world, rand)
      const mood = MOODS[Math.floor(Math.random() * MOODS.length)]
      shared.uZenith.value.setHex(mood.zenith)
      shared.uHorizon.value.setHex(mood.horizon)
      shared.uGlow.value.setHex(mood.glow).multiplyScalar(0.9)
      shared.uFunnelTint.value.setHex(mood.funnel)
      shared.uDebrisTint.value.setHex(mood.debris)
      cloudMaterial.uniforms.uCloudDark.value.setHex(mood.cloudDark)
      cloudMaterial.uniforms.uCloudLight.value.setHex(mood.cloudLight)
      ;(scene.fog as THREE.FogExp2).color.copy(shared.uGlow.value).multiplyScalar(0.22).add(shared.uHorizon.value)
      hemi.color.setHex(mood.hemiSky)
      hemi.groundColor.setHex(mood.hemiGround)
      sun.color.setHex(mood.sun)
      // The low light comes from behind the funnel (silhouette) or across/behind the camera
      const viewX = storm.x - cam.px
      const viewZ = storm.z - cam.pz
      const viewAngle = Math.atan2(viewZ, viewX)
      const backlit = Math.random() < 0.55
      const glowAngle = viewAngle + (backlit ? 0 : Math.PI * (Math.random() < 0.5 ? 0.6 : -0.6)) + (Math.random() - 0.5) * 0.9
      shared.uGlowDir.value.set(Math.cos(glowAngle), 0, Math.sin(glowAngle))
      const el = THREE.MathUtils.degToRad(SUN_ELEVATION_DEG)
      shared.uSunDir.value.set(Math.cos(glowAngle) * Math.cos(el), Math.sin(el), Math.sin(glowAngle) * Math.cos(el))
      sun.position.copy(shared.uSunDir.value).multiplyScalar(500)
      groundMaterial.uniforms.uRoadAngle.value = world.roadAngle
      groundMaterial.uniforms.uFieldSeed.value = Math.random() * 100
      funnelMaterial.uniforms.uWobblePh.value.set(storm.ph1, storm.ph2)
      scourTexture.needsUpdate = true
      writeAllBodies()
    }
    startCycle()

    // ------------------------------------------------------------------------------------------
    // Lightning

    let nextFlashAt = 2 + Math.random() * 4
    let flashStart = -1
    const pulses: number[] = []
    let flashClock = 0
    const flashDir = skyMaterial.uniforms.uFlashDir.value as THREE.Vector3
    const flashPos = cloudMaterial.uniforms.uFlashPos.value as THREE.Vector2

    const buildBolt = (x: number, z: number, top: number) => {
      // Camera-facing quads along a jagged path, plus one branch
      const toCam = new THREE.Vector3(camera.position.x - x, 0, camera.position.z - z).normalize()
      const side = new THREE.Vector3(-toCam.z, 0, toCam.x)
      const pts: THREE.Vector3[] = []
      let off = 0
      for (let i = 0; i < BOLT_POINTS; i++) {
        const t = i / (BOLT_POINTS - 1)
        off += (Math.random() - 0.5) * 34
        pts.push(new THREE.Vector3(x + side.x * off, top * (1 - t), z + side.z * off))
      }
      const branchFrom = 6 + Math.floor(Math.random() * 10)
      const branch: THREE.Vector3[] = [pts[branchFrom].clone()]
      const dirSign = Math.random() < 0.5 ? -1 : 1
      for (let i = 1; i < BOLT_BRANCH_POINTS; i++) {
        const p = branch[i - 1].clone()
        p.addScaledVector(side, dirSign * (8 + Math.random() * 14))
        p.y -= 6 + Math.random() * 10
        branch.push(p)
      }
      let v = 0
      const quad = (a: THREE.Vector3, b: THREE.Vector3, width: number) => {
        const sx = side.x * width
        const sz = side.z * width
        const verts = [
          a.x - sx, a.y, a.z - sz, a.x + sx, a.y, a.z + sz, b.x + sx, b.y, b.z + sz,
          a.x - sx, a.y, a.z - sz, b.x + sx, b.y, b.z + sz, b.x - sx, b.y, b.z - sz,
        ]
        boltPositions.set(verts, v)
        v += 18
      }
      for (let i = 0; i < pts.length - 1; i++) quad(pts[i], pts[i + 1], BOLT_WIDTH)
      for (let i = 0; i < branch.length - 1; i++) quad(branch[i], branch[i + 1], BOLT_WIDTH * 0.5)
      boltGeometry.setDrawRange(0, v / 3)
      boltAttr.needsUpdate = true
    }

    const strike = () => {
      flashStart = flashClock
      pulses.length = 0
      const n = 1 + Math.floor(Math.random() * 3)
      for (let i = 0; i < n; i++) pulses.push(i * (0.07 + Math.random() * 0.12))
      // Somewhere ahead of the camera, well beyond the tornado
      const fwd = new THREE.Vector3()
      camera.getWorldDirection(fwd)
      const ang = Math.atan2(fwd.z, fwd.x) + (Math.random() - 0.5) * 1.3
      const dist = 700 + Math.random() * 1000
      const x = camera.position.x + Math.cos(ang) * dist
      const z = camera.position.z + Math.sin(ang) * dist
      const top = storm.H + storm.wallDepth + 10
      flashPos.set(x, z)
      flashDir.set(x - camera.position.x, top * 0.6, z - camera.position.z).normalize()
      const visibleBolt = Math.random() < BOLT_CHANCE
      bolt.visible = visibleBolt
      if (visibleBolt) buildBolt(x, z, top)
    }

    // ------------------------------------------------------------------------------------------
    // Animation

    let mode: 'in' | 'run' | 'out' = 'in'
    let fade = 0
    let lastTime = performance.now()
    let physicsDebt = 0
    let slowFrameAvg = 1 / 60
    let slowTimer = 0
    let animationId = 0
    const center = { x: 0, z: 0 }
    const pose = { px: 0, py: 0, pz: 0, tx: 0, ty: 0, tz: 0 }
    const ambientScale = 1 / Math.PI

    const animate = () => {
      animationId = requestAnimationFrame(animate)
      const now = performance.now()
      const rawDt = (now - lastTime) / 1000
      const dt = Math.min(rawDt, MAX_FRAME_SECONDS)
      lastTime = now

      // Quality ratchet: sustained slow frames lower the resolution
      slowFrameAvg += (rawDt - slowFrameAvg) * 0.05
      if (rawDt < 0.25 && slowFrameAvg > SLOW_FRAME_SECONDS) {
        slowTimer += rawDt
        if (slowTimer > 2.5 && renderer.getPixelRatio() > MIN_PIXEL_RATIO + 0.01) {
          pixelRatioCap = Math.max(MIN_PIXEL_RATIO, renderer.getPixelRatio() * 0.85)
          handleResize()
          slowTimer = 0
          slowFrameAvg = 1 / 60
        }
      } else {
        slowTimer = 0
      }

      // Cycle: fade in, let the storm play out, fade to black, build a new town
      if (mode === 'in') {
        fade = Math.min(1, fade + dt / FADE_SECONDS)
        if (fade >= 1) mode = 'run'
      } else if (mode === 'run' && storm.phase === PH_DONE) {
        mode = 'out'
      } else if (mode === 'out') {
        fade = Math.max(0, fade - dt / FADE_SECONDS)
        if (fade <= 0) {
          startCycle()
          mode = 'in'
        }
      }

      // Fixed-step physics
      physicsDebt += dt
      let steps = 0
      while (physicsDebt >= PHYSICS_STEP && steps < MAX_SUBSTEPS) {
        physicsDebt -= PHYSICS_STEP
        stepStorm(storm, PHYSICS_STEP)
        stepWorld(world, storm, PHYSICS_STEP, rand)
        steps++
      }
      if (steps === MAX_SUBSTEPS) physicsDebt = 0
      stepDust(world, storm, dt, rand)
      paintScour(world, storm, dt)
      if (storm.intensity >= 0.15) scourTexture.needsUpdate = true
      dustPosAttr.needsUpdate = true
      dustSizeAttr.needsUpdate = true
      dustAlphaAttr.needsUpdate = true

      // Push moved bodies into their instance slots
      const t = storm.time
      const touched = [false, false, false, false, false]
      const touchedColor = [false, false, false, false, false]
      for (let i = 0; i < world.count; i++) {
        if (world.dirty[i]) {
          world.dirty[i] = 0
          const s = world.shape[i]
          writeInstanceMatrix(world, i, meshes[s].instanceMatrix.array as Float32Array, world.inst[i] * 16, t)
          touched[s] = true
        }
        if (world.colorDirty[i]) {
          world.colorDirty[i] = 0
          const s = world.shape[i]
          color.setHex(world.color[i])
          meshes[s].setColorAt(world.inst[i], color)
          touchedColor[s] = true
        }
      }
      for (let s = 0; s < SHAPE_COUNT; s++) {
        if (touched[s]) meshes[s].instanceMatrix.needsUpdate = true
        if (touchedColor[s]) meshes[s].instanceColor!.needsUpdate = true
      }
      // Shards: only the dirty span of each pool is rewritten and uploaded
      for (let p = 0; p < 2; p++) {
        const mesh = shardMeshes[p]
        const base = p === 0 ? 0 : TRI_POOL
        const lo = world.dirtyLo[p]
        const hi = world.dirtyHi[p]
        if (hi >= lo) {
          const arr = mesh.instanceMatrix.array as Float32Array
          for (let i = lo; i <= hi; i++) {
            if (!world.sdirty[i]) continue
            world.sdirty[i] = 0
            writeShardMatrix(world, i, arr, (i - base) * 16)
          }
          mesh.instanceMatrix.clearUpdateRanges()
          mesh.instanceMatrix.addUpdateRange((lo - base) * 16, (hi - lo + 1) * 16)
          mesh.instanceMatrix.needsUpdate = true
          world.dirtyLo[p] = MAX_SHARDS
          world.dirtyHi[p] = -1
        }
        const clo = world.colorLo[p]
        const chi = world.colorHi[p]
        if (chi >= clo) {
          const arr = mesh.instanceColor!.array as Float32Array
          for (let i = clo; i <= chi; i++) {
            if (!world.scolorDirty[i]) continue
            world.scolorDirty[i] = 0
            color.setHex(world.scolor[i])
            const o = (i - base) * 3
            arr[o] = color.r
            arr[o + 1] = color.g
            arr[o + 2] = color.b
          }
          mesh.instanceColor!.clearUpdateRanges()
          mesh.instanceColor!.addUpdateRange((clo - base) * 3, (chi - clo + 1) * 3)
          mesh.instanceColor!.needsUpdate = true
          world.colorLo[p] = MAX_SHARDS
          world.colorHi[p] = -1
        }
      }

      // Lightning: a few rapid pulses, then a quiet gap
      flashClock += dt
      if (flashClock >= nextFlashAt) {
        strike()
        nextFlashAt = flashClock + LIGHTNING_MIN_GAP + Math.random() * (LIGHTNING_MAX_GAP - LIGHTNING_MIN_GAP)
      }
      let flash = 0
      if (flashStart >= 0) {
        const ft = flashClock - flashStart
        for (const p of pulses) if (ft >= p) flash += Math.exp(-(ft - p) * 13)
        if (ft > 1.5) {
          flashStart = -1
          bolt.visible = false
        }
      }
      flash = Math.min(flash, 1.6)
      shared.uFlash.value = flash
      boltMaterial.opacity = Math.min(1, flash * 1.4)
      hemi.intensity = HEMI_INTENSITY * (1 + flash * 1.2)
      shared.uAmbient.value.copy(hemi.color).multiplyScalar(hemi.intensity * ambientScale)
      shared.uSunColor.value.copy(sun.color).multiplyScalar(SUN_INTENSITY * ambientScale)

      // Storm uniforms
      shared.uTime.value = t
      const fu = funnelMaterial.uniforms
      fu.uH.value = storm.H
      fu.uBase.value.set(storm.x, storm.z)
      fu.uLean.value.set(storm.leanX, storm.leanZ)
      fu.uWobble.value = storm.wobble
      fu.uRB.value = storm.rB
      fu.uRT.value = storm.rT
      fu.uRc.value = storm.Rc
      fu.uBottom.value = storm.funnelBottom
      fu.uVis.value = storm.funnelVis
      fu.uDebris.value = storm.debris
      funnel.visible = storm.phase < PH_AFTER

      centerAt(storm, storm.H, center)
      const cu = cloudMaterial.uniforms
      cu.uBaseH.value = storm.H + storm.wallDepth
      cu.uWallDepth.value = storm.wallDepth * (0.45 + 0.55 * Math.max(storm.intensity, storm.funnelVis * 0.6))
      cu.uWallRadius.value = Math.max(storm.RcBase * storm.topFrac, 40) * 2.2
      cu.uCenter.value.set(center.x, center.z)
      cu.uSwirl.value = t * 0.045
      cloud.position.set(center.x, 0, center.z)

      // Storm-chaser camera with a little hand-held drift
      updateChaseCam(cam, storm, world, dt)
      chaseCamPose(cam, storm, pose)
      camera.position.set(pose.px, pose.py, pose.pz)
      camera.lookAt(pose.tx, pose.ty, pose.tz)
      sky.position.copy(camera.position)

      // Scene into the HDR target, then tone map + fade the whole image in one final pass
      postMaterial.uniforms.uFade.value = fade * fade * (3 - 2 * fade)
      renderer.setRenderTarget(sceneTarget)
      renderer.render(scene, camera)
      renderer.setRenderTarget(null)
      renderer.render(postScene, postCamera)
    }

    animate()
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      cancelAnimationFrame(animationId)
      for (const mesh of meshes) mesh.dispose()
      for (const geometry of shapeGeometries) geometry.dispose()
      for (const mesh of shardMeshes) mesh.dispose()
      leafGeometry.dispose()
      tileGeometry.dispose()
      leafMaterial.dispose()
      bodyMaterial.dispose()
      skyGeometry.dispose()
      skyMaterial.dispose()
      groundGeometry.dispose()
      groundMaterial.dispose()
      scourTexture.dispose()
      cloudGeometry.dispose()
      cloudMaterial.dispose()
      funnelGeometry.dispose()
      funnelMaterial.dispose()
      dustGeometry.dispose()
      dustMaterial.dispose()
      puffTexture.dispose()
      boltGeometry.dispose()
      boltMaterial.dispose()
      postGeometry.dispose()
      postMaterial.dispose()
      sceneTarget.dispose()
      renderer.dispose()
      container.removeChild(renderer.domElement)
    }
  }, [])

  return <div ref={containerRef} className="w-full h-full" />
}
