import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { mergeGeometries, mergeVertices } from 'three/examples/jsm/utils/BufferGeometryUtils.js'
import { Flight, Forest, PERIOD, TREE_FADE_START, TREE_RADIUS, TREE_THIN_DISTANCE, type ForestBounds } from './terrainMath'

// Configuration constants
const RENDER_SCALE = 0.5 // raymarch resolution relative to CSS pixels, then upscaled
const MIN_RENDER_SCALE = 0.3 // floor for the automatic quality ratchet on slow GPUs
const MAX_RENDER_PIXELS = 700_000 // cap on large/4K screens so one frame can never stall the GPU
const MSAA_SAMPLES = 4 // tree edges are rasterized, so they get real antialiasing
const SLOW_FRAME_SECONDS = 1 / 45 // average frame time that triggers a resolution drop
const DAY_SECONDS = 240 // one full day/night cycle (night runs faster, see below)
const NIGHT_SPEEDUP = 0.7 // night passes this much faster than day
const BIOME_MIN_SECONDS = 45 // time between re-randomized biomes / flight styles
const BIOME_MAX_SECONDS = 80
const BIOME_BLEND_SECONDS = 15
const TAN_HALF_FOV = 0.5 // vertical, about 53 degrees
const CAMERA_NEAR = 0.006 // the flight floor keeps every tree at least ~0.01 away
const CAMERA_FAR = 40
const MAX_TREES = 30_000 // instance slots; a safety cap (the nearest trees win if a forest is ever denser)
const FOREST_BUDGET_MS = 1.2 // CPU time per frame for populating new forest tiles
const FOREST_STARTUP_MS = 300 // populated up front so the first frames already have forests
const FOREST_REBUILD_SECONDS = 0.4 // how often the visible instance list is rewritten
const CULL_FOV_MARGIN = 1.3 // the visible list covers a wider view than the camera's...
const CULL_ASPECT_MARGIN = 1.15 // ...so turning between rebuilds never shows a gap
const TRUNK_COLOR = '#3a2a1e'
const SKY_CLEAR_COLOR = 0x000000

// Biomes are palettes plus a few terrain-shading knobs. Colors are sRGB hex and get
// converted to linear by THREE.Color.
interface Biome {
  grass: string
  rock: string
  sand: string
  water: string
  snowLine: number // absolute height where snow starts (terrain spans roughly -0.9..1.2)
  strata: number // 0..1 strength of horizontal rock banding (canyon look)
  waterLevel: number
  conifer: [string, string] // needle colors (each tree picks a mix of the two)
  leaf: [string, string] // broadleaf crown colors
  treeDensity: number // 0 = no trees
  treeSize: number // 0..1 scale on the tallest tree (keep <= 1: the camera's floor assumes it)
  coniferShare: number // fraction of trees that are conifers (the rest are broadleaf)
}

const BIOMES: Biome[] = [
  // Alpine: green valleys, grey rock, snow caps, cold lakes, dark spruce forests
  { grass: '#3d5a1e', rock: '#6e655c', sand: '#a89a7a', water: '#0d2a33', snowLine: 0.36, strata: 0.1, waterLevel: -0.15,
    conifer: ['#1d3b2a', '#2e4a2a'], leaf: ['#3f6a2a', '#5a7a30'], treeDensity: 0.9, treeSize: 1, coniferShare: 0.85 },
  // Autumn highlands: rust and gold slopes, blazing maples among dark pines
  { grass: '#8a5420', rock: '#6a5a50', sand: '#a08a68', water: '#102a30', snowLine: 0.55, strata: 0.15, waterLevel: -0.14,
    conifer: ['#22402c', '#34502e'], leaf: ['#b8401a', '#e0a028'], treeDensity: 0.85, treeSize: 0.95, coniferShare: 0.2 },
  // Red desert canyons: banded sandstone, sparse olive scrub, turquoise water
  { grass: '#7a6234', rock: '#a0502a', sand: '#c89a68', water: '#0c3a3a', snowLine: 9, strata: 0.85, waterLevel: -0.24,
    conifer: ['#3a4a2a', '#4c5630'], leaf: ['#6a6a34', '#8a7a44'], treeDensity: 0.18, treeSize: 0.6, coniferShare: 0.1 },
  // Heather moors: purple-brown slopes, slate rock, scattered dark copses
  { grass: '#5a3a52', rock: '#5c5c64', sand: '#8a8070', water: '#0c1e2e', snowLine: 0.62, strata: 0.2, waterLevel: -0.12,
    conifer: ['#1e3226', '#2a3e2a'], leaf: ['#3a4a26', '#55602e'], treeDensity: 0.35, treeSize: 0.85, coniferShare: 0.6 },
  // Emerald lowlands: lush green, dark rock, lots of water, bright broadleaf woods
  { grass: '#2a6a2c', rock: '#4e4a42', sand: '#b0a07a', water: '#0a3030', snowLine: 0.8, strata: 0.05, waterLevel: -0.08,
    conifer: ['#1a4228', '#2a5a30'], leaf: ['#3a8a30', '#62a03a'], treeDensity: 1, treeSize: 1, coniferShare: 0.3 },
  // Volcanic: black rock, acid-green moss, heavy snow, a few hardy black pines
  { grass: '#4a5a1a', rock: '#2a2628', sand: '#4a4440', water: '#081a20', snowLine: 0.28, strata: 0.3, waterLevel: -0.18,
    conifer: ['#1a2a1e', '#24331f'], leaf: ['#3a4a1e', '#4a5a24'], treeDensity: 0.3, treeSize: 0.8, coniferShare: 0.8 },
]

// Sky keyframes indexed by sun elevation (sin of altitude). Dawn and dusk use different
// palettes: dawn leans gold, dusk leans magenta. Night is shared.
const SKY_ELEVATIONS = [-0.35, -0.12, 0.0, 0.08, 0.25, 0.6]
const DAWN = {
  zenith: ['#01030a', '#0b1230', '#2b3570', '#3d62a8', '#2f6cc4', '#1f5fbf'],
  horizon: ['#03060f', '#1d2340', '#e07848', '#f2b070', '#bcd4ec', '#a9cbee'],
  glow: ['#000000', '#2a120a', '#ff7a30', '#ffb050', '#ffe8c8', '#fff4e4'],
}
const DUSK = {
  zenith: ['#01030a', '#120c2c', '#3a2060', '#4a4890', '#2f6cc4', '#1f5fbf'],
  horizon: ['#03060f', '#2a1438', '#d84a7a', '#ff8a6a', '#c8d0e8', '#a9cbee'],
  glow: ['#000000', '#30081e', '#ff4a6a', '#ff7a40', '#ffe0c4', '#fff4e4'],
}
const MOONLIGHT = '#8fa8e0'
const NIGHT_AMBIENT = '#0a1226'

interface FlightStyle {
  clearance: number // cruise height above the typical (median-ish) ground around the flight
  speed: number // world units per second
  fogDensity: number
  cloudCover: number // 0 clear .. 1 overcast
}

// Shared fullscreen-triangle-ish quad; both passes write gl_Position directly
const QUAD_VERT = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`

// Everything the landscape, the trees and their ground shadows must agree on: the terrain
// function, sky and fog, terrain shadows and the final tonemap. Each fragment shader starts
// with this, so a tree and the ground behind it are lit, fogged and graded identically.
const COMMON_GLSL = /* glsl */ `
uniform vec2 uResolution;
uniform float uTime;
uniform vec3 uCamPos;
uniform vec3 uCamRight;
uniform vec3 uCamUp;
uniform vec3 uCamFwd;
uniform float uTanHalfFov;
uniform float uNear;
uniform float uFar;
uniform vec2 uOriginW;
uniform vec3 uSunDir;
uniform vec3 uMoonDir;
uniform vec3 uLightDir;
uniform vec3 uLightCol;
uniform vec3 uZenith;
uniform vec3 uHorizon;
uniform vec3 uSunGlow;
uniform vec3 uAmbient;
uniform float uNight;
uniform float uStarAngle;
uniform vec2 uCloudOffset;
uniform vec2 uWave1;
uniform vec2 uWave2;
uniform float uCloudCover;
uniform float uFogDensity;
uniform float uWater;
uniform vec3 uGrass;
uniform vec3 uRock;
uniform vec3 uSand;
uniform vec3 uWaterCol;
uniform float uSnowLine;
uniform float uStrata;

#define PER 256.0
#define MASK_PER 16.0
#define FAR 24.0
#define MAXH 1.4

// Integer hash (lowbias32-style); mirrored exactly in terrainMath.ts
uint ihash(uvec2 q) {
  uint h = (q.x * 0x8da6b343u) ^ (q.y * 0xd8163841u);
  h ^= h >> 16u;
  h *= 0x7feb352du;
  h ^= h >> 15u;
  h *= 0x846ca68bu;
  h ^= h >> 16u;
  return h;
}

float hash01(vec2 c) {
  return float(ihash(uvec2(c)) >> 8u) * (1.0 / 16777216.0);
}

// Periodic value noise with analytic derivatives: (value in [-1,1], d/dx, d/dy).
// p must already be in [0, per).
vec3 noised(vec2 p, float per) {
  vec2 fl = floor(p);
  vec2 i = mod(fl, per);
  vec2 f = p - fl;
  vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  vec2 du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
  vec2 j = mod(i + 1.0, per);
  float a = hash01(i);
  float b = hash01(vec2(j.x, i.y));
  float c = hash01(vec2(i.x, j.y));
  float d = hash01(j);
  float k1 = b - a;
  float k2 = c - a;
  float k4 = a - b - c + d;
  return vec3(-1.0 + 2.0 * (a + k1 * u.x + k2 * u.y + k4 * u.x * u.y),
              2.0 * du * vec2(k1 + k4 * u.y, k2 + k4 * u.x));
}

// Eroded fbm heightfield. Each octave applies the integer matrix [[2,-1],[1,2]] and wraps
// mod PER, so coordinates stay small (no float precision loss, ever) and the result is
// periodic. Mirrored in terrainMath.ts.
float terrain(vec2 x, int oct) {
  float region = 0.5 + 0.5 * noised(mod(x / MASK_PER, MASK_PER), MASK_PER).x;
  vec2 p = mod(x, PER);
  float a = 0.0;
  float b = 1.0;
  vec2 d = vec2(0.0);
  for (int i = 0; i < 11; i++) {
    if (i >= oct) break;
    vec3 n = noised(p, PER);
    d += n.yz;
    a += b * n.x / (1.0 + dot(d, d));
    b *= 0.47;
    p = mod(vec2(2.0 * p.x - p.y, p.x + 2.0 * p.y), PER);
  }
  float amp = mix(0.16, 0.7, smoothstep(0.25, 0.8, region));
  return amp * a + (region - 0.5) * 0.5;
}

// The same surface with a fractional octave count: the last octave fades in, so detail can
// follow distance continuously (no visible LOD steps). Used only for shading normals.
float terrainL(vec2 x, float oct) {
  float region = 0.5 + 0.5 * noised(mod(x / MASK_PER, MASK_PER), MASK_PER).x;
  vec2 p = mod(x, PER);
  float a = 0.0;
  float b = 1.0;
  vec2 d = vec2(0.0);
  for (int i = 0; i < 10; i++) {
    float w = clamp(oct - float(i), 0.0, 1.0);
    if (w <= 0.0) break;
    vec3 n = noised(p, PER);
    d += w * n.yz;
    a += w * b * n.x / (1.0 + dot(d, d));
    b *= 0.47;
    p = mod(vec2(2.0 * p.x - p.y, p.x + 2.0 * p.y), PER);
  }
  float amp = mix(0.16, 0.7, smoothstep(0.25, 0.8, region));
  return amp * a + (region - 0.5) * 0.5;
}

// Sky gradient + sun/moon glow only. Also used as the fog color so distant terrain melts
// seamlessly into the horizon.
vec3 skyColor(vec3 rd) {
  float y = clamp(rd.y, 0.0, 1.0);
  vec3 col = mix(uHorizon, uZenith, pow(y, 0.5));
  float sd = max(dot(rd, uSunDir), 0.0);
  col += uSunGlow * (0.16 * pow(sd, 3.0) + 0.45 * pow(sd, 20.0) + 0.6 * pow(sd, 300.0));
  float md = max(dot(rd, uMoonDir), 0.0);
  col += vec3(0.35, 0.45, 0.7) * uNight * (0.03 * pow(md, 8.0) + 0.12 * pow(md, 200.0));
  return col;
}

// Aerial perspective: distance haze plus mist that pools in the valleys. The view-distance
// limit fades in over a wide band (a third of FAR to FAR) so a big distant mountain never
// has its far half dissolving while its near half stays solid.
float fogAmount(float t, vec3 ro, vec3 rd) {
  float hb = 2.2;
  float base = exp(-max(ro.y, -1.0) * hb);
  float k = rd.y * hb;
  float heightTerm = abs(k) < 1e-3 ? base * t : base * (1.0 - exp(-t * k)) / k;
  float dens = uFogDensity * (0.35 * t + 0.6 * heightTerm);
  float fo = 1.0 - exp(-dens);
  return max(fo, smoothstep(FAR * 0.35, FAR, t));
}

vec3 applyFog(vec3 col, float t, vec3 ro, vec3 rd) {
  return mix(col, skyColor(rd), fogAmount(t, ro, rd));
}

float softShadow(vec3 ro, vec3 rd) {
  float res = 1.0;
  float t = 0.02;
  for (int i = 0; i < 32; i++) {
    vec3 p = ro + rd * t;
    if (p.y > MAXH) break;
    float h = p.y - terrain(p.xz, 5);
    res = min(res, 10.0 * h / t);
    if (res < 0.01) break;
    t += clamp(h * 0.7, 0.012, 0.35);
  }
  res = clamp(res, 0.0, 1.0);
  return res * res * (3.0 - 2.0 * res);
}

vec3 aces(vec3 x) {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

// Tonemap, gamma, and dither before the 8-bit render target (keeps dusk gradients smooth)
vec3 finish(vec3 col) {
  col = aces(col * 1.1);
  col = pow(col, vec3(1.0 / 2.2));
  float dn = hash01(mod(gl_FragCoord.xy, 4096.0));
  return col + (dn - 0.5) / 255.0;
}
`

// The landscape: raymarched heightfield, water with reflections, sky, fog, tonemapping.
// It also writes depth (matching the tree camera's projection) so the rasterized trees
// and the raymarched ground hide each other correctly.
const TERRAIN_FRAG = /* glsl */ `
${COMMON_GLSL}
varying vec2 vUv;

float fbmClouds(vec2 p) {
  p = mod(p, PER);
  float a = 0.0;
  float b = 0.55;
  for (int i = 0; i < 4; i++) {
    a += b * noised(p, PER).x;
    b *= 0.5;
    p = mod(vec2(2.0 * p.x - p.y, p.x + 2.0 * p.y), PER);
  }
  return a;
}

vec3 stars(vec3 rd) {
  // The celestial sphere turns slowly through the night
  float c = cos(uStarAngle);
  float s = sin(uStarAngle);
  vec3 d = vec3(rd.x, c * rd.y - s * rd.z, s * rd.y + c * rd.z);
  vec3 g = d * 150.0 + 512.0;
  vec3 cell = floor(g);
  uint h = ihash(uvec2(cell.xy) + uvec2(uint(cell.z) * 2654435761u, uint(cell.z)));
  float r = float(h >> 8u) * (1.0 / 16777216.0);
  if (r < 0.975) return vec3(0.0);
  float r2 = fract(r * 97.31);
  float r3 = fract(r * 531.7);
  vec3 star = cell + vec3(0.25 + 0.5 * r2, 0.25 + 0.5 * r3, 0.25 + 0.5 * fract(r * 13.7));
  vec3 v = g - star;
  v -= d * dot(v, d); // distance across the sky, ignoring depth within the cell
  float bright = pow((r - 0.975) / 0.025, 3.0) * 2.5 + 0.15;
  float twinkle = 0.75 + 0.25 * sin(uTime * (1.5 + 4.0 * r2) + r3 * 60.0);
  vec3 tint = mix(vec3(1.0, 0.82, 0.65), vec3(0.7, 0.82, 1.0), r3);
  return tint * bright * twinkle * smoothstep(0.32, 0.0, length(v));
}

// Full sky for primary and reflected rays: gradient, stars, sun and moon discs, clouds
vec3 skyFull(vec3 rd) {
  vec3 col = skyColor(rd);
  if (rd.y <= 0.0) return col;
  float hz = smoothstep(0.0, 0.1, rd.y);
  if (uNight > 0.01) col += stars(rd) * uNight * hz;
  float sd = dot(rd, uSunDir);
  col += uSunGlow * 14.0 * smoothstep(0.99955, 0.9998, sd);
  float md = dot(rd, uMoonDir);
  float moon = smoothstep(0.99962, 0.99982, md);
  // A little shading across the disc so it reads as a sphere, not a dot
  col += vec3(0.8, 0.86, 1.0) * 1.3 * moon * uNight * (0.75 + 0.25 * dot(rd - uMoonDir, uSunDir) * 40.0);

  // Cloud layer: a plane projected onto the dome, drifting with the wind
  vec2 cp = rd.xz / (rd.y + 0.09) * 0.4 + uCloudOffset;
  float n = fbmClouds(cp);
  float cov = smoothstep(0.35 - 0.5 * uCloudCover, 0.75 - 0.4 * uCloudCover, n * 0.5 + 0.5);
  float sunward = pow(max(sd, 0.0), 4.0);
  vec3 lit = uLightCol * 0.25 + uHorizon * 0.55 + uSunGlow * 0.5 * sunward;
  vec3 shade = uZenith * 0.45 + uHorizon * 0.25 + uAmbient * 0.3;
  float edge = smoothstep(0.25, 0.85, n * 0.5 + 0.5); // thin edges catch the light
  vec3 cloud = mix(lit * 1.15, shade, edge * 0.7);
  return mix(col, cloud, cov * hz * 0.85);
}

// Heightfield march. Steps are half the height above the ground but never shorter than
// about two pixels' footprint: rays that graze a ridge used to crawl along it in tiny
// steps and run out of iterations, and those pixels were then drawn as ground hanging in
// the air above the peak (the shimmering "melting" tops). Crossing the surface is refined
// by bisection, and a ray that does run out is sky, never phantom ground.
float march(vec3 ro, vec3 rd, float tmax) {
  float t = 0.004;
  float tPrev = t;
  for (int i = 0; i < 250; i++) {
    vec3 p = ro + rd * t;
    if (t > tmax || (rd.y >= 0.0 && p.y > MAXH)) return 1e5;
    float h = p.y - terrain(p.xz, 7);
    if (h < 0.0) {
      float a = tPrev;
      float b = t;
      for (int j = 0; j < 6; j++) {
        float m = 0.5 * (a + b);
        vec3 q = ro + rd * m;
        if (q.y - terrain(q.xz, 7) < 0.0) b = m;
        else a = m;
      }
      return 0.5 * (a + b);
    }
    if (h < 0.0003 * t) return t;
    tPrev = t;
    t += max(0.5 * h, 0.003 * t);
  }
  return 1e5;
}

// Cheaper march for reflections in water
float marchRefl(vec3 ro, vec3 rd, float tmax) {
  float t = 0.004;
  for (int i = 0; i < 60; i++) {
    vec3 p = ro + rd * t;
    if (t > tmax || p.y > MAXH) return 1e5;
    float h = p.y - terrain(p.xz, 5);
    if (h < 0.002 * t) return t;
    t += max(0.5 * h, 0.004 * t);
  }
  return 1e5;
}

// Shading normal with detail matched to the pixel footprint: octaves finer than a pixel
// only add per-frame sparkle (and flickering snow) as the camera moves
vec3 calcNormal(vec3 p, float t, float oct) {
  float e = max(0.0004, 0.0016 * t);
  return normalize(vec3(
    terrainL(p.xz - vec2(e, 0.0), oct) - terrainL(p.xz + vec2(e, 0.0), oct),
    2.0 * e,
    terrainL(p.xz - vec2(0.0, e), oct) - terrainL(p.xz + vec2(0.0, e), oct)));
}

float normalOctaves(float t) {
  return clamp(7.5 - 1.25 * log(t), 3.0, 10.0);
}

vec3 shadeTerrain(vec3 p, vec3 n, vec3 rd, float sh) {
  float aboveWater = p.y - uWater;
  float v1 = noised(mod(p.xz * 6.0, PER), PER).x;
  float v2 = noised(mod(p.xz * 37.0, PER), PER).x;

  vec3 rock = uRock * (0.8 + 0.22 * v1 + 0.12 * v2);
  float band = 0.5 + 0.5 * sin(p.y * 42.0 + v1 * 2.5);
  rock *= mix(1.0, 0.65 + 0.6 * band, uStrata);
  vec3 grass = uGrass * (0.8 + 0.2 * v1 + 0.15 * v2);

  float grassAmt = smoothstep(0.70, 0.86, n.y + 0.08 * v1)
                 * (1.0 - smoothstep(0.45, 0.85, aboveWater + 0.15 * v1));
  vec3 col = mix(rock, grass, grassAmt);
  float sandAmt = (1.0 - smoothstep(0.008, 0.035, aboveWater + 0.01 * v2)) * smoothstep(0.5, 0.8, n.y);
  col = mix(col, uSand, sandAmt);
  float snowAmt = smoothstep(uSnowLine - 0.05, uSnowLine + 0.08, p.y + 0.12 * v1 + 0.15 * (n.y - 0.7))
                * smoothstep(0.5, 0.72, n.y + 0.1 * v2);
  col = mix(col, vec3(0.8, 0.84, 0.9), snowAmt);

  vec3 L = uLightDir;
  float dif = clamp(dot(n, L), 0.0, 1.0) * sh;
  float amb = 0.55 + 0.45 * n.y;
  float bou = clamp(0.5 - 0.5 * dot(n, normalize(vec3(L.x, 0.0, L.z) + 1e-4)), 0.0, 1.0) * (1.0 - 0.5 * n.y);
  vec3 lin = uLightCol * dif * 2.2 + uAmbient * amb + uLightCol * 0.12 * bou;
  col *= lin;

  vec3 hv = normalize(L - rd);
  col += snowAmt * uLightCol * 0.35 * pow(clamp(dot(n, hv), 0.0, 1.0), 24.0) * sh;
  return col;
}

// Window depth of a point t along the ray, for the tree camera's perspective projection
float rayDepth(float t, vec3 rd) {
  float d = max(t * dot(rd, uCamFwd), uNear);
  float ndc = (uFar + uNear) / (uFar - uNear) - 2.0 * uFar * uNear / ((uFar - uNear) * d);
  return clamp(0.5 * ndc + 0.5, 0.0, 1.0);
}

void main() {
  vec2 uv = vUv * 2.0 - 1.0;
  uv.x *= uResolution.x / uResolution.y;
  vec3 ro = uCamPos;
  vec3 rd = normalize(uCamFwd + uTanHalfFov * (uv.x * uCamRight + uv.y * uCamUp));

  float tWater = rd.y < 0.0 ? (uWater - ro.y) / rd.y : 1e5;
  float tmax = min(FAR, tWater);
  float t = march(ro, rd, tmax);
  bool lit = dot(uLightCol, vec3(1.0)) > 0.002 && uLightDir.y > -0.05;

  vec3 col;
  float depth = 1.0;
  if (t < tmax) {
    vec3 p = ro + rd * t;
    vec3 n = calcNormal(p, t, normalOctaves(t));
    float sh = 1.0;
    if (lit && dot(n, uLightDir) > 0.0) sh = softShadow(p + n * 0.004, uLightDir);
    col = shadeTerrain(p, n, rd, sh);
    col = applyFog(col, t, ro, rd);
    depth = rayDepth(t, rd);
  } else if (tWater < FAR) {
    vec3 p = ro + rd * tWater;
    // Two drifting octaves of ripples; flatten with distance to avoid shimmer
    vec3 w1 = noised(mod(p.xz * 18.0 + uWave1, PER), PER);
    vec3 w2 = noised(mod(p.xz * 41.0 + uWave2, PER), PER);
    vec2 g = (w1.yz + 0.45 * w2.yz) * (0.035 / (1.0 + tWater * 0.6));
    vec3 n = normalize(vec3(-g.x, 1.0, -g.y));
    vec3 rr = reflect(rd, n);
    rr.y = max(rr.y, 0.002);
    float fres = 0.02 + 0.98 * pow(1.0 - clamp(dot(-rd, n), 0.0, 1.0), 5.0);

    vec3 refl;
    float tr = marchRefl(p + vec3(0.0, 0.002, 0.0), rr, 8.0);
    if (tr < 8.0) {
      vec3 q = p + rr * tr;
      vec3 nq = calcNormal(q, tr + tWater, 6.0);
      refl = shadeTerrain(q, nq, rr, 1.0);
      refl = applyFog(refl, tr, p, rr);
    } else {
      refl = skyFull(rr);
    }

    float sh = lit ? softShadow(p + vec3(0.0, 0.004, 0.0), uLightDir) : 0.0;
    float wd = uWater - terrain(p.xz, 4);
    vec3 body = uWaterCol * (uAmbient * 0.8 + uLightCol * (0.3 + 0.5 * sh) * max(uLightDir.y, 0.0));
    vec3 shallow = uSand * (uAmbient * 0.7 + uLightCol * sh * max(uLightDir.y, 0.0) * 1.6) * 0.6;
    body = mix(shallow, body, smoothstep(0.0, 0.06, wd));
    col = mix(body, refl, fres);
    col += uLightCol * 3.0 * pow(max(dot(rr, uLightDir), 0.0), 350.0) * sh;
    // A thin bright lip where water meets the shore
    float foam = smoothstep(0.012, 0.0, wd) * (0.6 + 0.4 * w2.x);
    col = mix(col, (uAmbient + uLightCol * sh) * 0.7, foam * 0.5);
    col = applyFog(col, tWater, ro, rd);
    depth = rayDepth(tWater, rd);
  } else {
    col = skyFull(rd);
  }

  gl_FragColor = vec4(finish(col), 1.0);
  gl_FragDepth = depth;
}
`

// How big a tree instance is drawn right now (0 = hidden), from its stored suitability and
// the current biome: density, water level, snow line and conifer share all blend smoothly,
// so trees grow and shrink with a biome change instead of popping. Far away only a thinned
// subset remains (drawn a little larger to keep the forest's coverage), and everything
// shrinks away toward the edge of the forest radius. Mirrors Forest.rebuild's bounds.
const TREE_SCALE_GLSL = /* glsl */ `
attribute vec4 aTint; // suitability, growth threshold, conifer rank, color mix
attribute vec4 aMisc; // crown aspect, ground slope x / z, thinning rank
uniform float uWater;
uniform float uSnowLine;
uniform float uTreeDensity;
uniform float uTreeSize;
uniform float uConifer;
uniform float uThinDistance;
uniform float uFadeStart;
uniform float uRadius;

float treeScale(vec3 base, float conifer) {
  float aw = base.y - uWater;
  float dens = uTreeDensity * aTint.x
             * smoothstep(0.025, 0.07, aw)
             * (1.0 - smoothstep(0.55, 0.95, aw))
             * (1.0 - smoothstep(uSnowLine - 0.25, uSnowLine - 0.03, base.y));
  float grow = clamp((dens - aTint.y) * 6.0, 0.0, 1.0);
  float con = smoothstep(aTint.z - 0.06, aTint.z + 0.06, uConifer * 1.12 - 0.06);
  float species = conifer > 0.5 ? con : 1.0 - con;
  float d = length(base.xz - cameraPosition.xz);
  float keep = min(1.0, uThinDistance / max(d, 1e-3));
  float thin = clamp((keep - aMisc.w) / 0.06, 0.0, 1.0);
  float enlarge = min(inversesqrt(keep), 1.8);
  float edge = 1.0 - smoothstep(uFadeStart, uRadius, d);
  return grow * species * thin * edge * enlarge * uTreeSize;
}
`

const TREE_VERT = /* glsl */ `
${TREE_SCALE_GLSL}
attribute float aLeaf; // 1 foliage, 0 trunk
attribute float aAo; // baked occlusion: dark low and inside the crown
uniform float uSpecies; // 1 for the conifer mesh, 0 for the broadleaf mesh
uniform vec3 uColA;
uniform vec3 uColB;
uniform vec3 uTrunk;
varying vec3 vWorld;
varying vec3 vNormal;
varying vec3 vColor;
varying float vAo;
varying float vLeaf;

void main() {
  vec3 base = instanceMatrix[3].xyz;
  float k = treeScale(base, uSpecies);
  if (k < 0.003) {
    // Hidden (wrong species here, not grown, or thinned out): outside the clip volume
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    return;
  }
  float asp = 0.85 + 0.3 * aMisc.x;
  vec4 wp = modelMatrix * instanceMatrix * vec4(position * vec3(asp, 1.0, asp) * k, 1.0);
  vWorld = wp.xyz;
  vNormal = normalize(mat3(modelMatrix) * mat3(instanceMatrix) * (normal * vec3(1.0 / asp, 1.0, 1.0 / asp)));
  float shade = 0.82 + 0.36 * fract(aTint.w * 13.7);
  vColor = mix(uTrunk, mix(uColA, uColB, aTint.w) * shade, aLeaf);
  vAo = aAo;
  vLeaf = aLeaf;
  gl_Position = projectionMatrix * viewMatrix * wp;
}
`

// Trees use the landscape's own light model: the sun/moon key light with terrain shadows
// (a directional light), sky ambient weighted by the normal's upward tilt (a hemisphere
// light), the same bounce, fog and tonemap. Leaves wrap light a little and glow when the
// sun is behind them.
const TREE_FRAG = /* glsl */ `
${COMMON_GLSL}
uniform float uSpecies;
varying vec3 vWorld;
varying vec3 vNormal;
varying vec3 vColor;
varying float vAo;
varying float vLeaf;

void main() {
  vec3 n = normalize(vNormal);
  vec3 toP = vWorld - cameraPosition;
  float t = length(toP);
  vec3 rd = toP / t;
  vec3 L = uLightDir;
  vec3 wp = vec3(vWorld.x + uOriginW.x, vWorld.y, vWorld.z + uOriginW.y);
  bool lit = dot(uLightCol, vec3(1.0)) > 0.002 && L.y > -0.05;
  float sh = lit ? softShadow(wp + vec3(0.0, 0.003, 0.0), L) : 0.0;
  float nl = dot(n, L);
  float dif = clamp(mix(nl, (nl + 0.5) / 1.5, vLeaf), 0.0, 1.0) * sh * mix(1.0, vAo, 0.5);
  float amb = (0.55 + 0.45 * n.y) * vAo;
  float bou = clamp(0.5 - 0.5 * dot(n, normalize(vec3(L.x, 0.0, L.z) + 1e-4)), 0.0, 1.0) * (1.0 - 0.5 * n.y);
  vec3 col = vColor * (uLightCol * dif * 2.2 + uAmbient * amb + uLightCol * 0.12 * bou * vAo);
  float back = pow(clamp(dot(rd, L), 0.0, 1.0), 5.0) * vLeaf * sh;
  col += vColor * uLightCol * back * mix(0.9, 0.35, uSpecies);
  col = applyFog(col, t, uCamPos, rd);
  gl_FragColor = vec4(finish(col), 1.0);
}
`

// Each tree's shadow on the ground: a soft ellipse laid on the local ground slope, cast
// away from the light and stretched as the sun gets low. Multiplied onto the frame.
const SHADOW_VERT = /* glsl */ `
${TREE_SCALE_GLSL}
uniform vec3 uLightDir;
uniform float uShadowStrength;
varying vec2 vQ;
varying vec3 vWorld;

void main() {
  vec3 base = instanceMatrix[3].xyz;
  float k = treeScale(base, 0.0) + treeScale(base, 1.0);
  if (k < 0.003 || uShadowStrength <= 0.0) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    return;
  }
  float h = length(instanceMatrix[1].xyz) * k;
  float r = h * 0.32 * (0.85 + 0.3 * aMisc.x);
  vec2 lh = uLightDir.xz;
  float ll = length(lh);
  vec2 dir = ll > 1e-4 ? -lh / ll : vec2(1.0, 0.0);
  float cotE = min(ll / max(uLightDir.y, 0.05), 4.0);
  // The crown (from about 0.3 h up) throws its shadow from c0 to c1 along dir
  float c0 = 0.3 * h * cotE;
  float c1 = h * cotE;
  float mid = 0.5 * (c0 + c1);
  float hl = 0.5 * (c1 - c0) + r;
  vec2 side = vec2(-dir.y, dir.x);
  vec2 off = dir * (mid + position.x * hl) + side * (position.y * r);
  float d = length(base.xz - cameraPosition.xz);
  float y = base.y + 0.0027 + 0.0003 * d + dot(aMisc.yz, off);
  vec3 wp = vec3(base.x + off.x, y, base.z + off.y);
  vQ = position.xy;
  vWorld = wp;
  gl_Position = projectionMatrix * viewMatrix * vec4(wp, 1.0);
}
`

const SHADOW_FRAG = /* glsl */ `
${COMMON_GLSL}
uniform float uShadowStrength;
varying vec2 vQ;
varying vec3 vWorld;

void main() {
  float m = 1.0 - smoothstep(0.3, 1.0, dot(vQ, vQ));
  vec3 toP = vWorld - cameraPosition;
  float t = length(toP);
  float fo = fogAmount(t, uCamPos, toP / t);
  // No double darkening where the ground is already in a hill's shadow
  vec3 wp = vec3(vWorld.x + uOriginW.x, vWorld.y, vWorld.z + uOriginW.y);
  float sh = softShadow(wp + vec3(0.0, 0.003, 0.0), uLightDir);
  float s = uShadowStrength * m * (1.0 - fo) * sh;
  gl_FragColor = vec4(vec3(1.0 - s), 1.0);
}
`

// Upscale to the screen with bilinear filtering and a soft vignette
const BLIT_FRAG = /* glsl */ `
varying vec2 vUv;
uniform sampler2D uScene;
void main() {
  vec3 col = texture2D(uScene, vUv).rgb;
  vec2 q = vUv - 0.5;
  col *= 1.0 - 0.45 * smoothstep(0.15, 0.75, dot(q, q) * 2.0);
  gl_FragColor = vec4(col, 1.0);
}
`

const mod = (x: number, p: number) => x - p * Math.floor(x / p)
const smoothstep = (a: number, b: number, x: number) => {
  const t = Math.min(Math.max((x - a) / (b - a), 0), 1)
  return t * t * (3 - 2 * t)
}
const rand = (a: number, b: number) => a + Math.random() * (b - a)
const fract = (x: number) => x - Math.floor(x)

const toColors = (hexes: string[]) => hexes.map(h => new THREE.Color(h))
const DAWN_COLORS = { zenith: toColors(DAWN.zenith), horizon: toColors(DAWN.horizon), glow: toColors(DAWN.glow) }
const DUSK_COLORS = { zenith: toColors(DUSK.zenith), horizon: toColors(DUSK.horizon), glow: toColors(DUSK.glow) }

// Piecewise-linear lookup into a keyframe color table by sun elevation
function sampleStops(stops: THREE.Color[], elevation: number, out: THREE.Color) {
  const e = SKY_ELEVATIONS
  if (elevation <= e[0]) return out.copy(stops[0])
  for (let i = 1; i < e.length; i++) {
    if (elevation <= e[i]) {
      return out.copy(stops[i - 1]).lerp(stops[i], (elevation - e[i - 1]) / (e[i] - e[i - 1]))
    }
  }
  return out.copy(stops[stops.length - 1])
}

const randomStyle = (): FlightStyle => ({
  clearance: rand(0.1, 0.3),
  speed: rand(0.13, 0.2),
  fogDensity: rand(0.08, 0.17),
  cloudCover: rand(0, 0.85),
})

// ---------------------------------------------------------------- tree models
// Unit-height models (base at y = 0, top at y <= 1); instances scale them to the tree's
// height and stretch the crown width by a per-tree aspect. Low-poly, but with smoothed,
// crown-shaped normals and baked occlusion so they shade like foliage rather than facets.

// Tags a part with the per-vertex attributes the tree shader expects
function tagPart(g: THREE.BufferGeometry, leaf: number, ao: (x: number, y: number, z: number) => number) {
  g.deleteAttribute('uv')
  const pos = g.getAttribute('position')
  const leafArr = new Float32Array(pos.count).fill(leaf)
  const aoArr = new Float32Array(pos.count)
  for (let i = 0; i < pos.count; i++) aoArr[i] = ao(pos.getX(i), pos.getY(i), pos.getZ(i))
  g.setAttribute('aLeaf', new THREE.BufferAttribute(leafArr, 1))
  g.setAttribute('aAo', new THREE.BufferAttribute(aoArr, 1))
  return g
}

// A spruce: three stacked, slightly ragged cones on a short trunk
function buildConifer() {
  const parts: THREE.BufferGeometry[] = []
  const trunk = new THREE.CylinderGeometry(0.03, 0.05, 0.4, 5, 1, true)
  trunk.translate(0, 0.15, 0)
  parts.push(tagPart(trunk, 0, () => 0.45))
  const tiers = [
    { y0: 0.12, y1: 0.62, r: 0.34 },
    { y0: 0.34, y1: 0.84, r: 0.27 },
    { y0: 0.56, y1: 1.0, r: 0.19 },
  ]
  tiers.forEach((tier, k) => {
    const h = tier.y1 - tier.y0
    const cone = new THREE.ConeGeometry(tier.r, h, 7, 1, true)
    cone.rotateY(k * 0.9)
    cone.translate(0, tier.y0 + h / 2, 0)
    // Ragged skirt: the rim droops unevenly
    const pos = cone.getAttribute('position')
    for (let i = 0; i < pos.count; i++) {
      if (pos.getY(i) < tier.y0 + 0.01) pos.setY(i, pos.getY(i) - 0.04 * fract(Math.sin(i * 12.9898 + k * 78.233) * 43758.5453))
    }
    cone.computeVertexNormals()
    parts.push(tagPart(cone, 1, (_x, y) => (0.35 + 0.65 * smoothstep(tier.y0 - 0.05, tier.y1, y)) * (0.75 + 0.125 * k)))
  })
  const g = mergeGeometries(parts)
  parts.forEach(p => p.dispose())
  return g
}

// A broadleaf: a lumpy crown of a few merged blobs on a taller trunk. The normals blend each
// blob's own with the direction from the crown's centre, so the crown shades as one mass.
function buildBroadleaf() {
  const parts: THREE.BufferGeometry[] = []
  const trunk = new THREE.CylinderGeometry(0.03, 0.05, 0.55, 5, 1, true)
  trunk.translate(0, 0.225, 0)
  parts.push(tagPart(trunk, 0, () => 0.4))
  const crown = new THREE.Vector3(0, 0.62, 0)
  const blobs = [
    { c: [0, 0.64, 0], r: 0.34 },
    { c: [0.2, 0.52, 0.12], r: 0.24 },
    { c: [-0.17, 0.55, -0.15], r: 0.23 },
    { c: [-0.04, 0.48, 0.22], r: 0.21 },
    { c: [0.06, 0.8, -0.04], r: 0.2 },
  ]
  const v = new THREE.Vector3()
  const nb = new THREE.Vector3()
  const nc = new THREE.Vector3()
  blobs.forEach((b, k) => {
    let g: THREE.BufferGeometry = new THREE.IcosahedronGeometry(1, 0)
    g.deleteAttribute('normal')
    g.deleteAttribute('uv')
    g = mergeVertices(g)
    const pos = g.getAttribute('position')
    const nrm = new Float32Array(pos.count * 3)
    for (let i = 0; i < pos.count; i++) {
      v.set(pos.getX(i), pos.getY(i), pos.getZ(i))
      nb.copy(v).normalize()
      // Lumpy silhouette: push each vertex in or out a little
      const j = 1 + 0.14 * (fract(Math.sin(i * 91.7 + k * 37.1) * 43758.5453) - 0.5)
      v.multiplyScalar(b.r * j)
      v.y *= 0.85
      v.add(nc.set(b.c[0], b.c[1], b.c[2]))
      pos.setXYZ(i, v.x, v.y, v.z)
      nc.copy(v).sub(crown).normalize()
      nb.lerp(nc, 0.55).normalize()
      nrm[i * 3] = nb.x
      nrm[i * 3 + 1] = nb.y
      nrm[i * 3 + 2] = nb.z
    }
    g.setAttribute('normal', new THREE.BufferAttribute(nrm, 3))
    parts.push(tagPart(g, 1, (x, y, z) => {
      const out = Math.min(1, Math.hypot(x - crown.x, (y - crown.y) * 1.2, z - crown.z) / 0.42)
      return (0.3 + 0.7 * smoothstep(0.3, 0.95, y)) * (0.6 + 0.4 * out)
    }))
  })
  const g = mergeGeometries(parts)
  parts.forEach(p => p.dispose())
  return g
}

export default function TerrainFlight() {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!containerRef.current) return
    const container = containerRef.current

    const renderer = new THREE.WebGLRenderer({ antialias: false })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(window.innerWidth, window.innerHeight)
    renderer.setClearColor(SKY_CLEAR_COLOR)
    renderer.autoClear = false // the landscape and the trees share one color + depth target
    container.appendChild(renderer.domElement)

    // The raymarch and the trees render into a small multisampled target that the blit
    // pass stretches to the screen
    let renderScale = RENDER_SCALE
    const targetSize = () => {
      const pixels = window.innerWidth * window.innerHeight * renderScale * renderScale
      const scale = renderScale * Math.min(1, Math.sqrt(MAX_RENDER_PIXELS / Math.max(1, pixels)))
      return [
        Math.max(1, Math.floor(window.innerWidth * scale)),
        Math.max(1, Math.floor(window.innerHeight * scale)),
      ]
    }
    const [w0, h0] = targetSize()
    const target = new THREE.WebGLRenderTarget(w0, h0, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      depthBuffer: true,
      samples: MSAA_SAMPLES,
    })
    // Depth is only needed while drawing; skip copying it out of the multisampled buffer
    target.resolveDepthBuffer = false

    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1)
    const quad = new THREE.PlaneGeometry(2, 2)

    // Rasterized trees are seen through a perspective camera that matches the raymarch
    // camera exactly: same position, basis, vertical field of view and aspect
    const fovDeg = THREE.MathUtils.radToDeg(2 * Math.atan(TAN_HALF_FOV))
    const treeCamera = new THREE.PerspectiveCamera(fovDeg, w0 / h0, CAMERA_NEAR, CAMERA_FAR)
    // A wider twin, only for choosing which forest tiles go into the instance buffers
    const cullCamera = new THREE.PerspectiveCamera(
      THREE.MathUtils.radToDeg(2 * Math.atan(TAN_HALF_FOV * CULL_FOV_MARGIN)),
      (w0 / h0) * CULL_ASPECT_MARGIN, CAMERA_NEAR, CAMERA_FAR)

    const uniforms = {
      uResolution: { value: new THREE.Vector2(w0, h0) },
      uTime: { value: 0 },
      uCamPos: { value: new THREE.Vector3() },
      uCamRight: { value: new THREE.Vector3() },
      uCamUp: { value: new THREE.Vector3() },
      uCamFwd: { value: new THREE.Vector3() },
      uTanHalfFov: { value: TAN_HALF_FOV },
      uNear: { value: CAMERA_NEAR },
      uFar: { value: CAMERA_FAR },
      uOriginW: { value: new THREE.Vector2() },
      uSunDir: { value: new THREE.Vector3() },
      uMoonDir: { value: new THREE.Vector3() },
      uLightDir: { value: new THREE.Vector3() },
      uLightCol: { value: new THREE.Color() },
      uZenith: { value: new THREE.Color() },
      uHorizon: { value: new THREE.Color() },
      uSunGlow: { value: new THREE.Color() },
      uAmbient: { value: new THREE.Color() },
      uNight: { value: 0 },
      uStarAngle: { value: 0 },
      uCloudOffset: { value: new THREE.Vector2() },
      uWave1: { value: new THREE.Vector2() },
      uWave2: { value: new THREE.Vector2() },
      uCloudCover: { value: 0.4 },
      uFogDensity: { value: 0.12 },
      uWater: { value: -0.15 },
      uGrass: { value: new THREE.Color() },
      uRock: { value: new THREE.Color() },
      uSand: { value: new THREE.Color() },
      uWaterCol: { value: new THREE.Color() },
      uSnowLine: { value: 0.4 },
      uStrata: { value: 0 },
      // Trees
      uTreeDensity: { value: 0 },
      uTreeSize: { value: 1 },
      uConifer: { value: 0.5 },
      uThinDistance: { value: TREE_THIN_DISTANCE },
      uFadeStart: { value: TREE_FADE_START },
      uRadius: { value: TREE_RADIUS },
      uTrunk: { value: new THREE.Color(TRUNK_COLOR) },
      uShadowStrength: { value: 0 },
    }
    const terrainMaterial = new THREE.ShaderMaterial({
      vertexShader: QUAD_VERT,
      fragmentShader: TERRAIN_FRAG,
      uniforms,
      // Depth comes from the ray hit (gl_FragDepth); the test must be on for it to be written
      depthTest: true,
      depthWrite: true,
      depthFunc: THREE.AlwaysDepth,
    })
    const blitMaterial = new THREE.ShaderMaterial({
      vertexShader: QUAD_VERT,
      fragmentShader: BLIT_FRAG,
      uniforms: { uScene: { value: target.texture } },
      depthTest: false,
      depthWrite: false,
    })
    // The landscape quad is drawn first in the same scene (and the same render call) as the
    // trees: three.js discards a multisampled target's color after each render() into it
    const terrainQuad = new THREE.Mesh(quad, terrainMaterial)
    terrainQuad.frustumCulled = false
    terrainQuad.renderOrder = -1
    const blitScene = new THREE.Scene()
    blitScene.add(new THREE.Mesh(quad, blitMaterial))

    // ---- trees: two instanced meshes (conifer, broadleaf) and their ground shadows, all
    // drawing the same instance buffers; each instance shows as one species or the other
    const coniferGeometry = buildConifer()
    const broadleafGeometry = buildBroadleaf()
    const shadowGeometry = new THREE.PlaneGeometry(2, 2)
    const tintAttr = new THREE.InstancedBufferAttribute(new Float32Array(MAX_TREES * 4), 4)
    const miscAttr = new THREE.InstancedBufferAttribute(new Float32Array(MAX_TREES * 4), 4)
    tintAttr.setUsage(THREE.DynamicDrawUsage)
    miscAttr.setUsage(THREE.DynamicDrawUsage)
    for (const g of [coniferGeometry, broadleafGeometry, shadowGeometry]) {
      g.setAttribute('aTint', tintAttr)
      g.setAttribute('aMisc', miscAttr)
    }
    const treeMaterial = (species: number) => new THREE.ShaderMaterial({
      vertexShader: TREE_VERT,
      fragmentShader: TREE_FRAG,
      uniforms: {
        ...uniforms,
        uSpecies: { value: species },
        uColA: { value: new THREE.Color() },
        uColB: { value: new THREE.Color() },
      },
    })
    const coniferMaterial = treeMaterial(1)
    const broadleafMaterial = treeMaterial(0)
    const shadowMaterial = new THREE.ShaderMaterial({
      vertexShader: SHADOW_VERT,
      fragmentShader: SHADOW_FRAG,
      uniforms,
      side: THREE.DoubleSide,
      depthWrite: false,
      // frame color *= shadow color
      blending: THREE.CustomBlending,
      blendEquation: THREE.AddEquation,
      blendSrc: THREE.ZeroFactor,
      blendDst: THREE.SrcColorFactor,
    })
    const conifers = new THREE.InstancedMesh(coniferGeometry, coniferMaterial, MAX_TREES)
    const broadleaves = new THREE.InstancedMesh(broadleafGeometry, broadleafMaterial, MAX_TREES)
    const shadows = new THREE.InstancedMesh(shadowGeometry, shadowMaterial, MAX_TREES)
    const instanceMatrix = conifers.instanceMatrix
    instanceMatrix.setUsage(THREE.DynamicDrawUsage)
    broadleaves.instanceMatrix = instanceMatrix
    shadows.instanceMatrix = instanceMatrix
    const scene = new THREE.Scene()
    scene.add(terrainQuad)
    // Landscape first, then tree shadows (they only darken the ground), then the trees
    shadows.renderOrder = 0
    conifers.renderOrder = 1
    broadleaves.renderOrder = 1
    for (const m of [shadows, conifers, broadleaves]) {
      m.count = 0
      m.visible = false
      m.frustumCulled = false // the forest manager already culls by tile
      scene.add(m)
    }

    // ---- biome + flight style, re-randomized periodically with a smooth crossfade
    interface Look {
      grass: THREE.Color
      rock: THREE.Color
      sand: THREE.Color
      water: THREE.Color
      snowLine: number
      strata: number
      waterLevel: number
      conA: THREE.Color
      conB: THREE.Color
      leafA: THREE.Color
      leafB: THREE.Color
      treeDensity: number
      treeSize: number
      coniferShare: number
      style: FlightStyle
    }
    const lookFromBiome = (b: Biome, style: FlightStyle): Look => ({
      grass: new THREE.Color(b.grass),
      rock: new THREE.Color(b.rock),
      sand: new THREE.Color(b.sand),
      water: new THREE.Color(b.water),
      snowLine: b.snowLine,
      strata: b.strata,
      waterLevel: b.waterLevel,
      conA: new THREE.Color(b.conifer[0]),
      conB: new THREE.Color(b.conifer[1]),
      leafA: new THREE.Color(b.leaf[0]),
      leafB: new THREE.Color(b.leaf[1]),
      treeDensity: b.treeDensity,
      treeSize: b.treeSize,
      coniferShare: b.coniferShare,
      style,
    })
    const cloneLook = (l: Look): Look => ({
      ...l,
      grass: l.grass.clone(),
      rock: l.rock.clone(),
      sand: l.sand.clone(),
      water: l.water.clone(),
      conA: l.conA.clone(),
      conB: l.conB.clone(),
      leafA: l.leafA.clone(),
      leafB: l.leafB.clone(),
      style: { ...l.style },
    })
    let biomeIndex = Math.floor(Math.random() * BIOMES.length)
    let fromLook = lookFromBiome(BIOMES[biomeIndex], randomStyle())
    let toLook = fromLook
    const look: Look = cloneLook(fromLook)
    let blendStart = -Infinity
    let nextBiomeAt = rand(BIOME_MIN_SECONDS, BIOME_MAX_SECONDS)

    const rerandomize = (now: number) => {
      let next = Math.floor(Math.random() * (BIOMES.length - 1))
      if (next >= biomeIndex) next++
      biomeIndex = next
      // Start from wherever the current blend has got to, so nothing ever jumps
      fromLook = cloneLook(look)
      toLook = lookFromBiome(BIOMES[next], randomStyle())
      blendStart = now
    }

    // Snow lines of "no snow" biomes are huge; blend in a clamped range so snow recedes over
    // the whole blend instead of vanishing in the first second
    const snowOf = (l: Look) => Math.min(l.snowLine, 1.6)

    const updateLook = (now: number) => {
      const k = smoothstep(0, 1, (now - blendStart) / BIOME_BLEND_SECONDS)
      const lerp = (a: number, b: number) => a + (b - a) * k
      look.grass.copy(fromLook.grass).lerp(toLook.grass, k)
      look.rock.copy(fromLook.rock).lerp(toLook.rock, k)
      look.sand.copy(fromLook.sand).lerp(toLook.sand, k)
      look.water.copy(fromLook.water).lerp(toLook.water, k)
      look.conA.copy(fromLook.conA).lerp(toLook.conA, k)
      look.conB.copy(fromLook.conB).lerp(toLook.conB, k)
      look.leafA.copy(fromLook.leafA).lerp(toLook.leafA, k)
      look.leafB.copy(fromLook.leafB).lerp(toLook.leafB, k)
      look.treeDensity = lerp(fromLook.treeDensity, toLook.treeDensity)
      look.treeSize = lerp(fromLook.treeSize, toLook.treeSize)
      look.coniferShare = lerp(fromLook.coniferShare, toLook.coniferShare)
      look.snowLine = lerp(snowOf(fromLook), snowOf(toLook))
      look.strata = lerp(fromLook.strata, toLook.strata)
      look.waterLevel = lerp(fromLook.waterLevel, toLook.waterLevel)
      look.style.clearance = lerp(fromLook.style.clearance, toLook.style.clearance)
      look.style.speed = lerp(fromLook.style.speed, toLook.style.speed)
      look.style.fogDensity = lerp(fromLook.style.fogDensity, toLook.style.fogDensity)
      look.style.cloudCover = lerp(fromLook.style.cloudCover, toLook.style.cloudCover)
    }

    // ---- flight state (see Flight in terrainMath.ts). Position is wrapped mod PERIOD (the
    // terrain is periodic), so it never grows and float32 uniforms stay precise forever.
    const flightParams = () => ({
      speed: look.style.speed,
      clearance: look.style.clearance,
      waterLevel: look.waterLevel,
    })
    const flight = new Flight(
      Math.random() * PERIOD,
      Math.random() * PERIOD,
      Math.random() * Math.PI * 2,
      [rand(0, 6.28), rand(0, 6.28), rand(0, 6.28)],
      flightParams(),
    )

    // ---- forest. Trees live in unwrapped world coordinates that follow the camera
    // continuously across the terrain's wrap, relative to a floating origin near the camera
    // (keeps float32 vertex positions precise however long it runs).
    const forest = new Forest()
    let camUX = flight.x
    let camUZ = flight.z
    let prevFlightX = flight.x
    let prevFlightZ = flight.z
    forest.recenter(camUX, camUZ)
    const frustum = new THREE.Frustum()
    const cullMatrix = new THREE.Matrix4()
    const box = new THREE.Box3()
    const inView = (x0: number, y0: number, z0: number, x1: number, y1: number, z1: number) =>
      frustum.intersectsBox(box.set(box.min.set(x0, y0, z0), box.max.set(x1, y1, z1)))
    const forestBounds: ForestBounds = { density: 0, waterMin: 0, snowMax: 0 }
    let rebuildTimer = 0
    const rebuildForest = () => {
      cullCamera.position.copy(treeCamera.position)
      cullCamera.quaternion.copy(treeCamera.quaternion)
      cullCamera.updateMatrixWorld()
      frustum.setFromProjectionMatrix(cullMatrix.multiplyMatrices(cullCamera.projectionMatrix, cullCamera.matrixWorldInverse))
      // Anything the biome blend could reach before the next rebuild
      forestBounds.density = Math.max(look.treeDensity, fromLook.treeDensity, toLook.treeDensity)
      forestBounds.waterMin = Math.min(look.waterLevel, fromLook.waterLevel, toLook.waterLevel)
      forestBounds.snowMax = Math.max(look.snowLine, snowOf(fromLook), snowOf(toLook))
      const n = forest.rebuild(inView, forestBounds,
        instanceMatrix.array as Float32Array, tintAttr.array as Float32Array, miscAttr.array as Float32Array, MAX_TREES)
      for (const m of [shadows, conifers, broadleaves]) {
        m.count = n
        m.visible = n > 0
      }
      // Upload only the part in use
      for (const [attr, size] of [[instanceMatrix, 16], [tintAttr, 4], [miscAttr, 4]] as const) {
        attr.clearUpdateRanges()
        attr.addUpdateRange(0, Math.max(n, 1) * size)
        attr.needsUpdate = true
      }
      uniforms.uOriginW.value.set(mod(forest.originX, PERIOD), mod(forest.originZ, PERIOD))
    }

    // Day phase: 0 = sunrise, pi/2 = noon, pi = sunset. Start somewhere in the morning.
    let dayPhase = rand(-0.2, 0.9)
    let starAngle = rand(0, 6.28)
    const cloudOffset = new THREE.Vector2(rand(0, PERIOD), rand(0, PERIOD))
    const wave1 = new THREE.Vector2()
    const wave2 = new THREE.Vector2()
    const windAngle = rand(0, 6.28)

    // Scratch objects reused every frame
    const fwd = new THREE.Vector3()
    const right0 = new THREE.Vector3()
    const up0 = new THREE.Vector3()
    const back = new THREE.Vector3()
    const basis = new THREE.Matrix4()
    const tmpColor = new THREE.Color()
    const moonLight = new THREE.Color(MOONLIGHT)
    const nightAmbient = new THREE.Color(NIGHT_AMBIENT)
    const luminance = (c: THREE.Color) => 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b

    const updateSky = () => {
      const sp = Math.sin(dayPhase)
      const cp = Math.cos(dayPhase)
      // The sun arcs across a tilted plane; elevation is biased up so days outlast nights
      const sun = uniforms.uSunDir.value.set(-cp, sp * 0.9 + 0.1, 0.45).normalize()
      const moon = uniforms.uMoonDir.value.set(cp, -sp * 0.9 + 0.05, -0.4).normalize()
      const elevation = sun.y
      // Rising (morning) uses the gold dawn palette, setting uses the magenta dusk one;
      // blend between them around noon and midnight so neither switch is visible
      const duskMix = smoothstep(0.35, -0.35, cp)
      const zen = uniforms.uZenith.value
      const hor = uniforms.uHorizon.value
      const glow = uniforms.uSunGlow.value
      sampleStops(DAWN_COLORS.zenith, elevation, zen)
      zen.lerp(sampleStops(DUSK_COLORS.zenith, elevation, tmpColor), duskMix)
      sampleStops(DAWN_COLORS.horizon, elevation, hor)
      hor.lerp(sampleStops(DUSK_COLORS.horizon, elevation, tmpColor), duskMix)
      sampleStops(DAWN_COLORS.glow, elevation, glow)
      glow.lerp(sampleStops(DUSK_COLORS.glow, elevation, tmpColor), duskMix)

      // Key light: the sun by day, the moon by night. Both are dark at the hand-over
      // (sun just below the horizon) so switching direction is invisible.
      const sunStrength = smoothstep(-0.04, 0.2, elevation)
      const moonStrength = smoothstep(0.04, 0.12, -elevation) * smoothstep(-0.02, 0.15, moon.y)
      if (elevation > -0.04) {
        uniforms.uLightDir.value.copy(sun)
        uniforms.uLightCol.value.copy(glow).multiplyScalar(sunStrength * 1.25)
      } else {
        uniforms.uLightDir.value.copy(moon)
        uniforms.uLightCol.value.copy(moonLight).multiplyScalar(moonStrength * 0.3)
      }
      uniforms.uAmbient.value.copy(zen).lerp(hor, 0.35).multiplyScalar(0.55).add(nightAmbient)
      uniforms.uNight.value = 1 - smoothstep(-0.22, -0.02, elevation)

      // Tree shadows are as deep as the key light's share of the light on open ground
      const L = uniforms.uLightDir.value
      const direct = 2.2 * luminance(uniforms.uLightCol.value) * Math.max(L.y, 0)
      const ambient = luminance(uniforms.uAmbient.value)
      uniforms.uShadowStrength.value = L.y > 0.02 ? 0.8 * direct / (direct + ambient + 1e-4) : 0
      return elevation
    }

    const updateCamera = (dt: number, time: number) => {
      flight.step(dt, time, flightParams())
      const dirX = Math.cos(flight.heading)
      const dirZ = Math.sin(flight.heading)
      const pitch = flight.pitch
      const roll = flight.roll

      const cp = Math.cos(pitch)
      fwd.set(dirX * cp, Math.sin(pitch), dirZ * cp)
      right0.set(-dirZ, 0, dirX)
      up0.crossVectors(right0, fwd)
      const cr = Math.cos(roll)
      const sr = Math.sin(roll)
      uniforms.uCamFwd.value.copy(fwd)
      uniforms.uCamUp.value.copy(up0).multiplyScalar(cr).addScaledVector(right0, sr)
      uniforms.uCamRight.value.copy(right0).multiplyScalar(cr).addScaledVector(up0, -sr)
      uniforms.uCamPos.value.set(flight.x, flight.y, flight.z)

      // Unwrapped camera track (flight.x/z wrap mod PERIOD)
      const wrapDelta = (d: number) => d - PERIOD * Math.round(d / PERIOD)
      camUX += wrapDelta(flight.x - prevFlightX)
      camUZ += wrapDelta(flight.z - prevFlightZ)
      prevFlightX = flight.x
      prevFlightZ = flight.z

      // The tree camera: same eye and basis (x right, y up, looking down -z = forward)
      treeCamera.position.set(camUX - forest.originX, flight.y, camUZ - forest.originZ)
      back.copy(fwd).negate()
      basis.makeBasis(uniforms.uCamRight.value, uniforms.uCamUp.value, back)
      treeCamera.quaternion.setFromRotationMatrix(basis)
      treeCamera.updateMatrixWorld()
    }

    const handleResize = () => {
      renderer.setSize(window.innerWidth, window.innerHeight)
      const [w, h] = targetSize()
      target.setSize(w, h)
      uniforms.uResolution.value.set(w, h)
      treeCamera.aspect = w / h
      treeCamera.updateProjectionMatrix()
      cullCamera.aspect = (w / h) * CULL_ASPECT_MARGIN
      cullCamera.updateProjectionMatrix()
    }
    window.addEventListener('resize', handleResize)

    // Quality ratchet: if the GPU can't keep up, permanently drop the raymarch resolution
    // a notch (never back up, so it can't oscillate between sizes)
    let slowSum = 0
    let slowCount = 0

    const startTime = performance.now()
    let lastTime = startTime
    let animationId = 0
    let firstFrame = true

    const animate = () => {
      animationId = requestAnimationFrame(animate)
      const now = performance.now()
      const rawDt = (now - lastTime) / 1000
      lastTime = now
      // Clamp so a backgrounded tab doesn't take one giant step on return
      const dt = Math.min(rawDt, 1 / 20)
      const time = (now - startTime) / 1000

      if (rawDt < 0.1 && !firstFrame) {
        slowSum += rawDt
        slowCount++
        if (slowCount >= 120) {
          if (slowSum / slowCount > SLOW_FRAME_SECONDS && renderScale > MIN_RENDER_SCALE) {
            renderScale = Math.max(MIN_RENDER_SCALE, renderScale - 0.07)
            handleResize()
          }
          slowSum = 0
          slowCount = 0
        }
      }

      if (time >= nextBiomeAt) {
        rerandomize(time)
        nextBiomeAt = time + rand(BIOME_MIN_SECONDS, BIOME_MAX_SECONDS)
      }
      updateLook(time)

      const elevation = updateSky()
      // Nights pass faster than days
      const nightness = smoothstep(0.0, -0.25, elevation)
      dayPhase = mod(dayPhase + ((Math.PI * 2) / DAY_SECONDS) * (1 + NIGHT_SPEEDUP * nightness) * dt, Math.PI * 2)
      starAngle = mod(starAngle + 0.012 * dt, Math.PI * 2)

      updateCamera(dt, time)

      // Forest: populate nearby tiles (the first frame does most of the view at once), and
      // periodically rewrite the visible instance list
      const recentered = forest.recenter(camUX, camUZ)
      if (recentered) {
        treeCamera.position.set(camUX - forest.originX, flight.y, camUZ - forest.originZ)
        treeCamera.updateMatrixWorld()
      }
      forest.update(camUX, camUZ, fwd.x, fwd.z, firstFrame ? FOREST_STARTUP_MS : FOREST_BUDGET_MS)
      rebuildTimer -= dt
      if (rebuildTimer <= 0 || recentered || firstFrame) {
        rebuildTimer = FOREST_REBUILD_SECONDS
        rebuildForest()
      }
      firstFrame = false

      const windX = Math.cos(windAngle)
      const windZ = Math.sin(windAngle)
      cloudOffset.set(mod(cloudOffset.x + windX * 0.02 * dt, PERIOD), mod(cloudOffset.y + windZ * 0.02 * dt, PERIOD))
      wave1.set(mod(wave1.x + windX * 0.35 * dt, PERIOD), mod(wave1.y + windZ * 0.35 * dt, PERIOD))
      wave2.set(mod(wave2.x - windZ * 0.5 * dt, PERIOD), mod(wave2.y + windX * 0.6 * dt, PERIOD))

      uniforms.uTime.value = time % 1000 // only drives star twinkle; wrap keeps it precise
      uniforms.uStarAngle.value = starAngle
      uniforms.uCloudOffset.value.copy(cloudOffset)
      uniforms.uWave1.value.copy(wave1)
      uniforms.uWave2.value.copy(wave2)
      uniforms.uCloudCover.value = look.style.cloudCover
      uniforms.uFogDensity.value = look.style.fogDensity
      uniforms.uWater.value = look.waterLevel
      uniforms.uGrass.value.copy(look.grass)
      uniforms.uRock.value.copy(look.rock)
      uniforms.uSand.value.copy(look.sand)
      uniforms.uWaterCol.value.copy(look.water)
      uniforms.uSnowLine.value = look.snowLine
      uniforms.uStrata.value = look.strata
      uniforms.uTreeDensity.value = look.treeDensity
      uniforms.uTreeSize.value = look.treeSize
      uniforms.uConifer.value = look.coniferShare
      coniferMaterial.uniforms.uColA.value.copy(look.conA)
      coniferMaterial.uniforms.uColB.value.copy(look.conB)
      broadleafMaterial.uniforms.uColA.value.copy(look.leafA)
      broadleafMaterial.uniforms.uColB.value.copy(look.leafB)

      renderer.setRenderTarget(target)
      renderer.clear(true, true, false)
      renderer.render(scene, treeCamera)
      renderer.setRenderTarget(null)
      renderer.render(blitScene, camera)
    }

    animate()

    return () => {
      window.removeEventListener('resize', handleResize)
      cancelAnimationFrame(animationId)
      quad.dispose()
      terrainMaterial.dispose()
      blitMaterial.dispose()
      coniferGeometry.dispose()
      broadleafGeometry.dispose()
      shadowGeometry.dispose()
      coniferMaterial.dispose()
      broadleafMaterial.dispose()
      shadowMaterial.dispose()
      conifers.dispose()
      broadleaves.dispose()
      shadows.dispose()
      target.dispose()
      renderer.dispose()
      container.removeChild(renderer.domElement)
    }
  }, [])

  return <div ref={containerRef} className="w-full h-full" />
}
