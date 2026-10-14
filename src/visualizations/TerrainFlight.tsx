import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { PERIOD, RAYMARCH_OCTAVES, terrainHeight } from './terrainMath'

// Configuration constants
const RENDER_SCALE = 0.5 // raymarch resolution relative to CSS pixels, then upscaled
const MIN_RENDER_SCALE = 0.3 // floor for the automatic quality ratchet on slow GPUs
const SLOW_FRAME_SECONDS = 1 / 45 // average frame time that triggers a resolution drop
const DAY_SECONDS = 240 // one full day/night cycle (night runs faster, see below)
const NIGHT_SPEEDUP = 0.7 // night passes this much faster than day
const BIOME_MIN_SECONDS = 45 // time between re-randomized biomes / flight styles
const BIOME_MAX_SECONDS = 80
const BIOME_BLEND_SECONDS = 15
const TAN_HALF_FOV = 0.5 // vertical, about 53 degrees
const LOOKAHEAD_DISTANCE = 1.6 // how far ahead (world units) the camera checks the ground
const LOOKAHEAD_SAMPLES = 17
const CLIMB_SLOPE = 0.07 // the camera starts rising this far ahead of a ridge (rise/run)
const MIN_CLEARANCE = 0.035 // hard floor above the ground directly beneath / just ahead
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
}

const BIOMES: Biome[] = [
  // Alpine: green valleys, grey rock, snow caps, cold lakes
  { grass: '#3d5a1e', rock: '#6e655c', sand: '#a89a7a', water: '#0d2a33', snowLine: 0.36, strata: 0.1, waterLevel: -0.15 },
  // Autumn highlands: rust and gold slopes
  { grass: '#8a5420', rock: '#6a5a50', sand: '#a08a68', water: '#102a30', snowLine: 0.55, strata: 0.15, waterLevel: -0.14 },
  // Red desert canyons: banded sandstone, sparse scrub, turquoise water
  { grass: '#7a6234', rock: '#a0502a', sand: '#c89a68', water: '#0c3a3a', snowLine: 9, strata: 0.85, waterLevel: -0.24 },
  // Heather moors: purple-brown slopes, slate rock
  { grass: '#5a3a52', rock: '#5c5c64', sand: '#8a8070', water: '#0c1e2e', snowLine: 0.62, strata: 0.2, waterLevel: -0.12 },
  // Emerald lowlands: lush green, dark rock, lots of water
  { grass: '#2a6a2c', rock: '#4e4a42', sand: '#b0a07a', water: '#0a3030', snowLine: 0.8, strata: 0.05, waterLevel: -0.08 },
  // Volcanic: black rock, acid-green moss, heavy snow
  { grass: '#4a5a1a', rock: '#2a2628', sand: '#4a4440', water: '#081a20', snowLine: 0.28, strata: 0.3, waterLevel: -0.18 },
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
  clearance: number // preferred height above the look-ahead ground envelope
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

// The landscape: raymarched heightfield, water with reflections, sky, fog, tonemapping.
// No ${} interpolations so the shipped source is byte-identical to what was validated.
const TERRAIN_FRAG = /* glsl */ `
varying vec2 vUv;

uniform vec2 uResolution;
uniform float uTime;
uniform vec3 uCamPos;
uniform vec3 uCamRight;
uniform vec3 uCamUp;
uniform vec3 uCamFwd;
uniform float uTanHalfFov;
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
#define FAR 16.0
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

// ---------------------------------------------------------------- sky

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

// ---------------------------------------------------------------- terrain

float march(vec3 ro, vec3 rd, float tmax) {
  float t = 0.004;
  for (int i = 0; i < 150; i++) {
    vec3 p = ro + rd * t;
    if (t > tmax || (rd.y >= 0.0 && p.y > MAXH)) return 1e5;
    float h = p.y - terrain(p.xz, 7);
    if (abs(h) < 0.0005 * t) return t;
    t += 0.4 * h;
  }
  return t < tmax ? t : 1e5;
}

// Cheaper march for reflections in water
float marchRefl(vec3 ro, vec3 rd, float tmax) {
  float t = 0.004;
  for (int i = 0; i < 60; i++) {
    vec3 p = ro + rd * t;
    if (t > tmax || p.y > MAXH) return 1e5;
    float h = p.y - terrain(p.xz, 5);
    if (abs(h) < 0.002 * t) return t;
    t += 0.5 * h;
  }
  return 1e5;
}

vec3 calcNormal(vec3 p, float t, int oct) {
  float e = max(0.0004, 0.0012 * t);
  return normalize(vec3(
    terrain(p.xz - vec2(e, 0.0), oct) - terrain(p.xz + vec2(e, 0.0), oct),
    2.0 * e,
    terrain(p.xz - vec2(0.0, e), oct) - terrain(p.xz + vec2(0.0, e), oct)));
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

// Aerial perspective: distance haze plus mist that pools in the valleys
vec3 applyFog(vec3 col, float t, vec3 ro, vec3 rd) {
  float hb = 2.2;
  float base = exp(-max(ro.y, -1.0) * hb);
  float k = rd.y * hb;
  float heightTerm = abs(k) < 1e-3 ? base * t : base * (1.0 - exp(-t * k)) / k;
  float dens = uFogDensity * (0.35 * t + 0.6 * heightTerm);
  float fo = 1.0 - exp(-dens);
  fo = max(fo, smoothstep(FAR * 0.65, FAR, t));
  return mix(col, skyColor(rd), fo);
}

vec3 aces(vec3 x) {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
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
  if (t < tmax) {
    vec3 p = ro + rd * t;
    vec3 n = calcNormal(p, t, 10);
    float sh = 1.0;
    if (lit && dot(n, uLightDir) > 0.0) sh = softShadow(p + n * 0.004, uLightDir);
    col = shadeTerrain(p, n, rd, sh);
    col = applyFog(col, t, ro, rd);
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
      vec3 nq = calcNormal(q, tr + tWater, 6);
      refl = shadeTerrain(q, nq, rr, 1.0);
      refl = applyFog(refl, tr, p, rr);
    } else {
      refl = skyFull(rr);
    }

    float sh = lit ? softShadow(p + vec3(0.0, 0.004, 0.0), uLightDir) : 0.0;
    float depth = uWater - terrain(p.xz, 4);
    vec3 body = uWaterCol * (uAmbient * 0.8 + uLightCol * (0.3 + 0.5 * sh) * max(uLightDir.y, 0.0));
    vec3 shallow = uSand * (uAmbient * 0.7 + uLightCol * sh * max(uLightDir.y, 0.0) * 1.6) * 0.6;
    body = mix(shallow, body, smoothstep(0.0, 0.06, depth));
    col = mix(body, refl, fres);
    col += uLightCol * 3.0 * pow(max(dot(rr, uLightDir), 0.0), 350.0) * sh;
    // A thin bright lip where water meets the shore
    float foam = smoothstep(0.012, 0.0, depth) * (0.6 + 0.4 * w2.x);
    col = mix(col, (uAmbient + uLightCol * sh) * 0.7, foam * 0.5);
    col = applyFog(col, tWater, ro, rd);
  } else {
    col = skyFull(rd);
  }

  col = aces(col * 1.1);
  col = pow(col, vec3(1.0 / 2.2));
  // Dither before the 8-bit render target to keep dusk gradients from banding
  float dn = hash01(mod(gl_FragCoord.xy, 4096.0));
  col += (dn - 0.5) / 255.0;
  gl_FragColor = vec4(col, 1.0);
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
  clearance: rand(0.07, 0.28),
  speed: rand(0.13, 0.2),
  fogDensity: rand(0.08, 0.17),
  cloudCover: rand(0, 0.85),
})

export default function TerrainFlight() {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!containerRef.current) return
    const container = containerRef.current

    const renderer = new THREE.WebGLRenderer({ antialias: false })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(window.innerWidth, window.innerHeight)
    renderer.setClearColor(SKY_CLEAR_COLOR)
    container.appendChild(renderer.domElement)

    // The raymarch renders into a small target that the blit pass stretches to the screen
    let renderScale = RENDER_SCALE
    const targetSize = () => [
      Math.max(1, Math.floor(window.innerWidth * renderScale)),
      Math.max(1, Math.floor(window.innerHeight * renderScale)),
    ]
    const [w0, h0] = targetSize()
    const target = new THREE.WebGLRenderTarget(w0, h0, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      depthBuffer: false,
    })

    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1)
    const quad = new THREE.PlaneGeometry(2, 2)

    const uniforms = {
      uResolution: { value: new THREE.Vector2(w0, h0) },
      uTime: { value: 0 },
      uCamPos: { value: new THREE.Vector3() },
      uCamRight: { value: new THREE.Vector3() },
      uCamUp: { value: new THREE.Vector3() },
      uCamFwd: { value: new THREE.Vector3() },
      uTanHalfFov: { value: TAN_HALF_FOV },
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
    }
    const terrainMaterial = new THREE.ShaderMaterial({
      vertexShader: QUAD_VERT,
      fragmentShader: TERRAIN_FRAG,
      uniforms,
      depthTest: false,
      depthWrite: false,
    })
    const blitMaterial = new THREE.ShaderMaterial({
      vertexShader: QUAD_VERT,
      fragmentShader: BLIT_FRAG,
      uniforms: { uScene: { value: target.texture } },
      depthTest: false,
      depthWrite: false,
    })
    const terrainScene = new THREE.Scene()
    terrainScene.add(new THREE.Mesh(quad, terrainMaterial))
    const blitScene = new THREE.Scene()
    blitScene.add(new THREE.Mesh(quad, blitMaterial))

    // ---- biome + flight style, re-randomized periodically with a smooth crossfade
    interface Look {
      grass: THREE.Color
      rock: THREE.Color
      sand: THREE.Color
      water: THREE.Color
      snowLine: number
      strata: number
      waterLevel: number
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
      style,
    })
    let biomeIndex = Math.floor(Math.random() * BIOMES.length)
    let fromLook = lookFromBiome(BIOMES[biomeIndex], randomStyle())
    let toLook = fromLook
    const look: Look = lookFromBiome(BIOMES[biomeIndex], { ...fromLook.style })
    let blendStart = -Infinity
    let nextBiomeAt = rand(BIOME_MIN_SECONDS, BIOME_MAX_SECONDS)

    const rerandomize = (now: number) => {
      let next = Math.floor(Math.random() * (BIOMES.length - 1))
      if (next >= biomeIndex) next++
      biomeIndex = next
      // Start from wherever the current blend has got to, so nothing ever jumps
      fromLook = {
        ...look,
        grass: look.grass.clone(),
        rock: look.rock.clone(),
        sand: look.sand.clone(),
        water: look.water.clone(),
        style: { ...look.style },
      }
      toLook = lookFromBiome(BIOMES[next], randomStyle())
      blendStart = now
    }

    const updateLook = (now: number) => {
      const k = smoothstep(0, 1, (now - blendStart) / BIOME_BLEND_SECONDS)
      const lerp = (a: number, b: number) => a + (b - a) * k
      look.grass.copy(fromLook.grass).lerp(toLook.grass, k)
      look.rock.copy(fromLook.rock).lerp(toLook.rock, k)
      look.sand.copy(fromLook.sand).lerp(toLook.sand, k)
      look.water.copy(fromLook.water).lerp(toLook.water, k)
      // Snow lines of "no snow" biomes are huge; blend in a clamped range so snow
      // recedes over the whole blend instead of vanishing in the first second
      look.snowLine = lerp(Math.min(fromLook.snowLine, 1.6), Math.min(toLook.snowLine, 1.6))
      look.strata = lerp(fromLook.strata, toLook.strata)
      look.waterLevel = lerp(fromLook.waterLevel, toLook.waterLevel)
      look.style.clearance = lerp(fromLook.style.clearance, toLook.style.clearance)
      look.style.speed = lerp(fromLook.style.speed, toLook.style.speed)
      look.style.fogDensity = lerp(fromLook.style.fogDensity, toLook.style.fogDensity)
      look.style.cloudCover = lerp(fromLook.style.cloudCover, toLook.style.cloudCover)
    }

    // ---- flight state. Position is wrapped mod PERIOD (the terrain is periodic), so it
    // never grows and float32 uniforms stay precise forever.
    const ground = (x: number, z: number) => Math.max(terrainHeight(x, z, RAYMARCH_OCTAVES), look.waterLevel)
    let camX = Math.random() * PERIOD
    let camZ = Math.random() * PERIOD
    let heading = Math.random() * Math.PI * 2
    const turnPhases = [rand(0, 6.28), rand(0, 6.28), rand(0, 6.28)]
    let steer = 0
    let roll = 0
    let pitch = -0.08
    let camY = ground(camX, camZ) + 0.2
    let lastAltitude = camY

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
    const tmpColor = new THREE.Color()
    const moonLight = new THREE.Color(MOONLIGHT)
    const nightAmbient = new THREE.Color(NIGHT_AMBIENT)

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
      return elevation
    }

    const updateCamera = (dt: number, time: number) => {
      const speed = look.style.speed
      // Wandering heading: a sum of incommensurate sines, plus a gentle pull toward the
      // lower side so the flight tends to follow valleys rather than slam into ridges
      const wander =
        0.07 * Math.sin(time * 0.031 + turnPhases[0]) +
        0.05 * Math.sin(time * 0.0173 + turnPhases[1]) +
        0.03 * Math.sin(time * 0.057 + turnPhases[2])
      const probe = 0.9
      const gl = ground(camX + Math.cos(heading - 0.5) * probe, camZ + Math.sin(heading - 0.5) * probe)
      const gr = ground(camX + Math.cos(heading + 0.5) * probe, camZ + Math.sin(heading + 0.5) * probe)
      const steerTarget = Math.max(-0.08, Math.min(0.08, (gl - gr) * 0.25))
      steer += (steerTarget - steer) * (1 - Math.exp(-dt / 3))
      const turnRate = wander + steer
      heading += turnRate * dt

      const dirX = Math.cos(heading)
      const dirZ = Math.sin(heading)
      camX = mod(camX + dirX * speed * dt, PERIOD)
      camZ = mod(camZ + dirZ * speed * dt, PERIOD)

      // Altitude: rise early for ridges ahead (an envelope that slopes down with
      // distance), descend lazily into valleys, and never dip below a hard floor
      let envelope = -Infinity
      for (let i = 0; i < LOOKAHEAD_SAMPLES; i++) {
        const s = (i / (LOOKAHEAD_SAMPLES - 1)) * LOOKAHEAD_DISTANCE
        envelope = Math.max(envelope, ground(camX + dirX * s, camZ + dirZ * s) - s * CLIMB_SLOPE)
      }
      const targetY = envelope + look.style.clearance
      const rate = targetY > camY ? 1.1 : 0.35
      camY += (targetY - camY) * (1 - Math.exp(-dt * rate))
      const floor = Math.max(ground(camX, camZ), ground(camX + dirX * 0.08, camZ + dirZ * 0.08)) + MIN_CLEARANCE
      camY = Math.max(camY, floor)

      // Nose follows the climb a little; the camera banks into turns
      const climb = dt > 0 ? (camY - lastAltitude) / dt / speed : 0
      lastAltitude = camY
      const pitchTarget = -0.1 + Math.max(-0.25, Math.min(0.3, climb * 0.5))
      pitch += (pitchTarget - pitch) * (1 - Math.exp(-dt / 1.5))
      const rollTarget = Math.max(-0.3, Math.min(0.3, turnRate * 2.2))
      roll += (rollTarget - roll) * (1 - Math.exp(-dt / 1.2))

      const cp = Math.cos(pitch)
      fwd.set(dirX * cp, Math.sin(pitch), dirZ * cp)
      right0.set(-dirZ, 0, dirX)
      up0.crossVectors(right0, fwd)
      const cr = Math.cos(roll)
      const sr = Math.sin(roll)
      uniforms.uCamFwd.value.copy(fwd)
      uniforms.uCamUp.value.copy(up0).multiplyScalar(cr).addScaledVector(right0, sr)
      uniforms.uCamRight.value.copy(right0).multiplyScalar(cr).addScaledVector(up0, -sr)
      uniforms.uCamPos.value.set(camX, camY, camZ)
    }

    const handleResize = () => {
      renderer.setSize(window.innerWidth, window.innerHeight)
      const [w, h] = targetSize()
      target.setSize(w, h)
      uniforms.uResolution.value.set(w, h)
    }
    window.addEventListener('resize', handleResize)

    // Quality ratchet: if the GPU can't keep up, permanently drop the raymarch resolution
    // a notch (never back up, so it can't oscillate between sizes)
    let slowSum = 0
    let slowCount = 0

    const startTime = performance.now()
    let lastTime = startTime
    let animationId = 0

    const animate = () => {
      animationId = requestAnimationFrame(animate)
      const now = performance.now()
      const rawDt = (now - lastTime) / 1000
      lastTime = now
      // Clamp so a backgrounded tab doesn't take one giant step on return
      const dt = Math.min(rawDt, 1 / 20)
      const time = (now - startTime) / 1000

      if (rawDt < 0.1) {
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

      renderer.setRenderTarget(target)
      renderer.render(terrainScene, camera)
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
      target.dispose()
      renderer.dispose()
      container.removeChild(renderer.domElement)
    }
  }, [])

  return <div ref={containerRef} className="w-full h-full" />
}
