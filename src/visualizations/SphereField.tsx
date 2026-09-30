import { useEffect, useRef } from 'react'

interface Sphere {
  x: number
  y: number
  vx: number
  vy: number
  radius: number
  hue: number
  r: number
  g: number
  b: number
}

const DOT_SPACING = 16
const MIN_DOT_RADIUS = 0.8
const MAX_DOT_RADIUS = DOT_SPACING * 0.48
const SPHERE_COUNT = 7
const SPHERE_SPEED = 0.6
// Sphere radius as a fraction of the smaller screen dimension
const MIN_SPHERE_SCALE = 0.12
const MAX_SPHERE_SCALE = 0.3
const HUE_DRIFT = 0.05
const BASE_COLOR = { r: 40, g: 44, b: 60 }

// HSL (s = 0.8, l = 0.6) to RGB 0-255
const hueToRgb = (hue: number) => {
  const s = 0.8
  const l = 0.6
  const c = (1 - Math.abs(2 * l - 1)) * s
  const hp = (hue % 360) / 60
  const x = c * (1 - Math.abs((hp % 2) - 1))
  let r = 0
  let g = 0
  let b = 0
  if (hp < 1) [r, g, b] = [c, x, 0]
  else if (hp < 2) [r, g, b] = [x, c, 0]
  else if (hp < 3) [r, g, b] = [0, c, x]
  else if (hp < 4) [r, g, b] = [0, x, c]
  else if (hp < 5) [r, g, b] = [x, 0, c]
  else [r, g, b] = [c, 0, x]
  const m = l - c / 2
  return { r: (r + m) * 255, g: (g + m) * 255, b: (b + m) * 255 }
}

export default function SphereField() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number | undefined>(undefined)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let spheres: Sphere[] = []

    const initSpheres = () => {
      const minDim = Math.min(canvas.width, canvas.height)
      spheres = Array.from({ length: SPHERE_COUNT }, () => {
        const angle = Math.random() * Math.PI * 2
        const speed = SPHERE_SPEED * (0.5 + Math.random())
        const hue = Math.random() * 360
        return {
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height,
          vx: Math.cos(angle) * speed,
          vy: Math.sin(angle) * speed,
          radius: minDim * (MIN_SPHERE_SCALE + Math.random() * (MAX_SPHERE_SCALE - MIN_SPHERE_SCALE)),
          hue,
          ...hueToRgb(hue),
        }
      })
    }

    const resizeCanvas = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
      initSpheres()
    }

    resizeCanvas()
    window.addEventListener('resize', resizeCanvas)

    const updateSpheres = () => {
      spheres.forEach(sphere => {
        sphere.x += sphere.vx
        sphere.y += sphere.vy

        // Bounce when the center reaches an edge, so spheres can half-leave the screen
        if (sphere.x < 0 || sphere.x > canvas.width) {
          sphere.vx *= -1
          sphere.x = Math.max(0, Math.min(canvas.width, sphere.x))
        }
        if (sphere.y < 0 || sphere.y > canvas.height) {
          sphere.vy *= -1
          sphere.y = Math.max(0, Math.min(canvas.height, sphere.y))
        }

        sphere.hue = (sphere.hue + HUE_DRIFT) % 360
        Object.assign(sphere, hueToRgb(sphere.hue))
      })
    }

    const draw = () => {
      const width = canvas.width
      const height = canvas.height

      ctx.fillStyle = '#000000'
      ctx.fillRect(0, 0, width, height)

      // Center the grid so the margins are even on all sides
      const cols = Math.floor(width / DOT_SPACING)
      const rows = Math.floor(height / DOT_SPACING)
      const offsetX = (width - (cols - 1) * DOT_SPACING) / 2
      const offsetY = (height - (rows - 1) * DOT_SPACING) / 2

      // Dots no sphere touches are identical, so they go into one batched path
      const flatDots = new Path2D()

      for (let row = 0; row < rows; row++) {
        const y = offsetY + row * DOT_SPACING
        for (let col = 0; col < cols; col++) {
          const x = offsetX + col * DOT_SPACING

          // Combined height toward the viewer: 1 - Π(1 - h) across overlapping spheres
          let flat = 1
          let weight = 0
          let r = 0
          let g = 0
          let b = 0
          for (const sphere of spheres) {
            const dx = x - sphere.x
            const dy = y - sphere.y
            const distSq = dx * dx + dy * dy
            const radiusSq = sphere.radius * sphere.radius
            if (distSq >= radiusSq) continue
            // Hemisphere profile: 1 at the center (closest to the screen), 0 at the rim
            const h = Math.sqrt(1 - distSq / radiusSq)
            flat *= 1 - h
            weight += h
            r += sphere.r * h
            g += sphere.g * h
            b += sphere.b * h
          }
          if (weight === 0) {
            flatDots.moveTo(x + MIN_DOT_RADIUS, y)
            flatDots.arc(x, y, MIN_DOT_RADIUS, 0, Math.PI * 2)
            continue
          }
          const lift = 1 - flat

          const dotRadius = MIN_DOT_RADIUS + lift * (MAX_DOT_RADIUS - MIN_DOT_RADIUS)
          // Blend from the dim base toward the spheres' mixed hue, brightening as it rises
          const t = Math.min(1, lift * 1.2)
          const bright = 0.6 + 0.4 * lift
          const cr = BASE_COLOR.r + ((r / weight) * bright - BASE_COLOR.r) * t
          const cg = BASE_COLOR.g + ((g / weight) * bright - BASE_COLOR.g) * t
          const cb = BASE_COLOR.b + ((b / weight) * bright - BASE_COLOR.b) * t

          ctx.fillStyle = `rgb(${cr | 0}, ${cg | 0}, ${cb | 0})`
          ctx.beginPath()
          ctx.arc(x, y, dotRadius, 0, Math.PI * 2)
          ctx.fill()
        }
      }

      ctx.fillStyle = `rgb(${BASE_COLOR.r}, ${BASE_COLOR.g}, ${BASE_COLOR.b})`
      ctx.fill(flatDots)
    }

    const animate = () => {
      updateSpheres()
      draw()
      animationRef.current = requestAnimationFrame(animate)
    }

    animate()

    return () => {
      window.removeEventListener('resize', resizeCanvas)
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [])

  return <canvas ref={canvasRef} style={{ display: 'block' }} />
}
