import { useEffect, useRef } from 'react'

interface Point {
  x: number
  y: number
  vx: number
  vy: number
  hue: number
}

const SEED_COUNT = 250
const SEED_SPEED = 1.5
const REPULSION_FORCE = 0.5
const MIN_DISTANCE = 50

export default function VoronoiDiagrams() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const seedsRef = useRef<Point[]>([])
  const animationRef = useRef<number | undefined>(undefined)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const resizeCanvas = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }

    resizeCanvas()
    window.addEventListener('resize', resizeCanvas)

    // Initialize seeds with random positions, velocities, and colors
    const initSeeds = () => {
      seedsRef.current = Array.from({ length: SEED_COUNT }, () => {
        const angle = Math.random() * Math.PI * 2
        return {
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height,
          vx: Math.cos(angle) * SEED_SPEED,
          vy: Math.sin(angle) * SEED_SPEED,
          hue: Math.random() * 360,
        }
      })
    }

    initSeeds()

    const updateSeeds = () => {
      const seeds = seedsRef.current

      // Apply repulsion between seeds
      seeds.forEach((seed, i) => {
        let repelX = 0
        let repelY = 0

        seeds.forEach((other, j) => {
          if (i === j) return

          const dx = seed.x - other.x
          const dy = seed.y - other.y
          const distSq = dx * dx + dy * dy
          const dist = Math.sqrt(distSq)

          // Apply repulsion if seeds are too close
          if (dist < MIN_DISTANCE && dist > 0) {
            const force = (MIN_DISTANCE - dist) / MIN_DISTANCE * REPULSION_FORCE
            repelX += (dx / dist) * force
            repelY += (dy / dist) * force
          }
        })

        seed.vx += repelX
        seed.vy += repelY

        // Damping to prevent excessive speeds
        const speed = Math.sqrt(seed.vx * seed.vx + seed.vy * seed.vy)
        if (speed > SEED_SPEED * 2) {
          seed.vx = (seed.vx / speed) * SEED_SPEED * 2
          seed.vy = (seed.vy / speed) * SEED_SPEED * 2
        }

        seed.x += seed.vx
        seed.y += seed.vy

        // Bounce off edges
        if (seed.x < 0 || seed.x > canvas.width) {
          seed.vx *= -1
          seed.x = Math.max(0, Math.min(canvas.width, seed.x))
        }
        if (seed.y < 0 || seed.y > canvas.height) {
          seed.vy *= -1
          seed.y = Math.max(0, Math.min(canvas.height, seed.y))
        }

        // Slowly shift hue
        seed.hue = (seed.hue + 0.2) % 360
      })
    }

    // Compute Voronoi cell for a seed by clipping with perpendicular bisectors
    const computeVoronoiCell = (seedIdx: number, seeds: Point[], width: number, height: number): [number, number][] => {
      const seed = seeds[seedIdx]
      const margin = 100
      
      // Start with bounding box
      let polygon: [number, number][] = [
        [-margin, -margin],
        [width + margin, -margin],
        [width + margin, height + margin],
        [-margin, height + margin],
      ]

      // Clip polygon by perpendicular bisector with each other seed
      for (let i = 0; i < seeds.length; i++) {
        if (i === seedIdx) continue
        
        const other = seeds[i]
        const midX = (seed.x + other.x) / 2
        const midY = (seed.y + other.y) / 2
        
        // Vector from other to seed (perpendicular bisector normal)
        const dx = seed.x - other.x
        const dy = seed.y - other.y
        
        // Clip polygon by this half-plane
        polygon = clipPolygonByLine(polygon, midX, midY, dx, dy)
        
        if (polygon.length === 0) break
      }

      return polygon
    }

    // Clip polygon by a line (keep points on the positive side of the line)
    const clipPolygonByLine = (
      polygon: [number, number][],
      lineX: number,
      lineY: number,
      normalX: number,
      normalY: number
    ): [number, number][] => {
      if (polygon.length === 0) return []
      
      const result: [number, number][] = []
      
      for (let i = 0; i < polygon.length; i++) {
        const [x1, y1] = polygon[i]
        const [x2, y2] = polygon[(i + 1) % polygon.length]
        
        const d1 = (x1 - lineX) * normalX + (y1 - lineY) * normalY
        const d2 = (x2 - lineX) * normalX + (y2 - lineY) * normalY
        
        if (d1 >= 0) {
          result.push([x1, y1])
          
          if (d2 < 0) {
            // Line crosses from inside to outside, add intersection
            const t = d1 / (d1 - d2)
            result.push([
              x1 + t * (x2 - x1),
              y1 + t * (y2 - y1),
            ])
          }
        } else if (d2 >= 0) {
          // Line crosses from outside to inside, add intersection
          const t = d1 / (d1 - d2)
          result.push([
            x1 + t * (x2 - x1),
            y1 + t * (y2 - y1),
          ])
        }
      }
      
      return result
    }

    const drawVoronoi = () => {
      const width = canvas.width
      const height = canvas.height
      const seeds = seedsRef.current

      // Clear canvas
      ctx.fillStyle = '#000000'
      ctx.fillRect(0, 0, width, height)

      // Draw each Voronoi cell
      seeds.forEach((seed, seedIdx) => {
        const polygon = computeVoronoiCell(seedIdx, seeds, width, height)
        
        if (polygon.length < 3) return

        // Create radial gradient from seed point
        const maxDist = Math.sqrt(width * width + height * height)
        const gradient = ctx.createRadialGradient(seed.x, seed.y, 0, seed.x, seed.y, maxDist * 0.5)
        gradient.addColorStop(0, `hsl(${seed.hue}, 100%, 85%)`)
        gradient.addColorStop(0.2, `hsl(${seed.hue}, 95%, 65%)`)
        gradient.addColorStop(0.5, `hsl(${seed.hue}, 80%, 45%)`)
        gradient.addColorStop(0.8, `hsl(${seed.hue}, 65%, 25%)`)
        gradient.addColorStop(1, `hsl(${seed.hue}, 50%, 10%)`)

        // Draw polygon
        ctx.beginPath()
        ctx.moveTo(polygon[0][0], polygon[0][1])
        for (let i = 1; i < polygon.length; i++) {
          ctx.lineTo(polygon[i][0], polygon[i][1])
        }
        ctx.closePath()
        
        ctx.fillStyle = gradient
        ctx.fill()
        
        // Draw border
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)'
        ctx.lineWidth = 1
        ctx.stroke()
      })

      // Draw seed points
      ctx.fillStyle = '#ffffff'
      ctx.shadowColor = '#000000'
      ctx.shadowBlur = 4
      seeds.forEach(seed => {
        ctx.beginPath()
        ctx.arc(seed.x, seed.y, 4, 0, Math.PI * 2)
        ctx.fill()
      })
      ctx.shadowBlur = 0
    }

    const animate = () => {
      updateSeeds()
      drawVoronoi()
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
