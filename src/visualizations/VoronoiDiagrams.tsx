import { useEffect, useRef } from 'react'

interface Point {
  x: number
  y: number
  vx: number
  vy: number
  hue: number
  prevX: number
  prevY: number
  speed: number
  cellSize: number
}

const SEED_COUNT = 250
const SEED_SPEED = 1.5
const REPULSION_FORCE = 0.5
const MIN_DISTANCE = 50
const WALL_REPULSION_DISTANCE = 30
const WALL_REPULSION_FORCE = 0.3

// Corner colors for 2D gradient mapping
// Top-left (slow, small), Top-right (fast, small), Bottom-left (slow, large), Bottom-right (fast, large)
const CORNER_COLORS = [
  { r: 0.2, g: 0.3, b: 0.8 },  // Top-left: Blue (slow, small)
  { r: 0.8, g: 0.2, b: 0.3 },  // Top-right: Red (fast, small)
  { r: 0.3, g: 0.8, b: 0.4 },  // Bottom-left: Green (slow, large)
  { r: 0.9, g: 0.7, b: 0.2 },  // Bottom-right: Yellow (fast, large)
]

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
        const x = Math.random() * canvas.width
        const y = Math.random() * canvas.height
        return {
          x,
          y,
          vx: Math.cos(angle) * SEED_SPEED,
          vy: Math.sin(angle) * SEED_SPEED,
          hue: Math.random() * 360,
          prevX: x,
          prevY: y,
          speed: 0,
          cellSize: 0,
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

        // Apply wall repulsion
        const distToLeft = seed.x
        const distToRight = canvas.width - seed.x
        const distToTop = seed.y
        const distToBottom = canvas.height - seed.y

        // Repel from left wall
        if (distToLeft < WALL_REPULSION_DISTANCE) {
          const force = (WALL_REPULSION_DISTANCE - distToLeft) / WALL_REPULSION_DISTANCE * WALL_REPULSION_FORCE
          seed.vx += force
        }
        // Repel from right wall
        if (distToRight < WALL_REPULSION_DISTANCE) {
          const force = (WALL_REPULSION_DISTANCE - distToRight) / WALL_REPULSION_DISTANCE * WALL_REPULSION_FORCE
          seed.vx -= force
        }
        // Repel from top wall
        if (distToTop < WALL_REPULSION_DISTANCE) {
          const force = (WALL_REPULSION_DISTANCE - distToTop) / WALL_REPULSION_DISTANCE * WALL_REPULSION_FORCE
          seed.vy += force
        }
        // Repel from bottom wall
        if (distToBottom < WALL_REPULSION_DISTANCE) {
          const force = (WALL_REPULSION_DISTANCE - distToBottom) / WALL_REPULSION_DISTANCE * WALL_REPULSION_FORCE
          seed.vy -= force
        }

        // Damping to prevent excessive speeds
        const speed = Math.sqrt(seed.vx * seed.vx + seed.vy * seed.vy)
        if (speed > SEED_SPEED * 2) {
          seed.vx = (seed.vx / speed) * SEED_SPEED * 2
          seed.vy = (seed.vy / speed) * SEED_SPEED * 2
        }

        seed.x += seed.vx
        seed.y += seed.vy

        // Calculate speed (distance traveled)
        const dx = seed.x - seed.prevX
        const dy = seed.y - seed.prevY
        seed.speed = Math.sqrt(dx * dx + dy * dy)
        seed.prevX = seed.x
        seed.prevY = seed.y

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

    // Calculate polygon area
    const calculatePolygonArea = (polygon: [number, number][]): number => {
      let area = 0
      for (let i = 0; i < polygon.length; i++) {
        const [x1, y1] = polygon[i]
        const [x2, y2] = polygon[(i + 1) % polygon.length]
        area += x1 * y2 - x2 * y1
      }
      return Math.abs(area) / 2
    }

    // Bilinear interpolation for 2D color mapping
    const interpolateColor2D = (x: number, y: number) => {
      // x = speed (0-1), y = size (0-1)
      // Corners: [0] = top-left, [1] = top-right, [2] = bottom-left, [3] = bottom-right
      const c00 = CORNER_COLORS[0] // top-left (slow, small)
      const c10 = CORNER_COLORS[1] // top-right (fast, small)
      const c01 = CORNER_COLORS[2] // bottom-left (slow, large)
      const c11 = CORNER_COLORS[3] // bottom-right (fast, large)

      // Interpolate along top edge
      const topR = c00.r * (1 - x) + c10.r * x
      const topG = c00.g * (1 - x) + c10.g * x
      const topB = c00.b * (1 - x) + c10.b * x

      // Interpolate along bottom edge
      const bottomR = c01.r * (1 - x) + c11.r * x
      const bottomG = c01.g * (1 - x) + c11.g * x
      const bottomB = c01.b * (1 - x) + c11.b * x

      // Interpolate between top and bottom
      const r = Math.round((topR * (1 - y) + bottomR * y) * 255)
      const g = Math.round((topG * (1 - y) + bottomG * y) * 255)
      const b = Math.round((topB * (1 - y) + bottomB * y) * 255)

      return `rgb(${r}, ${g}, ${b})`
    }

    const drawVoronoi = () => {
      const width = canvas.width
      const height = canvas.height
      const seeds = seedsRef.current

      // Clear canvas
      ctx.fillStyle = '#000000'
      ctx.fillRect(0, 0, width, height)

      // First pass: calculate cell sizes for all seeds
      const cellData: { polygon: [number, number][]; area: number }[] = []
      seeds.forEach((seed, seedIdx) => {
        const polygon = computeVoronoiCell(seedIdx, seeds, width, height)
        const area = polygon.length >= 3 ? calculatePolygonArea(polygon) : 0
        cellData.push({ polygon, area })
        seed.cellSize = area
      })

      // Find min/max for normalization
      const speeds = seeds.map(s => s.speed)
      const sizes = seeds.map(s => s.cellSize)
      const minSpeed = Math.min(...speeds)
      const maxSpeed = Math.max(...speeds)
      const minSize = Math.min(...sizes)
      const maxSize = Math.max(...sizes)

      // Avoid division by zero
      const speedRange = maxSpeed - minSpeed || 1
      const sizeRange = maxSize - minSize || 1

      // Draw each Voronoi cell with 2D color mapping
      seeds.forEach((seed, seedIdx) => {
        const { polygon } = cellData[seedIdx]
        
        if (polygon.length < 3) return

        // Normalize speed and size to 0-1
        const normalizedSpeed = (seed.speed - minSpeed) / speedRange
        const normalizedSize = (seed.cellSize - minSize) / sizeRange

        // Get color based on 2D position
        const baseColor = interpolateColor2D(normalizedSpeed, normalizedSize)

        // Create radial gradient from seed point using the mapped color
        const maxDist = Math.sqrt(width * width + height * height)
        const gradient = ctx.createRadialGradient(seed.x, seed.y, 0, seed.x, seed.y, maxDist * 0.5)
        
        // Parse RGB from base color
        const match = baseColor.match(/\d+/g)
        if (match) {
          const [r, g, b] = match.map(Number)
          gradient.addColorStop(0, `rgb(${Math.min(255, r + 100)}, ${Math.min(255, g + 100)}, ${Math.min(255, b + 100)})`)
          gradient.addColorStop(0.2, `rgb(${Math.min(255, r + 60)}, ${Math.min(255, g + 60)}, ${Math.min(255, b + 60)})`)
          gradient.addColorStop(0.5, baseColor)
          gradient.addColorStop(0.8, `rgb(${Math.round(r * 0.6)}, ${Math.round(g * 0.6)}, ${Math.round(b * 0.6)})`)
          gradient.addColorStop(1, `rgb(${Math.round(r * 0.3)}, ${Math.round(g * 0.3)}, ${Math.round(b * 0.3)})`)
        }

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
