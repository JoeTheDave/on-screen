import { useEffect, useState, useRef } from 'react'

// Configuration constants
const BOID_COUNT = 300
const BOID_SIZE = 3
const MAX_SPEED = 2.5
const MAX_FORCE = 0.05
const PERCEPTION_RADIUS = 50
const SEPARATION_RADIUS = 25
const SEPARATION_WEIGHT = 1.5
const ALIGNMENT_WEIGHT = 1.0
const COHESION_WEIGHT = 1.0

// Type definitions
interface Boid {
  x: number
  y: number
  vx: number
  vy: number
  hue: number
}

export default function FlockingBoids() {
  const [width, setWidth] = useState(0)
  const [height, setHeight] = useState(0)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number | undefined>(undefined)
  const boidsRef = useRef<Boid[]>([])

  // Handle window resize
  useEffect(() => {
    const updateSize = () => {
      setWidth(window.innerWidth)
      setHeight(window.innerHeight)
    }

    updateSize()
    window.addEventListener('resize', updateSize)
    return () => window.removeEventListener('resize', updateSize)
  }, [])

  // Main animation effect
  useEffect(() => {
    if (!canvasRef.current || width === 0 || height === 0) return

    const ctx = canvasRef.current.getContext('2d')
    if (!ctx) return

    /**
     * Create a boid with random position and velocity
     */
    const createBoid = (): Boid => {
      const angle = Math.random() * Math.PI * 2
      const speed = Math.random() * MAX_SPEED
      return {
        x: Math.random() * width,
        y: Math.random() * height,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        hue: Math.random() * 360,
      }
    }

    /**
     * Calculate distance between two points
     */
    const distance = (x1: number, y1: number, x2: number, y2: number) => {
      const dx = x2 - x1
      const dy = y2 - y1
      return Math.sqrt(dx * dx + dy * dy)
    }

    /**
     * Limit a vector to a maximum magnitude
     */
    const limit = (vx: number, vy: number, max: number) => {
      const mag = Math.sqrt(vx * vx + vy * vy)
      if (mag > max) {
        return { vx: (vx / mag) * max, vy: (vy / mag) * max }
      }
      return { vx, vy }
    }

    /**
     * Apply flocking rules to a boid
     */
    const applyFlockingRules = (boid: Boid, boids: Boid[]) => {
      let separationX = 0
      let separationY = 0
      let alignmentX = 0
      let alignmentY = 0
      let cohesionX = 0
      let cohesionY = 0
      let nearbyCount = 0
      let tooCloseCount = 0

      // Check all other boids
      boids.forEach((other) => {
        if (other === boid) return

        const dist = distance(boid.x, boid.y, other.x, other.y)

        // Separation: steer away from nearby boids
        if (dist < SEPARATION_RADIUS && dist > 0) {
          const diffX = boid.x - other.x
          const diffY = boid.y - other.y
          separationX += diffX / dist
          separationY += diffY / dist
          tooCloseCount++
        }

        // Alignment and Cohesion: only for boids within perception radius
        if (dist < PERCEPTION_RADIUS && dist > 0) {
          alignmentX += other.vx
          alignmentY += other.vy
          cohesionX += other.x
          cohesionY += other.y
          nearbyCount++
        }
      })

      let forceX = 0
      let forceY = 0

      // Average separation
      if (tooCloseCount > 0) {
        separationX /= tooCloseCount
        separationY /= tooCloseCount
        const sepMag = Math.sqrt(separationX * separationX + separationY * separationY)
        if (sepMag > 0) {
          separationX = (separationX / sepMag) * MAX_SPEED - boid.vx
          separationY = (separationY / sepMag) * MAX_SPEED - boid.vy
          const limited = limit(separationX, separationY, MAX_FORCE)
          forceX += limited.vx * SEPARATION_WEIGHT
          forceY += limited.vy * SEPARATION_WEIGHT
        }
      }

      // Average alignment and cohesion
      if (nearbyCount > 0) {
        // Alignment: steer towards average heading
        alignmentX /= nearbyCount
        alignmentY /= nearbyCount
        const alignMag = Math.sqrt(alignmentX * alignmentX + alignmentY * alignmentY)
        if (alignMag > 0) {
          alignmentX = (alignmentX / alignMag) * MAX_SPEED - boid.vx
          alignmentY = (alignmentY / alignMag) * MAX_SPEED - boid.vy
          const limited = limit(alignmentX, alignmentY, MAX_FORCE)
          forceX += limited.vx * ALIGNMENT_WEIGHT
          forceY += limited.vy * ALIGNMENT_WEIGHT
        }

        // Cohesion: steer towards average position
        cohesionX /= nearbyCount
        cohesionY /= nearbyCount
        const desiredX = cohesionX - boid.x
        const desiredY = cohesionY - boid.y
        const cohMag = Math.sqrt(desiredX * desiredX + desiredY * desiredY)
        if (cohMag > 0) {
          const steerX = (desiredX / cohMag) * MAX_SPEED - boid.vx
          const steerY = (desiredY / cohMag) * MAX_SPEED - boid.vy
          const limited = limit(steerX, steerY, MAX_FORCE)
          forceX += limited.vx * COHESION_WEIGHT
          forceY += limited.vy * COHESION_WEIGHT
        }
      }

      return { forceX, forceY }
    }

    /**
     * Initialize animation
     */
    const initializeAnimation = () => {
      ctx.fillStyle = 'black'
      ctx.fillRect(0, 0, width, height)
      boidsRef.current = Array.from({ length: BOID_COUNT }, createBoid)
    }

    initializeAnimation()

    /**
     * Main animation loop
     */
    const animate = () => {
      // Clear canvas
      ctx.fillStyle = 'black'
      ctx.fillRect(0, 0, width, height)

      // Update and draw boids
      boidsRef.current.forEach((boid) => {
        // Apply flocking rules
        const { forceX, forceY } = applyFlockingRules(boid, boidsRef.current)

        // Update velocity
        boid.vx += forceX
        boid.vy += forceY

        // Limit speed
        const limited = limit(boid.vx, boid.vy, MAX_SPEED)
        boid.vx = limited.vx
        boid.vy = limited.vy

        // Update position
        boid.x += boid.vx
        boid.y += boid.vy

        // Wrap around edges
        if (boid.x < 0) boid.x = width
        if (boid.x > width) boid.x = 0
        if (boid.y < 0) boid.y = height
        if (boid.y > height) boid.y = 0

        // Draw boid as a small triangle pointing in direction of velocity
        const angle = Math.atan2(boid.vy, boid.vx)
        ctx.save()
        ctx.translate(boid.x, boid.y)
        ctx.rotate(angle)
        ctx.fillStyle = `hsl(${boid.hue}, 80%, 65%)`
        ctx.beginPath()
        ctx.moveTo(BOID_SIZE * 2, 0)
        ctx.lineTo(-BOID_SIZE, BOID_SIZE)
        ctx.lineTo(-BOID_SIZE, -BOID_SIZE)
        ctx.closePath()
        ctx.fill()
        ctx.restore()
      })

      animationRef.current = requestAnimationFrame(animate)
    }

    animate()

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [width, height])

  return (
    <div className="w-full h-screen">
      <canvas ref={canvasRef} width={width} height={height} className="w-full h-full" />
    </div>
  )
}
