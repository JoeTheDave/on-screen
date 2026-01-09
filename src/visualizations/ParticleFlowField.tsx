import { useEffect, useState, useRef } from 'react'

// Configuration constants
const PARTICLE_COUNT = 3000
const FLOW_FIELD_RESOLUTION = 20
const FLOW_MAGNITUDE = 2
const PARTICLE_SPEED = 1.5
const TRAIL_FADE = 0.08
const LINE_WIDTH = 1.5
const PARTICLE_ALPHA = 0.8
const FADE_IN_FRAMES = 30
const VELOCITY_DAMPING = 0.95

// Type definitions
interface Particle {
  x: number
  y: number
  vx: number
  vy: number
  hue: number
  life: number
  maxLife: number
}

export default function ParticleFlowField() {
  const [width, setWidth] = useState(0)
  const [height, setHeight] = useState(0)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number | undefined>(undefined)
  const particlesRef = useRef<Particle[]>([])
  const flowFieldRef = useRef<{ angle: number; magnitude: number }[][]>([])
  const timeOffsetRef = useRef(0)

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

    const cols = Math.floor(width / FLOW_FIELD_RESOLUTION)
    const rows = Math.floor(height / FLOW_FIELD_RESOLUTION)

    /**
     * Create a particle with random position and color
     */
    const createParticle = (): Particle => {
      const maxLife = 180 + Math.random() * 180 // 3-6 seconds at 60fps
      return {
        x: Math.random() * width,
        y: Math.random() * height,
        vx: 0,
        vy: 0,
        hue: Math.random() * 360,
        life: 0,
        maxLife,
      }
    }

    /**
     * Generate flow field using layered sine/cosine waves
     */
    const generateFlowField = (time: number) => {
      const field: { angle: number; magnitude: number }[][] = []

      for (let i = 0; i < cols; i++) {
        field[i] = []
        for (let j = 0; j < rows; j++) {
          const x = i * FLOW_FIELD_RESOLUTION
          const y = j * FLOW_FIELD_RESOLUTION

          // Create complex flow pattern using multiple sine waves
          const angle1 = Math.sin(x * 0.01 + time * 0.5) * Math.cos(y * 0.01)
          const angle2 = Math.cos(x * 0.008 - time * 0.3) * Math.sin(y * 0.012 + time * 0.2)
          const angle3 = Math.sin((x + y) * 0.006 + time * 0.4)

          const angle = (angle1 + angle2 * 0.5 + angle3 * 0.3) * Math.PI * 2

          field[i][j] = {
            angle,
            magnitude: FLOW_MAGNITUDE,
          }
        }
      }

      return field
    }

    /**
     * Initialize animation
     */
    const initializeAnimation = () => {
      timeOffsetRef.current = Math.random() * 1000

      // Clear canvas
      ctx.fillStyle = 'black'
      ctx.fillRect(0, 0, width, height)

      // Initialize particles
      particlesRef.current = Array.from({ length: PARTICLE_COUNT }, createParticle)

      // Generate initial flow field
      flowFieldRef.current = generateFlowField(timeOffsetRef.current)
    }

    initializeAnimation()

    /**
     * Main animation loop
     */
    const animate = (frameCount = 0) => {
      // Fade effect for trails
      ctx.globalAlpha = TRAIL_FADE
      ctx.fillStyle = 'black'
      ctx.fillRect(0, 0, width, height)
      ctx.globalAlpha = 1

      // Update flow field with evolving time
      const time = timeOffsetRef.current + frameCount * 0.01
      flowFieldRef.current = generateFlowField(time)

      // Update and draw particles
      particlesRef.current.forEach((particle, index) => {
        const prevX = particle.x
        const prevY = particle.y

        // Apply flow field influence
        const col = Math.floor(particle.x / FLOW_FIELD_RESOLUTION)
        const row = Math.floor(particle.y / FLOW_FIELD_RESOLUTION)

        if (col >= 0 && col < cols && row >= 0 && row < rows) {
          const flow = flowFieldRef.current[col][row]
          particle.vx += Math.cos(flow.angle) * flow.magnitude * 0.1
          particle.vy += Math.sin(flow.angle) * flow.magnitude * 0.1
        }

        // Apply velocity with damping
        particle.vx *= VELOCITY_DAMPING
        particle.vy *= VELOCITY_DAMPING

        // Update position
        particle.x += particle.vx * PARTICLE_SPEED
        particle.y += particle.vy * PARTICLE_SPEED

        // Check if particle will wrap around edges
        const didWrap = particle.x < 0 || particle.x > width || particle.y < 0 || particle.y > height

        // Wrap around edges
        if (particle.x < 0) particle.x = width
        if (particle.x > width) particle.x = 0
        if (particle.y < 0) particle.y = height
        if (particle.y > height) particle.y = 0

        // Update life
        particle.life++

        // Calculate alpha with fade in/out
        let alpha = PARTICLE_ALPHA
        if (particle.life < FADE_IN_FRAMES) {
          alpha = (particle.life / FADE_IN_FRAMES) * PARTICLE_ALPHA
        } else if (particle.life > particle.maxLife - FADE_IN_FRAMES) {
          alpha = ((particle.maxLife - particle.life) / FADE_IN_FRAMES) * PARTICLE_ALPHA
        }

        // Calculate color based on velocity
        const speed = Math.sqrt(particle.vx * particle.vx + particle.vy * particle.vy)
        const saturation = 50 + Math.min(speed * 10, 50)
        const lightness = 40 + Math.min(speed * 5, 30)

        // Draw particle trail (skip if wrapped to avoid cross-screen lines)
        if (!didWrap) {
          ctx.strokeStyle = `hsla(${particle.hue}, ${saturation}%, ${lightness}%, ${alpha})`
          ctx.lineWidth = LINE_WIDTH
          ctx.lineCap = 'round'
          ctx.beginPath()
          ctx.moveTo(prevX, prevY)
          ctx.lineTo(particle.x, particle.y)
          ctx.stroke()
        }

        // Reset particle when it reaches max life
        if (particle.life >= particle.maxLife) {
          particlesRef.current[index] = createParticle()
        }
      })

      animationRef.current = requestAnimationFrame(() => animate(frameCount + 1))
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
