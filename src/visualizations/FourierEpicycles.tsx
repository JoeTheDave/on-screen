import { useEffect, useState, useRef } from 'react'

// Configuration constants
const ENABLE_CONTAINER_ROTATION = true
const ANIMATION_DURATION_SECONDS = 60
const PAUSE_DURATION_SECONDS = 10
const TARGET_FPS = 60
const MIN_ROTATIONS_PER_MINUTE = 6
const MAX_ROTATIONS_PER_MINUTE = 80
const LINE_WIDTH = 2

// Type definitions
interface Segments {
  length1: number
  length2: number
  speed1: number
  speed2: number
}

interface Point {
  x: number
  y: number
}

interface RotationAngles {
  angle1: number
  angle2: number
}

export default function FourierEpicycles() {
  // State and refs
  const [size, setSize] = useState(0)
  const canvas1Ref = useRef<HTMLCanvasElement>(null)
  const canvas2Ref = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number | undefined>(undefined)
  const rotationRef = useRef<RotationAngles>({ angle1: -Math.PI / 2, angle2: -Math.PI / 2 })
  const segmentsRef = useRef<Segments | null>(null)
  const prevTipRef = useRef<Point | null>(null)
  const traceColorsRef = useRef<{ slow: [number, number, number]; fast: [number, number, number] } | null>(null)
  const frameCountRef = useRef(0)

  // Handle window resize to keep canvas square based on smallest dimension
  useEffect(() => {
    const updateSize = () => {
      const minDimension = Math.min(window.innerWidth, window.innerHeight)
      setSize(minDimension)
    }

    updateSize()
    window.addEventListener('resize', updateSize)
    return () => window.removeEventListener('resize', updateSize)
  }, [])

  // Main animation effect
  useEffect(() => {
    if (!canvas1Ref.current || !canvas2Ref.current || size === 0) return

    const ctx1 = canvas1Ref.current.getContext('2d')
    const ctx2 = canvas2Ref.current.getContext('2d')
    if (!ctx1 || !ctx2) return

    const centerX = size / 2
    const centerY = size / 2

    /**
     * Calculate GCD (Greatest Common Divisor) to ensure coprime rotation counts
     */
    const gcd = (a: number, b: number): number => {
      return b === 0 ? a : gcd(b, a % b)
    }

    /**
     * Generate random HSL color with high saturation and medium-high lightness
     * Returns [hue, saturation, lightness] tuple
     */
    const generateRandomColor = (): [number, number, number] => {
      const hue = Math.floor(Math.random() * 360)
      const saturation = 70 + Math.floor(Math.random() * 30) // 70-100%
      const lightness = 50 + Math.floor(Math.random() * 20) // 50-70%
      return [hue, saturation, lightness]
    }

    /**
     * Interpolate between two HSL colors based on ratio (0-1)
     */
    const interpolateColor = (color1: [number, number, number], color2: [number, number, number], ratio: number): string => {
      const [h1, s1, l1] = color1
      const [h2, s2, l2] = color2
      
      // Interpolate hue (accounting for circular nature of hue)
      let hue: number
      const diff = h2 - h1
      if (Math.abs(diff) <= 180) {
        hue = h1 + diff * ratio
      } else {
        // Take the shorter path around the color wheel
        const shortDiff = diff > 0 ? diff - 360 : diff + 360
        hue = (h1 + shortDiff * ratio + 360) % 360
      }
      
      const saturation = s1 + (s2 - s1) * ratio
      const lightness = l1 + (l2 - l1) * ratio
      
      return `hsl(${Math.round(hue)}, ${Math.round(saturation)}%, ${Math.round(lightness)}%)`
    }

    /**
     * Initialize or reset animation with new random parameters
     */
    const initializeAnimation = () => {
      // Reset state
      frameCountRef.current = 0
      rotationRef.current = { angle1: -Math.PI / 2, angle2: -Math.PI / 2 }
      prevTipRef.current = null

      // Clear both canvases
      ctx1.clearRect(0, 0, size, size)
      ctx2.clearRect(0, 0, size, size)

      // Generate random segment lengths
      const maxDistance = size / 2 - 10
      const split = 0.2 + Math.random() * 0.3 // First segment: 20-50%, Second: 50-80%

      // Generate coprime rotation counts for 60-second synchronization
      // This ensures the pattern only repeats when both segments complete their cycles
      const rotations1 =
        Math.floor(Math.random() * (MAX_ROTATIONS_PER_MINUTE - MIN_ROTATIONS_PER_MINUTE + 1)) +
        MIN_ROTATIONS_PER_MINUTE

      let rotations2 =
        Math.floor(Math.random() * (MAX_ROTATIONS_PER_MINUTE - MIN_ROTATIONS_PER_MINUTE + 1)) +
        MIN_ROTATIONS_PER_MINUTE

      // Ensure coprime rotation counts
      while (gcd(rotations1, rotations2) !== 1) {
        rotations2 =
          Math.floor(Math.random() * (MAX_ROTATIONS_PER_MINUTE - MIN_ROTATIONS_PER_MINUTE + 1)) +
          MIN_ROTATIONS_PER_MINUTE
      }

      // Convert to radians per frame: (rotations * 2π / 60 seconds) / 60 fps
      const speed1 =
        ((rotations1 * 2 * Math.PI) / ANIMATION_DURATION_SECONDS / TARGET_FPS) *
        (Math.random() < 0.5 ? 1 : -1)
      const speed2 =
        ((rotations2 * 2 * Math.PI) / ANIMATION_DURATION_SECONDS / TARGET_FPS) *
        (Math.random() < 0.5 ? 1 : -1)

      segmentsRef.current = {
        length1: maxDistance * split,
        length2: maxDistance * (1 - split),
        speed1,
        speed2,
      }

      // Generate two complementary colors for the gradient
      const color1 = generateRandomColor()
      const color2 = generateRandomColor()
      traceColorsRef.current = { slow: color1, fast: color2 }
    }

    // Initialize/reinitialize on every size change to start fresh
    initializeAnimation()

    /**
     * Main animation loop
     */
    const animate = () => {
      if (!segmentsRef.current || !traceColorsRef.current) return

      frameCountRef.current++

      const totalFrames = (ANIMATION_DURATION_SECONDS + PAUSE_DURATION_SECONDS) * TARGET_FPS
      const animationFrames = ANIMATION_DURATION_SECONDS * TARGET_FPS

      // Restart animation after complete cycle (animation + pause)
      if (frameCountRef.current >= totalFrames) {
        initializeAnimation()
      }

      // Draw animation during the active phase
      if (frameCountRef.current <= animationFrames) {
        ctx2.clearRect(0, 0, size, size)

        const { length1, length2, speed1, speed2 } = segmentsRef.current
        const { angle1, angle2 } = rotationRef.current

        // Calculate first segment endpoint (rotates around center)
        const x1 = centerX + length1 * Math.cos(angle1)
        const y1 = centerY + length1 * Math.sin(angle1)

        // Draw first segment
        ctx2.strokeStyle = 'white'
        ctx2.lineWidth = LINE_WIDTH
        ctx2.beginPath()
        ctx2.moveTo(centerX, centerY)
        ctx2.lineTo(x1, y1)
        ctx2.stroke()

        // Calculate second segment endpoint (rotates around end of first)
        const x2 = x1 + length2 * Math.cos(angle2)
        const y2 = y1 + length2 * Math.sin(angle2)

        // Draw second segment
        ctx2.beginPath()
        ctx2.moveTo(x1, y1)
        ctx2.lineTo(x2, y2)
        ctx2.stroke()

        // Draw trace line from previous position to current
        if (prevTipRef.current) {
          // Calculate distance using distance formula
          const dx = x2 - prevTipRef.current.x
          const dy = y2 - prevTipRef.current.y
          const distance = Math.sqrt(dx * dx + dy * dy)
          console.log(`Line segment length: ${distance.toFixed(2)}`)

          // Map distance to color gradient (0-60 range)
          const maxDistance = 60
          const ratio = Math.min(distance / maxDistance, 1)
          const color = interpolateColor(traceColorsRef.current.slow, traceColorsRef.current.fast, ratio)

          ctx1.strokeStyle = color
          ctx1.lineWidth = LINE_WIDTH
          ctx1.beginPath()
          ctx1.moveTo(prevTipRef.current.x, prevTipRef.current.y)
          ctx1.lineTo(x2, y2)
          ctx1.stroke()
        }

        // Update state for next frame
        prevTipRef.current = { x: x2, y: y2 }
        rotationRef.current.angle1 += speed1
        rotationRef.current.angle2 += speed2
      } else {
        // During pause phase, hide segments but keep trace visible
        ctx2.clearRect(0, 0, size, size)
      }

      animationRef.current = requestAnimationFrame(animate)
    }

    animate()

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [size])

  return (
    <div className="w-full h-screen flex items-center justify-center">
      <div
        className={`relative ${ENABLE_CONTAINER_ROTATION ? 'animate-spin' : ''}`}
        style={{
          width: size,
          height: size,
          ...(ENABLE_CONTAINER_ROTATION ? { animationDuration: '10s' } : {}),
        }}
      >
        {/* Canvas 2: Animated line segments (bottom layer) */}
        <canvas
          ref={canvas2Ref}
          width={size}
          height={size}
          className="absolute top-0 left-0"
        />
        {/* Canvas 1: Persistent trace pattern (top layer) */}
        <canvas
          ref={canvas1Ref}
          width={size}
          height={size}
          className="absolute top-0 left-0"
        />
      </div>
    </div>
  )
}
