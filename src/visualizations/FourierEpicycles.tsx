import { useEffect, useState, useRef } from 'react'

const ENABLE_ROTATION = true

export default function FourierEpicycles() {
  const [size, setSize] = useState(0)
  const canvas1Ref = useRef<HTMLCanvasElement>(null)
  const canvas2Ref = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number | undefined>(undefined)
  const rotationRef = useRef({ angle1: -Math.PI / 2, angle2: -Math.PI / 2 })
  const segmentsRef = useRef<{ length1: number; length2: number; speed1: number; speed2: number } | null>(null)
  const prevTipRef = useRef<{ x: number; y: number } | null>(null)
  const traceColorRef = useRef<string | null>(null)
  const frameCountRef = useRef(0)

  useEffect(() => {
    const updateSize = () => {
      const minDimension = Math.min(window.innerWidth, window.innerHeight)
      setSize(minDimension)
    }

    updateSize()
    window.addEventListener('resize', updateSize)
    return () => window.removeEventListener('resize', updateSize)
  }, [])

  useEffect(() => {
    if (!canvas1Ref.current || !canvas2Ref.current || size === 0) return

    const initializeAnimation = () => {
      // Reset frame counter
      frameCountRef.current = 0
      
      // Reset angles
      rotationRef.current = { angle1: -Math.PI / 2, angle2: -Math.PI / 2 }
      
      // Reset previous tip
      prevTipRef.current = null
      
      // Clear both canvases
      const ctx1 = canvas1Ref.current?.getContext('2d')
      const ctx2 = canvas2Ref.current?.getContext('2d')
      if (ctx1) ctx1.clearRect(0, 0, size, size)
      if (ctx2) ctx2.clearRect(0, 0, size, size)
      
      // Generate new random segments
      const maxDistance = size / 2 - 10
      const split = 0.2 + Math.random() * 0.3
      
      // Synchronize at exactly 1 minute (60 seconds)
      // Each segment completes N rotations in exactly 60 seconds
      // Pattern only repeats at 60s when GCD(N1, N2) = 1 (coprime numbers)
      
      // Helper function to calculate GCD
      const gcd = (a: number, b: number): number => {
        return b === 0 ? a : gcd(b, a % b)
      }
      
      // Pick random number of rotations for first segment
      // Range: 6 to 80 rotations per minute for reasonable visual speeds
      const minRotations = 6
      const maxRotations = 80
      const rotations1 = Math.floor(Math.random() * (maxRotations - minRotations + 1)) + minRotations
      
      // Pick a coprime number for second segment
      let rotations2 = Math.floor(Math.random() * (maxRotations - minRotations + 1)) + minRotations
      while (gcd(rotations1, rotations2) !== 1) {
        rotations2 = Math.floor(Math.random() * (maxRotations - minRotations + 1)) + minRotations
      }
      
      // Convert rotations to radians per frame (at 60fps)
      // Speed in radians/second = rotations * 2π / 60
      // Speed in radians/frame = (rotations * 2π / 60) / 60
      const speed1 = ((rotations1 * 2 * Math.PI) / 60) / 60 * (Math.random() < 0.5 ? 1 : -1)
      const speed2 = ((rotations2 * 2 * Math.PI) / 60) / 60 * (Math.random() < 0.5 ? 1 : -1)
      
      segmentsRef.current = {
        length1: maxDistance * split,
        length2: maxDistance * (1 - split),
        speed1: speed1,
        speed2: speed2
      }
      
      // Generate random bright color (not white, good on black)
      const hue = Math.floor(Math.random() * 360)
      const saturation = 70 + Math.floor(Math.random() * 30) // 70-100%
      const lightness = 50 + Math.floor(Math.random() * 20)  // 50-70%
      traceColorRef.current = `hsl(${hue}, ${saturation}%, ${lightness}%)`
    }

    // Initialize on first run
    if (!segmentsRef.current) {
      initializeAnimation()
    }

    // Get contexts for both canvases
    const ctx1 = canvas1Ref.current.getContext('2d')
    const ctx2 = canvas2Ref.current.getContext('2d')
    const centerX = size / 2
    const centerY = size / 2

    const animate = () => {
      if (!ctx1 || !ctx2 || !segmentsRef.current || !traceColorRef.current) return

      frameCountRef.current++

      // After 70 seconds (60s animation + 10s pause), restart with new random values
      if (frameCountRef.current >= 4200) {
        initializeAnimation()
      }

      // Only draw if we're within the first 60 seconds
      if (frameCountRef.current <= 3601) {
        ctx2.clearRect(0, 0, size, size)

        const { length1, length2, speed1, speed2 } = segmentsRef.current
        const { angle1, angle2 } = rotationRef.current

        // First line segment - rotates around center
        const x1 = centerX + length1 * Math.cos(angle1)
        const y1 = centerY + length1 * Math.sin(angle1)

        ctx2.strokeStyle = 'white'
        ctx2.lineWidth = 2
        ctx2.beginPath()
        ctx2.moveTo(centerX, centerY)
        ctx2.lineTo(x1, y1)
        ctx2.stroke()

        // Second line segment - rotates around end of first
        const x2 = x1 + length2 * Math.cos(angle2)
        const y2 = y1 + length2 * Math.sin(angle2)

        ctx2.strokeStyle = 'white'
        ctx2.lineWidth = 2
        ctx2.beginPath()
        ctx2.moveTo(x1, y1)
        ctx2.lineTo(x2, y2)
        ctx2.stroke()

        // Draw trace on canvas 1
        if (prevTipRef.current) {
          ctx1.strokeStyle = traceColorRef.current
          ctx1.lineWidth = 2
          ctx1.beginPath()
          ctx1.moveTo(prevTipRef.current.x, prevTipRef.current.y)
          ctx1.lineTo(x2, y2)
          ctx1.stroke()
        }

        // Update previous tip position
        prevTipRef.current = { x: x2, y: y2 }

        // Update rotation angles
        rotationRef.current.angle1 += speed1
        rotationRef.current.angle2 += speed2
      } else {
        // Between 60-70 seconds, just clear canvas2 (hide line segments) but keep the trace
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
        className={`relative ${ENABLE_ROTATION ? 'animate-spin' : ''}`}
        style={{ 
          width: size, 
          height: size,
          ...(ENABLE_ROTATION ? { animationDuration: '10s' } : {})
        }}
      >
        <canvas
          ref={canvas2Ref}
          width={size}
          height={size}
          className="absolute top-0 left-0"
        />
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
