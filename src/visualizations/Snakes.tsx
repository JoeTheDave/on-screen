import { useEffect, useRef, useState } from 'react'

// Configuration
const SNAKE_COUNT = 5
const SEGMENT_COUNT = 10
const SEGMENT_RADIUS = 20
const SNAKE_SPEED = 2
const TURN_STRENGTH = 0.03 // radians per frame
const TURN_DURATION_FRAMES = 60 // 1 second at 60fps

// Types
interface Segment {
  x: number
  y: number
  radius: number
  direction: number // angle in radians pointing "up" for this segment
}

interface Snake {
  segments: Segment[]
  direction: number // angle in radians
  speed: number
  turnDirection: number // -1, 0, or 1
  turnFramesRemaining: number
}

export default function Snakes() {
  const [size, setSize] = useState({ width: 0, height: 0 })
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number | undefined>(undefined)
  const snakesRef = useRef<Snake[]>([])

  // Handle window resize
  useEffect(() => {
    const updateSize = () => {
      setSize({
        width: window.innerWidth,
        height: window.innerHeight
      })
    }

    updateSize()
    window.addEventListener('resize', updateSize)
    return () => window.removeEventListener('resize', updateSize)
  }, [])

  // Main animation effect
  useEffect(() => {
    if (!canvasRef.current || size.width === 0 || size.height === 0) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Initialize snakes
    if (snakesRef.current.length === 0) {
      for (let snakeIndex = 0; snakeIndex < SNAKE_COUNT; snakeIndex++) {
        const segments: Segment[] = []
        // Distribute snakes across the screen
        const startX = (size.width / (SNAKE_COUNT + 1)) * (snakeIndex + 1)
        const startY = size.height / 2
        const startDirection = (Math.PI * 2 / SNAKE_COUNT) * snakeIndex
        
        // Create segments in a line based on starting direction
        for (let i = 0; i < SEGMENT_COUNT; i++) {
          segments.push({
            x: startX - Math.cos(startDirection) * i * (SEGMENT_RADIUS * 2),
            y: startY - Math.sin(startDirection) * i * (SEGMENT_RADIUS * 2),
            radius: SEGMENT_RADIUS,
            direction: startDirection
          })
        }

        snakesRef.current.push({
          segments,
          direction: startDirection,
          speed: SNAKE_SPEED,
          turnDirection: 0,
          turnFramesRemaining: 0
        })
      }
    }

    // Animation loop
    const animate = () => {
      const snakes = snakesRef.current
      if (snakes.length === 0) return

      // Clear canvas
      ctx.fillStyle = '#000000'
      ctx.fillRect(0, 0, size.width, size.height)

      // Update and draw each snake
      for (const snake of snakes) {
        // Update turn direction if needed
        if (snake.turnFramesRemaining <= 0) {
          // Choose new turn direction: -1 (left), 0 (straight), 1 (right)
          snake.turnDirection = Math.floor(Math.random() * 3) - 1
          snake.turnFramesRemaining = TURN_DURATION_FRAMES
        } else {
          snake.turnFramesRemaining--
        }

        // Apply turn
        snake.direction += snake.turnDirection * TURN_STRENGTH

        // Update head position
        const head = snake.segments[0]
        head.x += Math.cos(snake.direction) * snake.speed
        head.direction = snake.direction
        head.y += Math.sin(snake.direction) * snake.speed

        // Wrap head around screen edges using modulo
        head.x = ((head.x % size.width) + size.width) % size.width
        head.y = ((head.y % size.height) + size.height) % size.height

        // Update each segment to follow the one before it
        for (let i = 1; i < snake.segments.length; i++) {
          const current = snake.segments[i]
          const parent = snake.segments[i - 1]

          // Calculate angle from parent to current segment, accounting for wrapping
          let dx = current.x - parent.x
          let dy = current.y - parent.y
          
          // Handle wrapping: if distance is more than half screen, wrap around
          if (Math.abs(dx) > size.width / 2) {
            dx = dx > 0 ? dx - size.width : dx + size.width
          }
          if (Math.abs(dy) > size.height / 2) {
            dy = dy > 0 ? dy - size.height : dy + size.height
          }
          
          const angle = Math.atan2(dy, dx)

          // Place current segment on the edge of parent's circle
          current.x = parent.x + Math.cos(angle) * parent.radius
          current.y = parent.y + Math.sin(angle) * parent.radius
          // Direction for non-head segments points back toward parent (opposite of angle)
          current.direction = angle + Math.PI
          
          // Wrap segment around screen edges
          current.x = ((current.x % size.width) + size.width) % size.width
          current.y = ((current.y % size.height) + size.height) % size.height
        }

        // Draw snake outline by connecting dots continuously
        ctx.strokeStyle = '#ff0000'
        ctx.lineWidth = 2
        ctx.beginPath()
        
        let lastX: number | null = null
        let lastY: number | null = null
        let firstX: number | null = null
        let firstY: number | null = null
        const maxDistance = Math.min(size.width, size.height) / 3 // Threshold for detecting wraps
        
        // Forward pass: dots 0-5 (first half) for each segment from head to tail
        for (let segIdx = 0; segIdx < snake.segments.length; segIdx++) {
          const segment = snake.segments[segIdx]
          
          for (let dotIdx = 0; dotIdx <= 5; dotIdx++) {
            const dotAngle = segment.direction + (dotIdx * Math.PI * 2 / 12)
            const dotX = segment.x + Math.cos(dotAngle) * segment.radius
            const dotY = segment.y + Math.sin(dotAngle) * segment.radius
            
            // Check if this dot is inside any other segment
            let isInsideOther = false
            for (let otherIdx = 0; otherIdx < snake.segments.length; otherIdx++) {
              if (otherIdx === segIdx) continue
              
              const other = snake.segments[otherIdx]
              const dx = dotX - other.x
              const dy = dotY - other.y
              const distance = Math.sqrt(dx * dx + dy * dy)
              
              if (distance < other.radius) {
                isInsideOther = true
                break
              }
            }
            
            if (!isInsideOther) {
              // Check if we need to start a new segment (due to screen wrap)
              if (lastX !== null && lastY !== null) {
                const distanceFromLast = Math.sqrt(
                  Math.pow(dotX - lastX, 2) + Math.pow(dotY - lastY, 2)
                )
                if (distanceFromLast > maxDistance) {
                  // Too far, likely a screen wrap - start new segment
                  ctx.moveTo(dotX, dotY)
                } else {
                  ctx.lineTo(dotX, dotY)
                }
              } else {
                ctx.moveTo(dotX, dotY)
                firstX = dotX
                firstY = dotY
              }
              lastX = dotX
              lastY = dotY
            }
          }
        }
        
        // Backward pass: dots 6-11 (second half) for each segment from tail to head
        for (let segIdx = snake.segments.length - 1; segIdx >= 0; segIdx--) {
          const segment = snake.segments[segIdx]
          
          for (let dotIdx = 6; dotIdx <= 11; dotIdx++) {
            const dotAngle = segment.direction + (dotIdx * Math.PI * 2 / 12)
            const dotX = segment.x + Math.cos(dotAngle) * segment.radius
            const dotY = segment.y + Math.sin(dotAngle) * segment.radius
            
            // Check if this dot is inside any other segment
            let isInsideOther = false
            for (let otherIdx = 0; otherIdx < snake.segments.length; otherIdx++) {
              if (otherIdx === segIdx) continue
              
              const other = snake.segments[otherIdx]
              const dx = dotX - other.x
              const dy = dotY - other.y
              const distance = Math.sqrt(dx * dx + dy * dy)
              
              if (distance < other.radius) {
                isInsideOther = true
                break
              }
            }
            
            if (!isInsideOther) {
              // Check if we need to start a new segment (due to screen wrap)
              if (lastX !== null && lastY !== null) {
                const distanceFromLast = Math.sqrt(
                  Math.pow(dotX - lastX, 2) + Math.pow(dotY - lastY, 2)
                )
                if (distanceFromLast > maxDistance) {
                  // Too far, likely a screen wrap - start new segment
                  ctx.moveTo(dotX, dotY)
                } else {
                  ctx.lineTo(dotX, dotY)
                }
              } else {
                ctx.moveTo(dotX, dotY)
              }
              lastX = dotX
              lastY = dotY
            }
          }
        }
        
        // Close the path by connecting back to the first point (if not wrapped)
        if (firstX !== null && firstY !== null && lastX !== null && lastY !== null) {
          const distanceToFirst = Math.sqrt(
            Math.pow(firstX - lastX, 2) + Math.pow(firstY - lastY, 2)
          )
          if (distanceToFirst <= maxDistance) {
            ctx.lineTo(firstX, firstY)
          }
        }
        
        ctx.stroke()
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
    <div className="w-screen h-screen bg-black">
      <canvas
        ref={canvasRef}
        width={size.width}
        height={size.height}
        className="w-full h-full"
      />
    </div>
  )
}
