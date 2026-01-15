import { useEffect, useRef, useState } from 'react'

// Configuration
const SNAKE_COUNT = 5
const SEGMENT_COUNT = 10
const SEGMENT_RADIUS = 8
const SNAKE_SPEED = 2
const TURN_STRENGTH = 0.03 // radians per frame
const TURN_DURATION_FRAMES = 60 // 1 second at 60fps
const FOOD_COUNT = 100
const FOOD_RADIUS = 5

// Types
interface Segment {
  x: number
  y: number
  radius: number
  direction: number // angle in radians pointing "up" for this segment
}

interface Food {
  x: number
  y: number
  radius: number
}

interface Snake {
  segments: Segment[]
  direction: number // angle in radians
  speed: number
  turnDirection: number // -1, 0, or 1
  turnFramesRemaining: number
  color: { light: string; dark: string } // for gradient and outline
  foodEaten: number
  alive: boolean
  opacity: number // for fade-out effect when dying
}

export default function Snakes() {
  const [size, setSize] = useState({ width: 0, height: 0 })
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number | undefined>(undefined)
  const snakesRef = useRef<Snake[]>([])
  const foodRef = useRef<Food[]>([])
  const animationTimeRef = useRef<number>(0)

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
      // Calculate buffer to keep snake bodies away from edges
      const bodyLength = SEGMENT_COUNT * SEGMENT_RADIUS * 2
      const buffer = bodyLength + SEGMENT_RADIUS
      
      // Divide canvas into grid regions for even distribution
      const cols = Math.ceil(Math.sqrt(SNAKE_COUNT))
      const rows = Math.ceil(SNAKE_COUNT / cols)
      const cellWidth = size.width / cols
      const cellHeight = size.height / rows
      
      for (let snakeIndex = 0; snakeIndex < SNAKE_COUNT; snakeIndex++) {
        const segments: Segment[] = []
        
        // Determine which grid cell this snake belongs to
        const col = snakeIndex % cols
        const row = Math.floor(snakeIndex / cols)
        
        // Random position within the cell, respecting edge buffer
        const cellMinX = col * cellWidth + buffer
        const cellMaxX = (col + 1) * cellWidth - buffer
        const cellMinY = row * cellHeight + buffer
        const cellMaxY = (row + 1) * cellHeight - buffer
        
        const startX = cellMinX + Math.random() * Math.max(0, cellMaxX - cellMinX)
        const startY = cellMinY + Math.random() * Math.max(0, cellMaxY - cellMinY)
        const startDirection = Math.random() * Math.PI * 2
        
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
          turnFramesRemaining: 0,
          color: { light: '#66ff66', dark: '#228822' }, // light green center, dark green outer
          foodEaten: 0,
          alive: true,
          opacity: 1
        })
      }
    }

    // Initialize food
    if (foodRef.current.length === 0) {
      for (let i = 0; i < FOOD_COUNT; i++) {
        foodRef.current.push({
          x: Math.random() * size.width,
          y: Math.random() * size.height,
          radius: FOOD_RADIUS
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

      // Increment animation time for pulsation
      animationTimeRef.current += 1

      // Draw food with pulsating glow
      const pulseScale = 1 + Math.sin(animationTimeRef.current * 0.1) * 0.105
      for (const food of foodRef.current) {
        const pulsatingRadius = food.radius * pulseScale
        const gradient = ctx.createRadialGradient(
          food.x, food.y, 0,
          food.x, food.y, pulsatingRadius
        )
        gradient.addColorStop(0, '#ffff00')
        gradient.addColorStop(0.4, '#ffdd00')
        gradient.addColorStop(1, 'rgba(255, 221, 0, 0)')
        
        ctx.fillStyle = gradient
        ctx.beginPath()
        ctx.arc(food.x, food.y, pulsatingRadius, 0, Math.PI * 2)
        ctx.fill()
      }

      // Update and draw each snake
      for (const snake of snakes) {
        if (!snake.alive) continue // Skip dead snakes

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

        const head = snake.segments[0]
        
        // DANGER AVOIDANCE: Only check what's directly ahead
        let bestTurnDirection = 0
        let minDanger = Infinity
        
        // Test three directions: left turn, straight, right turn
        for (const testTurn of [-1, 0, 1]) {
          const testAngle = snake.direction + testTurn * TURN_STRENGTH * 10
          const testX = head.x + Math.cos(testAngle) * head.radius * 4
          const testY = head.y + Math.sin(testAngle) * head.radius * 4
          
          let dangerScore = 0
          
          for (const otherSnake of snakes) {
            if (!otherSnake.alive) continue
            for (let i = 0; i < otherSnake.segments.length; i++) {
              // Skip own first 4 segments
              if (otherSnake === snake && i < 4) continue
              
              const seg = otherSnake.segments[i]
              const dist = Math.sqrt((seg.x - testX) ** 2 + (seg.y - testY) ** 2)
              
              if (dist < head.radius * 2) {
                dangerScore += 10
              }
            }
          }
          
          if (dangerScore < minDanger) {
            minDanger = dangerScore
            bestTurnDirection = testTurn
          }
        }
        
        // Apply avoidance turn if danger detected
        if (minDanger > 0) {
          snake.direction += bestTurnDirection * TURN_STRENGTH * 8
        }
        
        // FOOD CHASING: Find closest food within detection radius
        const detectionRadius = head.radius * 15
        let closestFood: Food | null = null
        let closestDist = Infinity
        
        for (const food of foodRef.current) {
          const dist = Math.sqrt((food.x - head.x) ** 2 + (food.y - head.y) ** 2)
          if (dist < detectionRadius && dist < closestDist) {
            closestFood = food
            closestDist = dist
          }
        }
        
        // Turn toward food (when safe)
        if (closestFood && minDanger === 0) {
          const angleToFood = Math.atan2(closestFood.y - head.y, closestFood.x - head.x)
          let diff = angleToFood - snake.direction
          while (diff > Math.PI) diff -= Math.PI * 2
          while (diff < -Math.PI) diff += Math.PI * 2
          
          if (Math.abs(diff) > 0.01) {
            snake.direction += Math.sign(diff) * TURN_STRENGTH * 1.5
          }
        }

        // Update head position
        head.x += Math.cos(snake.direction) * snake.speed
        head.direction = snake.direction
        head.y += Math.sin(snake.direction) * snake.speed

        // Wrap head around screen edges using modulo
        head.x = ((head.x % size.width) + size.width) % size.width
        head.y = ((head.y % size.height) + size.height) % size.height

        // Check for food collision
        for (let i = foodRef.current.length - 1; i >= 0; i--) {
          const food = foodRef.current[i]
          const dist = Math.sqrt((food.x - head.x) ** 2 + (food.y - head.y) ** 2)
          
          if (dist < head.radius + food.radius) {
            foodRef.current.splice(i, 1)
            snake.foodEaten++
            
            // Grow segment every 5 food
            if (snake.foodEaten % 5 === 0) {
              const tail = snake.segments[snake.segments.length - 1]
              const prevTail = snake.segments[snake.segments.length - 2]
              
              let dx = tail.x - prevTail.x
              let dy = tail.y - prevTail.y
              if (Math.abs(dx) > size.width / 2) dx = dx > 0 ? dx - size.width : dx + size.width
              if (Math.abs(dy) > size.height / 2) dy = dy > 0 ? dy - size.height : dy + size.height
              const angle = Math.atan2(dy, dx)
              
              snake.segments.push({
                x: tail.x + Math.cos(angle) * tail.radius,
                y: tail.y + Math.sin(angle) * tail.radius,
                radius: tail.radius,
                direction: angle + Math.PI
              })
            }
            
            // Grow radius every 30 food
            if (snake.foodEaten % 30 === 0) {
              for (const seg of snake.segments) {
                seg.radius += 1
              }
            }
            
            // Only spawn replacement food if we have less than 100
            if (foodRef.current.length < FOOD_COUNT) {
              foodRef.current.push({
                x: Math.random() * size.width,
                y: Math.random() * size.height,
                radius: FOOD_RADIUS
              })
            }
            break
          }
        }
        
        // Check for collisions with OTHER snakes
        for (const otherSnake of snakes) {
          if (!otherSnake.alive || otherSnake === snake) continue // Skip self and dead snakes
          
          // Check for HEAD-TO-HEAD collision first (both die)
          const otherHead = otherSnake.segments[0]
          const headToHeadDist = Math.sqrt((otherHead.x - head.x) ** 2 + (otherHead.y - head.y) ** 2)
          if (headToHeadDist < head.radius + otherHead.radius - 2) {
            // Both snakes die in head-to-head collision
            snake.alive = false
            snake.opacity = 1 // Start fading
            otherSnake.alive = false
            otherSnake.opacity = 1 // Start fading
            
            // Spawn food from both bodies
            for (const deadSnake of [snake, otherSnake]) {
              const foodToSpawn = Math.floor(deadSnake.foodEaten / 2)
              for (let f = 0; f < foodToSpawn; f++) {
                const randomSeg = deadSnake.segments[Math.floor(Math.random() * deadSnake.segments.length)]
                foodRef.current.push({
                  x: randomSeg.x + (Math.random() - 0.5) * 20,
                  y: randomSeg.y + (Math.random() - 0.5) * 20,
                  radius: FOOD_RADIUS
                })
              }
            }
            break
          }
          
          // Check collision with other snake's body segments
          for (let i = 0; i < otherSnake.segments.length; i++) {
            const seg = otherSnake.segments[i]
            const dist = Math.sqrt((seg.x - head.x) ** 2 + (seg.y - head.y) ** 2)
            
            if (dist < head.radius + seg.radius - 2) {
              // DIE
              snake.alive = false
              snake.opacity = 1 // Start fading
              
              // Spawn food from body
              const foodToSpawn = Math.floor(snake.foodEaten / 2)
              for (let f = 0; f < foodToSpawn; f++) {
                const randomSeg = snake.segments[Math.floor(Math.random() * snake.segments.length)]
                foodRef.current.push({
                  x: randomSeg.x + (Math.random() - 0.5) * 20,
                  y: randomSeg.y + (Math.random() - 0.5) * 20,
                  radius: FOOD_RADIUS
                })
              }
              break
            }
          }
          
          if (!snake.alive) break
        }

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

        // Draw filled circles with radial gradient (from last to first)
        ctx.globalAlpha = snake.opacity // Apply fade-out opacity
        for (let i = snake.segments.length - 1; i >= 0; i--) {
          const segment = snake.segments[i]
          
          // Create radial gradient
          const gradient = ctx.createRadialGradient(
            segment.x, segment.y, 0,
            segment.x, segment.y, segment.radius
          )
          gradient.addColorStop(0, snake.color.light) // light green in center
          gradient.addColorStop(1, snake.color.dark)  // dark green on outer edge
          
          // Draw filled circle
          ctx.fillStyle = gradient
          ctx.beginPath()
          ctx.arc(segment.x, segment.y, segment.radius, 0, Math.PI * 2)
          ctx.fill()
        }

        // Draw snake outline by connecting dots continuously
        ctx.strokeStyle = snake.color.light // outline matches light center color
        ctx.lineWidth = 1
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
        
        // Draw detection radius circle (reuse already calculated detectionRadius from above)
        ctx.strokeStyle = 'rgba(128, 128, 128, 0.3)'
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.arc(head.x, head.y, head.radius * 15, 0, Math.PI * 2)
        ctx.stroke()

        // Draw food counter
        ctx.fillStyle = '#ffffff'
        ctx.font = 'bold 16px monospace'
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.fillText(
          snake.foodEaten.toString(),
          head.x,
          head.y - head.radius - 15
        )
        
        // Draw eyes on the head segment (head already declared above)
        const eyeRadius = snake.segments[0].radius * 0.15
        const forwardOffset = snake.segments[0].radius * 0.4 // position towards front
        const lateralOffset = snake.segments[0].radius * 0.35 // spacing from center line
        
        // Calculate eye positions
        const leftEyeX = snake.segments[0].x + Math.cos(snake.direction) * forwardOffset + Math.cos(snake.direction + Math.PI / 2) * lateralOffset
        const leftEyeY = snake.segments[0].y + Math.sin(snake.direction) * forwardOffset + Math.sin(snake.direction + Math.PI / 2) * lateralOffset
        const rightEyeX = snake.segments[0].x + Math.cos(snake.direction) * forwardOffset + Math.cos(snake.direction - Math.PI / 2) * lateralOffset
        const rightEyeY = snake.segments[0].y + Math.sin(snake.direction) * forwardOffset + Math.sin(snake.direction - Math.PI / 2) * lateralOffset
        
        // Draw left eye (red - opposite hue from green)
        ctx.fillStyle = '#ff0000'
        ctx.beginPath()
        ctx.arc(leftEyeX, leftEyeY, eyeRadius, 0, Math.PI * 2)
        ctx.fill()
        
        // Draw right eye
        ctx.beginPath()
        ctx.arc(rightEyeX, rightEyeY, eyeRadius, 0, Math.PI * 2)
        ctx.fill()
        
        // Reset global alpha
        ctx.globalAlpha = 1
      }
      
      // FADE OUT and RESPAWN dead snakes
      for (let i = 0; i < snakes.length; i++) {
        if (!snakes[i].alive) {
          // Fade out gradually
          snakes[i].opacity -= 0.02
          
          // Only respawn when fully faded
          if (snakes[i].opacity <= 0) {
            const segments: Segment[] = []
            const startX = Math.random() * size.width
            const startY = Math.random() * size.height
            const startDirection = Math.random() * Math.PI * 2
            
            // Create segments in a line
            for (let j = 0; j < SEGMENT_COUNT; j++) {
              segments.push({
                x: startX - Math.cos(startDirection) * j * (SEGMENT_RADIUS * 2),
                y: startY - Math.sin(startDirection) * j * (SEGMENT_RADIUS * 2),
                radius: SEGMENT_RADIUS,
                direction: startDirection
              })
            }
            
            // Replace the dead snake
            snakes[i] = {
              segments,
              direction: startDirection,
              speed: SNAKE_SPEED,
              turnDirection: 0,
              turnFramesRemaining: 0,
              color: { light: '#66ff66', dark: '#228822' },
              foodEaten: 0,
              alive: true,
              opacity: 1
            }
          }
        }
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
