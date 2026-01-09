import { useEffect, useState, useRef } from 'react'

// Configuration constants
const CELL_SIZE = 10
const BUFFER_CELLS = 10
const PATTERN_SPAWN_INTERVAL = 12 // frames between pattern spawns
const CELL_COLOR_SATURATION = 70
const CELL_COLOR_LIGHTNESS = 60


export default function ConwayGameOfLife() {
  const [width, setWidth] = useState(0)
  const [height, setHeight] = useState(0)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number | undefined>(undefined)
  const gridRef = useRef<boolean[][]>([])
  const nextGridRef = useRef<boolean[][]>([])
  const colsRef = useRef(0)
  const rowsRef = useRef(0)
  const frameCountRef = useRef(0)

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

    // Calculate grid dimensions including buffer
    const cols = Math.ceil(width / CELL_SIZE) + BUFFER_CELLS * 2
    const rows = Math.ceil(height / CELL_SIZE) + BUFFER_CELLS * 2
    colsRef.current = cols
    rowsRef.current = rows

    /**
     * Initialize empty grid
     */
    const createEmptyGrid = (): boolean[][] => {
      return Array.from({ length: rows }, () => Array(cols).fill(false))
    }

    /**
     * Count live neighbors for a cell
     */
    const countNeighbors = (grid: boolean[][], row: number, col: number): number => {
      let count = 0
      for (let i = -1; i <= 1; i++) {
        for (let j = -1; j <= 1; j++) {
          if (i === 0 && j === 0) continue
          const newRow = row + i
          const newCol = col + j
          if (newRow >= 0 && newRow < rows && newCol >= 0 && newCol < cols) {
            if (grid[newRow][newCol]) count++
          }
        }
      }
      return count
    }

    /**
     * Apply Conway's Game of Life rules with aging and fading
     */
    const updateGrid = () => {
      const current = gridRef.current
      const next = nextGridRef.current

      for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
          const neighbors = countNeighbors(current, row, col)
          const isAlive = current[row][col]

          if (isAlive) {
            // Cell stays alive if it has 2 or 3 neighbors
            next[row][col] = neighbors === 2 || neighbors === 3
          } else {
            // Dead cell becomes alive if it has exactly 3 neighbors
            next[row][col] = neighbors === 3
          }
        }
      }

      // Swap grids
      gridRef.current = next
      nextGridRef.current = current
    }

    /**
     * Place a pattern on the grid
     */
    const placePattern = (pattern: number[][], row: number, col: number) => {
      pattern.forEach(([r, c]) => {
        const newRow = row + r
        const newCol = col + c
        if (newRow >= 0 && newRow < rows && newCol >= 0 && newCol < cols) {
          gridRef.current[newRow][newCol] = true
        }
      })
    }

    /**
     * Patterns library
     */
    const patterns = {
      glider: {
        'down-right': [[0, 1], [1, 2], [2, 0], [2, 1], [2, 2]],
        'down-left': [[0, 1], [1, 0], [2, 0], [2, 1], [2, 2]],
        'up-right': [[0, 0], [0, 1], [0, 2], [1, 2], [2, 1]],
        'up-left': [[0, 0], [0, 1], [0, 2], [1, 0], [2, 1]]
      },
      lwss: { // Lightweight spaceship
        'right': [[0, 1], [0, 4], [1, 0], [2, 0], [2, 4], [3, 0], [3, 1], [3, 2], [3, 3]],
        'left': [[0, 0], [0, 3], [1, 4], [2, 0], [2, 4], [3, 1], [3, 2], [3, 3], [3, 4]],
        'down': [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 4], [2, 4], [3, 1], [3, 4]],
        'up': [[0, 0], [0, 3], [1, 0], [2, 0], [2, 4], [3, 0], [3, 1], [3, 2], [3, 3]]
      },
      mwss: { // Middleweight spaceship
        'right': [[0, 2], [1, 0], [1, 4], [2, 5], [3, 0], [3, 5], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5]],
        'left': [[0, 3], [1, 1], [1, 5], [2, 0], [3, 0], [3, 5], [4, 0], [4, 1], [4, 2], [4, 3], [4, 4]]
      },
      blinker: [[0, 0], [0, 1], [0, 2]],
      toad: [[0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2]]
    }

    /**
     * Spawn a random pattern from off-screen area
     */
    const spawnRandomPattern = () => {
      const patternTypes = ['glider', 'glider', 'glider', 'lwss', 'mwss', 'blinker', 'toad']
      const patternType = patternTypes[Math.floor(Math.random() * patternTypes.length)]
      const side = Math.floor(Math.random() * 4) // 0=top, 1=right, 2=bottom, 3=left
      let row: number, col: number, pattern: number[][]

      switch (side) {
        case 0: // Top buffer
          row = Math.floor(Math.random() * BUFFER_CELLS)
          col = BUFFER_CELLS + Math.floor(Math.random() * (cols - BUFFER_CELLS * 2))
          if (patternType === 'glider') {
            pattern = patterns.glider[Math.random() < 0.5 ? 'down-right' : 'down-left']
          } else if (patternType === 'lwss') {
            pattern = patterns.lwss['down']
          } else if (patternType === 'mwss') {
            pattern = patterns.mwss['right']
          } else {
            pattern = patterns[patternType as 'blinker' | 'toad']
          }
          break
        case 1: // Right buffer
          row = BUFFER_CELLS + Math.floor(Math.random() * (rows - BUFFER_CELLS * 2))
          col = cols - BUFFER_CELLS + Math.floor(Math.random() * BUFFER_CELLS)
          if (patternType === 'glider') {
            pattern = patterns.glider[Math.random() < 0.5 ? 'down-left' : 'up-left']
          } else if (patternType === 'lwss') {
            pattern = patterns.lwss['left']
          } else if (patternType === 'mwss') {
            pattern = patterns.mwss['left']
          } else {
            pattern = patterns[patternType as 'blinker' | 'toad']
          }
          break
        case 2: // Bottom buffer
          row = rows - BUFFER_CELLS + Math.floor(Math.random() * BUFFER_CELLS)
          col = BUFFER_CELLS + Math.floor(Math.random() * (cols - BUFFER_CELLS * 2))
          if (patternType === 'glider') {
            pattern = patterns.glider[Math.random() < 0.5 ? 'up-right' : 'up-left']
          } else if (patternType === 'lwss') {
            pattern = patterns.lwss['up']
          } else if (patternType === 'mwss') {
            pattern = patterns.mwss['left']
          } else {
            pattern = patterns[patternType as 'blinker' | 'toad']
          }
          break
        default: // Left buffer
          row = BUFFER_CELLS + Math.floor(Math.random() * (rows - BUFFER_CELLS * 2))
          col = Math.floor(Math.random() * BUFFER_CELLS)
          if (patternType === 'glider') {
            pattern = patterns.glider[Math.random() < 0.5 ? 'down-right' : 'up-right']
          } else if (patternType === 'lwss') {
            pattern = patterns.lwss['right']
          } else if (patternType === 'mwss') {
            pattern = patterns.mwss['right']
          } else {
            pattern = patterns[patternType as 'blinker' | 'toad']
          }
          break
      }

      placePattern(pattern, row, col)
    }

    /**
     * Initialize animation
     */
    const initializeAnimation = () => {
      gridRef.current = createEmptyGrid()
      nextGridRef.current = createEmptyGrid()
      frameCountRef.current = 0

      // Spawn more initial patterns for immediate visual interest
      for (let i = 0; i < 8; i++) {
        spawnRandomPattern()
      }
    }

    initializeAnimation()

    /**
     * Main animation loop
     */
    const animate = () => {
      frameCountRef.current++

      // Spawn new patterns periodically
      if (frameCountRef.current % PATTERN_SPAWN_INTERVAL === 0) {
        spawnRandomPattern()
      }

      // Update game state
      updateGrid()

      // Clear canvas
      ctx.fillStyle = 'black'
      ctx.fillRect(0, 0, width, height)

      // Draw visible cells (only the viewport, not the buffer)
      const offsetX = BUFFER_CELLS * CELL_SIZE
      const offsetY = BUFFER_CELLS * CELL_SIZE
      
      // Animate hue over time
      const hueOffset = (frameCountRef.current * 0.5) % 360

      for (let row = BUFFER_CELLS; row < rows - BUFFER_CELLS; row++) {
        for (let col = BUFFER_CELLS; col < cols - BUFFER_CELLS; col++) {
          if (gridRef.current[row][col]) {
            const hue = (((row + col) * 7) + hueOffset) % 360
            ctx.fillStyle = `hsl(${hue}, ${CELL_COLOR_SATURATION}%, ${CELL_COLOR_LIGHTNESS}%)`
            
            const x = col * CELL_SIZE - offsetX
            const y = row * CELL_SIZE - offsetY
            ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE)
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
  }, [width, height])

  return (
    <div className="w-full h-screen">
      <canvas ref={canvasRef} width={width} height={height} className="w-full h-full" />
    </div>
  )
}
