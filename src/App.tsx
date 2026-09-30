import { useState, useEffect } from 'react'
import MenuModal from './components/MenuModal'
import FourierEpicycles from './visualizations/FourierEpicycles'
import ParticleFlowField from './visualizations/ParticleFlowField'
import FlockingBoids from './visualizations/FlockingBoids'
import ConwayGameOfLife from './visualizations/ConwayGameOfLife'
import VoronoiDiagrams from './visualizations/VoronoiDiagrams'
import CircularVoronoi from './visualizations/CircularVoronoi'
import OrbitingLights from './visualizations/OrbitingLights'
import SphereField from './visualizations/SphereField'
// import Snakes from './visualizations/Snakes'

const VISUALIZATIONS = [
  { id: 'fourier', name: 'Fourier Epicycles', component: FourierEpicycles },
  { id: 'flow-field', name: 'Particle Flow Field', component: ParticleFlowField },
  { id: 'flocking-boids', name: 'Flocking Boids', component: FlockingBoids },
  { id: 'game-of-life', name: "Game of Life", component: ConwayGameOfLife },
  { id: 'voronoi', name: 'Voronoi Diagram', component: VoronoiDiagrams },
  { id: 'circular-voronoi', name: 'Circular Voronoi', component: CircularVoronoi },
  { id: 'orbiting-lights', name: 'Orbiting Lights', component: OrbitingLights },
  { id: 'sphere-field', name: 'Sphere Field', component: SphereField },
  // { id: 'snakes', name: 'Snakes', component: Snakes },
]

const STORAGE_KEY = 'on-screen-current-viz'

function App() {
  const [isHovered, setIsHovered] = useState(false)
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const [currentViz, setCurrentViz] = useState(() => {
    // Load from localStorage on initial mount
    const stored = localStorage.getItem(STORAGE_KEY)
    return stored || 'fourier'
  })

  // Save to localStorage whenever currentViz changes
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, currentViz)
  }, [currentViz])

  const CurrentVisualization = VISUALIZATIONS.find(v => v.id === currentViz)?.component || FourierEpicycles

  return (
    <div className="relative w-screen h-screen overflow-hidden">
      <CurrentVisualization />

      <div 
        className="absolute -top-[100px] -left-[100px] w-[200px] h-[200px] rounded-full z-40"
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      />
      <button
        onClick={() => setIsMenuOpen(true)}
        className={`absolute top-4 left-4 w-14 h-14 rounded-full bg-white/95 backdrop-blur-sm text-black shadow-xl flex items-center justify-center transition-all duration-400 ease-out cursor-pointer hover:bg-white hover:shadow-2xl z-50 ${
          isHovered ? 'translate-x-0' : '-translate-x-20'
        }`}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
          strokeWidth={2}
          stroke="currentColor"
          className="w-6 h-6"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5"
          />
        </svg>
      </button>

      <MenuModal
        isOpen={isMenuOpen}
        onClose={() => setIsMenuOpen(false)}
        onSelect={setCurrentViz}
        visualizations={VISUALIZATIONS.map(v => ({ id: v.id, name: v.name }))}
        currentViz={currentViz}
      />
    </div>
  )
}

export default App
