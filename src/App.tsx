import { useState } from 'react'
import MenuModal from './components/MenuModal'
import FourierEpicycles from './visualizations/FourierEpicycles'
import ParticleFlowField from './visualizations/ParticleFlowField'

const VISUALIZATIONS = [
  { id: 'fourier', name: 'Fourier Epicycles', component: FourierEpicycles },
  { id: 'flow-field', name: 'Particle Flow Field', component: ParticleFlowField },
]

function App() {
  const [isHovered, setIsHovered] = useState(false)
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const [currentViz, setCurrentViz] = useState('fourier')

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
