interface MenuModalProps {
  isOpen: boolean
  onClose: () => void
  onSelect: (vizId: string) => void
  visualizations: { id: string; name: string }[]
  currentViz: string
}

export default function MenuModal({ isOpen, onClose, onSelect, visualizations, currentViz }: MenuModalProps) {
  if (!isOpen) return null

  return (
    <div 
      className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4"
      onClick={onClose}
    >
      <div 
        // Scrolls within the viewport on small screens; two columns once there's room
        className="relative bg-neutral-900 rounded-2xl p-4 sm:p-6 w-full max-w-sm sm:max-w-xl max-h-[calc(100dvh-2rem)] overflow-y-auto overscroll-contain shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {visualizations.map((viz) => (
            <button
              key={viz.id}
              onClick={() => {
                onSelect(viz.id)
                onClose()
              }}
              className={`w-full text-left px-4 py-3 rounded-lg transition-colors text-white cursor-pointer ${
                viz.id === currentViz ? 'bg-gray-500 ring-2 ring-white/70' : 'bg-gray-600 hover:bg-gray-700'
              }`}
            >
              {viz.name}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
