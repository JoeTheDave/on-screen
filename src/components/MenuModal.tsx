interface MenuModalProps {
  isOpen: boolean
  onClose: () => void
  onSelect: (vizId: string) => void
  visualizations: { id: string; name: string }[]
  currentViz: string
}

export default function MenuModal({ isOpen, onClose, onSelect, visualizations }: MenuModalProps) {
  if (!isOpen) return null

  return (
    <div 
      className="fixed inset-0 bg-black/70 flex items-center justify-center z-50"
      onClick={onClose}
    >
      <div 
        className="relative bg-neutral-900 rounded-2xl p-8 max-w-sm w-full mx-4 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="space-y-2">
          {visualizations.map((viz) => (
            <button
              key={viz.id}
              onClick={() => {
                onSelect(viz.id)
                onClose()
              }}
              className={`w-full text-left px-4 py-3 rounded-lg transition-colors bg-gray-600 text-white hover:bg-gray-700  cursor-pointer`}
            >
              {viz.name}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
