import { Component, type ReactNode } from 'react'

interface Props {
  children: ReactNode
}

interface State {
  error: Error | null
}

// Keeps one failing visualization (most often: the browser refused a WebGL context) from
// unmounting the whole app, so the menu stays usable. App keys this by visualization id,
// which resets it on every switch.
export default class VisualizationErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  render() {
    const { error } = this.state
    if (!error) return this.props.children

    const webgl = /webgl/i.test(error.message)
    return (
      <div className="w-full h-full flex items-center justify-center bg-black p-6">
        <div className="max-w-md text-center text-neutral-400 text-sm space-y-2">
          <p className="text-neutral-200">This visualization couldn't start.</p>
          {webgl ? (
            <p>
              The browser refused to create a WebGL (3D) context. Chrome turns WebGL off for the rest
              of the session after its GPU process crashes — restarting the browser usually brings it
              back. chrome://gpu shows the current status.
            </p>
          ) : (
            <p>{error.message}</p>
          )}
        </div>
      </div>
    )
  }
}
