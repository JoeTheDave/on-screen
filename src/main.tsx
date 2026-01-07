import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

// Request wake lock to keep screen on
let wakeLock: WakeLockSentinel | null = null

async function requestWakeLock() {
  try {
    if ('wakeLock' in navigator) {
      wakeLock = await navigator.wakeLock.request('screen')
      console.log('Wake lock acquired')
      
      wakeLock.addEventListener('release', () => {
        console.log('Wake lock released')
        wakeLock = null
      })
    } else {
      console.warn('Wake Lock API not supported')
    }
  } catch (err) {
    console.error('Failed to acquire wake lock:', err)
  }
}

// Re-request wake lock when page becomes visible again
document.addEventListener('visibilitychange', async () => {
  if (document.visibilityState === 'visible') {
    await requestWakeLock()
  }
})

// Also try to re-acquire on focus
window.addEventListener('focus', async () => {
  if (wakeLock === null || wakeLock.released) {
    await requestWakeLock()
  }
})

// Request wake lock on load
requestWakeLock()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
