import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'

const ORBIT_RADIUS = 8
const ORBIT_SPEED = 0.01
const SPHERE_SIZE = 0.5
const CUBE_SIZE = 3

export default function OrbitingLights() {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!containerRef.current) return

    const container = containerRef.current

    // Setup scene
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x000000)

    // Setup camera
    const camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    )
    camera.position.set(10, 10, 10)
    camera.lookAt(0, 0, 0)

    // Setup renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setSize(window.innerWidth, window.innerHeight)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1
    container.appendChild(renderer.domElement)

    // Setup post-processing for bloom effect
    const composer = new EffectComposer(renderer)
    const renderPass = new RenderPass(scene, camera)
    composer.addPass(renderPass)

    const bloomPass = new UnrealBloomPass(
      new THREE.Vector2(window.innerWidth, window.innerHeight),
      0.6, // strength
      1.2, // radius - larger for smoother falloff
      0.7  // threshold
    )
    composer.addPass(bloomPass)

    // Create icosahedron in center (no emissive, only lit by point lights)
    const icosahedronGeometry = new THREE.IcosahedronGeometry(CUBE_SIZE * 0.7, 0)
    const icosahedronMaterial = new THREE.MeshStandardMaterial({
      color: 0xcccccc,
      metalness: 0.3,
      roughness: 0.4,
    })
    const icosahedron = new THREE.Mesh(icosahedronGeometry, icosahedronMaterial)
    scene.add(icosahedron)

    // Add edges to make the icosahedron more visible
    const edges = new THREE.EdgesGeometry(icosahedronGeometry)
    const lineMaterial = new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 2 })
    const icosahedronEdges = new THREE.LineSegments(edges, lineMaterial)
    scene.add(icosahedronEdges)

    // Red sphere orbiting on X axis
    const redSphereGeometry = new THREE.SphereGeometry(SPHERE_SIZE, 64, 64)
    const redSphereMaterial = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      emissive: 0xff0000,
      emissiveIntensity: 4,
      roughness: 0.3,
      metalness: 0.1,
    })
    const redSphere = new THREE.Mesh(redSphereGeometry, redSphereMaterial)
    scene.add(redSphere)

    const redLight = new THREE.PointLight(0xff0000, 8, 50)
    scene.add(redLight)

    // Green sphere orbiting on Y axis
    const greenSphereGeometry = new THREE.SphereGeometry(SPHERE_SIZE, 64, 64)
    const greenSphereMaterial = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      emissive: 0x00ff00,
      emissiveIntensity: 4,
      roughness: 0.3,
      metalness: 0.1,
    })
    const greenSphere = new THREE.Mesh(greenSphereGeometry, greenSphereMaterial)
    scene.add(greenSphere)

    const greenLight = new THREE.PointLight(0x00ff00, 8, 50)
    scene.add(greenLight)

    // Blue sphere orbiting on Z axis
    const blueSphereGeometry = new THREE.SphereGeometry(SPHERE_SIZE, 64, 64)
    const blueSphereMaterial = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      emissive: 0x4444ff,
      emissiveIntensity: 6,
      roughness: 0.3,
      metalness: 0.1,
    })
    const blueSphere = new THREE.Mesh(blueSphereGeometry, blueSphereMaterial)
    scene.add(blueSphere)

    const blueLight = new THREE.PointLight(0x0000ff, 10, 50)
    scene.add(blueLight)

    // Animation
    let time = 0
    const animate = () => {
      time += ORBIT_SPEED

      // Red sphere orbits on X axis (YZ plane)
      redSphere.position.set(
        0,
        Math.cos(time) * ORBIT_RADIUS,
        Math.sin(time) * ORBIT_RADIUS
      )
      redLight.position.copy(redSphere.position)

      // Green sphere orbits on Y axis (XZ plane)
      greenSphere.position.set(
        Math.cos(time * 1.3) * ORBIT_RADIUS,
        0,
        Math.sin(time * 1.3) * ORBIT_RADIUS
      )
      greenLight.position.copy(greenSphere.position)

      // Blue sphere orbits on Z axis (XY plane)
      blueSphere.position.set(
        Math.cos(time * 0.7) * ORBIT_RADIUS,
        Math.sin(time * 0.7) * ORBIT_RADIUS,
        0
      )
      blueLight.position.copy(blueSphere.position)

      // Slowly rotate the camera for better viewing
      camera.position.x = Math.cos(time * 0.1) * 15
      camera.position.z = Math.sin(time * 0.1) * 15
      camera.lookAt(0, 0, 0)

      composer.render()
      requestAnimationFrame(animate)
    }

    animate()

    // Handle resize
    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight
      camera.updateProjectionMatrix()
      renderer.setSize(window.innerWidth, window.innerHeight)
      composer.setSize(window.innerWidth, window.innerHeight)
    }

    window.addEventListener('resize', handleResize)

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize)
      container.removeChild(renderer.domElement)
      renderer.dispose()
      icosahedronGeometry.dispose()
      icosahedronMaterial.dispose()
      edges.dispose()
      lineMaterial.dispose()
      redSphereGeometry.dispose()
      redSphereMaterial.dispose()
      greenSphereGeometry.dispose()
      greenSphereMaterial.dispose()
      blueSphereGeometry.dispose()
      blueSphereMaterial.dispose()
      scene.clear()
    }
  }, [])

  return <div ref={containerRef} style={{ width: '100vw', height: '100vh' }} />
}
