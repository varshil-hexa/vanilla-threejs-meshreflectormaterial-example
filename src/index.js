import './style/main.css'
import * as THREE from 'three'
import MeshReflectorMaterial from './MeshReflectorMaterial'
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js"
import { RGBELoader } from "three/examples/jsm/loaders/RGBELoader.js";
import { WebGPURenderer } from 'three/webgpu';

async function main() {
    const scene = new THREE.Scene()

    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 100)
    camera.position.set(2, 2, -2)
    scene.add(camera)

    const renderer = new WebGPURenderer({
        canvas: document.querySelector('.webgl'),
        antialias: true
    })
    await renderer.init()
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.setSize(window.innerWidth, window.innerHeight)

    const controls = new OrbitControls(camera, renderer.domElement);

    // Load HDR environment map
    const rgbeLoader = new RGBELoader();
    rgbeLoader.load("/env.hdr", (tex) => {
        tex.mapping = THREE.EquirectangularReflectionMapping;
        scene.environment = tex;
        scene.background = tex;
    });

    const cube = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial({ color: 0xff00ff }))
    scene.add(cube)

    const torusKnot = new THREE.Mesh(new THREE.TorusKnotGeometry(0.5, 0.1, 100, 16), new THREE.MeshStandardMaterial({ color: 0x00ff00 }));
    torusKnot.position.set(2, 0.5, 2)
    scene.add(torusKnot)

    const plane = new THREE.Mesh(new THREE.PlaneGeometry(10, 10))
    plane.position.y = -0.5
    plane.rotation.x = -Math.PI / 2
    scene.add(plane)

    plane.material = new MeshReflectorMaterial(renderer, camera, scene, plane, {
        resolution: 1024,
        blur: [512, 128],
        mixBlur: 2.5,
        mixContrast: 1.5,
        mirror: 0,
        bufferSamples: 4
    });

    plane.material.setValues({
        roughnessMap: new THREE.TextureLoader().load("/roughness.jpg"),
        normalMap: new THREE.TextureLoader().load("/normal.png"),
        normalScale: new THREE.Vector2(0.3, 0.3)
    })

    const loop = () => {
        cube.rotation.y += 0.01
        torusKnot.rotation.x += 0.01
        torusKnot.rotation.z += 0.01

        if (plane.material.update) plane.material.update()

        controls.update();

        renderer.render(scene, camera)
    }

    renderer.setAnimationLoop(loop)
}

main()