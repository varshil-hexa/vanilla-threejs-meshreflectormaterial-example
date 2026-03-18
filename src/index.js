import './style/main.css'
import * as THREE from 'three/webgpu'
import { pass } from 'three/tsl';
import { bloom } from 'three/examples/jsm/tsl/display/BloomNode.js';
import { computeBoundsTree, disposeBoundsTree } from 'three-mesh-bvh'; // Import BVH tools
import MeshRefractionMaterial from './MeshRefractionMaterial_WebGPU'
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js"
import { RGBELoader } from "three/examples/jsm/loaders/RGBELoader.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { DRACOLoader } from "three/examples/jsm/loaders/DRACOLoader.js";
import { GUI } from "three/examples/jsm/libs/lil-gui.module.min.js";

// Tell Three.js how to compute BVH trees
THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;

async function main() {
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 100)
    camera.position.set(0, 10, 0)
    scene.add(camera)

    const renderer = new THREE.WebGPURenderer({
        canvas: document.querySelector('.webgl'),
        antialias: true
    })
    await renderer.init()
    renderer.toneMapping = THREE.ACESFilmicToneMapping; 
    renderer.toneMappingExposure = 1.0;
    renderer.setSize(window.innerWidth, window.innerHeight)

    const controls = new OrbitControls(camera, renderer.domElement);

    // --- Post Processing ---
    const postProcessing = new THREE.PostProcessing(renderer);
    
    const scenePass = pass(scene, camera);
    const scenePassColor = scenePass.getTextureNode('output');
    
    // bloom( color, strength, radius, threshold )
    const bloomPass = bloom(scenePassColor, 0.5, 0.2, 0.9);
    
    const params = {
        enableBloom: true,
        bloomStrength: 1.5,
        bloomRadius: 0.2,
        bloomThreshold: 5.0,
    };

    // We can conditionally add bloom or just use mix
    // But since the API might vary, we can just rebuild the output node or use uniform nodes if needed.
    // For simplicity, let's mix based on a uniform, or just toggle postProcessing entirely if possible.
    // However, rebuilding outputNode is safest.
    
    function updatePostProcessing() {
        if (params.enableBloom) {
            const currentBloom = bloom(scenePassColor, params.bloomStrength, params.bloomRadius, params.bloomThreshold);
            postProcessing.outputNode = scenePassColor.add(currentBloom);
        } else {
            postProcessing.outputNode = scenePassColor;
        }
        postProcessing.needsUpdate = true;
    }

    updatePostProcessing();

    // --- GUI ---
    const gui = new GUI();
    const bloomFolder = gui.addFolder('Bloom');
    bloomFolder.add(params, 'enableBloom').onChange(updatePostProcessing);
    bloomFolder.add(params, 'bloomStrength', 0.0, 5.0, 0.01).onChange(updatePostProcessing);
    bloomFolder.add(params, 'bloomRadius', 0.0, 1.0, 0.01).onChange(updatePostProcessing);
    bloomFolder.add(params, 'bloomThreshold', 0.0, 20.0, 0.1).onChange(updatePostProcessing);

    const hdrLoader = new RGBELoader();
    hdrLoader.load("/ijewel01.hdr", (tex) => {
        tex.mapping = THREE.EquirectangularReflectionMapping;
        scene.environment = tex;
        scene.environmentIntensity = 1;
        // scene.background = tex;

        // Load GLTF model
        const dracoLoader = new DRACOLoader();
        dracoLoader.setDecoderPath('https://www.gstatic.com/draco/v1/decoders/');

        const gltfLoader = new GLTFLoader();
        gltfLoader.setDRACOLoader(dracoLoader);

        gltfLoader.load('/Diamond.glb', (gltf) => {
            gltf.scene.traverse((child) => {
                if (child.isMesh 
                    // && child.name.includes('Diam')
                ) {
                    
                    // 1. FORCE SHARP FACETS (Fixes the smooth mirror bug)
                    // We must ensure the geometry is non-indexed so the normals are perfectly flat
                    if (child.geometry.index) {
                        child.geometry = child.geometry.toNonIndexed();
                    }
                    child.geometry.computeVertexNormals();

                    // 2. REBUILD INDEX BUFFER (Required for the BVH to work after flattening)
                    if (!child.geometry.index) {
                        const indices = new Uint32Array(child.geometry.attributes.position.count);
                        for (let i = 0; i < indices.length; i++) indices[i] = i;
                        child.geometry.setIndex(new THREE.BufferAttribute(indices, 1));
                    }

                    // 3. COMPUTE BVH
                    child.geometry.computeBoundsTree();
                    
                    const original = child.material;
                    child.material = new MeshRefractionMaterial({
                        geometry: child.geometry,
                        bvh: child.geometry.boundsTree,
                        envMap: tex,
                        ior: 2.4,
                        bounces: 3,
                        aberrationStrength: 0.013,
                        fresnel: 1.0
                    });
                    if (original && original.dispose) original.dispose();
                }

            });
            scene.add(gltf.scene);
        });
    });

    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });

    const loop = () => {
        controls.update();

        if (params.enableBloom) {
            postProcessing.renderAsync();
            renderer.setClearColor(0xffffff, 1);
        } else 
            {
            renderer.renderAsync(scene, camera);
            renderer.setClearColor(0xffffff, 1);
        }
    }
    renderer.setAnimationLoop(loop);
}

main()