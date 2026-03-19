import './style/main.css'
import * as THREE from 'three/webgpu'
import { pass } from 'three/tsl';
import { bloom } from 'three/examples/jsm/tsl/display/BloomNode.js';
import { computeBoundsTree, disposeBoundsTree } from 'three-mesh-bvh'; 
import { mergeVertices } from 'three/examples/jsm/utils/BufferGeometryUtils.js'; // The Airtight Welder

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
    const camera = new THREE.PerspectiveCamera(25, window.innerWidth / window.innerHeight, 0.1, 100)
    camera.position.set(0, 10, 0)
    scene.add(camera)

    const params = {
        enableBloom: true,
        bloomStrength: 1.5,
        bloomRadius: 0.2,
        bloomThreshold: 5.0,
        pixelRatio: window.devicePixelRatio,
    };

    const renderer = new THREE.WebGPURenderer({
        canvas: document.querySelector('.webgl'),
        antialias: true
    })
    await renderer.init()
    renderer.toneMapping = THREE.ACESFilmicToneMapping; 
    renderer.toneMappingExposure = 1.0;
    renderer.setSize(window.innerWidth, window.innerHeight)
    renderer.setPixelRatio(params.pixelRatio)

    const controls = new OrbitControls(camera, renderer.domElement);

    // --- Post Processing ---
    const postProcessing = new THREE.PostProcessing(renderer);
    const scenePass = pass(scene, camera);
    const scenePassColor = scenePass.getTextureNode('output');
    
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

    gui.add(params, 'pixelRatio', 0.5, 4.0, 0.1).name('Resolution / DPI').onChange(() => {
        renderer.setPixelRatio(params.pixelRatio);
    });

    const hdrLoader = new RGBELoader();
    hdrLoader.load("/ijewel01.hdr", (tex) => {
        tex.mapping = THREE.EquirectangularReflectionMapping;
        scene.environment = tex;
        // scene.background = tex;
        scene.environmentIntensity = 1;

        const dracoLoader = new DRACOLoader();
        dracoLoader.setDecoderPath('https://www.gstatic.com/draco/v1/decoders/');

        const gltfLoader = new GLTFLoader();
        gltfLoader.setDRACOLoader(dracoLoader);

       gltfLoader.load('/15.glb', (gltf) => {
            gltf.scene.traverse((child) => {
                if (child.isMesh && child.name.includes('Diam')) {
                    
                    // --- DPI RESOLUTION FIX: CPU-Baked Flat Normals ---
                    // 1. Convert to non-indexed to bake perfectly sharp flat facets
                    let geom = child.geometry.clone();
                    if (geom.index) {
                        geom = geom.toNonIndexed();
                    }
                    geom.computeVertexNormals();

                    // 2. Create a sequential index buffer (Required for BVH to read the arrays)
                    const indices = new Uint32Array(geom.attributes.position.count);
                    for (let i = 0; i < indices.length; i++) indices[i] = i;
                    geom.setIndex(new THREE.BufferAttribute(indices, 1));

                    // 3. Compute the seamless BVH
                    geom.computeBoundsTree();
                    
                    const original = child.material;
                    child.material = new MeshRefractionMaterial({
                        geometry: geom, // Pass the new geometry
                        bvh: geom.boundsTree,
                        envMap: tex,
                        ior: 2.4,
                        bounces: 3,
                        aberrationStrength: 0.015,
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
        renderer.setPixelRatio(params.pixelRatio);
    });

    const loop = () => {
        controls.update();

        if (params.enableBloom) {
            postProcessing.renderAsync();
            renderer.setClearColor(0xffffff, 1);
        } else {
            renderer.renderAsync(scene, camera);
            renderer.setClearColor(0xffffff, 1);
        }
    }
    renderer.setAnimationLoop(loop);
}

main()