import * as THREE from 'three/webgpu';
import { storage, wgslFn, positionWorld, cameraPosition, normalWorld, Fn, texture, equirectUV, normalize, sub, add, vec3, mix, dot, pow, modelWorldMatrix, modelWorldMatrixInverse } from 'three/tsl';
import { bvhIntersectFirstHit } from './bvh_ray_functions.wgsl.js'; 
import { bvhNodeStruct } from './common_functions.wgsl.js'; 

// --- CRITICAL FIX: Memory Alignment Padder ---
// WebGPU requires vec3 arrays in storage buffers to be 16-byte aligned (like a vec4).
function padArrayVec3ToVec4(array, Type) {
    const numItems = array.length / 3;
    const padded = new Type(numItems * 4);
    for (let i = 0; i < numItems; i++) {
        padded[i * 4 + 0] = array[i * 3 + 0];
        padded[i * 4 + 1] = array[i * 3 + 1];
        padded[i * 4 + 2] = array[i * 3 + 2];
        padded[i * 4 + 3] = 0; // The 16-byte padding!
    }
    return padded;
}

export default class MeshRefractionMaterial extends THREE.NodeMaterial {
    constructor({ geometry, bvh, envMap, ior = 2.4, bounces = 3, aberrationStrength = 0.02, fresnel = 1.0 }) {
        super();
        this.transparent = true;

        // --- THE BRIDGE (CPU to GPU) ---
        // Pad the raw geometry arrays to satisfy WebGPU alignment rules
        const paddedPositions = padArrayVec3ToVec4(geometry.attributes.position.array, Float32Array);
        const paddedIndices = padArrayVec3ToVec4(geometry.index.array, Uint32Array);
        const bvhPackedArray = new Uint32Array(bvh._roots[0]); 

        // Notice itemSize is now 4 (for memory length), but WGSL type remains 'vec3'/'uvec3'
        const bvhPositionNode = storage(new THREE.StorageBufferAttribute(paddedPositions, 4), 'vec3', paddedPositions.length / 4).toReadOnly();
        const bvhIndexNode = storage(new THREE.StorageBufferAttribute(paddedIndices, 4), 'uvec3', paddedIndices.length / 4).toReadOnly();
        const bvhTreeDataNode = storage(new THREE.StorageBufferAttribute(bvhPackedArray, 8), 'BVHNode', bvhPackedArray.length / 8).toReadOnly();

        // --- CPU OPTIMIZATION: Dispersion IORs ---
        const iorR = Math.max(1.0, ior * (1.0 - aberrationStrength));
        const iorG = Math.max(1.0, ior);
        const iorB = Math.max(1.0, ior * (1.0 + aberrationStrength));

        // --- THE GPU LOGIC (WGSL) ---
        const calculateInternalBounces = wgslFn(`
            fn calculateInternalBounces(
                vWorldPos: vec3<f32>,
                rd: vec3<f32>, 
                normal: vec3<f32>, 
                ior: f32,
                bvh_index: ptr<storage, array<vec3u>, read>,
                bvh_position: ptr<storage, array<vec3f>, read>,
                bvh: ptr<storage, array<BVHNode>, read>,
                modelMatrixInverse: mat4x4<f32>,
                modelMatrix: mat4x4<f32>
            ) -> vec3<f32> {
                
                // 1. Enter the diamond
                var rayDirection = refract(rd, normal, 1.0 / ior);
                var rayOrigin = vWorldPos + rayDirection * 0.0001; 
                
                // 2. Transform to LOCAL SPACE so the ray perfectly matches the BVH geometry
                rayOrigin = (modelMatrixInverse * vec4<f32>(rayOrigin, 1.0)).xyz;
                rayDirection = normalize((modelMatrixInverse * vec4<f32>(rayDirection, 0.0)).xyz);
                
                let numBounces: i32 = ${parseInt(bounces)};
                
                // 3. BVH Bounce Loop
                for(var i: i32 = 0; i < numBounces; i = i + 1) {
                    var ray: Ray;
                    ray.origin = rayOrigin;
                    ray.direction = rayDirection;
                    
                    let hitResult = bvhIntersectFirstHit(bvh_index, bvh_position, bvh, ray);
                    
                    if (hitResult.didHit) {
                        let hitPos = rayOrigin + rayDirection * max(hitResult.dist - 0.001, 0.0);
                        let tempDir = refract(rayDirection, hitResult.normal, ior); 
                        
                        if (length(tempDir) != 0.0) {
                            rayDirection = tempDir; // Ray escapes!
                            break; 
                        }
                        
                        // Internal reflection
                        rayDirection = reflect(rayDirection, hitResult.normal);
                        rayOrigin = hitPos + rayDirection * 0.001;
                    } else {
                        break; // Nothing hit
                    }
                }
                
                // 4. Transform back to World Space to sample the HDRI
                rayDirection = normalize((modelMatrix * vec4<f32>(rayDirection, 0.0)).xyz);
                return rayDirection;
            }
        `, [ bvhNodeStruct, bvhIntersectFirstHit ]); 

        // --- THE MATERIAL OUTPUT ---
        this.colorNode = Fn(() => {
            const viewDirection = normalize(sub(positionWorld, cameraPosition));
            const normal = normalWorld;
            
            const baseArgs = {
                vWorldPos: positionWorld,
                rd: viewDirection, 
                normal: normal, 
                bvh_index: bvhIndexNode, 
                bvh_position: bvhPositionNode, 
                bvh: bvhTreeDataNode,
                modelMatrixInverse: modelWorldMatrixInverse,
                modelMatrix: modelWorldMatrix
            };

            const exitDirectionR = calculateInternalBounces({ ...baseArgs, ior: iorR });
            const exitDirectionG = calculateInternalBounces({ ...baseArgs, ior: iorG });
            const exitDirectionB = calculateInternalBounces({ ...baseArgs, ior: iorB });
            
            const colorR = texture(envMap, equirectUV(exitDirectionR)).r;
            const colorG = texture(envMap, equirectUV(exitDirectionG)).g;
            const colorB = texture(envMap, equirectUV(exitDirectionB)).b;
            
            const refractedColor = vec3(colorR, colorG, colorB);
            
            const dotProduct = dot(viewDirection, normal); 
            const nFresnel = pow(add(1.0, dotProduct), 10.0).mul(fresnel);
            
            return mix(refractedColor, vec3(1.0), nFresnel);
        })();
    }
}