import * as THREE from 'three/webgpu';
import { storage, wgslFn, positionWorld, cameraPosition, normalWorld, Fn, texture, equirectUV, normalize, sub, add, vec3, mix, dot, pow, modelWorldMatrix, modelWorldMatrixInverse } from 'three/tsl';
import { bvhIntersectFirstHit } from './bvh_ray_functions.wgsl.js'; 
import { bvhNodeStruct } from './common_functions.wgsl.js'; 

// --- CRITICAL FIX: Memory Alignment Padder ---
function padArrayVec3ToVec4(array, Type) {
    const numItems = array.length / 3;
    const padded = new Type(numItems * 4);
    for (let i = 0; i < numItems; i++) {
        padded[i * 4 + 0] = array[i * 3 + 0];
        padded[i * 4 + 1] = array[i * 3 + 1];
        padded[i * 4 + 2] = array[i * 3 + 2];
        padded[i * 4 + 3] = 0; 
    }
    return padded;
}

export default class MeshRefractionMaterial extends THREE.NodeMaterial {
    constructor({ geometry, bvh, envMap, ior = 2.4, bounces = 3, aberrationStrength = 0.013, fresnel = 1.0 }) {
        super();
        
        // --- FIX: Disable Transparency ---
        // Prevents MSAA from blending the triangle edges with the background
        this.transparent = false; 
        this.depthWrite = true;

        // --- THE BRIDGE (CPU to GPU) ---
        const paddedPositions = padArrayVec3ToVec4(geometry.attributes.position.array, Float32Array);
        const paddedIndices = padArrayVec3ToVec4(geometry.index.array, Uint32Array);
        const bvhPackedArray = new Uint32Array(bvh._roots[0]); 

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
                
                var worldRefractDir = refract(rd, normal, 1.0 / ior);
                
                var rayOrigin = (modelMatrixInverse * vec4<f32>(vWorldPos, 1.0)).xyz;
                var rayDirection = normalize((modelMatrixInverse * vec4<f32>(worldRefractDir, 0.0)).xyz);
                
                // Initial Entry Offset
                rayOrigin = rayOrigin + rayDirection * 0.001; 
                
                let numBounces: i32 = ${parseInt(bounces)};
                
                for(var i: i32 = 0; i < numBounces; i = i + 1) {
                    var ray: Ray;
                    ray.origin = rayOrigin;
                    ray.direction = rayDirection;
                    
                    let hitResult = bvhIntersectFirstHit(bvh_index, bvh_position, bvh, ray);
                    
                    if (hitResult.didHit) {
                        // --- PHYSICS FIX: Double Offset ---
                        let hitPos = rayOrigin + rayDirection * max(hitResult.dist - 0.0005, 0.0);
                        let tempDir = refract(rayDirection, hitResult.normal, ior); 
                        
                        if (length(tempDir) != 0.0) {
                            rayDirection = tempDir; 
                            break; 
                        }
                        
                        rayDirection = reflect(rayDirection, hitResult.normal);
                        rayOrigin = hitPos + rayDirection * 0.005;
                    } else {
                        break; 
                    }
                }
                
                rayDirection = normalize((modelMatrix * vec4<f32>(rayDirection, 0.0)).xyz);
                return rayDirection;
            }
        `, [ bvhNodeStruct, bvhIntersectFirstHit ]); 

        // --- THE MATERIAL OUTPUT ---
        // --- THE MATERIAL OUTPUT ---
        this.colorNode = Fn(() => {
            const viewDirection = normalize(sub(positionWorld, cameraPosition));
            
            // --- DPI FIX: Use the baked CPU Normal ---
            // No more dFdx! The geometry now provides a perfectly sharp normal natively.
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
            
            const colorR = texture(envMap, equirectUV(exitDirectionR)).level(0).r;
            const colorG = texture(envMap, equirectUV(exitDirectionG)).level(0).g;
            const colorB = texture(envMap, equirectUV(exitDirectionB)).level(0).b;
            
            const refractedColor = vec3(colorR, colorG, colorB);
            
            const dotProduct = dot(viewDirection, normal); 
            const nFresnel = pow(add(1.0, dotProduct), 10.0).mul(fresnel);
            
            return mix(refractedColor, vec3(1.0), nFresnel);
        })();
    }
}