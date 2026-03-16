import {
    DepthFormat, DepthTexture, LinearFilter, Matrix4,
    PerspectiveCamera, Plane, UnsignedShortType, Vector3, Vector4, RenderTarget
} from "three"
import { MeshStandardNodeMaterial } from 'three/webgpu'
import {
    texture, vec2, vec4, float, mix, smoothstep, max, min,
    uniform, positionLocal, varying, varyingProperty
} from 'three/tsl'

export default class MeshReflectorMaterial extends MeshStandardNodeMaterial {
    constructor(renderer, camera, scene, object, {
        mixBlur = 0,
        mixStrength = 1,
        resolution = 256,
        blur = [0, 0],
        minDepthThreshold = 0.9,
        maxDepthThreshold = 1,
        depthScale = 0,
        depthToBlurRatioBias = 0.25,
        mirror = 0,
        distortion = 1,
        mixContrast = 1,
        distortionMap,
        reflectorOffset = 0,
        bufferSamples = 8,
        planeNormal = new Vector3(0, 0, 1)
    } = {}) {
        super();

        this.gl = renderer
        this.mainCamera = camera
        this.scene = scene
        this.parent = object

        this.hasBlur = blur[0] + blur[1] > 0
        this.reflectorPlane = new Plane()
        this.normal = new Vector3()
        this.reflectorWorldPosition = new Vector3()
        this.cameraWorldPosition = new Vector3()
        this.rotationMatrix = new Matrix4()
        this.lookAtPosition = new Vector3(0, -1, 0)
        this.clipPlane = new Vector4()
        this.view = new Vector3()
        this.target = new Vector3()
        this.q = new Vector4()
        this.textureMatrix = new Matrix4()
        this.virtualCamera = new PerspectiveCamera()
        this.reflectorOffset = reflectorOffset;
        this.planeNormal = planeNormal

        this.setupBuffers(resolution, blur, bufferSamples);

        // Uniforms
        this.textureMatrixUniform = uniform(this.textureMatrix);
        this.mirrorUniform = uniform(mirror);
        this.mixBlurUniform = uniform(mixBlur);
        this.mixStrengthUniform = uniform(mixStrength);
        this.minDepthThresholdUniform = uniform(minDepthThreshold);
        this.maxDepthThresholdUniform = uniform(maxDepthThreshold);
        this.depthScaleUniform = uniform(depthScale);
        this.depthToBlurRatioBiasUniform = uniform(depthToBlurRatioBias);
        this.distortionUniform = uniform(distortion);
        this.mixContrastUniform = uniform(mixContrast);

        // Texture nodes — bound to the render targets
        this.tDiffuseNode = texture(this.fbo1.texture);
        this.tDiffuseBlurNode = texture(this.fbo2.texture);
        this.tDepthNode = texture(this.fbo1.depthTexture);

        this.setupNodes(distortionMap);
    }

    setupNodes(distortionMap) {
        // --- Vertex stage: compute projected UV and pass as varying ---
        // textureMatrix * vec4(position, 1.0) gives clip-space coords for the reflection
        const vReflectCoord = varyingProperty('vec4', 'vReflectCoord');

        // Override vertex position node to also compute vReflectCoord
        // We attach a vertex node that sets the varying
        this.vertexNode = (() => {
            // positionLocal is the vertex position in local space
            const projected = this.textureMatrixUniform.mul(vec4(positionLocal, 1.0));
            return vReflectCoord.assign(projected);
        })();

        // --- Fragment stage ---
        // Perspective-divide to get the final UV
        const coord = vReflectCoord;
        let refUv = coord.xy.div(coord.w);

        // Optional distortion
        if (distortionMap) {
            const distortionTexNode = texture(distortionMap);
            const distortionFactor = distortionTexNode.r.mul(this.distortionUniform);
            refUv = refUv.add(distortionFactor);
        }

        // Sample reflection textures using the projected UV
        let base = this.tDiffuseNode.uv(refUv);
        let blurSample = this.tDiffuseBlurNode.uv(refUv);
        let merge = base;

        // Depth-based fading
        if (this.depthScaleUniform.value > 0) {
            const depthSample = this.tDepthNode.uv(refUv);
            let depthFactor = smoothstep(
                this.minDepthThresholdUniform,
                this.maxDepthThresholdUniform,
                float(1.0).sub(depthSample.r.mul(depthSample.a))
            );
            depthFactor = depthFactor.mul(this.depthScaleUniform);
            depthFactor = max(float(0.0001), min(float(1.0), depthFactor));

            if (this.hasBlur) {
                blurSample = blurSample.mul(min(float(1.0), depthFactor.add(this.depthToBlurRatioBiasUniform)));
                merge = merge.mul(min(float(1.0), depthFactor.add(float(0.5))));
            } else {
                merge = merge.mul(depthFactor);
            }
        }

        // Roughness-based blur
        if (this.hasBlur) {
            const roughnessFactor = this.roughnessNode !== null ? this.roughnessNode : uniform(this.roughness);
            const blurFactor = min(float(1.0), this.mixBlurUniform.mul(roughnessFactor));
            merge = mix(merge, blurSample, blurFactor);
        }

        // Contrast adjustment
        const r = merge.r.sub(0.5).mul(this.mixContrastUniform).add(0.5);
        const g = merge.g.sub(0.5).mul(this.mixContrastUniform).add(0.5);
        const b = merge.b.sub(0.5).mul(this.mixContrastUniform).add(0.5);
        const newMerge = vec4(r, g, b, float(1.0));

        // Final color:
        // Original GLSL: diffuseColor.rgb = diffuseColor.rgb * ((1 - mirror) + newMerge.rgb * mixStrength)
        // mirror=0 → diffuseColor unchanged (full base color + reflection additive)
        // mirror=1 → diffuseColor * reflection only (pure reflective)
        const mirrorFactor = min(float(1.0), this.mirrorUniform);
        const baseContrib = float(1.0).sub(mirrorFactor);
        const reflectionContrib = newMerge.xyz.mul(this.mixStrengthUniform);

        // We override the diffuseColor-equivalent via colorNode
        // colorNode replaces the base diffuse modulation
        this.colorNode = this.color.mul(baseContrib.add(reflectionContrib));
    }

    setupBuffers(resolution, blur, bufferSamples) {
        const parameters = {
            minFilter: LinearFilter,
            magFilter: LinearFilter,
        }

        const fbo1 = new RenderTarget(resolution, resolution, parameters)
        fbo1.depthBuffer = true
        fbo1.depthTexture = new DepthTexture(resolution, resolution)
        fbo1.depthTexture.format = DepthFormat
        fbo1.depthTexture.type = UnsignedShortType

        const fbo2 = new RenderTarget(resolution, resolution, parameters)
        fbo1.samples = bufferSamples

        this.fbo1 = fbo1;
        this.fbo2 = fbo2;
    }

    beforeRender() {
        if (!this.parent) return

        this.reflectorWorldPosition.setFromMatrixPosition(this.parent.matrixWorld)
        this.cameraWorldPosition.setFromMatrixPosition(this.mainCamera.matrixWorld)
        this.rotationMatrix.extractRotation(this.parent.matrixWorld)

        this.normal.copy(this.planeNormal)
        this.normal.applyMatrix4(this.rotationMatrix)
        this.reflectorWorldPosition.addScaledVector(this.normal, this.reflectorOffset)
        this.view.subVectors(this.reflectorWorldPosition, this.cameraWorldPosition)

        if (this.view.dot(this.normal) > 0) return
        this.view.reflect(this.normal).negate()
        this.view.add(this.reflectorWorldPosition)
        this.rotationMatrix.extractRotation(this.mainCamera.matrixWorld)
        this.lookAtPosition.set(0, 0, -1)
        this.lookAtPosition.applyMatrix4(this.rotationMatrix)
        this.lookAtPosition.add(this.cameraWorldPosition)
        this.target.subVectors(this.reflectorWorldPosition, this.lookAtPosition)
        this.target.reflect(this.normal).negate()
        this.target.add(this.reflectorWorldPosition)
        this.virtualCamera.position.copy(this.view)
        this.virtualCamera.up.set(0, 1, 0)
        this.virtualCamera.up.applyMatrix4(this.rotationMatrix)
        this.virtualCamera.up.reflect(this.normal)
        this.virtualCamera.lookAt(this.target)
        this.virtualCamera.far = this.mainCamera.far
        this.virtualCamera.updateMatrixWorld()
        this.virtualCamera.projectionMatrix.copy(this.mainCamera.projectionMatrix)

        // Update texture matrix (maps world position to reflection UV)
        this.textureMatrix.set(0.5, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0)
        this.textureMatrix.multiply(this.virtualCamera.projectionMatrix)
        this.textureMatrix.multiply(this.virtualCamera.matrixWorldInverse)
        this.textureMatrix.multiply(this.parent.matrixWorld)

        // Oblique clipping plane (Lengyel technique)
        this.reflectorPlane.setFromNormalAndCoplanarPoint(this.normal, this.reflectorWorldPosition)
        this.reflectorPlane.applyMatrix4(this.virtualCamera.matrixWorldInverse)
        this.clipPlane.set(
            this.reflectorPlane.normal.x, this.reflectorPlane.normal.y,
            this.reflectorPlane.normal.z, this.reflectorPlane.constant
        )
        const projectionMatrix = this.virtualCamera.projectionMatrix
        this.q.x = (Math.sign(this.clipPlane.x) + projectionMatrix.elements[8]) / projectionMatrix.elements[0]
        this.q.y = (Math.sign(this.clipPlane.y) + projectionMatrix.elements[9]) / projectionMatrix.elements[5]
        this.q.z = -1.0
        this.q.w = (1.0 + projectionMatrix.elements[10]) / projectionMatrix.elements[14]
        this.clipPlane.multiplyScalar(2.0 / this.clipPlane.dot(this.q))

        projectionMatrix.elements[2] = this.clipPlane.x
        projectionMatrix.elements[6] = this.clipPlane.y
        projectionMatrix.elements[10] = this.clipPlane.z + 1.0
        projectionMatrix.elements[14] = this.clipPlane.w
    }

    update() {
        if (!this.parent || this.parent.material !== this) return;

        this.parent.visible = false
        this.beforeRender()

        this.gl.setRenderTarget(this.fbo1)
        if (!this.gl.autoClear) this.gl.clear()
        this.gl.render(this.scene, this.virtualCamera)

        this.parent.visible = true
        this.gl.setRenderTarget(null)
    }
}