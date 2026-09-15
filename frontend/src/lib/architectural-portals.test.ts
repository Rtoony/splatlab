import { describe, expect, it, vi } from "vitest";
import * as THREE from "three";
import type { SparkRenderer } from "@sparkjsdev/spark";
import {
  installPortalClipping,
  installMeshPortalClipping,
  portalFragmentShader,
  validatePortals,
  type ArchitecturalPortal,
} from "./architectural-portals";

const portal = (): ArchitecturalPortal => ({
  architecture_id: "architecture_" + "a".repeat(24),
  slug: "room",
  world_to_portal: [
    [1, 0, 0, -3],
    [0, 1, 0, 2],
    [0, 0, 1, -4],
    [0, 0, 0, 1],
  ],
  lower: [-0.5, 0.015, -0.3],
  upper: [0.5, 2.1, 0.3],
  provenance: "authored-inferred",
});
const shader = "in vec3 vNdc;\nvoid main() { fragColor = vec4(1.0); }";

describe("architectural portal clipping", () => {
  it("preserves the original fragment body and adds per-fragment volume clipping", () => {
    const result = portalFragmentShader(shader);
    expect(result).toContain("fragColor = vec4(1.0);");
    expect(result).toContain("gl_FragCoord.xy - architecturalViewport.xy");
    expect(result).toContain("architecturalRay / (-architecturalRay.z * gl_FragCoord.w)");
    expect(result).not.toContain("architecturalInverseProjection * vec4(vNdc, 1.0)");
    expect(result).toContain("architecturalWorldToPortal[portalIndex] * architecturalWorld");
    expect(result).toContain("discard;");
    expect(result.match(/void main/g)).toHaveLength(1);
  });
  it("refuses an unknown shader contract rather than rendering an uncut doorway", () => {
    expect(() => portalFragmentShader("void main() {}")).toThrow("fragment contract");
    expect(() => portalFragmentShader(shader + "void main() {}")).toThrow("fragment contract");
  });
  it.each(["matrix", "bounds", "basis", "provenance", "count"])("refuses invalid %s", (defect) => {
    const values = [portal()];
    if (defect === "matrix") values[0].world_to_portal[0][0] = NaN;
    if (defect === "bounds") values[0].upper = values[0].lower;
    if (defect === "basis") values[0].world_to_portal[1][0] = 0.5;
    if (defect === "provenance") values[0].provenance = "measured" as "authored-inferred";
    if (defect === "count") values.push(...Array.from({ length: 4 }, portal));
    expect(() => validatePortals(values)).toThrow();
  });
  it("updates the real camera uniforms and restores owned shader state", () => {
    const material = new THREE.ShaderMaterial({ fragmentShader: shader, uniforms: { untouched: { value: 7 } } });
    const before = vi.fn();
    const spark = { material, onBeforeRender: before } as unknown as SparkRenderer;
    const restore = installPortalClipping(spark, [portal()]);
    const camera = new THREE.PerspectiveCamera(60, 1.5, 0.1, 100);
    camera.position.set(1, 2, 3);
    camera.updateMatrixWorld();
    const viewport = new THREE.Vector4(31, 45, 1920, 1280);
    const renderer = { getCurrentViewport: (target: THREE.Vector4) => target.copy(viewport) } as THREE.WebGLRenderer;
    spark.onBeforeRender(renderer, new THREE.Scene(), camera);
    expect(before).toHaveBeenCalledOnce();
    expect(material.uniforms.architecturalViewToWorld.value.elements).toEqual(camera.matrixWorld.elements);
    expect(material.uniforms.architecturalViewport.value.toArray()).toEqual(viewport.toArray());
    viewport.set(5, 7, 960, 640);
    spark.onBeforeRender(renderer, new THREE.Scene(), camera);
    expect(material.uniforms.architecturalViewport.value.toArray()).toEqual(viewport.toArray());
    expect(material.uniforms.architecturalWorldToPortal.value[0].elements).toEqual(
      new THREE.Matrix4().set(...(portal().world_to_portal.flat() as Parameters<THREE.Matrix4["set"]>)).elements,
    );
    restore();
    expect(material.fragmentShader).toBe(shader);
    expect(spark.onBeforeRender).toBe(before);
    expect(material.uniforms).toEqual({ untouched: { value: 7 } });
    material.dispose();
  });
  it.each([
    [0, 0, 0, 640],
    [0, 0, 960, -1],
    [NaN, 0, 960, 640],
  ])("refuses invalid viewport %j", (...values) => {
    const material = new THREE.ShaderMaterial({ fragmentShader: shader });
    const spark = { material, onBeforeRender: vi.fn() } as unknown as SparkRenderer;
    const restore = installPortalClipping(spark, [portal()]);
    const renderer = { getCurrentViewport: (target: THREE.Vector4) => target.fromArray(values) } as THREE.WebGLRenderer;
    expect(() => spark.onBeforeRender(renderer, new THREE.Scene(), new THREE.PerspectiveCamera())).toThrow("viewport");
    restore();
    material.dispose();
  });
  it("clips the installed standard-mesh shader and restores its program identity", () => {
    const material = new THREE.MeshStandardMaterial();
    const beforeCompile = material.onBeforeCompile;
    const beforeKey = material.customProgramCacheKey;
    const restore = installMeshPortalClipping(material, [portal()]);
    const shader = {
      vertexShader: THREE.ShaderLib.standard.vertexShader,
      fragmentShader: THREE.ShaderLib.standard.fragmentShader,
      uniforms: THREE.UniformsUtils.clone(THREE.ShaderLib.standard.uniforms),
    } as Parameters<typeof material.onBeforeCompile>[0];
    material.onBeforeCompile(shader, {} as THREE.WebGLRenderer);
    expect(shader.vertexShader).toContain(
      "architecturalMeshWorldPosition = (modelMatrix * vec4(transformed, 1.0)).xyz",
    );
    expect(shader.fragmentShader).toContain(
      "meshWorldToPortal[portalIndex] * vec4(architecturalMeshWorldPosition, 1.0)",
    );
    expect(shader.uniforms.meshPortalCount.value).toBe(1);
    expect(material.customProgramCacheKey()).toContain("architectural-portals-v2");
    restore();
    expect(material.onBeforeCompile).toBe(beforeCompile);
    expect(material.customProgramCacheKey).toBe(beforeKey);
    material.dispose();
  });
  it("refuses an unsupported mesh shader instead of silently leaving the visual wall intact", () => {
    const material = new THREE.MeshBasicMaterial();
    const restore = installMeshPortalClipping(material, [portal()]);
    const shader = { vertexShader: "void main() {}", fragmentShader: "void main() {}", uniforms: {} } as Parameters<
      typeof material.onBeforeCompile
    >[0];
    expect(() => material.onBeforeCompile(shader, {} as THREE.WebGLRenderer)).toThrow("paired architectural cut");
    restore();
    material.dispose();
  });
  it("pairs room-envelope uniforms in the captured Gaussian and mesh shaders", () => {
    const value: ArchitecturalPortal = {
      ...portal(),
      appearance_method: "cut-and-authored-room/v1",
      room_lower: [-1.65, 0.005, 0.225],
      room_upper: [1.65, 2.85, 3.375],
    };
    validatePortals([value]);
    const material = new THREE.ShaderMaterial({ fragmentShader: shader });
    const spark = { material, onBeforeRender: vi.fn() } as unknown as SparkRenderer;
    const restore = installPortalClipping(spark, [value]);
    expect(material.uniforms.architecturalHasRoom.value).toEqual([1, 0, 0, 0]);
    expect(material.uniforms.architecturalRoomLower.value[0].toArray()).toEqual(value.room_lower);
    expect(material.fragmentShader).toContain("architecturalRoomUpper[portalIndex]");
    restore();
    expect(material.uniforms.architecturalHasRoom).toBeUndefined();
    material.dispose();
    const mesh = new THREE.MeshBasicMaterial();
    const restoreMesh = installMeshPortalClipping(mesh, [value]);
    const compiled = {
      vertexShader: THREE.ShaderLib.basic.vertexShader,
      fragmentShader: THREE.ShaderLib.basic.fragmentShader,
      uniforms: THREE.UniformsUtils.clone(THREE.ShaderLib.basic.uniforms),
    } as Parameters<typeof mesh.onBeforeCompile>[0];
    mesh.onBeforeCompile(compiled, {} as THREE.WebGLRenderer);
    expect(compiled.uniforms.meshPortalHasRoom.value).toEqual([1, 0, 0, 0]);
    expect(compiled.uniforms.meshPortalRoomUpper.value[0].toArray()).toEqual(value.room_upper);
    restoreMesh();
    mesh.dispose();
  });
  it.each(["missing", "unversioned", "unknown", "inverted", "nonfinite"])("refuses a %s room envelope", (defect) => {
    const value: ArchitecturalPortal = {
      ...portal(),
      appearance_method: "cut-and-authored-room/v1",
      room_lower: [-1.65, 0.005, 0.225],
      room_upper: [1.65, 2.85, 3.375],
    };
    if (defect === "missing") delete value.room_lower;
    if (defect === "unversioned") delete value.appearance_method;
    if (defect === "unknown") value.appearance_method = "unknown" as ArchitecturalPortal["appearance_method"];
    if (defect === "inverted") value.room_upper = value.room_lower;
    if (defect === "nonfinite") value.room_upper![0] = Infinity;
    expect(() => validatePortals([value])).toThrow();
  });
});
