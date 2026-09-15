import * as THREE from "three";
import type { SparkRenderer } from "@sparkjsdev/spark";

export type ArchitecturalPortal = {
  architecture_id: string;
  slug: string;
  world_to_portal: number[][];
  lower: number[];
  upper: number[];
  appearance_method?: "cut-only/v1" | "cut-and-authored-room/v1";
  room_lower?: number[];
  room_upper?: number[];
  provenance: "authored-inferred";
};

export function validatePortals(portals: ArchitecturalPortal[]): void {
  if (!Array.isArray(portals) || portals.length > 4)
    throw new Error("At most four architectural portal volumes are supported.");
  for (const portal of portals) {
    if (
      portal.provenance !== "authored-inferred" ||
      !/^architecture_[a-f0-9]{24}$/.test(portal.architecture_id) ||
      !Array.isArray(portal.world_to_portal) ||
      portal.world_to_portal.length !== 4 ||
      portal.world_to_portal.some(
        (row) =>
          !Array.isArray(row) ||
          row.length !== 4 ||
          row.some((value) => typeof value !== "number" || !Number.isFinite(value)),
      ) ||
      ![portal.lower, portal.upper].every(
        (vector) =>
          Array.isArray(vector) &&
          vector.length === 3 &&
          vector.every((value) => typeof value === "number" && Number.isFinite(value)),
      ) ||
      portal.upper.some((value, axis) => value <= portal.lower[axis]) ||
      portal.world_to_portal[3].some((value, column) => value !== (column === 3 ? 1 : 0))
    ) {
      throw new Error("Architectural clipping requires a finite authored world-metre frame.");
    }
    if (portal.appearance_method === "cut-and-authored-room/v1") {
      if (
        ![portal.room_lower, portal.room_upper].every(
          (bounds) => Array.isArray(bounds) && bounds.length === 3 && bounds.every(Number.isFinite),
        ) ||
        portal.room_upper!.some((value, axis) => value <= portal.room_lower![axis])
      )
        throw new Error("Authored-room clipping requires its complete bounded appearance envelope.");
    } else if (
      (portal.appearance_method !== undefined && portal.appearance_method !== "cut-only/v1") ||
      portal.room_lower !== undefined ||
      portal.room_upper !== undefined
    ) {
      throw new Error("Unknown or inconsistent architectural appearance method.");
    }
    const axes = portal.world_to_portal
      .slice(0, 3)
      .map((row) => new THREE.Vector3(...(row.slice(0, 3) as [number, number, number])));
    if (
      axes.some(
        (axis, index) =>
          Math.abs(axis.lengthSq() - 1) > 1e-6 ||
          axes.some((other, otherIndex) => index !== otherIndex && Math.abs(axis.dot(other)) > 1e-6),
      )
    )
      throw new Error("Architectural clipping requires an orthonormal frame.");
  }
}

export function portalFragmentShader(original: string): string {
  const entry = /void\s+main\s*\(\s*\)\s*\{/g;
  if ([...original.matchAll(entry)].length !== 1 || !original.includes("vNdc"))
    throw new Error("Installed Spark fragment contract does not support architectural clipping.");
  const declarations = `
uniform int architecturalPortalCount;
uniform mat4 architecturalWorldToPortal[4];
uniform vec3 architecturalLower[4];
uniform vec3 architecturalUpper[4];
uniform int architecturalHasRoom[4];
uniform vec3 architecturalRoomLower[4];
uniform vec3 architecturalRoomUpper[4];
uniform mat4 architecturalInverseProjection;
uniform mat4 architecturalViewToWorld;
uniform vec4 architecturalViewport;
`;
  const clip = `
  if (architecturalPortalCount > 0) {
    vec2 architecturalNdc = 2.0 * (gl_FragCoord.xy - architecturalViewport.xy) / architecturalViewport.zw - 1.0;
    vec3 architecturalRay = (architecturalInverseProjection * vec4(architecturalNdc, 0.0, 1.0)).xyz;
    vec3 architecturalView = architecturalRay / (-architecturalRay.z * gl_FragCoord.w);
    vec4 architecturalWorld = architecturalViewToWorld * vec4(architecturalView, 1.0);
    for (int portalIndex = 0; portalIndex < 4; portalIndex++) {
      if (portalIndex >= architecturalPortalCount) break;
      vec3 architecturalLocal = (architecturalWorldToPortal[portalIndex] * architecturalWorld).xyz;
      if (all(greaterThanEqual(architecturalLocal, architecturalLower[portalIndex]))
        && all(lessThanEqual(architecturalLocal, architecturalUpper[portalIndex]))) discard;
      if (architecturalHasRoom[portalIndex] == 1
        && all(greaterThanEqual(architecturalLocal, architecturalRoomLower[portalIndex]))
        && all(lessThanEqual(architecturalLocal, architecturalRoomUpper[portalIndex]))) discard;
    }
  }
`;
  return original.replace(entry, declarations + "\nvoid main() {\n" + clip);
}

export function installPortalClipping(spark: SparkRenderer, portals: ArchitecturalPortal[]): () => void {
  validatePortals(portals);
  const material = spark.material;
  const previousShader = material.fragmentShader;
  const previousRender = spark.onBeforeRender;
  const matrices = Array.from({ length: 4 }, (_, index) =>
    portals[index]
      ? new THREE.Matrix4().set(...(portals[index].world_to_portal.flat() as Parameters<THREE.Matrix4["set"]>))
      : new THREE.Matrix4(),
  );
  const lower = Array.from({ length: 4 }, (_, index) =>
    new THREE.Vector3().fromArray(portals[index]?.lower || [0, 0, 0]),
  );
  const upper = Array.from({ length: 4 }, (_, index) =>
    new THREE.Vector3().fromArray(portals[index]?.upper || [0, 0, 0]),
  );
  const inverseProjection = new THREE.Matrix4();
  const viewToWorld = new THREE.Matrix4();
  const viewport = new THREE.Vector4();
  const added = {
    architecturalPortalCount: { value: portals.length },
    architecturalWorldToPortal: { value: matrices },
    architecturalLower: { value: lower },
    architecturalUpper: { value: upper },
    architecturalHasRoom: {
      value: Array.from({ length: 4 }, (_, index) =>
        Number(portals[index]?.appearance_method === "cut-and-authored-room/v1"),
      ),
    },
    architecturalRoomLower: {
      value: Array.from({ length: 4 }, (_, index) =>
        new THREE.Vector3().fromArray(portals[index]?.room_lower || [0, 0, 0]),
      ),
    },
    architecturalRoomUpper: {
      value: Array.from({ length: 4 }, (_, index) =>
        new THREE.Vector3().fromArray(portals[index]?.room_upper || [0, 0, 0]),
      ),
    },
    architecturalInverseProjection: { value: inverseProjection },
    architecturalViewToWorld: { value: viewToWorld },
    architecturalViewport: { value: viewport },
  };
  material.fragmentShader = portalFragmentShader(previousShader);
  Object.assign(material.uniforms, added);
  material.needsUpdate = true;
  spark.onBeforeRender = function (renderer, scene, camera) {
    previousRender.call(this, renderer, scene, camera);
    const perspective = camera as THREE.PerspectiveCamera;
    if (!perspective.isPerspectiveCamera)
      throw new Error("Architectural clipping requires the calibrated perspective viewer.");
    renderer.getCurrentViewport(viewport);
    if (!viewport.toArray().every(Number.isFinite) || viewport.z <= 0 || viewport.w <= 0)
      throw new Error("Architectural clipping requires the current finite drawing-buffer viewport.");
    inverseProjection.copy(perspective.projectionMatrixInverse);
    viewToWorld.copy(camera.matrixWorld);
  };
  return () => {
    spark.onBeforeRender = previousRender;
    material.fragmentShader = previousShader;
    for (const key of Object.keys(added)) delete material.uniforms[key];
    material.needsUpdate = true;
  };
}

export function installMeshPortalClipping(material: THREE.Material, portals: ArchitecturalPortal[]): () => void {
  validatePortals(portals);
  const originalCompile = material.onBeforeCompile;
  const originalKey = material.customProgramCacheKey;
  const uniforms = {
    meshPortalCount: { value: portals.length },
    meshPortalHasRoom: {
      value: Array.from({ length: 4 }, (_, index) =>
        Number(portals[index]?.appearance_method === "cut-and-authored-room/v1"),
      ),
    },
    meshPortalRoomLower: {
      value: Array.from({ length: 4 }, (_, index) =>
        new THREE.Vector3().fromArray(portals[index]?.room_lower || [0, 0, 0]),
      ),
    },
    meshPortalRoomUpper: {
      value: Array.from({ length: 4 }, (_, index) =>
        new THREE.Vector3().fromArray(portals[index]?.room_upper || [0, 0, 0]),
      ),
    },
    meshWorldToPortal: {
      value: Array.from({ length: 4 }, (_, index) =>
        portals[index]
          ? new THREE.Matrix4().set(...(portals[index].world_to_portal.flat() as Parameters<THREE.Matrix4["set"]>))
          : new THREE.Matrix4(),
      ),
    },
    meshPortalLower: {
      value: Array.from({ length: 4 }, (_, index) => new THREE.Vector3().fromArray(portals[index]?.lower || [0, 0, 0])),
    },
    meshPortalUpper: {
      value: Array.from({ length: 4 }, (_, index) => new THREE.Vector3().fromArray(portals[index]?.upper || [0, 0, 0])),
    },
  };
  material.onBeforeCompile = function (shader, renderer) {
    originalCompile.call(this, shader, renderer);
    if (!shader.vertexShader.includes("#include <project_vertex>") || !shader.fragmentShader.includes("void main() {"))
      throw new Error("Captured mesh material does not support the paired architectural cut.");
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader =
      "varying vec3 architecturalMeshWorldPosition;\n" +
      shader.vertexShader.replace(
        "#include <project_vertex>",
        "#include <project_vertex>\narchitecturalMeshWorldPosition = (modelMatrix * vec4(transformed, 1.0)).xyz;",
      );
    shader.fragmentShader =
      `
varying vec3 architecturalMeshWorldPosition;
uniform int meshPortalCount;
uniform mat4 meshWorldToPortal[4];
uniform vec3 meshPortalLower[4];
uniform vec3 meshPortalUpper[4];
uniform int meshPortalHasRoom[4];
uniform vec3 meshPortalRoomLower[4];
uniform vec3 meshPortalRoomUpper[4];
` +
      shader.fragmentShader.replace(
        "void main() {",
        `void main() {
  for (int portalIndex = 0; portalIndex < 4; portalIndex++) {
    if (portalIndex >= meshPortalCount) break;
    vec3 architecturalLocal = (meshWorldToPortal[portalIndex] * vec4(architecturalMeshWorldPosition, 1.0)).xyz;
    if (all(greaterThanEqual(architecturalLocal, meshPortalLower[portalIndex]))
      && all(lessThanEqual(architecturalLocal, meshPortalUpper[portalIndex]))) discard;
    if (meshPortalHasRoom[portalIndex] == 1
      && all(greaterThanEqual(architecturalLocal, meshPortalRoomLower[portalIndex]))
      && all(lessThanEqual(architecturalLocal, meshPortalRoomUpper[portalIndex]))) discard;
  }
`,
      );
  };
  material.customProgramCacheKey = () => originalKey.call(material) + "|architectural-portals-v2";
  material.needsUpdate = true;
  return () => {
    material.onBeforeCompile = originalCompile;
    material.customProgramCacheKey = originalKey;
    material.needsUpdate = true;
  };
}
