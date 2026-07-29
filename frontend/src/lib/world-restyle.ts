// Restyle (W2-C2): apply a look over a captured world without touching it.
//
// Three verbs, matching backend/world_restyle.py exactly:
//   tint      multiply a colour into an element's baked albedo
//   material  replace the surface with a taxonomy CLASS tile
//   lighting  one scene-wide preset + intensity
//
// The class tile is generated HERE from the same `generative` numbers the
// backend bake reads out of class_taxonomy.json (base_rgb + noise), so the
// walker preview and a later permanent re-bake describe the same surface.
//
// It is projected TRIPLANARLY, in world space, on purpose: every element's
// UVs are its own atlas chart packing, so a tile pushed through them would
// scramble into confetti. World-space projection also means one tile scale
// reads consistently across elements — the thing you want when you are
// re-surfacing a room rather than texturing a model.

import * as THREE from "three";

export interface RestyleMaterial {
  id: string;
  display: string;
  category: string;
  color: string;
  base_rgb: [number, number, number] | number[];
  roughness: number;
  metallic: number;
  noise: { scale?: number; luma_jitter?: number; speckle?: number };
}

export interface RestyleEntry {
  tint?: string;
  material?: string;
  material_scale?: number;
}

export interface RestyleDoc {
  v: number;
  job_id: string;
  elements: Record<string, RestyleEntry>;
  lighting: { preset: string; intensity: number };
}

export interface LightPreset {
  /** Hemisphere sky/ground and the key light — the whole rig in four values. */
  sky: number;
  ground: number;
  key: number;
  keyIntensity: number;
  hemiIntensity: number;
  ambient: number;
  /** Renderer clear colour: the sky you see past an open shell. */
  background: number;
  /** as-captured means "do not light it at all" — the atlas already is lit. */
  asCaptured?: boolean;
}

export const LIGHT_PRESETS: Record<string, LightPreset> = {
  "as-captured": {
    sky: 0xdfe8ff, ground: 0x2a2620, key: 0xffffff,
    keyIntensity: 1.4, hemiIntensity: 2.0, ambient: 0.35,
    background: 0x0b0b0e, asCaptured: true,
  },
  noon: {
    sky: 0xeaf2ff, ground: 0x6b6558, key: 0xfff8e8,
    keyIntensity: 2.2, hemiIntensity: 1.6, ambient: 0.25, background: 0x9dc4ef,
  },
  "golden-hour": {
    sky: 0xffd9a0, ground: 0x4a3524, key: 0xffb257,
    keyIntensity: 2.0, hemiIntensity: 1.1, ambient: 0.22, background: 0xe8a765,
  },
  overcast: {
    sky: 0xdfe3e8, ground: 0x5c5c5c, key: 0xf2f4f7,
    keyIntensity: 0.7, hemiIntensity: 2.1, ambient: 0.5, background: 0xb9c0c7,
  },
  dusk: {
    sky: 0x5f6fa8, ground: 0x201c2c, key: 0xff9060,
    keyIntensity: 0.9, hemiIntensity: 0.9, ambient: 0.2, background: 0x3a3556,
  },
  night: {
    sky: 0x2a3b63, ground: 0x0a0c14, key: 0x9fb6ff,
    keyIntensity: 0.35, hemiIntensity: 0.5, ambient: 0.1, background: 0x070910,
  },
  dungeon: {
    sky: 0x2b1d10, ground: 0x0b0805, key: 0xff9a3c,
    keyIntensity: 1.3, hemiIntensity: 0.35, ambient: 0.08, background: 0x0a0705,
  },
};

export function emptyRestyle(jobId = ""): RestyleDoc {
  return { v: 1, job_id: jobId, elements: {},
           lighting: { preset: "as-captured", intensity: 1 } };
}

/** Deterministic 0..1 hash — the tile must be identical every reload. */
function hashRng(seedText: string): () => number {
  let h = 2166136261;
  for (let i = 0; i < seedText.length; i++) {
    h ^= seedText.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return () => {
    h += 0x6d2b79f5;
    let t = Math.imul(h ^ (h >>> 15), 1 | h);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * A seamless-enough procedural tile for one taxonomy class: base_rgb with
 * seeded luma jitter and speckle — the same recipe as the backend's
 * class_textures.tile_for, so preview and bake do not disagree in kind.
 */
export function classTile(cls: RestyleMaterial, size = 256): THREE.DataTexture {
  const rng = hashRng(cls.id);
  const noise = cls.noise || {};
  const cells = Math.max(2, Math.round(noise.scale ?? 12));
  const jitter = noise.luma_jitter ?? 0.1;
  const speckle = noise.speckle ?? 0.3;

  // Value-noise lattice, wrapped so opposite edges match (tiles repeat).
  const lattice: number[] = [];
  for (let i = 0; i < cells * cells; i++) lattice.push(rng());
  const at = (cx: number, cy: number) =>
    lattice[((cy % cells) + cells) % cells * cells + (((cx % cells) + cells) % cells)];

  const data = new Uint8Array(size * size * 4);
  const [br, bg, bb] = cls.base_rgb as number[];
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const fx = (x / size) * cells;
      const fy = (y / size) * cells;
      const x0 = Math.floor(fx), y0 = Math.floor(fy);
      const tx = fx - x0, ty = fy - y0;
      const n =
        at(x0, y0) * (1 - tx) * (1 - ty) + at(x0 + 1, y0) * tx * (1 - ty) +
        at(x0, y0 + 1) * (1 - tx) * ty + at(x0 + 1, y0 + 1) * tx * ty;
      const grain = 1 + (n - 0.5) * 2 * jitter
        + (rng() - 0.5) * speckle * 0.35;
      const k = (y * size + x) * 4;
      data[k] = Math.max(0, Math.min(255, br * 255 * grain));
      data[k + 1] = Math.max(0, Math.min(255, bg * 255 * grain));
      data[k + 2] = Math.max(0, Math.min(255, bb * 255 * grain));
      data[k + 3] = 255;
    }
  }
  const tex = new THREE.DataTexture(data, size, size);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.needsUpdate = true;
  return tex;
}

/**
 * A material that projects `tile` triplanarly in world space.
 *
 * `unitsPerTile` is in SCENE units, so callers convert from metres with the
 * one scale dial — no scene-unit constants anywhere (repo rule).
 */
export function triplanarMaterial(
  tile: THREE.Texture, unitsPerTile: number, cls: RestyleMaterial,
  tint: THREE.Color | null,
): THREE.MeshStandardMaterial {
  const mat = new THREE.MeshStandardMaterial({
    map: tile,
    color: tint ?? new THREE.Color(0xffffff),
    roughness: cls.roughness ?? 0.85,
    metalness: cls.metallic ?? 0,
    side: THREE.DoubleSide,
  });
  const scale = { value: 1 / Math.max(1e-4, unitsPerTile) };
  mat.userData.triplanarScale = scale;
  mat.onBeforeCompile = (shader) => {
    shader.uniforms.uTriplanarScale = scale;
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>",
        "#include <common>\nvarying vec3 vTriPos;\nvarying vec3 vTriNrm;")
      .replace("#include <worldpos_vertex>",
        `#include <worldpos_vertex>
         vec4 triWorld = modelMatrix * vec4(transformed, 1.0);
         vTriPos = triWorld.xyz;
         vTriNrm = mat3(modelMatrix) * objectNormal;`);
    shader.fragmentShader = shader.fragmentShader
      .replace("#include <common>",
        `#include <common>
         varying vec3 vTriPos;
         varying vec3 vTriNrm;
         uniform float uTriplanarScale;`)
      .replace("#include <map_fragment>",
        `#ifdef USE_MAP
           vec3 triN = normalize(abs(vTriNrm) + 1e-5);
           triN /= (triN.x + triN.y + triN.z);
           vec3 triUV = vTriPos * uTriplanarScale;
           vec4 triX = texture2D(map, triUV.zy);
           vec4 triY = texture2D(map, triUV.xz);
           vec4 triZ = texture2D(map, triUV.xy);
           vec4 sampledDiffuseColor = triX * triN.x + triY * triN.y + triZ * triN.z;
           diffuseColor *= sampledDiffuseColor;
         #endif`);
  };
  // Distinct key so three.js does not reuse another material's program.
  mat.customProgramCacheKey = () => "triplanar-restyle";
  return mat;
}

/** The four numbers a light rig needs for a preset at a given intensity. */
export function presetFor(name: string, intensity: number): LightPreset {
  const base = LIGHT_PRESETS[name] ?? LIGHT_PRESETS["as-captured"];
  const k = Math.max(0, Math.min(3, intensity));
  return {
    ...base,
    keyIntensity: base.keyIntensity * k,
    hemiIntensity: base.hemiIntensity * k,
    ambient: base.ambient * k,
  };
}
