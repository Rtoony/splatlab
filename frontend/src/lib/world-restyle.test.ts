// The restyle primitives: deterministic class tiles, real triplanar shader
// injection, and light presets that actually differ.

import { describe, expect, it } from "vitest";
import * as THREE from "three";

import {
  LIGHT_PRESETS,
  classTile,
  emptyRestyle,
  presetFor,
  triplanarMaterial,
  type RestyleMaterial,
} from "./world-restyle";

const GRASS: RestyleMaterial = {
  id: "grass", display: "Grass", category: "ground", color: "#4ade80",
  base_rgb: [0.28, 0.46, 0.2], roughness: 0.9, metallic: 0,
  noise: { scale: 14, luma_jitter: 0.1, speckle: 0.35 },
};
const CONCRETE: RestyleMaterial = {
  id: "concrete", display: "Concrete", category: "structure", color: "#9ca3af",
  base_rgb: [0.62, 0.62, 0.6], roughness: 0.8, metallic: 0,
  noise: { scale: 8, luma_jitter: 0.05, speckle: 0.2 },
};

describe("class tiles", () => {
  it("are bit-identical across runs (receipts must reproduce)", () => {
    const a = classTile(GRASS, 32).image.data as Uint8Array;
    const b = classTile(GRASS, 32).image.data as Uint8Array;
    expect(Array.from(a)).toEqual(Array.from(b));
  });

  it("differ per class and sit around the taxonomy's base colour", () => {
    const grass = classTile(GRASS, 32).image.data as Uint8Array;
    const concrete = classTile(CONCRETE, 32).image.data as Uint8Array;
    expect(Array.from(grass)).not.toEqual(Array.from(concrete));
    const mean = (d: Uint8Array, ch: number) => {
      let sum = 0;
      for (let i = ch; i < d.length; i += 4) sum += d[i];
      return sum / (d.length / 4);
    };
    // Grass is greenest of its own channels; both track base_rgb within the
    // jitter the taxonomy asks for.
    expect(mean(grass, 1)).toBeGreaterThan(mean(grass, 0));
    expect(mean(grass, 1)).toBeGreaterThan(mean(grass, 2));
    expect(Math.abs(mean(grass, 1) - 0.46 * 255)).toBeLessThan(20);
    expect(Math.abs(mean(concrete, 0) - 0.62 * 255)).toBeLessThan(20);
  });

  it("tile, so a repeated surface has no visible seam texture", () => {
    const tex = classTile(GRASS, 16);
    expect(tex.wrapS).toBe(THREE.RepeatWrapping);
    expect(tex.wrapT).toBe(THREE.RepeatWrapping);
  });
});

describe("triplanar material", () => {
  it("injects a world-space projection instead of using the atlas UVs", () => {
    const mat = triplanarMaterial(classTile(GRASS, 8), 2, GRASS, null);
    const shader = {
      uniforms: {} as Record<string, unknown>,
      vertexShader: "#include <common>\nvoid main(){\n#include <worldpos_vertex>\n}",
      fragmentShader: "#include <common>\nvoid main(){\n#include <map_fragment>\n}",
    };
    mat.onBeforeCompile!(shader as never, null as never);
    expect(shader.fragmentShader).toContain("uTriplanarScale");
    expect(shader.fragmentShader).toContain("texture2D(map, triUV.xz)");
    expect(shader.vertexShader).toContain("vTriPos");
    // The scale uniform is shared with the material so a live scale change
    // is one assignment, not a recompile.
    expect(shader.uniforms.uTriplanarScale).toBe(mat.userData.triplanarScale);
    expect((shader.uniforms.uTriplanarScale as { value: number }).value)
      .toBeCloseTo(0.5, 6); // 1 / 2 scene units per tile
  });

  it("carries the class's PBR numbers and an optional tint", () => {
    const tint = new THREE.Color("#ff8800");
    const mat = triplanarMaterial(classTile(CONCRETE, 8), 1, CONCRETE, tint);
    expect(mat.roughness).toBeCloseTo(0.8, 5);
    expect(mat.metalness).toBeCloseTo(0, 5);
    expect(mat.color.getHexString()).toBe("ff8800");
    expect(mat.side).toBe(THREE.DoubleSide);
  });

  it("uses its own program cache key so three.js cannot reuse a plain shader", () => {
    const mat = triplanarMaterial(classTile(GRASS, 8), 1, GRASS, null);
    expect(mat.customProgramCacheKey!()).toContain("triplanar");
  });
});

describe("light presets", () => {
  it("as-captured means do not light it — every other preset does", () => {
    expect(LIGHT_PRESETS["as-captured"].asCaptured).toBe(true);
    for (const [name, p] of Object.entries(LIGHT_PRESETS)) {
      if (name !== "as-captured") expect(p.asCaptured).toBeUndefined();
    }
  });

  it("scales the whole rig by intensity, clamped to the document's bounds", () => {
    const half = presetFor("noon", 0.5);
    const full = presetFor("noon", 1);
    expect(half.keyIntensity).toBeCloseTo(full.keyIntensity / 2, 5);
    expect(half.hemiIntensity).toBeCloseTo(full.hemiIntensity / 2, 5);
    // The backend caps intensity at 3; the walker refuses to exceed it even
    // if a hand-edited file slips through.
    expect(presetFor("noon", 99).keyIntensity)
      .toBeCloseTo(full.keyIntensity * 3, 5);
    expect(presetFor("noon", -5).keyIntensity).toBe(0);
  });

  it("falls back to as-captured for an unknown preset name", () => {
    expect(presetFor("vaporwave", 1).background)
      .toBe(LIGHT_PRESETS["as-captured"].background);
  });

  it("night and dungeon are actually darker than noon", () => {
    const noon = presetFor("noon", 1);
    for (const dark of ["night", "dungeon"]) {
      expect(presetFor(dark, 1).ambient).toBeLessThan(noon.ambient);
      expect(presetFor(dark, 1).hemiIntensity).toBeLessThan(noon.hemiIntensity);
    }
  });
});

describe("empty restyle", () => {
  it("is the capture, untouched", () => {
    const doc = emptyRestyle("splat_ab12cd34");
    expect(doc.elements).toEqual({});
    expect(doc.lighting).toEqual({ preset: "as-captured", intensity: 1 });
  });
});
