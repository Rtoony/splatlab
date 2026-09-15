import { describe, expect, it, vi } from "vitest";
import * as THREE from "three";
import { WorldWalker } from "./world-walker";

describe("paired authored-room lighting", () => {
  it.each([true, false])(
    "keeps captured texture policy while lighting explicit authored additions (unlit=%s)",
    (unlit) => {
      const worldGroup = new THREE.Group();
      const lights = new THREE.Group();
      const texture = new THREE.Texture();
      const captured = new THREE.Mesh();
      const authored = new THREE.Mesh();
      for (const mesh of [captured, authored]) {
        mesh.userData.litMaterial = new THREE.MeshStandardMaterial({ map: texture });
        mesh.userData.unlitMaterial = new THREE.MeshBasicMaterial({ map: texture });
        worldGroup.add(mesh);
      }
      authored.userData.authoredLighting = true;
      const portals = [{ architecture_id: "fixture" }];
      const rebind = vi.fn();
      const walker = Object.create(WorldWalker.prototype) as WorldWalker;
      Object.assign(walker, {
        params: { unlit },
        worldGroup,
        lights,
        architecturalPortals: portals,
        setArchitecturalPortals: rebind,
      });
      (walker as unknown as { applyMaterialMode(): void }).applyMaterialMode();
      expect(captured.material).toBe(unlit ? captured.userData.unlitMaterial : captured.userData.litMaterial);
      expect(authored.material).toBe(authored.userData.litMaterial);
      expect(lights.visible).toBe(true);
      expect(rebind).toHaveBeenCalledWith(portals);
      for (const mesh of [captured, authored]) {
        mesh.userData.litMaterial.dispose();
        mesh.userData.unlitMaterial.dispose();
        mesh.geometry.dispose();
      }
      texture.dispose();
    },
  );

  it("does not force lights on for legacy unlit scenes", () => {
    const worldGroup = new THREE.Group();
    const lights = new THREE.Group();
    const walker = Object.create(WorldWalker.prototype) as WorldWalker;
    Object.assign(walker, { params: { unlit: true }, worldGroup, lights, architecturalPortals: [] });
    (walker as unknown as { applyMaterialMode(): void }).applyMaterialMode();
    expect(lights.visible).toBe(false);
  });
});
