import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { makeSkyTexture, skyColumn } from "./world-walker";

describe("the daylight sky behind authored worlds", () => {
  const column = skyColumn(256);

  it("runs from ground (straight down) to blue sky (straight up)", () => {
    const [r0, g0, b0] = column[0];
    const [r1, g1, b1] = column[column.length - 1];
    expect(Math.min(r0, g0)).toBeGreaterThan(b0); // earthy: red/green above blue
    expect(b1).toBeGreaterThan(r1 + 60); // blue zenith
    expect(b1).toBeGreaterThan(g1);
  });

  it("is pale at the horizon, so a window looks out onto a skyline", () => {
    const [r, g, b] = column[Math.round((column.length - 1) * 0.5)];
    expect(Math.min(r, g, b)).toBeGreaterThan(190);
  });

  it("gets steadily bluer above the horizon", () => {
    const blueness = column.slice(128).map(([r, , b]) => b - r);
    for (let i = 1; i < blueness.length; i++) expect(blueness[i]).toBeGreaterThanOrEqual(blueness[i - 1] - 1);
  });

  it("is an equirectangular background with one texel per row", () => {
    const texture = makeSkyTexture(64);
    expect(texture.mapping).toBe(THREE.EquirectangularReflectionMapping);
    expect(texture.image.width).toBe(1);
    expect(texture.image.height).toBe(64);
    const data = texture.image.data as Uint8Array;
    expect(Array.from(data.slice(0, 4))).toEqual([...skyColumn(64)[0], 255]);
    expect(Array.from(data.slice(63 * 4, 64 * 4))).toEqual([...skyColumn(64)[63], 255]);
  });
});
