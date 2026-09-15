import { describe, expect, it } from "vitest";
import * as THREE from "three";

function multiplyFloat32(matrix: THREE.Matrix4, vector: number[]): number[] {
  return [0, 1, 2, 3].map((row) =>
    [0, 1, 2, 3].reduce(
      (sum, column) =>
        Math.fround(sum + Math.fround(Math.fround(matrix.elements[column * 4 + row]) * Math.fround(vector[column]))),
      0,
    ),
  );
}

function reconstruct(camera: THREE.PerspectiveCamera, point: number[], viewport: number[]) {
  const clip = multiplyFloat32(camera.projectionMatrix, [...point, 1]);
  const ndc = clip.slice(0, 3).map((value) => Math.fround(value / clip[3]));
  const pixels = ndc
    .slice(0, 2)
    .map((value, axis) => Math.fround(viewport[axis] + ((value + 1) * viewport[axis + 2]) / 2));
  const screen = pixels.map((value, axis) => Math.fround((2 * (value - viewport[axis])) / viewport[axis + 2] - 1));
  const ray = multiplyFloat32(camera.projectionMatrixInverse, [...screen, 0, 1]);
  const inverseW = Math.fround(1 / clip[3]);
  const stable = ray.slice(0, 3).map((value) => Math.fround(value / Math.fround(-ray[2] * inverseW)));
  const legacy = multiplyFloat32(camera.projectionMatrixInverse, [...ndc, 1]);
  return { stable, legacy: legacy.slice(0, 3).map((value) => Math.fround(value / legacy[3])) };
}

describe("architectural fragment position precision", () => {
  it("resolves millimetre separation at the observed camera range without moving either clipping plane", () => {
    const near = 0.0007571610382874666,
      far = 302.8644153149867,
      maximum = 2 ** 24 - 1;
    const linear = (distance: number) =>
      Math.round((far / (far - near) - (far * near) / ((far - near) * distance)) * maximum);
    const logarithmic = (distance: number) => Math.round((Math.log2(1 + distance) / Math.log2(1 + far)) * maximum);
    expect(linear(7)).toBe(linear(7.001));
    expect(logarithmic(7.001) - logarithmic(7)).toBeGreaterThan(300);
  });
  it("reproduces depth-unprojection boundary misclassification without changing the portal", () => {
    const camera = new THREE.PerspectiveCamera(65, 1.5, 0.0005, 200);
    const point = [0.39995, 0.3, -7];
    const result = reconstruct(camera, point, [0, 0, 960, 640]);
    expect(result.legacy[0]).toBeGreaterThan(0.4);
    expect(result.stable[0]).toBeLessThan(0.4);
    expect(Math.abs(result.legacy[2] - point[2])).toBeGreaterThan(0.001);
    expect(Math.abs(result.stable[2] - point[2])).toBeLessThan(0.000001);
  });
  it.each([0.0001, 0.0005, 0.005])("retains both sides of the cut at near plane %d", (near) => {
    const camera = new THREE.PerspectiveCamera(65, 1.5, near, 1000);
    for (const offset of [false, true]) {
      if (offset) camera.setViewOffset(1920, 1280, 240, 100, 960, 640);
      for (const distance of [1, 7, 20]) {
        for (const horizontal of [-0.40005, -0.39995, 0.39995, 0.40005]) {
          const point = [horizontal, 0.3, -distance];
          const result = reconstruct(camera, point, [31, 45, 1920, 1280]);
          expect(Math.abs(result.stable[0]) <= 0.4).toBe(Math.abs(horizontal) <= 0.4);
          expect(Math.max(...point.map((value, axis) => Math.abs(value - result.stable[axis])))).toBeLessThan(0.00001);
        }
      }
    }
  });
});
