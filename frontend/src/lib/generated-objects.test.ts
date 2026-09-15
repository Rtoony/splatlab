import { expect, it } from "vitest";
import { generatedObjectOperation, type GeneratedObject, type GeneratedCollision } from "./generated-objects";

const object: GeneratedObject = {
  generated_object_id: "generated_12345678",
  selected_slug: "box",
  selection_review_id: "selection_fixture",
  recovery_id: "recovery_87654321",
  input_image_id: 58,
  stale: false,
  result: { status: "needs-review", placement_resolved: true },
};
const collision: GeneratedCollision = {
  collision_id: "collision_fixture",
  selection_review_id: object.selection_review_id,
  selected_slug: "box",
  stale: false,
  result: { verdict: "PASS_LOCAL_EDIT" },
};

it("builds one reviewed generated-object and observed-background operation", () => {
  expect(generatedObjectOperation(object, true, collision)).toMatchObject({
    kind: "replace",
    selected_slug: "box",
    generated_object_id: object.generated_object_id,
    selection_collision_id: collision.collision_id,
    background: { recovery_id: object.recovery_id, slug: "background-87654321" },
  });
  expect(generatedObjectOperation(object, false)).toMatchObject({ kind: "place", background: undefined });
});
it("refuses stale, unbuilt and unplaced outputs", () => {
  for (const candidate of [
    { ...object, stale: true },
    { ...object, result: null },
    { ...object, result: { status: "unplaced", placement_resolved: false } },
  ]) {
    expect(() => generatedObjectOperation(candidate, false)).toThrow();
  }
});
it("does not borrow clearance from a different selection or object", () => {
  for (const candidate of [
    undefined,
    { ...collision, selection_review_id: "selection_other" },
    { ...collision, selected_slug: "chair" },
    { ...collision, stale: true },
  ]) {
    expect(() => generatedObjectOperation(object, true, candidate)).toThrow(/same refined selection/);
  }
});

it("pairs only the same object's completed native appearance with its compound revision", () => {
  const gaussian = {
    gaussians_id: "gaussians_fixture",
    generated_object_id: object.generated_object_id,
    status: "needs-review",
  };
  expect(generatedObjectOperation(object, true, collision, gaussian)).toMatchObject({
    gaussians_id: gaussian.gaussians_id,
    generated_object_id: object.generated_object_id,
    selection_collision_id: collision.collision_id,
    background: { recovery_id: object.recovery_id },
  });
  for (const candidate of [
    { ...gaussian, status: "failed" },
    { ...gaussian, generated_object_id: "generated_other" },
  ]) {
    expect(() => generatedObjectOperation(object, false, undefined, candidate)).toThrow(/must belong/);
  }
  expect(() => generatedObjectOperation({ ...object, stale: true }, false, undefined, gaussian)).toThrow(/current/);
});
