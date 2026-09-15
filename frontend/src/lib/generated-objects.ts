import type { CompletionCollision } from "./background-completion";

export type GeneratedObject = {
  generated_object_id: string;
  selected_slug: string;
  selection_review_id: string;
  recovery_id: string;
  input_image_id: number;
  stale: boolean;
  result: null | {
    status: string;
    placement_resolved: boolean;
    master_triangles?: number;
    delivery_triangles?: number;
    error?: string;
    views?: { image_id: number; split: string; master_mask_iou: number; delivery_mask_iou: number }[];
  };
};

export type GeneratedCollision = CompletionCollision & { selection_review_id: string };
export type GeneratedGaussianChoice = { gaussians_id: string; generated_object_id: string; status: string };

export function generatedObjectOperation(
  object: GeneratedObject,
  replace: boolean,
  collision?: GeneratedCollision,
  gaussian?: GeneratedGaussianChoice,
) {
  if (object.stale || object.result?.status !== "needs-review" || !object.result.placement_resolved) {
    throw new Error("A current, completed and placed generated candidate is required.");
  }
  if (
    replace &&
    (!collision ||
      collision.stale ||
      collision.result?.verdict !== "PASS_LOCAL_EDIT" ||
      collision.selected_slug !== object.selected_slug ||
      collision.selection_review_id !== object.selection_review_id)
  ) {
    throw new Error("Replacement needs current clearance from this same refined selection.");
  }
  if (gaussian && (gaussian.status !== "needs-review" || gaussian.generated_object_id !== object.generated_object_id)) {
    throw new Error("Native Gaussian appearance must belong to this completed generated object.");
  }
  return {
    kind: replace ? "replace" : "place",
    generated_object_id: object.generated_object_id,
    gaussians_id: gaussian?.gaussians_id,
    slug: `generated-${object.generated_object_id.slice(-8)}`,
    label: "Local generated replacement; inferred geometry",
    selected_slug: replace ? object.selected_slug : undefined,
    selection_collision_id: replace ? collision?.collision_id : undefined,
    background: replace
      ? {
          slug: `background-${object.recovery_id.slice(-8)}`,
          recovery_id: object.recovery_id,
          label: "Observed background; unknown cells absent",
        }
      : undefined,
  };
}
