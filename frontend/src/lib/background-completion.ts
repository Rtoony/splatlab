export type Completion = {
  completion_id: string;
  recovery_id: string;
  selected_slug: string;
  prompt: string;
  observed_fraction: number;
  generated_fraction: number;
  stale: boolean;
  result: null | {
    status: string;
    generated_image_name: string;
    generator: { provider: string; model: string };
    triangles: number;
    material_diagnostics?: {
      observed_reference_mae_rgb8: number;
      boundary_anchor_mae_rgb8: number;
      boundary_jump_mae_rgb8: number;
      generated_boundary_jump_mae_rgb8: number;
      boundary_pairs: number;
      automatic_acceptance: false;
    };
  };
};

export type CompletionCollision = {
  collision_id: string;
  selected_slug: string;
  stale: boolean;
  result: null | { verdict: string };
};

export function completionOperation(completion: Completion, replace: boolean, collision?: CompletionCollision) {
  if (completion.stale || completion.result?.status !== "needs-review") {
    throw new Error("Import a material into a current completion before proposing it.");
  }
  if (
    replace &&
    (!collision ||
      collision.stale ||
      collision.selected_slug !== completion.selected_slug ||
      collision.result?.verdict !== "PASS_LOCAL_EDIT")
  ) {
    throw new Error("Removal needs passing, current collision evidence for the same captured object.");
  }
  return {
    kind: replace ? "replace" : "place",
    selected_slug: replace ? completion.selected_slug : undefined,
    selection_collision_id: replace ? collision?.collision_id : undefined,
    completion_id: completion.completion_id,
    slug: `completion-${completion.completion_id.slice(-8)}`,
    label: "Observed + invented support material",
  };
}

export function completionUpload(file: File | null, provider: string, model: string) {
  if (!file || file.size === 0 || file.size > 16 * 1024 ** 2 || !["image/png", "image/jpeg"].includes(file.type)) {
    throw new Error("Choose a PNG/JPEG file up to 16 MiB. The server also checks square dimensions and opacity.");
  }
  if ([provider, model].some((value) => !value.trim() || value.trim().length > 120)) {
    throw new Error("Record the provider and model/source, at most 120 characters each.");
  }
  const body = new FormData();
  body.append("file", file);
  body.append("provider", provider.trim());
  body.append("model", model.trim());
  return body;
}
