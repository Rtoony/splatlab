import type { Completion, CompletionCollision } from "./background-completion";

export type RecoveryChoice = {
  recovery_id: string;
  selected_slug: string;
  stale: boolean;
  report: { supported_fraction: number };
};

export function replacementBackgrounds(selectedSlug: string, recoveries: RecoveryChoice[], completions: Completion[]) {
  return [
    ...recoveries
      .filter((item) => !item.stale && item.selected_slug === selectedSlug)
      .map((item) => ({
        key: item.recovery_id,
        label: `Observed only · ${(item.report.supported_fraction * 100).toFixed(1)}% supported · ${item.recovery_id.slice(-8)}`,
        operation: {
          slug: `background-${item.recovery_id.slice(-8)}`,
          label: "Photo-supported background; unknown cells absent",
          recovery_id: item.recovery_id,
        },
      })),
    ...completions
      .filter((item) => !item.stale && item.selected_slug === selectedSlug && item.result?.status === "needs-review")
      .map((item) => ({
        key: item.completion_id,
        label: `Includes invention · ${(item.generated_fraction * 100).toFixed(1)}% unknown · ${item.completion_id.slice(-8)}`,
        operation: {
          slug: `background-${item.completion_id.slice(-8)}`,
          label: "Observed + invented background",
          completion_id: item.completion_id,
        },
      })),
  ];
}

export function replacementClearance(selectedSlug: string, collisions: CompletionCollision[]) {
  return collisions
    .filter((item) => !item.stale && item.selected_slug === selectedSlug && item.result?.verdict === "PASS_LOCAL_EDIT")
    .at(-1);
}
