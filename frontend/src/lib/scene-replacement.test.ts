import { describe, expect, it } from "vitest";
import type { Completion, CompletionCollision } from "./background-completion";
import { replacementBackgrounds, replacementClearance, type RecoveryChoice } from "./scene-replacement";

const recovery: RecoveryChoice = {
  recovery_id: "recovery_12345678",
  selected_slug: "box",
  stale: false,
  report: { supported_fraction: 0.23 },
};
const completion: Completion = {
  completion_id: "completion_87654321",
  recovery_id: recovery.recovery_id,
  selected_slug: "box",
  stale: false,
  prompt: "fixture",
  observed_fraction: 0.23,
  generated_fraction: 0.77,
  result: {
    status: "needs-review",
    generated_image_name: "generated.png",
    generator: { provider: "fixture", model: "fixture" },
    triangles: 8192,
  },
};
const collision: CompletionCollision = {
  collision_id: "collision_current",
  selected_slug: "box",
  stale: false,
  result: { verdict: "PASS_LOCAL_EDIT" },
};

describe("compound replacement evidence", () => {
  it("keeps recovery and invented background choices distinct", () => {
    const choices = replacementBackgrounds("box", [recovery], [completion]);
    expect(choices).toHaveLength(2);
    expect(choices[0].label).toContain("23.0% supported");
    expect(choices[0].operation).toMatchObject({ recovery_id: recovery.recovery_id, slug: "background-12345678" });
    expect(choices[1].label).toContain("77.0% unknown");
    expect(choices[1].operation).toMatchObject({ completion_id: completion.completion_id });
  });
  it("excludes stale, foreign and unbuilt backgrounds", () => {
    expect(
      replacementBackgrounds(
        "box",
        [
          { ...recovery, stale: true },
          { ...recovery, selected_slug: "chair" },
        ],
        [
          { ...completion, stale: true },
          { ...completion, selected_slug: "chair" },
          { ...completion, result: null },
        ],
      ),
    ).toEqual([]);
  });
  it("never transfers a background when the selected object changes", () => {
    expect(replacementBackgrounds("chair", [recovery], [completion])).toEqual([]);
  });
  it("uses only current passing clearance for the selected instance", () => {
    expect(
      replacementClearance("box", [
        collision,
        { ...collision, stale: true },
        { ...collision, selected_slug: "chair" },
        { ...collision, result: null },
        { ...collision, result: { verdict: "FAIL" } },
      ]),
    ).toEqual(collision);
    expect(replacementClearance("chair", [collision])).toBeUndefined();
  });
});
