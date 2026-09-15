import { describe, expect, it } from "vitest";
import {
  completionOperation,
  completionUpload,
  type Completion,
  type CompletionCollision,
} from "./background-completion";

const completion: Completion = {
  completion_id: "completion_0123456789abcdef01234567",
  recovery_id: "recovery_0123456789abcdef01234567",
  selected_slug: "box",
  prompt: "Synthetic test fixture",
  observed_fraction: 0.25,
  generated_fraction: 0.75,
  stale: false,
  result: {
    status: "needs-review",
    generated_image_name: "generated.png",
    generator: { provider: "fixture", model: "fixture" },
    triangles: 8192,
  },
};
const collision: CompletionCollision = {
  collision_id: "collision_0123456789abcdef01234567",
  selected_slug: "box",
  stale: false,
  result: { verdict: "PASS_LOCAL_EDIT" },
};

describe("completion proposal contract", () => {
  it("places without silently hiding captured rows or adding a clearance study", () => {
    expect(completionOperation(completion, false, collision)).toMatchObject({
      kind: "place",
      completion_id: completion.completion_id,
      selected_slug: undefined,
      selection_collision_id: undefined,
    });
  });
  it("binds matching local clearance and replacement in one operation", () => {
    expect(completionOperation(completion, true, collision)).toMatchObject({
      kind: "replace",
      selected_slug: "box",
      selection_collision_id: collision.collision_id,
    });
  });
  it.each([
    { ...completion, stale: true },
    { ...completion, result: null },
    { ...completion, result: { ...completion.result!, status: "failed" } },
  ])("refuses unavailable completion", (candidate) => {
    expect(() => completionOperation(candidate, false)).toThrow("current completion");
  });
  it.each([
    undefined,
    { ...collision, stale: true },
    { ...collision, selected_slug: "other-box" },
    { ...collision, result: null },
    { ...collision, result: { verdict: "FAIL" } },
  ])("refuses missing or incompatible clearance", (candidate) => {
    expect(() => completionOperation(completion, true, candidate)).toThrow("same captured object");
  });
});

describe("completion import contract", () => {
  it("uses multipart without converting original file bytes", async () => {
    const file = new File([new Uint8Array([1, 2, 3])], "fixture.png", { type: "image/png" });
    const form = completionUpload(file, " local fixture ", " no inference ");
    expect(await (form.get("file") as File).arrayBuffer()).toEqual(await file.arrayBuffer());
    expect(form.get("provider")).toBe("local fixture");
    expect(form.get("model")).toBe("no inference");
  });
  it.each([
    null,
    new File([], "empty.png", { type: "image/png" }),
    new File(["bad"], "bad.svg", { type: "image/svg+xml" }),
    { type: "image/png", size: 16 * 1024 ** 2 + 1 } as File,
  ])("refuses missing, empty, unsupported or oversized images", (file) => {
    expect(() => completionUpload(file, "fixture", "fixture")).toThrow("PNG/JPEG");
  });
  it.each([
    ["", "fixture"],
    ["fixture", " "],
    ["x".repeat(121), "fixture"],
  ])("requires bounded source labels", (provider, model) => {
    expect(() =>
      completionUpload(new File(["fixture"], "fixture.jpg", { type: "image/jpeg" }), provider, model),
    ).toThrow("provider and model");
  });
});
