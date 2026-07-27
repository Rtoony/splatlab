import { describe, expect, it } from "vitest";
import {
  type InteractionRecord,
  effectFor,
  isStateful,
  maxReachInSceneUnits,
  nextState,
  promptFor,
  reachInSceneUnits,
  resolveState,
  targetInfo,
} from "./world-interactions";

const LAMP: InteractionRecord = {
  slug: "lamp",
  verb: "toggle",
  prompt: "the lamp",
  states: ["off", "on"],
  initial: "off",
  effects: { on: { tint: "#ffd27f" }, off: { tint: null } },
  range_m: 2,
};

const BOX: InteractionRecord = {
  slug: "box",
  verb: "inspect",
  prompt: "the box",
  states: ["seen"],
  initial: "seen",
  effects: {},
  range_m: 2.5,
  text: "Someone wrote KITCHEN on the side.",
};

describe("nextState", () => {
  it("cycles a two-state record", () => {
    expect(nextState(LAMP, "off")).toBe("on");
    expect(nextState(LAMP, "on")).toBe("off");
  });

  it("is stable on a single-state record", () => {
    expect(nextState(BOX, "seen")).toBe("seen");
  });

  it("recovers to the initial when the saved state is no longer authored", () => {
    expect(nextState(LAMP, "dimmed")).toBe("off");
    expect(nextState(LAMP, null)).toBe("off");
  });

  it("wraps a three-state record", () => {
    const dial = { ...LAMP, states: ["low", "mid", "high"], initial: "low" };
    expect(nextState(dial, "high")).toBe("low");
  });
});

describe("effectFor", () => {
  it("returns the authored effect", () => {
    expect(effectFor(LAMP, "on")).toEqual({ tint: "#ffd27f" });
  });

  it("falls back cleanly on a state with no effect", () => {
    expect(effectFor(BOX, "seen")).toEqual({});
    expect(effectFor(LAMP, "unauthored")).toEqual({});
  });
});

describe("promptFor", () => {
  it("says what the key will DO, not what the element is", () => {
    expect(promptFor(LAMP, "off")).toBe("Turn on the lamp");
    expect(promptFor(LAMP, "on")).toBe("Turn off the lamp");
  });

  it("reads naturally for a read-only verb", () => {
    expect(promptFor(BOX, "seen")).toBe("Look at the box");
  });

  it("reads naturally for pickup in both directions", () => {
    const bike: InteractionRecord = {
      ...LAMP, slug: "bike", verb: "pickup", prompt: "the bicycle",
      states: ["placed", "held"], initial: "placed",
    };
    expect(promptFor(bike, "placed")).toBe("Pick up the bicycle");
    expect(promptFor(bike, "held")).toBe("Put down the bicycle");
  });
});

describe("targetInfo", () => {
  it("offers a next state for a stateful record", () => {
    expect(targetInfo(LAMP, "off")).toMatchObject({ slug: "lamp", state: "off", nextState: "on" });
  });

  it("offers no next state for a single-state record, and carries its text", () => {
    const info = targetInfo(BOX, "seen");
    expect(info.nextState).toBeNull();
    expect(info.text).toContain("KITCHEN");
  });

  it("omits text when none is authored", () => {
    expect(targetInfo(LAMP, "off").text).toBeUndefined();
  });
});

describe("isStateful", () => {
  it("distinguishes a toggle from a one-shot", () => {
    expect(isStateful(LAMP)).toBe(true);
    expect(isStateful(BOX)).toBe(false);
  });
});

describe("resolveState", () => {
  const present = new Set(["lamp", "box"]);

  it("starts at the authored initials with no save", () => {
    expect(resolveState([LAMP, BOX], null, present).applied).toEqual({ lamp: "off", box: "seen" });
  });

  it("applies a saved state over the initial", () => {
    expect(resolveState([LAMP], { lamp: "on" }, present).applied).toEqual({ lamp: "on" });
  });

  it("drops an element the rebuilt world lost, and keeps the survivor", () => {
    const resolved = resolveState([LAMP, BOX], { lamp: "on", box: "seen" }, new Set(["lamp"]));

    expect(resolved.applied).toEqual({ lamp: "on" });
    expect(resolved.dropped).toEqual([
      { slug: "box", saved: "seen", reason: "element is not in the rebuilt world" },
    ]);
  });

  it("drops a state the author removed and falls back to the initial", () => {
    const resolved = resolveState([LAMP], { lamp: "dimmed" }, present);

    expect(resolved.applied).toEqual({ lamp: "off" });
    expect(resolved.dropped[0].reason).toContain("no longer authored");
  });

  it("drops a save for an element with no authored interaction", () => {
    const resolved = resolveState([LAMP], { chair: "on" }, new Set(["lamp", "chair"]));

    expect(resolved.applied).toEqual({ lamp: "off" });
    expect(resolved.dropped[0].reason).toContain("no interaction is authored");
  });

  it("does not discard nineteen survivors because one vanished", () => {
    const records = Array.from({ length: 20 }, (_, i) => ({ ...LAMP, slug: `prop-${i}` }));
    const saved = Object.fromEntries(records.map((r) => [r.slug, "on"]));
    const survivors = new Set(records.slice(0, 19).map((r) => r.slug));

    const resolved = resolveState(records, saved, survivors);

    expect(Object.keys(resolved.applied)).toHaveLength(19);
    expect(resolved.dropped).toHaveLength(1);
  });

  it("skips the presence rule when there is no manifest to check against", () => {
    expect(resolveState([LAMP], { lamp: "on" }, null).applied).toEqual({ lamp: "on" });
  });

  it("does not mutate its inputs", () => {
    const saved = { lamp: "on" };
    const before = JSON.stringify(saved);
    resolveState([LAMP], saved, present);
    expect(JSON.stringify(saved)).toBe(before);
  });
});

describe("reach", () => {
  it("converts authored metres into scene units", () => {
    expect(reachInSceneUnits(LAMP, 0.5)).toBe(1);
    expect(reachInSceneUnits(LAMP, 2)).toBe(4);
  });

  it("falls back to 1:1 on a nonsense scale rather than reaching zero", () => {
    expect(reachInSceneUnits(LAMP, 0)).toBe(2);
    expect(reachInSceneUnits(LAMP, Number.NaN)).toBe(2);
  });

  it("takes the largest authored reach so one raycast serves every candidate", () => {
    expect(maxReachInSceneUnits([LAMP, BOX], 1)).toBe(2.5);
    expect(maxReachInSceneUnits([], 1)).toBe(0);
  });
});
