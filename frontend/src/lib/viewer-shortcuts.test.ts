import { describe, expect, it } from "vitest";
import {
  type KeyEventLike,
  type ViewerToolState,
  escapeAction,
  resolveShortcut,
  stepRadius,
} from "./viewer-shortcuts";

const IDLE: ViewerToolState = {
  tool: null,
  confirmArmed: false,
  hasPlacement: false,
  selCount: 0,
  canRedo: false,
  paintTarget: "field",
  classCount: 9,
};

function state(over: Partial<ViewerToolState> = {}): ViewerToolState {
  return { ...IDLE, ...over };
}

function key(code: string, over: Partial<KeyEventLike> = {}): KeyEventLike {
  return {
    code,
    key: over.key ?? "",
    ctrlKey: false,
    metaKey: false,
    shiftKey: false,
    altKey: false,
    repeat: false,
    ...over,
  };
}

describe("escapeAction — the back-out ladder", () => {
  it("backs out one level per press, in order", () => {
    // A crop with a placed centre and an armed confirm unwinds confirm first,
    // then the placement, then the tool itself.
    expect(escapeAction(state({ tool: "crop", confirmArmed: true, hasPlacement: true }))).toBe(
      "cancel-confirm",
    );
    expect(escapeAction(state({ tool: "crop", hasPlacement: true }))).toBe("clear-placement");
    expect(escapeAction(state({ tool: "crop" }))).toBe("disarm");
  });

  it("does nothing when no tool is armed, so the event can bubble to a dialog", () => {
    expect(escapeAction(IDLE)).toBeNull();
  });

  it("never returns a step that discards an uncommitted paint selection", () => {
    // This is the guarantee behind the always-on selection chip: Esc gets you
    // out of the tool, it does not throw away 20 minutes of painting.
    const painting = state({ tool: "paint", selCount: 1204 });
    expect(escapeAction(painting)).toBe("disarm");
    expect(escapeAction(state({ selCount: 1204 }))).toBeNull();
  });
});

describe("stepRadius", () => {
  it("clamps at both ends", () => {
    const bounds = { min: 0.01, max: 5 };
    expect(stepRadius(0.01, -1, bounds)).toBe(0.01);
    expect(stepRadius(5, 1, bounds)).toBe(5);
  });

  it("is multiplicative and reversible, and honours the fine modifier", () => {
    const bounds = { min: 0.001, max: 100 };
    expect(stepRadius(1, 1, bounds)).toBeCloseTo(1.25);
    expect(stepRadius(stepRadius(1, 1, bounds), -1, bounds)).toBeCloseTo(1);
    expect(stepRadius(1, 1, bounds, true)).toBeCloseTo(1.05);
  });

  it("survives auto-repeat without running away", () => {
    const bounds = { min: 0.02, max: 5 };
    let r = 1;
    for (let i = 0; i < 200; i++) r = stepRadius(r, 1, bounds);
    expect(r).toBe(5);
  });
});

describe("resolveShortcut — modifiers", () => {
  it("undoes on Ctrl+Z and redoes on Ctrl+Shift+Z", () => {
    expect(resolveShortcut(key("KeyZ", { ctrlKey: true }), IDLE)).toEqual({ kind: "undo-stroke" });
    expect(
      resolveShortcut(key("KeyZ", { ctrlKey: true, shiftKey: true }), state({ canRedo: true })),
    ).toEqual({ kind: "redo-stroke" });
  });

  it("ignores redo when there is nothing to redo", () => {
    expect(resolveShortcut(key("KeyZ", { ctrlKey: true, shiftKey: true }), IDLE)).toBeNull();
  });

  it("treats bare Z as an undo alias only while painting", () => {
    expect(resolveShortcut(key("KeyZ"), state({ tool: "paint" }))).toEqual({ kind: "undo-stroke" });
    expect(resolveShortcut(key("KeyZ"), state({ tool: "crop" }))).toBeNull();
  });

  it("never swallows a browser shortcut", () => {
    // Ctrl+S must reach the browser, not pan the camera or arm a tool.
    for (const code of ["KeyS", "KeyW", "KeyC", "KeyB", "KeyH", "Digit0"]) {
      expect(resolveShortcut(key(code, { ctrlKey: true }), state({ tool: "paint" }))).toBeNull();
      expect(resolveShortcut(key(code, { metaKey: true }), state({ tool: "paint" }))).toBeNull();
    }
  });

  it("drops auto-repeat for everything except the resize keys", () => {
    expect(resolveShortcut(key("KeyB", { repeat: true }), IDLE)).toBeNull();
    expect(resolveShortcut(key("BracketRight", { repeat: true }), state({ tool: "paint" }))).toEqual(
      { kind: "step-radius", dir: 1, fine: false },
    );
  });
});

describe("resolveShortcut — tools", () => {
  it("maps the tool keys, with Shift+C for the box", () => {
    expect(resolveShortcut(key("KeyB"), IDLE)).toEqual({ kind: "toggle-tool", tool: "paint" });
    expect(resolveShortcut(key("KeyC"), IDLE)).toEqual({ kind: "toggle-tool", tool: "crop" });
    expect(resolveShortcut(key("KeyC", { shiftKey: true }), IDLE)).toEqual({
      kind: "toggle-tool",
      tool: "box",
    });
    expect(resolveShortcut(key("KeyM"), IDLE)).toEqual({ kind: "toggle-tool", tool: "measure" });
  });

  it("resizes only a tool that has a radius", () => {
    expect(resolveShortcut(key("BracketLeft"), state({ tool: "crop" }))).toEqual({
      kind: "step-radius",
      dir: -1,
      fine: false,
    });
    expect(resolveShortcut(key("BracketLeft"), state({ tool: "measure" }))).toBeNull();
    expect(resolveShortcut(key("BracketLeft"), IDLE)).toBeNull();
  });

  it("leaves WASD and the arrows to the camera bindings", () => {
    for (const code of ["KeyW", "KeyA", "KeyS", "KeyD", "ArrowLeft", "ArrowRight"]) {
      expect(resolveShortcut(key(code), state({ tool: "paint" }))).toBeNull();
    }
  });
});

describe("resolveShortcut — commit and discard", () => {
  it("commits only with a tool armed", () => {
    expect(resolveShortcut(key("Enter"), state({ tool: "paint" }))).toEqual({ kind: "commit" });
    expect(resolveShortcut(key("NumpadEnter"), state({ tool: "crop" }))).toEqual({ kind: "commit" });
    expect(resolveShortcut(key("Enter"), IDLE)).toBeNull();
  });

  it("discards only when there is something to discard", () => {
    expect(resolveShortcut(key("Delete"), state({ selCount: 3 }))).toEqual({ kind: "discard" });
    expect(resolveShortcut(key("Backspace"), state({ hasPlacement: true }))).toEqual({
      kind: "discard",
    });
    expect(resolveShortcut(key("Delete"), IDLE)).toBeNull();
  });
});

describe("resolveShortcut — class picking", () => {
  const classing = state({ tool: "paint", paintTarget: "class", classCount: 9 });

  it("picks a class only while the class brush is active", () => {
    expect(resolveShortcut(key("Digit1"), classing)).toEqual({ kind: "pick-class", index: 0 });
    expect(resolveShortcut(key("Digit9"), classing)).toEqual({ kind: "pick-class", index: 8 });
    expect(resolveShortcut(key("Digit1"), state({ tool: "paint" }))).toBeNull();
    expect(resolveShortcut(key("Digit1"), state({ tool: "crop" }))).toBeNull();
  });

  it("refuses digits past the end of the taxonomy", () => {
    expect(resolveShortcut(key("Digit9"), { ...classing, classCount: 4 })).toBeNull();
    expect(resolveShortcut(key("Digit4"), { ...classing, classCount: 4 })).toEqual({
      kind: "pick-class",
      index: 3,
    });
  });

  it("keeps 0 as reset-view rather than a tenth class", () => {
    expect(resolveShortcut(key("Digit0"), classing)).toEqual({ kind: "reset-view" });
  });
});

describe("resolveShortcut — view and help", () => {
  it("opens help on ? regardless of layout", () => {
    expect(resolveShortcut(key("Slash", { key: "?", shiftKey: true }), IDLE)).toEqual({
      kind: "toggle-help",
    });
  });

  it("maps 0 to reset-view and H to chrome", () => {
    expect(resolveShortcut(key("Digit0"), IDLE)).toEqual({ kind: "reset-view" });
    expect(resolveShortcut(key("KeyH"), IDLE)).toEqual({ kind: "toggle-chrome" });
  });
});
