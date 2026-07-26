// Keyboard-shortcut resolution for the Spark splat editor, kept as pure
// functions on purpose.
//
// Why a separate module: the viewer's entire input layer lives inside one
// useEffect keyed on [url], so nothing in it is reachable from a test — and
// this project has no DOM test runner (vitest, pure-function tests only).
// Pulling the decision "which key means what, given the current tool state"
// out here makes the part that actually encodes intent executable-checkable;
// spark-scene-viewer's onKeyDown becomes a thin dispatch over the result.

export type ToolName = "paint" | "crop" | "box" | "measure";

/** The subset of KeyboardEvent the resolver reads. */
export type KeyEventLike = {
  code: string;
  key: string;
  ctrlKey: boolean;
  metaKey: boolean;
  shiftKey: boolean;
  altKey: boolean;
  repeat: boolean;
};

/** Everything the resolver needs to know about the viewer right now. */
export type ViewerToolState = {
  /** The armed click-owner, or null when the canvas is just a camera. */
  tool: ToolName | null;
  /** A two-click destructive confirm is latched (crop/box "Sure?"). */
  confirmArmed: boolean;
  /** A crop/box centre is placed, or a first measure point is pending. */
  hasPlacement: boolean;
  /** Size of the uncommitted paint selection. */
  selCount: number;
  /** Whether an undone stroke is available to redo. */
  canRedo: boolean;
  paintTarget: "field" | "class";
  /** Number of chips in the loaded class taxonomy (gates the 1-9 keys). */
  classCount: number;
};

export type EscapeStep = "cancel-confirm" | "clear-placement" | "disarm";

export type ShortcutAction =
  | { kind: "escape"; step: EscapeStep }
  | { kind: "toggle-tool"; tool: ToolName }
  | { kind: "step-radius"; dir: -1 | 1; fine: boolean }
  | { kind: "undo-stroke" }
  | { kind: "redo-stroke" }
  | { kind: "commit" }
  | { kind: "discard" }
  | { kind: "pick-class"; index: number }
  | { kind: "toggle-help" }
  | { kind: "reset-view" }
  | { kind: "toggle-chrome" };

/**
 * The Esc ladder: back out exactly one level per press, and NEVER destroy an
 * uncommitted paint selection — that is what the always-on selection chip in
 * the HUD is for. Returns null when Esc has nothing to do here, so the event
 * can keep bubbling (a Radix dialog above us still gets to close).
 */
export function escapeAction(state: ViewerToolState): EscapeStep | null {
  if (state.confirmArmed) return "cancel-confirm";
  if (state.hasPlacement) return "clear-placement";
  if (state.tool) return "disarm";
  return null;
}

/**
 * Multiplicative resize clamped to the tool's own slider bounds. Multiplicative
 * because these radii are log-scaled (a fixed step is either useless at 0.02 or
 * glacial at 5); clamped because key auto-repeat fires ~30x/s and would
 * otherwise run the value to 0 or past the scene in under a second.
 */
export function stepRadius(
  current: number,
  dir: -1 | 1,
  bounds: { min: number; max: number },
  fine = false,
): number {
  const factor = fine ? 1.05 : 1.25;
  const next = dir > 0 ? current * factor : current / factor;
  return Math.min(Math.max(next, bounds.min), bounds.max);
}

const TOOL_KEYS: Record<string, ToolName> = {
  KeyB: "paint",
  KeyM: "measure",
};

/**
 * Map a keypress to an editor action, or null to let it fall through to the
 * camera bindings (WASD pan, arrow roll) that the viewer handles separately.
 *
 * Callers must apply the text-entry guard themselves before calling this — the
 * resolver has no opinion about focus.
 */
export function resolveShortcut(e: KeyEventLike, state: ViewerToolState): ShortcutAction | null {
  const mod = e.ctrlKey || e.metaKey;

  // Undo/redo is the one binding that WANTS the modifier. Handle it before the
  // blanket modifier bail below.
  if (e.code === "KeyZ" && mod && !e.altKey) {
    if (e.shiftKey) return state.canRedo ? { kind: "redo-stroke" } : null;
    return { kind: "undo-stroke" };
  }
  if (e.code === "KeyY" && mod && !e.altKey) {
    return state.canRedo ? { kind: "redo-stroke" } : null;
  }

  // Everything else is unmodified. Without this, Ctrl+S/Ctrl+W/Cmd+D would be
  // swallowed by the tool and camera bindings below.
  if (mod || e.altKey) return null;

  // Auto-repeat is meaningful only for the resize keys; a held B must not
  // toggle the brush 30x a second.
  if (e.repeat && e.code !== "BracketLeft" && e.code !== "BracketRight") return null;

  if (e.code === "Escape") {
    const step = escapeAction(state);
    return step ? { kind: "escape", step } : null;
  }

  if (e.code === "BracketLeft" || e.code === "BracketRight") {
    // Measure has no radius to grow.
    if (!state.tool || state.tool === "measure") return null;
    return { kind: "step-radius", dir: e.code === "BracketRight" ? 1 : -1, fine: e.shiftKey };
  }

  // Tools. C is the crop family: bare = sphere, Shift = box.
  if (e.code === "KeyC") return { kind: "toggle-tool", tool: e.shiftKey ? "box" : "crop" };
  if (!e.shiftKey && TOOL_KEYS[e.code]) return { kind: "toggle-tool", tool: TOOL_KEYS[e.code] };

  // Bare Z stays an undo alias while painting (it shipped that way in the
  // first cut, and it is the fast one-handed reach mid-stroke).
  if (e.code === "KeyZ" && !e.shiftKey && state.tool === "paint") return { kind: "undo-stroke" };

  if (e.code === "Enter" || e.code === "NumpadEnter") {
    return state.tool ? { kind: "commit" } : null;
  }

  if (e.code === "Delete" || e.code === "Backspace") {
    if (state.selCount > 0 || state.hasPlacement) return { kind: "discard" };
    return null;
  }

  // 1-9 pick a class chip, but only when the class brush is the active target.
  if (state.tool === "paint" && state.paintTarget === "class" && !e.shiftKey) {
    const digit = /^Digit([1-9])$/.exec(e.code);
    if (digit) {
      const index = Number(digit[1]) - 1;
      return index < state.classCount ? { kind: "pick-class", index } : null;
    }
  }

  // "?" is Shift+/ on US layouts but lives elsewhere on others — trust e.key.
  if (e.key === "?") return { kind: "toggle-help" };

  if (e.code === "Digit0" && !e.shiftKey) return { kind: "reset-view" };
  if (e.code === "KeyH" && !e.shiftKey) return { kind: "toggle-chrome" };

  return null;
}

/** True when the event came from somewhere the user is typing. */
export function isTextEntryTarget(t: EventTarget | null): boolean {
  const el = t as HTMLElement | null;
  if (!el) return false;
  return (
    el.tagName === "INPUT" ||
    el.tagName === "TEXTAREA" ||
    el.tagName === "SELECT" ||
    el.isContentEditable === true
  );
}
