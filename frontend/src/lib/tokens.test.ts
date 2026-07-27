import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import configRaw from "../../tailwind.config.js?raw";

// index.css is read with fs, not `?raw`: vitest stubs CSS imports to an empty
// string (its `css` option defaults to false), so a `?raw` import here would
// assert against "" and pass for the wrong reason forever.
const cssRaw = readFileSync(new URL("../index.css", import.meta.url), "utf8");

// Regression guard for a bug that shipped. The design tokens were hex-valued
// CSS vars mapped as a bare `var(--surface)`, so Tailwind could not build ANY
// opacity variant of them: `bg-surface/95`, `bg-surface/85`,
// `disabled:bg-accent/30` and every `border-accent/NN` compiled to no rule at
// all — with no build error — and the Edit/Export/Objects lane went live with
// literally no background, the splat scene reading straight through its text.
//
// Neither tsc nor eslint can catch this: the class name is a valid string, it
// just never becomes CSS. Asserted here as TEXT (via ?raw) rather than by
// importing the config, so the check needs no @types/node and still fails if
// someone edits either file back toward hex.

/** Tokens that must support opacity modifiers. */
const ALPHA_TOKENS = ["surface", "ink", "ink-muted", "accent", "accent-hover", "accent-ink"];

describe("design tokens support opacity modifiers", () => {
  it.each(ALPHA_TOKENS)("tailwind maps %s with an <alpha-value> slot", (name) => {
    expect(configRaw).toContain(`"rgb(var(--${name}) / <alpha-value>)"`);
  });

  it.each(ALPHA_TOKENS)("index.css defines %s as an 'r g b' triplet, not hex", (name) => {
    const m = new RegExp(`--${name}:\\s*([^;]+);`).exec(cssRaw);
    expect(m, `--${name} missing from index.css`).not.toBeNull();
    expect(m![1].trim()).toMatch(/^\d{1,3} \d{1,3} \d{1,3}$/);
  });

  it("never consumes an alpha token as a bare var() in raw CSS", () => {
    // `background: var(--surface)` would resolve to the literal "5 7 13" and
    // paint nothing — it has to be rgb(var(--surface)).
    const withoutDefinitions = cssRaw.replace(/--[a-z-]+:[^;]+;/g, "");
    for (const name of ALPHA_TOKENS) {
      expect(withoutDefinitions, `bare var(--${name}) in index.css`).not.toMatch(
        new RegExp(`(?<!rgb\\()var\\(--${name}\\)`),
      );
    }
  });

  it("keeps surface-raised bare — it is a baked-alpha overlay, not a triplet", () => {
    expect(configRaw).toContain(`"surface-raised": "var(--surface-raised)"`);
    expect(cssRaw).toMatch(/--surface-raised:\s*rgba\(/);
  });
});
