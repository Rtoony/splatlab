/// <reference types="vite/client" />
// Pulls in Vite's ambient module declarations — notably `*?raw`, which
// tokens.test.ts uses to read tailwind.config.js as text.

// Minimal shim for the ONE Node API a test needs. tokens.test.ts has to read
// src/index.css as text, and `?raw` cannot do it: vitest stubs every CSS
// import to an empty string (its `css` option defaults to false), which is why
// the first attempt silently asserted against "". @types/node is not installed
// and nothing else under src/ touches Node APIs, so this stays deliberately
// tiny rather than adding a dependency for one call.
declare module "node:fs" {
  export function readFileSync(path: string | URL, encoding: "utf8"): string;
}
