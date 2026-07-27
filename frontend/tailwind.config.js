/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      // `rgb(var(--x) / <alpha-value>)`, NOT a bare `var(--x)`. Tailwind needs
      // the alpha slot to generate bg-surface/85, disabled:bg-accent/30,
      // hover:border-accent/50 and friends; with a bare var it emits NO RULE
      // AT ALL for those classes and fails silently. That is how the Edit lane
      // shipped fully transparent. See src/index.css and tokens.test.ts.
      colors: {
        surface: "rgb(var(--surface) / <alpha-value>)",
        ink: "rgb(var(--ink) / <alpha-value>)",
        "ink-muted": "rgb(var(--ink-muted) / <alpha-value>)",
        accent: "rgb(var(--accent) / <alpha-value>)",
        "accent-hover": "rgb(var(--accent-hover) / <alpha-value>)",
        "accent-ink": "rgb(var(--accent-ink) / <alpha-value>)",
        // Baked-alpha overlay token — no <alpha-value> form exists for it.
        "surface-raised": "var(--surface-raised)",
      },
      // Collapse the historical radius drift onto a 3-step token scale:
      // every rounded/lg/xl/2xl in the codebase resolves to sm/md/lg vars.
      borderRadius: {
        DEFAULT: "var(--radius-sm)",
        sm: "var(--radius-sm)",
        md: "var(--radius-sm)",
        lg: "var(--radius-sm)",
        xl: "var(--radius-md)",
        "2xl": "var(--radius-lg)",
        "3xl": "var(--radius-lg)",
      },
    },
  },
  plugins: [],
};
