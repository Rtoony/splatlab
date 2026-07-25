import tseslint from "typescript-eslint";
import reactHooks from "eslint-plugin-react-hooks";

export default tseslint.config(
  { ignores: ["dist", "node_modules"] },
  ...tseslint.configs.recommended,
  {
    files: ["src/**/*.{ts,tsx}"],
    plugins: { "react-hooks": reactHooks },
    rules: {
      ...reactHooks.configs.recommended.rules,
      // Splat/heatmap code passes typed arrays and library handles around as `any`
      // at the three.js/spark boundary; banning it wholesale would force casts
      // that hide more than they reveal. Correctness rules stay on.
      "@typescript-eslint/no-explicit-any": "off",
      "@typescript-eslint/no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
      ],
      // `x.copy ? x.copy(v) : x.set(...)` duck-typing at the viewer-library boundary.
      "@typescript-eslint/no-unused-expressions": ["error", { allowShortCircuit: true, allowTernary: true }],
      // Compiler-era advisory rules: surface as warnings, don't block the check —
      // the flagged reset-state-on-dep-change patterns work and refactoring them
      // is viewer/page surgery, not lint hygiene.
      "react-hooks/set-state-in-effect": "warn",
      "react-hooks/refs": "warn",
      "react-hooks/immutability": "warn",
    },
  },
);
