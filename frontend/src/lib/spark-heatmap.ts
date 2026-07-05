// Shared Spark language-heatmap machinery — used by the /spark-test spike AND
// the SparkSceneViewer beta on the view page. One implementation so the spike
// stays an honest testbed for exactly what ships.

import * as THREE from "three";
import { dyno, RgbaArray, readRgbaArray } from "@sparkjsdev/spark";

// GPU worldModifier: reads a per-splat scalar out of an RgbaArray texture
// (Spark's splat-painter mechanism — readRgbaArray keyed by splat index),
// tints rgb through a 5-stop viridis-ish ramp, and fades opacity below a
// threshold ("spotlight"). All knobs are dyno uniforms mutated in place —
// REMEMBER: Spark only re-runs the generator on mesh.updateVersion(), so
// every uniform write must be followed by one or it is a visual no-op.
export function buildHeatmapModifier({
  scalarArray,
  heatmapEnabled,
  spotlightEnabled,
  spotlightThreshold,
}: {
  scalarArray: RgbaArray;
  heatmapEnabled: ReturnType<typeof dyno.dynoBool>;
  spotlightEnabled: ReturnType<typeof dyno.dynoBool>;
  spotlightThreshold: ReturnType<typeof dyno.dynoFloat>;
}) {
  return dyno.dynoBlock({ gsplat: dyno.Gsplat }, { gsplat: dyno.Gsplat }, ({ gsplat }) => {
    if (!gsplat) throw new Error("heatmap modifier: no gsplat input");
    const { rgb, opacity, index } = dyno.splitGsplat(gsplat).outputs;
    const raw = readRgbaArray(scalarArray.dyno, index);
    // NOTE: dyno's own .d.ts types `swizzle`'s single-component selector
    // incorrectly, so single-component reads go through split().outputs.x.
    const t = dyno.split(raw).outputs.x;

    const stop0 = dyno.dynoVec3(new THREE.Vector3(0.267, 0.005, 0.329));
    const stop1 = dyno.dynoVec3(new THREE.Vector3(0.229, 0.322, 0.545));
    const stop2 = dyno.dynoVec3(new THREE.Vector3(0.128, 0.567, 0.551));
    const stop3 = dyno.dynoVec3(new THREE.Vector3(0.369, 0.789, 0.383));
    const stop4 = dyno.dynoVec3(new THREE.Vector3(0.993, 0.906, 0.144));
    let ramp = dyno.mix(stop0, stop1, dyno.smoothstep(dyno.dynoFloat(0.0), dyno.dynoFloat(0.25), t));
    ramp = dyno.mix(ramp, stop2, dyno.smoothstep(dyno.dynoFloat(0.25), dyno.dynoFloat(0.5), t));
    ramp = dyno.mix(ramp, stop3, dyno.smoothstep(dyno.dynoFloat(0.5), dyno.dynoFloat(0.75), t));
    ramp = dyno.mix(ramp, stop4, dyno.smoothstep(dyno.dynoFloat(0.75), dyno.dynoFloat(1.0), t));

    const heatmapMask = dyno.float(heatmapEnabled);
    const tintedRgb = dyno.mix(rgb, ramp, heatmapMask);

    const belowThreshold = dyno.lessThan(t, spotlightThreshold);
    const spotlightActive = dyno.and(spotlightEnabled, belowThreshold);
    const dimmedOpacity = dyno.mul(opacity, dyno.dynoFloat(0.04));
    const newOpacity = dyno.select(spotlightActive, dimmedOpacity, opacity);

    const outGsplat = dyno.combineGsplat({ gsplat, rgb: tintedRgb, opacity: newOpacity });
    return { gsplat: outGsplat };
  });
}

// One relevancy byte per splat -> the RGBA layout readRgbaArray expects.
export function packRelevancyRgba(bytes: Uint8Array): Uint8Array {
  const rgba = new Uint8Array(bytes.length * 4);
  for (let i = 0; i < bytes.length; i += 1) {
    const b = bytes[i];
    const o = i * 4;
    rgba[o] = b;
    rgba[o + 1] = b;
    rgba[o + 2] = b;
    rgba[o + 3] = 255;
  }
  return rgba;
}

export interface RelevancyResult {
  bytes: Uint8Array;
  relMin: string | null;
  relMax: string | null;
  matchCount: number | null;
  ms: number;
}

// POST the query, return the raw uint8 vector + receipt metadata. The server
// serves EXPORTED-PLY row order (langfield_align map) — the caller MUST still
// verify bytes.length === its loaded splat count and refuse to tint on
// mismatch (a wrong tint is worse than an error).
export async function fetchRelevancy(jobId: string, text: string): Promise<RelevancyResult> {
  const started = performance.now();
  const res = await fetch(`/api/splat/jobs/${jobId}/langfield/relevancy`, {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (res.status === 401) {
    window.location.href = "/login";
    throw new Error("unauthorized");
  }
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status}: ${detail.slice(0, 300)}`);
  }
  const bytes = new Uint8Array(await res.arrayBuffer());
  let matchCount: number | null = null;
  const matchesHeader = res.headers.get("X-Matches");
  if (matchesHeader) {
    try {
      const parsed = JSON.parse(matchesHeader);
      if (Array.isArray(parsed)) matchCount = parsed.length;
      else if (parsed && Array.isArray(parsed.matches)) matchCount = parsed.matches.length;
    } catch {
      matchCount = null;
    }
  }
  return {
    bytes,
    relMin: res.headers.get("X-Min"),
    relMax: res.headers.get("X-Max"),
    matchCount,
    ms: Math.round(performance.now() - started),
  };
}

// ---------------------------------------------------------------------------
// Multi-query overlay: up to 4 simultaneous text queries live in ONE RgbaArray
// (R/G/B/A = query 0..3 relevancy), composited by a mode-baked modifier.
// Structural knobs (mode, ramp, colors, channel count) are BAKED — rebuild the
// modifier + updateGenerator() on change (cheap, ms). Only the threshold and
// per-channel enables are live uniforms (updateVersion() after writes).
// ---------------------------------------------------------------------------

export type OverlayMode = "tint" | "highlight" | "isolate" | "spotlight";

export const RAMPS: Record<string, [number, number, number][]> = {
  viridis: [
    [0.267, 0.005, 0.329],
    [0.229, 0.322, 0.545],
    [0.128, 0.567, 0.551],
    [0.369, 0.789, 0.383],
    [0.993, 0.906, 0.144],
  ],
  turbo: [
    [0.19, 0.072, 0.232],
    [0.155, 0.736, 0.925],
    [0.646, 0.99, 0.235],
    [0.987, 0.537, 0.129],
    [0.48, 0.016, 0.011],
  ],
  magma: [
    [0.001, 0.0, 0.014],
    [0.316, 0.071, 0.485],
    [0.716, 0.215, 0.475],
    [0.986, 0.535, 0.382],
    [0.987, 0.991, 0.75],
  ],
  grayscale: [
    [0.05, 0.05, 0.05],
    [0.29, 0.29, 0.29],
    [0.53, 0.53, 0.53],
    [0.76, 0.76, 0.76],
    [1.0, 1.0, 1.0],
  ],
};

export function rampCssGradient(ramp: string): string {
  const stops = RAMPS[ramp] ?? RAMPS.viridis;
  const css = stops
    .map((s, i) => `rgb(${Math.round(s[0] * 255)},${Math.round(s[1] * 255)},${Math.round(s[2] * 255)}) ${(i / (stops.length - 1)) * 100}%`)
    .join(", ");
  return `linear-gradient(90deg, ${css})`;
}

// Pack up to 4 per-splat relevancy vectors into one RGBA texture buffer.
export function packChannelsRgba(channels: Uint8Array[], numSplats: number): Uint8Array {
  const rgba = new Uint8Array(numSplats * 4);
  for (let c = 0; c < Math.min(channels.length, 4); c += 1) {
    const src = channels[c];
    for (let i = 0; i < numSplats; i += 1) rgba[i * 4 + c] = src[i];
  }
  return rgba;
}

// Per-channel percentile cutoff: the byte value (as 0..1) above which ~pct%
// of THIS query's splats fall. Relevancy bytes are per-query min-max
// normalized, so a raw shared threshold means something different for every
// query — "top X%" is the semantics a human actually wants.
export function cutoffForTopPercent(bytes: Uint8Array, pct: number): number {
  const hist = new Uint32Array(256);
  for (let i = 0; i < bytes.length; i += 1) hist[bytes[i]] += 1;
  const target = (pct / 100) * bytes.length;
  let above = 0;
  for (let b = 255; b >= 1; b -= 1) {
    above += hist[b];
    if (above >= target) return b / 255;
  }
  return 1 / 255;
}

export function buildOverlayModifier({
  scalarArray,
  channelCount,
  channelColors,
  channelEnabled,
  mode,
  ramp,
  channelCutoffs,
  tintChannel = 0,
}: {
  scalarArray: RgbaArray;
  channelCount: number;
  channelColors: [number, number, number][]; // baked per channel
  channelEnabled: ReturnType<typeof dyno.dynoBool>[]; // live uniforms
  mode: OverlayMode; // baked
  ramp: string; // baked; used by "tint" (single-query ramp view)
  channelCutoffs: ReturnType<typeof dyno.dynoFloat>[]; // live uniforms, one per channel
  tintChannel?: number; // baked; which channel the "tint" ramp reads
}) {
  return dyno.dynoBlock({ gsplat: dyno.Gsplat }, { gsplat: dyno.Gsplat }, ({ gsplat }) => {
    if (!gsplat) throw new Error("overlay modifier: no gsplat input");
    const { rgb, opacity, index } = dyno.splitGsplat(gsplat).outputs;
    const texel = dyno.split(readRgbaArray(scalarArray.dyno, index)).outputs;
    const chans = [texel.x, texel.y, texel.z, texel.w].slice(0, Math.max(1, channelCount));

    if (mode === "tint") {
      // Single-query scientific view: full ramp over the whole scene.
      const stops = (RAMPS[ramp] ?? RAMPS.viridis).map((s) =>
        dyno.dynoVec3(new THREE.Vector3(s[0], s[1], s[2])),
      );
      const t = chans[Math.min(Math.max(tintChannel, 0), chans.length - 1)];
      // widen to mix()'s DynoVal type so the loop reassignment type-checks
      let col = dyno.mix(stops[0], stops[0], dyno.dynoFloat(0));
      for (let i = 1; i < stops.length; i += 1) {
        col = dyno.mix(
          col,
          stops[i],
          dyno.smoothstep(dyno.dynoFloat((i - 1) / (stops.length - 1)), dyno.dynoFloat(i / (stops.length - 1)), t),
        );
      }
      const on = dyno.float(channelEnabled[Math.min(Math.max(tintChannel, 0), channelEnabled.length - 1)]);
      return { gsplat: dyno.combineGsplat({ gsplat, rgb: dyno.mix(rgb, col, on) }) };
    }

    // Multi-query modes: a splat "matches" channel i when that channel is
    // enabled AND its relevancy exceeds ITS OWN percentile cutoff (per-query
    // normalization makes a shared raw threshold meaningless). The winning
    // channel is the highest-relevancy match (select-chain).
    // Initializers go through no-op select()s so the accumulators carry
    // select's widened DynoVal types (reassignment stays type-compatible).
    const never = dyno.dynoBool(false);
    let bestT = dyno.select(never, dyno.dynoFloat(-1), dyno.dynoFloat(-1));
    let bestColor = dyno.select(never, rgb, rgb);
    let anyMatch = dyno.select(never, dyno.dynoFloat(0), dyno.dynoFloat(0));
    for (let i = 0; i < chans.length; i += 1) {
      const above = dyno.and(channelEnabled[i], dyno.lessThan(channelCutoffs[i], chans[i]));
      const wins = dyno.and(above, dyno.lessThan(bestT, chans[i]));
      const color = dyno.dynoVec3(new THREE.Vector3(...(channelColors[i] ?? [1, 1, 0])));
      bestT = dyno.select(wins, chans[i], bestT);
      bestColor = dyno.select(wins, color, bestColor);
      anyMatch = dyno.select(above, dyno.dynoFloat(1), anyMatch);
    }
    const isMatch = dyno.lessThan(dyno.dynoFloat(0.5), anyMatch);

    if (mode === "highlight") {
      // Natural scene; matches blend strongly toward their query color.
      const tinted = dyno.mix(rgb, bestColor, dyno.dynoFloat(0.8));
      const outRgb = dyno.select(isMatch, tinted, rgb);
      return { gsplat: dyno.combineGsplat({ gsplat, rgb: outRgb }) };
    }
    if (mode === "isolate") {
      // Only matches visible (natural color); everything else vanishes.
      const hidden = dyno.mul(opacity, dyno.dynoFloat(0.015));
      const outOpacity = dyno.select(isMatch, opacity, hidden);
      return { gsplat: dyno.combineGsplat({ gsplat, opacity: outOpacity }) };
    }
    // spotlight: matches tinted + full opacity; the rest natural but dimmed.
    const tinted = dyno.mix(rgb, bestColor, dyno.dynoFloat(0.65));
    const outRgb = dyno.select(isMatch, tinted, rgb);
    const dimmed = dyno.mul(opacity, dyno.dynoFloat(0.06));
    const outOpacity = dyno.select(isMatch, opacity, dimmed);
    return { gsplat: dyno.combineGsplat({ gsplat, rgb: outRgb, opacity: outOpacity }) };
  });
}
