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
