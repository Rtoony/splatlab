// Viewing-quality tiers (owner decision 2026-09-30): the full-quality splat is SHOWN only on the Nexus PC itself;
// phones and browsers outside the network get the compressed web or lite copy. Every tier stays downloadable
// everywhere (download menu). The server says whether this browser is the PC (GET /viewer-context); the tiers a
// scene has come with the job (quality_tiers).
import { useEffect, useState } from "react";
import { apiRequest } from "@/lib/api";
import type { QualityTier, QualityTierId } from "@/lib/contracts";

const PREF_KEY = "splatlab.quality";

export interface ViewerContext {
  local: boolean;
}

let contextPromise: Promise<ViewerContext> | null = null;

/** Once per page load; any failure reads as "not local" (never show full quality by accident). */
export function fetchViewerContext(): Promise<ViewerContext> {
  contextPromise ??= apiRequest<ViewerContext>("/api/splat/viewer-context").catch(() => ({ local: false }));
  return contextPromise;
}

export function isPhoneLike(): boolean {
  try {
    const coarse = window.matchMedia?.("(pointer: coarse)").matches ?? false;
    const small = Math.min(window.screen?.width ?? 9999, window.screen?.height ?? 9999) < 820;
    const mem = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
    return (coarse && small) || (typeof mem === "number" && mem <= 4);
  } catch {
    return false;
  }
}

/** The tiers this browser may VIEW: full only on the PC. */
export function viewableTiers(tiers: QualityTier[] | undefined, local: boolean): QualityTier[] {
  return (tiers ?? []).filter((t) => local || t.id !== "full");
}

export function readQualityPref(): QualityTierId | null {
  try {
    const v = window.localStorage.getItem(PREF_KEY);
    return v === "full" || v === "web" || v === "lite" ? v : null;
  } catch {
    return null;
  }
}

export function writeQualityPref(id: QualityTierId): void {
  try {
    window.localStorage.setItem(PREF_KEY, id);
  } catch {
    /* private window / blocked storage: the choice just is not remembered */
  }
}

/** Remembered choice if still viewable here; else PC -> full, phone -> lite, other browsers -> web. */
export function defaultTier(viewable: QualityTier[], local: boolean, phone: boolean,
                            pref: QualityTierId | null): QualityTier | null {
  if (!viewable.length) return null;
  const byId = (id: QualityTierId) => viewable.find((t) => t.id === id) ?? null;
  const order: QualityTierId[] = local && !phone ? ["full", "web", "lite"] : phone ? ["lite", "web"] : ["web", "lite"];
  return (pref && byId(pref)) || order.map(byId).find(Boolean) || viewable[0];
}

export function formatBytes(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes >= 1e6) return `${Math.round(bytes / 1e6)} MB`;
  return `${Math.max(1, Math.round(bytes / 1e3))} KB`;
}

export function formatSplats(n: number | null): string {
  if (!n) return "";
  const m = n / 1e6;
  return m >= 9.95 ? `${Math.round(m)}M` : m >= 1 ? `${m.toFixed(1)}M` : `${Math.round(n / 1e3)}k`;
}

/** The tier a viewer should load for this job, and a setter that remembers the choice on this device. */
export function useQualityTier(tiers: QualityTier[] | undefined) {
  const [local, setLocal] = useState<boolean | null>(null);
  const [chosen, setChosen] = useState<QualityTierId | null>(null);
  useEffect(() => {
    let live = true;
    void fetchViewerContext().then((c) => live && setLocal(c.local));
    return () => { live = false; };
  }, []);
  const viewable = viewableTiers(tiers, local === true);
  const current = local === null ? null
    : (chosen && viewable.find((t) => t.id === chosen)) || defaultTier(viewable, local, isPhoneLike(), readQualityPref());
  const choose = (id: QualityTierId) => {
    setChosen(id);
    writeQualityPref(id);
  };
  return { local, viewable, current, choose, ready: local !== null };
}
