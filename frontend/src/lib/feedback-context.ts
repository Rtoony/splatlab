import type { FeedbackContextSnapshot } from "@/lib/feedback-contracts";

type LastClickSummary = {
  tag?: string;
  role?: string | null;
  text?: string;
  aria_label?: string | null;
  title?: string | null;
  test_id?: string | null;
  href_path?: string | null;
  at: string;
};

type FailedApiCall = {
  method: string;
  path: string;
  status: number | "network_error";
  duration_ms: number;
  at: string;
};

type JsErrorSummary = {
  message: string;
  source?: string;
  line?: number;
  column?: number;
  at: string;
};

export type SplatlabFeedbackContext = Record<string, unknown>;

declare global {
  interface Window {
    __SPLATLAB_FEEDBACK_CONTEXT__?: SplatlabFeedbackContext | (() => SplatlabFeedbackContext | null | undefined);
  }
}

const MAX_RECENT = 12;
const SENSITIVE_QUERY_KEYS = /(token|secret|key|auth|code|password|passwd|session|jwt|bearer|credential|signature)/i;

const jsErrors: JsErrorSummary[] = [];
const failedApiCalls: FailedApiCall[] = [];
let lastClick: LastClickSummary | null = null;
let initialized = false;

function pushBounded<T>(items: T[], item: T) {
  items.unshift(item);
  if (items.length > MAX_RECENT) items.length = MAX_RECENT;
}

function compactText(text: string | null | undefined, max = 96): string | undefined {
  const trimmed = text?.replace(/\s+/g, " ").trim();
  if (!trimmed) return undefined;
  return trimmed.length > max ? `${trimmed.slice(0, max - 1)}…` : trimmed;
}

function sanitizeUrl(url: string): { url: string; path: string; search?: string; tab?: string | null } {
  const parsed = new URL(url, window.location.origin);
  const params = new URLSearchParams();
  let tab: string | null = null;
  parsed.searchParams.forEach((value, key) => {
    if (key === "tab") tab = compactText(value, 48) ?? null;
    if (!SENSITIVE_QUERY_KEYS.test(key) && ["tab", "view", "mode", "filter", "sort", "q", "job"].includes(key)) {
      params.set(key, compactText(value, 80) ?? "");
    }
  });
  const search = params.toString();
  return {
    url: `${parsed.origin}${parsed.pathname}${search ? `?${search}` : ""}${parsed.hash ? "#…" : ""}`,
    path: parsed.pathname,
    search: search || undefined,
    tab,
  };
}

function pathOnly(url: string): string {
  try {
    const parsed = new URL(url, window.location.origin);
    return parsed.pathname;
  } catch {
    return url.split("?")[0]?.slice(0, 160) || "unknown";
  }
}

function elementSummary(target: EventTarget | Element | null): Record<string, unknown> | null {
  if (!(target instanceof Element)) return null;
  const el = target.closest("button,a,input,textarea,select,[role],[data-testid]") ?? target;
  const summary: Record<string, unknown> = {
    tag: el.tagName.toLowerCase(),
    role: el.getAttribute("role"),
    aria_label: compactText(el.getAttribute("aria-label"), 80),
    title: compactText(el.getAttribute("title"), 80),
    test_id: compactText(el.getAttribute("data-testid"), 80),
  };
  if (el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement || el instanceof HTMLSelectElement) {
    summary.name = compactText(el.name, 80);
    summary.type = el instanceof HTMLInputElement ? el.type : undefined;
    summary.placeholder = compactText(el.getAttribute("placeholder"), 80);
  } else {
    summary.text = compactText(el.textContent, 96);
  }
  if (el instanceof HTMLAnchorElement) summary.href_path = pathOnly(el.href);
  return Object.fromEntries(Object.entries(summary).filter(([, value]) => value != null && value !== ""));
}

export function initializeFeedbackTelemetry() {
  if (initialized || typeof window === "undefined") return;
  initialized = true;

  window.addEventListener(
    "click",
    (event) => {
      const summary = elementSummary(event.target);
      lastClick = summary ? { ...summary, at: new Date().toISOString() } : null;
    },
    { capture: true },
  );

  window.addEventListener("error", (event) => {
    pushBounded(jsErrors, {
      message: compactText(event.message, 240) ?? "Script error",
      source: event.filename ? pathOnly(event.filename) : undefined,
      line: event.lineno || undefined,
      column: event.colno || undefined,
      at: new Date().toISOString(),
    });
  });

  window.addEventListener("unhandledrejection", (event) => {
    const reason = event.reason instanceof Error ? event.reason.message : String(event.reason ?? "Unhandled promise rejection");
    pushBounded(jsErrors, {
      message: compactText(reason, 240) ?? "Unhandled promise rejection",
      at: new Date().toISOString(),
    });
  });
}

export function recordFailedApiCall(input: RequestInfo | URL, init: RequestInit | undefined, status: number | "network_error", durationMs: number) {
  const raw = typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;
  const path = pathOnly(raw);
  if (!path.startsWith("/api/") || path.startsWith("/api/feedback")) return;
  pushBounded(failedApiCalls, {
    method: (init?.method ?? "GET").toUpperCase(),
    path,
    status,
    duration_ms: Math.round(durationMs),
    at: new Date().toISOString(),
  });
}

export function setSplatlabFeedbackContext(context: SplatlabFeedbackContext | null) {
  if (context) window.__SPLATLAB_FEEDBACK_CONTEXT__ = context;
  else delete window.__SPLATLAB_FEEDBACK_CONTEXT__;
}

async function fetchAppContext(): Promise<Record<string, unknown> | null> {
  try {
    const res = await fetch("/api/app-context", { credentials: "same-origin" });
    if (!res.ok) return null;
    const value = (await res.json()) as unknown;
    return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

function getSplatlabContext(): SplatlabFeedbackContext | null {
  try {
    const source = window.__SPLATLAB_FEEDBACK_CONTEXT__;
    const value = typeof source === "function" ? source() : source;
    return value && typeof value === "object" && !Array.isArray(value) ? value : null;
  } catch {
    return null;
  }
}

export async function collectFeedbackContext(): Promise<FeedbackContextSnapshot> {
  const route = sanitizeUrl(window.location.href);
  return {
    feedback_context_version: 1,
    captured_at: new Date().toISOString(),
    route,
    ui_state: {
      component_label: route.path.startsWith("/view/") ? "Splat viewer" : route.path === "/" ? "Splat Lab gallery" : null,
      scroll_x: Math.round(window.scrollX),
      scroll_y: Math.round(window.scrollY),
      splatlab: getSplatlabContext(),
    },
    active_element: elementSummary(document.activeElement),
    last_click: lastClick,
    browser: {
      user_agent: navigator.userAgent,
      language: navigator.language,
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      viewport: { width: window.innerWidth, height: window.innerHeight },
      screen: {
        width: window.screen.width,
        height: window.screen.height,
        device_pixel_ratio: window.devicePixelRatio || 1,
      },
    },
    recent_js_errors: jsErrors.slice(0, MAX_RECENT),
    recent_failed_api_calls: failedApiCalls.slice(0, MAX_RECENT),
    app_context: await fetchAppContext(),
  };
}

export function titleFromBody(body: string): string {
  const firstLine = body.replace(/\s+/g, " ").trim();
  if (!firstLine) return "Untitled feedback";
  return firstLine.length > 72 ? `${firstLine.slice(0, 69)}…` : firstLine;
}
