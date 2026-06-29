// Same-origin fetch helper. The splatlab backend proxies /api/* to the portal
// splat API with the bearer injected, so the browser only ever sees same-origin.
export async function apiRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { credentials: "same-origin", ...init });
  if (res.status === 401) {
    window.location.href = "/login";
    throw new Error("unauthorized");
  }
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(text || `HTTP ${res.status}`);
  }
  return (await res.json()) as T;
}
