// Shared 4s poll of GET /api/splat/activity — the server-truth "what is busy
// right now" layer. Local isPending state dies with a reload, so a refreshed
// tab couldn't tell a still-running build from a crashed one; the backend's
// own in-process locks can (and a backend restart that kills the operation
// also clears its lock, so the signal stays truthful).
import { useQuery } from "@tanstack/react-query";
import { fetchActivity } from "@/lib/api";

// `fast` tightens the poll to 2s — callers pass it while a local edit
// mutation is pending so the stepped edit_progress keeps up with the server's
// ~8-15s edit pipeline. All observers share the one ["activity"] cache entry;
// react-query refetches at the smallest interval any active observer asks for.
export function useActivity(fast = false) {
  return useQuery({
    queryKey: ["activity"],
    queryFn: fetchActivity,
    refetchInterval: fast ? 2000 : 4000,
  });
}
