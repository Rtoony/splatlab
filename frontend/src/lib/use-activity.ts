// Shared 4s poll of GET /api/splat/activity — the server-truth "what is busy
// right now" layer. Local isPending state dies with a reload, so a refreshed
// tab couldn't tell a still-running build from a crashed one; the backend's
// own in-process locks can (and a backend restart that kills the operation
// also clears its lock, so the signal stays truthful).
import { useQuery } from "@tanstack/react-query";
import { fetchActivity } from "@/lib/api";

export function useActivity() {
  return useQuery({
    queryKey: ["activity"],
    queryFn: fetchActivity,
    refetchInterval: 4000,
  });
}
