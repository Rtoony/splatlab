import type { SplatStatusResponse } from "@/lib/contracts";
import { Card } from "@/components/ui";
import { AlertTriangle, Cpu } from "lucide-react";

export function ComputeGateBanner({ compute }: { compute: SplatStatusResponse["compute"] }) {
  if (!compute) return null;
  if (compute.enabled && compute.mode === "supervised" && compute.supervised_unlock?.active) {
    const minutes = Math.max(0, Math.ceil((compute.supervised_unlock.seconds_remaining ?? 0) / 60));
    return (
      <Card className="mb-5 border-emerald-500/30 bg-emerald-500/10 p-4">
        <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
          <div className="flex min-w-0 gap-3">
            <Cpu className="mt-0.5 h-5 w-5 shrink-0 text-emerald-300" />
            <div className="min-w-0">
              <p className="text-sm font-semibold text-emerald-100">Supervised compute is armed</p>
              <p className="mt-1 text-sm text-emerald-100/80">
                One SplatLab GPU job can run during this test window. Expires in about {minutes} minute{minutes === 1 ? "" : "s"}.
              </p>
              <p className="mt-1 truncate font-mono text-[11px] text-emerald-100/50">{compute.unlock_path || compute.supervised_unlock.path}</p>
            </div>
          </div>
          <div className="grid gap-2 text-xs text-emerald-100/80 sm:grid-cols-2 md:w-[520px]">
            <div className="rounded-xl border border-emerald-400/20 bg-black/20 p-2.5">
              <p className="mb-1 font-semibold text-emerald-100">Available now</p>
              <p>{compute.safe_capabilities.slice(0, 3).join(" · ")}</p>
            </div>
            <div className="rounded-xl border border-emerald-400/20 bg-black/20 p-2.5">
              <p className="mb-1 font-semibold text-emerald-100">Still held back</p>
              <p>{compute.blocked_capabilities.slice(0, 3).join(" · ")}</p>
            </div>
          </div>
        </div>
      </Card>
    );
  }
  if (compute.enabled) return null;
  return (
    <Card className="mb-5 border-amber-500/30 bg-amber-500/10 p-4">
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div className="flex min-w-0 gap-3">
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber-300" />
          <div className="min-w-0">
            <p className="text-sm font-semibold text-amber-100">SplatLab is open in safe browse mode</p>
            <p className="mt-1 text-sm text-amber-100/80">
              GPU generation and scene mutations are blocked by the hardware-maintenance gate:{" "}
              {compute.reason || "hardware maintenance is active."}
            </p>
            <p className="mt-1 truncate font-mono text-[11px] text-amber-100/50">{compute.marker_path}</p>
          </div>
        </div>
        <div className="grid gap-2 text-xs text-amber-100/80 sm:grid-cols-2 md:w-[520px]">
          <div className="rounded-xl border border-amber-400/20 bg-black/20 p-2.5">
            <p className="mb-1 font-semibold text-amber-100">Available now</p>
            <p>{compute.safe_capabilities.slice(0, 3).join(" · ")}</p>
          </div>
          <div className="rounded-xl border border-amber-400/20 bg-black/20 p-2.5">
            <p className="mb-1 font-semibold text-amber-100">Paused</p>
            <p>{compute.blocked_capabilities.slice(0, 3).join(" · ")}</p>
          </div>
        </div>
      </div>
    </Card>
  );
}
