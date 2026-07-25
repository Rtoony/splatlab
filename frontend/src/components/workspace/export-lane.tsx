// Export lane — the /view workspace Export tab. v1 hosts the DownloadMenu's
// full artifact list inline plus per-family explainers; wave 6 grows this
// into the Export Center (collision, post-hoc mesh, contours triggers, format
// knobs, staleness handling).
import type { SplatJob } from "@/lib/contracts";
import { DownloadMenu } from "@/components/gallery/download-menu";
import { SectionLabel } from "@/components/ui";
import { PackageOpen } from "lucide-react";

const FAMILIES: { name: string; blurb: string }[] = [
  { name: "Splat formats", blurb: "SPZ for cross-tool interchange · SOG for web streaming · GLB (KHR_gaussian_splatting) for DCC tools." },
  { name: "Game engines", blurb: "Unreal 5.6 bundle — gaussians, collision, geometry twins, and survey artifacts with an import contract." },
  { name: "Mesh / DCC", blurb: "Triangle mesh (.ply/.glb) and the colored, decimated twin (.glb) for Blender, CAD, and printing." },
  { name: "Survey / CAD", blurb: "Georeferenced DXF, LandXML surface, contours, PNEZD ground points — needs scale calibration + a Locate anchor." },
];

export function ExportLane({ job }: { job: SplatJob }) {
  return (
    <div className="space-y-4 p-4">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <PackageOpen className="h-4 w-4 text-cyan-300" />
          <SectionLabel>Export</SectionLabel>
        </div>
        <DownloadMenu job={job} />
      </div>

      <div className="space-y-2">
        {FAMILIES.map((f) => (
          <div key={f.name} className="rounded-2xl border border-white/10 bg-white/[0.02] p-3">
            <p className="text-sm font-semibold text-zinc-100">{f.name}</p>
            <p className="mt-0.5 text-xs leading-relaxed text-zinc-400">{f.blurb}</p>
          </div>
        ))}
      </div>

      <p className="text-[11px] leading-relaxed text-zinc-600">
        Use the download button above for everything that's built. Format knobs, collision meshes, and on-demand
        contour/mesh builds land here next.
      </p>
    </div>
  );
}
