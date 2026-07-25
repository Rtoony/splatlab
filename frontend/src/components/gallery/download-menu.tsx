// Download-format menu: legacy per-artifact links + the portable pipeline
// (build/rebuild SPZ/SOG/GLB, package UE 5.6). Rendered on gallery cards,
// the featured viewer, and the /view header.
import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "wouter";
import { buildPortableExports, buildUnrealBundle, fetchPortableExports } from "@/lib/api";
import type { SplatJob } from "@/lib/contracts";
import { Button } from "@/components/ui";
import { Box, ChevronDown, Download, Layers, Loader2 } from "lucide-react";

export function DownloadMenu({ job }: { job: SplatJob }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement | null>(null);
  const qc = useQueryClient();
  const exportQuery = useQuery({
    queryKey: ["portable-exports", job.job_id],
    queryFn: () => fetchPortableExports(job.job_id),
    enabled: open,
    staleTime: 15_000,
    retry: false,
  });
  const exportMutation = useMutation({
    mutationFn: () => buildPortableExports(job.job_id),
    onSuccess: (data) => qc.setQueryData(["portable-exports", job.job_id], data),
  });
  const unrealMutation = useMutation({
    mutationFn: () => buildUnrealBundle(job.job_id),
    onSuccess: () => exportQuery.refetch(),
  });
  useEffect(() => {
    const close = (e: MouseEvent) => ref.current && !ref.current.contains(e.target as Node) && setOpen(false);
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);
  const portable = exportQuery.data;
  const portableOpts = [
    {
      url: portable?.artifacts.spz?.status === "ready" ? portable.artifacts.spz.url : null,
      label: "Interchange .spz",
      ext: "spz",
      hint: "SPZ v4 · fast cross-tool loading",
    },
    {
      url: portable?.artifacts.sog?.status === "ready" ? portable.artifacts.sog.url : null,
      label: "Bundled .sog",
      ext: "sog",
      hint: "web delivery · CPU-compressed",
    },
    {
      url: portable?.artifacts["streamed-sog"]?.status === "ready" ? portable.artifacts["streamed-sog"].url : null,
      label: "Streamed SOG manifest",
      ext: "json",
      hint: "multi-LOD runtime · chunk URLs stay relative",
    },
    {
      url: portable?.artifacts.gltf?.status === "ready" ? portable.artifacts.gltf.url : null,
      label: "Gaussian .glb",
      ext: "glb",
      hint: "KHR_gaussian_splatting interchange",
    },
    {
      url: portable?.unreal_bundle?.zip_url,
      label: "Unreal 5.6 bundle",
      ext: "zip",
      hint: "Gaussian + geometry + receipts + checksums",
    },
    {
      url: portable?.status === "ready" ? portable.manifest_url : null,
      label: "Export manifest",
      ext: "json",
      hint: "bounds · scale · provenance · SHA-256",
    },
  ];
  const opts = [
    { url: job.preview_web_url, label: "Web .ply", ext: "ply", hint: "small · for sharing/viewing" },
    { url: job.preview_spz_url, label: "Compressed .spz", ext: "spz", hint: "smallest · modern viewers" },
    { url: job.preview_file_url, label: "Full .ply", ext: "ply", hint: "full quality · for editing" },
    { url: job.mesh_file_url, label: "Mesh .ply", ext: "ply", hint: "triangle mesh · Blender/CAD" },
    { url: job.mesh_glb_url, label: "Mesh .glb", ext: "glb", hint: "triangle mesh · drag into Blender" },
    { url: job.twin_glb_url, label: "Twin .glb", ext: "glb", hint: "splat-colored · real meters · Blender-ready" },
    { url: job.survey_dxf_url, label: "Site DXF", ext: "dxf", hint: "georeferenced TIN · grid coordinates" },
    { url: job.survey_landxml_url, label: "LandXML surface", ext: "xml", hint: "imports as a Civil 3D surface" },
    { url: job.contours_dxf_url, label: "Contours DXF", ext: "dxf", hint: "ground contours · office layers" },
    { url: job.ground_points_url, label: "Ground points", ext: "txt", hint: "PNEZD survey points" },
    { url: job.sections_url, label: "Sections view", ext: "png", hint: "cross-sections vs scene" },
    { url: job.surface_iso_url, label: "3D surface view", ext: "png", hint: "isometric ground TIN" },
    // Only the APPROVED build — the one mandatory HITL gate P6f exists to
    // enforce. An unapproved draft stays inside the Scene panel's own
    // preview-download links, never surfaced here.
    {
      url: job.scene?.assemble?.state === "approved" ? `/api/splat/jobs/${job.job_id}/scene/assemble/file?fmt=glb` : null,
      label: "Scene .glb", ext: "glb", hint: "reassembled scene · fidelity dial · approved",
    },
    {
      url: job.scene?.assemble?.state === "approved" ? `/api/splat/jobs/${job.job_id}/scene/assemble/file?fmt=blend` : null,
      label: "Scene .blend", ext: "blend", hint: "reassembled scene · Blender-native",
    },
    ...portableOpts,
  ].filter((o) => o.url);

  const exportError = (exportMutation.error as Error | null)?.message;
  const unrealError = (unrealMutation.error as Error | null)?.message;
  const preparing = exportMutation.isPending || unrealMutation.isPending;

  return (
    <div ref={ref} className="relative">
      <Button size="sm" variant="outline" onClick={() => setOpen((v) => !v)}>
        <Download className="h-3.5 w-3.5" /> <ChevronDown className="h-3 w-3" />
      </Button>
      {open && (
        <div className="absolute right-0 z-20 mt-1 max-h-[70vh] w-72 overflow-y-auto rounded-xl border border-white/10 bg-[#0a0f1a] shadow-2xl">
          <div className="border-b border-white/10 p-2">
            <p className="px-1 pb-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-cyan-300/70">
              Portable pipeline
            </p>
            {portable?.status === "stale" && (
              <p className="mb-1 rounded-md bg-amber-400/10 px-2 py-1 text-[11px] text-amber-200">
                Canonical PLY changed. Rebuild before download.
              </p>
            )}
            <div className="grid grid-cols-2 gap-1">
              <button
                type="button"
                disabled={preparing}
                onClick={() => exportMutation.mutate()}
                className="flex items-center justify-center gap-1.5 rounded-lg border border-cyan-400/20 bg-cyan-400/10 px-2 py-2 text-xs font-medium text-cyan-100 hover:bg-cyan-400/15 disabled:opacity-50"
              >
                {exportMutation.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Box className="h-3.5 w-3.5" />}
                {portable?.status === "ready" ? "Rebuild formats" : "Build formats"}
              </button>
              <button
                type="button"
                disabled={preparing || portable?.status !== "ready"}
                onClick={() => unrealMutation.mutate()}
                className="flex items-center justify-center gap-1.5 rounded-lg border border-amber-300/20 bg-amber-300/10 px-2 py-2 text-xs font-medium text-amber-100 hover:bg-amber-300/15 disabled:opacity-40"
              >
                {unrealMutation.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Layers className="h-3.5 w-3.5" />}
                Package UE 5.6
              </button>
            </div>
            {(exportError || unrealError) && (
              <p className="mt-1 line-clamp-3 px-1 text-[10px] text-red-300">{exportError || unrealError}</p>
            )}
          </div>
          <div>
            {exportQuery.isLoading && (
              <div className="flex items-center gap-2 px-3 py-3 text-xs text-zinc-500">
                <Loader2 className="h-3.5 w-3.5 animate-spin" /> Checking formats…
              </div>
            )}
            {opts.map((o) => (
              <a
                key={o.label}
                href={o.url!}
                download={`${job.job_id}.${o.ext}`}
                onClick={() => setOpen(false)}
                className="block px-3 py-2 hover:bg-white/5"
              >
                <p className="text-sm font-medium text-zinc-100">{o.label}</p>
                <p className="text-[11px] text-zinc-500">{o.hint}</p>
              </a>
            ))}
            {/* The Export tab itself isn't deep-linkable yet (workspace mode is
                page-local state), so this lands on /view's View mode — one
                click from the Export tab. */}
            <Link
              href={`/view/${job.job_id}`}
              onClick={() => setOpen(false)}
              className="block border-t border-white/10 px-3 py-2 hover:bg-white/5"
            >
              <p className="text-sm font-medium text-cyan-200">Open export center →</p>
              <p className="text-[11px] text-zinc-500">format knobs · collision · mesh & contour builds (Export tab)</p>
            </Link>
          </div>
        </div>
      )}
    </div>
  );
}
