import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiRequest } from "@/lib/api";

type StructuralStudy = {
  structure_id: string;
  stale: boolean;
  scope: string;
  cameras: { image_id: number; split: string; photo: string; group: string }[];
  result: {
    status: string;
    accepted_points: number;
    check_scope: string;
    views: { image_id: number; observed_points: number; positive_points: number; overlay: string }[];
    planes: {
      plane_id: string;
      support_count: number;
      rms_m: number;
      lower_uv: number[];
      upper_uv: number[];
      support_grid_coverage: number;
      scope: string;
      appearance_checks: { image_id: number; observed_points: number; semantic_agreeing_points: number }[];
    }[];
  } | null;
};

export function StructuralSurfacePanel({ jobId, generation }: { jobId: string; generation: number }) {
  const [shown, setShown] = useState("");
  const [imageId, setImageId] = useState<number | null>(null);
  const base = `/api/splat/jobs/${jobId}/studio/structural-surfaces`;
  const query = useQuery({
    queryKey: ["structural-surfaces", jobId, generation],
    queryFn: () => apiRequest<{ studies: StructuralStudy[] }>(base),
  });
  const studies = query.data?.studies || [];
  const study = studies.find((item) => item.structure_id === shown) || studies.at(-1);
  const camera = study?.cameras.find((item) => item.image_id === imageId) || study?.cameras[0];
  const view = study?.result?.views.find((item) => item.image_id === camera?.image_id);
  const artifact = (name: string, stage = "receipt") =>
    `${base}/${study?.structure_id}/artifact?name=${encodeURIComponent(name)}&stage=${stage}`;
  const input = "w-full rounded-lg border border-white/15 bg-zinc-950 p-2 text-sm";
  return (
    <section className="space-y-3 rounded-xl border border-white/10 p-4">
      <h3 className="font-bold">Captured wall evidence</h3>
      <p className="text-xs text-zinc-400">
        Local wall masks identify candidate photo regions. Only verified tracked points supported by multiple fit views
        establish a surface. Reviewing this evidence does not cut a wall or change collision.
      </p>
      {query.error && (
        <p role="alert" className="text-xs text-red-300">
          {String(query.error)}
        </p>
      )}
      {!study && !query.isLoading && (
        <p className="text-sm text-zinc-400">
          No structural study yet. Ask the agent to inspect captured wall photographs.
        </p>
      )}
      {study && (
        <>
          <label className="block space-y-1 text-xs text-zinc-400">
            Retained wall study
            <select
              aria-label="Retained wall study"
              className={input}
              value={study.structure_id}
              onChange={(event) => setShown(event.target.value)}
            >
              {studies.map((item) => (
                <option key={item.structure_id} value={item.structure_id}>
                  {item.structure_id.slice(-8)} · {item.result?.status || "awaiting local worker"}
                  {item.stale ? " · stale" : ""}
                </option>
              ))}
            </select>
          </label>
          {study.stale && (
            <p className="text-xs text-amber-200">Historical evidence: the active scene or its sources changed.</p>
          )}
          <label className="block space-y-1 text-xs text-zinc-400">
            Wall evidence photograph
            <select
              aria-label="Wall evidence photograph"
              className={input}
              value={camera?.image_id || ""}
              onChange={(event) => setImageId(Number(event.target.value))}
            >
              {study.cameras.map((item) => (
                <option key={item.image_id} value={item.image_id}>
                  {item.image_id} · {item.group} · {item.split}
                </option>
              ))}
            </select>
          </label>
          {camera && (
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              <img
                src={artifact(camera.photo)}
                alt="Captured wall reference photograph"
                className="w-full rounded-lg"
              />
              {view && (
                <img
                  src={artifact(view.overlay, "result")}
                  alt="Wall mask and tracked photo observations"
                  className="w-full rounded-lg"
                />
              )}
            </div>
          )}
          {view && (
            <p className="text-xs text-zinc-400">
              View {view.image_id}: {view.positive_points.toLocaleString()} wall-labelled tracks out of{" "}
              {view.observed_points.toLocaleString()} verified observations. Teal is predicted wall; yellow marks
              tracked points inside it. Blank wall appearance does not imply geometric support.
            </p>
          )}
          {study.result && (
            <>
              <p className="text-sm text-sky-200">
                {study.result.accepted_points.toLocaleString()} multi-view-supported wall points ·{" "}
                {study.result.planes.length} fitted surfaces
              </p>
              {study.result.status === "no-supported-wall" && (
                <p className="text-sm text-amber-200">
                  No wall has enough tracked support for a fitted surface. Do not substitute furniture or silently
                  weaken the fit. Reviewed inferred structure or additional geometric references are needed before an
                  opening.
                </p>
              )}
              {study.result.planes.map((plane) => (
                <div key={plane.plane_id} className="space-y-1 rounded-lg bg-white/5 p-3 text-xs">
                  <h4 className="font-bold">{plane.plane_id}</h4>
                  <p>
                    {(plane.upper_uv[0] - plane.lower_uv[0]).toFixed(2)} ×{" "}
                    {(plane.upper_uv[1] - plane.lower_uv[1]).toFixed(2)} m fitted extent ·{" "}
                    {(plane.rms_m * 1000).toFixed(1)} mm fit RMS · {plane.support_count.toLocaleString()} points
                  </p>
                  <p className="text-zinc-400">
                    {(plane.support_grid_coverage * 100).toFixed(1)}% of 15cm support cells occupied. {plane.scope}.
                  </p>
                  {plane.appearance_checks.map((check) => (
                    <p key={check.image_id} className="text-zinc-400">
                      Check {check.image_id}: {check.semantic_agreeing_points}/{check.observed_points} observed tracks
                      agree with the wall label.
                    </p>
                  ))}
                </div>
              ))}
              <p className="text-xs text-zinc-500">{study.result.check_scope}. Fit RMS is not survey accuracy.</p>
              <a href={artifact("membership.npz", "result")} className="inline-block text-xs text-sky-300 underline">
                Download retained point membership
              </a>
            </>
          )}
        </>
      )}
    </section>
  );
}
