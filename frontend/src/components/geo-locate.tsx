// Locate-in-the-world: pin a splat scene to real WGS84 coordinates by draping
// its top-down footprint over live satellite imagery and dragging / rotating /
// scaling it into place. Saves to POST /jobs/{id}/geo (and optionally the
// survey-lane meters_per_unit via POST /jobs/{id}/scale when the user adjusts
// scale here). Lazy-loaded — Leaflet only ships when the modal opens.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { apiRequest } from "@/lib/api";
import type { SplatGeoAnchor, SplatGeoFootprint, SplatGeoSuggestion, SplatJob } from "@/lib/contracts";
import { Button, Card, Input, SectionLabel } from "@/components/ui";
import { Compass, Crosshair, Download, ExternalLink, Loader2, MapPin, Save, Search, Trash2, X } from "lucide-react";

const M_PER_DEG_LAT = 111320; // WGS84 meters per degree of latitude (good enough at map-align scale)

type NominatimHit = { display_name: string; lat: string; lon: string };

// Rotate the north-up footprint image clockwise by headingDeg onto a canvas
// sized to the rotated axis-aligned extents; same px scale as the source.
function rotateFootprint(img: HTMLImageElement, headingDeg: number): { url: string; w: number; h: number } {
  const th = (headingDeg * Math.PI) / 180;
  const w = Math.max(2, Math.ceil(Math.abs(img.width * Math.cos(th)) + Math.abs(img.height * Math.sin(th))));
  const h = Math.max(2, Math.ceil(Math.abs(img.width * Math.sin(th)) + Math.abs(img.height * Math.cos(th))));
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d")!;
  ctx.translate(w / 2, h / 2);
  ctx.rotate(th);
  ctx.drawImage(img, -img.width / 2, -img.height / 2);
  return { url: canvas.toDataURL("image/png"), w, h };
}

function fmtDeg(v: number): string {
  return v.toFixed(6);
}

const anchorIcon = L.divIcon({
  className: "",
  html: '<div style="width:18px;height:18px;border-radius:50%;background:#22d3ee;border:3px solid #04121a;box-shadow:0 0 10px rgba(34,211,238,.9)"></div>',
  iconSize: [18, 18],
  iconAnchor: [9, 9],
});

export default function GeoLocateModal({ job, onClose }: { job: SplatJob; onClose: () => void }) {
  const jobId = job.job_id;
  const queryClient = useQueryClient();
  const mapDivRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<L.Map | null>(null);
  const markerRef = useRef<L.Marker | null>(null);
  const overlayRef = useRef<L.ImageOverlay | null>(null);
  const footprintImgRef = useRef<HTMLImageElement | null>(null);
  const rotatedRef = useRef<{ url: string; w: number; h: number; heading: number } | null>(null);

  const [footprint, setFootprint] = useState<SplatGeoFootprint | null>(null);
  const [anchor, setAnchor] = useState<{ lat: number; lon: number } | null>(
    job.geo ? { lat: job.geo.lat, lon: job.geo.lon } : null,
  );
  const [heading, setHeading] = useState<number>(job.geo?.heading_deg ?? 0);
  const [altM, setAltM] = useState<string>(job.geo?.alt_m != null ? String(job.geo.alt_m) : "");
  // Working scale (meters per scene unit): starts from the survey calibration.
  const [mpu, setMpu] = useState<number>(job.meters_per_unit ?? 1);
  const [scaleDirty, setScaleDirty] = useState(false);
  const [overwriteScaleOk, setOverwriteScaleOk] = useState(false);
  const [opacity, setOpacity] = useState(0.85);
  const [fromSuggestion, setFromSuggestion] = useState(false);
  const [suggesting, setSuggesting] = useState(false);
  const [suggestNote, setSuggestNote] = useState<string | null>(null);
  const [searchText, setSearchText] = useState("");
  const [searchHits, setSearchHits] = useState<NominatimHit[]>([]);
  const [searching, setSearching] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [savedGeo, setSavedGeo] = useState<SplatGeoAnchor | null>(job.geo ?? null);
  const [confirmClear, setConfirmClear] = useState(false);

  const hadCalibration = job.meters_per_unit != null;
  const stopKeys = { onKeyDown: (e: React.KeyboardEvent) => e.stopPropagation(), onKeyUp: (e: React.KeyboardEvent) => e.stopPropagation() };

  // ── bootstrap: footprint bounds + image ────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    apiRequest<SplatGeoFootprint>(`/api/splat/jobs/${jobId}/geo/footprint`)
      .then((fp) => {
        if (cancelled) return;
        setFootprint(fp);
        if (fp.meters_per_unit != null) setMpu(fp.meters_per_unit);
        if (fp.available && fp.url) {
          const img = new Image();
          img.onload = () => {
            if (!cancelled) {
              footprintImgRef.current = img;
              setFootprint((prev) => (prev ? { ...prev } : prev)); // nudge overlay effect
            }
          };
          img.src = fp.url;
        }
      })
      .catch(() => !cancelled && setFootprint({ job_id: jobId, available: false, reason: "footprint fetch failed" }));
    return () => {
      cancelled = true;
    };
  }, [jobId]);

  // ── map lifecycle ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (!mapDivRef.current || mapRef.current) return;
    const start: [number, number] = job.geo ? [job.geo.lat, job.geo.lon] : [20, 0];
    const map = L.map(mapDivRef.current, { zoomControl: true, worldCopyJump: true }).setView(start, job.geo ? 18 : 2);
    const sat = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
      maxZoom: 21,
      maxNativeZoom: 19,
      attribution: "Imagery © Esri & contributors",
    }).addTo(map);
    const streets = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 21,
      maxNativeZoom: 19,
      attribution: "© OpenStreetMap contributors",
    });
    L.control.layers({ Satellite: sat, Streets: streets }, {}, { position: "topleft" }).addTo(map);
    map.on("click", (e: L.LeafletMouseEvent) => {
      setFromSuggestion(false);
      setAnchor({ lat: e.latlng.lat, lon: e.latlng.lng });
    });
    mapRef.current = map;
    return () => {
      map.remove();
      mapRef.current = null;
      markerRef.current = null;
      overlayRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── marker + footprint overlay follow the working state ───────────────────
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    if (!anchor) {
      markerRef.current?.remove();
      markerRef.current = null;
      overlayRef.current?.remove();
      overlayRef.current = null;
      return;
    }
    if (!markerRef.current) {
      const marker = L.marker([anchor.lat, anchor.lon], { draggable: true, icon: anchorIcon }).addTo(map);
      marker.on("drag", () => {
        const p = marker.getLatLng();
        setFromSuggestion(false);
        setAnchor({ lat: p.lat, lon: p.lng });
      });
      markerRef.current = marker;
    } else {
      const cur = markerRef.current.getLatLng();
      if (Math.abs(cur.lat - anchor.lat) > 1e-9 || Math.abs(cur.lng - anchor.lon) > 1e-9) {
        markerRef.current.setLatLng([anchor.lat, anchor.lon]);
      }
    }

    const img = footprintImgRef.current;
    const fp = footprint;
    if (!img || !fp?.available || fp.units_per_px == null) return;
    if (!rotatedRef.current || rotatedRef.current.heading !== heading) {
      const r = rotateFootprint(img, heading);
      rotatedRef.current = { ...r, heading };
    }
    const rot = rotatedRef.current;
    const mPerPx = fp.units_per_px * mpu;
    const halfWm = (rot.w * mPerPx) / 2;
    const halfHm = (rot.h * mPerPx) / 2;
    const dLat = halfHm / M_PER_DEG_LAT;
    const dLon = halfWm / (M_PER_DEG_LAT * Math.cos((anchor.lat * Math.PI) / 180) || 1e-9);
    const bounds = L.latLngBounds(
      [anchor.lat - dLat, anchor.lon - dLon],
      [anchor.lat + dLat, anchor.lon + dLon],
    );
    if (!overlayRef.current) {
      overlayRef.current = L.imageOverlay(rot.url, bounds, { opacity, interactive: false }).addTo(map);
    } else {
      overlayRef.current.setUrl(rot.url);
      overlayRef.current.setBounds(bounds);
      overlayRef.current.setOpacity(opacity);
    }
    markerRef.current?.setZIndexOffset(1000);
  }, [anchor, heading, mpu, opacity, footprint]);

  // ── actions ────────────────────────────────────────────────────────────────
  const applySuggestion = useCallback(async () => {
    setSuggesting(true);
    setSuggestNote(null);
    try {
      const res = await apiRequest<{ candidates: SplatGeoSuggestion[] }>(`/api/splat/jobs/${jobId}/geo/suggest`);
      const hit = res.candidates[0];
      if (!hit) {
        setSuggestNote("No GPS found in this capture's photos/video.");
        return;
      }
      setAnchor({ lat: hit.lat, lon: hit.lon });
      if (hit.alt_m != null) setAltM(String(Math.round(hit.alt_m * 10) / 10));
      setFromSuggestion(true);
      setSuggestNote(hit.detail);
      mapRef.current?.setView([hit.lat, hit.lon], 19);
    } catch {
      setSuggestNote("GPS lookup failed.");
    } finally {
      setSuggesting(false);
    }
  }, [jobId]);

  async function runSearch() {
    const q = searchText.trim();
    if (!q) return;
    setSearching(true);
    setSearchHits([]);
    try {
      const res = await fetch(`https://nominatim.openstreetmap.org/search?format=jsonv2&limit=5&q=${encodeURIComponent(q)}`);
      setSearchHits(((await res.json()) as NominatimHit[]) ?? []);
    } catch {
      setSearchHits([]);
    } finally {
      setSearching(false);
    }
  }

  function goToHit(hit: NominatimHit) {
    const lat = parseFloat(hit.lat);
    const lon = parseFloat(hit.lon);
    mapRef.current?.setView([lat, lon], 18);
    if (!anchor) {
      setFromSuggestion(false);
      setAnchor({ lat, lon });
    }
    setSearchHits([]);
  }

  async function save() {
    if (!anchor) return;
    setSaving(true);
    setSaveError(null);
    try {
      const alt = altM.trim() === "" ? null : Number(altM);
      if (alt != null && !Number.isFinite(alt)) throw new Error("Altitude must be a number (meters).");
      const body: { geo: Record<string, unknown> } = {
        geo: {
          lat: anchor.lat,
          lon: anchor.lon,
          alt_m: alt,
          heading_deg: heading,
          anchor_scene: footprint?.center ?? null,
          source: fromSuggestion ? "exif" : "map",
        },
      };
      const res = await apiRequest<{ geo: SplatGeoAnchor }>(`/api/splat/jobs/${jobId}/geo`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      // Scale writes are deliberate-only: never on the default 1.0 assumption,
      // and never over an existing survey calibration without the checkbox.
      if (scaleDirty && (!hadCalibration || overwriteScaleOk)) {
        await apiRequest(`/api/splat/jobs/${jobId}/scale`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ meters_per_unit: mpu }),
        });
      }
      setSavedGeo(res.geo);
      queryClient.invalidateQueries({ queryKey: ["status"] });
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : "Save failed.");
    } finally {
      setSaving(false);
    }
  }

  async function clearLocation() {
    if (!confirmClear) {
      setConfirmClear(true);
      setTimeout(() => setConfirmClear(false), 4000);
      return;
    }
    setConfirmClear(false);
    setSaving(true);
    try {
      await apiRequest(`/api/splat/jobs/${jobId}/geo`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ geo: null }),
      });
      setSavedGeo(null);
      setAnchor(null);
      queryClient.invalidateQueries({ queryKey: ["status"] });
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : "Clear failed.");
    } finally {
      setSaving(false);
    }
  }

  const footprintMeters = useMemo(() => {
    if (!footprint?.available || footprint.units_per_px == null || footprint.width == null) return null;
    return {
      w: footprint.width * footprint.units_per_px * mpu,
      h: (footprint.height ?? 0) * footprint.units_per_px * mpu,
    };
  }, [footprint, mpu]);

  return (
    <div className="fixed inset-0 z-50 flex bg-black/70 backdrop-blur-sm">
      <div className="relative m-2 flex min-w-0 flex-1 overflow-hidden rounded-2xl border border-white/10 bg-[#05070d] sm:m-4">
        {/* map */}
        <div ref={mapDivRef} className="h-full min-w-0 flex-1" />

        {/* controls */}
        <div className="flex w-[21rem] max-w-[45vw] shrink-0 flex-col gap-2 overflow-y-auto border-l border-white/10 bg-[#070b14]/95 p-3">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-1.5">
              <MapPin className="h-4 w-4 text-cyan-300" />
              <SectionLabel>Locate in the world</SectionLabel>
            </div>
            <button type="button" onClick={onClose} className="rounded-full p-1 text-zinc-400 transition hover:bg-white/10 hover:text-zinc-100" title="Close">
              <X className="h-4 w-4" />
            </button>
          </div>
          <p className="text-[11px] leading-snug text-zinc-500">
            Click the map (or search / use photo GPS) to drop the anchor, then drag it and rotate the footprint until it lines
            up with the satellite imagery.
          </p>

          {/* search */}
          <form
            className="flex items-center gap-1.5"
            onSubmit={(e) => {
              e.preventDefault();
              runSearch();
            }}
          >
            <Input value={searchText} onChange={(e) => setSearchText(e.target.value)} placeholder="Search a place or address…" autoComplete="off" {...stopKeys} />
            <Button type="submit" size="sm" variant="outline" disabled={searching || !searchText.trim()} className="shrink-0">
              {searching ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Search className="h-3.5 w-3.5" />}
            </Button>
          </form>
          {searchHits.length > 0 && (
            <Card className="border-white/10 bg-white/5 p-1">
              {searchHits.map((hit, i) => (
                <button key={i} type="button" onClick={() => goToHit(hit)} className="block w-full truncate rounded-md px-2 py-1 text-left text-xs text-zinc-300 hover:bg-white/10" title={hit.display_name}>
                  {hit.display_name}
                </button>
              ))}
            </Card>
          )}

          <Button type="button" variant="outline" size="sm" onClick={applySuggestion} disabled={suggesting}>
            {suggesting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Crosshair className="h-3.5 w-3.5" />} Use photo GPS
          </Button>
          {suggestNote && <p className="text-[11px] text-cyan-200/80">{suggestNote}</p>}

          {/* anchor readout */}
          <Card className="space-y-1 border-white/10 bg-white/5 p-2 text-xs text-zinc-300">
            {anchor ? (
              <>
                <div className="flex justify-between gap-2">
                  <span className="text-zinc-500">Anchor</span>
                  <span className="tabular-nums">{fmtDeg(anchor.lat)}, {fmtDeg(anchor.lon)}</span>
                </div>
                <div className="flex items-center justify-between gap-2">
                  <span className="text-zinc-500">Altitude (m)</span>
                  <input
                    value={altM}
                    onChange={(e) => setAltM(e.target.value)}
                    placeholder="optional"
                    className="w-24 rounded-md border border-white/10 bg-white/5 px-2 py-0.5 text-right text-xs text-zinc-200"
                    {...stopKeys}
                  />
                </div>
              </>
            ) : (
              <p className="text-zinc-500">No anchor yet — click the map to place one.</p>
            )}
          </Card>

          {/* heading */}
          <Card className="space-y-1.5 border-white/10 bg-white/5 p-2">
            <div className="flex items-center justify-between text-xs">
              <span className="flex items-center gap-1 text-zinc-400"><Compass className="h-3.5 w-3.5" /> Heading</span>
              <span className="tabular-nums text-zinc-200">{heading.toFixed(1)}° true</span>
            </div>
            <input type="range" min={0} max={359.5} step={0.5} value={heading} onChange={(e) => setHeading(Number(e.target.value))} className="w-full accent-cyan-300" />
            <p className="text-[10px] leading-snug text-zinc-500">Rotates the footprint clockwise until it matches the imagery.</p>
          </Card>

          {/* scale */}
          <Card className="space-y-1.5 border-white/10 bg-white/5 p-2">
            <div className="flex items-center justify-between text-xs">
              <span className="text-zinc-400">Scale (m / scene unit)</span>
              <input
                value={Number.isFinite(mpu) ? String(Math.round(mpu * 10000) / 10000) : ""}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  if (Number.isFinite(v) && v > 0) {
                    setMpu(v);
                    setScaleDirty(true);
                  }
                }}
                className="w-24 rounded-md border border-white/10 bg-white/5 px-2 py-0.5 text-right text-xs text-zinc-200"
                {...stopKeys}
              />
            </div>
            <input
              type="range"
              min={-2}
              max={2}
              step={0.01}
              value={Math.log10(Math.max(1e-6, mpu))}
              onChange={(e) => {
                setMpu(Math.pow(10, Number(e.target.value)));
                setScaleDirty(true);
              }}
              className="w-full accent-cyan-300"
            />
            {footprintMeters && (
              <p className="text-[10px] tabular-nums text-zinc-500">
                Footprint ≈ {footprintMeters.w.toFixed(1)} m × {footprintMeters.h.toFixed(1)} m
              </p>
            )}
            {!hadCalibration && !scaleDirty && (
              <p className="text-[10px] leading-snug text-amber-200/70">Scene has no scale calibration — stretch the footprint to match the imagery, or measure a reference in the viewer.</p>
            )}
            {hadCalibration && scaleDirty && (
              <label className="flex items-start gap-1.5 text-[10px] leading-snug text-amber-200/80">
                <input type="checkbox" checked={overwriteScaleOk} onChange={(e) => setOverwriteScaleOk(e.target.checked)} className="mt-0.5 accent-amber-300" />
                Overwrite the measured scale calibration ({job.meters_per_unit} m/unit) with this map-eyeballed value on save.
              </label>
            )}
          </Card>

          {/* overlay opacity + availability */}
          {footprint?.available ? (
            <Card className="space-y-1 border-white/10 bg-white/5 p-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-zinc-400">Footprint opacity</span>
                <span className="tabular-nums text-zinc-200">{Math.round(opacity * 100)}%</span>
              </div>
              <input type="range" min={0.1} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} className="w-full accent-cyan-300" />
            </Card>
          ) : footprint ? (
            <p className="text-[11px] leading-snug text-zinc-500">No footprint overlay ({footprint.reason ?? "unavailable"}) — you can still pin the anchor and heading.</p>
          ) : (
            <p className="flex items-center gap-1.5 text-[11px] text-zinc-500"><Loader2 className="h-3 w-3 animate-spin" /> Loading footprint…</p>
          )}

          {/* save / clear */}
          <div className="mt-auto space-y-2 pt-2">
            {saveError && <p className="text-[11px] text-red-300">{saveError}</p>}
            <Button type="button" onClick={save} disabled={!anchor || saving} className="w-full">
              {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />} Save location
            </Button>
            {savedGeo && (
              <>
                <p className="text-center text-[11px] text-emerald-300/80">
                  Located · {fmtDeg(savedGeo.lat)}, {fmtDeg(savedGeo.lon)} · {savedGeo.heading_deg.toFixed(1)}°
                </p>
                <div className="grid grid-cols-3 gap-1.5">
                  <a
                    href={`https://www.google.com/maps?q=${savedGeo.lat},${savedGeo.lon}`}
                    target="_blank"
                    rel="noreferrer"
                    className="flex items-center justify-center gap-1 rounded-lg border border-white/15 bg-white/5 px-2 py-1.5 text-[11px] font-semibold text-zinc-200 hover:bg-white/10"
                  >
                    <ExternalLink className="h-3 w-3" /> Maps
                  </a>
                  <a href={`/api/splat/jobs/${jobId}/geo/export?fmt=kml`} className="flex items-center justify-center gap-1 rounded-lg border border-white/15 bg-white/5 px-2 py-1.5 text-[11px] font-semibold text-zinc-200 hover:bg-white/10">
                    <Download className="h-3 w-3" /> KML
                  </a>
                  <a href={`/api/splat/jobs/${jobId}/geo/export?fmt=geojson`} className="flex items-center justify-center gap-1 rounded-lg border border-white/15 bg-white/5 px-2 py-1.5 text-[11px] font-semibold text-zinc-200 hover:bg-white/10">
                    <Download className="h-3 w-3" /> GeoJSON
                  </a>
                </div>
                <button type="button" onClick={clearLocation} className={`flex w-full items-center justify-center gap-1 rounded-lg px-2 py-1.5 text-[11px] font-semibold transition ${confirmClear ? "bg-red-400/20 text-red-200" : "text-zinc-500 hover:bg-white/5 hover:text-zinc-300"}`}>
                  <Trash2 className="h-3 w-3" /> {confirmClear ? "Click again to remove the location" : "Clear location"}
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
