export type SupportFeature = {
  point_id: string;
  pixel: [number, number];
  world: [number, number, number];
  reprojection_px: number;
};
export type SupportAnchor = { point_id: string; image_id: number; photo_sha256: string };
export type SupportAnchors = { evidence_sha256: string; points: SupportAnchor[] };

export function nearestSupportFeature(
  features: SupportFeature[],
  pixel: [number, number],
  pixelsPerScreenPixel: number,
) {
  if (!Number.isFinite(pixelsPerScreenPixel) || pixelsPerScreenPixel <= 0 || !pixel.every(Number.isFinite)) return null;
  let nearest: SupportFeature | null = null;
  let maximum = 12 * pixelsPerScreenPixel;
  for (const feature of features) {
    const distance = Math.hypot(feature.pixel[0] - pixel[0], feature.pixel[1] - pixel[1]);
    if (distance < maximum) {
      nearest = feature;
      maximum = distance;
    }
  }
  return nearest;
}
