// Shared viewer prop contract — originally exported by the classic mkkellogg
// viewer (deleted in professionalization wave 7.1); the Spark viewer and the
// /view page speak these shapes.
export type ViewerPoint = { point: [number, number, number]; radius: number };
export type ViewerOverlay = { matches: ViewerPoint[]; active: number; label: string } | null;
// A named, colored group of 3D points to highlight+label all at once (inventory legend).
export type ViewerHighlight = { label: string; color: string; points: [number, number, number][] };
export type ViewerCameraPose = {
  index: number;
  image_name: string;
  position: [number, number, number];
  forward: [number, number, number];
  up: [number, number, number];
  right: [number, number, number];
  fov_y_degrees?: number | null;
};
export type ViewerCameraOverlay = { cameras: ViewerCameraPose[]; displayScale: number; frame: "viewer" | "source" } | null;
export type ViewerCameraViewTarget = { camera: ViewerCameraPose; token: number; distance?: number } | null;
export type ViewerCameraNodeTarget = { camera: ViewerCameraPose; token: number; distance?: number } | null;
