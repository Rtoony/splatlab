globalThis.CaptureAtlasMath = (() => {
  function projectPixel(horizontal, vertical, width, height, yaw, pitch, fieldOfView, roll = 0) {
    const tangent = Math.tan(fieldOfView * Math.PI / 360);
    const screenX = (2 * (horizontal + 0.5) / width - 1) * tangent * width / height;
    const screenY = (1 - 2 * (vertical + 0.5) / height) * tangent;
    const rotatedX = screenX * Math.cos(roll) - screenY * Math.sin(roll);
    const rotatedY = screenX * Math.sin(roll) + screenY * Math.cos(roll);
    const elevatedY = rotatedY * Math.cos(pitch) + Math.sin(pitch);
    const elevatedZ = Math.cos(pitch) - rotatedY * Math.sin(pitch);
    const worldX = rotatedX * Math.cos(yaw) + elevatedZ * Math.sin(yaw);
    const worldZ = elevatedZ * Math.cos(yaw) - rotatedX * Math.sin(yaw);
    const length = Math.hypot(worldX, elevatedY, worldZ);
    return {u: ((Math.atan2(worldX, worldZ) / (2 * Math.PI) + 0.5) % 1 + 1) % 1,
      v: Math.max(0, Math.min(1, 0.5 - Math.asin(elevatedY / length) / Math.PI))};
  }

  function mapFrame(points, width = 600, height = 460, padding = 35) {
    if (!points.length) return null;
    const west = Math.min(...points.map(point => point.east_m));
    const east = Math.max(...points.map(point => point.east_m));
    const south = Math.min(...points.map(point => point.north_m));
    const north = Math.max(...points.map(point => point.north_m));
    const scale = Math.min((width - 2 * padding) / Math.max(1, east - west),
      (height - 2 * padding) / Math.max(1, north - south));
    return {scale, project: point => ({
      horizontal: width / 2 + (point.east_m - (east + west) / 2) * scale,
      vertical: height / 2 - (point.north_m - (north + south) / 2) * scale})};
  }
  return {projectPixel, mapFrame};
})();
