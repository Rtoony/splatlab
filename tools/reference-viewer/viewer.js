(() => {
  const data = JSON.parse(document.getElementById("reference-data").textContent);
  const math = globalThis.ReferenceMath;
  const camera = data.cameras;
  const frames = camera.frames;
  const byId = identifier => document.getElementById(identifier);
  const orbit = byId("orbit"), context = orbit.getContext("2d");
  const projected = byId("projected"), overlayContext = projected.getContext("2d");
  const source = byId("source"), photo = byId("photo");
  projected.width = camera.w;
  projected.height = camera.h;
  byId("photo-stack").style.aspectRatio = `${camera.w} / ${camera.h}`;
  const quantile = (values, amount) => [...values].sort((first, second) => first - second)[Math.floor((values.length - 1) * amount)];
  const target = [0, 1, 2].map(axis => quantile(data.points.map(point => point[axis]), .5));
  const distances = data.points.map(point => Math.hypot(...math.subtract(point, target)));
  const radius = Math.max(.1, quantile(distances, .95));
  const up = math.normalize([0, 1, 2].map(axis => frames.reduce((total, frame) => total + frame.transform_matrix[axis][1], 0)));
  const referenceAxis = Math.abs(math.dot(up, [0, 0, 1])) < .9 ? [0, 0, 1] : [1, 0, 0];
  const horizontal = math.normalize(math.cross(up, referenceAxis));
  const backward = math.normalize(math.cross(horizontal, up));
  let selected = Math.max(0, frames.findIndex(frame => frame.source_image === "lens-0/frame-000150.jpg" && frame.virtual_yaw_deg === 35));
  let yaw = .7, pitch = .35, zoom = 2.3, drag = null;
  for (const [index, frame] of frames.entries()) {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${frame.source_image} · ${frame.virtual_yaw_deg >= 0 ? "+" : ""}${frame.virtual_yaw_deg}° virtual direction`;
    source.append(option);
  }
  for (const text of [`${data.points.length.toLocaleString()} reconstructed points`, `${data.physical_source_images} posed raw images`, `${frames.length} virtual camera directions`, `${data.unlocalized_views_omitted} held-out crops still unlocalized`]) {
    const badge = document.createElement("span");
    badge.textContent = text;
    byId("counts").append(badge);
  }
  for (const text of data.limitations) {
    const item = document.createElement("li");
    item.textContent = text;
    byId("limitations").append(item);
  }
  byId("provenance").textContent = `Frozen reference receipt SHA-256: ${data.source_receipt_sha256}. Local, on-demand CPU canvas; no cloud calls.`;
  function renderOrbit() {
    const direction = math.add(math.scale(up, Math.sin(pitch)), math.scale(math.add(math.scale(backward, Math.cos(yaw)), math.scale(horizontal, Math.sin(yaw))), Math.cos(pitch)));
    const eye = math.add(target, math.scale(direction, radius * zoom));
    const project = point => math.orbitProjection(point, eye, target, up, orbit.width, orbit.height);
    context.fillStyle = "#0a121b";
    context.fillRect(0, 0, orbit.width, orbit.height);
    const points = data.points.map((point, index) => ({pixel: project(point), color: data.colors[index]})).filter(point => point.pixel).sort((first, second) => second.pixel[2] - first.pixel[2]);
    for (const point of points) {
      context.fillStyle = `rgb(${point.color.join(",")})`;
      const size = Math.max(1.8, Math.min(4, radius / point.pixel[2] * 5));
      context.fillRect(point.pixel[0] - size / 2, point.pixel[1] - size / 2, size, size);
    }
    let drawnCameras = 0;
    const drawCamera = (frame, active) => {
      const matrix = frame.transform_matrix;
      const origin = math.centre(matrix);
      const distance = radius * (active ? .12 : .045);
      const extentX = distance * camera.w / (2 * camera.fl_x), extentY = distance * camera.h / (2 * camera.fl_y);
      const corners = [[-extentX, extentY, -distance], [extentX, extentY, -distance], [extentX, -extentY, -distance], [-extentX, -extentY, -distance]].map(point => math.transform(matrix, point));
      context.strokeStyle = active ? "#ffc271" : "#398cae";
      context.lineWidth = active ? 2.5 : .9;
      const segment = (first, second) => {
        const start = project(first), end = project(second);
        if (!start || !end) return;
        context.beginPath(); context.moveTo(start[0], start[1]); context.lineTo(end[0], end[1]); context.stroke();
      };
      corners.forEach((corner, index) => {segment(origin, corner); segment(corner, corners[(index + 1) % 4]);});
      drawnCameras += 1;
    };
    if (byId("show-cameras").checked) {
      frames.forEach((frame, index) => {if (index !== selected) drawCamera(frame, false);});
      drawCamera(frames[selected], true);
    }
    context.fillStyle = "#adc2d4"; context.font = "16px system-ui";
    context.fillText("Sparse geometry · display orientation is not measured gravity", 18, 28);
    orbit.dataset.selectedFrame = frames[selected].file_path;
    orbit.dataset.drawnCameras = String(drawnCameras);
    orbit.dataset.drawnPoints = String(points.length);
  }
  function renderProjection() {
    overlayContext.clearRect(0, 0, projected.width, projected.height);
    let count = 0;
    for (const point of data.points) {
      const pixel = math.sourceProjection(point, frames[selected].transform_matrix, camera);
      if (!pixel) continue;
      count += 1;
      if (byId("overlay").checked) {
        overlayContext.beginPath(); overlayContext.arc(pixel[0], pixel[1], 2.3, 0, Math.PI * 2);
        overlayContext.fillStyle = "rgba(72,255,201,.85)"; overlayContext.fill();
        overlayContext.strokeStyle = "#142a33"; overlayContext.lineWidth = .8; overlayContext.stroke();
      }
    }
    projected.dataset.pointCount = String(count);
    byId("projection-count").textContent = `${count} sparse points project inside this image; visibility/occlusion has not been certified.`;
  }
  function select(index) {
    selected = Math.max(0, Math.min(frames.length - 1, index));
    const frame = frames[selected];
    source.value = String(selected);
    photo.src = frame.file_path;
    photo.alt = `Actual rectified ${frame.source_image}, yaw ${frame.virtual_yaw_deg} degrees`;
    byId("source-counter").textContent = `${selected + 1} / ${frames.length}`;
    byId("source-details").textContent = `Physical source: ${frame.source_image} · PTS ${frame.pts} × ${frame.time_base} · training pose only · photo SHA-256 ${frame.sha256}`;
    byId("previous").disabled = selected === 0;
    byId("next").disabled = selected === frames.length - 1;
    renderOrbit(); renderProjection();
  }
  source.addEventListener("change", () => select(Number(source.value)));
  byId("previous").addEventListener("click", () => select(selected - 1));
  byId("next").addEventListener("click", () => select(selected + 1));
  byId("overlay").addEventListener("change", renderProjection);
  byId("show-cameras").addEventListener("change", renderOrbit);
  byId("reset").addEventListener("click", () => {yaw = .7; pitch = .35; zoom = 2.3; renderOrbit();});
  orbit.addEventListener("pointerdown", event => {drag = [event.clientX, event.clientY]; orbit.setPointerCapture(event.pointerId); orbit.focus();});
  orbit.addEventListener("pointermove", event => {
    if (!drag) return;
    yaw -= (event.clientX - drag[0]) * .008;
    pitch = Math.max(-1.3, Math.min(1.3, pitch + (event.clientY - drag[1]) * .008));
    drag = [event.clientX, event.clientY]; renderOrbit();
  });
  for (const event of ["pointerup", "pointercancel", "lostpointercapture"]) orbit.addEventListener(event, () => {drag = null;});
  orbit.addEventListener("wheel", event => {event.preventDefault(); zoom = Math.max(.35, Math.min(10, zoom * Math.exp(event.deltaY * .001))); renderOrbit();}, {passive: false});
  orbit.addEventListener("keydown", event => {
    const changes = {ArrowLeft: [-.1, 0, 1], ArrowRight: [.1, 0, 1], ArrowUp: [0, .1, 1], ArrowDown: [0, -.1, 1], "+": [0, 0, .9], "-": [0, 0, 1.1]};
    if (!changes[event.key]) return;
    event.preventDefault(); const change = changes[event.key]; yaw += change[0]; pitch = Math.max(-1.3, Math.min(1.3, pitch + change[1])); zoom = Math.max(.35, Math.min(10, zoom * change[2])); renderOrbit();
  });
  select(selected);
})();
