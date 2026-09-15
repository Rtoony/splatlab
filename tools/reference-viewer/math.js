(function (root) {
  const dot = (first, second) => first.reduce((sum, value, index) => sum + value * second[index], 0);
  const subtract = (first, second) => first.map((value, index) => value - second[index]);
  const add = (first, second) => first.map((value, index) => value + second[index]);
  const scale = (vector, amount) => vector.map(value => value * amount);
  const normalize = vector => scale(vector, 1 / Math.max(1e-12, Math.hypot(...vector)));
  const cross = (first, second) => [first[1] * second[2] - first[2] * second[1], first[2] * second[0] - first[0] * second[2], first[0] * second[1] - first[1] * second[0]];
  const centre = matrix => matrix.slice(0, 3).map(row => row[3]);
  const transform = (matrix, point) => matrix.slice(0, 3).map(row => dot(row.slice(0, 3), point) + row[3]);
  function sourceProjection(point, matrix, camera) {
    const delta = subtract(point, centre(matrix));
    const local = [0, 1, 2].map(column => dot(delta, matrix.slice(0, 3).map(row => row[column])));
    const depth = -local[2];
    if (depth <= 1e-8) return null;
    const horizontal = camera.fl_x * local[0] / depth + camera.cx;
    const vertical = -camera.fl_y * local[1] / depth + camera.cy;
    if (horizontal < 0 || vertical < 0 || horizontal >= camera.w || vertical >= camera.h) return null;
    return [horizontal, vertical, depth];
  }
  function orbitProjection(point, eye, target, up, width, height) {
    const forward = normalize(subtract(target, eye));
    const right = normalize(cross(forward, up));
    const vertical = cross(right, forward);
    const delta = subtract(point, eye);
    const depth = dot(delta, forward);
    if (depth <= 1e-6) return null;
    const focal = Math.min(width, height) * .9;
    return [width / 2 + focal * dot(delta, right) / depth, height / 2 - focal * dot(delta, vertical) / depth, depth];
  }
  root.ReferenceMath = {dot, subtract, add, scale, normalize, cross, centre, transform, sourceProjection, orbitProjection};
})(globalThis);
