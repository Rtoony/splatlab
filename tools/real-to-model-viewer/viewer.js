import * as THREE from "three";
import {SparkRenderer, SplatFileType, SplatMesh} from "@sparkjsdev/spark";
import {OrbitControls} from "./vendor/OrbitControls.js";
import {GLTFLoader} from "./vendor/loaders/GLTFLoader.js";
import {captureSummary} from "./capture-summary.js";

const status = document.querySelector("#status");
const host = document.querySelector("#viewport");
try {
  const response = await fetch("evaluation/receipt.json");
  if (!response.ok) throw new Error(`Evaluation metadata: HTTP ${response.status}`);
  const receipt = await response.json();
  if (receipt.status !== "evaluated-needs-review") throw new Error("This evaluation is not ready to review");
  const summary = captureSummary(receipt);
  document.querySelector("#summary").textContent = `${receipt.gaussians.rows.toLocaleString()} trained Gaussians from real dual-lens video · ${receipt.training_iterations.toLocaleString()} iterations · ${summary.label}`;
  document.querySelector("#comparison-link").textContent = `All ${summary.viewCount} source/render/error comparisons`;
  const isSurfacePreview = receipt.method === "gsplat-2dgs";
  if (isSurfacePreview) {
    document.querySelector("#summary").textContent = `${receipt.gaussians.rows.toLocaleString()} trained 2D surfaces · ${receipt.training_iterations.toLocaleString()} iterations · ${summary.label} Native 2DGS renders below.`;
    document.querySelector("#representation-notice").textContent = "This interactive view uses a thin-3D approximation of trained 2D surfaces, not the native 2DGS renderer used for the validation images. It can look different at edges or oblique angles. Surface depth is inferred, not measured. Units, world up and architectural registration remain unverified; this is not your accepted condo model.";
    document.querySelector("#splat-download").textContent = "Thin-3D preview PLY";
  }
  const renderer = new THREE.WebGLRenderer({antialias: false, preserveDrawingBuffer: true});
  renderer.domElement.addEventListener("webglcontextlost", () => {
    renderer.setAnimationLoop(null);
    host.dataset.error = "WebGL context lost";
    status.textContent = "Live 3D lost its GPU context. The saved camera tour, comparisons and downloads remain available. Reload live 3D when the GPU is ready.";
    status.classList.add("error");
    document.querySelector("#reload-live").hidden = false;
  });
  document.querySelector("#reload-live").addEventListener("click", () => window.location.reload());
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
  renderer.setClearColor(0x080d13);
  host.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(100, 1, .005, 1000);
  let redrawUntil = 0;
  const spark = new SparkRenderer({renderer, onDirty: () => { redrawUntil = performance.now() + 600; }});
  scene.add(spark);
  const splat = new SplatMesh({url: "splat.ply", fileType: SplatFileType.PLY});
  scene.add(splat);
  const surfaceModels = new Map();
  const representation = document.querySelector("#representation");
  host.dataset.representation = "splat";
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = false;
  controls.maxDistance = 80;
  controls.minDistance = .05;
  const selector = document.querySelector("#view");
  for (const [index, view] of receipt.validation.entries()) {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = view.image;
    selector.appendChild(option);
  }
  selector.value = "1";
  let ready = false;
  const changed = () => {
    redrawUntil = performance.now() + 600;
    host.dataset.camera = JSON.stringify([...camera.position.toArray(), ...camera.quaternion.toArray()]);
  };
  const reset = () => {
    const index = Number(selector.value);
    const view = receipt.validation[index];
    const matrix = new THREE.Matrix4().set(...view.camera_to_world_opengl.flat(), 0, 0, 0, 1);
    matrix.decompose(camera.position, camera.quaternion, new THREE.Vector3());
    camera.up.set(0, 1, 0).applyQuaternion(camera.quaternion);
    camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(view.height / (2 * view.fy)));
    camera.aspect = view.width / view.height;
    camera.updateProjectionMatrix();
    const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
    controls.target.copy(camera.position).addScaledVector(forward, 3);
    controls.update();
    document.querySelector("#reference").src = `evaluation/renders/${String(index).padStart(3, "0")}-reference.png`;
    host.dataset.selectedView = view.image;
    changed();
  };
  controls.addEventListener("change", changed);
  selector.addEventListener("change", reset);
  document.querySelector("#reset").addEventListener("click", reset);
  host.addEventListener("keydown", event => {
    const offset = new THREE.Vector3();
    if (event.key === "ArrowLeft") offset.x = -.08;
    else if (event.key === "ArrowRight") offset.x = .08;
    else if (event.key === "ArrowUp") offset.z = -.08;
    else if (event.key === "ArrowDown") offset.z = .08;
    else return;
    event.preventDefault();
    offset.applyQuaternion(camera.quaternion);
    camera.position.add(offset);
    controls.target.add(offset);
    controls.update();
    changed();
  });
  new ResizeObserver(() => {
    renderer.setSize(host.clientWidth, host.clientHeight, false);
    changed();
  }).observe(host);
  document.addEventListener("visibilitychange", changed);
  reset();
  await splat.initialized;
  ready = true;
  host.dataset.splats = String(splat.numSplats);
  host.dataset.ready = "true";
  status.textContent = `Live ${isSurfacePreview ? "thin-3D approximation" : "3D"} ready: ${splat.numSplats.toLocaleString()} Gaussians. Rendering pauses when idle or this tab is hidden.`;
  changed();
  renderer.setAnimationLoop(() => {
    if (ready && !document.hidden && performance.now() < redrawUntil) {
      renderer.render(scene, camera);
      host.dataset.renderCount = String(Number(host.dataset.renderCount || 0) + 1);
    }
  });
  const delivery = await fetch("receipt.json").then(value => value.json());
  const meshSources = {};
  if (delivery.blender_scene) meshSources["mesh"] = {url: "blender/inferred-surface.glb", label: `${receipt.training_iterations.toLocaleString()}-step extracted mesh`};
  if (delivery.blender_alternative) meshSources["mesh-alternative"] = {url: "blender-alternative/inferred-surface.glb", label: "3,000-step extracted mesh"};
  for (const handoff of delivery.mesh_deliveries || []) {
    if (meshSources[handoff.representation] && handoff.mesh_href) meshSources[handoff.representation].url = handoff.mesh_href;
  }
  for (const [key, source] of Object.entries(meshSources)) {
    const option = document.createElement("option");
    option.value = key;
    option.textContent = source.label;
    representation.appendChild(option);
    document.querySelector("#representation-bar").hidden = false;
  }
  representation.addEventListener("change", async () => {
    const selected = representation.value;
    representation.disabled = true;
    try {
      if (selected !== "splat" && !surfaceModels.has(selected)) {
        status.textContent = "Loading the actual extracted mesh; preserving the current source viewpoint…";
        const loaded = await new GLTFLoader().loadAsync(meshSources[selected].url);
        loaded.scene.traverse(object => {
          if (!object.isMesh) return;
          if (object.material.isMeshBasicMaterial && object.geometry.attributes.color) return;
          for (const material of Array.isArray(object.material) ? object.material : [object.material]) material.dispose();
          object.material = new THREE.MeshBasicMaterial({vertexColors: true, side: THREE.DoubleSide});
        });
        scene.add(loaded.scene);
        surfaceModels.set(selected, loaded.scene);
      }
      for (const [key, model] of surfaceModels) model.visible = key === selected;
      splat.visible = spark.visible = selected === "splat";
      host.dataset.representation = selected;
      const label = selected === "splat" ? "Live radiance preview" : meshSources[selected].label;
      document.querySelector("#live-caption").textContent = `${label} · same source coordinates and camera · drag to inspect`;
      status.textContent = selected === "splat" ? "Live radiance preview. Rendering pauses when idle." : "Actual extracted triangle mesh with unlit captured vertex colors, not a splat or photo overlay. Holes and fragmentation are reconstruction defects; no architectural acceptance.";
      changed();
    } catch (error) {
      status.textContent = `Mesh loading failed: ${error.message}. The previous view and saved downloads remain available.`;
      representation.value = host.dataset.representation;
    } finally {
      representation.disabled = false;
    }
  });
  if (delivery.blender_scene) document.querySelector("#surface-link").hidden = false;
  if (delivery.blender_alternative) {
    document.querySelector("#alternative-surface-link").hidden = false;
    document.querySelector("#candidate-tradeoff").hidden = false;
    document.querySelector("#surface-link").textContent = "6,000-step visual candidate — Blender";
  }
  if (delivery.native_parameters) document.querySelector("#native-download").hidden = false;
  if (delivery.native_tour) {
    document.querySelector("#native-tour").hidden = false;
    document.querySelector("#tour-video").src = "tour/native-camera-tour.webm";
    document.querySelector("#tour-video").poster = "tour/tour-060.png";
  }
  for (const comparison of delivery.geometry_comparisons || []) {
    const link = document.createElement("a");
    link.href = comparison.href;
    link.textContent = comparison.label;
    document.querySelector("#geometry-links").appendChild(link);
  }
  for (const handoff of delivery.mesh_deliveries || []) {
    document.querySelector("#mesh-deliveries").hidden = false;
    const link = document.createElement("a");
    link.href = handoff.href;
    link.textContent = handoff.label;
    document.querySelector("#mesh-delivery-links").appendChild(link);
  }
  window.addEventListener("pagehide", () => {
    renderer.setAnimationLoop(null);
    controls.dispose();
    splat.dispose();
    spark.dispose();
    for (const model of surfaceModels.values()) model.traverse(object => {
      object.geometry?.dispose();
      for (const material of Array.isArray(object.material) ? object.material : object.material ? [object.material] : []) material.dispose();
    });
    renderer.dispose();
  }, {once: true});
} catch (error) {
  status.textContent = `Live viewer unavailable: ${error.message}. The saved source/render comparisons below remain usable.`;
  status.classList.add("error");
  host.dataset.error = error.message;
}
