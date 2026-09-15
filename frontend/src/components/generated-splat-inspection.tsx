import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { SparkRenderer, SplatFileType, SplatMesh } from "@sparkjsdev/spark";

export function GeneratedSplatInspection({ splatUrl, meshUrl }: { splatUrl: string; meshUrl: string }) {
  const host = useRef<HTMLDivElement>(null);
  const control = useRef<((mode: string) => void) | null>(null);
  const [opened, setOpened] = useState(false);
  const [mode, setMode] = useState("splat");
  const [message, setMessage] = useState("");
  const [ready, setReady] = useState(false);
  useEffect(() => {
    if (!opened || !host.current) return;
    const container = host.current;
    let disposed = false;
    let renderer: THREE.WebGLRenderer | null = null;
    let spark: SparkRenderer | null = null;
    let splat: SplatMesh | null = null;
    let mesh: THREE.Group | null = null;
    let orbit: OrbitControls | null = null;
    let frame = 0;
    let pending = false;
    let dirty = false;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(40, 1.5, 0.001, 1000);
    const draw = () => {
      dirty = true;
      if (pending || disposed) return;
      pending = true;
      frame = requestAnimationFrame(async () => {
        dirty = false;
        try {
          await spark?.update({ scene, camera });
          if (!disposed && renderer) {
            renderer.render(scene, camera);
            container.dataset.renderCount = String(Number(container.dataset.renderCount || 0) + 1);
          }
        } catch (error) {
          if (!disposed) setMessage(error instanceof Error ? error.message : String(error));
        } finally {
          pending = false;
          if (dirty && !disposed) draw();
        }
      });
    };
    const resize = new ResizeObserver(() => {
      if (!renderer || disposed) return;
      const width = Math.max(1, container.clientWidth);
      renderer.setSize(width, 320, false);
      camera.aspect = width / 320;
      camera.updateProjectionMatrix();
      draw();
    });
    const disposeMesh = (root: THREE.Group) =>
      root.traverse((object) => {
        if (!(object instanceof THREE.Mesh)) return;
        object.geometry.dispose();
        for (const material of Array.isArray(object.material) ? object.material : [object.material]) material.dispose();
      });
    void (async () => {
      try {
        renderer = new THREE.WebGLRenderer({ antialias: false, preserveDrawingBuffer: true });
        renderer.setPixelRatio(1);
        renderer.setClearColor(0x08090b);
        renderer.domElement.setAttribute("aria-label", "Generated mesh and splat 3D comparison");
        renderer.domElement.style.width = "100%";
        renderer.domElement.style.height = "320px";
        container.append(renderer.domElement);
        spark = new SparkRenderer({ renderer });
        spark.autoUpdate = false;
        scene.add(spark);
        splat = new SplatMesh({ url: splatUrl, fileType: SplatFileType.PLY });
        scene.add(splat);
        const [loaded] = await Promise.all([new GLTFLoader().loadAsync(meshUrl), splat.initialized]);
        if (disposed) {
          disposeMesh(loaded.scene);
          return;
        }
        mesh = loaded.scene;
        mesh.traverse((object) => {
          if (!(object instanceof THREE.Mesh)) return;
          const original = Array.isArray(object.material) ? object.material[0] : object.material;
          object.material = new THREE.MeshBasicMaterial({
            color: original.color,
            vertexColors: original.vertexColors,
            side: THREE.DoubleSide,
          });
          original.dispose();
        });
        mesh.visible = false;
        scene.add(mesh);
        if (disposed) return;
        const bounds = splat.getBoundingBox().union(new THREE.Box3().setFromObject(mesh));
        if (bounds.isEmpty()) throw new Error("Generated comparison has no geometry");
        const center = bounds.getCenter(new THREE.Vector3());
        const distance = bounds.getSize(new THREE.Vector3()).length() * 1.3;
        camera.position.copy(center).add(new THREE.Vector3(1, 0.7, 1).normalize().multiplyScalar(distance));
        orbit = new OrbitControls(camera, renderer.domElement);
        orbit.target.copy(center);
        orbit.enableDamping = false;
        orbit.update();
        orbit.addEventListener("change", draw);
        control.current = (shown) => {
          if (!splat || !mesh || disposed) return;
          splat.visible = shown === "splat";
          mesh.visible = shown === "mesh";
          container.dataset.mode = shown;
          draw();
        };
        container.dataset.splatCount = String(splat.numSplats);
        container.dataset.mode = "splat";
        resize.observe(container);
        setReady(true);
        setMessage(
          `${splat.numSplats.toLocaleString()} generated splats; drag to orbit. Mesh is the separate delivery, not a collider validation.`,
        );
        draw();
      } catch (error) {
        if (!disposed) setMessage(error instanceof Error ? error.message : String(error));
      }
    })();
    return () => {
      disposed = true;
      control.current = null;
      cancelAnimationFrame(frame);
      resize.disconnect();
      orbit?.dispose();
      if (mesh) disposeMesh(mesh);
      splat?.dispose();
      spark?.dispose();
      renderer?.dispose();
      container.replaceChildren();
    };
  }, [opened, splatUrl, meshUrl]);
  return (
    <div className="space-y-2">
      <button
        className="rounded-lg border border-white/15 px-3 py-2 text-xs"
        onClick={() => {
          setOpened(!opened);
          setReady(false);
          setMode("splat");
          setMessage(opened ? "" : "Loading retained generated geometry…");
        }}
      >
        {opened ? "Close 3D Gaussian comparison" : "Open 3D Gaussian comparison"}
      </button>
      {opened && (
        <>
          <label className="block text-xs">
            Generated 3D appearance
            <select
              aria-label="Generated 3D appearance"
              value={mode}
              disabled={!ready}
              className="ml-2 rounded bg-zinc-900 p-2"
              onChange={(event) => {
                setMode(event.target.value);
                control.current?.(event.target.value);
              }}
            >
              <option value="splat">Placed native Gaussians</option>
              <option value="mesh">Delivery mesh</option>
            </select>
          </label>
          <div ref={host} className="h-80 w-full min-w-0 overflow-hidden rounded-lg" />
          <p className="break-words text-xs text-zinc-400">{message}</p>
        </>
      )}
    </div>
  );
}
