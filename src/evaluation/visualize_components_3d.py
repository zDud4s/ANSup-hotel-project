"""Interactive 3D PCA component visualizer for clustering space.

The visualizer projects the governed clustering matrix to three principal
components and exports a standalone Three.js HTML file. PCA is diagnostic
only: clustering still happens in the full preprocessed feature space.
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

from ..clustering.ikmeans import fit_ikmeans
from ..data.validate import load_raw
from ..preprocessing.feature_config import FAST_MODE, FAST_N, FAST_SEED
from ..preprocessing.pipeline import (
    add_cyclic_seasonality,
    build_preprocessor,
    split_clustering_and_profiling,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "figures"
TABLES_DIR = PROJECT_ROOT / "tables"
DEFAULT_HTML = FIGURES_DIR / "task2_cluster_space_pca_3d.html"
DEFAULT_PNG = FIGURES_DIR / "task2_cluster_space_pca_3d.png"
DEFAULT_CSV = TABLES_DIR / "task2_cluster_space_pca_3d_sample.csv"

PROFILE_COLUMNS = [
    "hotel",
    "customer_type",
    "market_segment",
    "distribution_channel",
    "deposit_type",
    "reserved_room_type",
    "country",
    "meal",
    "lead_time",
    "total_nights",
    "party_size",
    "adr",
    "is_canceled",
]

PALETTE = [
    "#2d6cdf",
    "#d9480f",
    "#2b8a3e",
    "#9c36b5",
    "#e67700",
    "#0ca678",
    "#c92a2a",
    "#5f3dc4",
    "#087f5b",
    "#364fc7",
    "#f08c00",
    "#a61e4d",
]


def _progress(message: str) -> None:
    print(message, flush=True)


def _load_frames(fast: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_raw()
    if fast:
        df = df.sample(n=FAST_N, random_state=FAST_SEED).reset_index(drop=True)
        _progress(f"[FAST MODE] {len(df):,} rows")
    else:
        _progress(f"[FULL MODE] {len(df):,} rows")
    df = add_cyclic_seasonality(df)
    return split_clustering_and_profiling(df)


def _fit_labels(
    x: np.ndarray,
    labeler: str,
    seed: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray, str, int]:
    if labeler == "none":
        labels = np.zeros(x.shape[0], dtype=int)
        centres = np.empty((0, x.shape[1]))
        return labels, centres, "none", 1

    if labeler == "kmeans":
        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=seed,
            n_init=10,
            batch_size=1024,
            max_iter=300,
        )
        labels = km.fit_predict(x)
        return labels, km.cluster_centers_, "MiniBatchKMeans", k

    labels, centres, k_auto = fit_ikmeans(x, seed=seed, k_max=k)
    return labels, centres, "iKMeans", k_auto


def _sample_indices(n_rows: int, sample_size: int, seed: int) -> np.ndarray:
    if sample_size >= n_rows:
        return np.arange(n_rows)
    return np.sort(np.random.default_rng(seed).choice(n_rows, size=sample_size, replace=False))


def _combined_profile(x_input: pd.DataFrame, profiling_frame: pd.DataFrame) -> pd.DataFrame:
    frame = x_input.copy()
    for col in profiling_frame.columns:
        if col not in frame.columns:
            frame[col] = profiling_frame[col]
    return frame


def _json_value(value) -> str | int | float | None:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _build_payload(
    coords: np.ndarray,
    scene_coords: np.ndarray,
    labels: np.ndarray,
    profile: pd.DataFrame,
    sample_idx: np.ndarray,
) -> list[dict]:
    payload: list[dict] = []
    available_profile_cols = [col for col in PROFILE_COLUMNS if col in profile.columns]
    sampled_profile = profile.iloc[sample_idx].reset_index(drop=True)

    for row_no, original_idx in enumerate(sample_idx):
        profile_values = {
            col: _json_value(sampled_profile.at[row_no, col])
            for col in available_profile_cols
        }
        payload.append({
            "id": int(original_idx),
            "cluster": int(labels[original_idx]),
            "x": float(scene_coords[row_no, 0]),
            "y": float(scene_coords[row_no, 1]),
            "z": float(scene_coords[row_no, 2]),
            "pc1": float(coords[row_no, 0]),
            "pc2": float(coords[row_no, 1]),
            "pc3": float(coords[row_no, 2]),
            "profile": profile_values,
        })
    return payload


def _cluster_meta(labels: np.ndarray, sampled_labels: np.ndarray) -> list[dict]:
    rows: list[dict] = []
    total = len(labels)
    sampled_total = len(sampled_labels)
    for cluster in sorted(np.unique(labels)):
        rows.append({
            "cluster": int(cluster),
            "count": int((labels == cluster).sum()),
            "share": float((labels == cluster).mean()),
            "sampled": int((sampled_labels == cluster).sum()),
            "sampledShare": float((sampled_labels == cluster).sum() / max(sampled_total, 1)),
            "color": PALETTE[int(cluster) % len(PALETTE)],
        })
    return rows


def _centre_payload(
    centres: np.ndarray,
    pca: PCA,
    scale: float,
    labels: np.ndarray,
) -> list[dict]:
    if centres.size == 0:
        return []
    centre_coords = pca.transform(centres)
    counts = pd.Series(labels).value_counts().to_dict()
    payload = []
    for cluster, coord in enumerate(centre_coords):
        payload.append({
            "cluster": int(cluster),
            "x": float(coord[0] / scale * 4.0),
            "y": float(coord[1] / scale * 4.0),
            "z": float(coord[2] / scale * 4.0),
            "pc1": float(coord[0]),
            "pc2": float(coord[1]),
            "pc3": float(coord[2]),
            "count": int(counts.get(cluster, 0)),
            "color": PALETTE[int(cluster) % len(PALETTE)],
        })
    return payload


def _html_document(
    *,
    points_json: str,
    centres_json: str,
    clusters_json: str,
    meta_json: str,
) -> str:
    html = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" href="data:,">
  <title>3D PCA Component Visualizer</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101113;
      --panel: rgba(19, 20, 23, 0.92);
      --panel-soft: rgba(255, 255, 255, 0.055);
      --line: rgba(255, 255, 255, 0.14);
      --text: #f3f1ec;
      --muted: #a9aaa8;
      --accent: #f0c36a;
    }

    * {
      box-sizing: border-box;
    }

    html,
    body {
      width: 100%;
      height: 100%;
      margin: 0;
      overflow: hidden;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }

    #app {
      position: relative;
      width: 100vw;
      height: 100vh;
      min-height: 620px;
      background: linear-gradient(145deg, #101113 0%, #181918 54%, #0c0d0e 100%);
    }

    #scene {
      position: absolute;
      inset: 0;
      display: block;
      width: 100%;
      height: 100%;
      outline: none;
    }

    .panel {
      position: absolute;
      top: 18px;
      left: 18px;
      width: min(340px, calc(100vw - 36px));
      max-height: calc(100vh - 36px);
      overflow: auto;
      padding: 18px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      backdrop-filter: blur(18px);
      box-shadow: 0 18px 60px rgba(0, 0, 0, 0.28);
    }

    .title {
      margin: 0 0 4px;
      font-size: 18px;
      font-weight: 760;
      line-height: 1.15;
    }

    .subtitle {
      margin: 0 0 18px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }

    .stat-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin-bottom: 16px;
    }

    .stat {
      padding: 10px 8px;
      background: var(--panel-soft);
      border: 1px solid var(--line);
      border-radius: 7px;
    }

    .stat strong {
      display: block;
      color: var(--accent);
      font-size: 16px;
      line-height: 1.1;
    }

    .stat span {
      color: var(--muted);
      font-size: 11px;
    }

    .section {
      padding-top: 14px;
      margin-top: 14px;
      border-top: 1px solid var(--line);
    }

    .section h2 {
      margin: 0 0 10px;
      font-size: 12px;
      font-weight: 720;
      color: #dedbd2;
    }

    .cluster-list {
      display: grid;
      gap: 6px;
    }

    .cluster-row {
      display: grid;
      grid-template-columns: 20px 1fr auto;
      align-items: center;
      gap: 8px;
      min-height: 32px;
      color: #ebe7de;
      font-size: 12px;
    }

    .cluster-row input {
      width: 16px;
      height: 16px;
      margin: 0;
      accent-color: var(--accent);
    }

    .swatch {
      display: inline-block;
      width: 9px;
      height: 9px;
      margin-right: 6px;
      border-radius: 50%;
      vertical-align: middle;
    }

    .share {
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }

    .control-row {
      display: grid;
      grid-template-columns: 1fr auto;
      align-items: center;
      gap: 10px;
      margin: 10px 0;
      color: #dedbd2;
      font-size: 12px;
    }

    input[type="range"] {
      width: 100%;
      accent-color: var(--accent);
    }

    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }

    button {
      min-height: 34px;
      padding: 8px 10px;
      color: var(--text);
      background: rgba(255, 255, 255, 0.07);
      border: 1px solid var(--line);
      border-radius: 7px;
      font: inherit;
      font-size: 12px;
      cursor: pointer;
    }

    button:hover,
    button:focus-visible {
      background: rgba(240, 195, 106, 0.17);
      border-color: rgba(240, 195, 106, 0.55);
      outline: none;
    }

    .legend-note {
      margin: 10px 0 0;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.45;
    }

    #tooltip {
      position: absolute;
      z-index: 2;
      min-width: 210px;
      max-width: 280px;
      pointer-events: none;
      opacity: 0;
      transform: translate(10px, 10px);
      padding: 10px 12px;
      color: var(--text);
      background: rgba(15, 16, 18, 0.96);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 12px 36px rgba(0, 0, 0, 0.36);
      font-size: 12px;
      line-height: 1.35;
      transition: opacity 120ms ease;
    }

    #tooltip strong {
      color: var(--accent);
    }

    .footer {
      position: absolute;
      right: 18px;
      bottom: 16px;
      max-width: min(520px, calc(100vw - 36px));
      color: rgba(243, 241, 236, 0.72);
      font-size: 11px;
      text-align: right;
      text-shadow: 0 1px 8px rgba(0, 0, 0, 0.7);
    }

    @media (max-width: 760px) {
      #app {
        min-height: 720px;
      }

      .panel {
        top: 12px;
        left: 12px;
        right: 12px;
        width: auto;
        max-height: 43vh;
        padding: 14px;
      }

      .footer {
        left: 12px;
        right: 12px;
        text-align: left;
      }
    }
  </style>
</head>
<body>
  <main id="app">
    <canvas id="scene" aria-label="3D PCA component space"></canvas>
    <aside class="panel" aria-label="Visualizer controls">
      <h1 class="title">3D PCA Component Space</h1>
      <p class="subtitle" id="subtitle"></p>
      <div class="stat-grid" id="stats"></div>

      <div class="section">
        <h2>Clusters</h2>
        <div class="cluster-list" id="clusterList"></div>
        <p class="legend-note">Hover a point to inspect the booking profile. Uncheck clusters to isolate structure.</p>
      </div>

      <div class="section">
        <h2>Display</h2>
        <label class="control-row">
          <span>Point size</span>
          <input id="pointSize" type="range" min="0.025" max="0.16" step="0.005" value="0.07">
        </label>
        <label class="control-row">
          <span>Opacity</span>
          <input id="opacity" type="range" min="0.18" max="1" step="0.02" value="0.74">
        </label>
        <label class="control-row">
          <span>Auto rotate</span>
          <input id="autoRotate" type="checkbox" checked>
        </label>
        <div class="actions">
          <button id="resetCamera" type="button">Reset camera</button>
          <button id="showAll" type="button">Show all</button>
        </div>
      </div>
    </aside>
    <div id="tooltip"></div>
    <p class="footer">Diagnostic PCA projection. Distances and clusters are fitted in the full governed feature space.</p>
  </main>

  <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/build/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/examples/js/controls/OrbitControls.js"></script>
  <script>
    const POINTS = __POINTS_JSON__;
    const CENTRES = __CENTRES_JSON__;
    const CLUSTERS = __CLUSTERS_JSON__;
    const META = __META_JSON__;

    // Capture mode (for static report screenshots): ?capture=1&view=iso|front|side|top
    const PARAMS = new URLSearchParams(location.search);
    const CAPTURE = PARAMS.has("capture");
    const VIEW = (PARAMS.get("view") || "iso").toLowerCase();
    const DIST = parseFloat(PARAMS.get("dist") || "0.62");
    const PSIZE = parseFloat(PARAMS.get("size") || "0.16");

    const canvas = document.querySelector("#scene");
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x101113, 0.035);

    const camera = new THREE.PerspectiveCamera(48, 1, 0.1, 1000);
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.065;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.38;

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2(-10, -10);
    raycaster.params.Points.threshold = 0.13;

    const tooltip = document.querySelector("#tooltip");
    const pointObjects = new Map();
    const centreObjects = new Map();
    const clusterEnabled = new Map(CLUSTERS.map((cluster) => [cluster.cluster, true]));

    const maxExtent = Math.max(4.5, META.sceneExtent * 1.08);
    const ambient = new THREE.AmbientLight(0xffffff, 0.58);
    scene.add(ambient);
    const key = new THREE.DirectionalLight(0xffffff, 1.2);
    key.position.set(3, 6, 5);
    scene.add(key);

    function formatNumber(value, digits = 2) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
      return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
    }

    function makeMaterial(color, size, opacity) {
      return new THREE.PointsMaterial({
        color,
        size,
        transparent: true,
        opacity,
        sizeAttenuation: true,
        depthWrite: false,
      });
    }

    function addAxis(start, end, color) {
      const geometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(...start),
        new THREE.Vector3(...end),
      ]);
      const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.72 });
      const line = new THREE.Line(geometry, material);
      scene.add(line);
      return line;
    }

    function makeLabel(text, position, color) {
      const labelCanvas = document.createElement("canvas");
      labelCanvas.width = 512;
      labelCanvas.height = 128;
      const ctx = labelCanvas.getContext("2d");
      ctx.clearRect(0, 0, labelCanvas.width, labelCanvas.height);
      ctx.font = "700 44px system-ui, sans-serif";
      ctx.fillStyle = color;
      ctx.fillText(text, 16, 76);
      const texture = new THREE.CanvasTexture(labelCanvas);
      const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
      const sprite = new THREE.Sprite(material);
      sprite.position.set(...position);
      sprite.scale.set(1.4, 0.35, 1);
      scene.add(sprite);
      return sprite;
    }

    addAxis([-maxExtent, 0, 0], [maxExtent, 0, 0], 0xf0c36a);
    addAxis([0, -maxExtent, 0], [0, maxExtent, 0], 0x76d2a8);
    addAxis([0, 0, -maxExtent], [0, 0, maxExtent], 0x93b7ff);
    makeLabel(`PC1 ${formatNumber(META.variance[0] * 100, 1)}%`, [maxExtent + 0.18, 0, 0], "#f0c36a");
    makeLabel(`PC2 ${formatNumber(META.variance[1] * 100, 1)}%`, [0, maxExtent + 0.18, 0], "#76d2a8");
    makeLabel(`PC3 ${formatNumber(META.variance[2] * 100, 1)}%`, [0, 0, maxExtent + 0.18], "#93b7ff");

    const grid = new THREE.GridHelper(maxExtent * 2, 10, 0x59534a, 0x2c2c2c);
    grid.material.transparent = true;
    grid.material.opacity = 0.42;
    scene.add(grid);

    for (const cluster of CLUSTERS) {
      const rows = POINTS.filter((point) => point.cluster === cluster.cluster);
      const positions = new Float32Array(rows.length * 3);
      rows.forEach((point, index) => {
        positions[index * 3] = point.x;
        positions[index * 3 + 1] = point.y;
        positions[index * 3 + 2] = point.z;
      });
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geometry.userData.points = rows;
      const object = new THREE.Points(geometry, makeMaterial(cluster.color, 0.07, 0.74));
      object.userData.cluster = cluster.cluster;
      scene.add(object);
      pointObjects.set(cluster.cluster, object);
    }

    for (const centre of CENTRES) {
      const geometry = new THREE.SphereGeometry(0.16, 24, 18);
      const material = new THREE.MeshStandardMaterial({
        color: centre.color,
        emissive: new THREE.Color(centre.color).multiplyScalar(0.18),
        roughness: 0.4,
        metalness: 0.08,
      });
      const sphere = new THREE.Mesh(geometry, material);
      sphere.position.set(centre.x, centre.y, centre.z);
      sphere.userData.cluster = centre.cluster;
      scene.add(sphere);
      centreObjects.set(centre.cluster, sphere);
    }

    function buildPanel() {
      document.querySelector("#subtitle").textContent =
        `${META.method} labels, ${META.scaler} scaler, n=${META.sampleSize.toLocaleString()} sampled from ${META.rowCount.toLocaleString()} rows`;
      const stats = [
        ["PC1", `${formatNumber(META.variance[0] * 100, 1)}%`],
        ["PC2", `${formatNumber(META.variance[1] * 100, 1)}%`],
        ["PC3", `${formatNumber(META.variance[2] * 100, 1)}%`],
      ];
      document.querySelector("#stats").innerHTML = stats.map(([label, value]) => `
        <div class="stat"><strong>${value}</strong><span>${label} variance</span></div>
      `).join("");

      const clusterList = document.querySelector("#clusterList");
      clusterList.innerHTML = CLUSTERS.map((cluster) => `
        <label class="cluster-row">
          <input type="checkbox" data-cluster="${cluster.cluster}" checked>
          <span><i class="swatch" style="background:${cluster.color}"></i>cluster ${cluster.cluster}</span>
          <span class="share">${formatNumber(cluster.share * 100, 1)}%</span>
        </label>
      `).join("");

      clusterList.querySelectorAll("input[type='checkbox']").forEach((input) => {
        input.addEventListener("change", (event) => {
          const cluster = Number(event.currentTarget.dataset.cluster);
          clusterEnabled.set(cluster, event.currentTarget.checked);
          updateVisibility();
        });
      });
    }

    function updateVisibility() {
      for (const [cluster, object] of pointObjects) {
        object.visible = clusterEnabled.get(cluster);
      }
      for (const [cluster, object] of centreObjects) {
        object.visible = clusterEnabled.get(cluster);
      }
    }

    function updatePointMaterial() {
      const size = Number(document.querySelector("#pointSize").value);
      const opacity = Number(document.querySelector("#opacity").value);
      raycaster.params.Points.threshold = Math.max(0.08, size * 1.9);
      for (const object of pointObjects.values()) {
        object.material.size = size;
        object.material.opacity = opacity;
        object.material.needsUpdate = true;
      }
    }

    function applyView(view) {
      const f = CAPTURE ? DIST : 1.0;
      const d = maxExtent;
      const presets = {
        iso: [d * 1.35 * f, d * 0.92 * f, d * 1.45 * f],
        front: [0, d * 0.12, d * 2.05 * f],
        side: [d * 2.05 * f, d * 0.12, 0.001],
        top: [0.001, d * 2.3 * f, 0.001],
      };
      const pos = presets[view] || presets.iso;
      camera.position.set(pos[0], pos[1], pos[2]);
      camera.lookAt(0, 0, 0);
      controls.target.set(0, 0, 0);
      controls.update();
    }

    function resetCamera() {
      applyView("iso");
    }

    function showTooltip(point, x, y) {
      const profile = point.profile || {};
      const hotel = profile.hotel || "n/a";
      const segment = profile.market_segment || "n/a";
      const customer = profile.customer_type || "n/a";
      tooltip.innerHTML = `
        <strong>cluster ${point.cluster}</strong><br>
        PC1 ${formatNumber(point.pc1, 2)} | PC2 ${formatNumber(point.pc2, 2)} | PC3 ${formatNumber(point.pc3, 2)}<br>
        hotel: ${hotel}<br>
        segment: ${segment}<br>
        customer: ${customer}<br>
        lead time: ${formatNumber(profile.lead_time, 0)}
      `;
      tooltip.style.left = `${Math.min(x + 12, window.innerWidth - 300)}px`;
      tooltip.style.top = `${Math.min(y + 12, window.innerHeight - 170)}px`;
      tooltip.style.opacity = 1;
    }

    function hideTooltip() {
      tooltip.style.opacity = 0;
    }

    function onPointerMove(event) {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    }

    function resize() {
      const width = window.innerWidth;
      const height = window.innerHeight;
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    }

    function animate() {
      requestAnimationFrame(animate);
      controls.autoRotate = !CAPTURE && document.querySelector("#autoRotate").checked;
      controls.update();

      raycaster.setFromCamera(pointer, camera);
      const visiblePointObjects = [...pointObjects.values()].filter((object) => object.visible);
      const intersections = raycaster.intersectObjects(visiblePointObjects, false);
      if (intersections.length) {
        const hit = intersections[0];
        const point = hit.object.geometry.userData.points[hit.index];
        const x = (pointer.x + 1) * window.innerWidth / 2;
        const y = (-pointer.y + 1) * window.innerHeight / 2;
        showTooltip(point, x, y);
      } else {
        hideTooltip();
      }

      renderer.render(scene, camera);
    }

    buildPanel();
    document.querySelector("#pointSize").addEventListener("input", updatePointMaterial);
    document.querySelector("#opacity").addEventListener("input", updatePointMaterial);
    document.querySelector("#resetCamera").addEventListener("click", resetCamera);
    document.querySelector("#showAll").addEventListener("click", () => {
      document.querySelectorAll("#clusterList input[type='checkbox']").forEach((input) => {
        input.checked = true;
        clusterEnabled.set(Number(input.dataset.cluster), true);
      });
      updateVisibility();
    });
    window.addEventListener("resize", resize);
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerleave", hideTooltip);

    if (CAPTURE) {
      const panelEl = document.querySelector(".panel");
      if (panelEl) panelEl.style.display = "none";
      const footerEl = document.querySelector(".footer");
      if (footerEl) footerEl.style.display = "none";
      controls.autoRotate = false;
      controls.enabled = false;
      // Fog and faint dark points read well interactively but vanish in a flat
      // screenshot: drop the fog and make the cloud larger and brighter.
      scene.fog = null;
      for (const object of pointObjects.values()) {
        object.material.size = PSIZE;
        object.material.opacity = 0.92;
        object.material.needsUpdate = true;
      }
      window.__captureReady = false;
      // Signal readiness once a few frames have rendered, for screenshot tooling.
      setTimeout(() => { window.__captureReady = true; }, 1200);
    }

    applyView(CAPTURE ? VIEW : "iso");
    resize();
    animate();
  </script>
</body>
</html>
"""
    return (
        html
        .replace("__POINTS_JSON__", points_json)
        .replace("__CENTRES_JSON__", centres_json)
        .replace("__CLUSTERS_JSON__", clusters_json)
        .replace("__META_JSON__", meta_json)
    )


def _write_static_figure(
    *,
    coords: np.ndarray,
    labels_sample: np.ndarray,
    clusters: list[dict],
    variance: list[float],
    method_name: str,
    scaler: str,
    n_rows: int,
    n_sample: int,
    png_path: Path,
) -> None:
    """Render a browser-free static twin of the interactive 3D scene.

    This is the figure committed to the repository and referenced by the report:
    it reproduces the same PCA(3) cloud as the Three.js visualizer using only
    matplotlib, so ``run_all`` regenerates it with no browser dependency.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

    bg = "#101113"
    panel_bg = "#15171a"
    text_c = "#f3f1ec"
    muted_c = "#a9aaa8"
    grid_c = (1.0, 1.0, 1.0, 0.06)

    cluster_ids = [int(c["cluster"]) for c in clusters]
    color_for = {int(c["cluster"]): c["color"] for c in clusters}
    share_for = {int(c["cluster"]): float(c["share"]) for c in clusters}

    v = [float(x) * 100.0 for x in variance]
    lab = {"pc1": f"PC1  {v[0]:.1f}%", "pc2": f"PC2  {v[1]:.1f}%", "pc3": f"PC3  {v[2]:.1f}%"}

    # A handful of outliers stretch the raw PCA axes and leave the dense cloud
    # tiny. Zoom both panels to the central mass (the same trick the 2D view uses).
    def _bounds(values: np.ndarray, lo: float = 1.5, hi: float = 98.5, pad: float = 0.06):
        a, b = np.percentile(values, [lo, hi])
        span = (b - a) or 1.0
        return float(a - span * pad), float(b + span * pad)

    xlim, ylim, zlim = (_bounds(coords[:, 0]), _bounds(coords[:, 1]), _bounds(coords[:, 2]))
    in3d = (
        (coords[:, 0] >= xlim[0]) & (coords[:, 0] <= xlim[1])
        & (coords[:, 1] >= ylim[0]) & (coords[:, 1] <= ylim[1])
        & (coords[:, 2] >= zlim[0]) & (coords[:, 2] <= zlim[1])
    )
    in2d = (
        (coords[:, 0] >= xlim[0]) & (coords[:, 0] <= xlim[1])
        & (coords[:, 1] >= ylim[0]) & (coords[:, 1] <= ylim[1])
    )

    fig = plt.figure(figsize=(13.0, 6.2), facecolor=bg)

    # Isometric 3D panel: the "hero" view, matching the interactive scene.
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.set_facecolor(bg)
    for cid in cluster_ids:
        m = (labels_sample == cid) & in3d
        if m.any():
            ax.scatter(coords[m, 0], coords[m, 1], coords[m, 2], s=6,
                       c=color_for[cid], alpha=0.55, edgecolors="none", depthshade=True)
    for cid in cluster_ids:
        m = labels_sample == cid
        if m.any():
            cx, cy, cz = coords[m].mean(axis=0)
            ax.scatter([cx], [cy], [cz], s=110, c=color_for[cid],
                       edgecolors="white", linewidths=1.1, marker="o", depthshade=False)
    ax.set_xlim3d(xlim)
    ax.set_ylim3d(ylim)
    ax.set_zlim3d(zlim)
    ax.view_init(elev=18, azim=-62)
    ax.set_xlabel(lab["pc1"], color=text_c, fontsize=9, labelpad=4)
    ax.set_ylabel(lab["pc2"], color=text_c, fontsize=9, labelpad=4)
    ax.set_zlabel(lab["pc3"], color=text_c, fontsize=9, labelpad=4)
    ax.set_title("Isometric projection", color=text_c, fontsize=11, pad=6)
    ax.tick_params(colors=muted_c, labelsize=6)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.set_pane_color((1, 1, 1, 0.015))
            axis.pane.set_edgecolor((1, 1, 1, 0.08))
            axis._axinfo["grid"]["color"] = grid_c
            axis.line.set_color((1, 1, 1, 0.25))
        except Exception:
            pass

    # PC1 x PC2 plane: the analytical view where the segment split reads cleanly.
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_facecolor(panel_bg)
    for cid in cluster_ids:
        m = (labels_sample == cid) & in2d
        if m.any():
            ax2.scatter(coords[m, 0], coords[m, 1], s=7, c=color_for[cid],
                        alpha=0.5, edgecolors="none")
    for cid in cluster_ids:
        m = labels_sample == cid
        if m.any():
            ax2.scatter([coords[m, 0].mean()], [coords[m, 1].mean()], s=120,
                        c=color_for[cid], edgecolors="white", linewidths=1.1, marker="o")
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_xlabel(lab["pc1"], color=text_c, fontsize=9)
    ax2.set_ylabel(lab["pc2"], color=text_c, fontsize=9)
    ax2.set_title("PC1 x PC2 plane", color=text_c, fontsize=11, pad=6)
    ax2.tick_params(colors=muted_c, labelsize=7)
    for spine in ax2.spines.values():
        spine.set_color((1, 1, 1, 0.18))

    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=7,
               markerfacecolor=color_for[cid], markeredgecolor="none",
               label=f"segment {cid}  ({share_for[cid] * 100:.1f}%)")
        for cid in cluster_ids
    ]
    leg = ax2.legend(handles=handles, loc="upper right", fontsize=7.5, frameon=True,
                     facecolor=panel_bg, edgecolor=(1, 1, 1, 0.15), framealpha=0.85,
                     labelcolor=text_c, title="clusters", title_fontsize=8)
    if leg is not None and leg.get_title() is not None:
        leg.get_title().set_color(muted_c)

    fig.suptitle(
        f"Clustering space, diagnostic PCA projection  ({method_name} + {scaler} scaler, k={len(cluster_ids)})",
        color=text_c, fontsize=13, y=0.97,
    )
    fig.text(
        0.5, 0.035,
        "PCA is a viewing aid only: clustering uses the full preprocessed feature space.  "
        f"n={n_sample:,} points sampled from {n_rows:,} bookings.",
        ha="center", color=muted_c, fontsize=8.5,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.14, wspace=0.10)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150, facecolor=bg)
    plt.close(fig)


def run(
    *,
    fast: bool,
    scaler: str,
    labeler: str,
    k: int,
    sample_size: int,
    seed: int,
    html_path: Path,
    csv_path: Path,
    open_browser: bool,
    png_path: Path = DEFAULT_PNG,
) -> tuple[Path, Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    _progress("=== 3D PCA component visualizer ===")
    x_input, profiling_frame = _load_frames(fast)

    _progress(f"Fitting preprocessing pipeline with scaler={scaler}")
    preprocessor = build_preprocessor(scaler)
    x = preprocessor.fit_transform(x_input)
    _progress(f"X shape: {x.shape}")

    _progress(f"Fitting labels with {labeler}")
    labels, centres, method_name, k_used = _fit_labels(x, labeler=labeler, seed=seed, k=k)
    _progress(f"Label set: method={method_name}; k={k_used}")

    _progress("Projecting full clustering space to PCA(3)")
    pca = PCA(n_components=3, random_state=FAST_SEED)
    all_coords = pca.fit_transform(x)
    sample_idx = _sample_indices(len(x), sample_size, seed=seed)
    coords = all_coords[sample_idx]
    labels_sample = labels[sample_idx]

    # Uniform scaling preserves PCA-space geometry while keeping outliers visible.
    scale = float(np.quantile(np.abs(coords), 0.98))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    scene_coords = coords / scale * 4.0

    profile = _combined_profile(x_input, profiling_frame)
    export = pd.DataFrame({
        "pc1": coords[:, 0],
        "pc2": coords[:, 1],
        "pc3": coords[:, 2],
        "cluster": labels_sample,
        "source_row": sample_idx,
    })
    for col in PROFILE_COLUMNS:
        if col in profile.columns:
            export[col] = profile.iloc[sample_idx][col].to_numpy()
    export.to_csv(csv_path, index=False)

    points = _build_payload(coords, scene_coords, labels, profile, sample_idx)
    clusters = _cluster_meta(labels, labels_sample)
    centres_payload = _centre_payload(centres, pca, scale, labels)
    scene_extent = float(np.max(np.abs(scene_coords))) if len(scene_coords) else 4.0
    meta = {
        "method": method_name,
        "scaler": scaler,
        "k": int(k_used),
        "rowCount": int(len(x)),
        "sampleSize": int(len(sample_idx)),
        "variance": [float(v) for v in pca.explained_variance_ratio_],
        "scale": scale,
        "sceneExtent": scene_extent,
    }

    html = _html_document(
        points_json=json.dumps(points, ensure_ascii=True, separators=(",", ":")),
        centres_json=json.dumps(centres_payload, ensure_ascii=True, separators=(",", ":")),
        clusters_json=json.dumps(clusters, ensure_ascii=True, separators=(",", ":")),
        meta_json=json.dumps(meta, ensure_ascii=True, separators=(",", ":")),
    )
    html_path.write_text(html, encoding="utf-8")

    _write_static_figure(
        coords=coords,
        labels_sample=labels_sample,
        clusters=clusters,
        variance=meta["variance"],
        method_name=method_name,
        scaler=scaler,
        n_rows=int(len(x)),
        n_sample=int(len(sample_idx)),
        png_path=png_path,
    )

    _progress(f"Saved HTML visualizer: {html_path}")
    _progress(f"Saved static report figure: {png_path}")
    _progress(f"Saved sampled PCA coordinates: {csv_path}")
    if open_browser:
        webbrowser.open(html_path.resolve().as_uri())
    return html_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fast", action="store_true", help="Use the configured fast subsample.")
    mode.add_argument("--full", action="store_true", help="Use all available rows.")
    parser.add_argument("--scaler", choices=["standard", "robust"], default="standard")
    parser.add_argument("--labeler", choices=["ikmeans", "kmeans", "none"], default="ikmeans")
    parser.add_argument("--k", type=int, default=8, help="k for k-means, or k_max for iKMeans.")
    parser.add_argument("--sample-size", type=int, default=7000)
    parser.add_argument("--seed", type=int, default=FAST_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--png-output", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--open", action="store_true", help="Open the exported HTML in the browser.")
    args = parser.parse_args()

    if args.full:
        fast = False
    elif args.fast:
        fast = True
    else:
        fast = FAST_MODE
    run(
        fast=fast,
        scaler=args.scaler,
        labeler=args.labeler,
        k=args.k,
        sample_size=args.sample_size,
        seed=args.seed,
        html_path=args.output,
        csv_path=args.csv_output,
        open_browser=args.open,
        png_path=args.png_output,
    )


if __name__ == "__main__":
    main()
