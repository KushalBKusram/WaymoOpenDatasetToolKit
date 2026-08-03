"""Local Streamlit explorer and run-artifact evaluation dashboard.

Run: streamlit run app.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from modules.run_artifacts import RunArtifacts


st.set_page_config(page_title="Waymo Data Explorer", page_icon="🚗", layout="wide")


def application_default_credentials() -> tuple[str | None, str | None]:
    """Validate local ADC without attempting a network or GCS request."""
    configured = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.append(Path.home() / ".config" / "gcloud" / "application_default_credentials.json")
    credential_file = next((path for path in candidates if path.is_file()), None)
    if credential_file is None:
        return None, "No Application Default Credentials file was found."
    try:
        from google.auth import load_credentials_from_file
        load_credentials_from_file(str(credential_file))
    except Exception as exc:
        return None, f"Credentials at {credential_file} could not be loaded: {exc}"
    return str(credential_file), None


@st.cache_resource(show_spinner=False)
def toolkit_for(split: str) -> ToolKit:
    from modules.waymo_open_dataset import ToolKit
    return ToolKit(split=split)


@st.cache_data(show_spinner=False)
def segment_names(split: str) -> list[str]:
    return toolkit_for(split).list_segments()


def _format_bytes(size: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
        size /= 1024
    return f"{size:.1f} GiB"


def camera_mosaic(images: list[np.ndarray], names: list[str]) -> np.ndarray:
    """Render five annotated camera views as a labeled 3×2 BGR grid."""
    tile_w, tile_h = 640, 360
    tiles: list[np.ndarray] = []
    for image, name in zip(images, names):
        tile = cv2.resize(image, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (210, 34), (25, 25, 25), -1)
        cv2.putText(tile, name, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    blank = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    while len(tiles) < 6:
        tiles.append(blank.copy())
    return cv2.vconcat([cv2.hconcat(tiles[:2]), cv2.hconcat(tiles[2:4]), cv2.hconcat(tiles[4:6])])


def _move_frame(frame_key: str, frame_count: int, step: int) -> None:
    """Move a Streamlit frame widget without leaving its valid range."""
    current = int(st.session_state.get(frame_key, 0))
    st.session_state[frame_key] = (current + step) % frame_count


def _toggle_playback(play_key: str) -> None:
    st.session_state[play_key] = not bool(st.session_state.get(play_key, False))


def show_explorer() -> None:
    """Guide users from credentials → segment → cached workflow → frame."""
    st.title("Waymo Data Explorer")
    credential_file, credential_error = application_default_credentials()
    if credential_error:
        st.error("Google Cloud credentials are required before the explorer can connect to GCS.")
        st.code("gcloud auth application-default login", language="bash")
        st.caption(f"{credential_error} The app has not contacted GCS.")
        return

    from modules.segment_cache import SegmentCache, WORKFLOW_COMPONENTS
    st.caption(f"Credentials ready: {credential_file}")
    split = st.sidebar.selectbox("Dataset Split", ["training", "validation", "testing"])

    with st.status("Loading segment catalog…", expanded=True) as status:
        status.write("Starting the Waymo data reader…")
        from modules.waymo_open_dataset import CAMERA_NAMES, LABEL_TYPES, ToolKit, _L_BOX
        status.write(f"Listing {split} segment names from GCS. This downloads metadata only, not sensor data.")
        try:
            segments = segment_names(split)
        except Exception as exc:
            status.update(label="Could not load segment catalog", state="error")
            st.error(f"Could not list GCS segments: {exc}")
            return
        status.update(label=f"{len(segments)} segments available", state="complete", expanded=False)
    if not segments:
        st.warning("No segments found for this split.")
        return

    page_count = max(1, (len(segments) + 9) // 10)
    page = st.number_input("Segment Page", min_value=1, max_value=page_count, value=1, step=1)
    first = (page - 1) * 10
    page_segments = segments[first:first + 10]
    selected = st.radio(f"Select a Segment ({first + 1}–{first + len(page_segments)} of {len(segments)})", page_segments)

    workflow = st.radio("Choose a Workflow", list(WORKFLOW_COMPONENTS), horizontal=True)
    components = WORKFLOW_COMPONENTS[workflow]
    cache = SegmentCache()
    cached_files = cache.files_for_segment(split, selected)
    cache_size = cache.segment_cache_size(split, selected)
    with st.expander("Local Cache"):
        if cached_files:
            component_names = sorted({path.parent.name for path in cached_files if path.suffix == ".parquet"})
            decoded_count = sum(path.suffix == ".npz" for path in cached_files)
            st.caption(
                f"{len(cached_files)} files · {_format_bytes(cache_size)} · "
                f"Components: {', '.join(component_names) or 'frame indexes only'} · "
                f"Decoded LiDAR Frames: {decoded_count}"
            )
            confirm_key = f"clear-cache-confirm:{split}:{selected}"
            confirmed = st.checkbox("I Understand This Removes Only This Segment's Local Cache", key=confirm_key)
            if st.button("Remove Selected Segment From Local Cache", disabled=not confirmed, type="secondary"):
                removed_bytes = cache.remove_segment(split, selected)
                for key in list(st.session_state):
                    if key.startswith(("segment-analysis:", "mosaic:", "lidar:", "linked-mosaic:")) and f":{split}:{selected}" in key:
                        del st.session_state[key]
                toolkit_for(split).assign_segment(selected)
                st.success(f"Removed {_format_bytes(removed_bytes)} from the local cache.")
                st.rerun()
        else:
            st.caption("No Local Cache Files for This Segment Yet.")
    missing = [component for component in components if not cache.has_component(split, component, selected)]
    st.caption("Required Local Files: " + ", ".join(components))
    if missing:
        st.warning("Not cached yet: " + ", ".join(missing))
        if st.button(f"Download Files for {workflow}", type="primary"):
            progress_bar = st.progress(0, text="Preparing download…")
            def update_progress(component: str, done: int, total: int, index: int, count: int) -> None:
                fraction = done / total if total else 0.0
                overall = int((((index - 1) + fraction) / count) * 100)
                detail = f"{_format_bytes(done)} / {_format_bytes(total)}" if total else _format_bytes(done)
                progress_bar.progress(overall, text=f"Downloading {index}/{count}: {component} — {detail}")
            try:
                with st.status("Caching Selected Workflow…", expanded=True) as status:
                    paths = cache.download_workflow(split, selected, workflow, update_progress)
                    status.write("Cached: " + ", ".join(path.name for path in paths))
                    status.update(label="Local Cache Ready", state="complete", expanded=False)
                progress_bar.progress(100, text="Download complete")
                st.rerun()
            except Exception as exc:
                progress_bar.empty()
                st.error(f"Download failed: {exc}")
                return
    else:
        st.success("Required Files Are Available in the Local Cache.")

    toolkit = toolkit_for(split)
    if toolkit.context_name != selected:
        toolkit.assign_segment(selected)

    if workflow == "Segment Analysis":
        analysis_key = f"segment-analysis:{split}:{selected}"
        stats = st.session_state.get(analysis_key)
        if stats is None:
            with st.status("Analyzing Cached LiDAR Labels…", expanded=True) as status:
                status.write("Reading the selected segment's 3D labels once, then preparing training-relevant summaries.")
                try:
                    stats = toolkit.load_all_boxes_df()
                    st.session_state[analysis_key] = stats
                except Exception as exc:
                    status.update(label="Could Not Analyze Segment", state="error")
                    st.error(str(exc))
                    return
                status.update(label="Segment Analysis Ready", state="complete", expanded=False)

        type_column = f"{_L_BOX}.type"
        frame_column = "key.frame_timestamp_micros"
        center_x = f"{_L_BOX}.box.center.x"
        center_y = f"{_L_BOX}.box.center.y"
        size_x = f"{_L_BOX}.box.size.x"
        size_y = f"{_L_BOX}.box.size.y"
        size_z = f"{_L_BOX}.box.size.z"
        analysis = stats.copy()
        analysis["Class"] = analysis[type_column].map(lambda value: LABEL_TYPES.get(int(value), "TYPE_UNKNOWN")).str.replace("TYPE_", "", regex=False)
        analysis["Range (m)"] = np.hypot(analysis[center_x], analysis[center_y])
        analysis["Box Volume (m³)"] = analysis[size_x] * analysis[size_y] * analysis[size_z]
        frame_counts = analysis.groupby(frame_column).size().rename("Labels")

        st.subheader("Segment Overview")
        metric_columns = st.columns(4)
        metric_columns[0].metric("3D Labels", f"{len(analysis):,}")
        metric_columns[1].metric("Labeled Frames", f"{len(frame_counts):,}")
        metric_columns[2].metric("Labels per Frame", f"{frame_counts.mean():.1f}")
        metric_columns[3].metric("Median Label Range", f"{analysis['Range (m)'].median():.1f} m")

        left, right = st.columns(2)
        with left:
            st.subheader("Class Balance")
            class_counts = analysis["Class"].value_counts().rename("Labels")
            st.bar_chart(class_counts)
        with right:
            st.subheader("Labels per Frame")
            st.line_chart(frame_counts)

        left, right = st.columns(2)
        with left:
            st.subheader("Distance Coverage")
            distance_bins = pd.cut(
                analysis["Range (m)"],
                bins=[0, 10, 20, 30, 50, 75, 100, float("inf")],
                labels=["0–10 m", "10–20 m", "20–30 m", "30–50 m", "50–75 m", "75–100 m", "100+ m"],
                include_lowest=True,
            )
            st.bar_chart(distance_bins.value_counts(sort=False).rename("Labels"))
        with right:
            st.subheader("Class Coverage")
            class_summary = analysis.groupby("Class").agg(
                Labels=("Class", "size"),
                Frames=(frame_column, "nunique"),
                Median_Range_m=("Range (m)", "median"),
                Median_Box_Volume_m3=("Box Volume (m³)", "median"),
            ).sort_values("Labels", ascending=False)
            st.dataframe(class_summary, use_container_width=True)

        with st.expander("Raw Label Preview"):
            st.dataframe(analysis.head(100), use_container_width=True, hide_index=True)
        return

    timeline_component = "camera_image" if workflow == "Camera Frames" else "lidar"
    timeline_key = f"timestamps:{split}:{selected}:{timeline_component}"
    if st.button("Load Frame Timeline"):
        with st.status("Loading Frame Timeline…", expanded=True) as status:
            cached_timeline = cache.load_timeline(split, selected, timeline_component)
            if cached_timeline is not None:
                st.session_state[timeline_key] = cached_timeline
                status.update(label=f"Loaded {len(cached_timeline)} cached frame indexes", state="complete", expanded=False)
            else:
                status.write("Reading only the timestamp column, then saving a persistent frame index.")
                try:
                    timestamps_for_index = toolkit.get_timestamps(timeline_component)
                    cache.save_timeline(split, selected, timestamps_for_index, timeline_component)
                    st.session_state[timeline_key] = timestamps_for_index
                except Exception as exc:
                    status.update(label="Could not load timeline", state="error")
                    st.error(str(exc))
                    return
                status.update(label=f"Indexed {len(timestamps_for_index)} frames", state="complete", expanded=False)
    timestamps = st.session_state.get(timeline_key)
    if not timestamps:
        st.info("Choose **Load frame timeline** to select a frame.")
        return

    frame_key = f"frame:{split}:{selected}:{workflow}"
    play_key = f"playback:{split}:{selected}:{workflow}"
    advance_key = f"advance-playback:{split}:{selected}:{workflow}"
    if frame_key not in st.session_state:
        st.session_state[frame_key] = 0
    is_playing = bool(st.session_state.get(play_key, False))
    if not is_playing:
        st.session_state.pop(advance_key, None)
    elif st.session_state.pop(advance_key, False):
        _move_frame(frame_key, len(timestamps), 1)
    navigation = st.columns([1, 1, 1, 3])
    navigation[0].button("Previous Frame", on_click=_move_frame, args=(frame_key, len(timestamps), -1), use_container_width=True)
    navigation[1].button(
        "Pause" if st.session_state.get(play_key, False) else "Play",
        on_click=_toggle_playback,
        args=(play_key,),
        use_container_width=True,
    )
    navigation[2].button("Next Frame", on_click=_move_frame, args=(frame_key, len(timestamps), 1), use_container_width=True)
    navigation[3].caption(f"Frame {int(st.session_state[frame_key]) + 1} of {len(timestamps)} · Playback advances every 0.5 seconds")
    frame_index = st.slider("Frame", 0, len(timestamps) - 1, key=frame_key)
    timestamp = timestamps[frame_index]
    if workflow == "Camera Frames":
        from modules.visualize import draw_camera_boxes
        result_key = f"mosaic:{split}:{selected}:{timestamp}"
        if st.button("View All Camera Frames") or is_playing:
            with st.status("Building all-camera mosaic…", expanded=True) as status:
                images: dict[int, np.ndarray] = {}
                try:
                    for index, camera_id in enumerate(CAMERA_NAMES, start=1):
                        status.write(f"Reading camera {index}/5: {CAMERA_NAMES[camera_id]}")
                        images[camera_id] = draw_camera_boxes(
                            toolkit.load_camera_frame(timestamp, camera_id),
                            toolkit.load_camera_boxes(timestamp, camera_id),
                        )
                except Exception as exc:
                    status.update(label="Could not build camera mosaic", state="error")
                    st.error(str(exc))
                    return
                st.session_state[result_key] = {
                    "mosaic": camera_mosaic(list(images.values()), [CAMERA_NAMES[c] for c in images]),
                    "images": images,
                }
                status.update(label="All-camera mosaic ready", state="complete", expanded=False)
        result = st.session_state.get(result_key)
        if result is None:
            st.info("Choose **View all camera frames** to display the five synchronized views.")
            return
        st.image(cv2.cvtColor(result["mosaic"], cv2.COLOR_BGR2RGB), caption=f"All cameras · {timestamp}", use_container_width=True)
        st.caption("Open a Full-Resolution Camera View")
        detail_key = f"camera-detail:{split}:{selected}:{timestamp}"
        camera_columns = st.columns(5)
        for column, camera_id in zip(camera_columns, CAMERA_NAMES):
            if column.button(CAMERA_NAMES[camera_id], key=f"open:{detail_key}:{camera_id}", use_container_width=True):
                st.session_state[detail_key] = camera_id
        opened_camera = st.session_state.get(detail_key)
        if opened_camera in result["images"]:
            st.image(
                cv2.cvtColor(result["images"][opened_camera], cv2.COLOR_BGR2RGB),
                caption=f"{CAMERA_NAMES[opened_camera]} · {timestamp}",
                use_container_width=True,
            )
    else:
        from modules.visualize import plot_lidar_3d_interactive
        st.caption("Interactive 3D point cloud with a road-height reference surface. Z remains the true vehicle-frame height, so road points are commonly below zero.")
        top_lidar_only = True
        max_points = st.select_slider("Maximum Plotted Points", options=[25_000, 50_000, 100_000, 200_000], value=100_000)
        result_key = f"lidar:{split}:{selected}:{frame_index}:{top_lidar_only}"
        if st.button("View LiDAR Frame") or is_playing:
            progress = st.progress(0, text="Checking the decoded frame cache…")
            with st.status("Preparing LiDAR Frame…", expanded=True) as status:
                timings: dict[str, float | str] = {}
                try:
                    started = time.perf_counter()
                    points = cache.load_lidar_frame(split, selected, frame_index, top_lidar_only)
                    timings["frame_cache_read_s"] = time.perf_counter() - started
                    progress.progress(20, text="Checked decoded frame cache")
                    if points is None:
                        status.write("Cache miss: decoding selected LiDAR range-image rows. This runs once per frame and mode.")
                        progress.progress(30, text="Decoding LiDAR range image…")
                        started = time.perf_counter()
                        points = toolkit.load_lidar_points(timestamp, top_lidar_only=top_lidar_only)
                        timings["range_image_decode_s"] = time.perf_counter() - started
                        progress.progress(75, text="Saving decoded frame for future views…")
                        started = time.perf_counter()
                        cache.save_lidar_frame(split, selected, frame_index, top_lidar_only, points)
                        timings["frame_cache_write_s"] = time.perf_counter() - started
                        timings["cache_status"] = "miss — decoded and saved"
                    else:
                        status.write("Cache hit: loaded previously decoded points.")
                        progress.progress(75, text="Decoded frame cache loaded")
                        timings["cache_status"] = "hit — loaded decoded points"
                    status.write("Loading 3-D labels for this frame…")
                    started = time.perf_counter()
                    boxes = toolkit.load_lidar_boxes(timestamp)
                    timings["box_read_s"] = time.perf_counter() - started
                    progress.progress(100, text="LiDAR Frame Prepared")
                except Exception as exc:
                    progress.empty()
                    status.update(label="Could Not Load LiDAR Frame", state="error")
                    st.error(str(exc))
                    return
                st.session_state[result_key] = {"points": points, "boxes": boxes, "timings": timings}
                status.update(label="LiDAR Frame Ready", state="complete", expanded=False)
        result = st.session_state.get(result_key)
        if result is None:
            st.info("Choose **View LiDAR frame** to decode this frame once and save it for fast reuse.")
            return
        plot_progress = st.progress(0, text=f"Downsampling Up to {max_points:,} Points for the 3D Point Cloud…")
        with st.status("Rendering LiDAR 3D Point Cloud…", expanded=True) as status:
            plot_progress.progress(25, text="Building the Interactive Point-Cloud Scene…")
            started = time.perf_counter()
            fig = plot_lidar_3d_interactive(result["points"], result["boxes"], max_points=max_points)
            result["timings"]["plot_s"] = time.perf_counter() - started
            plot_progress.progress(100, text="3D Point Cloud Ready")
            status.update(label="3D Point Cloud Ready", state="complete", expanded=False)
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})
        timing = result["timings"]
        st.caption(
            f"{timing['cache_status']} · cache read {timing['frame_cache_read_s']:.2f}s · "
            f"box read {timing['box_read_s']:.2f}s · plot {timing['plot_s']:.2f}s"
            + (f" · decode {timing['range_image_decode_s']:.2f}s · cache write {timing['frame_cache_write_s']:.2f}s" if 'range_image_decode_s' in timing else "")
        )

        with st.expander("Synchronized Camera Mosaic"):
            camera_components = WORKFLOW_COMPONENTS["Camera Frames"]
            camera_missing = [
                component for component in camera_components
                if not cache.has_component(split, component, selected)
            ]
            linked_key = f"linked-mosaic:{split}:{selected}:{timestamp}"
            if camera_missing:
                st.caption("Camera files are not cached for this segment. Loading the mosaic will download them once with progress.")
                action_label = "Download and Load Synchronized Camera Mosaic"
            else:
                st.caption(f"Five Camera Views at the Current LiDAR Timestamp: {timestamp}")
                action_label = "Load Synchronized Camera Mosaic"
            if st.button(action_label, key=f"load-linked-mosaic:{split}:{selected}:{timestamp}"):
                download_progress = st.progress(0, text="Preparing Synchronized Camera Mosaic…")
                try:
                    if camera_missing:
                        with st.status("Caching Camera Frames for the Linked View…", expanded=True) as status:
                            def linked_download_progress(component: str, done: int, total: int, index: int, count: int) -> None:
                                fraction = done / total if total else 0.0
                                overall = int((((index - 1) + fraction) / count) * 100)
                                detail = f"{_format_bytes(done)} / {_format_bytes(total)}" if total else _format_bytes(done)
                                download_progress.progress(overall, text=f"Downloading {index}/{count}: {component} — {detail}")
                            cache.download_workflow(split, selected, "Camera Frames", linked_download_progress)
                            status.update(label="Camera Frames Cached", state="complete", expanded=False)
                    with st.status("Building Synchronized Camera Mosaic…", expanded=True) as status:
                        from modules.visualize import draw_camera_boxes
                        images: dict[int, np.ndarray] = {}
                        for index, camera_id in enumerate(CAMERA_NAMES, start=1):
                            status.write(f"Reading Camera {index}/5: {CAMERA_NAMES[camera_id]}")
                            images[camera_id] = draw_camera_boxes(
                                toolkit.load_camera_frame(timestamp, camera_id),
                                toolkit.load_camera_boxes(timestamp, camera_id),
                            )
                        st.session_state[linked_key] = camera_mosaic(
                            list(images.values()), [CAMERA_NAMES[camera_id] for camera_id in images]
                        )
                        status.update(label="Synchronized Camera Mosaic Ready", state="complete", expanded=False)
                    download_progress.progress(100, text="Synchronized Camera Mosaic Ready")
                except Exception as exc:
                    download_progress.empty()
                    st.error(f"Could Not Load Synchronized Camera Mosaic: {exc}")
            linked_mosaic = st.session_state.get(linked_key)
            if linked_mosaic is not None:
                st.image(
                    cv2.cvtColor(linked_mosaic, cv2.COLOR_BGR2RGB),
                    caption=f"Synchronized With LiDAR Frame {frame_index + 1} · {timestamp}",
                    use_container_width=True,
                )




    if is_playing:
        time.sleep(0.5)
        if st.session_state.get(play_key, False):
            st.session_state[advance_key] = True
            st.rerun()


def show_runs() -> None:
    st.title("Evaluation and Run Reports")
    run_root = Path(st.sidebar.text_input("Run Directory", "./runs/waymo"))
    if not run_root.exists():
        st.info("No run artifacts found yet. Train with `--drive-dir` or run `evaluate.py` first.")
        return
    candidates = [run_root] + sorted((p for p in run_root.iterdir() if p.is_dir()), reverse=True)
    run_dir = st.selectbox("Run", candidates, format_func=lambda p: str(p))
    artifacts = RunArtifacts(run_dir)
    metrics = artifacts.read_metrics()
    config_path = run_dir / "config.json"
    if config_path.exists():
        with st.expander("Run Configuration"):
            st.json(json.loads(config_path.read_text()))
    events = metrics.get("events", [])
    if not events:
        st.info("This run has no recorded metrics yet.")
        return
    table = pd.DataFrame(events)
    st.dataframe(table, use_container_width=True, hide_index=True)
    train_rows = table[table["type"] == "train_epoch"] if "type" in table else pd.DataFrame()
    if not train_rows.empty and "loss" in train_rows:
        st.subheader("Training Loss")
        st.line_chart(train_rows.set_index("global_epoch")["loss"])
    eval_rows = table[table["type"] == "evaluation"] if "type" in table else pd.DataFrame()
    if not eval_rows.empty:
        st.subheader("Evaluation Summary")
        st.json(eval_rows.iloc[-1].dropna().to_dict())
        report_path = run_dir / "evaluation.json"
        if report_path.exists():
            st.download_button("Download evaluation JSON", report_path.read_text(), file_name="evaluation.json")
    samples_dir = run_dir / "evaluation_samples"
    if samples_dir.exists():
        samples = sorted(samples_dir.glob("*.jpg"))
        if samples:
            st.subheader("Prediction Review")
            st.caption("Ground truth is on the left; prediction is on the right.")
            st.image([str(path) for path in samples], caption=[path.name for path in samples], width=520)


page = st.sidebar.radio("Page", ["Explorer", "Runs"])
if page == "Explorer":
    show_explorer()
else:
    show_runs()
