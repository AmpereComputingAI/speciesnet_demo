import multiprocessing
import os
import cv2
import types
from speciesnet.classifier import SpeciesNetClassifier
import torch
import numpy as np
from speciesnet.detector import SpeciesNetDetector
from speciesnet.utils import BBox
import PIL
import subprocess
import shutil

from fastapi import FastAPI, Response
from fastapi.responses import FileResponse, HTMLResponse

SKIP_FRAMES = int(os.environ.get("SKIP_FRAMES", 5))
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.25))
app = FastAPI()
STREAM_DIR = "/tmp/stream"
shutil.rmtree(STREAM_DIR, ignore_errors=True)
os.makedirs(STREAM_DIR, exist_ok=True)


def run_inference(video, index, assigned_cores, stop_event):
    pid = os.getpid()
    try:
        # psutil.Process(pid).cpu_affinity(assigned_cores)
        print(f"Process {pid} pinned to cores: {assigned_cores}")
    except Exception as e:
        print(f"Failed to assign cores to process {pid}")

    os.environ["AIO_NUMA_CPUS"] = " ".join((str(c) for c in assigned_cores))
    torch.set_num_threads(len(assigned_cores))

    detector = SpeciesNetDetector("kaggle:google/speciesnet/pyTorch/v4.0.2a/1")
    classifier = SpeciesNetClassifier("kaggle:google/speciesnet/pyTorch/v4.0.2a/1")
    detector.model = torch.compile(
        detector.model,
        backend="aio",
        options={
            "modelname": detector.model.__self__._get_name()
            if isinstance(detector.model, types.MethodType)
            else detector.model._get_name()
        },
    )
    print(f"Process {pid} done compiling detector")

    img_pil = PIL.Image.new(mode="RGB", size=(1000, 1000))
    example_input = torch.from_numpy(
        np.stack([(classifier.preprocess(img_pil).arr) / 255], axis=0, dtype=np.float32)
    )
    classifier.model = torch.jit.freeze(
        torch.jit.trace(
            classifier.model,
            example_inputs=[example_input],
        )
    )
    print(f"Process {pid} done tracing classifier")
    print(f"Process {pid} - {video}")
    cap = cv2.VideoCapture(video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    #    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
    #    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2
    width = 1920 // 2
    height = 1080 // 2
    previous_predictions = None
    frame_count = 0

    os.makedirs(f"{STREAM_DIR}/{index}", exist_ok=True)
    m3u8_path = f"{STREAM_DIR}/{index}/stream.m3u8"
    ffmpeg_cmd = [
        "./ffmpeg/ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps}",
        "-i",
        "pipe:0",
        "-i",
        video,
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-c:a",
        "copy",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx265",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-threads",
        "16",
        "-f",
        "hls",
        "-hls_time",
        "1",
        "-hls_list_size",
        "2",
        "-hls_flags",
        "delete_segments+append_list+independent_segments",
        m3u8_path,
    ]
    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            # Loop to the start if the video is finished
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        print(f"Process {pid} - {'full' if frame_count % SKIP_FRAMES == 0 else 'skip'}")
        frame = cv2.resize(frame, (width, height))
        pil_frame = PIL.Image.fromarray(frame)
        if frame_count % SKIP_FRAMES == 0 or previous_predictions is None:
            img_det = detector.preprocess(pil_frame)
            det_out = detector.predict("", img_det)
            detections = det_out["detections"]
            bboxes = [
                BBox(*det["bbox"])
                for det in detections
                if det["conf"] > CONFIDENCE_THRESHOLD
            ]
            classifier_frames = []
            for bbox in bboxes:
                classifier_frames.append(classifier.preprocess(pil_frame, [bbox]))
            classifier_outs = classifier.batch_predict(
                [str(i) for i in range(len(classifier_frames))],
                classifier_frames,
            )
            classes = []
            for classifier_out in classifier_outs:
                classes.append(
                    classifier_out["classifications"]["classes"][0].split(";")[-1]
                )

            bboxes = [
                [
                    bbox.xmin * frame.shape[1],
                    bbox.ymin * frame.shape[0],
                    (bbox.xmin + bbox.width) * frame.shape[1],
                    (bbox.ymin + bbox.height) * frame.shape[0],
                ]
                for bbox in bboxes
            ]
        else:
            bboxes, classes = previous_predictions
        frame_count += 1
        draw = PIL.ImageDraw.Draw(pil_frame)
        for bbox, classification in zip(bboxes, classes):
            if classification == "blank":
                continue
            draw.rectangle(bbox, width=10)
            draw.text(
                bbox[:2],
                classification,
                font_size=72,
                fill="white",
                stroke_fill="black",
                stroke_width=10,
                anchor="lb",
            )
        previous_predictions = (bboxes, classes)
        frame = np.asarray(pil_frame)
        resized_frame = cv2.resize(frame, (width, height))
        process.stdin.write(resized_frame.tobytes())
    cap.release()
    process.stdin.close()


def start(videos):
    stop_event = multiprocessing.Event()
    processes = []
    cores_per_process = multiprocessing.cpu_count() // len(videos)
    for i, v in enumerate(videos):
        start_core = i * cores_per_process
        assigned_cores = list(range(start_core, start_core + cores_per_process))
        p = multiprocessing.Process(
            target=run_inference,
            args=(v, i, assigned_cores, stop_event),
            daemon=True,
        )
        processes.append(p)
    for p in processes:
        p.start()


available_files = [
    ("Bears", "bearid-demo-raw.mp4"),
    # ("Clips", "usfq-demo-clips.mp4"),
    # ("Pandas", "pandas.mkv"),
    # ("Red Pandas", "redpandas.mp4"),
]


@app.get("/", response_class=HTMLResponse)
def index():
    players = "\n".join(
        [
            f"""
    <video id="v{i}" autoplay muted style="width:100%"></video>
    <script>
    </script>
    """
            for i in range(len(available_files))
        ]
    )

    grids = {1: ("1fr", "1fr"), 2: ("1fr 1fr", "1fr"), 4: ("1fr 1fr", "1fr 1fr")}
    grid_style = grids[len(available_files)]
    grid_template = (
        f"grid-template-columns: {grid_style[0]}; grid-template-rows: {grid_style[1]};"
    )

    scripts = "\n".join(
        [
            f"""
            (function() {{
            var v = document.getElementById('v{i}');
            var hls = new Hls({{ lowLatencyMode: true }});
            function tryLoad() {{
                fetch('/stream/{i}/stream.m3u8').then(r => {{
                if (r.ok) {{
                    hls.loadSource('/stream/{i}/stream.m3u8');
                    hls.attachMedia(v);
                }} else {{
                    setTimeout(tryLoad, 1000);
                }}
                }});
            }}
            tryLoad();
            }})();
            """
            for i in range(len(available_files))
        ]
    )

    return f"""
    <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
    <div style="display:grid; {grid_template} height:100vh; gap:10px; padding:10px; box-sizing: border-box">
            {players}
    </div>
    <script>{scripts}</script>
    """


@app.get("/stream/{index}/{filename}")
def serve_stream(index: int, filename: str):
    path = f"{STREAM_DIR}/{index}/{filename}"
    if not os.path.exists(path):
        return Response(status_code=404)
    return FileResponse(path)


@app.on_event("startup")
def startup():
    videos = [v[1] for v in available_files]
    start(videos)
