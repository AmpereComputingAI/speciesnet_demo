# hadolint ignore=DL3007
FROM amperecomputingai/pytorch:latest

# --- Environment ---
ENV SKIP_FRAMES=9 \
    AIO_IMPLICIT_FP16_TRANSFORM_FILTER=".*" \
    UVICORN_PORT=8000

# --- System deps + FFmpeg build (single layer, cache cleaned) ---
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
        nasm libx264-dev libx265-dev libsm6 libxext6 libgl1 jq \
    && rm -rf /var/lib/apt/lists/*

# --- Clone & extract speciesnet ---
RUN git clone --depth 1 https://github.com/google/cameratrapai.git \
    && mv cameratrapai/speciesnet /workspace/speciesnet \
    && rm -rf cameratrapai

# --- Build FFmpeg from source ---
WORKDIR /workspace/ffmpeg
RUN git clone --depth 1 https://github.com/FFmpeg/FFmpeg.git . \
    && ./configure \
        --arch=aarch64 \
        --cpu=native \
        --enable-neon \
        --enable-gpl \
        --enable-libx265 \
        --enable-libx264 \
    && make -j"$(nproc)" \
    && make install \
    && rm -rf /workspace/ffmpeg

# --- Python dependencies ---
# hadolint ignore=DL3013
RUN python3 -m pip install --no-cache-dir \
        "fastapi[standard]" \
        pillow \
        absl-py \
        humanfriendly \
        cloudpathlib \
        huggingface_hub \
        kagglehub \
        pandas \
        ultralytics \
        seaborn \
        reverse_geocoder \
        onnx2torch \
        "numpy==1.26.4" \
    && python3 -m pip install --no-cache-dir yolov5 --no-deps

WORKDIR /workspace

RUN mkdir classifier
WORKDIR /workspace/classifier
RUN curl -L -o classifier.tar.gz\
  https://www.kaggle.com/api/v1/models/google/speciesnet/pyTorch/v4.0.2a/1/download && tar -xvf classifier.tar.gz && rm classifier.tar.gz
RUN curl -L -o md_v1000.0.0-spruce.pt https://github.com/agentmorris/MegaDetector/releases/download/v1000.0/md_v1000.0.0-spruce.pt
RUN jq '.detector = "md_v1000.0.0-spruce.pt"' info.json > tmp.json && mv tmp.json info.json
WORKDIR /workspace

COPY main.py .
RUN mkdir videos
COPY videos/ videos/

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0"]
