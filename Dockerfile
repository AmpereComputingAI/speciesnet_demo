FROM amperecomputingai/pytorch

ENV SKIP_FRAMES=5
ENV AIO_IMPLICIT_FP16_TRANSFORM_FILTER=".*"
ENV UVICORN_PORT=8000

COPY main.py .
COPY bearid-demo-raw.mp4 .

RUN git clone --depth 1 https://github.com/google/cameratrapai.git
RUN mv cameratrapai/speciesnet .
RUN git clone --depth 1 https://github.com/FFmpeg/FFmpeg.git ffmpeg
RUN apt update && apt install -y nasm libx264-dev libx265-dev libsm6 libxext6 libgl1
WORKDIR /workspace/ffmpeg
RUN ./configure --arch=aarch64 --cpu=native --enable-neon --enable-gpl --enable-libx265 --enable-libx264
RUN make -j
RUN make install
WORKDIR /workspace
RUN python3 -m pip install "fastapi[standard]" pillow absl-py humanfriendly cloudpathlib huggingface_hub kagglehub pandas ultralytics seaborn reverse_geocoder onnx2torch numpy==1.26.4
RUN python3 -m pip install yolov5 --no-deps

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0"]
