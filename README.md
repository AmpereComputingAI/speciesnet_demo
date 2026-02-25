
Run with:
```bash
AIO_IMPLICIT_FP16_TRANSFORM_FILTER=".*" SKIP_FRAMES=15 uvicorn main:app  --host 0.0.0.0 --port 8000
```

# Docker
Run with
```bash
docker run -p 8000:8000 ghcr.io/amperecomputingai/speciesnet_demo:1.1
```

By default it will start the demo using port 8000 and run inference every 9th frame.
You can adjust these values if you want.

```bash
docker run -p 4321:8000 -e SKIP_FRAMES=11 ghcr.io/amperecomputingai/speciesnet_demo:1.1
```
