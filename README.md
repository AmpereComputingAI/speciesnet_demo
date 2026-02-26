
Run with:
```bash
AIO_IMPLICIT_FP16_TRANSFORM_FILTER=".*" SKIP_FRAMES=15 uvicorn main:app  --host 0.0.0.0 --port 8000
```

# Docker
Run with
```bash
docker run -p 8000:8000 ghcr.io/amperecomputingai/speciesnet_demo:1.2
```

By default it will start the demo using port 8000 and run inference every 9th frame.
You can adjust these values if you want.

```bash
docker run -p 4321:8000 -e SKIP_FRAMES=11 ghcr.io/amperecomputingai/speciesnet_demo:1.2
```

## Custom videos
You can pass your own videos to the docker image with the `-v /path/to/your/videos:/workspace/videos` argument to the `docker run` command.

There should be either 1, 2 or 4 videos in that directory and they have to be in the `.mp4` format. For best results, the videos should all be 30 fps.

# Displaying the results
To see the results, open a web browser and go to `<your-ip>:<port>`, for example `localhost:8000`.
