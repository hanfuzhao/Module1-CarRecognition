# Deployment

The app is deployed as a **Docker Space on Hugging Face** (chosen over a 512 MB
free tier because a torch + ResNet50 process needs more memory).

- **Live app:** https://HanfuZhao781-car-recognition.hf.space
- **Space page:** https://huggingface.co/spaces/HanfuZhao781/car-recognition
- **Source:** https://github.com/hanfuzhao/Module1-CarRecognition

The Space serves the same Flask app (`main.py`) via the `Dockerfile` in this
repo. It runs **inference only** — the trained MLP head ships in `models/`, and
the frozen ResNet50 backbone is baked into the image at build time so the first
request is fast.

## How it is built
`Dockerfile` (identical to the one running on the Space):
1. installs CPU PyTorch + the rest of `requirements.txt`,
2. copies the app and the trained model,
3. pre-downloads the ResNet50 ImageNet weights into the image,
4. serves with `gunicorn` on port 7860 (HF Spaces' expected port).

## Reproduce / redeploy
```bash
# 1. one-time: authenticate with a write token
huggingface-cli login

# 2. create a Docker Space (once)
python -c "from huggingface_hub import create_repo; \
  create_repo('<user>/car-recognition', repo_type='space', space_sdk='docker', exist_ok=True)"

# 3. push the app (Dockerfile + main.py + scripts/ + templates/ + static/ +
#    requirements.txt + models/ + data/raw/stanford-cars/metadata.json)
python -c "from huggingface_hub import upload_folder; \
  upload_folder(repo_id='<user>/car-recognition', repo_type='space', folder_path='.')"
```
The Space rebuilds automatically on each push; `get_space_runtime(...).stage`
reports `RUNNING` when live.

## Run locally
```bash
pip install -r requirements.txt
python main.py            # http://localhost:5000  (PORT env overrides)
```
Or with Docker:
```bash
docker build -t car-recognition .
docker run -p 7860:7860 car-recognition   # http://localhost:7860
```

## Health check
```bash
curl https://HanfuZhao781-car-recognition.hf.space/health
# {"num_classes":196,"status":"ok"}
```
