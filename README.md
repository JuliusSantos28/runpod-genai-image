# runpod-genai-image

**This repository only holds the install recipe for a Docker image.** There is no application code, model,
dataset or content here, so there is nothing else to look for.

The image is the public `runpod/pytorch` 2.8 / CUDA 12.8 base with a few open-source packages preinstalled, so a
rented GPU pod starts ready instead of spending ~20 minutes on `pip install`:

- [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) at a pinned commit, in `/opt/train-venv`
- Hugging Face Diffusers / Transformers (pinned versions in `requirements-qwen.txt`), in `/opt/qwen-venv`
- InsightFace with its public `buffalo_l` model files
- `handler.py`: a generic RunPod serverless worker for Qwen-Image-Edit-2511 or FLUX.2 Klein edits (weights come from RunPod's
  cached-model mount; images are sent with each request)

GitHub Actions builds it on each push to `main` and publishes `ghcr.io/juliussantos28/runpod-genai`
(`latest` and the short commit SHA). `constraints.txt` keeps the base image's PyTorch version during installs.
