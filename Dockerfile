# A RunPod PyTorch base image with common open-source image-generation packages preinstalled, so a new GPU pod
# skips ~20 minutes of pip installs. Contains software only: no model weights, data or configuration.
FROM docker.io/runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35
LABEL org.opencontainers.image.source=https://github.com/JuliusSantos28/runpod-genai-image
LABEL org.opencontainers.image.description="RunPod PyTorch 2.8 base with AI Toolkit, Diffusers and InsightFace preinstalled"
ENV PIP_NO_CACHE_DIR=1 NO_ALBUMENTATIONS_UPDATE=1 PYTHONUNBUFFERED=1
ARG AI_TOOLKIT_REV=ecee894ed2b1f3716d9d7326693061ec1a3105bb
COPY constraints.txt requirements-qwen.txt handler.py /opt/pod/

# AI Toolkit at a pinned revision in its own venv
RUN git clone https://github.com/ostris/ai-toolkit.git /opt/ai-toolkit \
 && git -C /opt/ai-toolkit checkout "$AI_TOOLKIT_REV" \
 && git -C /opt/ai-toolkit submodule update --init --recursive \
 && python -m venv --system-site-packages /opt/train-venv \
 && /opt/train-venv/bin/pip install -r /opt/ai-toolkit/requirements.txt -c /opt/pod/constraints.txt \
 && /opt/train-venv/bin/pip install insightface onnxruntime-gpu==1.22.0 -c /opt/pod/constraints.txt

# Diffusers venv (Qwen Image Edit support)
RUN python -m venv --system-site-packages /opt/qwen-venv \
 && /opt/qwen-venv/bin/pip install -r /opt/pod/requirements-qwen.txt insightface onnxruntime-gpu==1.22.0 runpod==1.9.1 \
      -c /opt/pod/constraints.txt

# InsightFace buffalo_l face model cached in the image, then record what got installed
RUN /opt/train-venv/bin/python -c "from insightface.utils.storage import ensure_available; ensure_available('models', 'buffalo_l', root='~/.insightface')" \
 && ls /root/.insightface/models/buffalo_l \
 && /opt/train-venv/bin/pip freeze > /opt/pod/train-freeze.txt \
 && /opt/qwen-venv/bin/pip freeze > /opt/pod/qwen-freeze.txt \
 && touch /opt/pod/ready

# Serverless use: set the endpoint template's start command to
#   /opt/qwen-venv/bin/python /opt/pod/handler.py
# and its cached model to Qwen/Qwen-Image-Edit-2511. Pods ignore this and start normally.
