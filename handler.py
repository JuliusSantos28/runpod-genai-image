"""RunPod serverless worker: Qwen-Image-Edit-2511 image edits (optional multi-angle LoRA) plus optional
InsightFace similarity. Model weights come from RunPod's cached-model mount; all images arrive in the request.

Input:
  image            base64 PNG/JPEG to edit (picture 1)
  extra_images     list of base64 images (picture 2, 3)
  prompt           text, used as given (camera_prompt is prepended only when set)
  negative, seed, steps, true_cfg_scale, guidance, flip, camera_prompt, lora_scale, width, height (optional)
  score_reference  optional base64 image: cosine similarity of the largest face to it is returned
Output: {"image": base64 PNG, "similarity", "face_fraction", "faces", "seconds", "model_snapshot", "gpu"}
"""
import base64
import hashlib
import io
import os
import time
from pathlib import Path

import runpod
import torch
from PIL import Image

PIPELINE = os.environ.get('PIPELINE', 'qwen')  # 'klein' runs a FLUX.2 Klein model (MODEL_ID) with reference images
MODEL_ID = os.environ.get('MODEL_ID', 'Qwen/Qwen-Image-Edit-2511')
CACHE = Path('/runpod-volume/huggingface-cache/hub')
ADAPTER = ('fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA', 'qwen-image-edit-2511-multiple-angles-lora.safetensors',
           'e3066224ab74263f4a5b6179cd1a3b0a15577e44')
UNCHAINED_LORA = ('Sentinel7/flux2', '2348977/2753532/KLEIN-Unchained-V2.safetensors',
                  '3375b906a95cc4b4853ccee35318975dd4c5ca68',
                  'bcf84b6b72ccef01f6e36565bd8b69ce8513907e80204476cb3e42098b453341')
PIPE, SNAPSHOT, FACE_APP, ADAPTER_LOADED, UNCHAINED_LOADED = None, None, None, False, False


def snapshot():
    """Path of the cached model snapshot (pinned revision if MODEL_REVISION is set and present)."""
    root = CACHE / ('models--' + MODEL_ID.replace('/', '--')) / 'snapshots'
    want = os.environ.get('MODEL_REVISION')
    if want and (root / want).is_dir():
        return root / want
    snaps = sorted(root.glob('*')) if root.is_dir() else []
    if not snaps:
        raise RuntimeError(f'{MODEL_ID} is not in the cached-model mount; set it as the endpoint model')
    return snaps[0]


def pipe():
    global PIPE, SNAPSHOT
    if PIPE is None:
        import diffusers
        cls = diffusers.Flux2KleinPipeline if PIPELINE == 'klein' else diffusers.QwenImageEditPlusPipeline
        SNAPSHOT = snapshot()
        PIPE = cls.from_pretrained(str(SNAPSHOT), torch_dtype=torch.bfloat16, local_files_only=True).to('cuda')
        PIPE.set_progress_bar_config(disable=True)
    return PIPE


def use_adapter(p, scale):
    global ADAPTER_LOADED
    if scale <= 0 and not ADAPTER_LOADED:
        return
    if not ADAPTER_LOADED:
        from huggingface_hub import hf_hub_download
        repo, name, rev = ADAPTER
        p.load_lora_weights(hf_hub_download(repo, name, revision=rev, cache_dir='/tmp/hf'), adapter_name='camera')
        ADAPTER_LOADED = True
    p.set_adapters('camera', adapter_weights=float(scale))


def use_unchained_lora(p, spec):
    """Load the pinned Klein 9B Base LoRA on opt-in; disable it on later requests that omit it."""
    global UNCHAINED_LOADED
    if not spec:
        if UNCHAINED_LOADED:
            p.set_adapters('unchained-v2', adapter_weights=0.0)
        return
    if PIPELINE != 'klein' or MODEL_ID != 'black-forest-labs/FLUX.2-klein-base-9B':
        raise ValueError('Unchained XXX v2 requires the FLUX.2 Klein Base 9B endpoint.')
    if spec.get('id') != 'unchained-v2':
        raise ValueError('Unknown external LoRA id.')
    strength = float(spec.get('strength', 0.8))
    if not 0.0 <= strength <= 2.0:
        raise ValueError('LoRA strength must be between 0 and 2.')
    if not UNCHAINED_LOADED:
        from huggingface_hub import hf_hub_download
        repo, filename, revision, expected_sha256 = UNCHAINED_LORA
        path = hf_hub_download(repo, filename, revision=revision, cache_dir='/tmp/hf')
        digest_hash = hashlib.sha256()
        with Path(path).open('rb') as f:
            for block in iter(lambda: f.read(1 << 20), b''):
                digest_hash.update(block)
        digest = digest_hash.hexdigest()
        if digest != expected_sha256:
            raise ValueError(f'Pinned LoRA checksum mismatch: {digest}')
        p.load_lora_weights(path, adapter_name='unchained-v2')
        UNCHAINED_LOADED = True
    p.set_adapters('unchained-v2', adapter_weights=strength)


def decode(b64):
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert('RGB')


def face_similarity(img, ref):
    global FACE_APP
    import cv2
    import numpy as np
    if FACE_APP is None:
        from insightface.app import FaceAnalysis
        FACE_APP = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition'],
                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        FACE_APP.prepare(ctx_id=0, det_size=(640, 640))

    def largest(im):
        found = FACE_APP.get(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR))
        return max(found, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])) if found else None, found
    r, _ = largest(ref)
    f, found = largest(img)
    if r is None or f is None:
        return None, None, len(found)
    x0, y0, x1, y1 = f.bbox
    frac = float((x1 - x0) * (y1 - y0) / (img.width * img.height))
    return round(float(np.dot(r.normed_embedding, f.normed_embedding)), 4), round(frac, 4), len(found)


def handler(job):
    a = job['input']
    t0 = time.monotonic()
    p = pipe()
    if PIPELINE != 'klein':
        use_adapter(p, float(a.get('lora_scale', 0)))
    else:
        use_unchained_lora(p, a.get('unchained_lora'))
    src = decode(a['image'])
    if a.get('flip'):
        src = src.transpose(Image.FLIP_LEFT_RIGHT)
    images = [src] + [decode(x) for x in a.get('extra_images', [])]
    prompt = f"{a['camera_prompt']}. {a['prompt']}" if a.get('camera_prompt') else a['prompt']
    size = ({'width': int(a['width']) // 16 * 16, 'height': int(a['height']) // 16 * 16}
            if a.get('width') and a.get('height') else {})
    gen = torch.Generator('cuda').manual_seed(int(a.get('seed', 0)))
    if PIPELINE == 'klein':
        out = p(image=images, prompt=prompt, generator=gen, guidance_scale=float(a.get('guidance', 1.0)),
                num_inference_steps=int(a.get('steps', 4)), **size).images[0]
    else:
        out = p(image=images, prompt=prompt, negative_prompt=a.get('negative', ' '), generator=gen,
                true_cfg_scale=float(a.get('true_cfg_scale', 4.0)), guidance_scale=float(a.get('guidance', 1.0)),
                num_inference_steps=int(a.get('steps', 40)), num_images_per_prompt=1, **size).images[0]
    if a.get('flip'):
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
    buf = io.BytesIO()
    out.save(buf, 'PNG')
    result = {'image': base64.b64encode(buf.getvalue()).decode(), 'full_prompt': prompt,
              'model_snapshot': SNAPSHOT.name, 'gpu': torch.cuda.get_device_name(0), 'dimensions': list(out.size)}
    if a.get('score_reference'):
        result['similarity'], result['face_fraction'], result['faces'] = face_similarity(out, decode(a['score_reference']))
    result['seconds'] = round(time.monotonic() - t0, 2)
    return result


if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
