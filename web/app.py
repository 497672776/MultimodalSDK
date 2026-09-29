#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the MultimodalSDK project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MultimodalSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
"""Web front end for MultimodalSDK.

Exposes the SDK's media preprocessing pipeline (image/video/audio decoding,
Qwen2-VL resize+normalize, SCC vision token compression) plus a vision-language
chat box backed by a local llama.cpp ``llama-server``.

Run it with ``python app.py`` or through ``web/run.sh``.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import platform
import shutil
import stat
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image as PILImage

import mm
from mm.acc._impl import acc as acc_backend

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
WORK_DIR = Path(os.environ.get("MM_WEB_WORK_DIR", str(Path.home() / "mm-sdk-web" / "work")))
LLAMA_URL = os.environ.get("MM_WEB_LLAMA_URL", "http://127.0.0.1:18810")
MAX_UPLOAD_BYTES = int(os.environ.get("MM_WEB_MAX_UPLOAD_MB", "64")) * 1024 * 1024
HOST = os.environ.get("MM_WEB_HOST", "0.0.0.0")
PORT = int(os.environ.get("MM_WEB_PORT", "8090"))

# Qwen2-VL / Qwen3-VL preprocessing constants.
QWEN_MEAN = [0.5, 0.5, 0.5]
QWEN_STD = [0.5, 0.5, 0.5]
PATCH_SIZE = 28
MERGE_SIZE = 2
MIN_PIXELS = PATCH_SIZE * PATCH_SIZE * 4
MAX_PIXELS = 1024 * 1024
IMAGE_MIN_SIDE = 10
IMAGE_MAX_SIDE = 8192
PREVIEW_MAX_SIDE = 512

app = FastAPI(title="MultimodalSDK web demo", version="1.0.0")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _save_upload(upload: UploadFile, suffix: str) -> Path:
    """Persist an upload in the work directory using SDK friendly permissions."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    target = WORK_DIR / f"{int(time.time() * 1000)}-{os.getpid()}{suffix}"
    size = 0
    with target.open("wb") as handle:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                handle.close()
                target.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail=f"upload exceeds {MAX_UPLOAD_BYTES} bytes")
            handle.write(chunk)
    if size == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="empty upload")
    # acc only reads files that grant nothing to "other".
    os.chmod(target, stat.S_IRUSR | stat.S_IRGRP)
    return target


def _cleanup(*paths) -> None:
    for path in paths:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass


def _encode_png(array: np.ndarray, max_side: int = PREVIEW_MAX_SIDE) -> str:
    image = PILImage.fromarray(np.ascontiguousarray(array, dtype=np.uint8), "RGB")
    if max_side and max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), PILImage.BILINEAR)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _to_uint8(tensor: np.ndarray, mean, std) -> np.ndarray:
    """Undo normalization for preview rendering, keep NCHW/NHWC layout."""
    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32)
    array = np.asarray(tensor, dtype=np.float32)
    if array.ndim == 4:
        array = array[0]
    if array.shape[0] == 3 and array.shape[-1] != 3:  # CHW -> HWC
        array = array.transpose(1, 2, 0)
        reshape = (1, 1, 3)
    else:
        reshape = (1, 1, 3)
    array = array * std_arr.reshape(reshape) + mean_arr.reshape(reshape)
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)


def _load_mm_image(path: Path):
    """Decode a still image into mm.Image, mirroring the SDK's own entry points."""
    with PILImage.open(path) as raw:
        image_format = raw.format
        size = raw.size
        rgb = np.asarray(raw.convert("RGB"), dtype=np.uint8)

    # Qwen/acc resize limits, keep the demo usable for very large uploads.
    if max(size) > IMAGE_MAX_SIDE or min(size) < IMAGE_MIN_SIDE:
        scale = IMAGE_MAX_SIDE / max(size) if max(size) > IMAGE_MAX_SIDE else 1.0
        target = (max(IMAGE_MIN_SIDE, int(size[0] * scale)), max(IMAGE_MIN_SIDE, int(size[1] * scale)))
        rgb = np.asarray(PILImage.fromarray(rgb, "RGB").resize(target, PILImage.BICUBIC), dtype=np.uint8)
        source = "PIL"
    else:
        source = "PIL"

    if image_format == "JPEG" and source == "PIL" and rgb.shape[:2] == (size[1], size[0]):
        # The native decoder path (libjpeg-turbo) is exercised for native-size JPEGs.
        return mm.Image.open(str(path)), "mm.Image.open (JPEG)"
    return mm.Image.from_pillow(PILImage.fromarray(rgb, "RGB")), "mm.Image.from_pillow"


def _smart_resize(height: int, width: int, factor: int = PATCH_SIZE, min_pixels: int = MIN_PIXELS,
                  max_pixels: int = MAX_PIXELS):
    """Qwen2-VL style resize so that both sides are multiples of ``factor``."""
    if height < 1 or width < 1:
        raise ValueError("height and width are required to be positive")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _llama_request(path: str, payload: dict | None = None, timeout: int = 15):
    url = LLAMA_URL.rstrip("/") + path
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _llama_status() -> dict:
    status = {"url": LLAMA_URL, "online": False}
    try:
        health = _llama_request("/health", timeout=5)
        status["online"] = health.get("status") == "ok"
    except Exception as e:  # noqa: BLE001 - any failure means "not reachable"
        status["error"] = f"{type(e).__name__}: {e}"
        return status
    try:
        props = _llama_request("/props", timeout=10)
        status["model"] = Path(props.get("model_path", "")).name
        status["chat_template"] = bool(props.get("chat_template"))
        status["n_ctx"] = props.get("default_generation_settings", {}).get("n_ctx")
    except Exception as e:  # noqa: BLE001
        status["props_error"] = f"{type(e).__name__}: {e}"
    return status


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.get("/api/status")
def api_status():
    return {
        "sdk": {
            "path": str(Path(mm.__file__).resolve().parent),
            "acc_backend": getattr(acc_backend, "__name__", str(acc_backend)),
            "acc_native": not getattr(acc_backend, "__name__", "").endswith("cpu_backend"),
            "frame_selectors": [cls.__name__ for cls in (mm.KFrameSelector, mm.KRangFrameSelector)],
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": _has_torch(),
            "ffmpeg": _has_ffmpeg(),
            "work_dir": str(WORK_DIR),
        },
        "llama": _llama_status(),
    }


@app.post("/api/image/preprocess")
def api_image_preprocess(
    file: UploadFile = File(...),
    target: str = Form("qwen"),
    size: int = Form(28),
):
    """Decode an image with mm.acc and run the Qwen2-VL resize+normalize step."""
    suffix = Path(file.filename or "image").suffix.lower() or ".png"
    path = _save_upload(file, suffix)
    try:
        started = time.perf_counter()
        mm_image, decode_note = _load_mm_image(path)
        decode_ms = (time.perf_counter() - started) * 1000.0
        original = np.asarray(mm_image.numpy())

        if target == "square":
            height = width = max(28, int(size) // 28 * 28)
        else:
            height, width = _smart_resize(mm_image.height, mm_image.width)

        result = {
            "decode_source": decode_note,
            "decode_ms": round(decode_ms, 2),
            "original": {
                "size": mm_image.size,
                "format": mm_image.format.name,
                "dtype": mm_image.dtype.name,
                "nbytes": mm_image.nbytes,
                "preview": _encode_png(original),
            },
            "target": {"height": height, "width": width, "patch_size": PATCH_SIZE,
                       "patches": (height // PATCH_SIZE) * (width // PATCH_SIZE)},
        }

        if not _has_torch():
            result["warning"] = "torch is not installed, using the acc operator path only"
            resized = acc_backend.Qwen2VLProcessor.Preprocess(
                [mm_image], QWEN_MEAN, QWEN_STD, width, height)[0]
            array = np.asarray(resized.numpy())
            result["tensor"] = {"shape": list(array.shape), "dtype": "float32",
                                "nbytes": int(array.nbytes)}
            result["preview"] = _encode_png(_to_uint8(array, QWEN_MEAN, QWEN_STD))
            return result

        import torch

        from mm.core.processor import resize_and_normalize

        frames = torch.from_numpy(np.ascontiguousarray(original)).permute(2, 0, 1).unsqueeze(0)
        started = time.perf_counter()
        normalized = resize_and_normalize(frames, height, width, QWEN_MEAN, QWEN_STD)
        preprocess_ms = (time.perf_counter() - started) * 1000.0

        array = normalized.numpy()
        result["preprocess_ms"] = round(preprocess_ms, 2)
        result["tensor"] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "nbytes": int(array.nbytes),
            "min": float(array.min()),
            "max": float(array.max()),
            "mean": float(array.mean()),
        }
        result["preview"] = _encode_png(_to_uint8(array.reshape(1, 3, height, width), QWEN_MEAN, QWEN_STD))
        return result
    finally:
        _cleanup(path)


@app.post("/api/video/decode")
def api_video_decode(
    file: UploadFile = File(...),
    sample_num: int = Form(8),
    frame_indices: str = Form(""),
):
    """Decode a video with mm.acc.video_decode (ffmpeg backed on the CPU backend)."""
    suffix = Path(file.filename or "video").suffix.lower() or ".mp4"
    path = _save_upload(file, suffix)
    try:
        indices = set()
        if frame_indices.strip():
            indices = {int(part) for part in frame_indices.replace(",", " ").split() if part.strip().isdigit()}

        started = time.perf_counter()
        frames = mm.video_decode(str(path), "cpu", indices or None, int(sample_num))
        decode_ms = (time.perf_counter() - started) * 1000.0

        previews = []
        for index, frame in enumerate(frames):
            previews.append({
                "index": index,
                "size": frame.size,
                "preview": _encode_png(np.asarray(frame.numpy()), max_side=256),
            })

        return {
            "requested": {"frame_indices": sorted(indices), "sample_num": int(sample_num)},
            "decoded": len(frames),
            "decode_ms": round(decode_ms, 2),
            "per_frame_ms": round(decode_ms / len(frames), 2) if frames else None,
            "frames": previews,
        }
    finally:
        _cleanup(path)


@app.post("/api/audio/load")
def api_audio_load(
    file: UploadFile = File(...),
    sample_rate: int = Form(16000),
):
    """Load a wav file with mm.acc.load_audio."""
    suffix = Path(file.filename or "audio").suffix.lower() or ".wav"
    if suffix != ".wav":
        raise HTTPException(status_code=400, detail="mm.load_audio only accepts .wav inputs")
    path = _save_upload(file, suffix)
    try:
        sr = int(sample_rate) if sample_rate else None
        started = time.perf_counter()
        tensor, loaded_sr = mm.load_audio(str(path), sr)
        load_ms = (time.perf_counter() - started) * 1000.0
        wave = np.asarray(tensor.numpy()).reshape(-1)
        duration = float(wave.size) / float(loaded_sr) if loaded_sr else 0.0
        points = 512
        if wave.size:
            step = max(1, wave.size // points)
            envelope = wave[: step * points].reshape(-1, step)
            waveform = np.concatenate([envelope.max(axis=1), envelope.min(axis=1)]).round(5).tolist()
        else:
            waveform = []
        return {
            "sample_rate": loaded_sr,
            "samples": int(wave.size),
            "duration_s": round(duration, 3),
            "dtype": str(wave.dtype),
            "peak": float(np.abs(wave).max()) if wave.size else 0.0,
            "rms": float(np.sqrt(np.mean(wave ** 2))) if wave.size else 0.0,
            "load_ms": round(load_ms, 2),
            "waveform": waveform,
        }
    finally:
        _cleanup(path)


@app.post("/api/scc/compress")
def api_scc_compress(
    file: UploadFile = File(...),
    ratio: float = Form(0.5),
    tau: float = Form(0.98),
    epsilon: float = Form(0.05),
):
    """Compress a grid of patch descriptors with the SCC token compression.

    True vision-encoder embeddings are not available on the CPU backend, so the
    demo derives one descriptor per 28x28 image patch (mean/std colour plus a
    downsampled grey pattern) and runs the SDK's SCC algorithm on them.
    """
    if not _has_torch():
        raise HTTPException(status_code=501, detail="SCC needs torch, which is not installed")
    import torch

    from mm.core.scc import scc_compress_to_target, scc_shrink, scc_should_run

    suffix = Path(file.filename or "image").suffix.lower() or ".png"
    path = _save_upload(file, suffix)
    try:
        mm_image, decode_note = _load_mm_image(path)
        height, width = _smart_resize(mm_image.height, mm_image.width, max_pixels=512 * 512)
        resized = np.asarray(
            PILImage.fromarray(np.asarray(mm_image.numpy()), "RGB").resize((width, height), PILImage.BICUBIC))
        grid_h, grid_w = height // PATCH_SIZE, width // PATCH_SIZE

        descriptors = []
        for row in range(grid_h):
            for col in range(grid_w):
                patch = resized[row * PATCH_SIZE:(row + 1) * PATCH_SIZE,
                                col * PATCH_SIZE:(col + 1) * PATCH_SIZE].astype(np.float32) / 255.0
                grey = patch.mean(axis=2)
                small = np.asarray(PILImage.fromarray((grey * 255).astype(np.uint8)).resize((7, 7), PILImage.BILINEAR),
                                   dtype=np.float32).reshape(-1) / 255.0
                descriptors.append(np.concatenate([patch.reshape(-1, 3).mean(axis=0),
                                                   patch.reshape(-1, 3).std(axis=0), small]))
        features = torch.from_numpy(np.ascontiguousarray(np.stack(descriptors))).float()
        norm = features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        features = features / norm

        n_tokens = int(features.shape[0])
        k_target = scc_shrink(n_tokens, float(ratio))
        started = time.perf_counter()
        compressed = scc_compress_to_target(features, k_target, 1, 0, float(tau), float(epsilon))
        scc_ms = (time.perf_counter() - started) * 1000.0

        def heatmap(tensor, max_side=256):
            data = tensor.detach().numpy()
            data = data - data.min()
            scale = data.max() if data.max() > 0 else 1.0
            image = PILImage.fromarray((data / scale * 255.0).astype(np.uint8), "L")
            ratio_side = max(1, max_side // max(image.size))
            image = image.resize((image.width * ratio_side, image.height * ratio_side), PILImage.NEAREST)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

        return {
            "decode_source": decode_note,
            "patch_grid": [grid_h, grid_w],
            "feature_dim": int(features.shape[1]),
            "tokens": n_tokens,
            "target_tokens": k_target,
            "keep_ratio": round(k_target / n_tokens, 4),
            "should_run": bool(scc_should_run(n_tokens)),
            "tau": float(tau),
            "epsilon": float(epsilon),
            "scc_ms": round(scc_ms, 2),
            "output_shape": list(compressed.shape),
            "features_heatmap": heatmap(features),
            "compressed_heatmap": heatmap(compressed),
        }
    finally:
        _cleanup(path)


@app.post("/api/vlm/chat")
def api_vlm_chat(
    file: UploadFile = File(...),
    question: str = Form("请用中文描述这张图片的内容。"),
    max_tokens: int = Form(256),
    temperature: float = Form(0.2),
    max_side: int = Form(768),
):
    """Answer a question about an image using the local llama-server."""
    suffix = Path(file.filename or "image").suffix.lower() or ".png"
    path = _save_upload(file, suffix)
    try:
        started = time.perf_counter()
        mm_image, decode_note = _load_mm_image(path)
        array = np.asarray(mm_image.numpy())
        pil_image = PILImage.fromarray(array, "RGB")
        if max_side and max(pil_image.size) > max_side:
            scale = max_side / max(pil_image.size)
            pil_image = pil_image.resize((max(1, int(pil_image.width * scale)), max(1, int(pil_image.height * scale))),
                                         PILImage.BICUBIC)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        data_uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        prepare_ms = (time.perf_counter() - started) * 1000.0

        payload = {
            "model": "local",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "stream": False,
        }
        started = time.perf_counter()
        try:
            response = _llama_request("/v1/chat/completions", payload, timeout=600)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:500]
            raise HTTPException(status_code=502, detail=f"llama-server error {e.code}: {detail}") from e
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"llama-server unreachable at {LLAMA_URL}: {e}") from e
        infer_ms = (time.perf_counter() - started) * 1000.0

        choice = (response.get("choices") or [{}])[0]
        usage = response.get("usage", {})
        return {
            "answer": choice.get("message", {}).get("content", ""),
            "finish_reason": choice.get("finish_reason"),
            "prepare_ms": round(prepare_ms, 2),
            "infer_ms": round(infer_ms, 2),
            "decode_source": decode_note,
            "image_size": list(pil_image.size),
            "usage": usage,
            "tokens_per_s": round(usage.get("completion_tokens", 0) / (infer_ms / 1000.0), 2) if infer_ms else None,
            "llama_url": LLAMA_URL,
            "preview": _encode_png(np.asarray(pil_image), max_side=384),
        }
    finally:
        _cleanup(path)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
