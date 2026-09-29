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
"""Pure-Python CPU backend for the ``mm.acc`` acceleration layer.

The production acceleration layer is the native ``_acc``/``libcore.so`` pair
built from AccSDK, which is only available on Ascend/aarch64 platforms. This
module implements the subset of that pybind11 API surface used by
``mm.acc.wrapper`` (plus ``Qwen2VLProcessor`` used by ``mm.core``) with NumPy,
Pillow and ffmpeg, so that MultimodalSDK runs on platforms without the native
library, for example RISC-V boards.

Behaviour mirrors the native operator semantics:
  * ``Image``/``Tensor`` conversions are zero-copy views over the input buffer.
  * ``Image.resize``/``Image.crop`` are bit-exact with Pillow's BICUBIC
    implementations, matching the native operator.
  * ``Image.to_tensor`` produces float32 in ``[0, 1]`` with a leading batch
    dimension, i.e. ``(1, C, H, W)`` for NCHW and ``(1, H, W, C)`` for NHWC.
  * ``normalize`` matches ``torchvision.transforms.Normalize``.
  * Video and audio decoding are delegated to the ffmpeg/ffprobe binaries and
    the Python standard library.
"""

import json
import os
import shutil
import stat
import subprocess

import numpy as np

# Enum values are kept identical to the native acc enums, because the wrappers
# convert their own Python enums to plain integers before calling into acc.
DataType_INT8 = 2
DataType_UINT8 = 4
DataType_FLOAT32 = 0

TensorFormat_ND = 2
TensorFormat_NHWC = 1
TensorFormat_NCHW = 0

ImageFormat_RGB = 12
ImageFormat_BGR = 13
ImageFormat_RGB_PLANAR = 69
ImageFormat_BGR_PLANAR = 70

DeviceMode_CPU = 0

Interpolation_BICUBIC = 2

_DEVICE_CPU = b"cpu"

_NUMPY_TO_DATATYPE = {
    np.dtype(np.int8): DataType_INT8,
    np.dtype(np.uint8): DataType_UINT8,
    np.dtype(np.float32): DataType_FLOAT32,
}

_NORMALIZE_ERROR = "Failed to execute normalize operator, please ensure your inputs are valid."
_TO_TENSOR_ERROR = "Failed to execute 'to tensor' operator, please ensure your inputs are valid."
_IMAGE_VALIDATION_ERROR = "The input image size or format is invalid."
_OPERATOR_ERROR = "Failed to execute operator, please ensure your inputs are valid."

# The native acceleration layer only accepts images whose width and height are
# within this inclusive range, and only reads input files that are not
# accessible to "other" (the packaged files are installed as 440/550/750).
MIN_IMAGE_SIZE = 10
MAX_IMAGE_SIZE = 8192

# Still-image decoding in the native layer goes through the bundled
# libjpeg-turbo (see mm.acc._impl._preload_shared_libraries); other formats
# have to be converted by the caller, e.g. through Image.from_pillow().
_SUPPORTED_STILL_IMAGE_FORMATS = frozenset({"JPEG"})


def _check_image_geometry(height: int, width: int) -> None:
    if not (MIN_IMAGE_SIZE <= height <= MAX_IMAGE_SIZE and MIN_IMAGE_SIZE <= width <= MAX_IMAGE_SIZE):
        raise RuntimeError(_IMAGE_VALIDATION_ERROR)


def _check_readable_file(path: str) -> None:
    """Reject media inputs that are accessible to "other".

    The native media decoders only read files whose mode grants no permission to
    "other" (the deployed package itself is installed as 440/550/640), so the
    CPU backend enforces the same rule to keep behaviour identical.
    """
    if not os.path.isfile(path):
        raise RuntimeError(f"The input path '{path}' is not a readable regular file.")

    mode = stat.S_IMODE(os.stat(path).st_mode)
    if mode & 0o007:
        raise RuntimeError(
            f"The permission of '{path}' is too open (mode {oct(mode)}), "
            "please remove all 'other' permission bits.")


def _check_still_image_support(pillow_image) -> None:
    """The native still-image decoder is libjpeg-turbo based, i.e. JPEG only."""
    if pillow_image.format not in _SUPPORTED_STILL_IMAGE_FORMATS:
        raise RuntimeError(
            f"The input image format '{pillow_image.format}' is not supported by the "
            f"acc image decoder, supported formats: {sorted(_SUPPORTED_STILL_IMAGE_FORMATS)}.")


def _supports_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _decode_path(path) -> str:
    if isinstance(path, (bytes, bytearray)):
        return os.fsdecode(bytes(path))
    return str(path)


def _as_numpy_interface(nd_array: np.ndarray) -> dict:
    """Build the ``__array_interface__`` payload expected by ``ObjectWrapper``.

    A reference to the owning array is stored inside the interface dictionary so
    that the buffer stays alive as long as NumPy keeps the wrapper object that
    exposes the interface.
    """
    interface = dict(nd_array.__array_interface__)
    interface["_mm_keepalive"] = nd_array
    return {"__array_interface__": interface}


def _empty_tensor_error() -> RuntimeError:
    return RuntimeError("Failed to execute normalize operator, please ensure your inputs are valid.")


class Tensor:
    """CPU tensor holder mirroring the native ``acc.Tensor``.

    The class is a thin container: NumPy arrays are stored by reference so that
    ``from_numpy``/``numpy`` conversions stay zero-copy, exactly like the
    native implementation.
    """

    def __init__(self):
        self._array = None
        self._format = TensorFormat_ND
        self._device = _DEVICE_CPU

    # -- construction -----------------------------------------------------
    @classmethod
    def from_numpy(cls, nd_array: np.ndarray) -> "Tensor":
        if not isinstance(nd_array, np.ndarray):
            raise TypeError("The input param 'nd_array' must be of numpy's ndarray type.")
        if not nd_array.flags["C_CONTIGUOUS"]:
            raise ValueError("The input param 'nd_array' must be c_contiguous.")
        if nd_array.dtype not in _NUMPY_TO_DATATYPE:
            raise ValueError("The input numpy's ndarray data type must be in [np.int8/np.uint8/np.float32]")

        tensor = cls.__new__(cls)
        tensor._array = nd_array
        tensor._format = TensorFormat_ND
        tensor._device = _DEVICE_CPU
        return tensor

    def _assign(self, other: "Tensor") -> None:
        self._array = other._array
        self._format = other._format
        self._device = other._device

    # -- properties -------------------------------------------------------
    @property
    def device(self) -> bytes:
        return self._device

    @property
    def dtype(self) -> int:
        # A default-constructed tensor reports float32, like the native class.
        if self._array is None:
            return DataType_FLOAT32
        return _NUMPY_TO_DATATYPE[self._array.dtype]

    @property
    def shape(self) -> list:
        if self._array is None:
            return []
        return list(self._array.shape)

    @property
    def format(self) -> int:
        return self._format

    @property
    def nbytes(self) -> int:
        if self._array is None:
            return 0
        return int(self._array.nbytes)

    # -- operators --------------------------------------------------------
    def clone(self) -> "Tensor":
        tensor = Tensor.__new__(Tensor)
        tensor._array = None if self._array is None else self._array.copy()
        tensor._format = self._format
        tensor._device = self._device
        return tensor

    def set_format(self, tensor_format: int) -> None:
        tensor_format = int(tensor_format)
        if tensor_format == TensorFormat_ND:
            self._format = tensor_format
            return
        if self._array is None or self._array.ndim != 4:
            raise RuntimeError(
                "Failed to execute set format operator, please ensure your inputs are valid.")
        self._format = tensor_format

    def numpy(self) -> dict:
        if self._array is None:
            raise RuntimeError("Failed to execute operator, please ensure your inputs are valid.")
        return _as_numpy_interface(self._array)

    def normalize(self, mean, std, device_mode: int = DeviceMode_CPU) -> "Tensor":
        if self._array is None or self._array.dtype != np.float32:
            raise _empty_tensor_error()
        result = _normalize_array(self._array, self._format, mean, std)
        tensor = Tensor.__new__(Tensor)
        tensor._array = result
        tensor._format = self._format
        tensor._device = self._device
        return tensor


def _normalize_array(array: np.ndarray, tensor_format: int, mean, std) -> np.ndarray:
    """Normalize per channel, replicating ``torchvision.transforms.Normalize``."""
    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32)

    channel_axis = 1 if tensor_format == TensorFormat_NCHW else -1

    if mean_arr.size != array.shape[channel_axis] or std_arr.size != array.shape[channel_axis]:
        raise _empty_tensor_error()
    if np.any(std_arr == 0):
        raise _empty_tensor_error()

    reshape = [1] * array.ndim
    reshape[channel_axis] = -1
    normalized = (array.astype(np.float32) - mean_arr.reshape(reshape)) / std_arr.reshape(reshape)
    return np.ascontiguousarray(normalized, dtype=np.float32)


class Image:
    """CPU image holder mirroring the native ``acc.Image``."""

    def __init__(self):
        self._array = None
        self._format = ImageFormat_RGB
        self._device = _DEVICE_CPU

    # -- construction -----------------------------------------------------
    @classmethod
    def open(cls, path, device: bytes = _DEVICE_CPU) -> "Image":
        try:
            from PIL import Image as PImage
        except ImportError as e:  # pragma: no cover - pillow is a hard requirement at runtime
            raise ImportError(f"Please install pillow firstly, error: {e}") from e

        image_path = _decode_path(path)
        if not image_path:
            _acc_log(3, "Check file path failed. The path is empty.", "ImageOpen")
            raise RuntimeError("The input image path is empty.")
        _check_readable_file(image_path)
        with PImage.open(image_path) as raw_image:
            _check_still_image_support(raw_image)
            array = np.asarray(raw_image.convert("RGB"), dtype=np.uint8)
        return cls.from_numpy(array, ImageFormat_RGB, device)

    @classmethod
    def _from_array(cls, nd_array: np.ndarray, image_format: int,
                    device: bytes = _DEVICE_CPU) -> "Image":
        """Internal constructor bypassing user-facing validation."""
        image = cls.__new__(cls)
        image._array = nd_array
        image._format = int(image_format)
        image._device = _DEVICE_CPU if device is None else device
        return image

    @classmethod
    def from_numpy(cls, nd_array: np.ndarray, image_format: int, device: bytes = _DEVICE_CPU) -> "Image":
        if not isinstance(nd_array, np.ndarray):
            raise TypeError("The input param 'nd_array' must be of numpy's ndarray type.")
        if not nd_array.flags["C_CONTIGUOUS"]:
            raise ValueError("The input param 'nd_array' must be c_contiguous.")
        if nd_array.dtype != np.uint8:
            raise ValueError("The input numpy's ndarray data type must be np.uint8")

        image = cls._from_array(nd_array, image_format, device)
        image._validate()
        return image

    def _validate(self) -> None:
        if self._array is None or self._array.ndim != 3:
            raise RuntimeError(_IMAGE_VALIDATION_ERROR)

        image_format = int(self._format)
        if image_format in (ImageFormat_RGB, ImageFormat_BGR):
            if self._array.shape[2] != 3:
                raise RuntimeError(_IMAGE_VALIDATION_ERROR)
        elif image_format in (ImageFormat_RGB_PLANAR, ImageFormat_BGR_PLANAR):
            if self._array.shape[0] != 3:
                raise RuntimeError(_IMAGE_VALIDATION_ERROR)
        else:
            raise RuntimeError(_IMAGE_VALIDATION_ERROR)

        _check_image_geometry(self.height, self.width)

    # -- properties -------------------------------------------------------
    @property
    def device(self) -> bytes:
        return self._device

    @property
    def dtype(self) -> int:
        return DataType_UINT8

    @property
    def size(self) -> list:
        return [self.width, self.height]

    @property
    def _planar(self) -> bool:
        return int(self._format) in (ImageFormat_RGB_PLANAR, ImageFormat_BGR_PLANAR)

    @property
    def format(self) -> int:
        return self._format

    @property
    def nbytes(self) -> int:
        return int(self._array.nbytes)

    @property
    def height(self) -> int:
        return int(self._array.shape[1] if self._planar else self._array.shape[0])

    @property
    def width(self) -> int:
        return int(self._array.shape[2] if self._planar else self._array.shape[1])

    # -- operators --------------------------------------------------------
    def numpy(self) -> dict:
        return _as_numpy_interface(self._array)

    def clone(self) -> "Image":
        image = Image.__new__(Image)
        image._array = self._array.copy()
        image._format = self._format
        image._device = self._device
        return image

    def crop(self, top: int, left: int, height: int, width: int, device_mode: int = DeviceMode_CPU) -> "Image":
        self._ensure_packed()
        if height <= 0 or width <= 0:
            raise RuntimeError(_OPERATOR_ERROR)
        if top < 0 or left < 0 or top + height > self.height or left + width > self.width:
            raise RuntimeError(_OPERATOR_ERROR)
        cropped = self._array[top:top + height, left:left + width, :]
        return Image._from_array(np.ascontiguousarray(cropped), self._format)

    def resize(self, width: int, height: int, interpolation: int, device_mode: int = DeviceMode_CPU) -> "Image":
        self._ensure_packed()
        _check_image_geometry(int(height), int(width))

        from PIL import Image as PImage

        pillow_image = PImage.fromarray(self._array, "RGB")
        resized = pillow_image.resize((int(width), int(height)), _pillow_resampling(interpolation))
        return Image._from_array(np.asarray(resized, dtype=np.uint8), self._format)

    def to_tensor(self, target_format: int = TensorFormat_NCHW, device_mode: int = DeviceMode_CPU) -> Tensor:
        self._ensure_packed()
        array = self._array.astype(np.float32) / np.float32(255.0)

        if int(target_format) == TensorFormat_NCHW:
            tensor_array = np.ascontiguousarray(array.transpose(2, 0, 1)[np.newaxis, ...])
        elif int(target_format) == TensorFormat_NHWC:
            tensor_array = np.ascontiguousarray(array[np.newaxis, ...])
        else:
            raise RuntimeError(_TO_TENSOR_ERROR)

        tensor = Tensor.__new__(Tensor)
        tensor._array = tensor_array
        tensor._format = int(target_format)
        tensor._device = self._device
        return tensor

    def _ensure_packed(self) -> None:
        if self._array is None or self._array.ndim != 3 or self._array.shape[2] != 3 or self._planar:
            raise RuntimeError(_OPERATOR_ERROR)


def _pillow_resampling(interpolation: int):
    from PIL import Image as PImage

    if int(interpolation) == Interpolation_BICUBIC:
        return PImage.BICUBIC
    return PImage.BICUBIC


class StringVector:
    """Minimal stand-in for the native ``std::vector<std::string>`` binding."""

    def __init__(self):
        self._items = []

    def push_back(self, item) -> None:
        self._items.append(item)

    def resize(self, size: int) -> None:
        while len(self._items) < size:
            self._items.append(b"")
        del self._items[size:]

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, index):
        return self._items[index]


class Tensorvector:
    """Minimal stand-in for the native ``std::vector<acc::Tensor>`` binding."""

    def __init__(self):
        self._items = []

    def resize(self, size: int) -> None:
        while len(self._items) < size:
            self._items.append(Tensor())
        del self._items[size:]

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, index):
        return self._items[index]

    def __setitem__(self, index, value) -> None:
        self._items[index] = value


_LOG_CALLBACK = None
_LOG_MIN_LEVEL = 1


class LogCallBacker:
    """Native-compatible log callback registration hook.

    ``mm.comm.log`` registers a callback object exposing ``log(level, message,
    file, line, function)``; backend diagnostics are pushed through it the same
    way the native layer does it.
    """

    def __init__(self):
        self.log = None

    @staticmethod
    def register_log_conf(min_level: int, callback) -> None:
        global _LOG_CALLBACK, _LOG_MIN_LEVEL
        _LOG_CALLBACK = callback
        _LOG_MIN_LEVEL = int(min_level)


def _acc_log(level: int, message: str, function: str = "acc") -> None:
    """Forward a diagnostic to the registered acc log callback, if any."""
    callback = _LOG_CALLBACK
    if callback is None or level < _LOG_MIN_LEVEL:
        return
    sink = getattr(callback, "log", None)
    if sink is None:
        return
    sink(level, message.encode("utf-8"), b"cpu_backend.py", 0, function.encode("utf-8"))


def normalize(src: Tensor, dst: Tensor, mean, std, device_mode: int = DeviceMode_CPU) -> None:
    """Native-style in-place normalize writing the result into ``dst``."""
    if not isinstance(src, Tensor) or not isinstance(dst, Tensor):
        raise RuntimeError(_NORMALIZE_ERROR)
    dst._assign(src.normalize(mean, std, device_mode))


# ---------------------------------------------------------------------------
# media decoding
# ---------------------------------------------------------------------------
def _probe_video(video_path: str) -> dict:
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe is required to decode video but was not found in PATH")
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_frames,avg_frame_rate,codec_type",
        "-of", "json", video_path,
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"failed to probe video '{video_path}': {completed.stderr.decode('utf-8', 'replace').strip()}")
    try:
        streams = json.loads(completed.stdout.decode("utf-8", "replace"))["streams"]
    except (ValueError, KeyError) as e:
        raise RuntimeError(f"failed to probe video '{video_path}': {e}") from e
    if not streams:
        raise RuntimeError(f"no video stream found in '{video_path}'")
    return streams[0]


def _video_frame_count(video_path: str) -> int:
    try:
        stream = _probe_video(video_path)
    except RuntimeError:
        return 0
    nb_frames = stream.get("nb_frames")
    try:
        return int(nb_frames)
    except (TypeError, ValueError):
        return 0


def _decode_frames(video_path: str, indices) -> list:
    stream = _probe_video(video_path)
    width = int(stream["width"])
    height = int(stream["height"])
    frame_bytes = width * height * 3

    select_expr = None
    if indices:
        select_expr = "+".join(f"eq(n\\,{int(i)})" for i in sorted(set(int(i) for i in indices)))
    command = ["ffmpeg", "-v", "error", "-i", video_path]
    if select_expr is not None:
        command += ["-vf", f"select='{select_expr}'", "-vsync", "0"]
    command += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frames = []
    try:
        while True:
            buffer = process.stdout.read(frame_bytes)
            if len(buffer) < frame_bytes:
                break
            frames.append(np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3).copy())
    finally:
        process.stdout.close()
        stderr = process.stderr.read()
        process.stderr.close()
        return_code = process.wait()
    if return_code != 0 and not frames:
        raise RuntimeError(
            f"failed to decode video '{video_path}': {stderr.decode('utf-8', 'replace').strip()}")
    return frames


def video_decode(path, device: bytes = _DEVICE_CPU, frame_indices=None, sample_num: int = -1) -> list:
    """Decode video frames with ffmpeg into ``Image`` objects (RGB, uint8)."""
    video_path = _decode_path(path)
    _check_readable_file(video_path)

    requested = set()
    if frame_indices:
        requested = {int(index) for index in frame_indices}
    elif sample_num and int(sample_num) > 0:
        total = _video_frame_count(video_path)
        if total > 0:
            picked = np.linspace(0, total - 1, num=min(int(sample_num), total))
            requested = {int(round(value)) for value in picked}

    frames = _decode_frames(video_path, requested)
    return [Image._from_array(frame, ImageFormat_RGB) for frame in frames]


def _read_wav_mono(path: str):
    import wave

    with wave.open(path, "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        raw = wav_file.readframes(wav_file.getnframes())

    if sample_width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"unsupported wav sample width: {sample_width}")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32), sample_rate


def _resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or samples.size == 0:
        return samples
    duration = samples.size / float(src_rate)
    target_size = int(round(duration * dst_rate))
    if target_size <= 0:
        return np.zeros(0, dtype=np.float32)
    src_positions = np.linspace(0.0, duration, num=samples.size, endpoint=False)
    dst_positions = np.linspace(0.0, duration, num=target_size, endpoint=False)
    return np.ascontiguousarray(np.interp(dst_positions, src_positions, samples), dtype=np.float32)


def _load_audio_into(path: str, dst: "Tensor", sr) -> int:
    audio_path = _decode_path(path)
    _check_readable_file(audio_path)
    samples, sample_rate = _read_wav_mono(audio_path)
    if sr is not None:
        samples = _resample_linear(samples, sample_rate, int(sr))
        sample_rate = int(sr)
    tensor = Tensor.__new__(Tensor)
    tensor._array = samples
    tensor._format = TensorFormat_ND
    tensor._device = _DEVICE_CPU
    dst._assign(tensor)
    return sample_rate


def load_audio(path, dst: "Tensor", sr=None) -> int:
    return _load_audio_into(path, dst, sr)


def load_audio_batch(paths, dst_tensors, sr=None) -> list:
    sample_rates = []
    for index, path in enumerate(paths):
        sample_rates.append(_load_audio_into(path, dst_tensors[index], sr))
    return sample_rates


# ---------------------------------------------------------------------------
# Qwen2-VL preprocessing
# ---------------------------------------------------------------------------
def _resize_hwc(array: np.ndarray, width: int, height: int) -> np.ndarray:
    from PIL import Image as PImage

    if array.shape[0] == height and array.shape[1] == width:
        return array
    pillow_image = PImage.fromarray(array, "RGB")
    return np.asarray(pillow_image.resize((int(width), int(height)), PImage.BICUBIC), dtype=np.uint8)


def _rescale_and_normalize(array: np.ndarray, mean, std, channel_axis: int) -> np.ndarray:
    data = array.astype(np.float32)
    if array.dtype == np.uint8:
        data = data / np.float32(255.0)
    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32)
    reshape = [1] * data.ndim
    reshape[channel_axis] = -1
    return np.ascontiguousarray(
        (data - mean_arr.reshape(reshape)) / std_arr.reshape(reshape), dtype=np.float32)


class Qwen2VLProcessor:
    """CPU implementation of the Qwen2-VL preprocessing operators."""

    @staticmethod
    def Preprocess(images, image_mean, image_std, width: int, height: int) -> list:
        outputs = []
        for image in images:
            array = np.asarray(_read_inner_array(image))
            resized = _resize_hwc(array, width, height)
            normalized = _rescale_and_normalize(resized, image_mean, image_std, channel_axis=-1)
            tensor = Tensor.__new__(Tensor)
            tensor._array = np.ascontiguousarray(normalized[np.newaxis, ...])
            tensor._format = TensorFormat_NHWC
            tensor._device = _DEVICE_CPU
            outputs.append(tensor)
        return outputs

    @staticmethod
    def PreprocessTensor(tensors, image_mean, image_std, width: int, height: int) -> list:
        outputs = []
        for tensor in tensors:
            array = _read_inner_array(tensor)
            tensor_format = tensor.format if isinstance(tensor, Tensor) else TensorFormat_ND
            if array.ndim == 3:
                array = array[np.newaxis, ...]

            if tensor_format == TensorFormat_NCHW:
                frames = [np.ascontiguousarray(frame.transpose(1, 2, 0)) for frame in array]
                resized = np.stack([_resize_hwc(frame, width, height) for frame in frames], axis=0)
                normalized = _rescale_and_normalize(resized, image_mean, image_std, channel_axis=-1)
                result = np.ascontiguousarray(normalized.transpose(0, 3, 1, 2))
            else:
                resized = np.stack([_resize_hwc(frame, width, height) for frame in array], axis=0)
                result = _rescale_and_normalize(resized, image_mean, image_std, channel_axis=-1)

            out_tensor = Tensor.__new__(Tensor)
            out_tensor._array = result
            out_tensor._format = tensor_format
            out_tensor._device = _DEVICE_CPU
            outputs.append(out_tensor)
        return outputs


def _read_inner_array(obj) -> np.ndarray:
    array = getattr(obj, "_array", None)
    if array is None:
        raise RuntimeError("Failed to execute operator, please ensure your inputs are valid.")
    return array
