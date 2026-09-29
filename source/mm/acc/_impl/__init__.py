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
import ctypes
import importlib
from pathlib import Path


def _preload_shared_libraries(base):
    opensource_lib_dirs = [
        base / "opensource" / "soxr" / "lib",
        base / "opensource" / "libjpeg-turbo" / "lib",
        base / "opensource" / "FFmpeg" / "lib",
    ]
    lib_files = []
    for lib_dir in opensource_lib_dirs:
        if not lib_dir.is_dir():
            continue
        for lib_file in lib_dir.iterdir():
            if lib_file.is_file() and ".so" in lib_file.name:
                lib_files.append(lib_file.resolve())

    loaded = set()
    for _ in range(len(lib_files) + 1):
        progress = False
        for lib_file in lib_files:
            lib_path = str(lib_file)
            if lib_path in loaded:
                continue
            try:
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                loaded.add(lib_path)
                progress = True
            except OSError:
                pass
        if not progress:
            break

    libcore = base / "lib" / "libcore.so"
    if not libcore.is_file():
        raise ImportError(f"libcore.so not found under {base / 'lib'}. Please reinstall the mm wheel package.")
    ctypes.CDLL(str(libcore.resolve()), mode=ctypes.RTLD_GLOBAL)


def _load_native_backend(base):
    """Load the native AccSDK backend when ``libcore.so`` is available.

    Returns ``None`` on platforms where the native acceleration layer is not
    deployed (for example RISC-V boards), so that callers fall back to the
    pure-Python CPU backend.
    """
    libcore = base / "lib" / "libcore.so"
    if not libcore.is_file():
        return None

    _preload_shared_libraries(base)
    for module_name in (f"{__name__}.acc", f"{__name__}._acc"):
        try:
            return importlib.import_module(module_name)
        except ImportError:
            continue
    raise ImportError(
        f"libcore.so was found under {base / 'lib'} but no importable acc backend was found. "
        "Please reinstall the mm wheel package."
    )


_impl_dir = Path(__file__).resolve().parent

acc = _load_native_backend(_impl_dir)
if acc is None:
    from . import cpu_backend as acc
