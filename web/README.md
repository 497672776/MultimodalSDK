# MultimodalSDK 网页演示

把 MultimodalSDK（`mm`）的媒体预处理能力做成一个网页，模型侧统一走 llama.cpp
（`llama-server`）。已在 SpacemiT K3（RISC-V / Bianbu 4.0.7）上部署验证。

## 功能

| 标签页 | 用到的 SDK 能力 | 说明 |
| --- | --- | --- |
| 图像预处理 | `mm.Image` → `mm.core.processor.resize_and_normalize` | 解码、smart resize 到 28 的整数倍、/255、按 mean/std 归一化，回显 tensor 形状与数值范围 |
| 视频抽帧 | `mm.acc.video_decode` | 支持均匀抽帧（`sample_num`）与指定帧号 |
| 音频加载 | `mm.acc.load_audio` | wav → 一维 float32 波形 + 采样率，页面画包络 |
| SCC 压缩 | `mm.core.scc.scc_compress_to_target` | 语义连通分量视觉 token 压缩，展示压缩前后特征热力图 |
| 视觉问答 | `mm.Image` + llama.cpp | 图片经 mm 解码缩放后以 base64 发给 `llama-server`（Qwen3-VL + mmproj） |

## 后端接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/status` | SDK 路径、acc 后端、torch/ffmpeg 可用性、llama-server 健康状态 |
| POST | `/api/image/preprocess` | `file`、`target=qwen|square`、`size` |
| POST | `/api/video/decode` | `file`、`sample_num`、`frame_indices` |
| POST | `/api/audio/load` | `file`、`sample_rate` |
| POST | `/api/scc/compress` | `file`、`ratio`、`tau`、`epsilon` |
| POST | `/api/vlm/chat` | `file`、`question`、`max_tokens`、`temperature`、`max_side` |

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MM_WEB_HOST` / `MM_WEB_PORT` | `0.0.0.0` / `8090` | 监听地址与端口 |
| `MM_WEB_LLAMA_URL` | `http://127.0.0.1:18810` | llama-server 地址 |
| `MM_WEB_WORK_DIR` | `$HOME/mm-sdk-web/work` | 上传临时目录（上传文件按 SDK 要求写成 0440） |
| `MM_WEB_MAX_UPLOAD_MB` | `64` | 单文件上限 |
| `MM_SDK_SOURCE` | 仓库 `source/` | `mm` 包所在目录（`PYTHONPATH`） |
| `MM_WEB_VENV` | `$HOME/mm-sdk-web/venv` | 运行网页的 Python 虚拟环境 |

## 在 K3 上部署

```bash
# 1) 板端运行环境（Python 3.13；torch/cv2 需要用 SpacemiT 源里的 riscv64 wheel）
python3.13 -m venv ~/mm-sdk-web/venv
~/mm-sdk-web/venv/bin/pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --extra-index-url https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple \
  --only-binary torch,torchvision,numpy,pillow,opencv-python-headless \
  "torch==2.8.0+spacemit.omp.2" "torchvision==0.23.0" numpy pillow \
  transformers fastapi uvicorn python-multipart "opencv-python-headless==4.12.0.88"
# torch 依赖的系统库
sudo apt-get install -y libsleef3 libomp5

# 2) 同步代码并启动
MM_WEB_SSH=bianbu@<board-ip> MM_WEB_SSH_PASSWORD=<password> ./web/deploy.sh
MM_WEB_HOME=~/mm-sdk-web bash ~/mm-sdk-web/web/start-llama-vlm.sh   # 板端启动 VLM

# 3) 打开 http://<board-ip>:8090/
```

## 说明

- `mm.acc` 的原生实现（`_acc` + `libcore.so`）只随 Ascend/aarch64 交付；在 RISC-V 上
  自动回退到 `mm/acc/_impl/cpu_backend.py`，接口与行为对齐原生算子（详见该文件注释）。
- 原生 `mm.Image.open` 只解码 JPEG、且只读「other 位为 0」的文件；网页上传会保存为
  `0440` 以满足该策略，非 JPEG 图片走 `mm.Image.from_pillow`。
- SCC 标签页用每 28×28 patch 的颜色/纹理描述子代替视觉编码器特征，仅演示压缩算法本身。
