# EE542-Lab7-Neural-Networks

USC EE 542 Lab 7：NumPy MLP、NumPy CNN 与 PyTorch/GPU 实验。

## 目录

```text
.
├── README.md                  # 项目结构与运行入口
├── DATA.md                    # 唯一的数据、预处理和结果格式说明
├── src/
│   ├── part1_mlp.py            # Part I：三个 NumPy MLP 示例
│   └── part3_torch.py          # Part III：PyTorch CNN、训练及分块评估
├── scripts/
│   ├── prepare_data.py         # 下载、导出、检查及划分图片
│   └── minipq.py               # prepare_data.py 的 Parquet 读取辅助模块
├── tests/
│   ├── test_prepare_data.py    # 数据准备流程测试
│   └── test_part3_torch.py     # PyTorch 实现的行为测试
├── data/                      # 图片、划分清单、统计与抽查图
└── results/                   # 本地实验结果，Git 已忽略
```

Python 实现只维护一份：模型放在 `src/`，数据准备及实验运行脚本放在
`scripts/`，测试放在 `tests/`。原始 Parquet 文件下载到 `raw/`。

## Python environment

Use a project-local virtual environment for every install and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-part1.txt
```

## 运行

以下命令在仓库根目录执行，并先激活 `.venv`。

```bash
# Part I：运行全部三个示例
python src/part1_mlp.py --output-dir results/part1

# 数据准备测试：无需下载真实数据
python tests/test_prepare_data.py

# 下载并生成图片数据；下载功能还需要 huggingface_hub
python scripts/prepare_data.py --download --raw raw --out data

# 已有 Parquet 文件时
python scripts/prepare_data.py --raw raw --out data

# 可选：增加真实 Parquet 读取测试
python tests/test_prepare_data.py --raw raw
```

数据准备要求输出目录为空；当前 `data/` 已有清单和统计，可以先使用
`--out data_restored` 恢复到新目录。`--overwrite` 会删除并重建指定输出目录。
当前仓库中的清单和统计不代表本地已包含全部图片。

## 分工

| 成员 | 负责内容 |
|---|---|
| A | Part II：NumPy CNN、训练与测试、计时和实验结果 |
| B | Part III：PyTorch CNN、CPU/GPU 设备管理、计时和实验结果 |
| C | 数据恢复、共享数据工具、实验运行与性能比较 |

当前已有 Part I、数据准备及 PyTorch CNN 代码。Part III 尚待队友执行测试和正式实验；
NumPy CNN、共享数据工具及实验运行/比较脚本尚待实现。数据约定与结果 JSON 字段见
[DATA.md](DATA.md)。

## Part III (team member B): PyTorch CNN

The dependency file targets Python 3.12, PyTorch 2.9.1, torchvision 0.24.1
and CUDA 12.6 wheels from the [official version matrix](https://pytorch.org/get-started/previous-versions/#v291).
Part III is currently an implementation handoff: the tests and real-data runs
below still need to be executed by the teammate responsible for verification.

```bash
source .venv/bin/activate
python -m pip install -r requirements-part3.txt
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
python -m unittest discover -s tests -p 'test_part3_torch.py' -v
```

The tests use generated fixtures and require no downloaded dataset. For training,
prepare `data/train/{cat,dog}`, `data/test/{cat,dog}`, `data/stats.json` and
`data/split_manifest.csv` according to [DATA.md](DATA.md). The script rejects
missing images or a mismatch between actual files and the recorded split.
If the data was restored elsewhere, pass that directory with `--data-dir`.

First verify the pipeline using a reproducible subset of the real images:

```bash
python src/part3_torch.py --device cuda --epochs 2 \
  --max-train-per-class 128 --max-test-per-class 64 \
  --output results/part3_smoke_cuda.json
```

Then run the full experiment:

```bash
set -o pipefail
mkdir -p results
python -u src/part3_torch.py --device cuda --epochs 10 \
  --output results/part3_torch_cuda.json | tee results/part3_torch_cuda.log
```

Use `--device cpu` for CPU execution or omit the device flag for automatic device
selection. An explicit `--device cuda` fails if CUDA is unavailable. Both the model
and tensors run on the selected device.

The model is `Conv(3,16) -> ReLU -> MaxPool -> Conv(16,32) -> ReLU -> MaxPool ->
FC(8192,64) -> ReLU -> FC(64,1) -> ReLU`, trained with MSE and Adam at 0.001. The
output layer starts with small weights and bias 0.5 to reduce inactive output
ReLUs. All convolution and fully connected parameters are trainable. This differs
from Part II's fixed convolution filter; comparisons must account for that.

Images are RGB, resized to 64x64 with bilinear interpolation, and divided by 255.
They are preloaded into CPU memory before timing (about 1.1 GiB for the full set).
Training batches are shuffled; test order is fixed. Labels are float32 columns
matching the `[N,1]` output. A prediction greater than 0.5 means dog.

The result JSON follows DATA.md and includes the effective split fingerprint,
source fingerprint, device, configuration, per-epoch loss/times, accuracy, test
blocks of 1,000 images, and throughput. Subset runs have their own fingerprint.
Training time spans all epochs and includes batching, transfers, forward/backward,
Adam, metrics and epoch logging; it excludes disk decoding, optimizer setup,
warmup and saving. Per-epoch times are also recorded separately.
Test time includes batching, transfers, inference, metrics and synchronization.
`test_compute_seconds` measures only forward calls, with CUDA synchronization.

Alongside the result JSON, a directory with the same stem contains `epochs.csv`,
`loss.png` and `model.pt` (model/optimizer state, configuration and label mapping).
Reload the model with:

```python
import torch
from src.part3_torch import CatDogCNN

checkpoint = torch.load("results/part3_torch_cuda/model.pt", map_location="cpu", weights_only=True)
model = CatDogCNN()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

Virtual environments, downloaded images, model weights and `results/` are ignored
by Git, following the repository's current `.gitignore` settings.
