# EE542-Lab7-Neural-Networks

USC EE 542 Lab 7：NumPy MLP、NumPy CNN 与 PyTorch/GPU 实验。

## 目录

```text
.
├── README.md                  # 项目结构与运行入口
├── DATA.md                    # 唯一的数据、预处理和结果格式说明
├── src/
│   └── part1_mlp.py            # Part I：三个 NumPy MLP 示例
├── scripts/
│   ├── prepare_data.py         # 下载、导出、检查及划分图片
│   └── minipq.py               # prepare_data.py 的 Parquet 读取辅助模块
├── tests/
│   └── test_prepare_data.py    # 数据准备流程测试
├── data/                      # 图片、划分清单、统计与抽查图
└── results/                   # 本地实验结果，Git 已忽略
```

Python 实现只维护一份：模型放在 `src/`，数据准备及实验运行脚本放在
`scripts/`，测试放在 `tests/`。原始 Parquet 文件下载到 `raw/`。

## 运行

以下命令在仓库根目录执行，使用已安装 NumPy、Matplotlib 和 Pillow 的 Python 环境。

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

当前已有 Part I 和数据准备代码。NumPy CNN、PyTorch CNN、共享数据工具及实验
运行/比较脚本尚待实现。数据约定与结果 JSON 字段见 [DATA.md](DATA.md)。
