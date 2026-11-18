# Python Project 1 - NumPy Image Encoder

本项目使用 NumPy 实现 OpenCLIP ViT-B-32 图像编码器的前向计算，不需要训练模型，也不依赖 PyTorch 或 OpenCLIP 包。

## 文件说明

- `image_encoder.py`：完成 patch 切分、多头注意力、Transformer block、图像 token 嵌入和最终输出。
- `numpy_homework.py`：8 道基础题和 4 道中等题的合并提交版。
- `lecture-demos/my_numpy_homework/`：每道 NumPy 练习的独立版本。
- `report/main.tex`：实验报告正文，在 Overleaf 中选择 XeLaTeX 编译。

## 环境安装

```bash
python -m pip install -r requirements.txt
```

要求 Python 3.9 或更高版本。

## 运行 NumPy 练习

```bash
python numpy_homework.py
```

## 检查图像编码器

```bash
python check.py --stage all
```

完整检查应依次通过 `assets`、`preprocess`、`patches`、`tokens`、`block0`、`last_tokens`、`output` 和 `batch`。

## 生成图像输出

```bash
python run.py --output output_tokens.npz
```

自定义图像：

```bash
python run.py --images path/to/a.jpg path/to/b.png --output my_tokens.npz
```

## 数值检查结果

| 阶段 | 输出 shape | 最大绝对误差 | 平均绝对误差 |
| --- | --- | ---: | ---: |
| preprocess | `(10,3,224,224)` | `0.000e+00` | `0.000e+00` |
| patches | `(2,49,3072)` | `0.000e+00` | `0.000e+00` |
| tokens | `(2,50,768)` | `4.005e-05` | `1.289e-06` |
| block0 | `(2,50,768)` | `7.063e-06` | `3.250e-07` |
| last_tokens | `(2,50,768)` | `4.768e-05` | `2.696e-06` |
| output | `(10,512)` | `5.439e-07` | `8.311e-08` |
| batch | `(3,512)` | `4.619e-07` | `7.632e-08` |

## 代码仓库

https://github.com/Miya555orz/my-numpy-homework

本仓库不包含课程提供的 `weights/` 和整个 `assets/` 目录。运行数值检查时，请把课程原始附件中的这两个目录放回项目根目录。

## 贡献与辅助说明

本人完成 NumPy 练习、理解数组 shape 与广播，并对最终代码和报告负责。实现与报告整理过程中使用了课程提供的 starter、数值检查器、SZU 实验报告模板和 OpenAI Codex 辅助；所有结果均通过本地程序重新运行验证。
