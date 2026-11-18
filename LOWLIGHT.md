# 低照度前处理 V2

`enhance_lowlight.py` 对文件夹逐张去噪、估计光照和受限提亮，并可直接完成双路配准与拼接。不覆盖原图，不缩放最终裁片，不进行图像几何变换。只依赖项目现有的 NumPy、SciPy、OpenCV，无需下载神经网络权重。

## Windows 使用

在 `multicamara` 目录运行：

```powershell
.\run_lowlight.bat
```

默认处理 `datasets\dark`，写出 `datasets\dark_natural_v2`，拼接到 `output\dark-natural-ges.png`。再次运行须指定另一个全景文件名：

```powershell
.\run_lowlight.bat output\dark-natural-repeat.png
```

BAT 会重新计算 `dark_natural_v2` 下的增强图和辅助图；已有的全景图及标定文件始终拒绝覆盖。当前 `run.bat` 保持原有输入设置；新版的入口是 `run_lowlight.bat`。

直接指定输入、输出和参数：

```powershell
.\.venv\Scripts\python.exe enhance_lowlight.py --input datasets\dark --output datasets\dark_custom --strength 0.45 --max-gain 3 --denoise 3 --check-matches --stitch output\dark-custom.png
```

省略 `--stitch` 则只生成前处理结果。默认仅扫描当前层的 `*.png`；JPG 或混合格式可加 `--pattern "*"`。自动排除 `ground_truth.png`、`ground-truth.png`、`gt.png` 等参考图。不读取 `tiles.json` 参与匹配、增强或拼接。

输出含义：

| 路径 | 用途 |
| --- | --- |
| `dark_natural_v2/00.png` 等 | 自然增强图，保留源尺寸，供结构检测与最终成像 |
| `dark_natural_v2/registration/00.png` 等 | 较强的对比增强，仅供 SIFT 提取对应点 |
| `dark_natural_v2/diagnostics/comparison.jpg` | 抽取四张裁片，左原图、右增强图 |
| `dark_natural_v2/enhancement-report.json` | 参数、源文件校验值、运行是否完成、可选匹配诊断 |
| 全景同名 `.npz`、`.report.json` | 网格标定及求解报告 |

可使用本脚本的 `--stitch`、BAT 或 app 的「暗光增强」走双路流程。app 点击「应用」后预览自然增强图；点击「开始拼接」时自动使用辅助图做 SIFT、自然增强图做结构检测与成像。详见 [app 使用说明](app/README.md)。`stitch.py stitch` 也支持 `--matching-images` 显式传入同顺序的辅助图。仅匹配自然增强图时，极暗的天空裁片仍可能不连通；辅助图不应作为最终纹理。

## 算法与参考项目

参考了以下项目的设计与依赖条件：

- [LIME/DUAL Python 实现](https://github.com/pvnieo/Low-light-Image-Enhancement)：基于光照估计与曝光增强。新版采用其光照分解思路，独立实现带梯度权重的稀疏最小二乘求解，并增加增益上限、阴影噪声限制及高光保护。没有照搬该仓库源码，也不是完整复现 LIME/DUAL。
- [Zero-DCE 官方实现](https://github.com/Li-Chongyi/Zero-DCE)：学习像素映射曲线，需要神经网络模型。当前版本不运行其模型。
- [HVI-CIDNet 官方实现](https://github.com/Fediory/HVI-CIDNet)：通过颜色与强度表示处理低照度增强，需要模型及权重。当前版本没有引入该网络，不能声称达到其效果。

自然成像分支先轻度彩色 NLM 去噪，以最大 RGB 通道初始化光照 `T0`，求解：

```text
min_T ||T - T0||² + alpha * sum_edges(w_pq * (T_p - T_q)²)
w_pq = 1 / (smoothed_absolute_gradient + 0.01)
```

光照图仅在低分辨率估计，再插值回源尺寸。根据 `T^(-strength)` 计算增益，并用 `T/(T+0.04)` 限制低信号区域的增益；默认最多 3 倍。三个颜色通道使用同一增益，避免强亮度通道均衡造成的泛灰。最终分支不使用 CLAHE，也不做锐化。增益不依赖逐像素边缘方差，以减轻沿轮廓发亮的问题。

配准分支仍可使用共享 gamma 和较强 CLAHE，只交给 SIFT 找对应点。`ges.calibrate(..., matching_images=helpers)` 校验两路图的数量、尺寸、类型，调用方须保持相同相机顺序。HED/Canny 和渲染都使用自然增强图。已有调用不传 `matching_images` 时保持原行为。

`--strength` 范围 0–1，越大越亮；0 时返回原图。`--max-gain` 为 1–8，建议先保留默认 3。`--denoise` 为 0–20，0 关闭降噪；更大可能损失暗区细节。`--work-size` 控制光照估计尺寸，默认 192，不控制输出分辨率。全组沿用相同参数，但每张估计自己的光照，边界光照估计仍可能造成接缝差异。

## 这组 dark 数据的验证

旧版 `dark-enhanced-ges.png` 中的灰块主要是强增强放大了暗区的量化轮廓。按 `tiles.json` 的裁片位置独立核对旧网格，去除整体平移后最大顶点偏移约 0.26 像素，不能把这类光度破坏全归为网格扭曲。

新版实际对全部 16 张图运行 SIFT、HED 和 GES-GSP 联合网格求解，最终画面仅来自自然增强图。匹配保留 3744 个内点；最终尺寸为 1441×1122。按已知裁片坐标核对最终网格，去除整体平移后的顶点偏移 RMS 约 0.18 像素、最大约 0.61 像素；记录在 `output/dark-v2-validation.json`。坐标仅用于独立核对，不参与重建。成功连通和拟合误差不能证明图像质量，仍需查看结果与局部细节。脚本不读取大图来替代拼接，也不会恢复传感器或量化过程已经丢失的暗区信息。

当前读取与输出统一为 8 位 BGR，不保留源文件的 alpha 或 16 位精度。

测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

覆盖黑场/高光、关闭增强、像素位置与颜色比例、辅助匹配与最终纹理分离、批处理排除参考图及保护原文件，以及原有拼接测试。
