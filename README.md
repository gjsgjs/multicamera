# GES-GSP 多相机网格拼接（Python / OpenCV）

在原始图片分辨率上，把多个有重叠视野的固定相机图片拼成更大视野、更多总像素的全景，并将标定网格复用于同步图片文件夹序列。输出 PNG 不会自动缩小，也不通过视频压缩。

算法参考：Du 等，CVPR 2022，**Geometric Structure Preserving Warp for Natural Image Stitching**：

- [作者项目](https://github.com/flowerDuo/GES-GSP-Stitching)
- [论文及补充材料](https://openaccess.thecvf.com/content/CVPR2022/html/Du_Geometric_Structure_Preserving_Warp_for_Natural_Image_Stitching_CVPR_2022_paper.html)

这是独立编写的工程复现，实际实现了网格联合优化与 GES 能量，并非给 OpenCV Stitcher 换名字。**不是作者 C++ 版本的逐行移植，也没有完成论文数据集上的等效复现验证。** 下文列出差异，避免把近似实现误认成论文原版。

## 暗光前处理 V2

界面入口：双击 `app/start.bat`，导入文件夹，在「前处理方式」选择「暗光增强」，点击「应用」预览自然增强图，再点击「开始拼接」。预览可切换原图，拼接继续使用增强结果。见 [app 使用说明](app/README.md)。

在项目目录运行 `.\run_lowlight.bat`，对 `datasets\dark` 生成自然增强图及独立的配准辅助图，再执行 HED/GES-GSP 拼接。辅助图只帮助 SIFT 匹配，最终像素来自自然增强图。相比旧版，去掉了最终成像中的强 CLAHE，改用光照估计、去噪、受限增益与颜色保护。

已生成的结果：`output/dark-natural-ges.png`；原始色调、旧增强、新增强的并排对比：`output/dark-v2-comparison.jpg`。已有输出不会覆盖，重跑时可用 `.\run_lowlight.bat output\dark-natural-next.png`。参数、参考项目及两路图像的用途见 [LOWLIGHT.md](LOWLIGHT.md)。

## 安装

在 `E:\a\stitchup\multicamara` 打开 PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_model.py
```

本工作区已准备好 `.venv` 和作者项目的 HED 模型。模型来源、SHA256 在 `models/sources.json`；下载脚本只下载模型和网络结构，不运行原项目 EXE。依赖为 NumPy、OpenCV、SciPy，SciPy 用于稀疏最小二乘；不需要编译 Eigen/VLFeat/C++ 扩展。

## 拼接 imgs 中的四张图片

### 生成有重叠的 4×4 测试集

```powershell
python .\generate_dataset.py
```

默认读取脚本目录下的 `datasets/img1/ground_truth.png`，在同一目录输出 `00.png`～`15.png` 和 `tiles.json`。按从左到右、从上到下的顺序编号，相邻裁片约有 25% 重叠，覆盖整张原图，且不缩放或改写原图。实际重叠像素数因整数坐标取整略有差异，记录在 JSON 中。裁片保留源图位深和通道数；注意拼接程序本身目前读取为 8 位 BGR。

可更换输入或重叠比例：

```powershell
python .\generate_dataset.py --input .\datasets\img1\ground_truth.png --overlap 0.30 --output .\datasets\img1\overlap30
```

默认拒绝覆盖已生成的裁片；重复生成同一目录时可显式加 `--overwrite`。工作区这张原图为 888×666，默认裁片为 274×205，共 16 张。

### 拼接示例

```powershell
.\.venv\Scripts\python.exe stitch.py stitch --images ../imgs/TL.jpg ../imgs/TR.jpg ../imgs/BL.jpg ../imgs/BR.jpg -o output/my-ges.png
```

会得到：

- `my-ges.png`：原尺寸网格拼接图，不缩图。
- `my-ges.npz`：可复用的相机网格标定。
- `my-ges.report.json`：有效匹配数、结构数、求解器信息和配准误差。

输出已存在时拒绝覆盖，请换名字。本次验证已生成 `output/ges-gsp-full.png` 和对应 NPZ/报告；图像 **3939×3901（约 1537 万像素）**，单路输入为 2048×2048（约 419 万像素）。这增加的是合并视野和总像素，不是插值放大，也不是单幅图像超分辨率。重叠区域不会提供两份独立视野，拼接不能恢复相机从未采到的细节。

## 固定网格生成图片序列

### 有雾图片的预处理

批量去雾脚本采用[暗通道先验与引导滤波](https://people.csail.mit.edu/kaiming/cvpr09/)，不需要下载额外神经网络或安装新依赖：

```powershell
python .\dehaze.py --input .\datasets\frog --output .\datasets\frog_dehazed --check-matches
```

Windows 也可双击 `dehaze.bat`，然后双击 `run_dehazed.bat` 拼接处理后的 `00.png`～`15.png`。原有 `run.bat` 保持不变。若处理结果已存在，直接用 `run_dehazed.bat` 即可；需要重新去雾时指定新输出目录，或显式加 `--overwrite`。

去雾默认共用大气光估计，减少同一批裁片的色调差异；保留图片宽高和文件编号，不进行空间变换。输出是无损 PNG 的 8 位 BGR 图片（输入的 alpha/高位深不保留），默认排除 `ground_truth.png` / `gt.png`，不覆盖源图片。`dehaze-report.json` 记录参数、源文件校验值和每张处理信息；`diagnostics/comparison.jpg` 是前四张图片的处理前后对比，不混入待拼接图片目录。

参数：`--omega 0.9` 控制去雾强度（0～1），`--tmin 0.15` 限制过度放大，`--work-size 1024` 是透射率估计长边上限（不是输出图片大小）。`--clahe 1.5` 可额外增强局部亮度对比，默认关闭。浓雾和噪点可能被同时增强，去雾不能恢复从未采集到的真实纹理；过强时降低 omega 或提高 tmin。`--airlight individual` 可对不同拍摄条件的独立照片分别估计大气光，拼接同一场景通常先试默认 shared。

`--check-matches` 使用当前拼接程序同样的 SIFT/RANSAC 条件比较前后匹配连通分量，会额外耗时。它只检查匹配关系，不能代替完整几何优化和视觉效果验证。本机 frog 的实测结果：处理前分为 `[0]`、`[1]`、`[2]`、`[3]`、`[4…15]`，24 对有效重叠；处理后全部 16 张连通，30 对有效重叠。继续运行完整 HED/GES-GSP 后已生成 `output/frog-dehazed-ges.png`，尺寸 **4482×3472**，保留匹配点的 RMS 约 0.17 像素。该数值是拟合误差，不是去雾画质指标。

### 无重叠的相邻裁块

自动 GES-GSP 配准需要图像中存在共同的场景特征。对于从大图裁出的相邻四块，使用显式布局：

```powershell
.\.venv\Scripts\python.exe stitch.py stitch --images datasets/TL.png datasets/TR.png datasets/BL.png datasets/BR.png --layout-grid 2 -o output/my-grid.png
```

`--layout-grid 2` 表示每行两张，按输入顺序从左到右、从上到下排列。该模式不运行 SIFT/HED/GES 优化，不推断重叠关系，也不缩放或拉伸原图；生成的 NPZ 仍可通过下述 `sequence` 命令复用。各列宽度、各行高度取该列/行图片的最大值，尺寸不一处补黑边。它不恢复裁切时丢失的像素，也不会自动移除画面内的 Camera 标签。

工作区 `datasets` 的四张图片宽高略有差异，已验证输出为 **1388×920**，每张输入区域像素保持不变。如果希望裁块边缘严密连接，应从原图重新导出准确的统一裁切边界。

自动模式出现 `Overlap graph is disconnected` 时，会输出每对图片的候选数、内点数与拒绝原因，并列出无法连接到参考相机的索引。重复叠加的标签、水印可能产生伪匹配，不应仅通过降低内点门槛强制连接。

### 有重叠相机的固定网格

每个相机一个文件夹，同一时刻各相机的图片具有相同文件名：

```text
data/TL/000001.png  data/TL/000002.png ...
data/TR/000001.png  data/TR/000002.png ...
data/BL/000001.png  data/BL/000002.png ...
data/BR/000001.png  data/BR/000002.png ...
```

先用同一时刻的四张图片完成一次 `stitch` 标定，再复用网格：

```powershell
.\.venv\Scripts\python.exe stitch.py sequence -c output/ges-gsp-full.npz --dirs data/TL data/TR data/BL data/BR --pattern "*.png" -o output/frames
```

相机顺序、分辨率、位置、焦距和预处理必须与标定一致。文件夹必须具有完全相同的非空文件名集合；按自然顺序处理，例如 `2.png` 在 `10.png` 前。不会按最短目录悄悄截断或跳过缺失帧。JPEG 序列改用 `--pattern "*.jpg"`。

输出 `frame_000000.png` 等，所有帧保持标定画布大小；`frames.jsonl` 记录输入与输出对应关系，`sequence.json` 记录完成数量与状态。逐组加载图片，不将整个序列读入内存。输出目录必须不存在且不能位于输入目录中。中途中断或出错时保留已写好的帧，并标记未完整完成。

`--max-frames 10` 可以先试处理 10 组。PNG 无损保存计算得到的 8 位像素；当前读取统一使用 8 位 BGR，**不保留 16 位 TIFF 原始位深**。

单组新图片可直接用：

```powershell
.\.venv\Scripts\python.exe stitch.py images -c output/ges-gsp-full.npz --images ../imgs/TL.jpg ../imgs/TR.jpg ../imgs/BL.jpg ../imgs/BR.jpg -o output/reuse.png
```

## 实现与原算法的对应关系

| 部分 | 本实现 | 与作者版本的区别 |
|---|---|---|
| 特征匹配 | SIFT、双向比值检验、RANSAC，所有有效重叠关系参与优化 | OpenCV SIFT 替代 VLFeat；特征默认长边 800，坐标映射回原尺寸 |
| 网格 | 每张图片独立矩形网格，匹配点双线性插值，全部图片联合求解 | 单应矩阵仅用于匹配内点筛选；初始化使用全部重叠关系的联合相似变换，避免长链投影奇点 |
| 对齐能量 | 匹配点经过网格变换后的位置差 | 使用原始分辨率坐标的 RANSAC 内点，未复现作者全部匹配预处理 |
| GSP 局部项 | 邻域边向量的相似变换拟合及边向量残差 | 与源码的邻域投影构造对应 |
| GSP 全局项 | 邻域相似参数接近全局旋转和尺度，非重叠处加大约束 | 全局尺度/旋转来自全部匹配点的联合二维相似拟合；未实现作者的三维旋转估计，空间权重也采用二维近似 |
| 结构提取 | 默认原项目 HED 模型，通过 OpenCV DNN 执行 | 长边 512 检测，轮廓筛选、远端锚点及采样；未逐步复现作者完整细化、连接和结构筛选流程 |
| GES 项 | 论文式 (5) 的三角形相似约束，结构采样点双线性插值到网格 | 实际进入稀疏矩阵，不是仅检测直线后忽略它们 |
| GES 权重 | 论文式 (8) 的余弦权重，最小值 0.01 | 用匹配点网格凸包近似重叠区域，距离在网格上计算 |
| 求解 | SciPy CSR 稀疏矩阵 + LSQR，检查收敛和网格翻折 | 参考相机网格作为锚定，不同于作者只固定两个自由度 |
| 成像 | 网格三角形分片仿射反向映射，OpenCV remap | 优化用双线性插值，成像用三角形插值，两者可能存在少量差异 |
| 融合 | 默认 `sharp` 选择离源图边界更远的有效像素；可选羽化 | 不复现作者的全部纹理映射和融合策略；不包含动态接缝和曝光补偿 |

因此，本版本可用于对固定相机做结构约束网格拼接，但不能宣称复现了论文的全部指标或解决所有大视差问题。HED/Canny 的选择、网格精度与参数会影响效果。

## 参数

- `--cell 128`：原图网格间距。更小可处理更细局部误差，但会增加耗时和自由度，可能引入局部形变。
- `--feature-size 800`：仅用于特征检测，不改变最终图片尺寸。内存紧张可用 `512`。
- `--reference 0`：显式固定参考相机（从 0 开始）；不指定时选择重叠图的中心节点，减少长链图像的累计变形。参考图决定坐标系和像素尺度，几何配准本身允许其他图片旋转、缩放和局部变形。
- `--local-weight 0.75`、`--ges-weight 1.5`：局部项和结构项的方程系数；最小二乘中有效能量权重为系数平方。
- `--edges hed`：默认结构提取。`--edges canny` 是显式可选的无模型近似，不是原论文的边缘检测配置。
- `--blend sharp`：保留单路细节，但存在曝光差或配准残差时可能看到接缝。`--blend feather` 平滑过渡，可能在运动/视差区域产生重影。
- `--tile 512`：融合工作块，减小可节约临时内存，不改变输出分辨率。
- `--max-canvas-pixels 120000000`：默认最多 1.2 亿画布像素，适用于标定、单组输出和序列加载。超过时报告实际宽高和像素数，明确停止，不自动缩图。较大的预算仍要求足够系统内存。

### 宽视野多图的初始化

早期版本将单应矩阵沿匹配链组合到第 0 张图；对较宽视野的图组，这会在远端图片内产生投影奇点。例如 WandaBuilding 的第 7 张图曾得到超过四千万的坐标。现在使用归一化、固定参考自由度的联合相似拟合，再进行完整 HED/GES-GSP 网格优化。原始分辨率的输入不因画布预算而自动缩小。

对 `GES-06_WandaBuilding-1` 的 8 张 4000×2252 原图，自动参考选择为 4，已完成联合求解，画布为 **9190×6148**，保留匹配点 1350 对。该组无需使用 `--layout-grid`。下面的 PowerShell 写法与逐一传入 8 个文件相同：

```powershell
$images = 0..7 | ForEach-Object { '.\datasets\GES50-Dataset\GES-06_WandaBuilding-1\{0:D2}.jpg' -f $_ }
python .\stitch.py stitch --images @images -o output/wanda-new.png
```

当前稀疏求解及 HED 推理使用 CPU。没有将普通 pip OpenCV 的 CUDA 模块视为可用 GPU 后端，也没有在本实现中保留上一版 OpenCL 选项。输出尺寸与是否用 GPU 没有关系。HED 需要一定系统内存，显存不能替代它。

建议在静态、有纹理、同步的画面上标定；广角畸变应先去畸变。近远景大视差、动态前景或自动曝光差仍可能造成接缝，网格优化不能保证三维场景处处对齐。

## 验证

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试涵盖 GES 对相似变换的不变性、网格采样精确像素、跨块连续性、标定保存加载、分辨率检查、四路低分辨率图片的联合优化、无重叠失败、八图链的投影奇点回归、参考相机选择和画布预算。旧版四图样例留下的报告记录 RMS 从 20.54 降到约 0.79 像素；采用联合相似初始化的 WandaBuilding 实测 RMS 从约 11.90 降到 3.41 像素，中位误差约 1.19 像素。初始化改变后，具体输出应以对应报告为准。这些都是参与拟合的匹配点误差，**不是独立测试集指标，也不等价于主观画质提升倍数**。请打开原尺寸 PNG 检查接缝和纹理。

引用：Peng Du, Jifeng Ning, Jiguang Cui, Shaoli Huang, Xinchao Wang, Jiaxin Wang. Geometric Structure Preserving Warp for Natural Image Stitching. CVPR 2022, pp. 3688–3696.
