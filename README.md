# Modeling the Neural Computational Mechanisms of Cuttlefish Color Change via Feedback Optimization

本项目是 Introduction to AI for Science 课程期末项目，目标是构建一个简化的、可解释的 in-silico biological agent，用反馈优化算法模拟乌贼 / 墨鱼伪装过程中的皮肤变色动态。项目不以传统机器学习意义上的分类准确率为主要目标，而是尝试复现真实伪装行为中观察到的动态特征：高维皮肤图案空间、非刻板化轨迹、曲折与间歇性运动、以及逐步稳定到目标背景附近的过程。

## 1. 研究问题

真实乌贼能够根据环境纹理快速改变皮肤图案，实现伪装。过去研究倾向于认为伪装图案可以被归纳为少数离散类别，且执行过程主要依赖预设运动程序，反馈修正的空间有限。Woo et al. (2023) 的研究则表明，乌贼皮肤图案空间高度连续且高维，伪装轨迹并非从初始状态直接走向目标状态，而是在皮肤图案空间中表现出曲折、间歇、减速与再加速等复杂动态。

本项目据此提出一个计算建模问题：

> 能否将乌贼伪装过程理解为一个在高维皮肤状态空间中进行反馈优化的过程，并通过不同优化控制规则比较其生物学合理性？

## 2. 核心假设

项目的基本假设是：乌贼变色可以被抽象为一个由环境输入、虚拟皮肤状态、视觉反馈损失和神经控制规则共同构成的闭环优化系统。

具体而言：

1. 虚拟皮肤不是任意像素图像，而是由一组可解释的皮肤图案基底组合生成。
2. 视觉目标不是像素级完全一致，而是与环境背景在纹理、边缘和高级视觉特征上相似。
3. 神经控制可以被近似为对皮肤激活向量的迭代更新。
4. 不同控制规则会产生不同的动态轨迹，其中部分轨迹可能更接近真实乌贼伪装行为。

## 3. 文献背景

### 3.1 乌贼伪装的生物学基础

乌贼伪装依赖视觉系统对环境的感知，并通过大脑控制皮肤中色素胞的收缩与扩张，形成与背景相匹配的图案。该行为涉及三个层面：

- 环境纹理感知；
- 视觉统计特征解释；
- 由大量色素胞共同构成的皮肤图案输出。

### 3.2 Woo et al. (2023)：伪装动态的复杂性

Woo et al. 对大量自然与人工背景下的乌贼伪装图像进行分析，发现皮肤图案空间并非少数离散类别，而是具有高维、连续的结构。PCA 结果提示皮肤图案可能需要多达约 60 个维度描述。伪装轨迹也不是固定模板式运动，而是在皮肤状态空间中表现出弯曲路径、停滞区域、速度波动和逐步稳定等现象。

这一发现为本项目提供了核心生物学动机：乌贼伪装可能更接近一种反馈驱动的动态搜索过程，而不是简单的记忆调用或固定程序执行。

### 3.3 Perceptual Loss：从像素相似到视觉相似

在本项目中，目标不是让虚拟皮肤与背景在像素层面完全一致，而是让它们在视觉特征空间中相似。因此项目引入 Johnson et al. (2016) 中的 perceptual loss 思路：使用预训练视觉网络提取图像特征，并在特征空间中比较两张图像的相似性。

在计算实现上，可使用 ImageNet 预训练的 VGG16 作为冻结特征提取器，选取中间层特征图，例如 ReLU_3_3，用于比较虚拟皮肤状态与目标背景之间的视觉差异。

## 4. 模型设计总览

项目目标是构建一个简化的白箱生物智能体，具有以下特征：

1. **Biologically grounded constraints**  
   模型中的皮肤状态由有限图案基底和激活权重表示，而非任意自由像素优化，从而保留色素胞式局部纹理模块的约束。

2. **Interpretability at every stage**  
   环境输入、图案基底、激活向量、损失函数、优化轨迹和最终皮肤状态均可被记录和可视化。

3. **Bridge between biological mechanism and computational optimization**  
   将乌贼伪装行为理解为从视觉反馈到皮肤运动控制的闭环优化过程，使生物行为和机器学习优化算法之间建立可比较的形式化联系。

## 5. Pipeline

### Phase 1: Environment & Body

构建环境与虚拟身体表示。

输入包括：

- 一组复杂自然背景图像，例如 sand、gravel、coral 等；
- 一个皮肤图案基底矩阵；
- 一组初始激活权重向量；
- 由基底和权重渲染得到的虚拟皮肤状态。

皮肤图案基底可以通过两种方式构建：

1. **Toy model basis**  
   使用随机噪声生成平滑纹理基底，并上采样到目标分辨率。

2. **Data-driven basis**  
   使用 VAE、NMF 或其他降维方法，从背景纹理或皮肤图案中学习局部 chromatophore-like texture modules。

虚拟皮肤状态可表示为：

```text
skin_state = render(basis_matrix, activation_vector)
```

### Phase 2: Vision & Loss

使用冻结的视觉网络计算虚拟皮肤与目标背景之间的视觉差异。

步骤包括：

1. 将 target background 和 rendered skin state 输入 VGG16；
2. 提取指定层的 feature maps；
3. 在特征空间中计算 perceptual loss；
4. 将损失作为反馈信号传递给优化器。

在初始 toy model 中，可以先使用 MSE loss 作为简化视觉反馈；在完整模型中再替换或扩展为 perceptual loss。

### Phase 3: Brain & Optimizer

优化器模拟神经控制规则，接收视觉反馈损失，并更新虚拟皮肤的激活向量。

拟比较三类控制方式：

1. **Baseline optimizer**
   - SGD
   - Adam

2. **Stochastic controller**
   - Langevin dynamics
   - 在梯度更新中加入噪声，用于模拟神经系统中的随机扰动或探索行为。

3. **Continuous-time controller**
   - Neural ODE
   - 将皮肤激活状态的变化建模为连续时间动力系统。

每一步更新后，新的 activation vector 被送回渲染模块，生成新的虚拟皮肤状态，再进入视觉反馈环节。

### Phase 4: Evaluation

记录整个优化过程中的激活向量序列和图像序列，并从视觉收敛、轨迹结构、参数可解释性和生物可比性四个方面评估模型。

评估输出包括：

- 最终皮肤图案是否接近目标背景；
- 高维激活轨迹是否呈现曲折、弯曲、停滞或间歇性动态；
- 学习率、噪声强度、基底数量等参数是否能解释不同动态行为；
- 模拟 PCA 轨迹是否与真实乌贼伪装轨迹具有可比性。

## 6. 核心交付物

### 6.1 Dynamic Visualization

生成 GIF 或视频，展示虚拟皮肤图案如何从初始随机状态逐步收敛到目标背景。

最低要求：

- 每隔若干 step 保存一次 rendered skin image；
- 将所有帧合成为 GIF；
- 同时保存 target image 作为对照。

### 6.2 PCA Trajectory Visualization

记录每一步的 activation vector，形成时间序列矩阵：

```text
W = [w_0, w_1, ..., w_T]
```

其中每一行或每一列对应一个时间步的高维激活状态。

随后使用 PCA 将高维轨迹投影到二维空间，并绘制优化路径。图中应标注：

- 初始点；
- 中间轨迹；
- 最终点；
- 可选：速度、停滞区域或 loss 变化。

### 6.3 Biological Comparison

将模拟得到的 PCA 轨迹与 Woo et al. 等文献中真实乌贼伪装轨迹进行比较。

可比较维度包括：

- 路径是否直线化；
- 是否存在曲折或迂回；
- 是否存在低速停滞区；
- 是否出现反复调整；
- 最终是否稳定收敛；
- 不同优化器的轨迹是否呈现不同生物合理性。

如有可用真实轨迹数据，可进一步使用 Dynamic Time Warping，比较模拟轨迹与真实轨迹的形状相似性。

## 7. 参考文献

- Woo, T., Liang, X., Evans, D. A., et al. (2023). *The dynamics of pattern matching in camouflaging cuttlefish*. Nature, 619, 122–128.
- Johnson, J., Alahi, A., & Fei-Fei, L. (2016). *Perceptual Losses for Real-Time Style Transfer and Super-Resolution*. ECCV 2016.

## 8. 当前代码实现

当前仓库已经从单文件 demo 扩展为可复现实验框架。核心代码位于 `src/`：

```text
src/
├── basis.py          # smooth chromatophore-like basis construction
├── render.py         # activation vector -> rendered skin
├── losses.py         # MSE and optional VGG16 perceptual feedback
├── controllers.py    # Adam, SGD, Langevin-like controllers
├── simulation.py     # closed-loop simulation and run saving
├── evaluation.py     # PCA, velocity, path and curvature metrics
├── visualization.py  # loss/PCA/velocity plots and GIF creation
├── utils.py
└── main.py           # command-line entry point
```

实验配置位于 `configs/`：

- `toy_adam.yaml`
- `toy_sgd.yaml`
- `toy_langevin.yaml`
- `perceptual_vgg16.yaml`
- `dtd_texture_adam.yaml`
- `dtd_texture_perceptual.yaml`
- `procedural_checkerboard_langevin.yaml`
- `procedural_checkerboard_perceptual.yaml`
- `toy_intermittent_decay.yaml`
- `toy_intermittent_decay_stochastic.yaml`
- `dtd_texture_intermittent_mse.yaml`
- `dtd_texture_intermittent_perceptual.yaml`
- `procedural_checkerboard_intermittent_perceptual.yaml`

历史原型保留在 `Demo 5.20/520demo.py`，新的可复现实验建议使用 `src.main` 运行。

## 9. 运行方式

安装依赖：

```bash
pip install -r requirements.txt
```

运行一个 toy Adam 实验：

```bash
python -m src.main --config configs/toy_adam.yaml
```

运行 Langevin-like stochastic controller：

```bash
python -m src.main --config configs/toy_langevin.yaml
```

`perceptual_vgg16.yaml` 会使用 `torchvision` 的 VGG16 特征提取器；如果本机没有缓存预训练权重，第一次运行可能需要联网下载模型权重。

运行真实纹理背景的 perceptual loss 实验：

```bash
python -m src.main --config configs/dtd_texture_perceptual.yaml
```

运行 checkerboard 控制背景的 perceptual loss 实验：

```bash
python -m src.main --config configs/procedural_checkerboard_perceptual.yaml
```

当前 perceptual 配置使用冻结 VGG16 的 `relu1_2` 和 `relu2_2` 特征，loss 完全来自特征空间，不再混入像素级 MSE。

运行间歇采样 + 指数衰减反馈 controller：

```bash
python -m src.main --config configs/toy_intermittent_decay.yaml
python -m src.main --config configs/dtd_texture_intermittent_perceptual.yaml
```

该 controller 在采样步计算并缓存视觉梯度，采样间隔内不重新观察梯度，而是使用：

```text
feedback_gain = exp(-steps_since_sample / decay_tau)
u_{t+1} = motor_momentum * u_t - learning_rate * feedback_gain * cached_gradient
w_{t+1} = w_t + u_{t+1} + noise
```

运行后会额外保存 `feedback.csv`、`feedback_gain.csv` 和 `sample_events.csv`，用于分析反馈输入的衰减与重新激活。

下载和预处理真实/控制背景：

```bash
.venv/bin/python -m src.download_dtd --output-root data/backgrounds_raw/dtd
.venv/bin/python -m src.procedural_backgrounds --output data/backgrounds_raw/procedural --size 256
.venv/bin/python -m src.data_prep --input data/backgrounds_raw/procedural --output data/backgrounds_processed/128_gray --size 128 --source procedural --license generated --manifest data/manifests/procedural_128_gray.csv
.venv/bin/python -m src.data_prep --input data/backgrounds_raw/dtd/dtd/images --output data/backgrounds_processed/128_gray --size 128 --source dtd --license see_dtd_terms --manifest data/manifests/dtd_128_gray.csv
```

覆盖随机种子：

```bash
python -m src.main --config configs/toy_adam.yaml --seed 123
```

每次实验会生成一个独立目录：

```text
outputs/runs/{run_name}_{YYYYMMDD_HHMMSS}_seed{seed}/
├── config.yaml
├── metadata.json
├── basis.pt
├── weights.npy
├── losses.csv
├── velocity.csv
├── metrics.json
├── pca_trajectory.npy
├── target.png
├── final_skin.png
├── skin_convergence.gif   # target / current skin / PCA trajectory 三联动态图
├── skin_only.gif          # 仅皮肤状态变化的辅助 GIF
├── frames/
└── figures/
    ├── loss_curve.png
    ├── pca_trajectory.png
    └── velocity_curve.png
```

`weights.npy` 保存完整高维 activation trajectory，是后续与 Woo et al. 真实轨迹进行 PCA/DTW/定性比较的主要数据对象。
`skin_convergence.gif` 会把目标背景、模拟皮肤状态和 PCA 轨迹放在同一张动态图中，便于直接展示反馈优化过程。

如果某些历史 run 缺少 GIF，或者需要强制重绘所有可视化结果：

```bash
.venv/bin/python -m src.render_visuals --all --force
```
