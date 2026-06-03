# Modeling the Neural Computational Mechanisms of Cuttlefish Color Change via Feedback Optimization

**Course Project:** Introduction to AI for Science End-of-Semester Project Proposal  
**Team Members:** Yueran Wang, Jiahao Hu, Yizhan Feng

## Abstract

Cuttlefish can rapidly change their skin patterns to match complex natural backgrounds, making camouflage a striking example of tight coupling among visual perception, neural control, and bodily execution. Traditional views often describe cuttlefish camouflage as a transition among a small number of low-dimensional, categorical pattern classes. Recent large-scale behavioral imaging studies, however, suggest that cuttlefish skin-pattern space is high-dimensional and that repeated responses to the same background transition are not stereotyped. Instead, trajectories can be curved, intermittent, and repeatedly corrected before stabilizing. This suggests that camouflage may be better understood as a continuous feedback-driven optimization process rather than a simple retrieval of stored templates.

This project proposes a simplified and interpretable in-silico biological agent model. We will represent cuttlefish skin states as weighted combinations of local chromatophore-like texture bases, and use perceptual loss in the feature space of a pretrained visual network to measure visual similarity between the current skin pattern and the target environment. We will then compare several candidate control rules, including SGD/Adam, Langevin dynamics, and Neural ODEs, to study whether different optimization dynamics can reproduce high-dimensional, curved, and intermittent camouflage trajectories observed in real animals.

**Keywords:** cuttlefish camouflage; feedback optimization; perceptual loss; skin-pattern space; Langevin dynamics; Neural ODE

## 1. Motivation

Natural camouflage provides a compelling setting for research at the intersection of artificial intelligence and life science. Cuttlefish can rapidly adjust skin color, texture, and spatial pattern in response to complex seafloor backgrounds such as sand, gravel, and coral. This behavior is not merely a color substitution problem. It involves visual extraction of environmental statistics, neural coordination of skin chromatophores, and continuous body-wide pattern updates over short timescales.

![Example of cuttlefish camouflage](figures/cuttlefish-camouflage-frame-1.png)

One traditional explanation is that cuttlefish camouflage can be compressed into a small set of low-dimensional pattern classes, such as uniform, mottle, or disruptive patterns. Under this view, the animal mainly switches among a few predefined templates when the environment changes. However, Woo et al. (2023) analyzed large numbers of cuttlefish camouflage images over natural and artificial backgrounds and found that skin-pattern space does not form a few discrete clusters. Instead, it appears to be a continuous, high-dimensional manifold. Moreover, trajectories during repeated responses to the same background transition are not stereotyped: each search may meander through skin-pattern space, with repeated deceleration, acceleration, and local stagnation.

This observation naturally turns the biological problem into a computational one. If the current skin pattern is treated as a high-dimensional state $S_t$ and the environment as a target $B_{\mathrm{target}}$, camouflage can be modeled as a dynamic optimization process that minimizes a visual discrepancy in skin-pattern space.

## 2. Core Research Question

The central research question of this project is:

> Can a white-box optimization model with perceptual feedback reproduce the high-dimensional, non-stereotyped, and intermittent trajectories observed in cuttlefish camouflage, and thereby help compare the biological plausibility of different control rules?

To address this question, we will construct a simplified in-silico cuttlefish agent. The agent will contain a parameterized virtual skin, a frozen visual evaluation module, and several candidate neural feedback controllers. The model should not only generate visualizations of skin patterns gradually approaching the target background, but also output state trajectories that can be compared with real behavioral data.

## 3. Biological and Computational Background

### 3.1 High-Dimensional Skin-Pattern Space

Cuttlefish skin contains large numbers of neurally controlled chromatophores and reflective structures. A macroscopic skin pattern is therefore a coordinated combination of many local units across multiple spatial scales. Even when the visible change appears as darkening, spot formation, or boundary disruption, the underlying control process involves many coupled local variables. If a skin pattern is represented as an image or feature vector, camouflage naturally takes place in a high-dimensional state space.

Woo et al. parameterized cuttlefish skin images with a pretrained neural network and visualized the resulting pattern space using UMAP and PCA. Their results showed that skin states do not separate into a few clear classes. Instead, they form a continuous cloud-like distribution, and PCA analysis suggests that many more dimensions are needed than earlier low-dimensional accounts assumed.

![Low-dimensional visualization of skin-pattern space](figures/skin-pattern-umap.png)

### 3.2 From Memory Templates to Feedback Updates

If cuttlefish simply retrieved a stored target pattern after an environmental change, skin-state transitions should be relatively stereotyped. Repeated trials with the same start and target should follow similar paths, and the direction of motion should largely align with the direct path from the initial state to the final state. In contrast, experimental data show that real trajectories vary substantially across trials, often curve through pattern space, and include low-velocity regions.

![Non-stereotyped trajectories under repeated background transitions](figures/trajectory-repeated-trials.png)

More importantly, after these low-velocity regions, the exit direction tends to point toward the final stable camouflage state rather than simply remaining parallel to the direct start-to-end path. This supports an update model in which the system repeatedly evaluates its current error and adjusts the next movement direction. It is less consistent with a memory model in which the system executes a fixed stored trajectory or directly retrieves a target template.

![Velocity changes along camouflage trajectories](figures/trajectory-velocity.png)

![Comparison between update model and memory model](figures/update-vs-memory-model.png)

## 4. Model Design

We propose a simplified white-box biological agent model. The model does not attempt to reproduce every physiological detail of cuttlefish skin. Instead, it keeps three components that are directly relevant to the research question:

1. a parameterized virtual skin;
2. a visual error evaluation system;
3. a neural feedback optimization loop.

Together, these components form a closed loop. The current skin state is evaluated by the visual module, the resulting error signal drives an optimizer to update skin-control parameters, and the updated skin state is fed back into the next round of evaluation.

![Overall model framework](figures/model-overview.png)

### 4.1 Environment and Virtual Skin

The environment module provides a repository of target background images, initially including complex natural textures such as sand, gravel, coral, and mixed backgrounds. We denote the target background as $B_{\mathrm{target}}$. The virtual skin is represented by $N$ local chromatophore-like texture bases:

$$
B_{\mathrm{skin}} = \{B_1, B_2, \ldots, B_N\}.
$$

Each $B_k$ can be interpreted as a local color or texture module. In the initial toy model, these bases can be constructed from smoothed random textures. In later versions, the bases may be learned from real skin or texture data using non-negative matrix factorization (NMF) or variational autoencoders (VAE).

Given an activation vector $W_t = (w_{t,1}, \ldots, w_{t,N})$, the virtual skin state is defined as:

$$
S_t = \sum_{k=1}^{N} w_{t,k} B_k.
$$

To preserve biological constraints, later implementations may restrict the range of weights, add local smoothness constraints, or impose sparsity on basis activations. These constraints correspond to limited chromatophore activation strength and spatial correlation among nearby skin regions.

### 4.2 Visual Evaluation and Perceptual Loss

The visual module will use an ImageNet-pretrained VGG16 network as a frozen feature extractor. At each time step $t$, both the current skin state $S_t$ and the target background $B_{\mathrm{target}}$ are fed into the same network. Feature maps from selected intermediate layers, such as ReLU_3_3, are extracted and compared:

$$
L_{\mathrm{perc}}(S_t, B_{\mathrm{target}})
=
\sum_l \lambda_l
\left\|
\phi_l(S_t) - \phi_l(B_{\mathrm{target}})
\right\|_2^2.
$$

Here, $\phi_l(\cdot)$ denotes the feature representation at layer $l$ of the pretrained visual network, and $\lambda_l$ controls the weight assigned to each layer. Perceptual loss is chosen because it captures texture, edges, and local structure better than pixel-wise error, while remaining differentiable and therefore suitable for feedback optimization.

![Perceptual loss framework](figures/perceptual-loss-network.png)

### 4.3 Neural Feedback Controllers

The optimizer module receives the visual error signal and updates the skin activation vector $W_t$. We will compare three types of candidate control rules:

1. **Baseline optimizers: SGD/Adam.** SGD provides a minimal gradient-descent baseline, while Adam introduces first- and second-moment estimates and is widely used in high-dimensional nonconvex optimization.
2. **Stochastic dynamics: Langevin dynamics.** Noise is added to the gradient update to model internal neural variability, exploratory search, or stochasticity in skin execution.
3. **Continuous-time control: Neural ODE.** The evolution of $W_t$ is treated as a continuous-time dynamical system parameterized by a neural network, $dW/dt=f_\theta(W,t)$.

A simple noisy first-order update can be written as:

$$
W_{t+1}
=
W_t
-
\eta \nabla_{W_t} L_{\mathrm{perc}}(S_t, B_{\mathrm{target}})
+
\sigma \epsilon_t,
\quad
\epsilon_t \sim \mathcal{N}(0, I).
$$

Here, $\eta$ is the learning rate and $\sigma$ controls the noise strength. By varying $\eta$, $\sigma$, and the optimizer type, we can test whether the resulting trajectories show curvature, stagnation, repeated correction, or fast convergence similar to real cuttlefish behavior.

## 5. Experimental Plan

### 5.1 Toy Model Validation

Before introducing real backgrounds and learned skin bases, we will first implement a toy model to verify that the closed-loop optimization framework works. The simplified version will use $32$ smoothed random basis patterns generated by bicubic interpolation. The virtual skin image will initially be $64 \times 64$ pixels, controlled by a $32$-dimensional activation vector. Adam can be used as the first optimizer, with MSE as an initial visual feedback loss before replacing it with perceptual loss. Langevin-like noise can then be added to simulate neural variability.

![Toy model demonstration frame](figures/toy-demo-frame.png)

The toy model is not intended to produce biological conclusions by itself. Instead, it will test three basic requirements:

1. whether the parameterized skin can gradually approach a target background under optimization;
2. whether the sequence $W_0, W_1, \ldots, W_T$ forms a meaningful visualizable trajectory;
3. whether learning rate, noise strength, and optimizer type significantly change trajectory structure.

### 5.2 Trajectory Visualization

For each simulation, the model will store all activation vectors over time, forming:

$$
\mathcal{W}
=
\begin{bmatrix}
W_0^\top \\
W_1^\top \\
\vdots \\
W_T^\top
\end{bmatrix}
\in \mathbb{R}^{(T+1)\times N}.
$$

PCA will then be used to project this high-dimensional trajectory into two dimensions. We will examine whether the path is smooth, curved, stagnant, exploratory, or repeatedly corrected. This visualization parallels the trajectory analysis used by Woo et al. for real cuttlefish skin patterns.

### 5.3 Comparison with Real Biological Data

We will attempt to extract real cuttlefish camouflage trajectories from published figures, supplementary materials, or accessible datasets. If time-series data are available, simulated and real trajectories can be represented in a common low-dimensional space and compared using Dynamic Time Warping (DTW), which measures trajectory similarity even when temporal speeds differ. If complete raw data are unavailable, we will first conduct a qualitative comparison based on published trajectory figures and explicitly discuss the data limitation.

## 6. Evaluation Criteria

The model will be evaluated from four perspectives:

1. **Visual convergence.** Does the virtual skin gradually approach the target background, and does perceptual loss decrease and stabilize?
2. **Trajectory structure.** Does the PCA trajectory show smooth curvature, local stagnation, indirect movement, or exploratory perturbation?
3. **Parameter interpretability.** Can learning rate, noise strength, number of bases, and optimizer type explain different dynamic behaviors?
4. **Biological comparability.** Do simulated trajectories resemble real cuttlefish camouflage trajectories, either through DTW distance or qualitative structure?

These criteria ensure that the project does not only evaluate whether the final image looks similar to the background, but also whether the dynamic process that generates the image resembles real animal behavior.

## 7. Expected Deliverables

The project is expected to produce three main deliverables:

1. **Dynamic camouflage visualization.** A GIF or image sequence showing the virtual cuttlefish skin gradually approaching the target background, together with the loss curve.
2. **High-dimensional state trajectory analysis.** PCA projections of skin activation dynamics under different controllers.
3. **Biological plausibility comparison.** A comparison of trajectories generated by SGD/Adam, Langevin dynamics, and Neural ODE controllers against real cuttlefish camouflage trajectories.

## 8. Work Plan

The project will proceed in four stages:

1. **Prototype stage.** Implement the toy model with smoothed random skin bases, MSE or single-layer perceptual loss, and basic closed-loop optimization.
2. **Perceptual loss stage.** Introduce multi-layer VGG16 feature loss and test convergence under different target backgrounds and basis sizes.
3. **Controller comparison stage.** Compare Adam, Langevin-like noisy updates, and Neural ODE controllers; record loss, trajectory curvature, stagnation intervals, and convergence speed.
4. **Biological comparison stage.** Organize available data or figures from Woo et al., perform PCA trajectory comparison and DTW analysis when possible, and formulate the final biological interpretation.

## 9. Risks and Alternatives

The main challenges are the availability of real cuttlefish trajectory data and the mismatch between VGG16 features and the true cuttlefish visual system. We will use a two-level strategy. If raw or supplementary time-series data are available, we will perform quantitative DTW comparison. If only published trajectory figures are available, we will conduct qualitative trajectory-structure comparison and clearly state the limitation.

For the visual module, VGG16 will serve as the first differentiable perceptual proxy. Later versions may test lighter texture-statistics features or self-supervised visual models to evaluate whether the conclusions depend on a specific network architecture.

Overall, the goal is not to claim that a particular deep learning model is equivalent to the cuttlefish nervous system. Rather, the goal is to build a runnable, interpretable, and comparable computational framework in which the hypothesis “cuttlefish camouflage is a feedback optimization process” can be explicitly simulated and tested.

## References

- Woo, T., Liang, X., Evans, D. A., et al. (2023). *The dynamics of pattern matching in camouflaging cuttlefish*. Nature, 619, 122-128. https://doi.org/10.1038/s41586-023-06259-2
- Johnson, J., Alahi, A., & Fei-Fei, L. (2016). *Perceptual Losses for Real-Time Style Transfer and Super-Resolution*. ECCV.
- Simonyan, K., & Zisserman, A. (2015). *Very Deep Convolutional Networks for Large-Scale Image Recognition*. ICLR.
- Lee, D. D., & Seung, H. S. (1999). *Learning the parts of objects by non-negative matrix factorization*. Nature, 401, 788-791.
- Kingma, D. P., & Welling, M. (2014). *Auto-Encoding Variational Bayes*. ICLR.
- Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization*. ICLR.
- Welling, M., & Teh, Y. W. (2011). *Bayesian Learning via Stochastic Gradient Langevin Dynamics*. ICML.
- Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K. (2018). *Neural Ordinary Differential Equations*. NeurIPS.
- Sakoe, H., & Chiba, S. (1978). *Dynamic programming algorithm optimization for spoken word recognition*. IEEE Transactions on Acoustics, Speech, and Signal Processing, 26(1), 43-49.
