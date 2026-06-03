import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import imageio
import os
from IPython.display import Image, display

# ==========================================
# 1. 物理环境与生物基底初始化
# ==========================================
torch.manual_seed(1)
np.random.seed(1)

# 生成 32 个平滑的"生物图斑基底" (模拟真实的运动单元，而不是纯噪点)
# 方法：生成低分辨率噪点，然后双三次插值放大，得到极其自然的类似细胞的平滑斑块
raw_basis = torch.randn(32, 1, 8, 8)
basis = F.interpolate(raw_basis, size=(64, 64), mode='bicubic', align_corners=False)
basis = torch.tanh(basis) # 限制在 -1 到 1 之间

# 随机生成一个极其复杂的"自然背景"作为终极目标
target_weights = torch.randn(32) * 1.5
target_bg = torch.sum(target_weights.view(-1, 1, 1, 1) * basis, dim=0)
target_bg = torch.sigmoid(target_bg).detach()

# ==========================================
# 2. 模拟包含"神经突触噪声"的优化大脑
# ==========================================
# 初始化章鱼的起始状态 (完全不伪装的初始权重)
w_brain = torch.randn(32, requires_grad=True)
optimizer = optim.Adam([w_brain], lr=0.2)

history_skin = []
history_weights = []
history_loss = []

print("正在模拟章鱼大脑的迭代寻优过程 (包含神经噪音)...")
for step in range(200):
    optimizer.zero_grad()
    
    # 根据当前的大脑信号 (权重)，渲染出当前的皮肤状态
    current_skin_raw = torch.sum(w_brain.view(-1, 1, 1, 1) * basis, dim=0)
    current_skin = torch.sigmoid(current_skin_raw)
    
    # 视觉反馈 (计算差异)
    loss = torch.mean((current_skin - target_bg)**2)
    loss.backward()
    
    # 更新大脑信号
    optimizer.step()
    
    # 【核心生物学设定】：注入神经传导的内禀噪声 (Langevin 动力学)
    # 这种噪声会让变色过程走弯路，但符合生物真实情况
    with torch.no_grad():
        noise = torch.randn(32) * 0.3 * (0.97 ** step) # 噪声随着接近目标逐渐减弱
        w_brain += noise
        
    # 每 2 步记录一次状态用于做动画
    if step % 2 == 0:
        history_skin.append(current_skin.squeeze().detach().numpy())
        history_weights.append(w_brain.detach().numpy().copy())
        history_loss.append(loss.item())

# ==========================================
# 3. 数据降维与 GIF 动画生成
# ==========================================
print("正在生成 PCA 轨迹并渲染 GIF 动画，请稍候...")
# 使用 PCA 将 32 维的神经信号降维到 2D 轨迹
pca = PCA(n_components=2)
traj_2d = pca.fit_transform(history_weights)

# 创建临时文件夹保存每一帧
os.makedirs("frames", exist_ok=True)
filenames = []

for i in range(len(traj_2d)):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor('#f8f9fa') # 加上高级的浅灰背景
    
    # 图 1：目标环境
    axes[0].set_title("Target Environment", fontsize=14, fontweight='bold')
    axes[0].imshow(target_bg.squeeze().numpy(), cmap='viridis')
    axes[0].axis('off')
    
    # 图 2：实时演化的皮肤
    axes[1].set_title(f"Virtual Cuttlefish Skin (Step {i*2})", fontsize=14, fontweight='bold')
    axes[1].imshow(history_skin[i], cmap='viridis')
    axes[1].axis('off')
    
    # 图 3：PCA 动态轨迹 (重点！)
    axes[2].set_title(f"Neural Trajectory in PCA Space\nLoss: {history_loss[i]:.4f}", fontsize=14, fontweight='bold')
    axes[2].plot(traj_2d[:i+1, 0], traj_2d[:i+1, 1], color='#e63946', linewidth=2, alpha=0.8) # 走过的历史轨迹
    axes[2].scatter(traj_2d[i, 0], traj_2d[i, 1], color='#e63946', s=100, zorder=5) # 当前点
    if i == 0:
        axes[2].scatter(traj_2d[0, 0], traj_2d[0, 1], color='green', s=150, marker='*', zorder=6, label="Start")
    axes[2].scatter(traj_2d[-1, 0], traj_2d[-1, 1], color='blue', s=150, marker='X', zorder=6, label="Goal")
    
    axes[2].set_xlim(traj_2d[:, 0].min() - 0.5, traj_2d[:, 0].max() + 0.5)
    axes[2].set_ylim(traj_2d[:, 1].min() - 0.5, traj_2d[:, 1].max() + 0.5)
    axes[2].grid(True, linestyle='--', alpha=0.6)
    if i == 0:
        axes[2].legend(loc='lower left')
    
    plt.tight_layout()
    
    # 保存当前帧
    filename = f"frames/frame_{i:03d}.png"
    plt.savefig(filename, dpi=100)
    filenames.append(filename)
    plt.close()

# 拼接为 GIF
gif_path = "cuttlefish_dynamics.gif"
with imageio.get_writer(gif_path, mode='I', duration=0.1) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# 在 Colab 中显示生成的动图
print("✅ GIF 生成完毕！")
display(Image(filename=gif_path))