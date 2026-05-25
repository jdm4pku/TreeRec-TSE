import matplotlib.pyplot as plt
import numpy as np

# 数据
agents = [
    "Expert + Facilitative",
    "Base + Facilitative",
    "Directive",
    "Base + Interrogative",
    "Expert + Interrogative"
]

data = np.array([
    [13, 6, 5, 3, 4],
    [8, 10, 6, 5, 2],
    [7, 6, 6, 5, 7],
    [3, 8, 6, 6, 8],
    [1, 8, 12, 10, 0]
])

ranks = ["Rank 1", "Rank 2", "Rank 3", "Rank 4", "Rank 5"]

# 每个 agent 一种颜色，Rank 越靠后颜色越浅
base_colors = [
    "#F47C6B",  # red
    "#F5B52E",  # orange
    "#69BE8F",  # green
    "#5DA9E9",  # blue
    "#CFE8F3"   # light blue
]

fig, ax = plt.subplots(figsize=(8.5, 2.4))

y_pos = np.arange(len(agents))

for i, agent in enumerate(agents):
    left = 0
    total = data[i].sum()
    
    for j in range(data.shape[1]):
        value = data[i, j]
        if value == 0:
            continue
        
        # 用 alpha 制造从深到浅的效果
        alpha = 0.95 - j * 0.14
        
        ax.barh(
            y_pos[i],
            value,
            left=left,
            height=0.75,
            color=base_colors[i],
            alpha=alpha,
            edgecolor="white",
            linewidth=0.5
        )
        
        # 在每一段中间标数字
        ax.text(
            left + value / 2,
            y_pos[i],
            str(value),
            ha="center",
            va="center",
            fontsize=9
        )
        
        left += value

# y 轴标签
ax.set_yticks(y_pos)
ax.set_yticklabels(agents, fontsize=10)
ax.invert_yaxis()

# x 轴只显示 Rank 1 和 Rank 5
ax.set_xlim(0, data.sum(axis=1).max())
ax.set_xticks([0, data.sum(axis=1).max()])
ax.set_xticklabels(["Rank 1", "Rank 5"], fontsize=10)

# 标题
ax.set_title("Rankings of Each Agent", fontsize=11)

# 去掉多余边框
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

# 保留底部轴线
ax.spines["bottom"].set_color("#666666")

ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", length=0)

plt.tight_layout()
plt.savefig("agent_rankings.pdf", bbox_inches="tight")
plt.savefig("agent_rankings.png", dpi=300, bbox_inches="tight")
plt.show()