import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

# === 全局字体 ===
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['figure.dpi'] = 200

# ========== 数据：运行时间（来自你的表格） ==========
ecosystems = ["JavaScript", "HuggingFace", "Linux"]
x = range(len(ecosystems))

non_llm = {
    "TF-IDF":     [0.0029, 0.0264, 0.0008],
    "BM25":       [0.0193, 0.0211, 0.0005],
    "LSI":        [0.0179, 0.0039, 0.0009],
    "JenS":       [3.4352, 11.239, 0.0067],
    "Word2Vec":   [0.0071, 0.0022, 0.0007],
    "FastText":   [0.0070, 0.0027, 0.0006]
}

llm = {
    "GPT-4":       [643, 525, 11],
    "DeepSeek-R1": [814, 449, 297],
    "Qwen3-8B":    [80, 292, 42],
    "Qwen3-14B":   [44, 718, 80],
    "Qwen3-32B":   [61, 853, 35]
}

# ========== 创建三段断轴 ==========
fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
    3, 1, sharex=True, figsize=(7.4, 6.2),
    gridspec_kw={'height_ratios': [3, 2, 1]}
)

# 根据数值自动设置断轴区间
ax_top.set_ylim(50, 900)     # LLM section
ax_mid.set_ylim(2, 18)     # Middle section
ax_bot.set_ylim(0, 0.03)      # Tiny section

dashed_style = (0, (4, 3))
method_colors = {}
method_markers = {}
method_styles = {}

# ===== 绘矩统一函数 =====
def plot_all(ax):
    # Non-LLM: dashed + circle
    for name, vals in non_llm.items():
        line = ax.plot(
            x, vals, marker='o', markersize=7,
            linewidth=2, linestyle=dashed_style
        )[0]
        method_colors[name] = line.get_color()
        method_markers[name] = 'o'
        method_styles[name] = dashed_style

    # LLM: solid + square
    for name, vals in llm.items():
        line = ax.plot(
            x, vals, marker='s', markersize=7,
            linewidth=2.5, linestyle='-'
        )[0]
        method_colors[name] = line.get_color()
        method_markers[name] = 's'
        method_styles[name] = '-'

plot_all(ax_top)
plot_all(ax_mid)
plot_all(ax_bot)

# ========== 断裂符号 ==========
for ax1, ax2 in [(ax_top, ax_mid), (ax_mid, ax_bot)]:
    ax1.spines["bottom"].set_visible(False)
    ax2.spines["top"].set_visible(False)

d = 0.015
kw = dict(color="k", clip_on=False, linewidth=1.2)

# top
ax_top.plot((-d, d), (-d, d), transform=ax_top.transAxes, **kw)
ax_top.plot((1-d, 1+d), (-d, d), transform=ax_top.transAxes, **kw)

# mid top
ax_mid.plot((-d, d), (1-d, 1+d), transform=ax_mid.transAxes, **kw)
ax_mid.plot((1-d, 1+d), (1-d, 1+d), transform=ax_mid.transAxes, **kw)

# mid bottom
ax_mid.plot((-d, d), (-d, d), transform=ax_mid.transAxes, **kw)
ax_mid.plot((1-d, 1+d), (-d, d), transform=ax_mid.transAxes, **kw)

# bottom
ax_bot.plot((-d, d), (1-d, 1+d), transform=ax_bot.transAxes, **kw)
ax_bot.plot((1-d, 1+d), (1-d, 1+d), transform=ax_bot.transAxes, **kw)

# ========== 轴设置 ==========
ax_bot.set_xticks(list(x))
ax_bot.set_xticklabels(ecosystems, fontsize=14, fontweight='bold')

for ax in [ax_top, ax_mid, ax_bot]:
    ax.grid(True, linestyle='--', alpha=0.5)

ax_top.set_ylabel("Runtime (ms)", fontsize=16, fontweight='bold')

plt.subplots_adjust(top=0.88, bottom=0.14, hspace=0.08)

# ========== Legend ==========
handles = []
labels = []

for name in list(non_llm.keys()) + list(llm.keys()):
    h = Line2D(
        [0], [0],
        color=method_colors[name],
        marker=method_markers[name],
        markersize=7,
        linewidth=2.5 if name in llm else 2,
        linestyle=method_styles[name]
    )
    handles.append(h)
    labels.append(name)

fig.legend(
    handles, labels,
    loc='upper center',
    ncol=5,
    fontsize=10,
    framealpha=0.9,
    columnspacing=1.2
)

plt.savefig("runtime.pdf", format="pdf", bbox_inches="tight")
plt.show()