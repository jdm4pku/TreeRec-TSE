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

# ========== 数据：DCG@2 ==========
ecosystems = ["JavaScript", "HuggingFace", "Linux"]
x = range(len(ecosystems))

non_llm = {
    "TF-IDF": [0.30, 0.23, 0.43],
    "BM25": [0.25, 0.30, 0.50],
    "LSI": [0.22, 0.15, 0.65],
    "JenS": [0.32, 0.22, 0.59],
    "Word2Vec": [0.03, 0.08, 0.15],
    "FastText": [0.04, 0.09, 0.09]
}

llm = {
    "GPT-4": [0.52, 0.20, 0.80],
    "DeepSeek-R1": [0.56, 0.26, 0.61],
    "Qwen3-8B": [0.56, 0.21, 0.79],
    "Qwen3-14B": [0.66, 0.33, 0.68],
    "Qwen3-32B": [0.66, 0.32, 0.79]
}

# ======== 创建三段断轴结构 ========
fig, (ax_top, ax_mid, ax_bottom) = plt.subplots(
    3, 1, sharex=True, figsize=(7.4, 6.2),
    gridspec_kw={'height_ratios': [3, 2, 1]}
)

# DCG@2 三段区间
ax_top.set_ylim(0.54, 0.85)
ax_mid.set_ylim(0.18, 0.54)
ax_bottom.set_ylim(0.00, 0.18)

# 样式
dashed_style = (0, (4, 3))
method_colors = {}
method_markers = {}
method_styles = {}

# ======== 统一绘线函数 ========
def plot_all(ax):
    # 非 LLM
    for name, vals in non_llm.items():
        line = ax.plot(
            x, vals,
            marker='o', markersize=7,
            linewidth=2, linestyle=dashed_style
        )[0]
        method_colors[name] = line.get_color()
        method_markers[name] = 'o'
        method_styles[name] = dashed_style

    # LLM
    for name, vals in llm.items():
        line = ax.plot(
            x, vals,
            marker='s', markersize=7,
            linewidth=2.5, linestyle='-'
        )[0]
        method_colors[name] = line.get_color()
        method_markers[name] = 's'
        method_styles[name] = '-'

plot_all(ax_top)
plot_all(ax_mid)
plot_all(ax_bottom)

# ======== 断轴符号 ========
for ax1, ax2 in [(ax_top, ax_mid), (ax_mid, ax_bottom)]:
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)

d = 0.015
kwargs = dict(color='k', clip_on=False, linewidth=1.2)

# top
ax_top.plot((-d, d), (-d, d), transform=ax_top.transAxes, **kwargs)
ax_top.plot((1 - d, 1 + d), (-d, d), transform=ax_top.transAxes, **kwargs)

# mid top
ax_mid.plot((-d, d), (1 - d, 1 + d), transform=ax_mid.transAxes, **kwargs)
ax_mid.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_mid.transAxes, **kwargs)

# mid bottom
ax_mid.plot((-d, d), (-d, d), transform=ax_mid.transAxes, **kwargs)
ax_mid.plot((1 - d, 1 + d), (-d, d), transform=ax_mid.transAxes, **kwargs)

# bottom
ax_bottom.plot((-d, d), (1 - d, 1 + d), transform=ax_bottom.transAxes, **kwargs)
ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_bottom.transAxes, **kwargs)

# ======== 坐标轴 ========
ax_bottom.set_xticks(list(x))
ax_bottom.set_xticklabels(ecosystems, fontsize=14, fontweight='bold')

for ax in [ax_top, ax_mid, ax_bottom]:
    ax.grid(True, linestyle='--', alpha=0.5)

ax_top.set_ylabel("DCG@2", fontsize=16, fontweight='bold')

plt.subplots_adjust(top=0.88, bottom=0.14, hspace=0.08)

# ======== Legend ========
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
    loc='upper center', ncol=5,
    fontsize=10, framealpha=0.9,
    columnspacing=1.2
)

plt.savefig("dcg2.pdf", format="pdf", bbox_inches="tight")
plt.show()