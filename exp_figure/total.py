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

# ===============================
# 一共四组指标的原始数据
# ===============================
metrics_data = {
    "P@1": {
        "non_llm": {
            "TF-IDF": [0.24, 0.19, 0.31],
            "BM25": [0.20, 0.25, 0.41],
            "LSI": [0.18, 0.13, 0.54],
            "JenS": [0.26, 0.18, 0.49],
            "Word2Vec": [0.02, 0.05, 0.11],
            "FastText": [0.03, 0.07, 0.04]
        },
        "llm": {
            "GPT-4": [0.49, 0.20, 0.73],
            "DeepSeek-R1": [0.54, 0.21, 0.55],
            "Qwen3-8B": [0.52, 0.21, 0.73],
            "Qwen3-14B": [0.60, 0.20, 0.61],
            "Qwen3-32B": [0.61, 0.20, 0.63]
        },
        "ranges": [(0.45, 0.75), (0.12, 0.45), (0.00, 0.12)]
    },

    "P@4": {
        "non_llm": {
            "TF-IDF": [0.42, 0.36, 0.62],
            "BM25": [0.36, 0.40, 0.67],
            "LSI": [0.32, 0.21, 0.82],
            "JenS": [0.45, 0.30, 0.81],
            "Word2Vec": [0.04, 0.14, 0.20],
            "FastText": [0.08, 0.15, 0.21]
        },
        "llm": {
            "GPT-4": [0.67, 0.40, 0.87],
            "DeepSeek-R1": [0.63, 0.32, 0.67],
            "Qwen3-8B": [0.66, 0.34, 0.82],
            "Qwen3-14B": [0.70, 0.50, 0.69],
            "Qwen3-32B": [0.70, 0.30, 0.74]
        },
        "ranges": [(0.60, 0.90), (0.24, 0.60), (0.00, 0.24)]
    },

    "DCG@2": {
        "non_llm": {
            "TF-IDF": [0.30, 0.23, 0.43],
            "BM25": [0.25, 0.30, 0.50],
            "LSI": [0.22, 0.15, 0.65],
            "JenS": [0.32, 0.22, 0.59],
            "Word2Vec": [0.03, 0.08, 0.15],
            "FastText": [0.04, 0.09, 0.09]
        },
        "llm": {
            "GPT-4": [0.52, 0.20, 0.80],
            "DeepSeek-R1": [0.56, 0.26, 0.61],
            "Qwen3-8B": [0.56, 0.21, 0.79],
            "Qwen3-14B": [0.66, 0.33, 0.68],
            "Qwen3-32B": [0.66, 0.32, 0.79]
        },
        "ranges": [(0.54, 0.85), (0.18, 0.54), (0.00, 0.18)]
    },

    "DCG@5": {
        "non_llm": {
            "TF-IDF": [0.36, 0.29, 0.49],
            "BM25": [0.30, 0.34, 0.58],
            "LSI": [0.27, 0.19, 0.71],
            "JenS": [0.38, 0.26, 0.68],
            "Word2Vec": [0.04, 0.10, 0.16],
            "FastText": [0.06, 0.12, 0.14]
        },
        "llm": {
            "GPT-4": [0.55, 0.34, 0.82],
            "DeepSeek-R1": [0.56, 0.26, 0.63],
            "Qwen3-8B": [0.56, 0.31, 0.82],
            "Qwen3-14B": [0.66, 0.37, 0.72],
            "Qwen3-32B": [0.66, 0.32, 0.80]
        },
        "ranges": [(0.63, 0.90), (0.19, 0.63), (0.00, 0.19)]
    }
}

ecosystems = ["JavaScript", "HuggingFace", "Linux"]
x = range(len(ecosystems))
dashed_style = (0, (4, 3))

# ===============================
#   统一绘图函数：创建三段断轴图
# ===============================
def create_broken_axis(ax, metric_name, data):
    non_llm = data["non_llm"]
    llm = data["llm"]
    (top, mid, bottom) = data["ranges"]

    ax_top, ax_mid, ax_bottom = ax

    ax_top.set_ylim(*top)
    ax_mid.set_ylim(*mid)
    ax_bottom.set_ylim(*bottom)

    method_colors, method_markers, method_styles = {}, {}, {}

    def plot_all(target_ax):
        # 非 LLM
        for name, vals in non_llm.items():
            line = target_ax.plot(
                x, vals,
                marker='o', markersize=7, linewidth=2,
                linestyle=dashed_style
            )[0]
            method_colors[name] = line.get_color()
            method_markers[name] = 'o'
            method_styles[name] = dashed_style

        # LLM
        for name, vals in llm.items():
            line = target_ax.plot(
                x, vals,
                marker='s', markersize=7, linewidth=2.5,
                linestyle='-'
            )[0]
            method_colors[name] = line.get_color()
            method_markers[name] = 's'
            method_styles[name] = '-'

        return method_colors, method_markers, method_styles

    # 绘制三段
    plot_all(ax_top)
    plot_all(ax_mid)
    plot_all(ax_bottom)

    # 断轴符号
    for ax1, ax2 in [(ax_top, ax_mid), (ax_mid, ax_bottom)]:
        ax1.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)

    dd = 0.015
    kw = dict(color='k', clip_on=False, linewidth=1.2)

    # top
    ax_top.plot((-dd, dd), (-dd, dd), transform=ax_top.transAxes, **kw)
    ax_top.plot((1-dd, 1+dd), (-dd, dd), transform=ax_top.transAxes, **kw)

    # mid top
    ax_mid.plot((-dd, dd), (1-dd, 1+dd), transform=ax_mid.transAxes, **kw)
    ax_mid.plot((1-dd, 1+dd), (1-dd, 1+dd), transform=ax_mid.transAxes, **kw)

    # mid bottom
    ax_mid.plot((-dd, dd), (-dd, dd), transform=ax_mid.transAxes, **kw)
    ax_mid.plot((1-dd, 1+dd), (-dd, dd), transform=ax_mid.transAxes, **kw)

    # bottom
    ax_bottom.plot((-dd, dd), (1-dd, 1+dd), transform=ax_bottom.transAxes, **kw)
    ax_bottom.plot((1-dd, 1+dd), (1-dd, 1+dd), transform=ax_bottom.transAxes, **kw)

    # X 轴
    ax_bottom.set_xticks(list(x))
    ax_bottom.set_xticklabels(ecosystems, fontsize=12, fontweight='bold')

    for a in [ax_top, ax_mid, ax_bottom]:
        a.grid(True, linestyle='--', alpha=0.5)

    ax_top.set_ylabel(metric_name, fontsize=15, fontweight='bold')

    return method_colors, method_markers, method_styles


# ===============================
# 创建 2×2 大图
# ===============================
fig = plt.figure(figsize=(14, 11))

# 子图布局（每个子图包含 3 行断轴子图）
gs = fig.add_gridspec(6, 2, height_ratios=[1,1,1,1,1,1])

ax_positions = {
    "P@1": (0, 0),
    "P@4": (0, 1),
    "DCG@2": (3, 0),
    "DCG@5": (3, 1),
}

all_colors, all_markers, all_styles = {}, {}, {}

for metric_name, (start_row, col) in ax_positions.items():
    ax_top = fig.add_subplot(gs[start_row, col])
    ax_mid = fig.add_subplot(gs[start_row+1, col], sharex=ax_top)
    ax_bottom = fig.add_subplot(gs[start_row+2, col], sharex=ax_mid)

    colors, markers, styles = create_broken_axis(
        (ax_top, ax_mid, ax_bottom),
        metric_name,
        metrics_data[metric_name]
    )
    all_colors.update(colors)
    all_markers.update(markers)
    all_styles.update(styles)

# ===============================
# 全局 Legend
# ===============================
handles = []
labels = []

for name in list(metrics_data["P@1"]["non_llm"].keys()) + list(metrics_data["P@1"]["llm"].keys()):
    h = Line2D(
        [0], [0],
        color=all_colors[name],
        marker=all_markers[name],
        markersize=7,
        linewidth=2.5 if all_styles[name] == '-' else 2,
        linestyle=all_styles[name]
    )
    handles.append(h)
    labels.append(name)

fig.legend(
    handles, labels,
    loc='upper center',
    ncol=5,
    fontsize=11,
    framealpha=0.9,
    columnspacing=1.4
)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("all_metrics.pdf", format="pdf", bbox_inches="tight")
plt.show()