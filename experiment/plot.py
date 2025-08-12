import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from model.join_selector_settings import JoinSelectorSettings
import pandas as pd

settings = JoinSelectorSettings()

# print(plt.style.available)
# exit()


def read_data(benchmark, sheet_idx="Query_Cost"):
    name_mapping = {"TPC-H": "tpch", "TPC-DS": "tpcds", "JOB": "imdb"}
    # 从excel中读取数据
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = f"{base_dir}/result/sf={settings.scale_factor}/bs={settings.block_size}/rep={settings.replication_factor}/{name_mapping[benchmark]}_results.xlsx"

    # 读取xlsx文件
    data = pd.read_excel(data_path, sheet_name=sheet_idx)
    # 读取列名，和除列名行之外的数据
    columns = list(data.columns)
    data = data.values
    return columns, data


def plot_opt_routing_time(benchmarks):
    # methods = ["PAW runtime","PAW routing time", "AD-MTO runtime", "AD-MTO routing time", "PAC-Tree optimization time","PAC-Tree routing time"]  # Example methods
    methods = ["PAW runtime", "AD-MTO runtime", "PAC-Tree runtime"]  # Example methods
    cols = len(benchmarks)
    fig, axes = plt.subplots(1, cols, figsize=(12, 3.5))

    bar_width = 0.35
    titles = [f"{benchmark}" for benchmark in benchmarks]

    routing_time_list = [1.42, 1.35, 1.02]
    hatch_patterns = ["//", "xx", ".."]
    # hatch_colors = ["blue", "orange", "green"]
    hatch_colors = ["#4C72B0", "#55A868", "#C44E52"]
    idx = 0
    for j in range(cols):
        ax = axes[j]
        # data=[np.random.randint(5, 15) for _ in range(len(methods))]
        columns, data_arr = read_data(benchmarks[j], sheet_idx="Time_Cost")
        data = []
        for cnt in [1, 2]:
            data += data_arr[:, columns.index("Opt Time")].T.tolist()

        for method_idx, method in enumerate(methods):
            ax.bar(
                method_idx * 0.5,
                data[method_idx],
                width=bar_width,
                label=method,
                hatch=hatch_patterns[method_idx],
                edgecolor="black",  # Set bar border to black
                facecolor=hatch_colors[method_idx],
            )
            #为 ax 添加数值标签
            v = data[method_idx]
            ax.text(
                method_idx * 0.5,
                v,
                f"{v:.1f}s",
                ha="center",
                va="bottom",
                fontsize=12,
            )

        # 加一条分割线
        ax.axvline(x=1.26, color="black", linestyle="--", linewidth=0.5)

        ax_right = ax.twinx()
        ax_right.bar(
            3 * 0.5,
            routing_time_list[j],
            width=bar_width,
            label="Average data routing time",
            hatch="++",
            edgecolor="gray",  # Set bar border to black
            facecolor="none",
        )
        ax_right.set_ylabel("Routing Time (h)", fontsize=14, color="gray")
        ax_right.set_ylim(0, max(routing_time_list) * 1.2)
        # 为 ax_right 添加数值标签
        v = routing_time_list[j]
        # 右侧y轴设为gray色
        ax_right.spines["right"].set_color("gray")
        ax_right.text(3 * 0.5, v, f"{v:.2f}h", ha="center", va="bottom", fontsize=12)

        ax.set_xlabel("")
        # 设置y轴上限为柱形数值最大值的1.2倍
        ax.set_ylim(0, max(data) * 1.2)
        # 设置科学计数法
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

        ax.set_ylabel("Runtime Overhead (s)", fontsize=14)
        ax.set_title(titles[idx], y=-0.15, fontsize=14)  # Place title below
        ax.set_xticks([])  # Hide x-axis ticks
        idx += 1
        if j == 0:
            # Move legend to the center top of the entire figure
            handles, labels = ax.get_legend_handles_labels()
            handles2, labels2 = ax_right.get_legend_handles_labels()
            handles += handles2
            labels += labels2

            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.085),
                ncol=len(methods) + 1,
                fontsize=14,
            )
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.6)
    plt.savefig(f"./experiment/images/opt-routing-time.pdf", bbox_inches="tight")
    plt.show()


def estimate_query_trend(benchmarks):
    """Plot cumulative scan ratio reduction trend across query templates for different benchmarks."""
    # Plot configuration
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]  # Professional color scheme
    plt.figure(figsize=(6, 4))
    markers = ["P", "P", "P"]  # Reduced marker set for clarity

    # Style settings
    plt.style.use(["seaborn-v0_8-white"])
    plt.rcParams.update({"font.size": 12, "font.family": "serif"})

    # Plot data for each benchmark
    for b_id, benchmark in enumerate(benchmarks):
        # Read and preprocess data
        columns, data_arr = read_data(benchmark)
        data_arr = [row for row in data_arr if row[0] != "avg"]
        template_num = len(data_arr)

        # Calculate ratio reduction
        ratio_reduction = []
        for i in range(template_num):
            ad_mto_ratio = data_arr[i][columns.index("AD-MTO_Scan Ratio")]
            pac_tree_ratio = data_arr[i][columns.index("PAC-Tree_Scan Ratio")]
            reduction = (
                (ad_mto_ratio - pac_tree_ratio) * 100 / ad_mto_ratio
                if ad_mto_ratio > 0
                else 0
            )
            ratio_reduction.append(max(0, reduction))

        # Calculate cumulative sum and normalize x-axis
        y = np.cumsum(ratio_reduction)
        x = np.array(range(template_num)) / (template_num - 1)  # Normalize to [0,1]

        # Plot line with markers
        plt.plot(
            x,
            y,
            label=benchmark,
            color=colors[b_id],
            linestyle="-",
            marker=markers[b_id],
            markersize=6,
            linewidth=2,
            markeredgewidth=1.5,
        )

    # Configure axes
    plt.ylabel("Cumulative Scan Ratio Reduction (%)", fontsize=12)
    plt.xlabel("CDF Over Query Templates", fontsize=13)
    plt.xlim(0, 1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add legend
    plt.legend(
        loc="lower right", fontsize=13, frameon=True, edgecolor="black", fancybox=False
    )

    # Add grid and adjust layout
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    # Save figure
    plt.savefig(
        "./experiment/images/averge-query-trend.pdf", bbox_inches="tight", dpi=300
    )



def estimate_average_blocks(benchmarks):
    methods = ["PAW", "AD-MTO", "PAC-Tree"]
    data = []
    hatch_patterns = ["//", "xx", ".."]
    hatch_colors = ["#4C72B0", "#55A868", "#C44E52"]
    benchmark_num = len(benchmarks)
    # 设置量纲权重
    weights = [1, 2, 1]
    bar_width = 0.2

    # 使用 seaborn-whitegrid 风格
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(7, 4))
    
    x_labels = []
    for cnt in range(benchmark_num):
        columns, data_arr = read_data(benchmarks[cnt], sheet_idx="Table_Cost")
        # 取出 data_arr[:,0] 为 "avg" 的行，并按权重进行调整
        data_arr = [
            data_arr[i] * weights[cnt]
            for i in range(len(data_arr))
            if data_arr[i][0] == "avg"
        ]
        # 将每种方法的数据依次放入 data 数组
        data += [data_arr[0][columns.index(f"{method}_Scan Block")] for method in methods]
        # 坐标位置，生成例如: [cnt+1-0.2, cnt+1, cnt+1+0.2]
        x_labels += [cnt + 1 + op * bar_width for op in [-1, 0, 1]]
    
    # 绘制柱状图
    for i in range(len(x_labels)):
        method_idx = i % 3  # 依次分配到三种方法
        plt.bar(
            x_labels[i],
            data[i],
            width=bar_width,
            edgecolor=hatch_colors[method_idx],
            facecolor="none",
            hatch=hatch_patterns[method_idx],
            lw=1.5
        )
    
    # 为每个benchmark添加垂直箭头标识PAC-Tree相对于AD-MTO的降低比例
    ax = plt.gca()
    # For all benchmarks, we fix the horizontal and vertical offsets.
    offset_x = 0.1  # constant horizontal offset
    offset_y = 0.06  # constant vertical offset
    for cnt in range(benchmark_num):
        # 对于每个benchmark，PAW位于 cnt*3，AD-MTO位于 cnt*3+1，PAC-Tree位于 cnt*3+2
        ad_mto_val = data[cnt * 3 + 1]
        pac_tree_val = data[cnt * 3 + 2]
        if ad_mto_val > 0:
            reduction = ((ad_mto_val - pac_tree_val) / ad_mto_val) * 100
            # Place the arrow at the top right of the PAC-Tree bar.
            # PAC-Tree bar is at x = cnt+1+0.2 (see x_labels computation).
            tip_x = cnt + 1 + 0.14
            tip_y = pac_tree_val
            # The tail of the arrow is offset to the right and upward.
            tail_x = tip_x + offset_x
            tail_y = tip_y + offset_y
            ax.annotate(
                f"-{reduction:.0f}%",
                xy=(tip_x, tip_y),     # arrow tip at the top right of PAC-Tree bar
                xytext=(tail_x, tail_y),   # fixed offset for all benchmarks
                textcoords="data",
                ha="left",
                va="bottom",
                fontsize=13,
                fontweight="bold",
                color="black",
                arrowprops=dict(
                    arrowstyle="->",
                    color="black",
                    lw=3,  # thicker arrow line for clarity
                    connectionstyle="arc3,rad=-0.2"  # slight curvature
                )
            )
    
    # 设置x轴刻度为每个benchmark的中心位置
    centers = [cnt + 1 for cnt in range(benchmark_num)]
    plt.xticks(centers, benchmarks, fontsize=15, fontfamily="serif")
    
    plt.ylabel("Average number of blocks", fontsize=15, fontfamily="serif")
    plt.ylim(0, max(data) * 1.2)
    # 设置y轴刻度的大小
    plt.yticks(fontsize=12, fontfamily="serif")
    plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    
    # 显示图例：使用 Patch 对象来表示不同方法的样式
    handles = [
        Patch(facecolor="none", edgecolor=hatch_colors[i], hatch=hatch_patterns[i])
        for i in range(len(methods))
    ]
    plt.legend(handles, methods, loc="best", ncol=len(methods), fontsize=14, frameon=True)
    
    plt.tight_layout()
    plt.savefig(f"./experiment/images/average-blocks.pdf", bbox_inches="tight", dpi=300)
    plt.show()


def data_access_ratio_experiments(benchmarks):
    """Plot data access ratio comparison with improvement annotations."""
    # Basic setup
    plt.style.use(["seaborn-v0_8-white"])
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "figure.facecolor": "white"
    })

    methods = ["PAW", "AD-MTO", "PAC-Tree"]
    methods_colors = ["#4C72B0", "#55A868", "#C44E52"]
    hatch_patterns = ["//", "xx", ".."]
    total_length = 60

    for th, benchmark in enumerate(benchmarks):
        # Create figure
        fig = plt.figure(figsize=(12, 3.5))
        ax = plt.gca()
        
        # Set background style
        ax.set_facecolor("#f8f9fa")
        ax.set_axisbelow(True)
        ax.grid(True, color="white", linewidth=1.3, alpha=0.8)
        
        # Get query indices
        if benchmark == "TPC-H":
            query_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 19]
        elif benchmark == "JOB":
            query_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        elif benchmark == "TPC-DS":
            query_idx = [0, 1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13]

        query_num = len(query_idx)
        x_data = np.linspace(0, total_length - 1, query_num, dtype=int)
        
        # Read data
        columns, data_arr = read_data(benchmark)
        used_legend = False

        # Plot bars for each query
        for j in range(len(x_data)):
            for z in [-1, 0, 1]:
                x = x_data[j] + z
                y = data_arr[query_idx[j]][columns.index(f"{methods[z+1]}_Scan Ratio")]
                plt.bar(
                    x, y,
                    width=0.8,
                    label=methods[z+1] if not used_legend else "",
                    edgecolor=methods_colors[z+1],
                    facecolor="none",
                    hatch=hatch_patterns[z+1],
                    lw=1.5,
                    zorder=3
                )

            # Calculate and annotate improvement
            ad_mto = data_arr[query_idx[j]][columns.index("AD-MTO_Scan Ratio")]
            pac_tree = data_arr[query_idx[j]][columns.index("PAC-Tree_Scan Ratio")]
            
            if ad_mto > 0:
                improvement = ((ad_mto - pac_tree) / ad_mto) * 100
                ann_color = "#2ecc71" if improvement >= 0 else "#e74c3c"
                
                plt.annotate(
                    f"{improvement:+.0f}%",
                    xy=(x_data[j] + 1, max(ad_mto, pac_tree)),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    color=ann_color,
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        fc="white",
                        ec=ann_color,
                        lw=1.5,
                        alpha=0.9
                    ),
                    zorder=4
                )

            if not used_legend:
                used_legend = True
                plt.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.25),
                    ncol=3,
                    fontsize=12,
                    frameon=True,
                    edgecolor="black",
                    columnspacing=1.0
                )

        # Configure axes
        plt.yscale("log")
        plt.ylabel("Data Scan Ratio", fontsize=13, fontweight="bold")
        plt.xticks(x_data, [f"Q{i+1}" for i in range(query_num)], fontsize=12)
        plt.xlim(-2, total_length + 2)

        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)

        plt.tight_layout()
        # plt.savefig(
        #     f"./experiment/images/{benchmark}-access-ratio.pdf",
        #     bbox_inches="tight",
        #     dpi=300,
        #     facecolor="white",
        #     edgecolor="none"
        # )
        plt.show()
        plt.close()
        
        # Create latency statistics dictionary
        latency_stats = {
            method: {
                'under_1s': {bench: 0 for bench in benchmarks},
                '1s_to_3s': {bench: 0 for bench in benchmarks},
                '3s_to_5s': {bench: 0 for bench in benchmarks},
                '5s_to_10s': {bench: 0 for bench in benchmarks}
            } for method in methods
        }
        
        # Collect latency statistics for all methods
        for benchmark in benchmarks:
            columns, data_arr = read_data(benchmark)
            query_rows = [row for row in data_arr if row[0] != "avg"]
            
            # Calculate latencies for each method
            for method in methods:
                latencies = [
                    estimate_query_latency(row[columns.index(f"{method}_Scan Ratio")])
                    for row in query_rows
                ]
                
                # Categorize latencies
                for latency in latencies:
                    if latency < 1.0:
                        latency_stats[method]['under_1s'][benchmark] += 1
                    elif 1.0 <= latency < 3.0:
                        latency_stats[method]['1s_to_3s'][benchmark] += 1
                    elif 3.0 <= latency < 5.0:
                        latency_stats[method]['3s_to_5s'][benchmark] += 1
                    elif 5.0 <= latency < 10.0:
                        latency_stats[method]['5s_to_10s'][benchmark] += 1

        # Print formatted tables for each method
        for method in methods:
            print(f"\nQuery Execution Time Distribution for {method}:")
            print("-" * 60)
            print(f"{'Exec-time':<15} {'TPC-H':<15} {'TPC-DS':<15} {'JOB':<15}")
            print("-" * 60)
            print(f"{'< 1s':<15} {latency_stats[method]['under_1s']['TPC-H']:<15} {latency_stats[method]['under_1s']['TPC-DS']:<15} {latency_stats[method]['under_1s']['JOB']:<15}")
            print(f"{'1s - 3s':<15} {latency_stats[method]['1s_to_3s']['TPC-H']:<15} {latency_stats[method]['1s_to_3s']['TPC-DS']:<15} {latency_stats[method]['1s_to_3s']['JOB']:<15}")
            print(f"{'3s - 5s':<15} {latency_stats[method]['3s_to_5s']['TPC-H']:<15} {latency_stats[method]['3s_to_5s']['TPC-DS']:<15} {latency_stats[method]['3s_to_5s']['JOB']:<15}")
            print(f"{'5s - 10s':<15} {latency_stats[method]['5s_to_10s']['TPC-H']:<15} {latency_stats[method]['5s_to_10s']['TPC-DS']:<15} {latency_stats[method]['5s_to_10s']['JOB']:<15}")
            print("-" * 60)
        
        



def estimate_data_access_per_query(benchmark="TPC-H"):
    methods = ["PAW", "AD-MTO", "PAC-Tree"]  # Example methods
    if benchmark == "TPC-H":
        rows = 2
        cols = 7
        showed_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 19]
    elif benchmark == "JOB":
        rows = 2
        cols = 5
        showed_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    elif benchmark == "TPC-DS":
        rows = 2
        cols = 7
        showed_idxs = range(0, 14)

    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.7))

    bar_width = 0.35
    titles = [f"{benchmark} Q{i+1}" for i in showed_idxs]

    hatch_patterns = ["//", "xx", ".."]
    hatch_colors = ["blue", "orange", "green"]
    # 获取数据
    columns, data_arr = read_data(benchmark)
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= len(data_arr):
                break
            ax = axes[i][j]
            # data=[np.random.randint(5, 15) for _ in range(len(methods))]
            data = [
                data_arr[showed_idxs[idx]][columns.index(f"{method}_Scan Ratio")]
                for method in methods
            ]
            for method_idx, method in enumerate(methods):
                bars = ax.bar(
                    method_idx * 0.5,
                    data[method_idx],
                    width=bar_width,
                    label=method,
                    hatch=hatch_patterns[method_idx],
                    edgecolor=hatch_colors[method_idx],  # Set bar border to black
                    facecolor="none",  # Remove background color
                )
            ax.set_xlabel("")
            # 设置y轴上限为柱形数值最大值的1.2倍
            ax.set_ylim(0, max(data) * 1.3)
            # 设置科学计数法
            ax.ticklabel_format(
                style="sci", axis="y", scilimits=(0, 0), useOffset=False
            )
            # from matplotlib.ticker import FormatStrFormatter
            # ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            if j == 0:
                ax.set_ylabel("Scanning Ratio", fontsize=12)
            ax.set_title(titles[idx], y=-0.25, fontsize=13)  # Place title below
            ax.set_xticks([])  # Hide x-axis ticks
            idx += 1

    # Move legend to the center top of the entire figure
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(methods),
        bbox_to_anchor=(0.5, 1.08),
        fontsize=12,
    )
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5, hspace=0.33)
    plt.savefig(
        f"./experiment/images/{benchmark}-block-estimate.pdf", bbox_inches="tight"
    )
    # plt.show()



def scaling_datasize():
    """Plot model overhead analysis showing query response time and model runtime."""
    # Plot configuration
    benchmark = "TPC-DS"
    markers = ["o", "s"]  # 更专业的标记样式
    methods = ["PAC-Tree"]
    metric_colors = ["#2171b5", "#454545"]  # 使用更柔和的蓝色和深灰色
    metrics = ["Query Response Time", "Model Runtime"]
    metric_line_styles = ["-", "--"]  # 实线和虚线区分

    # Style settings
    plt.style.use(["seaborn-v0_8-white"])
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "figure.facecolor": "white",
        # "axes.facecolor": "#f8f9fa",  # 浅灰色背景
    })

    # Data scale constants
    K, M, G = 1024, 1024 * 1024, 1024 * 1024 * 1024
    machine_parallelism = 22
    block_sizes = [1000, 5000, 10000, 20000, 50000]
    scale_factor = [1, 5, 10, 20, 50]
    x_data = [x * G for x in [1, 5, 10, 20, 50]]
    normalized_x_data = [int(x / G) for x in x_data]

    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(6.2, 4.5))
    # ax1.set_facecolor("#f8f9fa")
    ax1.grid(True, linestyle="--", alpha=0.3, color="gray", zorder=0)

    # Plot query response time
    for method_idx, method in enumerate(methods):
        y1_data = []
        for data_idx, bs in enumerate(block_sizes):
            settings.block_size = bs
            columns, data_arr = read_data(benchmark)
            data_arr = [row for row in data_arr if row[0] == "avg"]

            data_read_scale = (
                x_data[data_idx] / G
                if x_data[data_idx] / G <= 2
                else max(2, (x_data[data_idx] / G) / machine_parallelism)
            )
            estimate_query_time = (
                data_arr[0][columns.index(f"{method}_Scan Ratio")]
                * data_read_scale
                * 1024
                / 1500
            )
            y1_data.append(estimate_query_time)

        # Plot line with enhanced style
        line1 = ax1.plot(
            normalized_x_data,
            y1_data,
            label=metrics[0],
            color=metric_colors[0],
            linestyle=metric_line_styles[0],
            marker=markers[0],
            markersize=8,
            linewidth=2.5,
            markerfacecolor="white",
            markeredgewidth=2,
            zorder=3
        )

        # Add data labels with enhanced style
        for i, txt in enumerate(y1_data):
            ax1.annotate(
                f"{txt:.2f}",
                (normalized_x_data[i], y1_data[i]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color=metric_colors[0]
            )

    # Add parallel processing annotation with improved style
    ax1.axvline(x=15, color=metric_colors[0], linestyle="--", 
                linewidth=1.5, alpha=0.5, zorder=2)
    ax1.text(
        23, 2.4,
        "Start Parallel",
        color=metric_colors[0],
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        bbox=dict(
            facecolor="white",
            edgecolor=metric_colors[0],
            alpha=0.8,
            pad=0.5,
            boxstyle="round"
        )
    )

    # Configure primary axis with improved style
    ax1.set_xlabel("Data Scale (GB)", fontsize=14)
    ax1.set_ylabel(f"{metrics[0]} (s)", fontsize=14, color=metric_colors[0])
    # 科学计数法
    # 显示整型数值刻度
    # ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax1.ticklabel_format(
        style="sci", axis="y", scilimits=(0, 0), useOffset=False
    )
    
    ax1.set_ylim(0, 2.6)
    ax1.tick_params(axis="y", colors=metric_colors[0], labelsize=12.5)
    ax1.tick_params(axis="x", labelsize=12.5)

    # Create secondary axis for model runtime
    ax2 = ax1.twinx()

    # Plot model runtime with enhanced style
    for method_idx, method in enumerate(methods):
        y2_data = []
        for sf in scale_factor:
            settings.block_size = int(10000 / sf)
            settings.scale_factor = sf
            columns, data_arr = read_data(benchmark, sheet_idx="Time_Cost")
            y2_data.append(data_arr[:, columns.index("Opt Time")].T.tolist()[2])

        line2 = ax2.plot(
            normalized_x_data,
            y2_data,
            label=metrics[1],
            color=metric_colors[1],
            linestyle=metric_line_styles[1],
            marker=markers[1],
            markersize=8,
            linewidth=2.5,
            markerfacecolor="white",
            markeredgewidth=2,
            zorder=3
        )

        # Add data labels with enhanced style
        for i, txt in enumerate(y2_data):
            ax2.annotate(
                f"{txt:.2f}",
                (normalized_x_data[i], y2_data[i]),
                xytext=(0, -13),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
                color=metric_colors[1]
            )

    # Configure secondary axis with improved style
    ax2.set_ylabel(f"{metrics[1]} (s)", fontsize=14, color=metric_colors[1])
    ax2.tick_params(axis="y", colors=metric_colors[1], labelsize=12.5)
    ax2.set_ylim(0, 19)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax2.ticklabel_format(
        style="sci", axis="y", scilimits=(0, 0), useOffset=False
    )

    # Create legend with improved style
    lines = [
        Line2D([0], [0], color=metric_colors[0], linestyle=metric_line_styles[0],
               marker=markers[0], linewidth=2, label=metrics[0],
               markerfacecolor="white", markeredgewidth=2, markersize=8),
        Line2D([0], [0], color=metric_colors[1], linestyle=metric_line_styles[1],
               marker=markers[1], linewidth=2, label=metrics[1],
               markerfacecolor="white", markeredgewidth=2, markersize=8)
    ]
    
    plt.legend(
        handles=lines,
        loc="lower right",
        # bbox_to_anchor=(0.5, 1),
        ncol=1,
        fontsize=12,
        frameon=True,
        edgecolor="black",
        columnspacing=1.0
    )

    # Final adjustments
    plt.tight_layout()
    
    # Save figure with high quality
    plt.savefig(
        "./experiment/images/model-overhead-sensitivity.pdf",
        bbox_inches="tight",
        dpi=300,
        facecolor="white",
        edgecolor="none"
    )
    plt.show()
    plt.close()


def scaling_datasize_old():
    """Plot model overhead analysis showing query response time and model runtime."""
    # Plot configuration
    benchmark = "TPC-DS"
    markers = ["P"]
    methods = ["PAC-Tree"]
    metric_colors = ["#2171b5", "black"]  # Blue for response time, Red for runtime
    metrics = ["Query Response Time (s)", "Model Runtime (s)"]
    metric_line_styles = ["-", "-"]

    # Style settings
    plt.style.use(["seaborn-v0_8-white"])
    plt.rcParams.update({"font.size": 12, "font.family": "serif"})

    # Data scale constants
    K, M, G = 1024, 1024 * 1024, 1024 * 1024 * 1024
    machine_parallelism = 22
    block_sizes = [1000, 5000, 10000, 20000, 50000]
    scale_factor = [1, 5, 10, 20, 50]
    x_data = [x * G for x in [1, 5, 10, 20, 50]]
    normalized_x_data = [int(x / G) for x in x_data]

    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(6, 4.5))
    ax1.set_prop_cycle(color=[metric_colors[0]])

    # Plot query response time
    for method_idx, method in enumerate(methods):
        y1_data = []
        for data_idx, bs in enumerate(block_sizes):
            settings.block_size = bs
            columns, data_arr = read_data(benchmark)
            data_arr = [row for row in data_arr if row[0] == "avg"]

            data_read_scale = (
                x_data[data_idx] / G
                if x_data[data_idx] / G <= 2
                else max(2, (x_data[data_idx] / G) / machine_parallelism)
            )
            estimate_query_time = (
                data_arr[0][columns.index(f"{method}_Scan Ratio")]
                * data_read_scale
                * 1024
                / 1500
            )
            y1_data.append(estimate_query_time)

        # Plot line and markers
        ax1.plot(
            normalized_x_data,
            y1_data,
            label=method,
            color=metric_colors[0],
            linestyle=metric_line_styles[0],
            marker=markers[method_idx],
            markersize=8,
            linewidth=2,
        )

        # Add data labels
        for i, txt in enumerate(y1_data):
            ax1.annotate(
                f"{txt:.2f}",
                (normalized_x_data[i], y1_data[i]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=10,
            )

    # Add parallel processing annotation
    ax1.text(
        22,
        max(y1_data),
        "Start Parallel",
        color=metric_colors[0],
        ha="center",
        fontsize=11,
        alpha=0.9,
    )
    ax1.plot([15, 15], [2.5, 0], "k--", linewidth=1, alpha=0.8, color=metric_colors[0])

    # Configure primary axis
    ax1.set_xlabel("Data Scale (Gigabyte)", fontsize=13)
    ax1.set_ylabel(metrics[0], fontsize=13, color=metric_colors[0])
    ax1.set_ylim(0, 2.5)
    ax1.tick_params(axis="y", colors=metric_colors[0])

    # Create secondary axis for model runtime
    ax2 = ax1.twinx()
    ax2.set_prop_cycle(color=[metric_colors[1]])

    # Plot model runtime
    for method_idx, method in enumerate(methods):
        y2_data = []
        for sf in scale_factor:
            settings.block_size = int(10000 / sf)
            settings.scale_factor = sf
            columns, data_arr = read_data(benchmark, sheet_idx="Time_Cost")
            y2_data.append(data_arr[:, columns.index("Opt Time")].T.tolist()[2])

        ax2.plot(
            normalized_x_data,
            y2_data,
            color=metric_colors[1],
            linestyle=metric_line_styles[1],
            marker=markers[method_idx],
            markersize=8,
            linewidth=2,
        )

        # Add data labels
        for i, txt in enumerate(y2_data):
            ax2.annotate(
                f"{txt:.2f}",
                (normalized_x_data[i], y2_data[i]),
                textcoords="offset points",
                xytext=(0, -17),
                ha="center",
                fontsize=10,
            )

    # Configure secondary axis
    ax2.set_ylabel(metrics[1], fontsize=13, color=metric_colors[1])
    ax2.tick_params(axis="y", colors=metric_colors[1])
    ax2.set_ylim(0, 18)

    # Create legend
    lines = [
        Line2D(
            [0],
            [0],
            color=metric_colors[0],
            linestyle="-",
            linewidth=2,
            label=metrics[0],
        ),
        Line2D(
            [0],
            [0],
            color=metric_colors[1],
            linestyle="-.",
            linewidth=2,
            label=metrics[1],
        ),
    ]
    plt.legend(
        handles=lines,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        fontsize=12,
        frameon=True,
    )

    # Add grid and adjust layout
    plt.grid(True, linestyle="--", alpha=0.6, color="black", axis="x")
    plt.tight_layout()

    # Save figure
    plt.savefig(
        "./experiment/images/model-overhead-sensitivity.pdf",
        bbox_inches="tight",
        dpi=300,
    )


def scaling_join_tables():

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = f"{base_dir}/result/sf={settings.scale_factor}/bs={settings.block_size}/rep={settings.replication_factor}/join_results.xlsx"

    data = pd.read_excel(data_path, sheet_name="join_results")
    columns = list(data.columns)
    data = data.values
    # 取出"JoinSet"列，作为一维数组
    join_set = [data[i][columns.index("JoinSet")] for i in range(len(data))]
    mto_scan_ratios = [
        data[i][columns.index("MTO Scan Ratio")] for i in range(len(data))
    ]
    pac_tree_scan_ratios = [
        data[i][columns.index("PAC-Tree Scan Ratio")] for i in range(len(data))
    ]
    mto_shuffle_times = [
        data[i][columns.index("MTO Shuffle Times")] for i in range(len(data))
    ]
    mto_hyper_times = [
        data[i][columns.index("MTO Hyper Times")] for i in range(len(data))
    ]
    pac_tree_shuffle_times = [
        data[i][columns.index("PAC-Tree Shuffle Times")] for i in range(len(data))
    ]
    pac_tree_hyper_times = [
        data[i][columns.index("PAC-Tree Hyper Times")] for i in range(len(data))
    ]

    # 绘制美观学术的图形，横坐标是join_set，为1-4,纵轴有两个，左轴为"Data Scan Ratio"，右轴为"Join Times"，
    # 左轴对MTO和PAC-Tree的数值进行绘制，使用柱形图；右图绘制MTO和PAC-Tree的Join Times，包括Shuffle Times 和 Hyper Times 使用不同颜色的折线图

    x = np.arange(len(join_set))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(6, 4))

    # 柱状图：Scan Ratio
    bar1 = ax1.bar(
        x - width / 2,
        mto_scan_ratios,
        width,
        label="MTO Scan Ratio",
        color="#4C72B0",
        alpha=0.85,
    )
    bar2 = ax1.bar(
        x + width / 2,
        pac_tree_scan_ratios,
        width,
        label="PAC-Tree Scan Ratio",
        color="#55A868",
        alpha=0.85,
    )
    ax1.set_xlabel("Number of Joined Tables", fontdict={"fontsize": 13})
    ax1.set_ylabel("Data Scan Ratio", fontdict={"fontsize": 13})
    ax1.set_xticks(x)
    ax1.set_xticklabels(join_set)
    ax1.tick_params(axis="y")
    ax1.legend(loc="upper left")

    # 折线图：Join Times
    ax2 = ax1.twinx()
    # line1, = ax2.plot(x, mto_shuffle_times, marker='o', color='#C44E52', label='MTO Shuffle Times', linestyle='-')
    # line2, = ax2.plot(x, mto_hyper_times, marker='s', color='#C44E52', label='MTO Hyper Times', linestyle='--')
    # line3, = ax2.plot(x, pac_tree_shuffle_times, marker='o', color='#8172B2', label='PAC-Tree Shuffle Times', linestyle='-')
    # line4, = ax2.plot(x, pac_tree_hyper_times, marker='s', color='#8172B2', label='PAC-Tree Hyper Times', linestyle='--')

    (line1,) = ax2.plot(
        x,
        mto_shuffle_times,
        marker="o",
        color="#8172B2",
        label="MTO Shuffle Times",
        linestyle="-",
    )
    (line2,) = ax2.plot(
        x,
        mto_hyper_times,
        marker="s",
        color="#8172B2",
        label="MTO Hyper Times",
        linestyle="--",
    )
    # 绿色，但和#55A868进行区分
    (line3,) = ax2.plot(
        x,
        pac_tree_shuffle_times,
        marker="o",
        color="#006d2c",
        label="PAC-Tree Shuffle Times",
        linestyle="-",
    )
    (line4,) = ax2.plot(
        x,
        pac_tree_hyper_times,
        marker="s",
        color="#006d2c",
        label="PAC-Tree Hyper Times",
        linestyle="--",
    )

    ax2.set_ylabel("Join Times", fontdict={"fontsize": 13})
    ax2.tick_params(axis="y")

    # 图例合并
    lines = [bar1, bar2, line1, line2, line3, line4]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", ncol=1, frameon=False, fontsize=11.5)

    # plt.title('Scalability: Join Table Number vs. Scan Ratio & Join Times')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"./experiment/images/scalability-join-table.pdf", bbox_inches="tight")



def scalability():
    """Plot scalability analysis results in a 2x2 grid."""
    # plt.style.use(["seaborn-v0_8-white"])
    # plt.rcParams.update({
    #     "font.family": "serif",
    #     "font.size": 12,
    #     "axes.labelsize": 13,
    #     "axes.titlesize": 14,
    # })

    methods = ["PAW", "AD-MTO", "PAC-Tree"]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    benchmark = "TPC-DS"
    markers = ["o", "s", "^"]

    x_data = [
        [0, 1, 2, 3, 4],  # Replica factors
        [1000, 5000, 10000, 20000, 50000],  # Block sizes (MB)
    ]
    uniform_x = [0.25,0.8,1.5,2.5,4]

    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(8, 6.5))
    
    for i in range(len(x_data)):
        y_data = []
        for x_elem in x_data[i]:
            if i == 0:
                settings.replication_factor = x_elem + 1
            else:
                settings.replication_factor = 2
                settings.block_size = x_elem

            columns, data_arr = read_data(benchmark)
            avg_data = [row for row in data_arr if row[0] == "avg"]
            y_elem = [avg_data[0][columns.index(f"{method}_Scan Ratio")] 
                     for method in methods]
            y_data.append(y_elem)

        y_data = np.array(y_data)
        
        # Left subplot (Scatter plot)
        ax_scatter = axes[i, 0]
        scatter_handles = []

        all_values = y_data.flatten()
        min_value = np.min(all_values)
        max_value = np.max(all_values)
        
        for method_idx, method in enumerate(methods):
            # 使用所有数据的范围来计算颜色深浅
            normalized_values = (y_data[:, method_idx] - min_value) / (max_value - min_value)
            
            for j, (x, y, intensity) in enumerate(zip(x_data[i], y_data[:, method_idx], normalized_values)):
                scatter = ax_scatter.scatter(
                    uniform_x[j], y,
                    s=80,
                    c=[colors[method_idx]],
                    marker=markers[method_idx],
                    alpha=0.4 + 0.6 * intensity,  # 透明度范围：0.4-1.0
                    label=method if j == 0 else "",
                )
                if j == 0:
                    scatter_handles.append(scatter)
        
        # Set axis limits
        ax_scatter.set_ylim(0, np.max(y_data) * 1.1)

        # Right subplot (Growth rate)
        
        growth_handles = []
        ax_trend = axes[i, 1]
        for method_idx, method in enumerate(methods):
            # 计算增长率
            method_data = y_data[:, method_idx]
            growth_rates = []
            for j in range(1, len(method_data)):
                rate = ((method_data[j] - method_data[0]))
                growth_rates.append(rate)
            
            # 只画增长率线
            line = ax_trend.plot(uniform_x[1:], growth_rates, 
                         color=colors[method_idx],
                         label=f"{method}",
                         marker='o',
                         markerfacecolor='white',
                         markeredgecolor=colors[method_idx],
                         markersize=5)[0]
            growth_handles.append(line)

        # Configure axes
        label_size = 17.3
        ax_scatter.set_xlabel("Number of Data Replicas" if i == 0 else "Block Size (MB)", fontsize=label_size)
        ax_scatter.set_ylabel("Data Scan Ratio", fontsize=label_size)
        ax_trend.set_xlabel("Number of Data Replicas" if i == 0 else "Block Size (MB)", fontsize=label_size)
        ax_trend.set_ylabel("Ratio Growth Rate", fontsize=label_size)
        
        # 科学计数法
        ax_scatter.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useOffset=False)
        ax_trend.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useOffset=False)
        
        if i == 0:
            ax_scatter.xaxis.set_major_locator(MultipleLocator(1))
            ax_trend.xaxis.set_major_locator(MultipleLocator(1))
            # 散点图图例
            ax_scatter.legend(scatter_handles, methods,
                            bbox_to_anchor=(0.5, 1.2),
                            loc='center',
                            frameon=True,
                            ncol=3,
                            fontsize=13.5,
                            columnspacing=0.3,  # 减小列间距
                            handletextpad=0.1)
            ax_trend.legend(growth_handles, methods,
                        bbox_to_anchor=(0.5, 1.2),
                        loc='center',
                        ncol=3,
                        frameon=True,
                        fontsize=13.5,
                        columnspacing=0.3,  # 减小列间距
                        handletextpad=0.1)
            
            # 设置label大小
            ax_scatter.set_xticks(x_data[i])
            ax_trend.set_xticks(x_data[i][1:])
            ax_scatter.set_xticklabels([f"{x+1}" for x in x_data[i]],
                                       fontsize=13,
                                       ha='right')
            ax_trend.set_xticklabels([f"{x+1}" for x in x_data[i][1:]],
                                     fontsize=13,
                                     ha='right')
            
        else:
            ax_scatter.set_xticks(uniform_x)
            ax_scatter.set_xticklabels([f"{int(x*256/10000)}" for x in x_data[i]],
                                    fontsize=15,
                                    ha='center')

            ax_trend.set_xticks(uniform_x[1:])
            ax_trend.set_xticklabels([f"{int(x*256/10000)}" for x in x_data[i][1:]],
                                    fontsize=15,
                                    ha='center')

        for ax in [ax_scatter, ax_trend]:
            ax.grid(True, linestyle="--", alpha=0.3)
            
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4,wspace=0.35)
    plt.savefig("./experiment/images/scalability.pdf", 
                bbox_inches="tight", 
                dpi=300)
    plt.show()


# def spark_experiments(benchmarks):
#     methods = ["PAW", "AD-MTO", "JT"]
#    fontsize=12)
#         plt.ylabel("Query Latency (s)" if th==0 else "Throughput",fontsize=12)
#         plt.xticks([_*3 for _ in range(len(benchmarks))], benchmarks)
#         plt.tight_layout()
#         # if th==0:
#         #     plt.savefig(f"./experiment/images/spark-experiments-latency.pdf",bbox_inches='tight')
#         # else:
#         #     plt.savefig(f"./experiment/images/spark-experiments-throughput.pdf",bbox_inches='tight')
#         plt.show()


def estimate_query_latency(
    access_ratio, table_scale=100, machine_parallelism=30, bandwidth=2000  # 单位：GB
):  # 单位：MB/s
    """
    access_ratio: 查询访问数据比例（0~1）
    table_scale: 数据总量（GB）
    machine_parallelism: 机器数
    bandwidth: 单机带宽（MB/s）
    """
    # 1GB = 1024MB
    if access_ratio <= 0:
        return 0.0

    total_data_GB = table_scale
    total_data_MB = total_data_GB * 1024
    data_to_read_MB = total_data_MB * access_ratio

    # 每台机器平均分配的数据量
    data_per_machine_MB = total_data_MB / machine_parallelism

    # 实际需要参与工作的机器数
    used_machines = min(
        machine_parallelism, int(np.ceil(data_to_read_MB / data_per_machine_MB))
    )
    if used_machines == 0:
        used_machines = 1

    total_bandwidth = used_machines * bandwidth  # MB/s
    query_latency = data_to_read_MB / total_bandwidth  # 秒

    return query_latency


def estimate_system_throughout_old(query_latency_list):
    if not query_latency_list:
        return 0

    return len(query_latency_list) / max(query_latency_list)


def estimate_system_throughout(query_latency_list, parallel_limit=7):
    """Calculate system throughput considering parallel execution limits.
    
    Args:
        query_latency_list: List of query latencies
        parallel_limit: Maximum number of queries that can be executed in parallel
        
    Returns:
        float: Estimated throughput (queries per second)
    """
    if not query_latency_list:
        return 0
    
    # Sort latencies in ascending order
    sorted_latencies = sorted(query_latency_list)
    total_queries = len(sorted_latencies)
    
    # Process queries in groups based on parallel_limit
    total_time = 0
    current_idx = 0
    
    while current_idx < total_queries:
        # Take next batch of queries
        batch = sorted_latencies[current_idx:min(current_idx + parallel_limit, total_queries)]
        # Add the maximum latency of current batch to total time
        total_time += max(batch)
        current_idx += parallel_limit
        
    return total_queries / total_time



def spark_experiments(benchmarks):
    """
    Plot analysis for query performance:
    Top: Throughput (bar chart)
    Bottom: Query latency (boxplot)
    """
    import matplotlib.ticker as mticker

    methods = ["PAW", "AD-MTO", "PAC-Tree"]
    methods_colors = ["#4C72B0", "#55A868", "#C44E52"]
    hatch_patterns = ["//", "xx", ".."]
    
    # Data containers
    throughput_data = {}  # {benchmark: {method: throughput_value}}
    latency_data = {}     # {benchmark: {method: list of latencies}}
    
    # Collect data
    for benchmark in benchmarks:
        columns, data_arr = read_data(benchmark)
        query_rows = [row for row in data_arr if row[0] != "avg"]
        
        throughput_data[benchmark] = {}
        latency_data[benchmark] = {}
        
        for m in methods:
            latencies = [
                estimate_query_latency(row[columns.index(f"{m}_Scan Ratio")])
                for row in query_rows
            ]
            latency_data[benchmark][m] = latencies
            throughput= estimate_system_throughout(latencies)
            throughput_data[benchmark][m] = throughput

    # Plot settings
    num_benchmarks = len(benchmarks)
    x = np.arange(num_benchmarks)
    width = 0.25

    # Create figure
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6.5, 6.5), 
        gridspec_kw={"height_ratios": [1, 1]}
    )

    # Top subplot: Throughput bars
    for m, method in enumerate(methods):
        bars = ax1.bar(
            x + (m - 1) * width,
            [throughput_data[bench][method] for bench in benchmarks],
            width,
            label=method,
            alpha=0.85,
            hatch=hatch_patterns[m],
            edgecolor=methods_colors[m],
            facecolor="none"
        )
        # Add value labels
        for rect in bars:
            height = rect.get_height()
            ax1.annotate(
                f"{height:.1f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=10,
                color=methods_colors[m]
            )

    ax1.set_ylabel("Throughput (qps)", fontsize=15, fontfamily="serif")
    #设置y上界为1.5
    ax1.set_ylim(0, max(throughput_data[benchmarks[2]].values()) * 1.1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(benchmarks, fontsize=14, fontfamily="serif")
    ax1.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax1.legend(loc="upper left", fontsize=13, ncol=3, frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.3)


    # Bottom subplot: Latency boxplots
    boxplot_positions = []
    boxplot_data = []
    for i, bench in enumerate(benchmarks):
        for j, method in enumerate(methods):
            pos = i + (j - 1) * 0.25
            boxplot_positions.append(pos)
            boxplot_data.append(latency_data[bench][method])

    bp = ax2.boxplot(
        boxplot_data,
        positions=boxplot_positions,
        widths=0.15,
        patch_artist=True,
        showfliers=False
    )

    # Style boxplots
    for i in range(len(boxplot_data)):
        method_idx = i % len(methods)
        bp["boxes"][i].set(
            facecolor="white",
            alpha=0.7,
            edgecolor=methods_colors[method_idx]
        )
        bp["medians"][i].set(color="black", linewidth=1.5)
        # Fix the whiskers and caps coloring
        bp["whiskers"][i*2].set_color(methods_colors[method_idx])
        bp["whiskers"][i*2+1].set_color(methods_colors[method_idx])
        bp["caps"][i*2].set_color(methods_colors[method_idx])
        bp["caps"][i*2+1].set_color(methods_colors[method_idx])
        
        # Add median value annotation
        median = np.median(boxplot_data[i])
        ax2.annotate(
            f'{median:.1f}',
            xy=(boxplot_positions[i], median),
            xytext=(0, 3),  # 5 points vertical offset
            textcoords='offset points',
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=9.5,
            color=methods_colors[method_idx]
        )
    

    ax2.set_ylabel("Query Latency (s)", fontsize=15, fontfamily="serif")
    ax2.set_xlabel("Benchmark", fontsize=15, fontfamily="serif")
    ax2.set_xticks(x)
    ax2.set_xticklabels(benchmarks, fontsize=14, fontfamily="serif")
    ax2.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    plt.savefig("./experiment/images/spark-experiments.pdf", bbox_inches="tight", dpi=300)
    plt.show()


benchmarks = ["TPC-H", "TPC-DS", "JOB"]

# 估计不同布局下，查询的访问数据比例
# estimate_data_access_per_query(benchmark="TPC-H")
# estimate_data_access_per_query(benchmark="JOB")
# estimate_data_access_per_query(benchmark="TPC-DS")
# data_access_ratio_experiments(benchmarks)


# 估计不同布局下，整个查询负载的访问数据块
# estimate_average_blocks(benchmarks)

# 估计不同查询负载下，随着查询执行数量的增加，PAC-Tree相对于AD-MTO的累计数据扫描比例减少量
# estimate_query_trend(benchmarks)

# 估计不同查询负载下， 不同分区树的树构建时间和布局路由时间
# plot_opt_routing_time(benchmarks)

# 设置不同参数 replica factor 和 Scale Factor
scalability()


# 估计不同参与join的表数量对查询性能的影响
# scaling_join_tables()

# 扩缩 block size
# scaling_datasize()


# 估计不同布局下，查询执行时间和整体的系统吞吐量
# spark_experiments(benchmarks)
