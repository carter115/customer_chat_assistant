import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# 设置Matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def generate_chart_png(df: pd.DataFrame, output_path: str):
    """
    根据DataFrame的内容自动生成图表并保存为PNG。
    尝试识别数据类型以绘制合适的图表。
    """
    if df.empty:
        print("DataFrame为空，无法生成图表。")
        return

    # 尝试将第一列转换为日期时间类型，如果失败则保持原样
    try:
        df_copy = df.copy()
        df_copy.iloc[:, 0] = pd.to_datetime(df_copy.iloc[:, 0])
        df_copy = df_copy.sort_values(by=df_copy.columns[0])
        is_time_series = True
    except (ValueError, TypeError):
        is_time_series = False

    plt.figure(figsize=(12, 6))

    if is_time_series and len(df_copy.columns) >= 2 and pd.api.types.is_numeric_dtype(df_copy.iloc[:, 1]):
        # 如果第一列是日期，第二列是数值，则绘制折线图
        plt.plot(df_copy.iloc[:, 0], df_copy.iloc[:, 1])
        plt.xlabel(df_copy.columns[0])
        plt.ylabel(df_copy.columns[1])
        plt.title(f'{df_copy.columns[1]} 随时间变化趋势')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate() # 自动调整日期标签，防止重叠
    elif len(df.columns) >= 2 and pd.api.types.is_numeric_dtype(df.iloc[:, 1]):
        # 如果第一列是分类，第二列是数值，则绘制柱状图
        # 确保第一列是字符串类型，以便作为分类轴
        df_copy = df.copy()
        df_copy.iloc[:, 0] = df_copy.iloc[:, 0].astype(str)
        plt.bar(df_copy.iloc[:, 0], df_copy.iloc[:, 1])
        plt.xlabel(df_copy.columns[0])
        plt.ylabel(df_copy.columns[1])
        plt.title(f'{df_copy.columns[0]} vs {df_copy.columns[1]}')
        plt.xticks(rotation=45, ha='right') # 旋转X轴标签，防止重叠
    elif len(df.columns) == 1 and pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
        # 如果只有一列数值数据，绘制直方图
        plt.hist(df.iloc[:, 0], bins=10)
        plt.xlabel(df.columns[0])
        plt.ylabel('频数')
        plt.title(f'{df.columns[0]} 分布')
    else:
        # 默认情况下，尝试绘制表格或散点图，或者提示无法可视化
        print(f"无法自动为DataFrame生成图表，请检查数据类型或提供更具体的绘图指令。DataFrame列: {df.columns.tolist()}")
        plt.text(0.5, 0.5, '无法生成图表', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=20, color='red')
        plt.axis('off') # 隐藏坐标轴

    plt.grid(True)
    plt.tight_layout()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(output_path)
    plt.close() # 关闭图表，释放内存

    print(f"图表已保存到: {output_path}")