import json
import pandas as pd
from flask import Flask, render_template, jsonify,request

app = Flask(__name__)

# 数据加载 (建议：从数据库加载数据)
def load_data():
    """
    加载客户基本信息和行为资产数据。
    """
    customer_base_df = pd.read_csv('customer_base.csv')
    customer_behavior_df = pd.read_csv('customer_behavior_assets.csv')
    return customer_base_df, customer_behavior_df

# 基础信息统计
def get_basic_stats(customer_base_df, customer_behavior_df):
    """
    计算客户总数、总资产规模和平均资产。
    """
    total_customers = customer_base_df['customer_id'].nunique()
    total_assets_scale = customer_behavior_df['total_assets'].sum() / 100000000  # 转换为亿
    average_assets = customer_behavior_df['total_assets'].mean() / 10000  # 转换为万
    return {
        'total_customers': total_customers,
        'total_assets_scale': round(total_assets_scale, 2),
        'average_assets': round(average_assets, 2)
    }


# 客户资产分层饼图数据
def get_asset_level_pie_data(df):
    """
    生成客户资产分层饼图所需数据。
    """
    asset_level_counts = df['asset_level'].value_counts().reset_index()
    asset_level_counts.columns = ['name', 'value']
    return asset_level_counts.to_dict(orient='records')

# 客户生命周期漏斗图数据
def get_lifecycle_funnel_data(df):
    """
    生成客户生命周期漏斗图所需数据。
    """
    # 定义生命周期阶段的顺序
    lifecycle_order = ['新客户', '成长客户', '成熟客户', '忠诚客户', '价值客户']
    # 统计每个阶段的客户数量
    lifecycle_counts = df['lifecycle_stage'].value_counts().reindex(lifecycle_order).fillna(0)
    data = []
    for stage in lifecycle_order:
        if stage in lifecycle_counts.index:
            data.append({'name': stage, 'value': int(lifecycle_counts[stage])})
    return data

# 高潜力客户雷达图数据 (示例，需要根据实际高潜力客户定义来调整)
def get_high_potential_radar_data(customer_base_df, customer_behavior_df):
    """
    生成高潜力客户雷达图所需数据。
    这里仅为示例，实际需要根据业务定义高潜力客户。
    """
    merged_df = pd.merge(customer_base_df, customer_behavior_df, on='customer_id', how='inner')

    # 示例：定义高潜力客户的筛选条件
    # 假设高潜力客户是：月收入高，总资产高，APP登录次数多
    high_potential_df = merged_df[
        (merged_df['monthly_income'] > merged_df['monthly_income'].quantile(0.75)) &
        (merged_df['total_assets'] > merged_df['total_assets'].quantile(0.75)) &
        (merged_df['app_login_count'] > merged_df['app_login_count'].quantile(0.75))
    ]

    if high_potential_df.empty:
        return {
            'indicator': [
                {'name': '月收入', 'max': 100},
                {'name': '总资产', 'max': 100},
                {'name': 'APP登录次数', 'max': 100}
            ],
            'value': [0, 0, 0]
        }

    # 计算高潜力客户的平均值（归一化到0-100）
    avg_monthly_income = high_potential_df['monthly_income'].mean()
    avg_total_assets = high_potential_df['total_assets'].mean()
    avg_app_login_count = high_potential_df['app_login_count'].mean()

    # 获取所有客户的最大值用于归一化
    max_monthly_income = merged_df['monthly_income'].max()
    max_total_assets = merged_df['total_assets'].max()
    max_app_login_count = merged_df['app_login_count'].max()

    # 归一化处理
    norm_monthly_income = round((avg_monthly_income / max_monthly_income) * 100, 2) if max_monthly_income > 0 else 0
    norm_total_assets = round((avg_total_assets / max_total_assets) * 100, 2) if max_total_assets > 0 else 0
    norm_app_login_count = round((avg_app_login_count / max_app_login_count) * 100, 2) if max_app_login_count > 0 else 0

    indicator = [
        {'name': '月收入', 'max': 100},
        {'name': '总资产', 'max': 100},
        {'name': 'APP登录次数', 'max': 100}
    ]
    value = [norm_monthly_income, norm_total_assets, norm_app_login_count]

    return {'indicator': indicator, 'value': value}

# 营销效果热力图数据 (示例，需要根据实际营销结果来调整)
def get_marketing_heatmap_data(customer_base_df, customer_behavior_df):
    """
    生成营销效果热力图所需数据。
    这里仅为示例，实际需要根据业务定义营销成功。
    """
    merged_df = pd.merge(customer_base_df, customer_behavior_df, on='customer_id', how='inner')

    # 假设 'contact_result' 为 '成功' 表示营销成功
    marketing_success_df = merged_df[merged_df['contact_result'] == '成功']

    # 统计不同城市和职业类型的营销成功次数
    heatmap_data = marketing_success_df.groupby(['city_level', 'occupation_type']).size().reset_index(name='count')

    # 获取所有城市和职业类型，确保图表完整性
    all_cities = merged_df['city_level'].unique()
    all_occupations = merged_df['occupation_type'].unique()

    data = []
    for i, city in enumerate(all_cities):
        for j, occupation in enumerate(all_occupations):
            count = heatmap_data[(heatmap_data['city_level'] == city) & (heatmap_data['occupation_type'] == occupation)]['count'].sum()
            data.append([i, j, int(count)])

    return {
        'cities': all_cities.tolist(),
        'occupations': all_occupations.tolist(),
        'data': data
    }

# 客户活跃度趋势图数据 (示例，需要根据实际时间序列数据来调整)
def get_activity_trend_data(df):
    """
    生成客户活跃度趋势图所需数据。
    """
    # 确保 'stat_month' 是日期格式
    df['stat_month'] = pd.to_datetime(df['stat_month'])
    # 按月份聚合，计算平均APP登录次数和金融页面查看时长
    monthly_activity = df.groupby(df['stat_month'].dt.to_period('M'))[['app_login_count', 'app_financial_view_time']].mean().reset_index()
    monthly_activity['stat_month'] = monthly_activity['stat_month'].astype(str)

    return {
        'months': monthly_activity['stat_month'].tolist(),
        'app_login_count': monthly_activity['app_login_count'].tolist(),
        'app_financial_view_time': monthly_activity['app_financial_view_time'].tolist()
    }

@app.route('/')
def index():
    """
    渲染主页面。
    """
    return render_template('index.html')

@app.route('/chat/send_message', methods=['POST'])
def chat_send_message():
    """
    发送消息接口
    :return: json格式的响应数据
    """
    try:
        # 获取请求中的question参数
        data = request.get_json()
        print(f"user request: {data}")
        if not data or 'question' not in data:
            return jsonify({
                "code": 400,
                "content": "请求参数错误,缺少question字段"
            })
            
        question = data['question']
        from customer_operation_assistant import send_message
        md_content = send_message(question)
#         md_content = ''' 执行 exc_sql 结果如下表：
# |   customer_count |
# |-----------------:|
# |             9748 |
# 月收入大于10000的客户数量为9748人。'''
        print("md_content:", md_content)
        
        if md_content:
            return jsonify({
                "code": 0,
                "content": md_content
            })
        else:
            return jsonify({
                "code": 0,
                "content": "### 未查询到结果"
            })

    except Exception as e:
        return jsonify({
            "code": 500,
            "content": f"服务器错误: {str(e)}"
        })

@app.route('/data')
def data():
    """
    提供所有图表所需的数据。
    """
    customer_base_df, customer_behavior_df = load_data()

    basic_stats = get_basic_stats(customer_base_df, customer_behavior_df)
    asset_level_data = get_asset_level_pie_data(customer_behavior_df)
    lifecycle_funnel_data = get_lifecycle_funnel_data(customer_base_df)
    high_potential_radar_data = get_high_potential_radar_data(customer_base_df, customer_behavior_df)
    marketing_heatmap_data = get_marketing_heatmap_data(customer_base_df, customer_behavior_df)
    activity_trend_data = get_activity_trend_data(customer_behavior_df)

    return jsonify({
        'basic_stats': basic_stats,
        'asset_level': asset_level_data,
        'lifecycle_funnel': lifecycle_funnel_data,
        'high_potential_radar': high_potential_radar_data,
        'marketing_heatmap': marketing_heatmap_data,
        'activity_trend': activity_trend_data
    })

if __name__ == '__main__':
    app.run(debug=True)