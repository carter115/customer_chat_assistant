import json
import os
import time
from datetime import datetime
from typing import List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
from qwen_agent.tools.base import BaseTool, register_tool
from sqlalchemy import create_engine
from statsmodels.tsa.arima.model import ARIMA

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']  # 优先使用的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 配置 DashScope
dashscope_model = "qwen-turbo-latest"
dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY", "sk-xxxxx")  # 从环境变量获取 API Key，如果未设置则使用默认值

# ====== 百万客群运营助手 system prompt 和函数描述 ======
system_prompt = f"""我是百万客群运营助手，当前时间是 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}，以下是关于客户数据表相关的字段，我可能会编写对应的SQL，对数据进行查询
-- 客户基础信息表
CREATE TABLE IF NOT EXISTS customer_base (
    customer_id VARCHAR(32) PRIMARY KEY COMMENT '客户ID',
    name VARCHAR(100) COMMENT '客户姓名',
    age INT COMMENT '年龄',
    gender VARCHAR(10) COMMENT '性别',
    occupation VARCHAR(100) COMMENT '职业',
    occupation_type VARCHAR(50) COMMENT '职业类型标签（如：企业高管/互联网从业者/私营业主）',
    monthly_income DECIMAL(12,2) COMMENT '月收入',
    open_account_date VARCHAR(10) COMMENT '开户日期',
    lifecycle_stage VARCHAR(50) COMMENT '客户生命周期',
    marriage_status VARCHAR(20) COMMENT '婚姻状态',
    city_level VARCHAR(20) COMMENT '城市等级（一线/二线城市）',
    branch_name VARCHAR(100) COMMENT '开户网点'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='客户基础信息表';

-- 客户行为资产表
CREATE TABLE IF NOT EXISTS customer_behavior_assets (
    id VARCHAR(32) PRIMARY KEY COMMENT '主键ID',
    customer_id VARCHAR(32) COMMENT '客户ID（关联customer_base表）',
    
    -- 资产相关
    total_assets DECIMAL(16,2) COMMENT '总资产',
    deposit_balance DECIMAL(16,2) COMMENT '存款余额',
    financial_balance DECIMAL(16,2) COMMENT '理财余额',
    fund_balance DECIMAL(16,2) COMMENT '基金余额',
    insurance_balance DECIMAL(16,2) COMMENT '保险余额',
    asset_level VARCHAR(20) COMMENT '资产分层（50万以下、50-80万、80-100万、100万+）',
    
    -- 产品持有
    deposit_flag TINYINT COMMENT '是否持有存款（1是0否）',
    financial_flag TINYINT COMMENT '是否持有理财（1是0否）',
    fund_flag TINYINT COMMENT '是否持有基金（1是0否）',
    insurance_flag TINYINT COMMENT '是否持有保险（1是0否）',
    product_count INT COMMENT '持有产品数量',
    
    -- 交易行为
    financial_repurchase_count INT COMMENT '近1年理财复购次数',
    credit_card_monthly_expense DECIMAL(12,2) COMMENT '信用卡月均消费',
    investment_monthly_count INT COMMENT '月均投资交易次数',
    
    -- APP行为
    app_login_count INT COMMENT 'APP月均登录次数',
    app_financial_view_time INT COMMENT '理财页面月均停留时长(秒)',
    app_product_compare_count INT COMMENT '产品对比点击次数',
    last_app_login_time VARCHAR(19) NULL COMMENT '最近APP登录时间',
    
    -- 营销触达
    last_contact_time VARCHAR(19) NULL COMMENT '最近联系时间',
    contact_result VARCHAR(50) COMMENT '联系结果',
    marketing_cool_period VARCHAR(10) COMMENT '营销冷却期（30天）',
    
    stat_month VARCHAR(7) COMMENT '统计月份（YYYY-MM）',
    
    -- 外键约束
    FOREIGN KEY (customer_id) REFERENCES customer_base(customer_id),
    
    -- 创建联合唯一索引，确保每个客户每月只有一条记录
    UNIQUE KEY uk_customer_month (customer_id, stat_month)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='客户行为资产表';
我将回答用户关于客户运营相关的问题

每当 'exc_sql', 'arima_aum', 'decision_tree' 工具返回 markdown 表格或图片时，你必须原样输出工具返回的全部内容（包括图片 markdown）。
"""

functions_desc = ['exc_sql', 'arima_aum', 'decision_tree']


# ====== exc_sql 工具类实现 ======
@register_tool('exc_sql')
class ExcSQLTool(BaseTool):
    """
    SQL查询工具，执行传入的SQL语句并返回结果，并自动进行可视化。
    """
    description = '对于生成的SQL，进行SQL查询，并自动可视化'
    parameters = [
        {
            'name': 'sql_input',
            'type': 'string',
            'description': '生成的SQL语句',
            'required': True
        },
        {
            'name': 'need_visualize',
            'type': 'boolean',
            'description': '是否需要可视化和统计信息，默认为True',
            'required': False
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        """
        执行SQL查询并生成数据可视化图表。

        Args:
            params (str): 包含'sql_input'和可选'database'的JSON字符串。
            **kwargs: 其他关键字参数。

        Returns:
            str: 包含Markdown表格和图片链接的字符串，或错误信息。
        """

        args = json.loads(params)
        print(f"agent args: {args}")
        sql_input = args['sql_input']
        database = args.get('database', 'ai_operation')  # 修改数据库为 ai_operation
        try:
            engine = create_engine(
                f'mysql+mysqlconnector://stxxx:stxxx@rm-xxxxxxxx.rds.aliyuncs.com:3306/{database}?charset=utf8mb4',
                connect_args={'connect_timeout': 10}, pool_size=10, max_overflow=20
            )
            df = pd.read_sql(sql_input, engine)
            # 如果df没有数据，则返回提示
            if len(df) == 0:
                return "查询成功，但没有找到数据。"
            # 如果df只有一行数据，则不需要画图
            elif len(df) == 1:
                md = df.to_markdown(index=False)
            else:
                # 如果df的行数大于10，则md的内容使用前5行和后5行
                if len(df) > 10:
                    md = pd.concat([df.head(5), df.tail(5)]).to_markdown(index=False)
                else:
                    md = df.to_markdown(index=False)

            # 检查是否需要生成图表
            need_visualize = kwargs.get('need_visualize', True)  # 默认为True

            if need_visualize and len(df) > 1:  # 只有多于一行数据时才尝试画图
                # 自动创建目录
                save_dir = os.path.join(os.path.dirname(__file__), 'static')
                os.makedirs(save_dir, exist_ok=True)

                # 将DataFrame保存为CSV文件
                csv_filename = 'query_result.csv'
                csv_path = os.path.join(save_dir, csv_filename)
                df.to_csv(csv_path, index=False)
                print(f"查询结果已保存到 {csv_path}")

                # 生成带时间戳的文件名
                timestamp = int(time.time() * 1000)  # 毫秒级时间戳
                image_filename = f"chart_{timestamp}.png"
                image_path = os.path.join(save_dir, image_filename)

                # 调用图表生成函数
                from chart_generator import generate_chart_png  # 假设存在 chart_generator.py
                generate_chart_png(df, image_path)
                print(f"图表已保存到 {image_path}")

                # 返回Markdown格式的图片路径
                relative_image_path = os.path.join('static', image_filename).replace('\\', '/')
                result = f"{md}\n\n![Chart]({relative_image_path})"
            else:
                result = md
            return result

        except Exception as e:
            return f"SQL执行或可视化出错: {str(e)}"


# ====== ARIMA AUM 预测工具类实现 ======
@register_tool('arima_aum')
class ArimaAUMTool(BaseTool):
    """
    ARIMA AUM预测工具，用于预测客户AUM增长趋势。
    """
    description = '使用ARIMA模型预测客户AUM增长趋势，并返回预测结果和图表。'
    parameters = [
        {
            'name': 'customer_id',
            'type': 'string',
            'description': '要预测AUM增长趋势的客户ID。如果为空，则预测所有客户的总AUM。',
            'required': False
        },
        {
            'name': 'forecast_steps',
            'type': 'integer',
            'description': '预测未来多少个季度（默认为3）。',
            'required': False,
            'default': 3
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        customer_id = args.get('customer_id')
        forecast_steps = args.get('forecast_steps', 3)

        # 1. 数据加载与预处理
        df_assets = pd.read_csv('customer_behavior_assets.csv')
        df_assets['stat_month'] = pd.to_datetime(df_assets['stat_month'], format='%Y-%m')

        if customer_id:
            aum_data = df_assets[df_assets['customer_id'] == customer_id].sort_values(by='stat_month')
            if aum_data.empty:
                return f"未找到客户ID为 {customer_id} 的AUM数据。"
            aum_monthly = aum_data.set_index('stat_month')['total_assets']
            title_prefix = f"客户 {customer_id} 的"
        else:
            aum_monthly = df_assets.groupby('stat_month')['total_assets'].sum()
            title_prefix = "所有客户的"

        # 将总资产单位转换为“亿元”
        aum_monthly = aum_monthly / 100000000

        # 确保时间序列是连续的，并填充缺失值（如果需要）
        full_month_range = pd.date_range(start=aum_monthly.index.min(), end=aum_monthly.index.max(), freq='MS')
        aum_monthly = aum_monthly.reindex(full_month_range, fill_value=0)  # 填充0或使用插值

        if len(aum_monthly) < 5:  # ARIMA模型通常需要至少5个观测值
            return "数据点过少，无法进行ARIMA预测。至少需要5个历史数据点。"

        # 2. 差分处理 (根据aum_prediction_arima.py的分析，使用二阶差分)
        aum_diff = aum_monthly.diff().diff().dropna()

        # 3. 模型训练与预测
        try:
            # 重新训练模型，使用(1,2,1)阶ARIMA模型
            model = ARIMA(aum_monthly, order=(1, 2, 1))
            model_fit = model.fit()

            # 预测未来N个季度
            forecast = model_fit.get_forecast(steps=forecast_steps)
            forecast_index = pd.date_range(start=aum_monthly.index[-1] + pd.DateOffset(months=1),
                                           periods=forecast_steps, freq='MS')
            forecast_values = forecast.predicted_mean
            forecast_ci = forecast.conf_int()

            # 4. 绘制最终的预测图
            plt.figure(figsize=(14, 7))
            plt.plot(aum_monthly, label='历史AUM')
            plt.plot(forecast_index, forecast_values, label='未来AUM预测', color='green', marker='o')
            plt.fill_between(forecast_index,
                             forecast_ci.iloc[:, 0],
                             forecast_ci.iloc[:, 1], color='k', alpha=.15)
            plt.title(f'{title_prefix}AUM历史数据与未来预测')
            plt.xlabel('日期')
            plt.ylabel('总AUM（亿元）')
            plt.legend()
            plt.grid(True)

            # 保存图表
            save_dir = os.path.join(os.path.dirname(__file__), 'static')
            os.makedirs(save_dir, exist_ok=True)
            timestamp = int(time.time() * 1000)
            image_filename = f"aum_forecast_{timestamp}.png"
            image_path = os.path.join(save_dir, image_filename)
            plt.savefig(image_path)
            plt.close()  # 关闭图表，释放内存

            # 格式化预测结果
            forecast_results = []
            for i, date in enumerate(forecast_index):
                forecast_results.append({
                    '月份': date.strftime('%Y-%m'),
                    '预测值(亿元)': round(forecast_values[i], 3),
                    '95%置信区间下限(亿元)': round(forecast_ci.iloc[i, 0], 3),
                    '95%置信区间上限(亿元)': round(forecast_ci.iloc[i, 1], 3)
                })
            result_df = pd.DataFrame(forecast_results)
            md_table = result_df.to_markdown(index=False)

            relative_image_path = os.path.join('static', image_filename).replace('\\', '/')
            return f"### {title_prefix}AUM增长趋势预测结果\n\n{md_table}\n\n![AUM Forecast]({relative_image_path})"

        except Exception as e:
            return f"ARIMA模型预测出错: {str(e)}. 请检查数据是否符合时间序列预测要求。"


# ====== Decision Tree 客户资产增长预测工具类实现 ======
@register_tool('decision_tree')
class DecisionTreeTool(BaseTool):
    """
    决策树模型识别未来3个月资产容易提升至100万+的客户群体。
    """
    description = '使用决策树模型识别未来3个月资产容易提升至100万+的客户群体，并返回决策树规则和可视化图。'
    parameters = [
        {
            'name': 'max_depth',
            'type': 'integer',
            'description': '决策树的最大深度（默认为4）。',
            'required': False,
            'default': 4
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        max_depth = args.get('max_depth', 4)

        # 1. 数据加载
        customer_base_df = pd.read_csv('customer_base.csv')
        customer_behavior_df = pd.read_csv('customer_behavior_assets.csv')

        # 2. 数据预处理
        merged_df = pd.merge(customer_base_df, customer_behavior_df, on='customer_id', how='inner')

        # 模拟目标变量：未来3个月资产提升至100万+的概率
        merged_df['target'] = ((merged_df['total_assets'] + merged_df['monthly_income'] * 12) > 1000000).astype(int)

        # 选择特征
        numerical_features = ['monthly_income', 'total_assets', 'app_login_count', 'app_financial_view_time']
        categorical_features = ['gender', 'city_level', 'occupation_type', 'lifecycle_stage']

        # 填充缺失值
        for col in numerical_features:
            merged_df[col] = merged_df[col].fillna(0)
        for col in categorical_features:
            merged_df[col] = merged_df[col].fillna('Unknown')

        # 定义预处理步骤，数值特征不进行归一化
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ], remainder='passthrough')  # 'passthrough' 表示保留未处理的列

        X = merged_df[numerical_features + categorical_features]
        y = merged_df['target']

        # 3. 训练决策树模型
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(max_depth=max_depth, random_state=42))
        ])

        model_pipeline.fit(X, y)

        # 获取处理后的特征名称
        ohe_feature_names = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(
            categorical_features)
        all_feature_names = numerical_features + list(ohe_feature_names)
        class_names = ['小于等于100万', '大于100万']  # 目标变量的类别名称

        # 4. 可视化决策树
        dt_model = model_pipeline.named_steps['classifier']

        # 文本打印决策树
        from sklearn.tree import export_text
        tree_text = export_text(dt_model, feature_names=all_feature_names, class_names=class_names)

        # 生成决策树可视化图片 (需要安装 graphviz)
        import graphviz
        from sklearn.tree import export_graphviz

        dot_data = export_graphviz(dt_model, out_file=None,
                                   feature_names=all_feature_names,
                                   class_names=class_names,
                                   filled=True, rounded=True,
                                   special_characters=True)

        # 尝试设置 GRAPHVIZ_DOT 环境变量，如果 dot.exe 不在系统 PATH 中
        # 请将 'C:\Program Files\Graphviz\bin\dot.exe' 替换为您的 dot.exe 实际路径
        if "GRAPHVIZ_DOT" not in os.environ:
            os.environ["GRAPHVIZ_DOT"] = r"C:\Program Files\Graphviz\bin\dot.exe"  # 示例路径，请根据实际情况修改

        dot_data = dot_data.replace("helvetica", "SimHei")  # 写死
        graph = graphviz.Source(dot_data, encoding='utf-8')

        # 保存图表
        save_dir = os.path.join(os.path.dirname(__file__), 'static')
        os.makedirs(save_dir, exist_ok=True)
        timestamp = int(time.time() * 1000)
        image_filename = f"decision_tree_{timestamp}.png"
        image_path = os.path.join(save_dir, image_filename)
        graph.render(image_path.replace('.png', ''), format='png', cleanup=True)  # graphviz render 会自动添加 .png

        relative_image_path = os.path.join('static', image_filename).replace('\\', '/')

        return f"### 客户资产增长预测决策树分析\n\n**决策树规则:**\n```\n{tree_text}\n```\n\n![Decision Tree]({relative_image_path})"

    """
    description = '使用ARIMA模型预测客户AUM增长趋势，并返回预测结果和图表。'
    parameters = [
        {
            'name': 'customer_id',
            'type': 'string',
            'description': '要预测AUM增长趋势的客户ID。如果为空，则预测所有客户的总AUM。',
            'required': False
        },
        {
            'name': 'forecast_steps',
            'type': 'integer',
            'description': '预测未来多少个季度（默认为3）。',
            'required': False,
            'default': 3
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        customer_id = args.get('customer_id')
        forecast_steps = args.get('forecast_steps', 3)

        # 1. 数据加载与预处理
        df_assets = pd.read_csv('customer_behavior_assets.csv')
        df_assets['stat_month'] = pd.to_datetime(df_assets['stat_month'], format='%Y-%m')

        if customer_id:
            aum_data = df_assets[df_assets['customer_id'] == customer_id].sort_values(by='stat_month')
            if aum_data.empty:
                return f"未找到客户ID为 {customer_id} 的AUM数据。"
            aum_monthly = aum_data.set_index('stat_month')['total_assets']
            title_prefix = f"客户 {customer_id} 的"
        else:
            aum_monthly = df_assets.groupby('stat_month')['total_assets'].sum()
            title_prefix = "所有客户的"

        # 将总资产单位转换为“亿元”
        aum_monthly = aum_monthly / 100000000

        # 确保时间序列是连续的，并填充缺失值（如果需要）
        full_month_range = pd.date_range(start=aum_monthly.index.min(), end=aum_monthly.index.max(), freq='MS')
        aum_monthly = aum_monthly.reindex(full_month_range, fill_value=0) # 填充0或使用插值

        if len(aum_monthly) < 5: # ARIMA模型通常需要至少5个观测值
            return "数据点过少，无法进行ARIMA预测。至少需要5个历史数据点。"

        # 2. 差分处理 (根据aum_prediction_arima.py的分析，使用二阶差分)
        aum_diff = aum_monthly.diff().diff().dropna()

        # 3. 模型训练与预测
        try:
            # 重新训练模型，使用(1,2,1)阶ARIMA模型
            model = ARIMA(aum_monthly, order=(1, 2, 1))
            model_fit = model.fit()

            # 预测未来N个季度
            forecast = model_fit.get_forecast(steps=forecast_steps)
            forecast_index = pd.date_range(start=aum_monthly.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')
            forecast_values = forecast.predicted_mean
            forecast_ci = forecast.conf_int()

            # 4. 绘制最终的预测图
            plt.figure(figsize=(14, 7))
            plt.plot(aum_monthly, label='历史AUM')
            plt.plot(forecast_index, forecast_values, label='未来AUM预测', color='green', marker='o')
            plt.fill_between(forecast_index,
                             forecast_ci.iloc[:, 0],
                             forecast_ci.iloc[:, 1], color='k', alpha=.15)
            plt.title(f'{title_prefix}AUM历史数据与未来预测')
            plt.xlabel('日期')
            plt.ylabel('总AUM（亿元）')
            plt.legend()
            plt.grid(True)

            # 保存图表
            save_dir = os.path.join(os.path.dirname(__file__), 'static')
            os.makedirs(save_dir, exist_ok=True)
            timestamp = int(time.time() * 1000)
            image_filename = f"aum_forecast_{timestamp}.png"
            image_path = os.path.join(save_dir, image_filename)
            plt.savefig(image_path)
            plt.close() # 关闭图表，释放内存

            # 格式化预测结果
            forecast_results = []
            for i, date in enumerate(forecast_index):
                forecast_results.append({
                    '月份': date.strftime('%Y-%m'),
                    '预测值(亿元)': round(forecast_values[i], 3),
                    '95%置信区间下限(亿元)': round(forecast_ci.iloc[i, 0], 3),
                    '95%置信区间上限(亿元)': round(forecast_ci.iloc[i, 1], 3)
                })
            result_df = pd.DataFrame(forecast_results)
            md_table = result_df.to_markdown(index=False)

            relative_image_path = os.path.join('static', image_filename).replace('\\', '/')
            return f"### {title_prefix}AUM增长趋势预测结果\n\n{md_table}\n\n![AUM Forecast]({relative_image_path})"

        except Exception as e:
            return f"ARIMA模型预测出错: {str(e)}. 请检查数据是否符合时间序列预测要求。"    """
    description = 'ARIMA AUM预测工具，用于预测客户AUM增长趋势'
    parameters = [
        {
            'name': 'predict_months',
            'type': 'integer',
            'description': '预测月数，例如：3',
            'required': True
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        """
        执行 ARIMA AUM 预测并生成预测图表。

        Args:
            params (str): 包含'predict_months'的JSON字符串。
            **kwargs: 其他关键字参数。

        Returns:
            str: 包含Markdown表格和图片链接的字符串，或错误信息。
        """
        args = json.loads(params)
        predict_months = args['predict_months']

        try:
            # 1. 数据加载与预处理
            # 从CSV文件加载数据，这里需要从数据库获取数据
            engine = create_engine(
                f'mysql+mysqlconnector://stxxxx:stxxxx@rm-xxxxxx.rds.aliyuncs.com:3306/ai_operation?charset=utf8mb4',
                connect_args={'connect_timeout': 10}, pool_size=10, max_overflow=20
            )
            query = "SELECT stat_month, total_assets FROM customer_behavior_assets ORDER BY stat_month ASC"
            df_assets = pd.read_sql(query, engine)

            if df_assets.empty:
                return "未找到客户行为资产数据。"

            df_assets['stat_month'] = pd.to_datetime(df_assets['stat_month'], format='%Y-%m')
            aum_monthly = df_assets.groupby('stat_month')['total_assets'].sum()
            aum_monthly = aum_monthly / 100000000  # 转换为亿元

            # 确保数据是连续的，处理缺失月份
            full_month_range = pd.date_range(start=aum_monthly.index.min(), end=aum_monthly.index.max(), freq='MS')
            aum_monthly = aum_monthly.reindex(full_month_range)
            aum_monthly = aum_monthly.interpolate(method='time')  # 使用时间插值填充缺失值

            from statsmodels.tsa.arima.model import ARIMA
            # 训练 ARIMA 模型 (使用固定阶数 (1,2,1) 作为示例，实际应用中可能需要更复杂的阶数选择逻辑)
            model = ARIMA(aum_monthly, order=(1, 2, 1))
            model_fit = model.fit()

            # 预测未来走势
            forecast_result = model_fit.get_forecast(steps=predict_months)
            forecast = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int(alpha=0.05)  # 95% 置信区间

            last_month = aum_monthly.index[-1]
            forecast_index = pd.date_range(start=last_month + pd.DateOffset(months=1), periods=predict_months,
                                           freq='MS')

            forecast_df = pd.DataFrame({
                'stat_month': forecast_index,
                'predicted_aum': forecast,
                'lower_bound': conf_int.iloc[:, 0],
                'upper_bound': conf_int.iloc[:, 1]
            })
            forecast_df['stat_month'] = forecast_df['stat_month'].dt.strftime('%Y-%m')

            # 自动创建目录
            save_dir = os.path.join(os.path.dirname(__file__), 'static')
            os.makedirs(save_dir, exist_ok=True)

            # 生成预测图表
            timestamp = int(time.time() * 1000)  # 毫秒级时间戳
            image_filename = f"chart_aum_forecast_{timestamp}.png"
            image_path = os.path.join(save_dir, image_filename)

            plt.figure(figsize=(12, 6))
            plt.plot(aum_monthly.index, aum_monthly, label='历史AUM')
            plt.plot(forecast_index, forecast, label='预测AUM', color='red')
            plt.fill_between(forecast_index,
                             conf_int.iloc[:, 0],
                             conf_int.iloc[:, 1], color='pink', alpha=0.3, label='95% 置信区间')

            plt.title(f'未来 {predict_months} 个月AUM预测')
            plt.xlabel('日期')
            plt.ylabel('总AUM（亿元）')
            plt.legend()
            plt.grid(True)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.gcf().autofmt_xdate()  # 自动调整日期标签，防止重叠
            plt.tight_layout()
            plt.savefig(image_path)
            plt.close()
            print(f"ARIMA AUM 预测图表已保存到 {image_path}")

            # 返回Markdown格式的图片路径和预测数据表格
            relative_image_path = os.path.join('static', image_filename).replace('\\', '/')
            md_table = forecast_df.to_markdown(index=False)
            result = f"### 未来 {predict_months} 个月AUM预测结果：\n\n{md_table}\n\n![ARIMA AUM Forecast]({relative_image_path})"
            return result

        except Exception as e:
            return f"ARIMA AUM 预测出错: {str(e)}"


def init_agent_service():
    """初始化百万客群运营助手服务"""
    llm_cfg = {
        'model': dashscope_model,
        'model_type': 'qwen_dashscope',
        'api_key': dashscope_api_key,
        'timeout': 30,
        'retry_count': 3,
    }
    try:
        bot = Assistant(
            llm=llm_cfg,
            name='百万客群运营助手',
            description='客户数据查询与分析',
            system_message=system_prompt,
            function_list=functions_desc,
            files=[]  # 暂时没有faq文件
        )
        print("助手初始化成功！")
        return bot
    except Exception as e:
        print(f"助手初始化失败: {str(e)}")
        raise


# 初始化助手
bot = init_agent_service()


def bot_response(messages: List[dict]):
    from qwen_agent.llm.schema import ASSISTANT, FUNCTION

    full_text = ''
    content = []
    for msg in messages:
        if msg['role'] == ASSISTANT:
            if msg.get('content'):
                assert isinstance(msg['content'], str), 'Now only supports text messages'
                content.append(msg['content'])
            if msg.get('function_call'):
                content.append(f'{msg["function_call"]["name"]}\n{msg["function_call"]["arguments"]}')
        else:
            raise TypeError
    if content:
        full_text = '\n'.join(content)
    return full_text


# 对话历史
def send_message(query, file=None):
    try:
        # 输入验证
        if not query:
            print('user question cannot be empty！')
            return ''

        print("正在处理您的请求...")
        messages = []

        messages.append({'role': 'user', 'content': query})
        # 运行助手并处理响应
        response = []
        for response in bot.run(messages):
            pass
        return bot_response(response)

    except Exception as e:
        print(f"处理请求时出错: {str(e)}")
        print("请重试或输入新的问题")



if __name__ == '__main__':
    # app_tui()
    # app_gui()
    send_message("查询月收入大于10000的客户数量")
