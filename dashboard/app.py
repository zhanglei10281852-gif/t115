import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


class DashboardApp:
    def __init__(self, results_dict: dict, market_data: pd.DataFrame,
                 batteries_df: pd.DataFrame, dr_events: pd.DataFrame):
        self.results_dict = results_dict
        self.market_data = market_data
        self.batteries_df = batteries_df
        self.dr_events = dr_events
        self.strategy_names = list(results_dict.keys())

        self.app = dash.Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()

    def _setup_layout(self):
        self.app.layout = html.Div([
            html.Div([
                html.H1("虚拟电厂回测分析看板",
                        style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
                html.P("分布式储能聚合调度策略回测平台",
                       style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '20px'}),
            ], style={'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '10px',
                      'marginBottom': '20px'}),

            html.Div([
                html.Label("选择调度策略：", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                dcc.Dropdown(
                    id='strategy-selector',
                    options=[{'label': name, 'value': name} for name in self.strategy_names],
                    value=self.strategy_names[0] if self.strategy_names else None,
                    multi=False,
                    style={'width': '400px', 'display': 'inline-block'}
                ),
            ], style={'marginBottom': '20px', 'padding': '15px', 'backgroundColor': '#f8f9fa',
                      'borderRadius': '8px'}),

            html.Div(id='kpi-cards', style={'display': 'flex', 'flexWrap': 'wrap',
                                            'gap': '15px', 'marginBottom': '20px'}),

            html.Div([
                html.Div([
                    html.H3("收益拆解", style={'textAlign': 'center', 'color': '#2c3e50'}),
                    dcc.Graph(id='revenue-breakdown-chart'),
                ], style={'flex': '1', 'minWidth': '400px', 'backgroundColor': 'white',
                          'padding': '15px', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),

                html.Div([
                    html.H3("电池收益贡献排名", style={'textAlign': 'center', 'color': '#2c3e50'}),
                    dcc.Graph(id='battery-ranking-chart'),
                ], style={'flex': '1', 'minWidth': '400px', 'backgroundColor': 'white',
                          'padding': '15px', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '15px', 'marginBottom': '20px'}),

            html.Div([
                html.H3("电池群荷电状态热力图", style={'textAlign': 'center', 'color': '#2c3e50'}),
                dcc.Graph(id='soc-heatmap'),
            ], style={'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '8px',
                      'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'marginBottom': '20px'}),

            html.Div([
                html.H3("削峰填谷效果 - 区域净负荷对比", style={'textAlign': 'center', 'color': '#2c3e50'}),
                dcc.Graph(id='net-load-chart'),
            ], style={'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '8px',
                      'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'marginBottom': '20px'}),

            html.Div([
                html.H3("电池充放电功率曲线", style={'textAlign': 'center', 'color': '#2c3e50'}),
                dcc.Graph(id='power-chart'),
            ], style={'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '8px',
                      'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'marginBottom': '20px'}),

            html.Div([
                html.H3("警告与异常信息", style={'textAlign': 'center', 'color': '#e74c3c'}),
                html.Div(id='warnings-panel',
                         style={'maxHeight': '200px', 'overflowY': 'scroll',
                                'backgroundColor': '#fdf2f2', 'padding': '10px',
                                'borderRadius': '5px', 'border': '1px solid #f5b7b1'}),
            ], style={'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '8px',
                      'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),

            html.Div([
                html.P(f"数据周期：{self.market_data.index[0].strftime('%Y-%m-%d')} 至 "
                       f"{self.market_data.index[-1].strftime('%Y-%m-%d')} | "
                       f"电池数量：{len(self.batteries_df)} 块 | "
                       f"时间粒度：15分钟",
                       style={'textAlign': 'center', 'color': '#95a5a6', 'fontSize': '12px',
                              'marginTop': '20px'}),
            ]),
        ], style={'padding': '20px', 'backgroundColor': '#f5f6fa'})

    def _setup_callbacks(self):
        @self.app.callback(
            Output('kpi-cards', 'children'),
            Output('revenue-breakdown-chart', 'figure'),
            Output('battery-ranking-chart', 'figure'),
            Output('soc-heatmap', 'figure'),
            Output('net-load-chart', 'figure'),
            Output('power-chart', 'figure'),
            Output('warnings-panel', 'children'),
            Input('strategy-selector', 'value'),
        )
        def update_all_charts(selected_strategy):
            result = self.results_dict[selected_strategy]

            kpi_cards = self._create_kpi_cards(result)
            revenue_fig = self._create_revenue_breakdown_chart(result)
            ranking_fig = self._create_battery_ranking_chart(result)
            heatmap_fig = self._create_soc_heatmap(result)
            load_fig = self._create_net_load_chart(result)
            power_fig = self._create_power_chart(result)
            warnings_html = self._create_warnings_panel(result)

            return kpi_cards, revenue_fig, ranking_fig, heatmap_fig, load_fig, power_fig, warnings_html

    def _create_kpi_cards(self, result):
        cards_data = [
            {'title': '总净收益', 'value': f"¥{result.total_revenue():,.2f}",
             'color': '#27ae60', 'icon': '💰'},
            {'title': '套利收益', 'value': f"¥{result.total_arbitrage_revenue:,.2f}",
             'color': '#3498db', 'icon': '⚡'},
            {'title': '需求响应补偿', 'value': f"¥{result.total_dr_revenue:,.2f}",
             'color': '#9b59b6', 'icon': '📋'},
            {'title': '调频收益', 'value': f"¥{result.total_reg_revenue:,.2f}",
             'color': '#f39c12', 'icon': '📈'},
            {'title': '损耗成本', 'value': f"-¥{result.total_degradation_cost:,.2f}",
             'color': '#e74c3c', 'icon': '🔋'},
            {'title': '参与DR事件', 'value': f"{len(result.dr_participations)} 次",
             'color': '#1abc9c', 'icon': '🎯'},
        ]

        cards = []
        for card in cards_data:
            cards.append(
                html.Div([
                    html.Div(card['icon'],
                             style={'fontSize': '24px', 'marginBottom': '5px'}),
                    html.Div(card['title'],
                             style={'fontSize': '12px', 'color': '#7f8c8d',
                                    'marginBottom': '5px'}),
                    html.Div(card['value'],
                             style={'fontSize': '20px', 'fontWeight': 'bold',
                                    'color': card['color']}),
                ], style={'flex': '1', 'minWidth': '150px', 'padding': '15px',
                          'backgroundColor': 'white', 'borderRadius': '8px',
                          'textAlign': 'center', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                          'borderTop': f'3px solid {card["color"]}'})
            )
        return cards

    def _create_revenue_breakdown_chart(self, result):
        categories = ['套利收益', '需求响应补偿', '调频收益', '损耗成本']
        values = [
            result.total_arbitrage_revenue,
            result.total_dr_revenue,
            result.total_reg_revenue,
            -result.total_degradation_cost,
        ]
        colors = ['#3498db', '#9b59b6', '#f39c12', '#e74c3c']

        fig = go.Figure(data=[go.Pie(
            labels=categories,
            values=values,
            hole=0.4,
            marker=dict(colors=colors),
            textinfo='label+percent',
            hovertemplate='%{label}: ¥%{value:,.2f}<extra></extra>',
        )])

        fig.update_layout(
            showlegend=True,
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        return fig

    def _create_battery_ranking_chart(self, result):
        ranking_df = result.get_battery_revenue_ranking()
        ranking_df = ranking_df.head(20)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=ranking_df['battery_id'],
            x=ranking_df['arbitrage_revenue'],
            name='套利收益',
            orientation='h',
            marker_color='#3498db',
        ))

        fig.add_trace(go.Bar(
            y=ranking_df['battery_id'],
            x=ranking_df['dr_revenue'],
            name='需求响应',
            orientation='h',
            marker_color='#9b59b6',
        ))

        fig.add_trace(go.Bar(
            y=ranking_df['battery_id'],
            x=ranking_df['reg_revenue'],
            name='调频收益',
            orientation='h',
            marker_color='#f39c12',
        ))

        fig.add_trace(go.Bar(
            y=ranking_df['battery_id'],
            x=-ranking_df['degradation_cost'],
            name='损耗成本',
            orientation='h',
            marker_color='#e74c3c',
        ))

        fig.update_layout(
            barmode='stack',
            yaxis=dict(autorange='reversed'),
            height=500,
            margin=dict(l=80, r=20, t=20, b=20),
            xaxis_title='收益 (元)',
            legend=dict(orientation='h', y=-0.15),
            hovermode='y unified',
        )
        return fig

    def _create_soc_heatmap(self, result):
        soc_df = result.get_soc_heatmap_data()

        fig = go.Figure(data=go.Heatmap(
            z=soc_df.values.T,
            x=soc_df.index,
            y=soc_df.columns,
            colorscale='RdYlGn',
            zmin=0,
            zmax=1,
            colorbar=dict(title='SOC', tickformat='.0%'),
            hovertemplate='时间: %{x}<br>电池: %{y}<br>SOC: %{z:.1%}<extra></extra>',
        ))

        fig.update_layout(
            height=500,
            margin=dict(l=80, r=20, t=20, b=40),
            xaxis_title='时间',
            yaxis_title='电池编号',
        )
        return fig

    def _create_net_load_chart(self, result):
        time_index = result.time_index

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=time_index,
            y=result.net_load_without_bess,
            mode='lines',
            name='无储能净负荷',
            line=dict(color='#e74c3c', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(231, 76, 60, 0.1)',
        ))

        fig.add_trace(go.Scatter(
            x=time_index,
            y=result.net_load_with_bess,
            mode='lines',
            name='有储能净负荷',
            line=dict(color='#27ae60', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(39, 174, 96, 0.1)',
        ))

        fig.update_layout(
            height=400,
            margin=dict(l=60, r=20, t=20, b=40),
            xaxis_title='时间',
            yaxis_title='净负荷 (kW)',
            legend=dict(orientation='h', y=-0.15),
            hovermode='x unified',
        )
        return fig

    def _create_power_chart(self, result):
        time_index = result.time_index

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=time_index,
            y=result.total_power_history,
            mode='lines',
            name='总功率',
            line=dict(color='#3498db', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(52, 152, 219, 0.2)',
        ))

        fig.add_hline(y=0, line_dash="dash", line_color="gray")

        fig.update_layout(
            height=350,
            margin=dict(l=60, r=20, t=20, b=40),
            xaxis_title='时间',
            yaxis_title='功率 (kW) <br><span style="font-size:10px">(正=充电, 负=放电)</span>',
            hovermode='x unified',
        )
        return fig

    def _create_warnings_panel(self, result):
        if not result.warnings:
            return html.P("暂无异常警告", style={'color': '#27ae60', 'textAlign': 'center'})

        warning_items = []
        for w in result.warnings[:50]:
            warning_items.append(
                html.Div(f"⚠️ {w}",
                         style={'padding': '5px', 'borderBottom': '1px solid #f5b7b1',
                                'fontSize': '13px', 'color': '#c0392b'})
            )

        if len(result.warnings) > 50:
            warning_items.append(
                html.Div(f"... 还有 {len(result.warnings) - 50} 条警告未显示",
                         style={'padding': '5px', 'fontSize': '12px',
                                'color': '#7f8c8d', 'textAlign': 'center'})
            )

        return warning_items

    def run(self, port=8744, debug=False):
        print(f"🚀 虚拟电厂回测看板已启动，请访问 http://localhost:{port}")
        self.app.run_server(port=port, debug=debug)
