import sys
import argparse
import numpy as np

from data import generate_all_data
from backtest import BacktestEngine
from strategies import (
    ThresholdStrategy,
    GreedyStrategy,
    RollingOptimizationStrategy,
)
from dashboard import DashboardApp


def parse_args():
    parser = argparse.ArgumentParser(description='虚拟电厂回测分析系统')
    parser.add_argument('--port', type=int, default=8744, help='Dash 服务端口')
    parser.add_argument('--days', type=int, default=30, help='回测天数')
    parser.add_argument('--batteries', type=int, default=50, help='电池数量')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    return parser.parse_args()


def run_backtests(market_data, batteries_df, dr_events):
    strategies = [
        ThresholdStrategy(buy_threshold=0.4, sell_threshold=0.6, charge_rate=0.8),
        GreedyStrategy(min_spread=0.03),
        RollingOptimizationStrategy(look_ahead_steps=96, charge_rate=0.9),
    ]

    results = {}
    for strategy in strategies:
        print(f"  ↻ 正在运行策略: {strategy.name} ...")
        engine = BacktestEngine(
            market_data=market_data,
            batteries_df=batteries_df,
            dr_events=dr_events,
            strategy=strategy,
            dt_hours=0.25,
        )
        result = engine.run()
        results[strategy.name] = result
        print(f"    ✓ {strategy.name} 完成 - 总净收益: ¥{result.total_revenue():,.2f}")

    return results


def main():
    args = parse_args()

    print("=" * 60)
    print("  虚拟电厂回测分析系统")
    print("=" * 60)

    print(f"\n[1/4] 正在生成模拟数据...")
    print(f"    回测天数: {args.days} 天")
    print(f"    电池数量: {args.batteries} 块")

    np.random.seed(42)
    data = generate_all_data(days=args.days, num_batteries=args.batteries)
    market_data = data['market_data']
    batteries_df = data['batteries']
    dr_events = data['dr_events']

    print(f"    ✓ 数据生成完成")
    print(f"      时间点数: {len(market_data)}")
    print(f"      在线电池: {(~batteries_df['offline']).sum()} 块")
    print(f"      离线检修: {batteries_df['offline'].sum()} 块")
    print(f"      DR事件数: {len(dr_events)} 次")

    print(f"\n[2/4] 正在运行回测引擎...")
    results = run_backtests(market_data, batteries_df, dr_events)

    print(f"\n[3/4] 策略对比总结:")
    print("-" * 60)
    print(f"{'策略名称':<15} {'套利收益':>12} {'DR补偿':>12} {'调频收益':>12} {'损耗成本':>12} {'净收益':>12}")
    print("-" * 60)
    for name, result in results.items():
        print(f"{name:<15} "
              f"¥{result.total_arbitrage_revenue:>10,.2f} "
              f"¥{result.total_dr_revenue:>10,.2f} "
              f"¥{result.total_reg_revenue:>10,.2f} "
              f"-¥{result.total_degradation_cost:>9,.2f} "
              f"¥{result.total_revenue():>10,.2f}")
    print("-" * 60)

    print(f"\n[4/4] 启动 Dash 看板...")
    dashboard = DashboardApp(
        results_dict=results,
        market_data=market_data,
        batteries_df=batteries_df,
        dr_events=dr_events,
    )

    print(f"\n🎉 系统已就绪！访问 http://localhost:{args.port} 查看看板")
    print(f"\n提示: 按 Ctrl+C 停止服务\n")

    try:
        dashboard.run(port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\n\n👋 服务已停止，再见！")
        sys.exit(0)


if __name__ == '__main__':
    main()
