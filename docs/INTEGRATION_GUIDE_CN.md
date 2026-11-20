# GitHub 与 CV 对齐指南

## 项目统一原则

以根目录 `run_demo.py` 作为唯一支持的主入口，使用 `quant_research/` 中的新版因子、预测、配置、优化、回测和诊断模块。旧脚本、旧测试和旧结果放在 `legacy/` 中保留，不覆盖或混用旧数据。

当前默认 ETF 与保留的 `data/prices.csv` 列一致：SPY、QQQ、IWM、TLT、GLD、EFA、EEM、VNQ、XLE、XLK。

## 本地验证

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m examples.inspect_derivatives
python run_demo.py --source synthetic --days 500 --estimators 8 --output example_results
```

示例数据是合成价格，不能将其收益率或夏普比率当成真实市场表现。

若已核实保留的历史数据来源，可执行：

```bash
python run_demo.py --source csv --csv data/prices.csv --output results_historical
```

对于这份旧数据，目前缺少可验证的来源说明，因此结果仅供技术检查。应进一步核实是否为复权收盘价、数据源、时间戳、缺失交易日及可复现实验设置。

## 上传 GitHub

1. 在 GitHub 仓库创建新分支，例如 `unify-research-framework`，不要直接覆盖主分支。
2. 将整理后的文件内容同步到仓库（最好使用 GitHub Desktop 或 git），保留 `.git` 目录。
3. 确认 `.github/workflows/tests.yml` 已上传，且 GitHub Actions 通过。
4. 检查 README 显示的 ETF、项目入口和数据来源说明与代码一致。
5. 再将分支合并到主分支。

## CV 描述

可据代码陈述：10 只 ETF、9 个价格因子、扩展窗口 Random Forest、21 日收益预测、样本外 walk-forward、逆波动率与约束均值–方差配置、仓位上限、交易成本、SPY 基准、解析梯度及 Hessian、SLSQP 和数值导数验证。

需要注意：SLSQP 使用解析梯度；Hessian 是独立计算和检验的，并非直接作为 SciPy SLSQP 的 Hessian 输入。只有你亲自完成和理解的工作才可写成自己的个人贡献。未经核实的历史绩效或合成样本表现不得写入申请材料。
