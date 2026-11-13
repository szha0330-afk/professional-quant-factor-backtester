# GitHub 上传与迁移说明（中文）

## 重要：这是什么版本？

这是一份**完整可运行的重构版**（包括所有核心模块、测试、文档和模拟数据示例），**不是通过 GitHub 克隆后直接修改的原仓库副本**。当前环境未能直接读取原仓库完整源码。因此它适合新建仓库展示，或与旧仓库逐文件比较后合并；**不要不核对就整包覆盖原仓库**，否则可能丢失原有数据、个人改动和 Git 历史。

## 最快运行方式

1. 下载 ZIP 并在电脑上“全部解压”。
2. 在解压后的项目文件夹地址栏输入 `powershell`，回车。
3. 运行：

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python run_demo.py --source synthetic --days 500 --estimators 8 --output example_results
```

4. 查看 `example_results/comparison.png`，以及 `strategy_metrics.csv` 和 `derivative_check.json`。

**注意：这里使用模拟随机价格，只是检验代码能运行，不是 ETF 的真实回测业绩！**

## 上传 GitHub（推荐先新建版本仓库）

推荐新建 `professional-quant-factor-backtester-enhanced` 仓库，完整上传以下目录和文件：`quant_research/`、`tests/`、`docs/`、`examples/`、`.github/workflows/`、`README.md`、`run_demo.py`、`requirements.txt`、`requirements-optional.txt`、`pyproject.toml` 和 `.gitignore`。建议先保留旧仓库，验证后再考虑合并。

如果必须覆盖旧仓库：先在 GitHub 下载旧项目 ZIP 备份；逐一比较模块签名、数据字段、回测结果；不要删除你自己的数据和历史报告。切忌把模拟报告当真实交易表现。

## 多元微积分证据在哪？

- `quant_research/optimization.py`：目标函数、解析梯度、Hessian、均值—方差优化。
- `quant_research/diagnostics.py`：数值微分校验、Euler 风险贡献。
- `tests/test_optimization.py`：解析梯度 / Hessian 与数值差分匹配、权重限制、未来价格不影响当前权重。
- `docs/MATHEMATICAL_FOUNDATIONS.md`：详细推导和 KKT 条件。
- `example_results/derivative_check.json`：实际运行的数学检验结果（非收益数据）。

**这些可以体现多元偏导和约束最优化，但不能替代学校对多重积分课程的先修要求。**

## 以后怎样改进？

用可靠来源的真实 ETF 调整收盘价 CSV 重新回测，再验证包含滑点的成交价假设；开展滚动协方差稳健性和预测收益校准研究。任何绩效数值必须以真实数据运行后为准。
