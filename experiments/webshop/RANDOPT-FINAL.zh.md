# 第三库封存题补充评测

原预登记提交 `707798d` 的结论为证据不足，原计划未提名候选。本次按 2026-11-22 的指令单独启用第三库封存的 60 题，追加一次探索性独立评测。

只比较原模型和上轮全模型扰动在 24 道开发题上得分最高的方向 `1732518584`。候选固定使用绝对 `σ = 0.0002578125` 以及已经校准的全模型实现。开发集平均分差为 +0.0104167。沿用原 NF4 基线、相同模型修订与 Harness，16 轮上限和贪心解码。每组 60 局，共 120 局。题库、索引、价格与任务记录须和封存清单一致。

主要结果为按题配对的官方终局分差与 100,000 次题目重采样所得的 95% 百分位区间。报告原有 +0.03 实用阈值和区间下界是否大于零，同时报告满分、下单、有效动作、模型轮次与 token 成本，保留逐题轨迹。评测结束后本组实验收尾。新题结果只用于报告这个已冻结的全局候选。

首次运行原模型时，第 75 号题的商品页面超过 4096 token 并中止。修复保留原限制及任一可容纳的原提示。仅在任务和当前页单独就越界时，先保留任务与菜单，截取网页正文首尾。若菜单本身仍超出预算，则优先保留下单和导航，再按原顺序保留其他动作。保留 8 个已完成的基线结果和越界时的部分轨迹，补丁对两组适用。日志标记发生过缩短的提示。

## 从 Windows 启动

在已配置的 Ubuntu 24.04 环境执行。若中断，使用同一命令和目录续跑。

```powershell
wsl -d Ubuntu-24.04 -e bash -lc 'cd ~/rsi-data/webshop && export JAVA_HOME="$PWD/java/usr/lib/jvm/java-17-openjdk-amd64" && export PATH="$JAVA_HOME/bin:$PATH" && ~/.venvs/rsi-alfworld/bin/python -B -u /mnt/d/GitHub/RSI/experiments/webshop/run_randopt_global_final.py --checkout "$PWD" --model-dir ~/rsi-data/models/Qwen3-4B-1cfa9a7 --results-root /mnt/d/RSI-outputs'
```
