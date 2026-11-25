# WebShop 随机权重扰动复跑

使用 D 盘上的原始结果和原 Ubuntu 推理环境。在 Windows PowerShell 中从仓库根目录执行。第三库只需准备一次。

```powershell
& 'C:/Users/lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B experiments/webshop/prepare_direction_catalog.py --offset 20000 D:/RSI-outputs/webshop-data/direction-02
wsl -d Ubuntu-24.04 -e bash /mnt/d/GitHub/RSI/experiments/webshop/run_randopt_followup.sh seal
```

将登记和题目清单一同提交到 Git 后，从 Windows 运行下列命令。

```powershell
wsl -d Ubuntu-24.04 -e bash /mnt/d/GitHub/RSI/experiments/webshop/run_randopt_followup.sh all
```

出现中断可重跑相同命令，已有完整局通过日志核对后直接复用。也可把 `all` 改为 `stage0`、`replay`、`complete`、`scope` 或 `final`。原始轨迹、模型与运行环境记录及分析均位于 `D:/RSI-outputs/webshop/randopt-followup-01`。复现失败时先看 `replay.json` 中的首个分歧。校准轨迹和幅度位于 `scope` 下。独立验证由 `nomination.json` 中冻结的单一提名启动。
