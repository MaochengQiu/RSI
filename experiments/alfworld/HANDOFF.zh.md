# ALFWorld 局部权重扰动实验交接

## 实验目的

在本地用 Qwen3-4B 跑通 ALFWorld 文本任务。验证单个权重矩阵的小幅随机扰动是否进入前向计算，是否改变 ReAct 文本或实际环境动作，以及是否改善任务完成情况。正式的 4B 与 8B 对比留给 H100，本地结果用于验证流程和观察行为。

## 环境与配置

| 项目 | 配置 |
|---|---|
| 硬件与系统 | RTX 5060 Laptop GPU，8 GB 显存，WSL 2 Ubuntu 24.04 |
| 模型 | `Qwen/Qwen3-4B`，revision `1cfa9a7208912126459214e8b04321603b3df60c` |
| 环境依赖 | ALFWorld 0.4.2，TextWorld 1.7.0，PyTorch 2.8.0+cu128，Transformers 4.57.6，Accelerate 1.10.1，bitsandbytes 0.50.2 |
| 量化 | 主体 NF4 4-bit，双重量化，FP16 计算。目标矩阵和 `lm_head` 跳过 4-bit 量化 |
| 目标矩阵 | `model.layers.18.self_attn.v_proj.weight`，FP16，1024 × 2560 |
| ReAct | 官方示例，revision `6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9`。每类任务用两个示例，将示例放置动作改写为环境接受的 `move X to Y` |
| 解码 | 贪心，单 beam，关闭 Qwen thinking。每轮最多生成 64 token，第一行非空文本、EOS 或 token 上限时停止。输入最多 4096 token，每局最多 50 个 ReAct 轮次 |
| 环境重置 | 每次从同一个 game file 重建环境，固定环境种子 20261120，关掉物品命名打乱，核对场景文件哈希和初始观察 |

三道题均来自 `json_2.1.1/valid_seen`。完整相对路径和 SHA-256 见 [基线配置](handoff-data/baseline-configuration.json)。

| 任务 | 场景目录 | Trial |
|---|---|---|
| 把书放到边桌 | `pick_and_place_simple-Book-None-SideTable-329` | `trial_T20190908_050633_745514` |
| 洗净黄油刀并放到台面 | `pick_clean_then_place_in_recep-ButterKnife-None-CounterTop-8` | `trial_T20190909_105559_983897` |
| 冷却苹果并放到台面 | `pick_cool_then_place_in_recep-Apple-None-CounterTop-14` | `trial_T20190909_044933_815840` |

## 实验方法

1. 用同一混合精度配置生成基线。每轮保存实际提示、原始输出、解析后的命令、观察、环境反馈和成败。只有提交给环境的动作计入环境步数，`think:` 计入 ReAct 轮次。
2. 保存原矩阵 \(W\)。用 CPU 固定种子生成标准高斯张量 \(Z\)，计算 \(W'=\operatorname{FP16}(W+s\operatorname{RMS}(W)Z)\)。同一方向两档幅度共用 \(Z\)。报告写回后实际的 \(\lVert W'-W\rVert_F/\lVert W\rVert_F\)。
3. 首轮用种子 20261120 和 1% 幅度。加噪前用相同 token 做两次无 KV 缓存前向，比较目标层输入、输出和末 token logits。加噪后重查，恢复原权重后再查并复跑放书题。
4. 第二轮固定三个种子 20261120、20261121、20261122，各试 1% 和 5%。校验重建的首个候选与历史权重哈希一致后复用其三条轨迹，实跑其余 15 局。每个候选结束后恢复原矩阵并校验哈希。额外重放苹果题第 8 轮决策，比较相同 token 前缀下的分数。

## 结果

基线完成 2/3，三题分别为成功 10 步、成功 7 步、失败 49 步。首轮 1% 扰动下，三题的 70 轮 ReAct 文本和 66 个环境动作逐项一致。固定诊断输入上，两次基线测量差异为零。扰动后目标层输入差异为零，层输出 RMS 差异为 0.00651617，末 token logits RMS 差异为 0.01043966。恢复后权重哈希一致，数值差异归零，放书题轨迹复现。

| 种子 | 名义幅度 | 实际权重变化 | 目标层输出 RMS 差异 | 放书 | 洗刀 | 冷却苹果 | 首次环境动作差异 |
|---|---:|---:|---:|---|---|---|---|
| 20261120 | 1% | 0.999473% | 0.006516 | 成功 10 步 | 成功 7 步 | 失败 49 步 | 无 |
| 20261120 | 5% | 4.997933% | 0.032495 | 成功 10 步 | 成功 7 步 | 失败 49 步 | 苹果题第 17 步 |
| 20261121 | 1% | 1.000057% | 0.006545 | 成功 10 步 | 成功 7 步 | 失败 49 步 | 无 |
| 20261121 | 5% | 5.000776% | 0.032661 | 成功 10 步 | 成功 7 步 | 失败 49 步 | 无 |
| 20261122 | 1% | 0.999973% | 0.006745 | 成功 10 步 | 成功 7 步 | 失败 49 步 | 无 |
| 20261122 | 5% | 5.000412% | 0.033654 | 成功 10 步 | 成功 7 步 | 失败 49 步 | 无 |

- **实际动作变化。** 20261120 的 5% 候选在苹果题第 18 轮，也就是第 17 个环境动作，将 `go to countertop 1` 改成 `go to cabinet 1`。柜子只显示肥皂，模型随后仍尝试从柜子拿苹果，环境返回 `Nothing happens.`。
- **仅文本格式变化。** 20261122 的 5% 候选在苹果题第 28 轮首次出现原始输出差异。两边解析后均执行 `go to cabinet 3`。
- **苹果题卡在搜索。** 场景标记可解，苹果在微波炉里。所有候选均未拿到或冷却苹果。
- **资源开销。** 第二轮总耗时 509.39 秒，15 局推理合计 470.75 秒，PyTorch 峰值已分配 3332.98 MiB、峰值保留 3608 MiB。

## 决策分数

下表的分数是模型为地点 token 生成的原始分数。在第 8 和第 18 轮，均在原模型和候选仍共享相同 token 前缀的位置比较。此处重放的两个决策点使用种子 20261120。

| 轮次 | 条件 | 台面词片段 ` countert` | 柜子 ` cabinet` | 冰箱 ` fridge` | 选中地点 |
|---|---|---:|---:|---:|---|
| 8 | 原模型 | 37.31250 | 22.703125 | 23.40625 | 台面 |
| 8 | 1% | 37.18750 | 22.81250 | 23.484375 | 台面 |
| 8 | 5% | 36.65625 | 23.078125 | 23.81250 | 台面 |
| 18 | 原模型 | 33.18750 | 32.78125 | 28.75000 | 台面 |
| 18 | 1% | 33.09375 | 32.81250 | 28.78125 | 台面 |
| 18 | 5% | 32.75000 | 32.96875 | 28.890625 | 柜子 |

第 8 轮六个候选都生成 `go to countertop 1`。完整逐 token 相对分数保存在 [六候选汇总](handoff-data/sweep-results.json) 的 `candidates[].decision.shared_prefix_scores`。其中包括原模型与候选各自的前两名分差，以及原 token 在候选下领先最强对手的分数。恢复权重后，该决策的 token 和重放分数均与基线相同。整段任务的逐轮轨迹保存了文本和反馈，未保存每一步的全词表 logits。

## 复跑与文件

在 WSL 中执行。环境安装和模型准备见 [README](README.md)。复跑时给输出目录使用新名称。

```bash
cd /mnt/d/GitHub/RSI/experiments/alfworld
source ~/.venvs/rsi-alfworld/bin/activate
python -B run_experiment.py compare \
  --model-dir ~/rsi-data/models/Qwen3-4B-1cfa9a7 \
  --output /mnt/d/RSI-outputs/alfworld/compare-rerun
python -B sweep_perturbations.py \
  --reference /mnt/d/RSI-outputs/alfworld/compare-rerun \
  --model-dir ~/rsi-data/models/Qwen3-4B-1cfa9a7 \
  --output /mnt/d/RSI-outputs/alfworld/sweep-rerun
```

- 随文携带 [基线结果](handoff-data/baseline-results.json)、[权重前向验证](handoff-data/weight-verification.json)、[扫描配置](handoff-data/sweep-config.json) 与 [决策重放输入](handoff-data/decision-input.json)。
- [完整轨迹压缩包](handoff-data/traces.zip) 包含去重后的 22 个 JSONL，覆盖三题基线、六个候选和一条权重恢复复跑。压缩包保留 `compare-01` 和 `sweep-01` 两级目录，便于与汇总 JSON 中的轨迹路径对照。原始运行位于 Windows 的 `D:\RSI-outputs\alfworld`。

## 结论与待验证问题

局部扰动经验证进入了前向计算，同幅度下不同方向产生了不同的行为结果。第 18 轮本来接近打平，5% 扰动足以改变排序。六个候选的任务完成数均为 2/3。本轮只覆盖一个矩阵和三个开发场景，且只有一道失败题可供改善。尚未比较其他矩阵或验证新场景上的收益。若继续判断这套方法是否值得扩大，可冻结现有候选，先在一批事先选定的新场景上对照其救回和做坏的题数。
