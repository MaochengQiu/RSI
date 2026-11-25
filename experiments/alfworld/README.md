# ALFWorld 局部权重扰动

在 WSL 2 的 Ubuntu 24.04 中运行。RSI 仓库用 `/mnt/d/GitHub/RSI` 访问，结果保存到 `/mnt/d/RSI-outputs/alfworld`。Python 环境和模型缓存放在 Ubuntu 的 Linux 文件系统内，本机 Ubuntu 虚拟磁盘位于 D 盘。

## 安装环境

```bash
sudo apt install -y git curl wget build-essential python3-dev libffi-dev python3-pip python3-venv
python3 -m venv ~/.venvs/rsi-alfworld
source ~/.venvs/rsi-alfworld/bin/activate

python -m pip install --upgrade pip
python -m pip install cmake==4.4.5 alfworld==0.4.2 textworld==1.7.0
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install transformers==4.57.6 accelerate==1.10.1 bitsandbytes==0.50.2
python -m pip check
```

安装 ALFWorld 前激活 venv，Fast Downward 的构建脚本从 `PATH` 调用 `python` 和 `cmake`。本机若遇到大 wheel 下载中断，可以从已校验的 `~/rsi-data/wheels/cu128` 目录安装 CUDA 版 PyTorch 及其运行库。

当前环境里，pip 26.3 的 `pip check` 会对 TextWorld 1.7.0 的组合 manylinux 标签给出平台提示。用下文的 `check_environment.py textworld` 实测游戏加载、重置和动作执行。

## 准备文本场景和 ReAct 示例

```bash
export ALFWORLD_DATA="$HOME/rsi-data/alfworld"
mkdir -p "$ALFWORLD_DATA" /mnt/d/RSI-outputs/alfworld

curl -fL -o "$ALFWORLD_DATA/json_2.1.3_tw-pddl.zip" \
  https://github.com/alfworld/alfworld/releases/download/0.4.2/json_2.1.3_tw-pddl.zip
python -m zipfile -t "$ALFWORLD_DATA/json_2.1.3_tw-pddl.zip"
python -m zipfile -e "$ALFWORLD_DATA/json_2.1.3_tw-pddl.zip" "$ALFWORLD_DATA"
curl -fL -o "$ALFWORLD_DATA/react-alfworld_3prompts.json" \
  https://raw.githubusercontent.com/ysymyth/ReAct/6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9/prompts/alfworld_3prompts.json
```

## 运行

```bash
cd /mnt/d/GitHub/RSI/experiments/alfworld
python -B check_environment.py textworld
python -B check_environment.py cuda

python -B run_experiment.py smoke --output /mnt/d/RSI-outputs/alfworld/smoke-01
python -B run_experiment.py compare --output /mnt/d/RSI-outputs/alfworld/compare-01
```

每次选择一个尚未创建的输出目录。Qwen3-4B 的权重与 tokenizer 使用脚本内固定的同一 revision。调试场景取自 train，正式对照按相对路径顺序从 valid_seen 的三类任务分别取首个场景。

本机已校验的模型位于 `~/rsi-data/models/Qwen3-4B-1cfa9a7`。运行命令时可加 `--model-dir "$HOME/rsi-data/models/Qwen3-4B-1cfa9a7"` 直接读取本地权重。目录内的 `source-revision.json` 记录了源 revision 和文件哈希。

`configuration.json` 记录场景、版本和生成设置，逐轮 JSONL 保存实际提示、模型输出、动作和反馈。`weight-verification.json` 保存实际权重变化及两次基线、扰动、恢复前向的差值。`results.md` 是三行对照表，`results.json` 含各回合的动作、时间和 PyTorch 峰值显存。

## 比较三个扰动方向和两档幅度

```bash
cd /mnt/d/GitHub/RSI/experiments/alfworld
~/.venvs/rsi-alfworld/bin/python -B sweep_perturbations.py \
  --reference /mnt/d/RSI-outputs/alfworld/compare-01 \
  --output /mnt/d/RSI-outputs/alfworld/sweep-01 \
  --model-dir ~/rsi-data/models/Qwen3-4B-1cfa9a7
```

复用 `compare-01` 的三道题和已有的 1% 候选轨迹。固定种子 20261120、20261121、20261122，每个方向用同一份高斯噪声分别生成 1% 和 5% 两档候选。每次从原始权重出发，记录写回 FP16 后权重的实际变化比例及目标层输出变化。复跑时把 `--output` 指向一个新目录。

汇总 `results.json` 记录每题成败、动作数、首次不同的 ReAct 文本和环境动作，以及逐轮轨迹的文件路径。`decision-input.json` 保存苹果题第一次尝试取苹果失败后的提示。各候选的 `decision-replay.json` 保存该提示下贪心生成的一行文本，以及原模型和候选仍共享相同 token 前缀时的分差。候选对原 token 的领先分数为负，表示已有另一个 token 超过原选择。

## 标准 ESOpt 的候选复验与聚合对照

已完成的三批结果见 [ALFWorld ESOpt 报告](ESOPT.zh.md)。

```bash
bash run_esopt_on_wsl.sh --output /mnt/d/RSI-outputs/alfworld/esopt-01
~/.venvs/rsi-alfworld/bin/python -B audit_esopt.py --output /mnt/d/RSI-outputs/alfworld/esopt-01
```

沿用原模型、NF4 配置、TextWorld、ReAct 提示、贪心解码和 50 轮预算。扰动和更新使用 Sudoku 实验锁定的同一官方 ES 实现，依赖旁边的 `agentic-esopt-sudoku/upstream-snapshot/` 或固定版本的官方 checkout。标准高斯扰动应用到量化模型中仍为浮点的参数，四个方向共用 `sigma=0.0007`。更新用开发集官方任务成功率做 z-score，`alpha=0.0005`。每次从精确中心复制恢复，再核对权重哈希。

固定抽取三批新场景，每批六场，放置、清洗、冷却各两场。开发与候选复验从 `valid_seen` 分别取不同布局，各含一场冷却苹果。聚合对照取 `valid_unseen`，布局与前两批互斥，同批内多个任务可能共享布局。候选跨组比较和苹果行为预测从开发轨迹确定，再复验四个重建候选。聚合阶段仅使用开发任务的二元成功率计算新中心，在第三批新场景上成对比较原中心和更新中心。

`protocol.json` 记录场景哈希和运行设置。`development.json` 保存复验前的预测和方向权重哈希，`confirmation.json` 保存复验结果。`update.json` 保存可从同一中心重放聚合的种子、奖励和权重哈希。每场的 JSONL 保留完整原始动作和观察，CSV 汇总任务成功与行为指标。以同一输出路径复跑可以核验并继续已完成的场景。CPU 审计用 TextWorld 重放所有动作并复算官方成败。

若更新的系数范数为零，审计命令可加 `--check-zero-update`。该选项需要 GPU，并将逐值比较写入 `zero-update-diagnostics.json`。
