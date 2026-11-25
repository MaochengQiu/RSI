# Agentic ESOpt Sudoku Mask15 本机复评

当前结果、复现范围和保留的产物见 [Sudoku 复现收尾](CLOSEOUT.zh.md)。

本项目对比 Qwen3.5-4B 原权重与作者发布的 Mask15 权重。在 RTX 5060 Laptop 8 GB 上，用官方 Sudoku 环境和评测器，经单个本机模型服务依次加载两份权重。两模型使用同一量化方式、题序和进程级采样种子。官方评测封装使用多端点，本项目的单端点调度会改变随机数的消耗顺序。

- 官方代码固定在 `3c6cd6992df3c353299cb9c6d33f733ede7854c8`。
- 发布权重用 `zz1358m/Qwen3.5-4B-Sudoku-Mask15-Agentic-ESOpt@1ed50da`。
- Pilot 取官方评测集筛选出 Mask15 后的前两题，各模型各跑一次。完整复评使用全部 32 题，各模型三次。
- 保留 45 轮动作预算、每轮 64 token、温度 0.7、top-p 0.8、top-k 20、presence penalty 1.5、repetition penalty 1.0，并关闭 thinking。
- `runs/` 保存命令和数据哈希、逐题评测历史、服务日志以及显存采样。模型、环境、上游副本和运行目录在本地忽略。

在 Ubuntu 24.04 中使用既有的 CUDA PyTorch 环境。实验的 Transformers 5.3.0、Flask 及 Flask-CORS 安装在 `.runtime/`。

```bash
cd /mnt/d/GitHub/RSI/experiments/agentic-esopt-sudoku
bash run_on_wsl.sh pilot --bits 8 --run-id pilot-8bit-01
bash run_on_wsl.sh full --bits 8 --run-id full-8bit-01
```

若两模型统一改用 4-bit，给新运行指定 `--bits 4` 和新的 `--run-id`。复评比较的是本机量化部署下的结果。服务器重训方案见 [Sudoku 复现建议](../../literature/2026-11-23-ESOpt/Sudoku复现建议.md)。

本机的 Ubuntu 24.04 虚拟磁盘在 `D:\WSL\Ubuntu-24.04`。启动脚本将实验的临时文件以及 Hugging Face、CUDA 和 Triton 缓存定向至项目的 `.cache/`。

## 试跑后的分批预算

两模型在同两题都填满了棋盘，均因棋盘有冲突得零分。原模型两局 91.4 秒，作者权重两局 117.1 秒。按此速度外推，32 题各评一次约需 56 分钟，三批约 2 小时 47 分钟，另计每批模型加载。正式题可能需要更多动作，两题外推只用于排机。

本机优先用 8-bit 单端点串行评测。每批同时覆盖原模型和作者权重，在两次模型加载之间释放显存。第一批使用全部 32 题建立配对结果。若只需初步方向，可在此收尾并标为单次采样；若要估计三次采样的表现，按预定种子完成另两批，无论首批成绩高低。三批独立重启进程和种子，与官方一次启动后连续采样三次的随机数序列有差异。历史四局只用于工程诊断。

```bash
bash run_on_wsl.sh paired32 --bits 8 --seed 20261124 --run-id paired32-s1
bash run_on_wsl.sh paired32 --bits 8 --seed 20261125 --run-id paired32-s2
bash run_on_wsl.sh paired32 --bits 8 --seed 20261126 --run-id paired32-s3
```

每批保存逐题结果、成功数、运行时间和整卡显存峰值。核对完成数、通信错误、显存溢出和动作轨迹。汇总三批时先按题计算三次的平均配对差，再对 32 道题汇总。需要区间时按题整体重采样。若出现量化结果与论文差距较大的情况，优先用更大显存的服务器对同一对权重做原精度复评，再决定是否花费训练算力。

三批结果、训练题重叠检查和作者日志复算见 [Mask15 复现观察](analysis-mask15/复现观察.md)。

## 一轮带留出复评的 ES 试跑

```bash
bash run_on_wsl.sh observable-es --run-id observable-es-01
```

试跑用 4-bit 作者权重作中心，更新量化模型中仍以浮点存储的参数。四个方向共用四道开发题，Mask5 和 Mask15 各两道，每题采样两次。方向分别从同一中心恢复后生成。聚合沿用官方代码的均分、z-score 和更新公式。留出题从官方评测文件各取两道 Mask5 和 Mask15，使用独立的采样种子。先基于开发集检查是否存在难度间表现交叉的候选对，再在留出题上复评中心、更新后的中心及四个重建候选。两种 Mask 表示空格数。[已完成的试跑结果](analysis-mask15/短程ES试跑.md)

运行目录保存 `manifest.json`、逐批逐动作 JSON 和可直接读取的 `scores.csv`。逐批 JSON 原子写入。用同一命令重启时，脚本核对原设置、权重哈希和轨迹后复用已完成的批次。扰动前后的浮点参数哈希用于检查中心恢复和候选重建。聚合更新另从原中心重放一次。每条轨迹重放后再写入汇总。

## 四个固定方向的跨题复评

```bash
python -B analyze_direction_transfer.py
bash run_on_wsl.sh direction-transfer --run-id direction-transfer-01
python -B analyze_direction_transfer.py --audit-run direction-transfer-01
```

第一条命令在 CPU 上复算作者日志和已有轨迹，需要 NumPy。第二条命令读取原运行 `runs/observable-es-01/` 的中心及候选哈希，在新生成棋盘上依次做校准、发现与确认。每两个棋盘写入一个可恢复的轨迹文件。确认记录绑定事先写入的 `prediction.json` 的 SHA-256。协议和结果见 [方向跨题复评](analysis-mask15/方向跨题复评.md)。
