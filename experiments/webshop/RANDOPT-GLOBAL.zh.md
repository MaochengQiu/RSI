# WebShop 全模型高斯扰动探索

检验在相近 Token KL 下，全模型加性高斯扰动是否产生更好的候选分布。复用第二库 24 道开发题、NF4 原模型和同一购物流程。第三库封存题维持封存。固定随机种子 2026112107 抽取八个新方向，每个最多 24 局，总上限 192 局。无论结果如何，本次预算用完即结束范围扩大实验。

- 覆盖所有权重，包括 Attention、MLP、Embedding 和 RMSNorm。实际输入、输出权重须共享，合用一份高斯扰动。量化线性层保留原路径，叠加 FP16 高斯线性增量。非量化权重以 FP32 相加后写回 FP16，结束恢复。每个张量的随机流由候选种子分派并固定。缓存部分增量，其余按层在 GPU 上重建。
- 复用上轮的 12 道校准题、4 道复核题及每题的搜索与商品提示。在最后 16 个有效位置计算全词表 Token KL，复用上轮目标值和 ±10%／±20% 容差。初值 0.0001875，按二倍增减至夹住目标，沿用上轮幅度边界和最多八轮二分。冻结选出的幅度后再评测。
- 前向门槛要求零增量与原 logits 一致、全部目标模块被调用、相同候选的固定输入重复前向一致、校准及复核 logits 有限。用前两道校准题的四个固定提示生成动作，至少两条须为当前页面允许的动作。单个候选不达标则退出且不补抽。
- 报告平均终局分差、动作变化、升降分、下单和满分变化。按题重采样生成校正区间，单层、多层、全模型仅在这批题上作探索性对照，各用不同方向。较大正收益作为线索报告。

在原 Ubuntu 环境执行，逐局落盘，同一命令可续跑。

```bash
cd ~/rsi-data/webshop
export JAVA_HOME="$PWD/java/usr/lib/jvm/java-17-openjdk-amd64"
export PATH="$JAVA_HOME/bin:$PATH"
~/.venvs/rsi-alfworld/bin/python -B -u /mnt/d/GitHub/RSI/experiments/webshop/run_randopt_global.py \
  --checkout "$PWD" --model-dir ~/rsi-data/models/Qwen3-4B-1cfa9a7 \
  --results-root /mnt/d/RSI-outputs
```
