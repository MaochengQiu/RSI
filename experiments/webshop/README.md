# WebShop 局部权重扰动

在 WebShop 官方模拟文本环境上使用 Qwen3-4B，复用 ALFWorld 的 NF4 混合精度、FP16 目标矩阵、三种噪声种子和两档幅度。使用 WebShop 固定 10,000 商品子集的人工任务池作为本地开发样本。官方 1,000 商品小库只匹配到 13 条人工指令。以 20261120 从较大子集固定抽取 20 条，预留任务 0 做环境检查。先运行基线，基线至少 10 局下单且下单分数至少有两个档位时，脚本才进入六候选阶段。

## 数据与搜索设置

- 官方代码 `princeton-nlp/WebShop@64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd` 位于 WSL 的 `~/rsi-data/webshop`。
- 三个 JSON 文件从 [固定版本的 WebShop 数据镜像](https://huggingface.co/datasets/YWZBrandon/webshop-data/tree/ce990ff/subsets/10k) 下载至 checkout 的 `data`，文件名为 `items_shuffle_10000.json`、`items_ins_v2_10000.json` 和 `items_human_ins_10000.json`。
- 沿用 [ALFWorld 推理环境](../alfworld/README.md)。在其中安装 `gym==0.24.0 flask beautifulsoup4 cleantext rank_bm25 rich thefuzz pyjnius pandas scikit-learn faiss-cpu onnxruntime nmslib sentencepiece selenium requests_mock`，再用 `pip install --no-deps pyserini==0.17.0` 安装搜索引擎，用 `python -m spacy download en_core_web_sm` 安装官方评分所需语言模型。
- 在 OpenJDK 17 环境下，用官方商品转换器和索引参数构建小库索引。

```bash
cd ~/rsi-data/webshop/search_engine
mkdir -p resources resources_100 resources_1k resources_100k
python -B - <<'PY'
import runpy
import sys
from pathlib import Path
root = Path.cwd().parent
sys.path.insert(0, str(root))
from web_agent_site import utils
from web_agent_site.engine import engine
utils.DEFAULT_FILE_PATH = str(root / "data/items_shuffle_10000.json")
engine.DEFAULT_ATTR_PATH = str(root / "data/items_ins_v2_10000.json")
runpy.run_path("convert_product_file_format.py")
PY
python -B -m pyserini.index.lucene --collection JsonCollection \
  --input resources --index indexes \
  --generator DefaultLuceneDocumentGenerator --threads 1 \
  --storePositions --storeDocvectors --storeRaw
```

## 运行

```bash
cd ~/rsi-data/webshop
source ~/.venvs/rsi-alfworld/bin/activate
export JAVA_HOME="$PWD/java/usr/lib/jvm/java-17-openjdk-amd64"
export PATH="$JAVA_HOME/bin:$PATH"
python -B /mnt/d/GitHub/RSI/experiments/webshop/run_experiment.py \
  --checkout "$PWD" --model-dir ~/rsi-data/models/Qwen3-4B-1cfa9a7 \
  --weight-reference /mnt/d/RSI-outputs/alfworld/compare-01 \
  --output /mnt/d/RSI-outputs/webshop/pilot-03 --stage sweep
```

可用 `--stage probe` 单独检查搜索，用 `--stage baseline` 跑完基线即退出。恢复运行时用同一命令和输出目录。每局独立落盘，重启会校验任务、价格、商品库、索引和脚本的 manifest，再跳过已完整保存的局。

每局最多 16 个模型决策轮次。每轮列出官方页面提供的动作。裸按钮名只有在完全匹配当前页面按钮时才会转为点击。日志区分模型原文、可执行环境动作和解析错误，记录搜索词、结果 ASIN、有序结果列表、打开的 ASIN、选项和购买。JSONL 保留实际提示及动作前后状态，局级 JSON 给出购买的 ASIN、规格、步数、官方分及其组成。每局结束后才调用官方评分函数计算已查看商品的即时购买分。

`baseline-summary.json` 含购买、部分分、满分、购买分数分布和解析错误。`candidate-summary.json` 含各候选动作变化数、升分数、降分数和平均分差。条件目录的 `paired.json` 保存首次动作差异及其前后页面、商品和规格状态。

用 `python summarize.py /mnt/d/RSI-outputs/webshop/pilot-03` 汇总首次不同的动作类型，并查看同分和分数变化的示例。

## 两个冻结候选的留出评测

固定 20261121 和 20261122 两个 5% 候选。权重张量哈希分别与 ALFWorld 扫描中保存的候选一致。沿用开发轮的购物提示、商品库、索引、价格与 16 轮上限。用固定种子 20261123 抽取 40 条新指令，排除开发任务、预留的试跑任务、同目标商品和规范化后相同的指令。任务列表和候选哈希在推理前写入 manifest。

```bash
python -B /mnt/d/GitHub/RSI/experiments/webshop/run_holdout.py \
  --checkout "$PWD" --model-dir ~/rsi-data/models/Qwen3-4B-1cfa9a7 \
  --pilot-dir /mnt/d/RSI-outputs/webshop/pilot-03 \
  --weight-reference /mnt/d/RSI-outputs/alfworld/compare-01 \
  --sweep-reference /mnt/d/RSI-outputs/alfworld/sweep-01 \
  --output /mnt/d/RSI-outputs/webshop/holdout-01
python -B /mnt/d/GitHub/RSI/experiments/webshop/analyze_holdout.py \
  /mnt/d/RSI-outputs/webshop/holdout-01
```

主要指标使用逐题配对的官方终局分，未购买记零。满分、正分和下单分别报告新增与丢失的题数。结账时机用离线即时购买分核对，包括到过满分状态却没下单、买单分低于此前已见的商品或规格，以及同一提示下先买的一方得分低于另一方后来买到的分数。成本按每局模型轮次、提交给环境的有效动作、搜索动作、输入 token 和墙钟时间统计。

## 开发集强度补充对照

在 Ubuntu 原推理环境中运行下列命令。补跑 20 条原开发指令、三个原方向、两个绝对高斯标准差 0.002 和 0.003，共 120 局。模型加载后先重放第一条基线轨迹。每个候选都用同一串 token 检查目标层和末 token logits，并在恢复原权重后复查。中断后可用相同命令续跑。

```bash
cd ~/rsi-data/webshop
source ~/.venvs/rsi-alfworld/bin/activate
export JAVA_HOME="$PWD/java/usr/lib/jvm/java-17-openjdk-amd64"
export PATH="$JAVA_HOME/bin:$PATH"
scripts=/mnt/d/GitHub/RSI/experiments/webshop
results=/mnt/d/RSI-outputs/webshop
python -B "$scripts/run_strength_sweep.py" \
  --checkout "$PWD" --model-dir ~/rsi-data/models/Qwen3-4B-1cfa9a7 \
  --pilot-dir "$results/pilot-03" --weight-reference /mnt/d/RSI-outputs/alfworld/compare-01 \
  --output "$results/strength-01"
python -B "$scripts/analyze_strength_sweep.py" --pilot-dir "$results/pilot-03" \
  --strength-dir "$results/strength-01" --output "$scripts/strength-01"
```

逐步轨迹和权重检查保存在 D 盘 `RSI-outputs/webshop/strength-01`。汇总脚本核对基线和 12 个候选的原始轨迹后，输出四档强度和各方向的汇总表 `strength-summary.json`，以及包含 240 条任务与候选配对的 `strength-pairs.csv`。每档 60 个配对来自同样的 20 个任务，每题三个方向。

## 新指令上的下单时机

沿用同一 Ubuntu 环境、商品库、价格、提示、16 轮上限和贪心解码。用开发轮保存的目标矩阵哈希冻结全部三个方向的 5% 和绝对 σ = 0.003 候选。推理前排除试跑题、开发 20 题、首轮留出 40 题，以及这些题涉及的目标商品和规范化指令。对剩余池按原 ID 顺序每个商品和规范化指令各保留一次，取全部合格指令。原模型与六个候选逐题运行。

```bash
cd ~/rsi-data/webshop
source ~/.venvs/rsi-alfworld/bin/activate
export JAVA_HOME="$PWD/java/usr/lib/jvm/java-17-openjdk-amd64"
export PATH="$JAVA_HOME/bin:$PATH"
scripts=/mnt/d/GitHub/RSI/experiments/webshop
output=/mnt/d/RSI-outputs/webshop/checkout-holdout-02
model=~/rsi-data/models/Qwen3-4B-1cfa9a7
python -B "$scripts/run_checkout_holdout.py" --checkout "$PWD" --model-dir "$model" \
  --results-root /mnt/d/RSI-outputs --output "$output"
python -B "$scripts/replay_checkout_holdout.py" --model-dir "$model" "$output"
python -B "$scripts/analyze_checkout_holdout.py" "$output" --output "$scripts/checkout-holdout-02"
```

逐局对比包括原模型对两档扰动，以及每个方向内的强弱对照。按题先平均三个方向，再对题目重采样计算平均分差的描述性区间。首次分岔只有在提示和页面状态都一致时才归入下单或其他动作。继续操作的一侧如耗尽预算，终局分记零。

另从原模型轨迹提取允许下单的商品页面，每题的每个商品和规格组合只取首次出现时的完整提示。离线记下当时立刻购买的官方分；各模型看到逐字相同的提示。原模型重新生成动作，候选在命中完全相同的在线提示和状态时复用其动作，其余提示逐条回放。此项测量以原模型访问到的状态为范围。部分分页面下单是否过早，还要与继续操作一方的最终得分合看。各条件的 `checkout-probes.json` 保存逐条回放结果。

## 固定强度筛选八个新方向

从同版本原始商品数组取第 10,001 至 20,000 条，使用随附属性与原始人工指令。单独构建官方格式的 Lucene 索引。开发集和测试集共用此库、价格、提示及评分规则。排除旧库商品和重复指令，并按目标商品和规范化指令双重去重。随机划分使用固定种子 2026112102。99 个合格目标中，前 24 个用作开发集，随后 60 个用作测试集。

对 Qwen3-4B 的 `model.layers.18.self_attn.v_proj.weight` 加绝对标准差 0.003 的高斯噪声，FP32 计算后写入 FP16。用 2026112101 抽取八个 CPU 噪声种子。根据 24 题平均官方终局分选出一人，均分保留 12 位小数后如并列，取抽取顺序靠前者。第一顺位种子同时是预设的无筛选对照。胜者落盘后开始测试，只评估原模型、胜者和预设对照。每题的动作序列、终局分、调用开销和峰值显存均保留。

```bash
cd ~/rsi-data/webshop
source ~/.venvs/rsi-alfworld/bin/activate
export JAVA_HOME="$PWD/java/usr/lib/jvm/java-17-openjdk-amd64"
export PATH="$JAVA_HOME/bin:$PATH"
scripts=/mnt/d/GitHub/RSI/experiments/webshop
results=/mnt/d/RSI-outputs
catalog=$results/webshop-data/direction-01
output=$results/webshop/direction-selection-01
# 首次准备公开数据及独立索引，使用已建好的索引时从评测命令开始。
python -B "$scripts/prepare_direction_catalog.py" "$catalog"
python -B "$scripts/index_direction_catalog.py" "$PWD" "$catalog"
python -B "$scripts/run_direction_selection.py" --checkout "$PWD" --catalog "$catalog" \
  --model-dir ~/rsi-data/models/Qwen3-4B-1cfa9a7 --results-root "$results" --output "$output"
python -B "$scripts/analyze_direction_selection.py" "$output" --output "$scripts/direction-selection-01"
python -B "$scripts/export_selected_matrix.py" "$output" "$results/alfworld/compare-01/original-target.safetensors"
```

如中断，可在原目录重跑评测命令，完整局会读取已存轨迹，半局从对应任务的初始状态重新开始。`manifest.json` 保存切分、权重哈希及选优规则，`selection.json` 保存锁定的种子。`selected-target.safetensors` 是胜者推理时使用的完整目标矩阵。分析命令从逐步轨迹重算排名，并按题配对重采样估计测试均分差的 95% 区间。
