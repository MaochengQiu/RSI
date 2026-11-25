# RSI 与 Agent Rollout 训练调研

2026-11-19。围绕 11 月 18 日交流，核对了直接相关的 RSI 工作、用多步交互轨迹更新 Task Agent 的工作，以及 RandOpt 原文。需要从论文里找明白三件事：谁执行任务，轨迹送到哪里，哪些题真正参与了更新或候选筛选。

## 论文对照

| 工作 | 用什么数据收集 Rollout 或做搜索 | 用什么 Benchmark 报告结果 | 用什么模型、轨迹怎么用 |
| --- | --- | --- | --- |
| RSIFlow（2026 预印本） | 代码用 TACO 的 DeepCoder 子集，工具用 EnvScaler，检索问答用 HotpotQA 与 2WikiMultiHopQA。三域各 1,000 个训练任务，分五轮。 | HumanEval+ / MBPP+；BFCL-v3 / ACEBench；HotpotQA / 2WikiMultiHopQA 的独立报告样本。代码测测试通过，工具用原生评分，检索问答主报答案 F1，另留 EM。 | Task-Agent 用 Qwen3-4B 或 8B，Meta-Agent 用 GPT-5.6 Sol。SFT 复用成功任务中符合条件的 assistant 调用。GRPO 按题采样完整轨迹，用环境的二元任务成败做奖励。Meta-Agent 也用执行反馈判断该更新权重、Harness 还是当前产物。见 §3、§4.1、附录 A.1–A.3、B.3。 [原文](https://arxiv.org/html/2610.21791v1) |
| HarnessForge（2026 预印本） | 3,800 题训练与演化池。EnvScaler-RL 2,000，ToolHop 800，NQ / HotpotQA / 2Wiki 的离线检索任务共 1,000。 | ToolHop、RestBench-TMDB、API-Bank，以及 HotpotQA / 2Wiki 的 SearchQA，均用留出样本。部分评测还比较调用路径或参数。 | Task Agent 为 Qwen3-4B / 8B，Meta-Agent 为 GPT-5.5。新 Harness 执行时收集轨迹，筛成功轨迹做 SFT，只更新该 Harness 对应的 LoRA；另比较 GRPO、RLOO。见 §4.1、§4.4 及附录 B、C、E.3。 [原文](https://arxiv.org/html/2606.01779) |
| Search-R1（COLM 2025） | Natural Questions 和 HotpotQA 的训练集。每题可多次搜索和读返回的证据，再提交答案。 | 同域 NQ、HotpotQA；跨数据集 TriviaQA、PopQA、2WikiMultiHopQA、MuSiQue、Bamboogle。主指标 EM。 | 主表 Qwen2.5-3B / 7B，均试 Base 和 Instruct；作者也提供 Llama-3.2-3B 系列 checkpoint。默认 PPO，另比 GRPO。奖励只看最终答案 EM，返回的检索文本作为上下文，训练损失屏蔽这些 token。见 §3、§4 与附录。 [原文](https://arxiv.org/html/2503.09516v5) · [代码](https://github.com/PeterGriffinJin/Search-R1) · [权重及数据](https://huggingface.co/collections/PeterJinGo/search-r1) |
| Agent-R（2025，arXiv 版本） | 在 WebShop、ScienceWorld 和 TextCraft 的交互环境内收集轨迹。用 MCTS 展开动作分支，定位出错步，把错误前缀接到共享父节点的正确分支，构造修正轨迹。 | 同三个环境的测试任务，论文分别取 200、200、100 题。另将失败轨迹截在中间一步，比较模型接手后纠错的效果。 | 主基座 Llama-3.1-8B-Instruct，迭代三轮 SFT。能看到改动作是否挽回失败、是否减少重复动作。见 §3、§4.1、§4.4、附录案例。 [原文](https://arxiv.org/html/2501.11425) · [代码](https://github.com/ByteDance-Seed/Agent-R) |
| ETO（ACL 2024） | 在 WebShop、ScienceWorld 和 ALFWorld 训练任务中探索并采集失败轨迹，与成功或更高回报的轨迹配对。 | 三个环境的留出任务，ScienceWorld 和 ALFWorld 还测未见过的任务变体。WebShop、ScienceWorld 有 0–1 终局分数，ALFWorld 为二元终局成败。 | 主基座 Llama-2-7B-Chat，另试 Llama-2-13B-Chat、Mistral-7B。先从示范做 SFT，再用成败轨迹对做 DPO。见 §3、§4、附录 B、D。 [原文](https://arxiv.org/html/2403.02502) |
| PEARL（2026 预印本） | 先离线试跑工具以记录用法和失败条件，再让 Planner 产出计划、Executor 调用工具，收集反馈。 | ToolHop 做同域测试；T-Eval 取 100 个规划任务改成可执行的 ToolHop 风格，测迁移。报告最终成功率、调用错误率。 | Qwen2.5-7B 架构，用 GRPO 更新 Planner。奖励含规划信号，Executor 的工具知识来自离线试跑。见 §3、§4.1–4.3。 [原文](https://arxiv.org/html/2601.20439) |
| Evo²（2026 预印本） | 记录 Proposer 的 Harness 编辑、候选执行结果、Reflector 的反馈及修订，并利用执行得分估计编辑或反馈的价值。 | 代码用 Polyglot、SWE-bench Verified、LiveCodeBench，另有跨 Benchmark 迁移。记忆任务在 LoCoMo 演化、报留出会话结果，再测 LongMemEval-S。 | 主实验固定 Qwen3.8-27B Task Executor，用同款模型初始化共享的 Proposer / Reflector，再对后者做 SFT 和交错 GRPO。闭源对照另用 GPT-5-mini、Claude Haiku 4.5，权重固定。见 §3.3、§4.1–4.5。 [原文](https://arxiv.org/html/2610.13025v1) |
| HGM（ICLR 2026，配置按 arXiv v3） | 用编码 Agent 执行 SWE-bench Verified / Polyglot；按后代的任务成绩估计分支潜力，决定扩展哪个 Harness。 | 小预算对比包括 SWE-Verified-60 与 Polyglot，大规模实验还包括全量 SWE-Verified。用 SWE-Lite 做迁移，并单列去掉与 Verified 重合任务的结果。 | 对比实验中，SWE 使用 GPT-5 扩展 Harness、GPT-5-mini 执行评测；Polyglot 使用 Qwen3-Coder-480B-A35B-Instruct 扩展、Qwen3-Coder-30B-A3B-Instruct 评测。SWE-Lite 也测试 GPT-5。更新对象是 Harness 源码。见 §4.2、§4.3、附录 C.1。 [原文](https://arxiv.org/html/2510.21614v3) · [正式论文](https://openreview.net/forum?id=T0EiEuhOOL) · [代码](https://github.com/metauto-ai/HGM) |
| MGM（2026 预印本） | 在 SWE-bench Verified-60 和 Polyglot-60 比较 Harness 演化。可让编辑器看同一 Agent 在多题上的轨迹，或同一题上两个谱系的轨迹。 | 主实验跑两个 60 题设置；跨 Benchmark 从 Polyglot 迁移至 SWE-bench Pro / Multilingual；另测试跨模型迁移。 | 主干 Qwen3.6-35B-A3B。修改的是 Harness，轨迹作为编辑证据。主实验用 8 张 H100，200 次任务评估和 24 次 Harness 扩展。见 §4、§5、附录 C。 [原文](https://arxiv.org/html/2608.07645v1) |
| R2E-Gym（2025，arXiv 版本） | 在 R2E-Gym 提供的仓库修复环境采集多步轨迹，通过单元测试做筛选。 | SWE-bench Verified，用测试确认补丁是否解决 issue。 | 编辑 Agent 全量 SFT 的基座为 Qwen2.5-Coder 7B / 14B / 32B，另训练 Qwen2.5-Coder-14B verifier，对轨迹和补丁进行判别。见 §3–4、附录 B 与 C.2。 [原文](https://arxiv.org/html/2504.07164) · [代码](https://github.com/R2E-Gym/R2E-Gym) |
| RandOpt / Neural Thickets（2026，按 arXiv v1） | 采样权重扰动，用训练样本上的任务得分筛选候选；推理时让入选候选投票。 | 主表覆盖 Countdown、GSM8K、MATH-500、OlympiadBench、MBPP、ROCStories、USPTO，另有视觉问答 GQA。 | Qwen2.5-Instruct 0.5B / 1.5B / 3B，另试 Llama-3.1-8B-Instruct、OLMo3-7B Base 等。相关证据主要来自可判分的最终输出。见 §4–5、附录 A、I。 [原文](https://arxiv.org/html/2603.12228v1) · [代码](https://github.com/sunrainyg/RandOpt) |

## Benchmark 与方法的对应关系

- **Search-R1 的直接信号是答对。** NQ 和 HotpotQA 有可自动核对的答案，允许在每条多轮搜索后给终局奖励。HotpotQA、2Wiki 还提供跨文档查证据的机会，可以从日志看到是搜漏了、搜错了，还是已经查到证据却整合错了。Search-R1 的 EM 与 RSIFlow 的答案 F1 是两个协议，跨论文比较时应分别记录。 [Search-R1 §3–4](https://arxiv.org/html/2503.09516v5) · [RSIFlow 附录 A](https://arxiv.org/html/2610.21791v1)
- **Agent-R 和 ETO 需要任务出现可回溯的失败。** WebShop 可以观察搜索、属性选择、购买，TextCraft 可以观察拿取和合成步骤。Agent-R 的测试会让模型接手截断的失败轨迹，测它能否采取挽回任务的动作。ETO 的专家轨迹平均约 4.9 轮（WebShop）、10.1 轮（ALFWorld）、14.4 轮（ScienceWorld）。这些数字反映该论文的参考轨迹长度。 [Agent-R §4](https://arxiv.org/html/2501.11425) · [ETO 表 1、附录 B](https://arxiv.org/html/2403.02502)
- **ToolHop 把多步调用本身变成可观察对象。** 工具在本地执行，上一步结果常是下一步的输入。PEARL 同时看最终成功和调用错误，HarnessForge 把 ToolHop 同时纳入演化数据与留出评测，能进一步用 RestBench-TMDB、API-Bank 看迁移。与只看问答 EM 相比，这里多了直接可观察的选工具、构参和处理调用错误的行为。 [ToolHop](https://arxiv.org/abs/2501.02506) · [PEARL §4](https://arxiv.org/html/2601.20439) · [HarnessForge 附录 B](https://arxiv.org/html/2606.01779)
- **HGM 和 MGM 用测试结果给 Harness 变体打分。** Polyglot 比较独立编程题的测试通过，SWE-bench 比较真实仓库修复。两者都可产生搜索、编写、测试、返工的轨迹，SWE-bench 还包含逐仓库的环境与依赖成本。DGM 附录 E.1 特别把 Polyglot 作为较低成本设置，文中的美元估算采用各自指定的 API 模型。 [HGM §4](https://arxiv.org/html/2510.21614v3) · [MGM §5](https://arxiv.org/html/2608.07645v1) · [DGM 附录 E.1](https://arxiv.org/html/2505.22954)

## 对 11 月 18 日方案的候选

| 想观察的行为 | 优先对照 | 要落实的工程变量 |
| --- | --- | --- |
| 证据不够时，继续查还是马上提交 | Search-R1 的 Qwen2.5-3B 公布权重 + HotpotQA 留出样本；2Wiki 作跨数据集对照 | 固定同一检索库、索引、提示和工具回合上限。查阅实例时对齐查询、返回证据和停止动作。作者提供本地 Wikipedia 检索语料、索引和推理入口。 [官方代码](https://github.com/PeterGriffinJin/Search-R1) · [官方权重](https://huggingface.co/collections/PeterJinGo/search-r1) |
| 工具调用或状态转移出错后，能否纠错 | ToolHop，或 Agent-R 的 WebShop / TextCraft | ToolHop 容易记录错工具、错参和错误传播。WebShop / TextCraft 可观察环境反馈后补救，需要部署各自的 AgentGym 环境。 [ToolHop](https://arxiv.org/abs/2501.02506) · [Agent-R 环境与评测入口](https://github.com/ByteDance-Seed/Agent-R) |
| 代码 Agent 何时运行测试、从报错中恢复或停止修改 | 如果沿用 HGM 的代码路线，先比较 Polyglot 与 SWE-bench Lite 的环境成本 | HGM 仓库公开了 §4.3 的 SWE 系列 best-belief Agent。代码含 vLLM 兼容入口，对本地候选权重是现成接口。Polyglot 是另一个评测设置。将该 Harness 接到新的模型或任务后的基线表现、工具调用兼容性和单条成本待跑。 [best_agent 说明](https://github.com/metauto-ai/HGM/blob/main/best_agent/README.md) · [模型接口](https://github.com/metauto-ai/HGM/blob/main/best_agent/llm.py) |

第一轮如果以观察行为为主，倾向 Search-R1 + HotpotQA。任务的典型分岔可以直接落到可记录的动作上，模型和本地检索接口也有公开起点。如果目标更偏代码 Agent，HGM 的公开 Harness 和 MGM 的同题对照更接近 Harness 演化路线。建议把预算记成每个模型变体的任务数、每题重复数、每条平均步数及生成 token，再分别计推理和环境判分耗时。N 个扰动、M 道题、每题 R 次重复，仅扰动侧就有 N × M × R 条轨迹。第一轮若用开发集选强度或挑候选，最终收益用独立题集衡量。
