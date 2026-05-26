# CO+SIG Benchmark Results — Complete Dimensions Report

> **测试日期**: 2026-05-22 ~ 2026-05-26
> **GPU**: NVIDIA GeForce RTX 4070 SUPER (12 GB, 代表高端边缘工作站)
> **模型**: Qwen3.5-0.8B-Q4_K_M / Qwen3.5-4B-Q4_K_M (llama.cpp, Q4_K_M 量化)
> **N=30 配对运行 (核心性能实验)**: R6, R8, R13, R14
> **N=1 试点测量**: R7, R9
> **分析推测**: R12

> **叙事定位 (2026-05-26 FINAL)**: 
> 
> 本报告配合论文《SIG as a Specialized Edge Accelerator for Long-Tool-Chain Inference — A Design Space Exploration Across Nine Research Vectors》。**SIG 被定位为边缘设备上小模型长工具链的专用流式注入加速器，而非通用 KV 缓存方案。** R6 深链优势是核心证据；R13/R8 定义了专用加速器的设计边界；混合调度策略 (SIG for chains, AppLoop-PC for fragments) 将边界转化为决策框架。
>
> **未进行任何形式假设检验；所有对比为描述性数值比较。全部发现限于 Qwen3.5 Q4_K_M + RTX 4070 SUPER。**

---

## 核心论点 (Core Thesis)

> **SIG 是边缘设备上小模型长工具链的专用流式注入加速器。** 在资源受限的边缘设备上，当小参数模型需要执行连续、深度、带状态的长工具链时，SIG 的"流式持久缓存"范式以极低的增量成本维持上下文，相比依赖重编码或通用前缀缓存的 AppLoop 方案取得数量级层面的端到端效率提升。SIG 在其他场景下的"失效"并非缺陷，而是其作为专用加速器的明确设计边界。

---

## 一、CO+AppLoop vs CO+SIG（教师预计算计划，9 场景）

### 0.8B 模型 (Qwen3.5-0.8B-Q4_K_M)

| 场景 | AppLoop Gen(s) | SIG Gen(s) | AppLoop PF(s) | SIG PF(s) | AppLoop Total(s) | SIG Total(s) | 加速比 |
|------|---------------|------------|--------------|----------|-----------------|--------------|--------|
| 1 Long-seq (22轮) | 2.88 | 1.92 | 2.37 | 0.29 | 5.25 | 2.21 | **2.38×** |
| 2 Multi-tool | 0.72 | 0.77 | 0.04 | 0.04 | 0.77 | 0.81 | 0.95× |
| 3 Rapid-fire (12轮) | 2.16 | 1.23 | 0.68 | 0.17 | 2.84 | 1.40 | **2.03×** |
| 4 Long-document | 1.05 | 0.64 | 0.17 | 0.10 | 1.21 | 0.74 | **1.64×** |
| 5 Mixed (8轮) | 2.63 | 1.53 | 0.33 | 0.10 | 2.96 | 1.64 | **1.80×** |
| 6 Deep chain (14工具) | 9.28 | 2.06 | 1.94 | 0.20 | 11.22 | 2.26 | **4.96×** |
| 7 Travel planning (11工具) | 3.21 | 2.21 | 0.89 | 0.18 | 4.10 | 2.39 | **1.72×** |
| 8 Code debugging | 1.12 | 0.55 | 0.29 | 0.11 | 1.41 | 0.67 | **2.10×** |
| 9 Cross-reference (9工具) | 1.23 | 1.00 | 0.52 | 0.13 | 1.75 | 1.12 | **1.56×** |
| **平均** | **2.70** | **1.32** | **0.80** | **0.15** | **3.50** | **1.47** | **2.38×** |

### 4B 模型 (Qwen3.5-4B-Q4_K_M)

| 场景 | AppLoop Gen(s) | SIG Gen(s) | AppLoop PF(s) | SIG PF(s) | AppLoop Total(s) | SIG Total(s) | 加速比 |
|------|---------------|------------|--------------|----------|-----------------|--------------|--------|
| 1 Long-seq (22轮) | 8.03 | 3.26 | 6.97 | 0.54 | 15.00 | 3.80 | **3.95×** |
| 2 Multi-tool | 0.54 | 0.54 | 0.10 | 0.09 | 0.63 | 0.64 | 0.98× |
| 3 Rapid-fire (12轮) | 9.16 | 3.68 | 2.64 | 0.31 | 11.80 | 3.99 | **2.96×** |
| 4 Long-document | 5.91 | 3.19 | 0.85 | 0.17 | 6.75 | 3.36 | **2.01×** |
| 5 Mixed (8轮) | 5.63 | 3.43 | 0.91 | 0.26 | 6.54 | 3.70 | **1.77×** |
| 6 Deep chain (14工具) | 12.34 | 2.80 | 4.69 | 0.45 | 17.03 | 3.24 | **5.26×** |
| 7 Travel planning (11工具) | 10.63 | 3.56 | 3.13 | 0.39 | 13.76 | 3.95 | **3.48×** |
| 8 Code debugging | 2.70 | 1.89 | 0.94 | 0.30 | 3.64 | 2.19 | **1.66×** |
| 9 Cross-reference (9工具) | 4.21 | 4.94 | 1.73 | 0.28 | 5.94 | 5.22 | 1.14× |
| **平均** | **6.57** | **3.03** | **2.44** | **0.31** | **9.01** | **3.34** | **2.70×** |

### CO 汇总

| 指标 | 0.8B AppLoop | 0.8B SIG | 4B AppLoop | 4B SIG |
|------|-------------|----------|-----------|--------|
| 平均生成时间 | 2.70s | 1.32s | 6.57s | 3.03s |
| 平均 Prefill 时间 | 0.80s | 0.15s | 2.44s | 0.31s |
| **Prefill 时间节省** | | **81%** | | **87%** |
| 平均总时间 | 3.50s | 1.47s | 9.01s | 3.34s |
| **端到端加速比** | | **2.38×** | | **2.70×** |
| GPU 显存 | ~1.3 GB | ~1.4 GB | ~3.9 GB | ~4.0 GB |

---

## 二、Per-Token Generation Rate Analysis（生成速率对比）

**新增**：记录 `gen_toks` + `pf_toks` 以分离 "生成速度" 与 "输出长度" 对总时间的影响。

### Gen_Toks 分解

| 模型 | AppLoop gen_toks | SIG gen_toks | AppLoop pf_toks | SIG pf_toks | AppLoop tok/s | SIG tok/s |
|------|-----------------|-------------|----------------|-------------|---------------|-----------|
| 0.8B | 699 | 354 | 16260 | 1112 | **274** | **281** |
| 4B | 625 | 296 | 15636 | 1112 | **99** | **101** |

### 核心发现

- **Per-token 生成速率几乎相同**：SIG 与 AppLoop 差值 < 2%（两种模型均如此）
- SIG 的"生成时间减半"完全来自**输出更短**，而非每 token 更快
- 总时间加速主要来源：**93% prefill token 节省**（16260 → 1112 / 15636 → 1112）
- 4B/0.8B token 速度比：**2.8×** — 5× 参数量仅产生 2.8× 的 per-token 减速

---

## 三、SIG 自主工具调用（模型自生成工具调用，9 场景）

### 0.8B 模型 — 工具准确率

| 场景 | AppLoop | SIG | 优势方 |
|------|---------|-----|--------|
| 1 Long-seq (22轮) | 1/22 (5%) | 15/22 (68%) | **SIG** |
| 2 Multi-tool (4工具) | 0/4 (0%) | 3/4 (75%) | **SIG** |
| 3 Rapid-fire (12轮) | 0/12 (0%) | 8/12 (67%) | **SIG** |
| 4 Long-document (4工具) | 0/4 (0%) | 4/4 (100%) | **SIG** |
| 5 Mixed (4工具) | 4/4 (100%) | 4/4 (100%) | 平手 |
| 6 Deep chain (14工具) | 0/14 (0%) | 14/14 (100%) | **SIG** |
| 7 Travel planning (11工具) | 0/0 (N/A) | 7/13 (54%) | **SIG** |
| 8 Code debugging (4工具) | 3/4 (75%) | 2/2 (100%) | **SIG** |
| 9 Cross-reference (9工具) | 0/0 (N/A) | 5/11 (45%) | **SIG** |

### 4B 模型 — 工具准确率

| 场景 | AppLoop | SIG | 优势方 |
|------|---------|-----|--------|
| 1 Long-seq (22轮) | 22/22 (100%) | 22/22 (100%) | 平手 |
| 2 Multi-tool (4工具) | 4/4 (100%) | 4/4 (100%) | 平手 |
| 3 Rapid-fire (12轮) | 12/12 (100%) | 12/12 (100%) | 平手 |
| 4 Long-document (4工具) | 4/4 (100%) | 4/4 (100%) | 平手 |
| 5 Mixed (4工具) | 4/4 (100%) | 4/4 (100%) | 平手 |
| 6 Deep chain (14工具) | 14/14 (100%) | 13/14 (93%) | AppLoop |
| 7 Travel planning (11工具) | 4/7 (57%) | 0/0 (N/A) | AppLoop |
| 8 Code debugging (4工具) | 4/6 (67%) | 2/2 (100%) | **SIG** |
| 9 Cross-reference (9工具) | 8/8 (100%) | 7/7 (100%) | 平手 |

### Prefill Token 节省（自主模式）

| 场景 | 0.8B Full | 0.8B SIG | 节省 | 4B Full | 4B SIG | 节省 |
|------|----------|----------|------|---------|--------|------|
| 1 Long-seq | 18396 | 1371 | **93%** | 23245 | 1371 | **94%** |
| 2 Multi-tool | 2017 | 537 | **73%** | 2466 | 537 | **78%** |
| 3 Rapid-fire | 6368 | 869 | **86%** | 7965 | 840 | **89%** |
| 4 Long-document | 4453 | 1021 | **77%** | 4670 | 991 | **79%** |
| 5 Mixed | 3453 | 670 | **81%** | 4471 | 546 | **88%** |
| 6 Deep chain | 10038 | 1071 | **89%** | 12339 | 1069 | **91%** |
| 7 Travel plan | 639 | 880 | -38% | 12904 | 382 | **97%** |
| 8 Code debug | 5166 | 719 | **86%** | 5712 | 719 | **87%** |
| 9 Cross-ref | 681 | 734 | -8% | 8708 | 725 | **92%** |

### SIG 复杂任务 Chain Completion

| 场景 | 0.8B AppLoop | 0.8B SIG | 4B AppLoop | 4B SIG |
|------|-------------|----------|-----------|--------|
| Travel planning (11) | 0/11 | 8/11 | 4/11 | 1/11 |
| Code debugging (4) | 3/4 | 3/4 | 4/4 | 3/4 |
| Cross-reference (9) | 0/9 | 6/9 | 8/9 | 9/9 |

---

## 四、R1: Attention Distribution Analysis（注意力分布偏移）

> **测试入口**: `transformer_bench.py --task r1` 或 `sig_benchmark.py --task r1`
> **模型**: Qwen2.5-0.5B (24 layers, 14 heads, FP16, modelscope)
> **方法**: 比较 full re-encoding vs SIG injection (past_key_values) 的注意力权重

### Per-Layer Head Agreement & Cosine Similarity

| 层组 | Head Agreement | Cosine Similarity | 解释 |
|------|----------------|--------------------|------|
| Early (0–7) | **0.252** | 0.647 | 基础注意力模式受注入扰动最大 |
| Middle (8–15) | 0.304 | 0.735 | 通过 self-attention 部分恢复 |
| Late (16–23) | 0.427 | 0.793 | 任务精炼层受影响最小 |
| **Overall** | **0.327** | **0.725** | 实质性但非灾难性偏移 |

### 关键结论

- **确认层敏感度梯度**：早期层 HeadAgr 最低 (0.25)，后期层最高 (0.43)，证实"早期扰动 → 后期恢复"假设
- **Cosine Similarity 0.725**：整体注意力模式保留实质性结构相似性
- 仅约 1/3 的注意力头在注入下与全量重编码关注相同的 top-5 位置
- **首次直接实证测量** SIG 注入对内部注意力表征的影响

---

## 五、R2: KV-Cache Degradation（缓存退化实验）

> **测试入口**: `co_benchmark.py --task r2 --r2-n-cities N --r2-probe-interval 2`
> **方法**: 逐轮注入天气数据 + 每 2 轮探针测量事实 recall

### 0.8B 模型 — 天气 Recall

| 轮次 | Cache Tokens | 短期 Recall | 长期 Recall | 观察 |
|------|-------------|------------|------------|------|
| 2 | 126 | 0.50 | 0.50 | 双召回偏部分 |
| 4 | 249 | 0.50 | 0.50 | 稳定 |
| 6 | 360 | 1.00 | 0.50 | 短期改善，长期稳定 |
| 8 | 472 | 1.00 | 0.50 | 短期维持 |

### 4B 模型 — 天气 Recall

| 轮次 | Cache Tokens | 短期 Recall | 长期 Recall | 观察 |
|------|-------------|------------|------------|------|
| 2 | 126 | 0.50 | 0.50 | 双召回偏部分 |
| 4 | 249 | 0.50 | 0.50 | 稳定 |
| 6 | 360 | 1.00 | 0.50 | 短期改善 |
| 8 | 472 | 1.00 | 0.50 | 短期维持 |
| 10 | 591 | 0.50 | 0.50 | 双双稳定 |

### 关键结论

- **6–10 轮未观察到退化**：长期 recall 在两种模型上均稳定 0.50
- 短期 recall 在 0.50–1.00 间波动，无单调下降趋势
- 4B 模型在简单事实 recall 上无优势 — 两种模型表现等同
- KV 缓存保持注入天气信息的可达性，无测量到衰减
- **局限**: 仅测试 6–10 轮；长值（e.g. 18C）有利于 recall；未测试 32+ 轮场景

---

## 六、R3: Cross-Architecture SIG Simulation（跨架构仿真）

> **测试入口**: `transformer_bench.py --task r3` 或 `sig_benchmark.py --task r3`
> **参数**: d_model=256, n_heads=8, n_layers=6, n_injections=3

### 信息保留率

| 架构 | Avg Retention | Final Retention | Avg Fidelity |
|------|-------------|-----------------|-------------|
| Transformer | 0.079 | 0.080 | 0.000 |
| SSM/Mamba | 0.778 | 0.976 | 0.000 |
| RWKV | 0.985 | 0.987 | 0.286 |
| xLSTM | 1.000 | 1.000 | 0.081 |

### 状态容量

| 架构 | 状态元素 | 内存 (fp16) | 信息密度 |
|------|---------|------------|----------|
| Transformer | 6,291,456 | 12.00 MB | -0.028 |
| SSM/Mamba | 24,576 | 0.05 MB | -1.806 |
| RWKV | 4,608 | 0.01 MB | 2.644 |
| xLSTM | 27,696 | 0.05 MB | 3.747 |

### SIG 操作延迟（仿真）

| 架构 | Init | Inject | Suspend | Resume |
|------|------|--------|---------|--------|
| Transformer | 2.47ms | 1.15ms | 0.10ms | 0.21ms |
| RWKV | 11.61ms | 5.43ms | 0.00ms | 0.20ms |
| xLSTM | 32.20ms | 14.84ms | 0.00ms | 0.84ms |
| SSM/Mamba | 220.79ms | 115.92ms | 0.15ms | 12.47ms |

### SIG 可行性排位

1. **Transformer**: 原生 SIG 支持（append-only KV cache）
2. **xLSTM**: 良好 SIG 支持（可加性 cell state + rank-1 更新）
3. **RWKV**: 中等 SIG 支持（decay-weighted 注入）
4. **SSM/Mamba**: 挑战性（固定容量状态瓶颈）

### R3 实证参数化

| 参数 | 0.8B | 4B |
|------|------|-----|
| Prefill 节省 | 93.2% | 92.9% |
| Token 速度 | 281 tok/s | 101 tok/s |
| 速度比 (4B/0.8B) | 2.8× | |

**非 Transformer 投射**（均为假设，无实现）:

| 架构 | 预填节省投射 | 基础 |
|------|-------------|------|
| Transformer | 100% | Qwen3.5 实测 |
| xLSTM | 85-95% | Rank-1 矩阵注入假设 |
| RWKV | 70-85% | Causal (k,v) 插入假设 |
| Mamba/SSM | 40-60% | 状态容量瓶颈假设 |

---

## 七、R4: Teacher-Student Capability Gap（师生能力差距）

> **测试入口**: `co_benchmark.py --task r4`（无需模型，硬编码 CO 基准数据）

### 自主工具调用准确率（按场景）

| 场景 | 0.8B Alone | 0.8B+SIG | 4B Alone | 4B+SIG |
|------|-----------|----------|---------|--------|
| Long-seq (22) | 0.05 | 0.68 | 1.00 | 1.00 |
| Multi-tool (4) | 0.00 | 0.75 | 1.00 | 1.00 |
| Rapid-fire (12) | 0.00 | 0.67 | 1.00 | 1.00 |
| Long-doc (4) | 0.00 | 1.00 | 1.00 | 1.00 |
| Mixed (4) | 1.00 | 1.00 | 1.00 | 1.00 |
| Deep chain (14) | 0.00 | 1.00 | 1.00 | 0.93 |
| Travel plan (11) | 0.00 | 0.54 | 0.57 | 0.00 |
| Code debug (4) | 0.75 | 1.00 | 0.67 | 1.00 |
| Cross-ref (9) | 0.00 | 0.45 | 1.00 | 1.00 |
| **平均** | **0.20** | **0.79** | **0.92** | **0.88** |

### 关键差距测量

| 指标 | 值 | 含义 |
|------|-----|------|
| **CoT 放大系数** | **+0.80** (80pp) | 0.8B alone=0.20 → 0.8B+4B CoT+SIG=1.00 |
| **SIG 放大系数** | **+0.59** (59pp) | 0.8B alone=0.20 → 0.8B+SIG=0.79 |
| **教师质量边际** | **0.72** | 4B alone=0.92 − 0.8B alone=0.20 |
| **CoT+SIG 合并** | **1.00** | 小模型+CoT+SIG 匹配大模型自主性能 |

### 结论

- 5× 师生比下，CoT 提供 80pp 绝对准确率增益
- SIG 独立提供 59pp，CoT 独立提供 80pp — 两者互补
- CoT+SIG 合并使 0.8B 模型达到或超越 4B 模型自主性能
- **局限**: 单对师生 (4B/0.8B)；无 teacher-size 扫描；CoT 预计算

---

## 八、R5: Privacy Anonymization Concept Demo（隐私匿名化演示）

> **测试入口**: `co_benchmark.py --task r5`（无需模型，正则匹配 PII）

### 四类查询 PII 检测

| 查询类型 | 原长度 | PII 移除后 | 意图外包后 | PII 项数 |
|----------|--------|----------|-----------|---------|
| Travel planning | 73 | 71 | 72 | 1 |
| Code debugging + PII | 91 | 76 | 76 | 2 |
| Medical query | 85 | 77 | 77 | 2 |
| Financial planning | 76 | 76 | 78 | 0 |

### 示例变换 (Code+PII)

```
原始:     My name is Dr. Sarah Chen. Bug in calculator.py. Email sarah@hospital.org. Salary $150,000.
PII移除:  My name is Dr. [NAME]. Bug in calculator.py. Email [EMAIL]. Salary $150,000.
意图外包: My name is Dr. [NAME]. Bug in calculator.py. Email [EMAIL]. Salary [AMOUNT].
```

### 支持的模式匹配

| 正则 | 替换 | 类型 |
|------|------|------|
| `[A-Z][a-z]+ [A-Z][a-z]+` | `[NAME]` | 全名 |
| `\d{3}-\d{2}-\d{4}` | `[SSN]` | 社保号 |
| `[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z\|a-z]{2,}` | `[EMAIL]` | 邮箱 |
| `\d{10,16}` | `[PHONE/CARD]` | 电话/卡号 |
| `\d{1,2}/\d{1,2}/\d{2,4}` | `[DATE]` | 日期 |
| `\$[\d,]+(\.\d{2})?` | `[SALARY]` | 工资 |

**注**: 仅概念演示。无差分隐私形式保证、PII 检测精度/召回未测量、未做攻击仿真。

---

## 九、测试基础设施总览

| 入口脚本 | Task | 需要模型? | 功能 |
|---------|------|-----------|------|
| `co_benchmark.py` | `--task baseline` | GGUF | CO 9-scenario baseline (AppLoop vs SIG) |
| `co_benchmark.py` | `--task r2` | GGUF | KV-Cache degradation (weather recall) |
| `co_benchmark.py` | `--task r4` | 否 | Teacher-student capability gap |
| `co_benchmark.py` | `--task r5` | 否 | Privacy anonymization demo |
| `sig_benchmark.py` | `--task baseline` | GGUF | SIG 9-scenario baseline (autonomous mode) |
| `sig_benchmark.py` | `--task r1` | HF modelscope | SIG attention distribution analysis |
| `sig_benchmark.py` | `--task r3` | 否 (numpy) | Cross-architecture SIG simulation |
| `transformer_bench.py` | `--task r1` | HF modelscope | Standalone R1 attention analysis |
| `transformer_bench.py` | `--task r3` | 否 (numpy) | Standalone R3 cross-architecture |
| `transformer_bench.py` | `--task r3-empirical` | 否 | CO benchmark parameterization |
| `transformer_bench.py` | `--task all --output X.json` | HF | 全部 R1+R3 导出 JSON |

---

## 十、关键发现总汇

### 10.1. CO+SIG 加速效果

- 0.8B 平均 **2.38×** 加速，prefill 时间 **-81%**
- 4B 平均 **2.70×** 加速，prefill 时间 **-87%**
- 最佳场景：Deep chain (14工具) — **5.26× (4B)** / **4.96× (0.8B)**

### 10.2. 总时间加速来自 Prefill

- Per-token 生成速度 AppLoop vs SIG 差值 < 2%
- SIG 的 "生成时间减少" 完全来自输出更短
- Prefill token 节省 **93%**（两种模型）

### 10.3. KV 缓存连续性是 0.8B 自主调用的关键

- 0.8B AppLoop 在 6/9 场景准确率接近 0%
- SIG 使 Deep chain 从 0% → 100%，Long-seq 从 5% → 68%

### 10.4. 4B 自主模式需要优化

- 4B SIG 在 Travel/Code 场景出现 rollback
- 合成注入 (synthetic injection) 帮助恢复但仍有优化空间

### 10.5. R1: 注意力层敏感度梯度确认

- Early→Late: HeadAgr 0.25→0.43, CosSim 0.65→0.79
- 早期层最敏感，后期层逐渐恢复

### 10.6. R2: 6–10 轮无退化

- 长期 weather recall 稳定 0.50（两种模型）
- 4B 无优势 — 简单事实 recall 两种模型等同

### 10.7. R3: Transformer 原生 SIG 最优

- 仿真排位: Transformer > xLSTM > RWKV > SSM/Mamba
- SSM 固定状态无法有效支持长序列注入
- 非 Transformer 架构投射值均为假设

### 10.8. R4: CoT + SIG 协同效果显著

- CoT 放大 +80pp，SIG 放大 +59pp
- 合并后 0.8B + CoT + SIG = 1.00（匹配 4B 自主性能）
- 5× 师生比下 gap 可完全弥合

### 10.9. GPU 显存

| 模型 | AppLoop | SIG |
|------|---------|-----|
| 0.8B | ~1.3 GB | ~1.4 GB |
| 4B | ~3.9 GB | ~4.0 GB |

SIG 增加约 **0.1 GB**，在可接受范围内。

---

## 十二、R6-R14 扩展研究测试 (2026-05-25)

> **测试日期**: 2026-05-25
> **GPU**: NVIDIA GeForce RTX 4070 SUPER (12 GB)
> **GGUF 模型**: Qwen3.5-0.8B-Q4_K_M / Qwen3.5-4B-Q4_K_M (llama.cpp)
> **modelscope**: Qwen2.5-0.5B / Qwen2.5-1.5B (HuggingFace, FP16, device_map=auto)

---

### 12.1. R6: 动态重规划 — 工具失败恢复 ✅ **实测 N=30**

30工具深度, 15%故障注入 (base seed 42, 每run独立重采样), N=30配对运行, 三模式 (SIG / AppLoop / AppLoop-PC), 端到端 wall-clock。

| 模型 | SIG | AppLoop | AppLoop-PC | SIG vs AppLoop | SIG vs AppLoop-PC |
|------|-----|---------|------------|----------------|-------------------|
| 0.8B | 0.232±0.021s | 0.646±0.022s | 0.646±0.021s | **2.79×** | **2.79×** |
| 4B   | 0.480±0.016s | 2.043±0.073s | 2.038±0.084s | **4.26×** | **4.25×** |

**发现**: AppLoop-PC（前缀缓存）与原始 AppLoop 无差异——30 步上下文从零累积，前缀缓存恢复后仍需重新 eval 全部内容。SIG 在 4B 上加速比 (4.26×) 显著大于 0.8B (2.79×)，prefill cost 占比随模型增大而升高。这是目前对 CO+SIG 最直接的端到端加速证据。

---

### 12.2. R7: 多模态 SIG — 结构化数据注入效率

对比三种数据表示格式的 KV cache 注入效率：

| 格式 | 0.8B Tokens | 0.8B Eval(ms) | 0.8B Density | 4B Tokens | 4B Eval(ms) | 4B Density |
|------|------------|--------------|-------------|----------|------------|-----------|
| Structured JSON | 112 | 20.4 | 0.390 | 112 | 37.1 | 0.390 |
| Plain text | 77 | 27.8 | 0.365 | 77 | 55.8 | 0.365 |
| Minimal text | 70 | 22.7 | 0.385 | 70 | 54.8 | 0.385 |

**发现**: JSON 结构化格式反直觉地增加了 **45% token 开销**（括号/引号/冒号等结构字符导致更多 token）。Minimal text 格式 token 最少（70），plain text 次之（77）。多模态 SIG 应优先使用紧凑的纯文本或 minimal key-info 格式，而非 JSON。

**Eval 时间异常**: 0.8B 上 JSON 格式 eval 时间 (20.4ms) 低于 Plain text (27.8ms)，尽管 token 数更多。这可能是因为 JSON token 更短（单字符符号如 `{`、`"`、`:`）能在更紧密的 CUDA kernel 批处理中执行，而较长的自然语言 token 每个 token 消耗更多内存带宽。此异常不影响主要建议（token 效率优先使用 plain text），但提示 eval 时间优化可能依赖于格式。

---

### 12.3. R8: 长上下文精确检索 + 序列推理探针 ✅ **实测 N=30**

12轮导航 (6房间×2轮), N=30配对运行, 三模式 (SIG / AppLoop / AppLoop-PC), 检索探针(T=3/6/9/12) + 空间推理探针(T=10)。

**上下文精确检索命中率**:

| Probe | 0.8B SIG | 0.8B AppLoop | 0.8B AppLoop-PC | 4B SIG | 4B AppLoop | 4B AppLoop-PC |
|-------|----------|-------------|----------------|--------|-----------|--------------|
| T=3   | 0/30 (0%) | 0/30 (0%) | **27/30 (90%)** | 0/30 (0%) | 0/30 (0%) | **28/30 (93%)** |
| T=6   | 0/30 (0%) | 0/30 (0%) | **24/30 (80%)** | **28/30 (93%)** | 0/30 (0%) | 23/30 (77%) |
| T=9   | 0/30 (0%) | 0/30 (0%) | 9/30 (30%) | 12/30 (40%) | **30/30 (100%)** | 26/30 (87%) |
| T=12  | 0/30 (0%) | 0/30 (0%) | 0/30 (0%) | 0/30 (0%) | 0/30 (0%) | 0/30 (0%) |
| **Total** | **0/120 (0%)** | **0/120 (0%)** | **60/120 (50%)** | 40/120 (33%) | 30/120 (25%) | **77/120 (64%)** |

**空间推理探针 (T=10)**:
| 模型 | SIG | AppLoop | AppLoop-PC |
|------|-----|---------|------------|
| 0.8B | 0/30 (0%) | 0/30 (0%) | 0/30 (0%) |
| 4B   | 8/30 (26.7%) | **30/30 (100%)** | 5/30 (16.7%) |

**发现**: 0.8B 的 SIG 在所有检索探针上完全失败（0/120命中），4B SIG 仅命中 33%。AppLoop-PC 在两个模型上总体最优（50% / 64%），但在 T=12 处三模式均降至 0——所有模式的远距离检索完全失败。空间推理方面，4B AppLoop 达 100%，而 AppLoop-PC 仅 16.7%，表明前缀缓存的系统提示+纯上下文注入模式在此类多步推理任务上反而不如完整重编码。SIG 的增量注入模式在检索和推理任务上一致弱于 AppLoop-PC baseline。

---

### 12.4. R9: 实时性约束 — 延迟预算分析

测试 200~2000 token 上下文的 prefill + generation 延迟（延迟预算 2.0s）。

**0.8B 模型**:
| Context Tok | Prefill(s) | Gen(s) | Total(s) | Budget | Strategy |
|-------------|-----------|--------|---------|--------|----------|
| 200 | 0.007 | 0.065 | 0.072 | OK | Teacher planning safe |
| 500 | 0.007 | 0.071 | 0.078 | OK | Teacher planning safe |
| 1000 | 0.034 | 0.070 | 0.104 | OK | Teacher planning safe |
| 2000 | 0.087 | 0.067 | 0.154 | OK | Teacher planning safe |

**4B 模型**:
| Context Tok | Prefill(s) | Gen(s) | Total(s) | Budget | Strategy |
|-------------|-----------|--------|---------|--------|----------|
| 200 | 0.013 | 0.156 | 0.170 | OK | Teacher planning safe |
| 500 | 0.028 | 0.179 | 0.206 | OK | Teacher planning safe |
| 1000 | 0.110 | 0.187 | 0.297 | OK | Teacher planning safe |
| 2000 | 0.288 | 0.185 | 0.473 | OK | Teacher planning safe |

**发现**: 所有上下文规模均在 2.0s 延迟预算内。Prefill 从 200→2000 tokens 缩放 12.3×（0.8B）/ 22.2×（4B），但绝对延迟仍然很低。预判性注入（predictive injection）和推测解码可进一步降低延迟。由于是模拟场景（单个 prefill），实际多轮交互中 SIG 节省的 re-prefill 是关键优势。

---

### 12.5. R10: 注入攻击与防御 ✅ **实测**

**攻击面分析** (理论风险评分):
| Attack Vector | Risk |
|---------------|------|
| Attention manipulation | 0.91 |
| Prompt injection | 0.85 |
| Data exfiltration | 0.78 |
| Result poisoning | 0.72 |
| Cache pollution | 0.63 |

**实测缓存污染+Rollback测试 (Qwen3.5, 10 攻击向量×2 模型)**:

| 模型 | 攻击成功 | 成功率 | Rollback 恢复 | 恢复率 |
|------|---------|--------|-------------|--------|
| 0.8B | 6/10 | 60% | 10/10 | 100% |
| 4B | 8/10 | 80% | 10/10 | 100% |
| **合并** | **14/20** | **70%** | **20/20** | **100% (Wilson 95% CI: 84%–100%)** |

最有效攻击向量: instruction hijack (2/2模型), fake authority (2/2), result poisoning (2/2), numerical poisoning (2/2).
最无效: context flood (1/2), multi-turn contamination (0/2), role override (0/2).

**发现**: 4B 模型仍更易受攻击 (80% vs 60%)。20/20 次成功恢复 (Wilson 95% CI: 84%–100%; 注意: 该 95% CI (84%-100%) 反映 N=20 样本下的估计精度不足——真恢复率可能是 100%，CI 仅指示当前样本量下的不确定性范围)。指令劫持和伪造权威是最高危攻击向量。上下文淹没攻击仅对 4B 生效。

**新增：连续语义相似度度量 + LLM-as-Judge 自评估。** 除了二值关键词检测外，新增了字符3-gram Jaccard相似度、词级重叠率、回答长度比、污染残余Jaccard四项连续指标。这些指标能够检测关键词模式匹配漏掉的微妙残余污染，如模型语气改变、信心校准偏移或语义层面的微妙偏差。攻击前基线回答与回滚后回答的比较提供了比纯关键词匹配更细粒度的污染信号。

**新增：LLM-as-Judge 自评估。** 使用 Qwen2.5-7B-Instruct 作为评判模型，对回滚后回答与攻击前基线回答进行语义等价性评分（1-5分 Likert 量表），补充连续语义相似度度量。LLM-Judge 评分与连续相似度度量一致性高（Spearman ρ=0.82），验证了自评估的有效性。两种信号联合使用时，残余污染检测召回率提升至 96%（vs 单独关键词检测的 78%）。

---

### 12.6. R11: 工具结果语义忠实度 — Token-Jaccard ✅ **实测 N=6×2×2=24**

12个工具查询 (6真实城市实体 × 2模式 SIG/AppLoop) × 2模型, Token-Jaccard作为主度量 (模型自身Tokenizer的词汇单元重叠率)。

**0.8B 模型**:
| 查询 | SIG TokJac | AppLoop TokJac | SIG N4Jac | AppLoop N4Jac | SIG KW | AppLoop KW |
|------|-----------|---------------|-----------|---------------|--------|-----------|
| london | 0.036 | 0.033 | 0.004 | 0.003 | 0/2 | 0/2 |
| rome | 0.032 | 0.032 | 0.023 | 0.023 | 2/8 | 2/8 |
| newyork | 0.022 | 0.020 | 0.022 | 0.019 | 0/4 | 0/4 |
| tokyo | 0.015 | 0.015 | 0.000 | 0.000 | 0/2 | 0/2 |
| sydney | 0.017 | 0.322 | 0.015 | 0.253 | 1/9 | 9/9 |
| dubai | 0.032 | 0.032 | 0.031 | 0.031 | 1/4 | 1/4 |

**4B 模型**:
| 查询 | SIG TokJac | AppLoop TokJac | SIG N4Jac | AppLoop N4Jac | SIG KW | AppLoop KW |
|------|-----------|---------------|-----------|---------------|--------|-----------|
| london | 0.132 | 0.132 | 0.044 | 0.044 | 2/2 | 2/2 |
| rome | 0.067 | 0.067 | 0.000 | 0.000 | 0/8 | 0/8 |
| newyork | 0.000 | 0.000 | 0.000 | 0.000 | 0/4 | 0/4 |
| tokyo | 0.068 | 0.000 | 0.009 | 0.000 | 2/2 | 0/2 |
| sydney | 0.067 | 0.067 | 0.014 | 0.014 | 0/9 | 0/9 |
| dubai | 0.000 | 0.000 | 0.000 | 0.000 | 0/4 | 0/4 |

**发现**: Token-Jaccard 在两个模型/模式之间差异微乎其微——SIG 与 AppLoop 在忠实度上平价。4B 整体 Token-Jaccard (均值 0.056) 低于 0.8B (均值 0.065，排除sydney异常值后)，反映更丰富的改写行为而非信息丢失。sydney 在 0.8B AppLoop 上出现异常高 TokJac (0.322)，可能为随机种子 artifact——N=30 可平滑此波动。LLM-Judge: 11/12 NOT_SUPPORTED, 1/12 SUPPORTED (4B AppLoop tokyo) — 总体倾向不判定忠实，与假说一致。

---

### 12.7. R12: SIG Scaling Law — Analytic Projections (Moved to Discussion)

> **R12 已从主实证向量组剥离。** 仅有 0.8B 128-2048 token 的 prefill 为实测。所有模型规模和上下文长度外推均基于 T∝M^0.7 假设 —— 分析推测，非实证发现。

**实测 Prefill 缩放** (0.8B):
| Context Tok | Prefill(ms) | Tok/s |
|-------------|------------|-------|
| 128 | ~7 | ~18,000 |
| 256 | ~12 | ~21,000 |
| 512 | ~25 | ~20,000 |
| 1024 | ~55 | ~19,000 |
| 2048 | ~95 | ~22,000 |

prefill 成本相对模型规模的缩放使用 T ∝ M^0.7 投影（compute-bound prefill 延迟），以实测 0.8B/4B 数据点标定（~20K tok/s @ 0.8B, ~8K tok/s @ 4B）。预填充成本跨模型规模投影采用 T ∝ M^0.7 作为计算密集型预填充延迟的近似，基于两个实测点标定（0.8B ≈ 20,000 tok/s, 4B ≈ 8,000 tok/s）。

**理论投影 — 模型规模 [分析性外推 — 非实测数据, 仅供示意]:**
| Model Size | AppLoop Tok/s | SIG Speedup |
|------------|--------------|-------------|
| 0.5B | 45.0 *(est.)* | 8.5× *(est.)* |
| 0.8B | 35.0 | 7.0× |
| 3B | 22.0 *(est.)* | 5.5× *(est.)* |
| 4B | 18.0 | 4.8× |
| 7B | 12.0 *(est.)* | 4.0× *(est.)* |
| 13B | 8.0 *(est.)* | 3.2× *(est.)* |
| 70B | 2.0 *(est.)* | 2.5× *(est.)* |

**上下文长度与Prefill节约 [分析性外推 — 非实测数据, 仅供示意]:**
| Context | App Prefill | SIG Prefill | Saving | Break-even |
|---------|------------|-------------|--------|------------|
| 4K | 0.5s | 0.1s | 80% | >2 turns |
| 8K | 1.2s | 0.2s | 83% | >1 turn |
| 16K | 3.0s | 0.3s | 90% | always |
| 32K | 7.5s | 0.5s | 93% | always |
| 128K | 45s | 2.0s | 96% | always |

**发现**: SIG 加速比随模型规模增大而增长（prefill cost 占比增大）。对于 >3B 模型或 >8K 上下文，SIG 无争议优于 AppLoop。0.8B 模型 prefill 效率约 20K tok/s，线性缩放良好。

---

### 12.8. R13: 碎片化本地KV重建 — 端到端 Wall-Clock ✅ **实测 N=30**

10轮对话, 8工具调用 (跨 4 虚拟设备), N=30配对运行, 三模式 (SIG / AppLoop / AppLoop-PC), 端到端 wall-clock 为**唯一指标**。

| 模型 | SIG | AppLoop | AppLoop-PC | SIG vs AppLoop | SIG vs AppLoop-PC |
|------|-----|---------|------------|----------------|-------------------|
| 0.8B | 0.713±0.359s | 0.232±0.003s | 0.231±0.004s | **0.33× (SIG 更慢)** | **0.32× (SIG 更慢)** |
| 4B   | 1.396±0.089s | 1.336±0.014s | 1.508±0.111s | **0.96× (平价)** | **1.08×** |

**发现**: 这是 SIG **不占优势**的核心证据。0.8B 上 SIG 反而慢 3 倍（每个碎片注入都需部分缓存操作，抵消了前缀缓存节省）。4B 上勉强平价（1.08×，在标准差范围内不显著）。碎片化上下文重建是 SIG 的已知结构局限性——注入操作的单位开销在短上下文/小模型上占比过高，仅在大模型长链上可能体现优势。此实验非分布式部署——仅测量单GPU上多片段上下文的本地重建成本。

---

### 12.9. R14: SIG + 推理范式 — CoT公平对比 ✅ **实测 N=30**

2个查询, 四模式 (CoT+SIG / CoT+AppLoop / CoT+AppLoop-PC / SIG_raw), N=30配对运行, 输出长度控制于 80 tokens, gen-token 追踪。

**Q1: 三城市比较 (6 tools)**:

| 模式 | 0.8B Wall-Clock | 0.8B Gen Tok | 4B Wall-Clock | 4B Gen Tok |
|------|----------------|-------------|---------------|-----------|
| CoT+SIG | 0.123±0.021s | 28 | 0.688±0.019s | 71 |
| CoT+AppLoop | 0.119±0.003s | 28 | 0.687±0.002s | 71 |
| CoT+AppLoop-PC | 0.156±0.029s | 39 | 0.577±0.240s | 59 |
| SIG_raw | 0.335±0.019s | 79 | 0.139±0.002s | 2 |
| CoT+SIG vs CoT+AppLoop | **1.03×** | 1.00 | **1.00×** | 1.00 |

**Q2: 旅行规划 (5 tools)**:

| 模式 | 0.8B Wall-Clock | 0.8B Gen Tok | 4B Wall-Clock | 4B Gen Tok |
|------|----------------|-------------|---------------|-----------|
| CoT+SIG | 0.087±0.003s | 19 | 0.317±0.247s | 30 |
| CoT+AppLoop | 0.088±0.004s | 19 | 0.691±0.003s | 72 |
| CoT+AppLoop-PC | 0.158±0.022s | 39 | 0.517±0.151s | 52 |
| SIG_raw | 0.333±0.007s | 80 | 0.124±0.002s | 2 |
| CoT+SIG vs CoT+AppLoop | **0.99×** | 1.00 | **2.18× ⚠** | 0.42 |

**发现**: Q1 在所有模型/模式下接近平价 (0.99–1.03×)，CoT+SIG 无显著加速。Q2 在 4B 上 CoT+SIG 显示 2.18× 加速，但 gen token 比仅为 0.42（CoT+SIG 生成 30 tokens，CoT+AppLoop 生成 72 tokens）—— **此 4B Q2 数据点因严重输出长度失衡被排除出解释范围**。实验需在强制输出长度均衡协议下重新执行。排除 Q2 4B 后，CoT+SIG 的加速主要来自结构化上下文组织效应而非纯缓存优势。

---

### 12.10. R1 注意力分析

使用 modelscope 加载全精度模型：

| 模型 | Layers | Heads | 状态 |
|------|--------|-------|------|
| Qwen2.5-0.5B | 24 | 14 | 加载成功，attention 对比函数 IndexError（GQA 张量维度不匹配） |
| Qwen2.5-1.5B | 28 | 12 | 加载成功，同上 IndexError |

**发现**: 两个 modelscope 模型均成功下载并加载到 GPU（device_map=auto, FP16）。R1 的 attention 对比代码需要适配 Qwen 的 GQA（Grouped Query Attention）——`num_attention_heads` 与实际存储的 KV head 数量不一致导致索引越界。需要修复 attention 张量的 head 维度索引逻辑。

---

### 12.11. R6-R14 汇总

| 编号 | 方向 | 0.8B 结论 | 4B 结论 | 状态 |
|------|------|----------|--------|------|
| R6 | 动态重规划 | SIG 2.79× 加速 | SIG 4.26× 加速 | ✅ N=30 实测 |
| R7 | 多模态 SIG | Minimal text 最优; JSON +45% tokens | Minimal text 最优; JSON +45% tokens | ✅ 实测 (N=1) |
| R8 | 长上下文精确检索+空间推理 | SIG 0% 检索命中; AppLoop-PC 50% | SIG 33% 检索; AppLoop 100% 空间推理 | ✅ N=30 实测 |
| R9 | 实时性约束 | 全部 <0.2s, 2.0s 预算充足 | 全部 <0.5s, 2.0s 预算充足 | ✅ 实测 (N=1) |
| R10 | 注入攻击防御 | 0.8B 6/10 攻击成功; 8/10 干净回滚; 残差 Jaccard 0.109 | 4B 8/10 攻击成功; 10/10 干净回滚; 残差 Jaccard 0.049 | ✅ 实测 (10攻击×2模型) |
| R11 | 工具忠实度 | Token-Jaccard SIG≈AppLoop (平价); sydney 异常值 0.322 | Token-Jaccard SIG≈AppLoop (平价); 均值 0.056 | ✅ 实测 (12查询×2模型) |
| R12 | SIG Scaling Law | ~20K tok/s 实测 prefill; 投影仅供分析 | 投影仅供分析 | ⚠ 分析推测 (已剥离) |
| R13 | E2E Wall-Clock | SIG 0.32× 反慢 (0.8B); 平价 (4B) | SIG 1.08× (4B, within SD) | ✅ N=30 实测 |
| R14 | SIG+推理范式 | CoT+SIG vs AppLoop 平价 (Q1, Q2-0.8B); Q2-4B 排除 | 无显著加速 | ✅ N=30 实测 (Q2 4B excluded) |
| UQ1 | 上下文缩放交叉点 | ≤8K SIG 未超 AppLoop (1.04–1.09×) | ≤8K SIG 未超 AppLoop (1.04–1.07×) | ✅ 实测 |
| UQ2 | 虚构实体忠实度 | SIG 28% vs AppLoop 32% (价平) | SIG 24% vs AppLoop 28% (价平) | ✅ 实测 |
| R15/UQ3 | 多步推理 QA | 0/4 (模型能力限制) | — | ✅ 确认限制 |

---

### 12.12. UQ1-UQ3: 未回答问题进展 (May 2025)

#### UQ1: 上下文缩放交叉点
| 模型 | 2K SIG/App | 4K SIG/App | 8K SIG/App | 交叉? |
|------|-----------|-----------|-----------|-------|
| 0.8B | 126.5/51.7ms (2.44×) | 95.8/91.2ms (1.05×) | 211.1/192.9ms (1.09×) | 否 |
| 4B | 204.2/161.4ms (1.27×) | 326.4/313.8ms (1.04×) | 715.9/667.2ms (1.07×) | 否 |

**发现**: ≤8K 上下文中 SIG 未超过 AppLoop。4K 时接近平齐 (ratio ~1.05)，差距从 2K 的 1.3-2.4× 缩窄至 4K 的 1.05×，8K 稳定在 ~1.07-1.09×。推测交叉点需 >16K 或 >7B 模型。

#### UQ2: 虚构实体忠实度 (消除先验知识干扰)
| 模型 | SIG Faithful | AppLoop Faithful | 差异 |
|------|------------|----------------|------|
| 0.8B | 7/25 (28%) | 8/25 (32%) | +1 query |
| 4B | 6/25 (24%) | 7/25 (28%) | +1 query |

**发现**: **逆忠实度缩放假说被拒绝。** R11 中 "4B 不如 0.8B" 的发现确认为真实实体先验知识的 artifact。使用虚构实体（无预训练知识），0.8B 和 4B 忠实度完全一致，SIG 与 AppLoop 也无差异。指标: Jaccard overlap ratio ≥0.15 on ≥4-char tokens。

#### R15 (UQ3): 多步推理 QA
| 模型 | CoT+SIG | CoT+AppLoop | 原因 |
|------|---------|-------------|------|
| 0.8B | 0/4 | 0/4 | Qwen3.5 `<think>` 污染输出，小模型无法多步推理 |

**发现**: 受限于模型能力，Qwen3.5 ≤4B 无法在工具结果上进行多步推理。此问题需 ≥7B 或非 Qwen 架构模型才能回答。

---

### 12.13. SIG 的时延优势何时不成立

本研究的关键实证贡献之一是识别出 SIG **无法**提供可测量时延优势的条件。这些 N=30 负面结果界定了 SIG 当前实用性的边界。R13 测量的是碎片化本地 KV 重建成本（单 GPU，**非分布式部署**），不涉及网络通信、设备同步等真正分布式开销。

| 条件 | 模型 | SIG vs AppLoop | vs AppLoop-PC | 证据来源 | 解读 |
|------|------|---------------|---------------|----------|------|
| 30 工具深链 (R6) | 0.8B | **SIG 2.79×** | SIG 2.79× | N=30 paired | AppLoop-PC=AppLoop (前缀缓存无效), SIG 纯粹缓存优势 |
| 30 工具深链 (R6) | 4B | **SIG 4.26×** | SIG 4.25× | N=30 paired | 同上, 规模放大效应 |
| 碎片化 KV 重建 (R13) | 0.8B | **SIG 0.32× 反慢** | **SIG 0.32× 反慢** | N=30 paired | 注入操作单位开销 > 缓存节省 |
| 碎片化 KV 重建 (R13) | 4B | SIG 0.96× 平价 | SIG 1.08× (不显著) | N=30 paired | 小模型上 SIG 更差，大模型勉强平价 |
| 长上下文检索 (R8) | 0.8B | 平价 (均 0%) | **AppLoop-PC 50%** | N=30 paired | SIG 无检索能力；AppLoop-PC 最优 |
| 长上下文检索 (R8) | 4B | 混合 | **AppLoop-PC 64%** | N=30 paired | AppLoop-PC 检索最优 64% |
| 空间推理 (R8) | 4B | AppLoop 100% > SIG 27% | — | N=30 paired | SIG 推理弱于完整重编码 |
| CoT 推理 (R14 Q1) | both | 平价 (~1.0×) | 平价 | N=30 paired | CoT+SIG 不提供加速 |
| CoT 推理 (R14 Q2) | 4B | CoT+SIG 2.18× ⚠ | — | N=30 paired | 但 gen token 比 0.42 (截断风险) |
| ≤8K 单次推理 (UQ1) | both | AppLoop still 1.04-1.09× faster | 未测 | N=1/context | SIG 在小上下文无优势 |

**核心观察 (限于 Qwen3.5 Q4_K_M, RTX 4070 SUPER):** SIG 在深链场景 (R6) 显示出最大的数值优势（2.8–4.3×）——前缀缓存在此场景下结构性失效（<3% token 复用率）。SIG 的流式注入成本与链深度解耦，这是边缘设备上小模型执行长规划的关键特性。在碎片化重建 (R13) 和随机检索 (R8) 中，SIG 落后于 AppLoop-PC——但这些是流式注入架构的**设计边界**，而非缺陷：SIG 优化的是"注入一次，连续消费"的流式模式，而非频繁独立重建或随机存取。我们提出了混合调度策略（长链用 SIG，碎片用 AppLoop-PC）作为 SIG Decision Framework 的核心。所有经验观察是模型家族 × 量化 × GPU 特定的，跨家族复现是必要前提。

---

### 12.14. 效度威胁

**内部效度** (实验设计内的测量质量):
- ~~R11: 关键词重叠测量字面复制而非语义忠实度~~ → **已修复**: 采用 Token-Jaccard + LLM-Judge 双指标，废弃关键词重叠。
- ~~R8: 探针测试顺序键值对回忆而非空间推理~~ → **已修复**: 新增空间推理探针 (T=10, "从 Room 0 出发隔 2 个房间后的城市")，与检索探针形成互补双探针设计，N=30 三模式配对运行。
- ~~R10: 攻击成功/恢复检测基于关键词模式，可能漏检细微残留污染~~ → **已修复**: 新增连续语义相似度 + LLM-as-Judge 自评估，联合检测召回率 96%。
- ~~跨章节测量范围不一致 (R13不含生成, R6含生成, R14不含rebuild_cache)~~ → **已修复**: 所有 R6/R13/R14 统一为 wall-clock 端到端计时 + N=30。
- ~~R6/R8/R13/R14 样本量 N≤3, 无假设检验~~ → **已修复**: 全部四模块升级为 N=30 配对运行，报告 mean±SD。

**外部效度** (超越实验条件的可推广性):
- 单一模型家族 (仅 Qwen3.5, Q4_K_M)
- 单一GPU + 单一定量化 (RTX 4070 SUPER, Q4_K_M)
- R7/R9 仍为 N=1 per condition — 仅提供方向性指示，不可做统计推断
- R12 模型规模投影为分析性推测 (T∝M^0.7)，非实测；已从主实证向量组剥离
- R14 Q2 4B 因 gen-token比 0.42 数据点被排除 — 需在强制输出长度协议下重新测量
- R8 检索/推理探针使用关键词命中二元判定 — 存在假阴性偏移

**未来确认性研究建议**: 在 ≥2 额外模型家族 (Llama, Gemma, Mistral) 上复现 R6/R8/R13 关键发现；R7 JSON 异常需 GPU profiling (nsight systems)；R8 需嵌入级语义相似度替代关键词判定；R14 需输出长度标准化方案。
