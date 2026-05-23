# SIG / CO 后续研究方向

> 本文档记录 Cognitive Outsourcing 与 Suspend-and-Inject Generation 的后续研究路线。
> 每个方向对应一个独立的 git worktree（分支），便于并行推进。

---

## 一、SIG 机制基础理论

### R1: 注入对注意力状态的信息论分析

- **分支**: `R1-injection-information-theory`
- **Worktree**: `d:\trunk\SIG\output\cognitive-outsourcing-R1`
- **核心问题**: 注入 vs 完整重编码之间，是否存在可证明的信息损失上界？能否用互信息（Mutual Information）量化"认知连续性"？
- **研究内容**:
  - 信息论框架下 SIG 注入的信息保留量分析
  - 注意力分布偏移的层间敏感性分析（哪些层对注入最敏感？是否存在"注入临界层"？）
  - 注入粒度研究：整段批量注入 vs 逐步渐进注入的效果对比
- **难度**: ⭐⭐⭐
- **状态**: 🔧 开发中
- **实现文件**:
  - `r1_info_theory.py` — 信息论度量核心（KL散度、JS散度、互信息估计、熵、信息保留比）
  - `r1_probe.py` — 扩展MeaningCompiler捕获logits和KV-cache状态的探针模块
  - `r1_attention_analyzer.py` — 基于HuggingFace的注意力层分析模块
  - `r1_benchmark.py` — R1主实验框架（4组实验）
  - `research/info_analysis.py` — 信息分析工具函数

### R2: KV 缓存生命周期与退化机制

- **分支**: `R2-kv-cache-degradation`
- **Worktree**: `d:\trunk\SIG\output\cognitive-outsourcing-R2`
- **核心问题**: KV 缓存在多轮注入后是否必然退化？退化规律是什么？
- **研究内容**:
  - 缓存退化曲线：随注入轮次增加，早期信息的"信噪比"变化规律
  - 选择性缓存淘汰策略（LRU、重要性评分），在有限显存下保留最关键的注意力状态
  - 缓存压缩：对已缓存 KV 对进行低秩近似或量化，在保持生成质量前提下降低显存
- **难度**: ⭐⭐⭐
- **状态**: 🔧 开发中

### R3: SIG 与非 Transformer 架构的兼容性

- **分支**: `R3-sig-beyond-transformer`
- **Worktree**: `d:\trunk\SIG\output\cognitive-outsourcing-R3`
- **核心问题**: 非 Transformer 架构（Mamba/SSM、RWKV、xLSTM）能否实现 SIG？
- **研究内容**:
  - SSM 隐状态的 Suspend-and-Inject 可行性分析
  - 不同架构下 SIG 的信息保留能力差异
  - 混合架构（如 Jamba = Transformer + Mamba）中 SIG 的作用层选择
- **难度**: ⭐⭐⭐
- **状态**: 🔧 开发中
- **已实现模块**:
  - `r3_core.py` — 架构状态模型抽象基类 + 信息论度量（fidelity, retention, capacity）
  - `r3_ssm.py` — SSM/Mamba SIG 适配器（HiPPO 矩阵、离散化、选择性扫描注入）
  - `r3_rwkv.py` — RWKV SIG 适配器（WKV 状态动力学、时间衰减注入）
  - `r3_xlstm.py` — xLSTM SIG 适配器（sLSTM 指数门控 + mLSTM 矩阵记忆、rank-1 注入）
  - `r3_hybrid.py` — 混合架构分析器（Jamba/Griffin/Zamba 配置、注入层选择器）
  - `r3_benchmark.py` — 跨架构基准测试（9 项 benchmark，含延迟/容量/衰减分析）
- **初步发现**:
  - 🥇 Transformer: 原生 SIG 支持（append-only KV cache），最高信息保留
  - 🥈 xLSTM: 良好 SIG 支持（加性 cell state + rank-1 矩阵更新）
  - 🥉 RWKV: 中等 SIG 支持（时间衰减加权注入，状态极紧凑）
  - 4️⃣ SSM: SIG 具有挑战性（固定维度状态瓶颈，压缩比随序列长度恶化）

---

## 二、CO 架构深化

### R4: 教师模型最优选择与知识蒸馏

- **分支**: `R4-teacher-student-distillation`
- **Worktree**: `d:\trunk\SIG\output\cognitive-outsourcing-R4`
- **核心问题**: 教师-学生能力差距的最优匹配关系是什么？
- **研究内容**:
  - 教师-学生能力匹配的理论框架：教师太强（GPT-4）vs 学生太弱（0.8B），推理链可能无法被学生理解
  - 推理链复杂度自适应控制：根据学生模型容量调整 CoT 抽象层级和详细程度
  - 多教师协同：不同领域教师的动态选择机制，多教师 CoT 融合策略
- **难度**: ⭐⭐
- **状态**: 🔧 开发中
- **实现文件**:
  - `r4_benchmark.py` — R4 基准测试主程序（含三组实验）
  - `r4_gen_plans.py` — 多教师等级预计算计划生成器

### R5: 认知外包的隐私边界

- **分支**: `R5-privacy-guarantees`
- **Worktree**: `d:\trunk\SIG\output\cognitive-outsourcing-R5`
- **核心问题**: 认知外包的隐私泄漏能否形式化量化？
- **研究内容**:
  - 差分隐私（Differential Privacy）或信息论方法量化"发送给云教师的信息泄露量"
  - 教师推理链中的隐私泄漏检测与过滤
  - 本地化教师方案：在本地运行中等规模模型（3B-7B）作为教师，完全避免云端通信
- **难度**: ⭐⭐⭐
- **状态**: 🔧 开发中
- **实现文件**:
  - `r5_privacy.py` — 核心模块（PrivacyQuantifier, PrivacyFilter, LocalTeacherModule, PrivacyAwareCOAgent）
  - `r5_benchmark.py` — 隐私基准测试（PII检测、过滤效果、隐私量化、信息论分析）

### R6: 从静态规划到动态重规划

- **分支**: `R6-dynamic-replanning`
- **Worktree**: `d:\trunk\SIG\output\cognitive-outsourcing-R6`
- **核心问题**: CO 能否从"一次规划，全程执行"演进为在线动态调整？
- **研究内容**:
  - 执行时发现计划不充分的在线检测机制
  - 计划失败恢复：工具调用失败或返回意外结果时的在线调整策略
  - 交互式 CO：学生-教师多轮协商机制，学生执行中遇困时暂停请求教师补充指导
- **难度**: ⭐⭐
- **状态**: ✅ 已完成（仿真框架）
- **实现文件**:
  - `r6_dynamic_replanning.py` — 失败模拟、恢复策略分析、交互式CO协商、成本收益分析

---

## 三、具身智能场景

### R7: 多模态 SIG

- **分支**: `R7-multimodal-sig`
- **Worktree**: `d:\trunk\SIG\output\cognitive-outsourcing-R7`
- **核心问题**: 多模态特征能否直接注入 KV 缓存？
- **研究内容**:
  - 视觉注入：将感知模块输出的视觉特征（而非文本描述）直接注入 KV 缓存
  - 跨模态对齐：视觉 token 和语言 token 在 KV 缓存中的对齐方式
  - 传感器流式注入：持续将感知信息注入缓存而不中断推理的"流式 SIG"
- **难度**: ⭐⭐⭐
- **状态**: ✅ 已完成（仿真框架）
- **实现文件**:
  - `r7_multimodal_sig.py` — 视觉/音频/传感器注入仿真、跨模态对齐、模态比例分析

### R8: 空间认知与持续注意

- **分支**: `R8-spatial-cognition`
- **Worktree**: `d:\trunk\SIG\output\cognitive-outsourcing-R8`
- **核心问题**: SIG 是否真的比 AppLoop 更好地保持空间认知和长时程任务记忆？
- **研究内容**:
  - 空间记忆基准：需要维护空间地图的多轮任务（多房间导航并记住物体位置）
  - 长时程任务：从 22 轮扩展到数百轮，SIG 在超长交互中的认知保持能力
  - 任务切换与恢复：智能体被中断后恢复任务时，SIG vs AppLoop 的上下文恢复对比
- **难度**: ⭐⭐
- **状态**: ✅ 已完成（仿真框架）
- **实现文件**:
  - `r8_spatial_cognition.py` — 2D空间网格导航、空间记忆探测、长时程任务、任务切换恢复

### R9: 实时性约束下的 SIG

- **分支**: `R9-realtime-sig`
- **Worktree**: `d:\trunk\SIG\output\cognitive-outsourcing-R9`
- **核心问题**: 在固定延迟预算下，如何最优分配教师规划时间和本地执行时间？
- **研究内容**:
  - 延迟预算分配：教师规划时间 vs 本地执行时间的最优权衡
  - 预测性注入：在请求发出前预判需要的工具调用并提前注入结果
  - SIG 与推测执行（Speculative Decoding）的协同优化
- **难度**: ⭐⭐
- **状态**: ✅ 已完成（仿真框架）
- **实现文件**:
  - `r9_realtime_sig.py` — 延迟预算优化器、预测性注入、推测解码SIG协同

---
## 四、鲁棒性与安全性

### R10: 注入攻击与防御

- **分支**: `R10-injection-attacks`
- **Worktree**: `d:\trunk\SIG\output\cognitive-outsourcing-R10`
- **核心问题**: SIG 注入是否比 AppLoop 更容易受到工具结果投毒攻击？
- **研究内容**:
  - 工具结果投毒：恶意工具返回精心构造的结果，通过 SIG 注入后对模型行为的影响
  - 缓存污染传播：一次恶意注入对后续所有轮次的影响范围（"传播半径"）
  - 防御机制：基于注意力权重的异常检测——监控注入后注意力分布是否异常偏移
- **难度**: ⭐⭐
- **状态**: ✅ 已完成（仿真框架）
- **实现文件**:
  - `r10_injection_attacks.py` — 5种攻击向量模拟、污染传播分析、注意力异常检测、5种防御策略矩阵

### R11: 事实性与幻觉

- **分支**: `R11-factuality-hallucination`
- **Worktree**: `d:\trunk\SIG\output\cognitive-outsourcing-R11`
- **核心问题**: SIG 注入的工具结果是否会被模型"忠实"引用？覆盖率提升是否以牺牲准确性为代价？
- **研究内容**:
  - 注入信息的忠实度：SIG vs AppLoop 下工具结果利用率和幻觉率对比
  - 信息覆盖 vs 准确性权衡：3× 覆盖率提升背后的准确性代价
  - 冲突信息处理：多个工具返回矛盾信息时，SIG 连续注意力 vs AppLoop 重编码的处理差异
- **难度**: ⭐⭐
- **状态**: ✅ 已完成（仿真框架）
- **实现文件**:
  - `r11_factuality.py` — 引用忠实度检测、覆盖-准确性权衡分析、冲突消解、幻觉率对比

---
## 五、规模化与泛化

### R12: SIG 的 Scaling Law

- **分支**: `R12-sig-scaling-law`
- **Worktree**: `d:\trunk\SIG\output\cognitive-outsourcing-R12`
- **核心问题**: SIG 的收益随模型规模、上下文长度、工具链深度如何变化？
- **研究内容**:
  - 模型规模 vs SIG 收益：0.8B → 3B → 7B → 13B，收益递减拐点
  - 上下文长度 vs SIG 收益：16K → 32K → 128K → 1M，prefill 节省增长曲线
  - 工具链深度 vs SIG 收益：14 步 → 50 步 → 100 步，极端链式调用下的优势变化
- **难度**: ⭐⭐
- **状态**: ✅ 已完成（仿真框架）
- **实现文件**:
  - `r12_scaling_law.py` — 解析缩放模型、4B/7B/13B速度预测、上下文长度vs工具深度分析

### R13: 分布式认知外包

- **分支**: `R13-distributed-co`
- **Worktree**: `d:\trunk\SIG\output\cognitive-outsourcing-R13`
- **核心问题**: 多边缘设备能否通过 SIG 共享 KV 缓存片段实现分布式推理？
- **研究内容**:
  - 多边缘设备协作：多个小模型设备通过 SIG 共享 KV 缓存片段
  - 层级 CO：边缘(0.8B) → 本地服务器(7B) → 云端(70B) 的三级认知外包策略
  - 联邦 SIG：联邦学习场景下各客户端 KV 缓存的聚合与有效性验证
- **难度**: ⭐⭐⭐
- **状态**: ✅ 已完成（仿真框架）
- **实现文件**:
  - `r13_distributed_co.py` — KV缓存片段共享、3层层级CO路由、联邦SIG聚合、可扩展性分析

### R14: SIG 与新兴推理范式

- **分支**: `R14-sig-reasoning-paradigms`
- **Worktree**: `d:\trunk\SIG\output\cognitive-outsourcing-R14`
- **核心问题**: SIG 能否与 Chain-of-Thought、Tree-of-Thought 等推理范式结合？
- **研究内容**:
  - SIG + 自主 CoT：本地模型自主生成 CoT 并通过 SIG 注入中间推理结果
  - SIG + Tree-of-Thought：搜索树推理中缓存公共前缀，仅注入不同分支差异
  - SIG + Tool Learning：通过 SIG 持续学习新工具使用模式，实现在线工具学习
- **难度**: ⭐⭐⭐
- **状态**: ✅ 已完成（仿真框架）
- **实现文件**:
  - `r14_reasoning_paradigms.py` — CoT-SIG集成、ToT-SIG前缀缓存、在线工具学习、统一范式对比

---
## 优先级总览

| 编号 | 方向 | 难度 | 创新性 | 影响力 | 推荐优先级 |
|------|------|------|--------|--------|-----------|
| R1 | 注入信息论分析 | ⭐⭐⭐ | 极高 | 极高 | 🥇 最高 |
| R7 | 多模态 SIG | ⭐⭐⭐ | 极高 | 极高 | 🥇 最高 |
| R2 | KV 缓存退化 | ⭐⭐⭐ | 高 | 高 | 🥈 高 |
| R5 | 隐私边界 | ⭐⭐⭐ | 高 | 高 | 🥈 高 |
| R13 | 分布式 CO | ⭐⭐⭐ | 高 | 高 | 🥈 高 |
| R14 | SIG+推理范式 | ⭐⭐⭐ | 高 | 高 | 🥈 高 |
| R3 | 非Transformer兼容 | ⭐⭐⭐ | 高 | 中 | 🥉 中 |
| R4 | 教师最优选择 | ⭐⭐ | 中 | 中 | 🥉 中 |
| R6 | 动态重规划 | ⭐⭐ | 中 | 中 | 🥉 中 |
| R8 | 空间认知 | ⭐⭐ | 中 | 中 | 🥉 中 |
| R9 | 实时性约束 | ⭐⭐ | 中 | 中 | 🥉 中 |
| R10 | 注入攻击防御 | ⭐⭐ | 中 | 中 | 🥉 中 |
| R11 | 事实性与幻觉 | ⭐⭐ | 中 | 中 | 🥉 中 |
| R12 | SIG Scaling Law | ⭐⭐ | 中 | 中 | 🥉 中 |
