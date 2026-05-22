# Cognitive Outsourcing (CO) — Code Wiki

> **项目名称**: Cognitive Outsourcing with Suspend-and-Inject Generation (SIG)
> **许可证**: MIT License (Copyright © 2026 MonSp)
> **论文**: *Cognitive Outsourcing with Suspend-and-Inject Generation for Scalable Embodied Intelligence*

---

## 目录

1. [项目概述](#1-项目概述)
2. [整体架构](#2-整体架构)
3. [文件结构与职责](#3-文件结构与职责)
4. [核心类详解](#4-核心类详解)
5. [关键函数说明](#5-关键函数说明)
6. [数据流与执行流程](#6-数据流与执行流程)
7. [基准测试场景](#7-基准测试场景)
8. [依赖关系](#8-依赖关系)
9. [项目运行方式](#9-项目运行方式)
10. [配置与扩展](#10-配置与扩展)

---

## 1. 项目概述

**Cognitive Outsourcing (CO)** 是一种边缘 AI 架构，使轻量级设备端语言模型（小至 0.8B 参数）能够通过一种新颖的 **Suspend-and-Inject Generation (SIG)** 原语，动态访问外部认知资源来编排复杂任务。

### 核心问题

传统工具调用循环（Tool-Calling Loop）在每次外部交互后，强制模型重新编码整个对话历史。这会导致：
- 丢弃模型内部注意力状态
- 产生二次方 prefill 开销
- 破坏认知连续性——对需要保持持续空间和任务感知的具身智能体而言是致命缺陷

### SIG 解决方案

SIG 允许运行中的模型：
1. **暂停**自回归解码
2. **调用**外部认知模块（云 LLM "教师"、感知 API、本地技能库）
3. **无缝吸收**其响应到模型的 KV 缓存中，**无需昂贵的重新编码**

本地模型成为隐私保护枢纽，按需召唤世界级专业知识，同时维持连续的注意力状态。

### 关键成果

| 指标 | 数值 |
|------|------|
| Prefill Token 减少 | 最高 96% |
| Prefill 时间节省 | 86% |
| 端到端加速 | 1.57×（0.8B 模型） |
| 长上下文信息覆盖提升 | 3× |
| GPU 显存占用 | < 1.5 GB |

---

## 2. 整体架构

CO 框架由三层组成：

```
┌─────────────────────────────────────────────────────────┐
│              Cognitive Module Ecosystem                  │
│   (Cloud Teacher LLMs, Perception APIs, Local Skills)   │
└──────────────────────────┬──────────────────────────────┘
                           │ Tool Results / Plans
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  Injection Engine (SIG)                   │
│   KV-cache continuity management · Result injection      │
│   Cache tracking · Rollback recovery                     │
└──────────────────────────┬──────────────────────────────┘
                           │ Token IDs
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   Meaning Compiler                       │
│   Lightweight local model · Intent parsing               │
│   Tokenization · Autoregressive generation               │
└─────────────────────────────────────────────────────────┘
```

### 两种执行模式对比

| 特性 | CO + AppLoop | CO + SIG |
|------|-------------|----------|
| Prefill 策略 | 每轮完整重新 prefill | 仅 prefill 增量 token |
| KV 缓存 | 每轮丢弃重建 | 持续维护 |
| 认知连续性 | 中断 | 保持 |
| Prefill 开销 | O(n²) 累积 | O(增量) |
| 适用场景 | 基线对照 | 生产推荐 |

---

## 3. 文件结构与职责

```
cognitive-outsourcing/
├── co_benchmark.py              # [核心] CO 基准测试：CO+AppLoop vs CO+SIG
├── sig_benchmark.py             # [核心] SIG 基准测试：AppLoop vs SIG（传统工具调用模式）
├── gen_plans.py                 # [工具] 调用云 LLM 生成教师计划
├── _gen_plans.py                # [内部] 从预计算数据导出计划 JSON
├── _gen_prompts.py              # [内部] 生成云教师提示词 JSON
├── co_benchmark_prompts.json    # [数据] 9 个场景的云教师提示词
├── co_benchmark_plans.json      # [数据] 9 个场景的预计算云教师计划
├── co_benchmark_plans copy.json # [备份] 计划文件备份
├── requirements.txt             # [配置] Python 依赖
├── README.md                    # [文档] 项目说明
├── LICENSE                      # [法律] MIT 许可证
├── Cognitive_Outsourcing_Paper.pdf  # [论文] 技术论文 PDF
└── Cognitive Outsourcing with Suspend-and-Inject Generation for Scalable Embodied Intelligence.md
                                # [论文] 技术论文 Markdown 版本
```

### 文件职责详解

| 文件 | 类型 | 职责 |
|------|------|------|
| `co_benchmark.py` | 核心代码 | CO 模式基准测试入口，包含 MeaningCompiler、InjectionEngine、ToolRegistry、CloudTeacherModule、COAppLoopAgent、COSIGAgent 等核心类，以及 9 个场景构建器和完整的评测流程 |
| `sig_benchmark.py` | 核心代码 | 传统 SIG vs AppLoop 基准测试，包含 EnhancedLlamaBench 类，模型自主生成工具调用（非预计算计划），测试 SIG 在自主推理链中的表现 |
| `gen_plans.py` | 辅助工具 | 通过 OpenAI 兼容 API 调用云 LLM，为 9 个场景生成 chain-of-thought 教师计划，输出到 `co_benchmark_plans.json` |
| `_gen_plans.py` | 内部脚本 | 从 `co_benchmark` 模块的 `PRECOMPUTED_PLANS` 导出计划 JSON（开发辅助用） |
| `_gen_prompts.py` | 内部脚本 | 从 `co_benchmark` 模块的场景构建器生成提示词 JSON（开发辅助用） |
| `co_benchmark_prompts.json` | 数据文件 | 包含 9 个场景的 system prompt 和 user message，用于调用云 LLM 生成计划 |
| `co_benchmark_plans.json` | 数据文件 | 包含 9 个场景的预计算 chain_of_thought 和 nodes，CO 基准测试直接加载使用 |

---

## 4. 核心类详解

### 4.1 `MeaningCompiler`（co_benchmark.py）

**职责**: 轻量级本地模型的封装层，负责 tokenization、生成、KV 缓存管理。

**核心思想**: 作为 CO 架构的第一层——"意义编译器"，将用户意图解析为可执行的 token 序列，并管理模型的注意力状态。

```python
class MeaningCompiler:
    TOOL_MARK = "<<<TOOL>>>"     # 工具调用起始标记
    TOOL_END = "<<</TOOL>>>"     # 工具调用结束标记
```

**关键方法**:

| 方法 | 签名 | 说明 |
|------|------|------|
| `__init__` | `(model_path, n_ctx=8192, n_threads=4, n_gpu_layers=0)` | 加载 GGUF 模型，初始化 llama.cpp 实例 |
| `tokenize` | `(text, add_bos=False) -> List[int]` | 文本转 token ID |
| `detokenize` | `(ids) -> str` | Token ID 转文本 |
| `reset_cache` | `() -> None` | 清空 KV 缓存（通过 `kv_cache_seq_rm`） |
| `eval` | `(tokens) -> None` | 对 token 序列执行前向传播（prefill） |
| `rebuild_cache` | `(token_ids) -> None` | 完全重建 KV 缓存（reset + eval） |
| `sample` | `(temp=0.0) -> int` | 采样下一个 token（支持贪心/温度采样） |
| `generate_until_ids` | `(stop_ids, max_new=300) -> Tuple[str, List[int]]` | 生成直到遇到指定 token ID 序列 |
| `generate_until_str` | `(stop_str, max_new=300) -> Tuple[str, List[int]]` | 生成直到遇到指定字符串 |
| `generate_until_any` | `(stop_strs, max_new=300) -> Tuple[str, List[int], Optional[str]]` | 生成直到遇到任一停止字符串 |
| `sanitize_generation` | `(n_before, gen_text, gen_ids, cached_prefix_ids) -> Tuple[str, List[int], bool]` | 清理异常生成（角色标签泄漏、重复），必要时回滚缓存 |

**内部机制**:
- 使用 `SEQ_ID = 0` 作为 KV 缓存序列标识
- `_ids_endswith`: 检查 token ID 序列是否以指定后缀结尾
- `_detect_repetition`: 检测退化重复模式（最小长度 6，阈值 3 次重复）

---

### 4.2 `InjectionEngine`（co_benchmark.py）

**职责**: SIG 运行时，管理 KV 缓存连续性和工具结果注入。

**核心思想**: 作为 CO 架构的第二层——"注入引擎"，确保工具结果无缝融入模型的注意力状态，避免完整重新编码。

```python
class InjectionEngine:
    def __init__(self, compiler: MeaningCompiler):
        self.compiler = compiler
        self.cached_ids: List[int] = []   # 追踪当前缓存中的所有 token ID
```

**关键方法**:

| 方法 | 签名 | 说明 |
|------|------|------|
| `make_result_block` | `(tool_name, tool_args, result) -> str` | 构造工具结果文本块，格式: `[Observation from tool(args)]: result` |
| `inject` | `(token_ids) -> Tuple[int, float]` | 将 token 注入 KV 缓存，返回 (token数, 耗时) |
| `inject_and_track` | `(token_ids, metrics, key_prefix) -> None` | 注入并同步更新性能指标和缓存追踪 |
| `update_cache` | `(new_ids) -> None` | 追加新 token ID 到缓存追踪列表 |
| `rollback` | `(target_ids) -> None` | 回滚缓存到指定状态（重建 KV 缓存） |
| `reset` | `() -> None` | 完全重置缓存和追踪 |

**设计要点**:
- `cached_ids` 维护了当前 KV 缓存中所有 token 的完整列表，用于回滚操作
- `inject_and_track` 是 SIG 模式的核心方法，同时处理注入和指标收集
- `rollback` 在生成异常时提供恢复机制

---

### 4.3 `ToolRegistry`（co_benchmark.py）

**职责**: 模拟工具执行环境，提供 6 种工具的确定性返回值。

**工具清单**:

| 工具名 | 参数 | 说明 | 数据覆盖 |
|--------|------|------|----------|
| `search_attractions` | `city` | 返回城市顶级景点 | 8 个城市 |
| `get_weather` | `city` | 返回当前天气 | 8 个城市 |
| `get_flight_info` | `origin, destination` | 返回航班信息 | 11 条航线 |
| `read_file` | `path/file` | 返回源代码文件内容 | 5 个文件 |
| `search_code` | `query` | 搜索代码库 | 8 个查询 |
| `run_test` | `test_name/name` | 运行测试套件 | 4 个测试集 |

**城市名规范化**: 通过 `CITY_ALIASES` 字典将常见别名映射为标准名称（如 "ny" → "newyork", "new york" → "newyork"）。

---

### 4.4 `CloudTeacherModule`（co_benchmark.py）

**职责**: 云端教师 LLM 接口，用于生成 chain-of-thought 工具调用计划。

**核心思想**: CO 架构的第三层——"认知模块生态"中的云教师组件。大型云 LLM 生成包含 `<<NODE:N>>` 标记的推理链，本地小模型只需执行而非规划。

**提示词模板**:

| 模板 | 用途 |
|------|------|
| `TEACHER_PLANNING_PROMPT` | 单查询工具链规划 |
| `TEACHER_CONVERSATION_PROMPT` | 多轮对话规划 |

**输出格式**:
```json
{
  "chain_of_thought": "推理文本...<<NODE:1>>...评估...<<NODE:2>>...",
  "nodes": {
    "1": {"tool": "tool_name", "arguments": {"param": "value"}},
    "2": {"tool": "tool_name", "arguments": {"param": "value"}}
  }
}
```

**关键方法**:

| 方法 | 签名 | 说明 |
|------|------|------|
| `plan_tool_chain` | `(query, tool_descriptions) -> Dict` | 为单查询生成工具链计划 |
| `plan_conversation` | `(turns, tool_descriptions) -> Dict` | 为多轮对话生成计划 |
| `_parse_cot_plan` | `(content) -> Dict` | 解析 LLM 返回的 JSON 计划（含容错） |

---

### 4.5 `COAppLoopAgent`（co_benchmark.py）

**职责**: CO + AppLoop 模式的智能体实现——每轮完整重新 prefill。

**执行流程**:
1. Phase-1: 云教师生成 chain-of-thought 计划
2. Phase-2: 组装工具结果到 CoT
3. 每轮：完整重新 prefill 整个对话历史 → 生成回答

**关键方法**:

| 方法 | 签名 | 说明 |
|------|------|------|
| `run_conversation` | `(turns, system_prompt, ...) -> Dict` | 执行多轮对话基准测试 |
| `run_complex_task` | `(user_query, expected_chain, ...) -> Dict` | 执行复杂任务基准测试 |
| `_full_prefill` | `(text, metrics) -> List[int]` | 完整重新 prefill（清空缓存 + 编码全部文本） |

**特点**:
- 每轮调用 `_full_prefill` 完全重建 KV 缓存
- 作为 SIG 的基线对照
- CoT 在第一个工具调用轮次注入

---

### 4.6 `COSIGAgent`（co_benchmark.py）

**职责**: CO + SIG 模式的智能体实现——KV 缓存连续性维护。

**执行流程**:
1. Phase-1: 云教师生成 chain-of-thought 计划（与 AppLoop 相同）
2. Phase-2: 组装工具结果到 CoT
3. 首轮：prefill 系统提示 → 后续轮次仅 prefill 增量 token
4. CoT 通过 `InjectionEngine.inject_and_track` 注入 KV 缓存
5. 最后一轮注入 `SIG_ANSWER_REMINDER` 提示模型综合回答

**关键方法**:

| 方法 | 签名 | 说明 |
|------|------|------|
| `run_conversation` | `(turns, system_prompt, ...) -> Dict` | 执行多轮对话基准测试（SIG 模式） |
| `run_complex_task` | `(user_query, expected_chain, ...) -> Dict` | 执行复杂任务基准测试（SIG 模式） |

**与 AppLoop 的关键差异**:
- 使用 `InjectionEngine` 管理 KV 缓存
- 仅 prefill 新增 token（用户输入 + CoT 块 + 工具结果）
- 最后一轮非工具轮次注入 `SIG_ANSWER_REMINDER`
- 使用 `generate_until_any` 支持多种停止条件

---

### 4.7 `EnhancedLlamaBench`（sig_benchmark.py）

**职责**: 增强版 SIG 基准测试类，模型自主生成工具调用（非预计算计划模式）。

**与 CO 基准测试的区别**:
- CO 模式：云教师预计算计划 → 本地模型执行
- SIG 模式：本地模型自主生成 `<<<TOOL>>>` 工具调用 → 应用层执行 → 注入结果

**关键方法**:

| 方法 | 签名 | 说明 |
|------|------|------|
| `run_complex_task` | `(mode, user_query, expected_chain, ...) -> Dict` | 运行复杂任务（支持 app_loop/sig 模式） |
| `run_mode_on_turns` | `(mode, turns, ...) -> Dict` | 运行多轮对话 |
| `_parse_tool_call` | `(text) -> Tuple[Optional[str], Optional[Dict]]` | 从生成文本中解析工具调用 JSON |
| `make_tool_result_block` | `(tool_name, tool_args, result) -> str` | 构造工具结果块 |

**SIG 特有机制**:
- **合成注入 (Synthetic Injection)**: 当模型未能生成工具调用时，SIG 模式可合成 `<<<TOOL>>>` 块注入缓存，确保工具链继续执行
- **回滚恢复 (Rollback)**: 检测到异常生成（角色标签泄漏、重复）时，回滚 KV 缓存到安全状态
- **结构性隔离 (Structural Isolation)**: 可选的 `<<<SAFE>>>` 标记包裹工具结果，防止提示注入

---

### 4.8 `GPUMonitor`（co_benchmark.py / sig_benchmark.py）

**职责**: 通过 NVML 监控 GPU 显存使用情况。

| 方法 | 说明 |
|------|------|
| `__init__` | 初始化 NVML，记录基线显存 |
| `snapshot` | 返回当前显存使用量和增量 |
| `shutdown` | 关闭 NVML |

**容错设计**: pynvml 不可用时自动降级，所有方法返回零值。

---

## 5. 关键函数说明

### 5.1 场景构建函数（co_benchmark.py）

| 函数 | 场景 | 返回类型 | 说明 |
|------|------|----------|------|
| `build_scenario1_long_sequence` | 1 | `List[Dict]` | 长序列压力测试：22 轮循环查询 6 个城市 |
| `build_scenario2_multi_tool_chain` | 2 | `List[Dict]` | 多工具链：4 次工具调用 + 1 次总结 |
| `build_scenario3_rapid_fire` | 3 | `List[Dict]` | 快速短查询：12 个独立查询 |
| `build_scenario4_long_document` | 4 | `Tuple[str, List[Dict]]` | 长文档 + 工具调用：大背景文本 + 4 次工具调用 |
| `build_scenario5_mixed_conversation` | 5 | `List[Dict]` | 混合对话：工具调用与闲聊交替 |
| `build_scenario6_deep_tool_chain` | 6 | `List[Dict]` | 深度工具链：15 轮 14 次工具调用（5 城市） |
| `build_scenario7_travel_planning_chain` | 7 | `List[Dict]` | 旅行规划：12 轮 11 次工具调用（4 城市） |
| `build_scenario8_code_debugging_chain` | 8 | `List[Dict]` | 代码调试：5 轮 4 次工具调用 |
| `build_scenario9_cross_reference_chain` | 9 | `List[Dict]` | 交叉引用：10 轮 9 次工具调用（3 城市对比） |

### 5.2 核心辅助函数

| 函数 | 文件 | 说明 |
|------|------|------|
| `normalize_city` | 两者 | 城市名规范化（别名映射 + 空格移除） |
| `assemble_chain_of_thought` | co_benchmark.py | CO 核心组装函数：将 `<<NODE:N>>` 替换为工具结果 |
| `_init_metrics` | co_benchmark.py | 初始化性能指标字典 |
| `_make_observation_block` | co_benchmark.py | 构造观察文本块 |
| `_load_precomputed_plans` | co_benchmark.py | 加载预计算教师计划 JSON |
| `average_metrics` | 两者 | 对多次运行指标取平均 |
| `evaluate_answer_quality` | co_benchmark.py | 评估回答质量（信息覆盖率） |
| `_extract_key_facts` | co_benchmark.py | 从工具结果中提取关键事实 |
| `parse_cot_plan` | gen_plans.py | 解析云 LLM 返回的 CoT 计划 |
| `call_llm` | gen_plans.py | 调用 OpenAI 兼容 API |
| `execute_tool` | sig_benchmark.py | 执行工具调用（独立函数版） |

### 5.3 `assemble_chain_of_thought` 详解

这是 CO 模式的核心组装函数：

```python
def assemble_chain_of_thought(cot, nodes, module, expected_chain, metrics, debug=True):
    """
    1. 在教师的 chain-of-thought 中查找所有 <<NODE:N>> 标记
    2. 对每个节点，调用对应工具获取结果
    3. 将 <<NODE:N>> 替换为工具结果
    4. 返回组装后的 CoT 和匹配计数
    """
```

**匹配逻辑**: 使用 `normalize_city` 规范化参数后比较工具名和参数，确保 "New York" 与 "newyork" 正确匹配。

---

## 6. 数据流与执行流程

### 6.1 CO + SIG 模式执行流程（多轮对话）

```
┌──────────────────────────────────────────────────────────────┐
│ Phase 1: Cloud Teacher Planning                              │
│                                                              │
│  多轮对话 ──→ CloudTeacherModule.plan_conversation()         │
│              ──→ 云 LLM 返回 {chain_of_thought, nodes}       │
│              ──→ 或加载 precomputed_plan                      │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ Phase 2: Assembly                                            │
│                                                              │
│  assemble_chain_of_thought()                                 │
│  ──→ 遍历 <<NODE:N>> 标记                                    │
│  ──→ ToolRegistry.execute() 获取工具结果                      │
│  ──→ 替换标记为结果文本                                       │
│  ──→ 输出: assembled_cot                                     │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ Phase 3: SIG Execution (per turn)                            │
│                                                              │
│  Turn 0: prefill(system_prompt) ──→ update_cache             │
│                                                              │
│  For each turn:                                              │
│    ├─ prefill("User: ... Assistant:") ──→ update_cache       │
│    ├─ if 首个工具轮:                                          │
│    │    inject(assembled_cot + "Answer:") ──→ generate       │
│    ├─ elif 非工具轮:                                          │
│    │    generate_until_any(...)                               │
│    └─ update_cache(gen_ids)                                   │
│                                                              │
│  最后一轮非工具轮:                                            │
│    inject(SIG_ANSWER_REMINDER) ──→ generate                  │
└──────────────────────────────────────────────────────────────┘
```

### 6.2 CO + AppLoop 模式执行流程（多轮对话）

```
┌──────────────────────────────────────────────────────────────┐
│ Phase 1 & 2: 同 SIG 模式                                     │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ Phase 3: AppLoop Execution (per turn)                        │
│                                                              │
│  Turn 0: full_prefill(system_prompt) ──→ 重置缓存            │
│                                                              │
│  For each turn:                                              │
│    ├─ history += "User: ... Assistant:"                      │
│    ├─ if 首个工具轮: history += assembled_cot + "Answer:"     │
│    ├─ full_prefill(history) ──→ 完全重建缓存                  │
│    ├─ generate_until_str("\nUser:")                           │
│    └─ history += gen_text                                    │
└──────────────────────────────────────────────────────────────┘
```

### 6.3 SIG 自主模式执行流程（sig_benchmark.py）

```
┌──────────────────────────────────────────────────────────────┐
│ 模型自主工具调用模式                                          │
│                                                              │
│  Init: prefill(system_prompt + "User: query Assistant:")     │
│                                                              │
│  Loop (max_steps):                                           │
│    ├─ AppLoop: full_prefill(history) ──→ 重建缓存            │
│    ├─ SIG: 仅 prefill 增量 token                             │
│    ├─ generate_until_ids(<<</TOOL>>>)                        │
│    ├─ _parse_tool_call(gen_text)                             │
│    ├─ if 工具调用:                                           │
│    │    ├─ execute_tool() ──→ make_result_block()            │
│    │    ├─ AppLoop: 追加到 history，下轮 full_prefill        │
│    │    └─ SIG: inject(result_ids) ──→ update_cache          │
│    ├─ if 无工具调用 (SIG + rollback):                        │
│    │    └─ synthetic_injection: 合成工具调用注入缓存          │
│    └─ if 连续无工具 或 链完成: break                          │
└──────────────────────────────────────────────────────────────┘
```

---

## 7. 基准测试场景

### 场景总览

| # | 名称 | 类型 | 轮次 | 工具调用数 | 领域 | 测试重点 |
|---|------|------|------|-----------|------|----------|
| 1 | Long-sequence | 多轮 | 22 | 22 | 旅行 | 累积 prefill 优势 |
| 2 | Multi-tool chain | 多轮 | 5 | 4 | 旅行 | 复杂推理链 |
| 3 | Rapid-fire | 多轮 | 12 | 12 | 旅行 | 短查询开销 |
| 4 | Long-document | 多轮 | 5 | 4 | 旅行 | 长前缀重 prefill 代价 |
| 5 | Mixed conversation | 多轮 | 8 | 4 | 旅行 | 工具+闲聊混合 |
| 6 | Deep tool chain | 多轮 | 15 | 14 | 旅行 | 持续链式调用 |
| 7 | Travel planning | 多轮 | 12 | 11 | 旅行 | 多城市规划 |
| 8 | Code debugging | 多轮 | 5 | 4 | 开发 | 编程智能体 |
| 9 | Cross-reference | 多轮 | 10 | 9 | 旅行 | 跨结果引用 |

### 场景分类

**多轮对话模式（Scenario 1-6）**: 每轮一个用户输入，模型响应后进入下一轮。测试 SIG 在多轮累积中的 prefill 节省。

**复杂任务模式（Scenario 7-9）**: 单个复杂查询，模型需要自主链式调用多个工具。测试 SIG 在不间断推理链中的优势。

### 评测指标

| 指标 | 说明 |
|------|------|
| `total_ttf` | 总时间到首 token |
| `total_gen_time` | 总生成时间 |
| `total_prefill_time` | 总 prefill 时间 |
| `total_prefill_tokens` | 总 prefill token 数 |
| `total_gen_tokens` | 总生成 token 数 |
| `tool_calls_ok` | 正确工具调用数 |
| `total_tool_calls` | 总工具调用数 |
| `chain_depth` | 匹配的预期工具链深度 |
| `chain_total` | 预期工具链总深度 |
| `peak_gpu_delta` | 峰值 GPU 显存增量 |
| `rollback_count` | SIG 回滚次数 |
| `coverage` | 回答信息覆盖率 |

---

## 8. 依赖关系

### 8.1 外部依赖

| 包 | 版本约束 | 用途 | 必需 |
|----|----------|------|------|
| `llama-cpp-python` | >=0.2.80, <0.3.0 | GGUF 模型加载与推理（llama.cpp Python 绑定） | 是 |
| `pynvml` | >=8.0.0 | GPU 显存监控（NVML Python 绑定） | 否（降级运行） |
| `requests` | >=2.31.0 | HTTP 请求（云教师 API 调用） | 否（CO 基准可离线运行） |

### 8.2 模块间依赖图

```
co_benchmark.py
├── llama_cpp.Llama          (模型推理)
├── pynvml                   (GPU 监控，可选)
├── requests                 (云教师 API，可选)
├── co_benchmark_plans.json  (预计算计划数据)
└── 内部类依赖:
    MeaningCompiler ← InjectionEngine
    MeaningCompiler ← COAppLoopAgent
    MeaningCompiler + InjectionEngine ← COSIGAgent
    ToolRegistry ← COAppLoopAgent / COSIGAgent
    CloudTeacherModule ← COAppLoopAgent / COSIGAgent (可选)

sig_benchmark.py
├── llama_cpp.Llama          (模型推理)
├── pynvml                   (GPU 监控，可选)
└── 内部类依赖:
    EnhancedLlamaBench (自包含，包含所有逻辑)

gen_plans.py
├── requests                 (调用云 LLM API)
├── co_benchmark_prompts.json (读取提示词)
└── → co_benchmark_plans.json (写入计划)

_gen_prompts.py
└── co_benchmark (导入场景构建器和提示词模板)

_gen_plans.py
└── co_benchmark (导入 PRECOMPUTED_PLANS)
```

### 8.3 关键外部依赖说明

**llama-cpp-python**: 项目基于 llama.cpp 构建，使用其 Python 绑定进行模型推理。关键 API 使用：
- `Llama()`: 模型加载
- `llm.tokenize()`: 文本编码
- `llm.detokenize()`: token 解码
- `llm.eval()`: 前向传播（prefill）
- `llm.sample()`: token 采样
- `llm._ctx.kv_cache_seq_rm()`: KV 缓存序列删除（SIG 核心操作）

---

## 9. 项目运行方式

### 9.1 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 准备 GGUF 模型文件（如 Qwen2.5-0.8B-Instruct 等）
# 下载地址示例: https://huggingface.co/
```

### 9.2 生成云教师计划（可选，已有预计算数据）

```bash
# 使用 Ollama 本地模型
python gen_plans.py --api-base http://localhost:11434/v1 --model gpt-4o-mini

# 使用 OpenAI API
python gen_plans.py --api-base https://api.openai.com/v1 --model gpt-4o-mini --api-key sk-xxx

# 仅运行特定场景
python gen_plans.py --api-base ... --model ... --only 1,3,7

# 跳过特定场景
python gen_plans.py --api-base ... --model ... --skip 3,5
```

### 9.3 运行 CO 基准测试

```bash
# 基本运行（CPU）
python co_benchmark.py --model /path/to/model.gguf

# GPU 加速
python co_benchmark.py --model /path/to/model.gguf --n-gpu-layers 99

# 自定义参数
python co_benchmark.py \
  --model /path/to/model.gguf \
  --n-ctx 16384 \          # 上下文窗口大小
  --n-threads 4 \          # CPU 线程数
  --n-gpu-layers 0 \       # GPU 层数（0=纯CPU）
  --runs 10 \              # 每模式每场景运行次数
  --long-turns 22 \        # 场景1轮次数
  --rapid-queries 12 \     # 场景3查询数
  --max-new 600 \          # 最大生成 token 数
  --max-new-tool 300 \     # 工具轮最大生成 token 数
  --max-new-tool-sig 150 \ # SIG 工具轮最大生成 token 数
  --skip 3,5 \             # 跳过场景
  --no-debug               # 关闭调试输出
```

### 9.4 运行 SIG 基准测试

```bash
# 基本运行
python sig_benchmark.py --model /path/to/model.gguf

# GPU 加速
python sig_benchmark.py --model /path/to/model.gguf --n-gpu-layers 99

# 自定义参数
python sig_benchmark.py \
  --model /path/to/model.gguf \
  --n-ctx 16384 \
  --n-threads 4 \
  --n-gpu-layers 0 \
  --runs 10 \
  --long-turns 22 \
  --rapid-queries 12 \
  --skip 3,5 \
  --no-debug
```

### 9.5 输出说明

两个基准测试均输出以下维度的对比表：

1. **Total Time Breakdown**: 生成时间 vs Prefill 时间
2. **Prefill Time Comparison**: AppLoop vs SIG prefill 时间及节省百分比
3. **Prefill Token Comparison**: AppLoop vs SIG prefill token 数及节省百分比
4. **End-to-End Total Time**: 端到端总时间及归一化加速比
5. **Generation Time & Tokens**: 生成时间和 token 数
6. **Answer Quality**: 信息覆盖率评估
7. **Tool vs Chat Turn TTF**: 工具轮 vs 闲聊轮延迟
8. **Peak GPU Delta**: 峰值 GPU 显存增量

---

## 10. 配置与扩展

### 10.1 系统提示词

项目内置两套系统提示词：

| 常量 | 内容 | 用途 |
|------|------|------|
| `SYSTEM_PROMPT` | 旅行助手 | 场景 1-7, 9 |
| `SYSTEM_PROMPT_DEV` | 软件开发专家 | 场景 8 |

### 10.2 工具描述

| 常量 | 工具集 | 用途 |
|------|--------|------|
| `TOOL_DESCRIPTIONS_TRAVEL` | search_attractions, get_weather, get_flight_info | 旅行场景 |
| `TOOL_DESCRIPTIONS_DEV` | run_test, read_file, search_code | 开发场景 |

### 10.3 扩展新场景

1. 在 `co_benchmark.py` 中添加 `build_scenarioN_xxx()` 函数，返回 `List[Dict]`（每项含 `user`, `tool`, `tool_args`）
2. 在 `main()` 中添加场景运行逻辑
3. 运行 `_gen_prompts.py` 生成提示词
4. 运行 `gen_plans.py` 生成预计算计划
5. 在 `sig_benchmark.py` 的 `EnhancedLlamaBench` 中添加对应场景构建方法

### 10.4 扩展新工具

1. 在 `ToolRegistry.tools` 字典中添加工具名和数据
2. 在 `ToolRegistry.execute()` 方法中添加工具执行逻辑
3. 更新 `TOOL_DESCRIPTIONS_TRAVEL` 或 `TOOL_DESCRIPTIONS_DEV`
4. 同步更新 `sig_benchmark.py` 中的 `TOOL_REGISTRY` 和 `execute_tool()`

### 10.5 使用不同云 LLM

`CloudTeacherModule` 和 `gen_plans.py` 均支持任何 OpenAI 兼容 API：

```python
# Ollama
teacher = CloudTeacherModule(api_base="http://localhost:11434/v1", model="llama3")

# OpenAI
teacher = CloudTeacherModule(api_base="https://api.openai.com/v1", model="gpt-4o", api_key="sk-xxx")

# vLLM
teacher = CloudTeacherModule(api_base="http://localhost:8000/v1", model="meta-llama/Meta-Llama-3-8B")
```

---

## 附录: 关键常量与全局变量

| 名称 | 值 | 说明 |
|------|----|------|
| `SEQ_ID` | `0` | KV 缓存序列标识符 |
| `CITY_ALIASES` | 字典 | 城市名别名映射 |
| `SIG_ANSWER_REMINDER` | `"\nBased on all the observations above..."` | SIG 最后一轮回答提示 |
| `LOCAL_CO_PROMPT` | 模板字符串 | CO 本地模型提示词模板 |
| `NODE_PATTERN` | `re.compile(r'<<NODE:(\d+)>>')` | CoT 节点标记正则 |
| `LONG_TRAVEL_GUIDE` | 长文本 | 场景 4 背景参考文档 |
