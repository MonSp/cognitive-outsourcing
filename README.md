# Cognitive Outsourcing (CO)

**Cognitive Outsourcing (CO)** is an edge-AI architecture that empowers lightweight on-device language models—as small as 0.8B parameters—to orchestrate complex tasks by dynamically accessing external cognitive resources through a novel **Suspend-and-Inject Generation (SIG)** primitive.

## The Problem

Traditional tool-calling loops force models to re-encode the entire conversation history after every external interaction. This discards the model's internal attention state, incurs quadratic prefill costs, and breaks cognitive continuity—a critical limitation for embodied agents that must maintain persistent spatial and task awareness.

## Our Approach

SIG enables a running model to pause autoregressive decoding, invoke external cognitive modules (cloud LLM "teachers", perception APIs, local skill libraries), and seamlessly absorb their responses into the model's key-value (KV) cache **without costly re-encoding**. The local model becomes a privacy-preserving hub that summons world-class expertise on demand while maintaining continuous attention state.

## Key Results

- **Up to 96% prefill token reduction** and **86% prefill time savings**
- **1.57× end-to-end speedup** on 0.8B models
- **3× improvement in answer information coverage** in long-context scenarios
- Memory footprint under 1.5 GB GPU, suitable for smartphones and embedded devices

## Architecture

The CO framework consists of three layers:
1. **Meaning Compiler** — lightweight local model for intent parsing and orchestration
2. **Injection Engine** — SIG runtime managing KV-cache continuity
3. **Cognitive Module Ecosystem** — pluggable tools, cloud teachers, and local caches

## Contents

This repository contains the full implementation (based on llama.cpp), benchmark suite across 9 multi-turn scenarios, and pre-computed cloud teacher plans.

## Paper

For technical details, see our paper: *"Cognitive Outsourcing with Suspend-and-Inject Generation for Scalable Embodied Intelligence"*.

---

**Keywords**: edge AI, tool calling, KV-cache injection, embodied intelligence, small language models, privacy-preserving ML, agent architectures, LLM orchestration
