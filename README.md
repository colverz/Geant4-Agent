# Geant4-Agent

A Geant4-oriented geometry assembly prototype: DSL + feasibility checker, plus a BERT lab for
structure/parameter extraction.

## Repository Layout

This repo is split into four layers:

- `geometry/`: DSL + feasibility checker + experiments (Geant4-style assemblies)
- `bert_lab/`: A small, isolated starting point for BERT-based parameter extraction
- `knowledge/`: Materials/environment knowledge (JSON schema + validation)
- `llm/`: Ollama-driven prompt flows and JSON-schema constrained generation

## Architecture Overview

- **Geometry core** (`geometry/`): DSL + analytical feasibility checks. Produces errors, warnings, and suggestions.
- **Language core** (`bert_lab/`): Structure classification + parameter extraction + postprocess.
- **Knowledge layer** (`knowledge/`): JSON schema, validated lists, and validation.
- **LLM layer** (`llm/`): Prompt flows and schema-constrained outputs via Ollama.

## Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## Geometry Quick Start

```powershell
python -m geometry.cli run_all --outdir geometry/out --n_samples 200 --n_param_sets 100 --seed 7 --dataset geometry/examples/coverage.csv
```

Expected outputs:
- `geometry/out/coverage_summary.json`
- `geometry/out/coverage_checked.jsonl`
- `geometry/out/feasibility_summary.json`
- `geometry/out/ambiguity_summary.json`

## BERT Lab Quick Start

```powershell
python bert_lab/bert_lab_data.py --out bert_lab/bert_lab_samples.jsonl --n 200 --seed 7
```

## Knowledge Quick Start

Fetch Geant4 NIST material names (official list):

```powershell
python knowledge\\tools\\fetch_geant4_materials.py
```

## Local Web UI

```powershell
python llm/web/server.py
```

Then open:

- http://127.0.0.1:8088

## Current Limitations

- **No full Geant4 runtime config**: schema exists, but no full generator of G4 macro or C++ config.
- **Physics lists are reference-only**: fetched from official “reference” list, not a complete superset.
- **Output formats are project-defined**: not an official Geant4 list.
- **RAG not implemented**: `knowledge/rag/` is a placeholder; no retrieval index yet.
- **Material ↔ volume mapping is manual**: needs explicit `volume_names` to validate mappings.
- **BERT data is synthetic-heavy**: real-world robustness still unverified.

---

# Geant4-Agent（中文）

面向 Geant4 的几何装配原型：包含 DSL 与可行性检查，以及用于结构/参数抽取的 BERT Lab。

## 仓库结构

本仓库分为四层：

- `geometry/`: DSL + 理论可行性检查 + 实验
- `bert_lab/`: BERT 结构/参数抽取实验区
- `knowledge/`: 材料/环境知识层（JSON schema + 校验）
- `llm/`: LLM 层（Ollama 驱动的 prompt 流程与 schema 约束）

## 架构概览

- **几何核心**（`geometry/`）：DSL + 解析可行性判定，输出错误/警告/建议。
- **语言核心**（`bert_lab/`）：结构分类 + 参数抽取 + 后处理。
- **知识层**（`knowledge/`）：JSON schema、可溯源列表与校验。
- **LLM 层**（`llm/`）：Ollama prompt 流程与 schema 约束输出。

## 虚拟环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 几何快速开始

```powershell
python -m geometry.cli run_all --outdir geometry/out --n_samples 200 --n_param_sets 100 --seed 7 --dataset geometry/examples/coverage.csv
```

预期输出：
- `geometry/out/coverage_summary.json`
- `geometry/out/coverage_checked.jsonl`
- `geometry/out/feasibility_summary.json`
- `geometry/out/ambiguity_summary.json`

## BERT 快速开始

```powershell
python bert_lab/bert_lab_data.py --out bert_lab/bert_lab_samples.jsonl --n 200 --seed 7
```

## 知识层快速开始

抓取官方材料名单：

```powershell
python knowledge\\tools\\fetch_geant4_materials.py
```

## 本地 Web 界面

```powershell
python llm/web/server.py
```

然后访问：

- http://127.0.0.1:8088

## 当前限制

- **尚无完整 Geant4 运行配置**：目前只有 schema，没有 G4 宏或 C++ 配置生成。
- **物理列表仅覆盖 reference 列表**：非完整集合。
- **输出格式为项目定义**：非官方 Geant4 列表。
- **RAG 尚未实现**：`knowledge/rag/` 仅为占位目录。
- **材料与体积映射需手工指定**：需要 `volume_names` 才能验证映射。
- **BERT 数据以合成为主**：真实输入鲁棒性待验证。
