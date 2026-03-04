# Geant4-Agent

A Geant4-oriented geometry assembly prototype: DSL + feasibility checker, plus a BERT-based NLU lab and an LLM-assisted planner for multi-turn clarification.

## Repository Layout

This repo is split into layered components:

- `builder/geometry/`: DSL + feasibility checker + experiments (Geant4-style assemblies)
- `nlu/bert_lab/`: BERT-based NLU (structure + parameters + entities)
- `planner/`: LLM-driven planning + clarification flows
- `knowledge/`: Materials/environment knowledge (JSON schema + validation)
- `ui/web/`: Local web UI for multi-turn dialogue
- `core/`: Shared schemas and semantic frame

## Architecture Overview

- **Geometry core** (`builder/geometry/`): DSL + analytical feasibility checks. Produces errors, warnings, and suggestions.
- **NLU core** (`nlu/bert_lab/`): Structure classification + parameter/entity extraction + postprocess.
- **Planner layer** (`planner/`): LLM-driven question planning and schema-constrained outputs.
- **Knowledge layer** (`knowledge/`): JSON schema, validated lists, and validation.
- **UI layer** (`ui/web/`): Local multi-turn interface.
- **Core** (`core/`): Shared schema + semantic frame.

See: `docs/ARCHITECTURE.md`

## Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## Geometry Quick Start

```powershell
python -m builder.geometry.cli run_all --outdir builder/geometry/out --n_samples 200 --n_param_sets 100 --seed 7 --dataset builder/geometry/examples/coverage.csv
```

Expected outputs:
- `builder/geometry/out/coverage_summary.json`
- `builder/geometry/out/coverage_checked.jsonl`
- `builder/geometry/out/feasibility_summary.json`
- `builder/geometry/out/ambiguity_summary.json`

## BERT Lab Quick Start

```powershell
python nlu/bert_lab/bert_lab_data.py --out nlu/bert_lab/data/bert_lab_samples.jsonl --n 200 --seed 7
```

Note:
- Trained checkpoints and model weights are not shipped in this repository.
- If you want to use the BERT lab locally, generate data and train the models on your own machine.
- Keep `nlu/bert_lab/models/` as a local-only directory.

## Knowledge Quick Start

Fetch Geant4 NIST material names (official list):

```powershell
python knowledge\tools\fetch_geant4_materials.py
```

## Local Web UI

```powershell
python ui/web/server.py
```

Then open:
- http://127.0.0.1:8088

## Current Limitations

- **No full Geant4 runtime config**: schema exists, but no full generator of G4 macro or C++ config.
- **Physics lists are reference-only**: fetched from official reference list, not a complete superset.
- **Output formats use the official Geant4 analysis file types** (`csv`, `hdf5`, `root`, `xml`) plus a project-local `json` extension.
- **RAG not implemented**: `knowledge/rag/` is a placeholder; no retrieval index yet.
- **Material → volume mapping is manual**: needs explicit `volume_names` to validate mappings.
- **BERT data is synthetic-heavy**: real-world robustness still unverified.

---

# Geant4-Agent（中文）

面向 Geant4 的几何装配原型：包含 DSL 与可行性检查，同时提供 BERT 语义解析与 LLM 规划的多轮对话能力。

## 仓库结构

仓库按分层组件组织：

- `builder/geometry/`：DSL + 理论可行性检查 + 实验
- `nlu/bert_lab/`：BERT 语义解析（结构/参数/实体）
- `planner/`：LLM 规划与追问流程
- `knowledge/`：材料/环境知识（JSON schema + 校验）
- `ui/web/`：本地多轮对话界面
- `core/`：共享 schema 与语义帧

## 架构概览

- **几何核心**（`builder/geometry/`）：DSL + 解析可行性判定，输出错误/警告/建议。
- **语义核心**（`nlu/bert_lab/`）：结构分类 + 参数/实体抽取 + 后处理。
- **规划层**（`planner/`）：LLM 驱动的追问与 schema 约束输出。
- **知识层**（`knowledge/`）：JSON schema、可溯源列表与校验。
- **UI 层**（`ui/web/`）：本地多轮对话界面。
- **Core**（`core/`）：共享 schema 与语义帧。

详见：`docs/ARCHITECTURE.md`

## 虚拟环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 几何快速开始

```powershell
python -m builder.geometry.cli run_all --outdir builder/geometry/out --n_samples 200 --n_param_sets 100 --seed 7 --dataset builder/geometry/examples/coverage.csv
```

预期输出：
- `builder/geometry/out/coverage_summary.json`
- `builder/geometry/out/coverage_checked.jsonl`
- `builder/geometry/out/feasibility_summary.json`
- `builder/geometry/out/ambiguity_summary.json`

## BERT 快速开始

```powershell
python nlu/bert_lab/bert_lab_data.py --out nlu/bert_lab/data/bert_lab_samples.jsonl --n 200 --seed 7
```

说明：
- 本仓库不提供已训练好的 checkpoint 或模型权重。
- 如需使用 BERT Lab，请先在本地生成数据并自行训练模型。
- `nlu/bert_lab/models/` 应仅作为本地目录使用，不要纳入版本控制。

## 知识层快速开始

抓取 Geant4 NIST 材料名（官方列表）：

```powershell
python knowledge\tools\fetch_geant4_materials.py
```

## 本地 Web 界面

```powershell
python ui/web/server.py
```

然后访问：
- http://127.0.0.1:8088

## 当前限制

- **尚无完整 Geant4 运行配置**：只有 schema，没有完整的 G4 宏或 C++ 生成器。
- **物理过程列表仅覆盖 reference**：非完整集合。
- **输出格式已接入 Geant4 官方分析文件类型**（`csv`、`hdf5`、`root`、`xml`），并保留项目本地 `json` 扩展。
- **RAG 尚未实现**：`knowledge/rag/` 为占位。
- **材料 → 体积映射需手工指定**：需要 `volume_names` 才能验证。
- **BERT 数据以合成为主**：真实输入鲁棒性待验证。
