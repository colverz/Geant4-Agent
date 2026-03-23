# Geant4-Agent

A Geant4-oriented geometry assembly prototype: DSL + feasibility checker, plus a BERT-based NLU lab and an LLM-assisted planner for multi-turn clarification.

## Repository Layout

This repo is now easier to read if you treat it as three groups rather than one flat tree:

- Runtime path: `core/`, `nlu/`, `planner/`, `ui/web/`
- Deterministic builders and knowledge: `builder/geometry/`, `knowledge/`
- Historical assets and generated deliverables: `docs/archive/`, `legacy/`, `docs/reports/`

Main directories:

- `core/`: shared contracts, orchestration, dialogue policy, validation, config registries
- `nlu/`: runtime NLU, LLM adapters, BERT extractor, and separated training assets
- `builder/geometry/`: DSL + feasibility checker + geometry synthesis
- `planner/`: clarification planning and question rendering
- `knowledge/`: materials, particles, physics lists, schemas, and validation tools
- `ui/web/`: local multi-turn web UI
- `docs/`: active docs, release docs, reports, and archives
- `legacy/`: frozen legacy programs and reports

## Architecture Overview

- **Geometry core** (`builder/geometry/`): DSL + analytical feasibility checks. Produces errors, warnings, and suggestions.
- **NLU core** (`nlu/`): Runtime semantic extraction, LLM-assisted parsing, structure extraction, and separated BERT training assets.
- **Planner layer** (`planner/`): LLM-driven question planning and schema-constrained outputs.
- **Knowledge layer** (`knowledge/`): JSON schema, validated lists, and validation.
- **UI layer** (`ui/web/`): Local multi-turn interface.
- **Core** (`core/`): Shared contracts, orchestration, dialogue state, and validation.

See: `docs/architecture/ARCHITECTURE.md` and `docs/PROJECT_CONCLUSION_2026-03-23.md`.

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

## BERT Training Quick Start

```powershell
python nlu/training/bert_lab/bert_lab_data.py --out nlu/training/bert_lab/data/bert_lab_samples.jsonl --n 200 --seed 7
```

Note:
- Trained checkpoints and model weights are not shipped in this repository.
- Training assets now live under `nlu/training/bert_lab/`, outside the runtime path.
- If you want to train locally, generate data and train the models on your own machine.
- Keep `nlu/training/bert_lab/models/` as a local-only directory.

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

## Archived Tooling

Evaluation, regression, and one-off dataset scripts have been moved out of the main tree into local-only archived tooling under `legacy/tooling/`.

Generated reports and local test outputs are intentionally ignored by git.

## LLM Provider Config (Ollama / OpenAI-Compatible)

The project now supports multiple LLM backends through config files:

- `provider=ollama` (default): uses `POST /api/generate`
- `provider=openai_compatible` (also supports aliases like `deepseek`): uses `POST /v1/chat/completions`

Example configs:
- `nlu/llm_support/configs/ollama_config.json`
- `nlu/llm_support/configs/openai_compatible_config.example.json`
- `nlu/llm_support/configs/deepseek_api_config.example.json`

To use API-key based providers, set `api_key` or `api_key_env` in the config.

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

现在更适合按三类理解这个仓库：

- 运行主链路：`core/`、`nlu/`、`planner/`、`ui/web/`
- 确定性构建与知识：`builder/geometry/`、`knowledge/`
- 历史归档与生成物：`docs/archive/`、`legacy/`、`docs/reports/`

主要目录：

- `core/`：共享契约、编排、对话状态、校验和配置注册表
- `nlu/`：运行时语义解析、LLM 适配层、BERT 提取器，以及拆分后的训练资产
- `builder/geometry/`：DSL、可行性检查、几何合成
- `planner/`：追问规划和问题渲染
- `knowledge/`：材料、粒子、物理列表、schema 与校验工具
- `ui/web/`：本地多轮 Web UI
- `docs/`：当前文档、发布材料、回归报告、归档材料
- `legacy/`：冻结的旧程序和历史报告

## 架构概览

- **几何核心**（`builder/geometry/`）：DSL + 解析可行性判定，输出错误/警告/建议。
- **语义核心**（`nlu/`）：运行时语义抽取、LLM 辅助解析、结构识别，以及拆分后的 BERT 训练资产。
- **规划层**（`planner/`）：LLM 驱动的追问与 schema 约束输出。
- **知识层**（`knowledge/`）：JSON schema、可溯源列表与校验。
- **UI 层**（`ui/web/`）：本地多轮对话界面。
- **Core**（`core/`）：共享契约、编排、对话状态与校验。

详见：`docs/architecture/ARCHITECTURE.md` 与 `docs/PROJECT_CONCLUSION_2026-03-23.md`

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

## BERT 训练快速开始

```powershell
python nlu/training/bert_lab/bert_lab_data.py --out nlu/training/bert_lab/data/bert_lab_samples.jsonl --n 200 --seed 7
```

说明：
- 本仓库不提供已训练好的 checkpoint 或模型权重。
- 训练相关资产已迁移到 `nlu/training/bert_lab/`，不再放在运行时主链路里。
- 如需训练，请先在本地生成数据并自行训练模型。
- `nlu/training/bert_lab/models/` 应仅作为本地目录使用，不要纳入版本控制。

## 已归档工具

回归、评测、数据探针等一次性脚本已经迁入 `legacy/tooling/`，并作为本地工具处理，不再作为主仓库代码的一部分维护。

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

## LLM 提供方配置（Ollama / OpenAI 兼容）

项目现已支持多种 LLM 后端配置：

- `provider=ollama`（默认）：调用 `POST /api/generate`
- `provider=openai_compatible`（也支持 `deepseek` 等别名）：调用 `POST /v1/chat/completions`

示例配置文件：
- `nlu/llm_support/configs/ollama_config.json`
- `nlu/llm_support/configs/openai_compatible_config.example.json`
- `nlu/llm_support/configs/deepseek_api_config.example.json`

如需 API Key 模式，可在配置中设置 `api_key` 或 `api_key_env`。

## 当前限制

- **尚无完整 Geant4 运行配置**：只有 schema，没有完整的 G4 宏或 C++ 生成器。
- **物理过程列表仅覆盖 reference**：非完整集合。
- **输出格式已接入 Geant4 官方分析文件类型**（`csv`、`hdf5`、`root`、`xml`），并保留项目本地 `json` 扩展。
- **RAG 尚未实现**：`knowledge/rag/` 为占位。
- **材料 → 体积映射需手工指定**：需要 `volume_names` 才能验证。
- **BERT 数据以合成为主**：真实输入鲁棒性待验证。

