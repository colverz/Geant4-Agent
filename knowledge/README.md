# Knowledge Layer (Stage 1)

This module provides **structured, verifiable knowledge** for materials and
environment parameters. JSON is the source of truth; RAG is a helper for
explanations and disambiguation.

## Contents

- `schema/`: JSON schema for materials and environment.
- `data/`: Material lists and environment defaults/constraints.
- `validate.py`: Validation helpers.
- `cli.py`: Validate a JSON payload.
- `tools/`: Data fetchers (official Geant4 sources).
- `rag/`: Placeholder for future retrieval index.

## Data Sources (Stage 1)

- Geant4 Application Developers Guide: material definitions and concepts.
- Geant4 NIST material name list (official material names).

See `data/materials_geant4_nist.json` for the fetched list and metadata.

## Project Lists (Stage 1)

- `data/physics_lists.json`: reference physics lists (official source).
- `data/particles.json`: particle names (official source).
- `data/output_formats.json`: output formats (official Geant4 analysis formats + project extensions).

Notes:
- `data/physics_lists.json` is fetched from the official reference physics list index.
- `data/particles.json` is fetched from the official Geant4 particle list pages.
- `data/output_formats.json` now tracks the official Geant4 analysis manager file types (`csv`, `hdf5`, `root`, `xml`) and keeps `json` as a project-local extension.

---

# 知识层（阶段 1）

该模块提供**结构化、可验证**的材料与环境知识。JSON 作为真值来源；RAG 仅用于解释与消歧。

## 内容

- `schema/`：材料与环境 JSON schema
- `data/`：材料列表与环境默认/约束
- `validate.py`：校验入口
- `cli.py`：校验命令行
- `tools/`：官方数据抓取脚本
- `rag/`：检索占位

## 数据来源（阶段 1）

- Geant4 Application Developers Guide 中的材料定义与概念
- Geant4 NIST 材料名官方列表

已抓取列表见 `data/materials_geant4_nist.json`。

## 列表说明（阶段 1）

- `data/physics_lists.json`：官方 reference physics lists
- `data/particles.json`：官方粒子列表
- `data/output_formats.json`：项目自定义输出格式

备注：
- `data/physics_lists.json` 来自官方 reference physics list 页面。
- `data/particles.json` 来自官方 particle list 页面。
- `data/output_formats.json` 为项目自定义（非官方）。
