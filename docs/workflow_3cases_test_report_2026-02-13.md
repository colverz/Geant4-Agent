# Geant4-Agent 端到端工作流测试报告（3例）

## 1. 测试目的
验证当前项目在**完整 workflow**下的可用性：

用户自然语言输入 -> Ollama 路由/标准化/追问 -> BERT 结构与参数抽取 -> 几何可行性与最小配置导出。

本次重点确认：
1. 材料匹配修复是否生效。
2. 结构与几何参数链路是否能稳定闭环。
3. 几何之外（source/physics/output）是否可用。

## 2. 测试配置
- 测试时间：2026-02-13
- 测试样本数：3
- Ollama 配置：`nlu/bert_lab/configs/ollama_config_expert_fast.json`
- 关键开关：`llm_router=true`，`llm_question=true`，`normalize_input=true`，`autofix=true`
- 结果文件：`nlu/bert_lab/data/eval/workflow_3cases_results.json`

## 3. 用例设计
1. `case_ideal_en_complete`：英文单轮完整输入（含 geometry/material/source/physics/output）。
2. `case_realistic_en_two_turns`：英文两轮补全（先几何+材料，再源+物理+输出）。
3. `case_realistic_zh_three_turns`：中文三轮补全（依赖 LLM 标准化）。

## 4. 总体结果
- `n_cases`: 3
- `n_complete`: 0
- `completion_rate`: 0.0

结论：当前版本在完整 workflow 下仍未达到可闭环完成标准。

## 5. 分项结果（按能力）

| 能力项 | 结果 | 说明 |
|---|---:|---|
| 几何结构闭环 | 失败 | 2/3 用例结构误判为 `grid`，导致缺失 `grid` 参数集 |
| 几何参数闭环 | 失败 | 几何分支未稳定，缺参持续存在 |
| 材料识别 | 部分通过 | 英文2例材料正确识别为 `G4_Cu`、`G4_Si`；中文1例未形成 `volume_material_map` |
| Source 识别 | 通过 | 3/3 提取到 `particle/energy/position/direction` |
| Physics 识别 | 通过 | 3/3 提取到 physics list |
| Output 识别 | 通过 | 3/3 提取到 format/path |

## 6. 关键观察

### 6.1 材料匹配修复已生效
修复前存在前缀误匹配（如 `G4_Cu -> G4_C`）。

本轮英文样例中：
- `G4_Cu` 保持为 `G4_Cu`
- `G4_Si` 保持为 `G4_Si`

说明材料匹配核心 bug 已被修复。

### 6.2 当前主瓶颈不在材料，而在几何链路
两个英文用例均被收敛到 `grid`，最终缺失 `module_x/module_y/module_z/nx/ny/pitch_x/pitch_y` 等字段。
这表明问题主要出在：
1. LLM 标准化文本与 BERT 训练分布对齐不足；
2. 结构判别策略对 `single_box/ring/grid` 的边界不稳。

### 6.3 中文流程存在单位语义风险
中文样例最终几何参数为 `module_x=1.0,module_y=1.0,module_z=1.0`，与“1m”语义不一致（期望 1000 mm）。
当前单位解析仍有缺口（尤其 `m` 单位）。

## 7. 逐例摘要

### 7.1 `case_ideal_en_complete`
- 最终状态：未完成
- 最终 phase：`geometry_params`
- 关键缺失：`geometry.params.module_x/module_y/module_z/nx/ny/pitch_x/pitch_y/clearance`
- 已正确：material=`G4_Cu`，source/physics/output 全部可用

### 7.2 `case_realistic_en_two_turns`
- 最终状态：未完成
- 最终 phase：`geometry_params`
- 关键缺失：`geometry.params.module_x/module_y/module_z/nx/ny/pitch_x/pitch_y`
- 已正确：material=`G4_Si`，source/physics/output 全部可用

### 7.3 `case_realistic_zh_three_turns`
- 最终状态：未完成
- 最终 phase：`materials`
- 关键缺失：`materials.volume_material_map`
- 已正确：geometry 结构为 `single_box`，source/physics/output 可用

## 8. 后续修正项（按优先级）

### P0（必须优先）
1. **几何结构判别稳定化**
- 针对 `single_box/ring/grid` 建立专用回归集（基于 LLM 标准化输出），并强制回归测试。
- 对结构预测加入“强约束一致性检查”（例如出现 `radius+n+module_size` 时优先 ring）。

2. **几何参数键契约统一**
- 固化 LLM 标准化输出 schema（字段名、单位表达、次序）。
- BERT 训练集只围绕该 schema 生成，减少分布漂移。

### P1（高优先）
3. **单位解析补全**
- 明确支持 `m/cm/mm`，统一转换到 `mm`。
- 增加单位回归用例，覆盖中文和英文表达。

4. **材料映射补全策略**
- 中文场景下若已识别材料但缺 `volume_material_map`，允许在“单体几何”时自动回填 `target -> material`。

### P2（中优先）
5. **workflow 评测自动化**
- 固化“3例快速回归 + 20例扩展回归”两级套件。
- 输出统一 KPI：completion_rate、structure_acc、geometry_missing_count、material_map_acc。

## 9. 结论
- 材料匹配低级错误已修复并通过英文样例验证。
- 当前阻断闭环的首要问题是几何结构与参数链路，而非 source/physics/output。
- 下一阶段应集中于“标准化文本 -> BERT 几何解析”契约化与回归化。
