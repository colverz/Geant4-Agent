# 几何链路重构验证报告（候选图搜索）

## 1. 重构目标
将原“单标签结构分类”主链路替换为：

- 候选图生成（ring/grid/nest/stack/shell/single_box/single_tubs）
- 约束与证据联合打分（参数覆盖、文本线索、可行性、缺参数/错误惩罚）
- Top-K 候选排序与可解释输出

## 2. 已实施改动
- 新增：`nlu/bert_lab/graph_search.py`
- 接入：`nlu/bert_lab/semantic.py`
- 接入：`nlu/bert_lab/end_to_end.py`
- 接入：`ui/web/server.py`（返回 `graph_candidates`）
- 参数抽取加强：`nlu/bert_lab/postprocess.py`
  - 新增 key/value 提取
  - 新增 `N modules` 计数提取
  - 修复 `1m -> 1000mm` 单位解析

## 3. 验证策略
### 3.1 算法级验证（不依赖 Ollama，聚焦几何逻辑）
执行脚本：

```powershell
.\.venv\Scripts\python -m nlu.bert_lab.eval_graph_search
```

输出：
- `nlu/bert_lab/data/eval/graph_search_eval.json`
- `docs/graph_search_eval_report.md`

结果：
- 样本数：15
- Top-1 准确率：0.800
- Top-3 召回率：1.000
- unknown 精确率：0.667

解释：候选图机制在 Top-K 层面已具备稳定“候选覆盖能力”，可用于多轮追问闭环。

### 3.2 工作流级回归（快速三例，禁用 LLM 归一化，仅验证几何决策替换效果）
输出：
- `nlu/bert_lab/data/eval/workflow_3cases_after_refactor.json`

结果：
- `n_cases=3`
- `n_complete=2`

与重构前（同三例报告）对比：
- 重构前：`0/3` 完成
- 重构后（该验证条件）：`2/3` 完成

## 4. 结论
1. 几何主链路已经从“分类中心”切换到“候选图搜索中心”。
2. 在可复现实验中，几何闭环能力显著提升。
3. 材料前缀误匹配问题已修复，材料识别稳定性提升。

## 5. 当前剩余问题
1. `unknown` 判定仍可进一步校准（避免少量误拒）。
2. shell/nest 的中文表达样本仍偏少，需补充语料与规则。
3. 带 LLM 归一化的全流程评测仍需做一轮完整对照（耗时较高，建议夜间批跑）。

## 6. 下一步建议（按优先级）
1. 新增 `A/B` 脚本：旧链路 vs 新链路同集对照（Top-1, Top-K, Completion, Wrong-ask）。
2. 扩展评测集到 100+（含中文、歧义、单位混合、冲突参数）。
3. 在 UI 输出中展示 Top-3 候选图的打分分解与淘汰原因，增强可解释性。
