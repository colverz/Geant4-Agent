# 阶段更新（BERT 训练策略）- 2026-03-05

## 1) 本轮已完成

- 完整回归（双大类 + 三领域）重新执行：
  - 数据：`docs/dual_class_domain_test_data_2026-03-05.json`
  - 报告：`docs/dual_class_domain_test_report_2026-03-05.tex`
  - 结果：`strict_pass_rate=1.0`，`closure_rate=0.9167`，`failed_ids=[]`
- 修复了 `beam` 被误判为 `point` 的关键问题（“pointing” 误触发）：
  - 代码：`nlu/bert/extractor.py`
  - 代码：`nlu/llm/slot_frame.py`
  - 对应测试：
    - `tests/test_extractor_fallbacks.py`
    - `tests/test_llm_slot_frame.py`

## 2) 对“当前 BERT 训练策略是否最优”的结论

- 仅看历史离线评估（in_dist/hard/realnorm）会得到“几乎最优”结论。
- 但加入本轮新增的“噪声标准化文本探针集”后，不再是最优：
  - 探针结果：`docs/structure_probe_eval_2026-03-05.json`
  - `accuracy=0.92`
  - 薄弱类别：`nest=0.8507`，`shell=0.7021`
- 因此结论应为：**当前策略可用，但不是最终最优；仍需针对归一化噪声文本做强化训练。**

## 3) 已落地的训练策略改造（可直接执行）

- 新增 v2 数据构建脚本（面向“LLM 归一化文本”风格）：
  - `scripts/build_structure_v2_dataset.py`
  - 产物：`nlu/bert_lab/data/controlled_structure_v2.jsonl`
  - 统计：`nlu/bert_lab/data/controlled_structure_v2.summary.json`
- 新增 v2 训练配置：
  - `nlu/bert_lab/configs/structure_train_v2.json`
- 新增 v2 训练脚本（含 class weight + label smoothing + early stopping）：
  - `scripts/train_structure_v2.py`
- 新增训练策略审计脚本：
  - `scripts/audit_bert_strategy.py`
  - 产物：
    - `docs/bert_strategy_audit_2026-03-05.json`
    - `docs/bert_strategy_audit_2026-03-05.md`

## 4) 下一步建议（按优先级）

1. 先训练 `structure_controlled_v5_e2`（用 v2 数据）并替换运行时默认结构模型。
2. 复跑 `scripts/eval_structure_probe.py`，目标把探针准确率提升到 `>=0.95`，并重点观察 `nest/shell`。
3. 再跑一次全链路双大类报告，确认端到端稳定收益，而不是仅离线分类收益。

