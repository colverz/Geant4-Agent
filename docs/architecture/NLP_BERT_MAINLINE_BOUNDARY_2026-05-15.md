# NLP-BERT 主链边界说明

## 结论

当前主链应逐步移出 NLP 侧 BERT 依赖。主链默认目标是：

```text
用户自然语言
-> LLM slot/semantic frame
-> runtime_semantic rules + knowledge validation
-> geometry/source v2 bridge
-> typed runtime payload
-> Geant4 runtime
-> structured result
```

这里的 BERT 指项目 NLP 侧的 `nlu/training/bert_lab` 模型先验，不是 Geant4 物理列表 `FTFP_BERT`。`FTFP_BERT` 仍然是合法 physics list，不能因为 NLP-BERT 退主链而删除或改名。

## 为什么移出

- NLP-BERT 对早期 parser 有帮助，但现在主链已经有 LLM 解释层、知识库白名单、validator、v2 geometry/source bridge 和 runtime benchmark。
- BERT 先验容易把系统拉回“分类器 + 字典”的旧路线，尤其在复杂工业场景里，真正需要的是 LLM 解释候选、deterministic validator 和 Geant4 runtime 结果闭环。
- 保留 BERT 在 legacy fallback 中仍有价值：无 LLM、无网络、旧测试、旧 demo 可以继续运行。

## 当前边界

- Web UI 默认选择 `geometry=v2`、`source=v2`，面向主链。
- `ui.web.strict_api` 只在 LLM 主链启用时默认注入 v2 pipeline；显式关闭 LLM 的 fallback 保持 selector 默认 legacy。
- `nlu.runtime_semantic.extract_runtime_semantic_frame(..., enable_nlp_bert_prior=False)` 不调用 NLP-BERT NER 或 structure model，只用规则、知识库和 graph search。
- `nlu.runtime_extractor` 是当前 runtime semantic extractor 的主入口。
- `nlu/bert/extractor.py` 只保留 thin compatibility wrapper，旧 import 仍可用，但新代码不应依赖 `nlu.bert.extractor`。
- legacy pipeline 或显式兼容路径仍可启用 NLP-BERT prior。
- `Producer.RUNTIME_SEMANTIC` 是新的主链 producer；`Producer.BERT_EXTRACTOR` 保留为兼容标识，不再代表主链事实来源。

## 后续目标

- 保持 `nlu/bert/extractor.py` 为 compatibility wrapper，后续只允许 legacy/test 兼容场景引用。
- 将复杂 graph geometry 的 v2 bridge 做厚，避免依赖旧结构分类器。
- 用 industrial runtime benchmark 验证最终效果，不能只用 config diff 证明成功。

## Benchmark 评估方式

`docs/eval/agentic_benchmark_v1.json` 已加入 `nlu_boundary` capability 和 `expected_nlu` 契约。当前覆盖 box、beam/cylinder、多轮 guarded runtime、复杂 detector/scoring 未支持请求、非法 physics list 和 live LLM baseline。关键断言是：

```json
{
  "expected_nlu": {
    "must_disable_nlp_bert_prior": true,
    "expected_inference_backend": "runtime_semantic_rules"
  }
}
```

`tools/evaluate_geant4_agent_benchmark.py --dry-run` 会输出 `nlu_boundary_summary`，用于确认 no-BERT 主链不是“结果碰巧正确”，而是真的没有启用 NLP-BERT model prior。

这层评估只证明 NLU 主链边界；工业可用性仍必须继续看 runtime benchmark，包括真实 Geant4 运行后的能量沉积、穿透率、detector crossing 等定量结果。
