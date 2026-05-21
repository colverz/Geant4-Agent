# Geant4 Agent 工业 Runtime Benchmark 状态报告

生成日期：2026-05-15

本报告用于快速了解当前项目是否已经从“配置生成器”推进到“可真实运行、可量化评估的 Geant4 agent”。结论先写在前面：项目已经有第一个完整通过案例，但整体还远未完成工业级覆盖。

## 当前结论

当前已经跑通并审核通过的完整链路是：

```text
raw dialogue
-> live LLM candidate config
-> deterministic contract check
-> typed SimulationSpec / RuntimePayload
-> real local Geant4 runtime
-> structured metrics
-> reviewed golden numeric comparison
```

当前 official evaluator 结果：

| 指标 | 数量 |
|---|---:|
| Benchmark cases | 22 |
| Passed | 1 |
| Failed | 0 |
| Not evaluable | 19 |
| Explicit unsupported | 2 |

当前 compiler 覆盖：

| Compile status | 数量 | 含义 |
|---|---:|---|
| compiled | 4 | 当前 runtime payload 可以表达，下一步主要缺 golden 或 live full-chain 覆盖 |
| compiled_with_gaps | 5 | 可表达一部分，但 metric/scoring 或多 run 能力不足 |
| unsupported_capability | 13 | 当前 geometry/source/scoring/runtime executor 不能诚实支持 |

这不是一个“高通过率”阶段，而是一个“真实评估门槛已经立起来”的阶段。现在通过率低反而是有价值的，因为它暴露了真实能力边界。

## 已通过 Case

### shielding_lead_gamma_transmission

Raw dialogue:

> Use a 1 MeV gamma beam through 10 mm lead shielding and measure transmission at a silicon detector.

任务：计算 1 MeV gamma 穿过 10 mm lead shield 后在 silicon detector 的透过响应。

当前状态：`passed`

Full-chain gate：

| 检查项 | 结果 |
|---|---|
| Live LLM used | true |
| Prompt profile | `slot_extract_en_strict_slot_v2` |
| Candidate contract | passed |
| Real Geant4 runtime | completed |
| Golden status | reviewed |
| Metric comparison | passed |

Runtime 配置摘要：

| 字段 | 值 |
|---|---|
| Geometry | `LeadShield`, `G4_Pb`, `100 x 100 x 10 mm` |
| Detector | `Detector`, `G4_Si`, position `(0, 0, 50) mm`, size `20 x 20 x 2 mm` |
| Source | beam gamma, `1.0 MeV`, position `(0, 0, -100) mm`, direction `+z` |
| Physics | `FTFP_BERT` |
| Events | `10000` |
| Seed | `1337` |
| Threads | `1` |

评估指标：

| Metric | Actual | Golden | Tolerance | Result |
|---|---:|---:|---:|---|
| detector_crossing_count | 4659 | 4659 | 0 | passed |
| detector_edep_total_mev | 53.9888 | 53.9888 | 0.0 | passed |
| transmission_factor | 0.4659 | 0.4659 | 0.0 | passed |

Golden 文件：

```text
docs/eval/golden/industrial_runtime/shielding_lead_gamma_transmission.golden.json
```

审核说明：

- runtime payload hash 与 deterministic compiler 输出一致。
- run summary 中的指标可直接追溯到 structured Geant4 result。
- 同 seed、单线程重跑复现三项指标。
- 10 mm Pb / 1 MeV gamma 的透过率量级与 NIST XCOM Pb 1 MeV 衰减数据一致。
- 该审核是内部 benchmark engineering review，不是外部物理认证。

参考：NIST XCOM Pb mass attenuation coefficients  
https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z82.html

## Casebank 明细

### 1. ndt_steel_step_wedge_gamma_detector

Domain: `industrial_ndt`

Raw dialogue:

> Model a 1 MeV gamma beam through a steel step wedge and score the silicon detector response behind it.

Status: `not_evaluable`

主要阻塞：

- `step_wedge_geometry_not_supported_by_current_single_volume_runtime`
- `missing_golden_metrics`
- `transmission_ratio` 尚不能完整提取/计算

下一步：需要支持 step wedge / multi-thickness geometry，不能简化成单一 box 来“刷过”。

### 2. ndt_aluminum_block_void_contrast

Domain: `industrial_ndt`

Raw dialogue:

> Simulate a gamma radiography setup for an aluminum block with a central cylindrical air void and compare detector response through void and solid regions.

Status: `not_evaluable`

主要阻塞：

- `embedded_void_geometry_not_supported_by_current_single_volume_runtime`
- `region_scoring_not_supported_by_current_runtime_summary`
- `contrast_ratio`、`void_region_count`、`solid_region_count` 尚不能完整评估

下一步：需要 embedded void geometry 和 detector region scoring。

### 3. ndt_pipe_wall_corrosion_gamma

Domain: `industrial_ndt`

Raw dialogue:

> Compare detector transmission for a steel pipe section with nominal wall thickness and a corroded thinned wall region.

Status: `not_evaluable`

主要阻塞：

- `curved_pipe_geometry_not_supported_by_current_single_volume_runtime`
- `paired_runtime_runs_not_supported_by_current_benchmark_executor`
- nominal/corroded 两组结果和 relative change 尚不能评估

下一步：需要 curved geometry 或至少 paired scenario executor。

### 4. ndt_weld_defect_slab_radiography

Domain: `industrial_ndt`

Raw dialogue:

> Set up a steel slab weld inspection scene with a low-density defect zone and evaluate detector contrast.

Status: `not_evaluable`

主要阻塞：

- `defect_region_geometry_not_supported_by_current_single_volume_runtime`
- region contrast metrics 尚不能评估

下一步：需要 defect insert geometry 和 detector contrast scoring。

### 5. ndt_tungsten_inclusion_contrast

Domain: `industrial_ndt`

Raw dialogue:

> Put a small tungsten inclusion inside an aluminum block and quantify the detector response change for a gamma beam.

Status: `not_evaluable`

主要阻塞：

- `inclusion_geometry_not_supported_by_current_single_volume_runtime`
- `contrast_ratio` 尚不能完整评估

下一步：需要 multi-material inclusion geometry。

### 6. shielding_lead_gamma_transmission

Domain: `shielding`

Raw dialogue:

> Use a 1 MeV gamma beam through 10 mm lead shielding and measure transmission at a silicon detector.

Status: `passed`

结果摘要：

- detector_crossing_count = `4659`
- detector_edep_total_mev = `53.9888`
- transmission_factor = `0.4659`
- reviewed golden comparison = passed

### 7. shielding_concrete_gamma_transmission

Domain: `shielding`

Raw dialogue:

> Compare detector response before and after a 100 mm concrete shield for a 2 MeV gamma beam.

Status: `not_evaluable`

主要阻塞：

- compiled_with_gaps
- `paired_runtime_runs_not_supported_by_current_benchmark_executor`
- unshielded/shielded paired comparison 尚不能评估

下一步：需要 paired run executor。

### 8. shielding_polyethylene_neutron_moderation

Domain: `shielding`

Raw dialogue:

> Send 5 MeV neutrons through a polyethylene slab and score downstream neutron crossings and target energy deposition.

Status: `not_evaluable`

主要阻塞：

- compiled
- missing reviewed golden

下一步：这是下一批较适合推进的 case。当前 runtime payload 可表达，主要需要真实 runtime 生成 golden、审核指标，并确认 neutron scoring 是否稳定。

### 9. shielding_graded_lead_poly_gamma

Domain: `shielding`

Raw dialogue:

> Build a graded shield with lead followed by polyethylene and measure detector response for a gamma beam.

Status: `not_evaluable`

主要阻塞：

- `multi_layer_geometry_not_supported_by_current_runtime_payload`
- missing golden

下一步：需要 multi-layer geometry，不能把 lead/poly 顺序抹平。

### 10. medical_proton_water_depth_dose

Domain: `medical_phantom`

Raw dialogue:

> Send a 150 MeV proton beam into a water phantom and score depth-binned energy deposition.

Status: `not_evaluable`

主要阻塞：

- compiled_with_gaps
- `depth_binned_scoring_not_supported_by_current_runtime_summary`
- `bragg_peak_metric_not_supported_without_depth_binned_scoring`

下一步：需要 depth-binned scoring，这是医疗 phantom 方向的核心能力。

### 11. medical_electron_water_surface_dose

Domain: `medical_phantom`

Raw dialogue:

> Use a 12 MeV electron beam incident on water and score near-surface energy deposition bins.

Status: `not_evaluable`

主要阻塞：

- compiled_with_gaps
- depth-binned scoring 缺失
- surface region dose 尚不能结构化输出

下一步：同样需要 depth-binned scoring。

### 12. medical_gamma_water_depth_bins

Domain: `medical_phantom`

Raw dialogue:

> Fire a 6 MeV gamma beam into a water box and score energy deposition by depth bins.

Status: `not_evaluable`

主要阻塞：

- compiled_with_gaps
- depth-bin edep hash/分层 dose profile 尚不能输出

下一步：需要 depth-binned result summary。

### 13. detector_silicon_gamma_response

Domain: `detector_response`

Raw dialogue:

> Place a silicon detector behind an air gap and measure detector energy deposition from a 1 MeV gamma beam.

Status: `not_evaluable`

主要阻塞：

- compiled
- missing reviewed golden

下一步：这是下一批很适合推进的 case。runtime payload 可表达，指标也简单，适合作为第二个 official pass 候选。

### 14. detector_scintillator_gamma_response

Domain: `detector_response`

Raw dialogue:

> Use a plastic scintillator detector and score deposited energy from a 662 keV gamma beam.

Status: `not_evaluable`

主要阻塞：

- compiled
- missing reviewed golden

下一步：也是较适合推进的 case，但需要确认当前 Geant4 material `G4_PLASTIC_SC_VINYLTOLUENE` 在本地 runtime 中可用且行为稳定。

### 15. detector_position_acceptance_sweep

Domain: `detector_response`

Raw dialogue:

> Run the same gamma beam with the silicon detector at z=50 mm and z=100 mm and compare hit counts.

Status: `not_evaluable`

主要阻塞：

- `paired_runtime_runs_not_supported_by_current_benchmark_executor`
- `acceptance_ratio` 尚不能通过 paired result 计算

下一步：需要 multi-run / parameter sweep executor。

### 16. beam_gaussian_spread_plane

Domain: `beam_source`

Raw dialogue:

> Use a Gaussian gamma beam with 3 mm spot sigma and score the spatial spread at a plane 100 mm downstream.

Status: `not_evaluable`

主要阻塞：

- compiled_with_gaps
- `plane_spatial_distribution_metrics_not_supported_by_current_runtime_summary`
- sigma_x / sigma_y 尚不能评估

下一步：需要 plane crossing spatial distribution output。

### 17. beam_collimated_gamma_aperture

Domain: `beam_source`

Raw dialogue:

> Collimate a gamma beam with a lead aperture and measure downstream plane crossings.

Status: `not_evaluable`

主要阻塞：

- `collimator_aperture_geometry_not_supported_by_current_runtime_payload`
- `aperture_acceptance_requires_collimator_geometry`

下一步：需要 aperture/collimator geometry。

### 18. beam_isotropic_source_solid_angle

Domain: `beam_source`

Raw dialogue:

> Use an isotropic gamma point source and compute the fraction of events crossing a small silicon detector.

Status: `not_evaluable`

主要阻塞：

- `isotropic_source_sampling_not_supported_by_current_source_runtime`
- `solid_angle_acceptance_requires_isotropic_source_support`

下一步：需要 isotropic source sampling 和 solid-angle acceptance 评估。

### 19. multiturn_shield_thickness_update

Domain: `multi_turn_engineering`

Raw dialogue:

> Set up a 1 MeV gamma beam through a 5 mm lead shield and score the detector.

> Increase the lead thickness to 20 mm and rerun the same events.

> Compare the detector transmission against the original run.

Status: `not_evaluable`

主要阻塞：

- `multi_turn_benchmark_execution_not_supported_by_current_runtime_compiler`
- `paired_runtime_runs_not_supported_by_current_benchmark_executor`
- `paired_metric_delta_not_supported_without_multi_run_execution`

下一步：需要真正的 multi-turn runtime scenario，而不是只改 session config。

### 20. multiturn_source_energy_update

Domain: `multi_turn_engineering`

Raw dialogue:

> Run a gamma beam through a copper block at 1 MeV and score target energy deposition.

> Change the source energy to 2 MeV, keep the geometry and scoring unchanged, and rerun.

> Compare target deposited energy between the two runs.

Status: `not_evaluable`

主要阻塞：

- `multi_turn_benchmark_execution_not_supported_by_current_runtime_compiler`
- `multi_turn_source_energy_preservation_not_supported_by_current_benchmark_executor`
- `paired_metric_delta_not_supported_without_multi_run_execution`

下一步：需要 multi-run delta evaluator，并验证 slot/static context 在多轮里不会污染配置。

### 21. unsupported_cad_import_detector_housing

Domain: `unsupported_boundary`

Raw dialogue:

> Import this detector housing CAD model and simulate gamma response inside it.

Status: `unsupported_capability`

说明：这是故意保留的边界 case。当前项目不支持 CAD import，不应该假装支持。

### 22. unsupported_rotating_ct_gantry

Domain: `unsupported_boundary`

Raw dialogue:

> Build a rotating CT gantry with a moving x-ray source and collect projections around the object.

Status: `unsupported_capability`

说明：这是故意保留的边界 case。当前项目不支持 moving geometry / multi-projection runtime，不应该降级成静态单次模拟。

## 当前最有价值的下一步

建议下一轮不要先碰最复杂的 NDT / medical / multi-turn case，而是先扩充“可完整跑通”的 official pass 数量。

优先候选：

1. `detector_silicon_gamma_response`
2. `detector_scintillator_gamma_response`
3. `shielding_polyethylene_neutron_moderation`

这三个 case 的共同特点是：当前 compiler 已经能产出 runtime payload，主要缺真实 runtime golden 与 review。它们适合作为第二、第三、第四个 official full-chain pass。

随后再进入结构能力扩展：

1. Paired run executor：支持 shielded/unshielded、near/far detector、before/after update。
2. Depth-binned scoring：支撑 medical phantom 和 Bragg peak 类 case。
3. Region scoring / contrast：支撑 NDT void、defect、inclusion。
4. Multi-layer / inclusion / aperture geometry：支撑更接近工业场景的复杂几何。

## 风险说明

- 当前 `passed=1` 不能说明项目已经工业可用，只能说明主链已经闭合并可被严格评估。
- 当前 benchmark 标准没有降低：not_evaluable 是诚实失败，不是隐藏失败。
- LLM 目前只负责生成 candidate config；最终判断来自 deterministic contract、real runtime 和 reviewed golden metrics。
- 下一阶段最重要的是继续增加 reviewed runtime goldens，同时扩展 runtime/scoring 能力，而不是增加 parser-only case。
