# Ollama 开启回归报告（多轮）

## 运行配置
- config_path: nlu/bert_lab/configs/deepseek_api.local.json
- provider: deepseek
- base_url: https://api.deepseek.com
- model: deepseek-chat
- min_confidence: 0.6

## 连接探针
- connectivity_ok: True
- model_count: 2
- models(sample): ['deepseek-chat', 'deepseek-reasoner']
- error: none

## 总体指标
- 用例数: 5
- 语言自然化平均分: 0.913
- 准确性平均分: 1.000
- 严格通过率: 100.00%
- LLM 使用回合占比: 100.00%
- 未通过用例: []

## 逐例（含 raw dialogue）
### ollama_case01_box_full
- naturalness: {'internal_field_leak_rate': 0.0, 'template_rate': 0.0, 'short_reply_rate': 1.0, 'naturalness_score': 0.8}
- accuracy: 1.000 (6/6)
- strict_pass: True
- llm_used_turn_ratio: 1.00
- turns:
  - turn 1 user: 我要一个铜立方体，尺寸 1m x 1m x 1m。Gamma 点源，能量 1 MeV，位置在中心，方向 +z，物理列表 FTFP_BERT，输出 json。
    assistant: 配置已完成。
    action=finalize, llm_used=True, fallback=None, missing=[]
- raw_dialogue:
```json
[
  {
    "role": "user",
    "content": "我要一个铜立方体，尺寸 1m x 1m x 1m。Gamma 点源，能量 1 MeV，位置在中心，方向 +z，物理列表 FTFP_BERT，输出 json。"
  },
  {
    "role": "assistant",
    "content": "配置已完成。"
  }
]
```

### ollama_case02_box_two_turns
- naturalness: {'internal_field_leak_rate': 0.0, 'template_rate': 0.0, 'short_reply_rate': 0.0, 'naturalness_score': 1.0}
- accuracy: 1.000 (5/5)
- strict_pass: True
- llm_used_turn_ratio: 1.00
- turns:
  - turn 1 user: 铜立方体 1m x 1m x 1m，gamma 点源，1 MeV，位置原点，方向 +z。
    assistant: 目前配置进度如下：

- **几何**：已设置立方体结构，尺寸为 1m × 1m × 1m。
- **材料**：已选择铜作为材料，并绑定到几何体。
- **源**：已设置为 gamma 点源，能量 1 MeV，位于原点，方向沿 +z。
- **物理**：物理列表尚未选择。
- **输出**：输出格式和保存路径还未指定。

请继续补充物理列表、输出格式和输出路径，完成后即可生成完整的模拟配置。
    action=summarize_progress, llm_used=True, fallback=None, missing=['physics.physics_list', 'output.format', 'output.path']
  - turn 2 user: 物理列表 QBBC，输出 root。
    assistant: 配置已完成。物理列表已设为 QBBC，输出格式为 ROOT 文件，输出路径将使用默认位置。
    action=finalize, llm_used=True, fallback=None, missing=[]
- raw_dialogue:
```json
[
  {
    "role": "user",
    "content": "铜立方体 1m x 1m x 1m，gamma 点源，1 MeV，位置原点，方向 +z。"
  },
  {
    "role": "assistant",
    "content": "目前配置进度如下：\n\n- **几何**：已设置立方体结构，尺寸为 1m × 1m × 1m。\n- **材料**：已选择铜作为材料，并绑定到几何体。\n- **源**：已设置为 gamma 点源，能量 1 MeV，位于原点，方向沿 +z。\n- **物理**：物理列表尚未选择。\n- **输出**：输出格式和保存路径还未指定。\n\n请继续补充物理列表、输出格式和输出路径，完成后即可生成完整的模拟配置。"
  },
  {
    "role": "user",
    "content": "物理列表 QBBC，输出 root。"
  },
  {
    "role": "assistant",
    "content": "配置已完成。物理列表已设为 QBBC，输出格式为 ROOT 文件，输出路径将使用默认位置。"
  }
]
```

### ollama_case03_tubs_two_turns
- naturalness: {'internal_field_leak_rate': 0.0, 'template_rate': 0.0, 'short_reply_rate': 0.5, 'naturalness_score': 0.9}
- accuracy: 1.000 (5/5)
- strict_pass: True
- llm_used_turn_ratio: 1.00
- turns:
  - turn 1 user: 我要圆柱铜靶，半径 30 mm，半长 50 mm，gamma 点源 2 MeV，位置(0,0,-100)，方向(0,0,1)。
    assistant: 当前配置进度如下：

几何部分已根据你的输入，设定了圆柱体结构和尺寸（半径30mm、半高50mm）。  
材料部分已选定铜，并完成了与体积的绑定。  
源部分已设置为2 MeV的伽马点源，并指定了位置和方向。  

接下来还需要确认以下设置：  
- 物理列表  
- 输出格式  
- 输出路径  

请补充这些信息，以便继续完成配置。
    action=summarize_progress, llm_used=True, fallback=None, missing=['physics.physics_list', 'output.format', 'output.path']
  - turn 2 user: 物理列表 QBBC，输出 json。
    assistant: 配置已完成。
    action=finalize, llm_used=True, fallback=None, missing=[]
- raw_dialogue:
```json
[
  {
    "role": "user",
    "content": "我要圆柱铜靶，半径 30 mm，半长 50 mm，gamma 点源 2 MeV，位置(0,0,-100)，方向(0,0,1)。"
  },
  {
    "role": "assistant",
    "content": "当前配置进度如下：\n\n几何部分已根据你的输入，设定了圆柱体结构和尺寸（半径30mm、半高50mm）。  \n材料部分已选定铜，并完成了与体积的绑定。  \n源部分已设置为2 MeV的伽马点源，并指定了位置和方向。  \n\n接下来还需要确认以下设置：  \n- 物理列表  \n- 输出格式  \n- 输出路径  \n\n请补充这些信息，以便继续完成配置。"
  },
  {
    "role": "user",
    "content": "物理列表 QBBC，输出 json。"
  },
  {
    "role": "assistant",
    "content": "配置已完成。"
  }
]
```

### ollama_case04_overwrite_confirm
- naturalness: {'internal_field_leak_rate': 0.0, 'template_rate': 0.0, 'short_reply_rate': 0.3333333333333333, 'naturalness_score': 0.9333333333333333}
- accuracy: 1.000 (3/3)
- strict_pass: True
- llm_used_turn_ratio: 1.00
- turns:
  - turn 1 user: 铜立方体 1m x 1m x 1m，gamma 点源 1 MeV，位置中心，方向 +z，物理 FTFP_BERT，输出 json。
    assistant: 配置已完成。
    action=finalize, llm_used=True, fallback=None, missing=[]
  - turn 2 user: 把材料改成 G4_Al。
    assistant: 检测到您要将材料从 G4_Cu 改为 G4_Al，这会覆盖已确认的设置。请确认是否应用此修改？
    action=confirm_overwrite, llm_used=True, fallback=None, missing=[]
  - turn 3 user: 确认。
    assistant: 好的，配置已更新完成。已将材料从 G4_Cu 改为 G4_Al，并应用到对应的立方体体积上。
    action=finalize, llm_used=True, fallback=None, missing=[]
- raw_dialogue:
```json
[
  {
    "role": "user",
    "content": "铜立方体 1m x 1m x 1m，gamma 点源 1 MeV，位置中心，方向 +z，物理 FTFP_BERT，输出 json。"
  },
  {
    "role": "assistant",
    "content": "配置已完成。"
  },
  {
    "role": "user",
    "content": "把材料改成 G4_Al。"
  },
  {
    "role": "assistant",
    "content": "检测到您要将材料从 G4_Cu 改为 G4_Al，这会覆盖已确认的设置。请确认是否应用此修改？"
  },
  {
    "role": "user",
    "content": "确认。"
  },
  {
    "role": "assistant",
    "content": "好的，配置已更新完成。已将材料从 G4_Cu 改为 G4_Al，并应用到对应的立方体体积上。"
  }
]
```

### ollama_case05_source_split
- naturalness: {'internal_field_leak_rate': 0.0, 'template_rate': 0.0, 'short_reply_rate': 0.3333333333333333, 'naturalness_score': 0.9333333333333333}
- accuracy: 1.000 (5/5)
- strict_pass: True
- llm_used_turn_ratio: 1.00
- turns:
  - turn 1 user: 铜立方体 1m x 1m x 1m，gamma 点源。
    assistant: 当前配置进度如下：

**几何**：已设置立方体结构，尺寸为 1m x 1m x 1m。  
**材料**：已选择铜，并完成与体积的绑定。  
**源**：已确定为 gamma 点源，还需补充能量和位置信息。  
**物理**：待选择物理列表。  
**输出**：待指定输出格式和路径。

请继续提供缺失的源参数、物理列表和输出设置。
    action=summarize_progress, llm_used=True, fallback=None, missing=['source.energy', 'source.position', 'source.direction', 'physics.physics_list', 'output.format', 'output.path']
  - turn 2 user: 能量 1 MeV，位置 (0,0,0)，方向 +z。
    assistant: 当前配置进度如下：

**几何**：已确认几何根体积名。  
**材料**：已确认材料选择依据和来源。  
**源**：已同步能量、位置和方向设置。  
**物理**：还需补充物理列表。  
**输出**：还需指定输出格式和输出路径。
    action=summarize_progress, llm_used=True, fallback=None, missing=['physics.physics_list', 'output.format', 'output.path']
  - turn 3 user: 物理 FTFP_BERT，输出 json。
    assistant: 配置已完成。
    action=finalize, llm_used=True, fallback=None, missing=[]
- raw_dialogue:
```json
[
  {
    "role": "user",
    "content": "铜立方体 1m x 1m x 1m，gamma 点源。"
  },
  {
    "role": "assistant",
    "content": "当前配置进度如下：\n\n**几何**：已设置立方体结构，尺寸为 1m x 1m x 1m。  \n**材料**：已选择铜，并完成与体积的绑定。  \n**源**：已确定为 gamma 点源，还需补充能量和位置信息。  \n**物理**：待选择物理列表。  \n**输出**：待指定输出格式和路径。\n\n请继续提供缺失的源参数、物理列表和输出设置。"
  },
  {
    "role": "user",
    "content": "能量 1 MeV，位置 (0,0,0)，方向 +z。"
  },
  {
    "role": "assistant",
    "content": "当前配置进度如下：\n\n**几何**：已确认几何根体积名。  \n**材料**：已确认材料选择依据和来源。  \n**源**：已同步能量、位置和方向设置。  \n**物理**：还需补充物理列表。  \n**输出**：还需指定输出格式和输出路径。"
  },
  {
    "role": "user",
    "content": "物理 FTFP_BERT，输出 json。"
  },
  {
    "role": "assistant",
    "content": "配置已完成。"
  }
]
```
