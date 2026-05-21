# Geant4-Agent UI 重建计划

## 目标

当前 UI 不再继续修补。新的 UI 只复用后端 API 与业务能力，不复用旧的 sidebar/card/chip/inspector/debug-panel 设计范式。

新界面目标是成为一个桌面优先的 agent simulation workspace：

- 用户默认通过自然语言发起模拟目标。
- Agent 默认先做方案设计，而不是让用户点击“设计候选方案”。
- 可公开的 agent activity 显式展示，包括需求理解、知识参考、能力校验、配置草案、runtime preflight、run result。
- 隐藏的模型内部推理不伪造、不展示；界面展示的是可验证的过程记录。
- Runtime 操作可以明确触发，但不再用吓人的 guard 文案阻断正常模拟。
- Debug 和 raw JSON 默认放入抽屉，不污染主对话。

## 设计原则

参考已安装 skills 的本地规范：

- `figma-generate-design`：先定义 screen sections、tokens、组件语义，再实现。
- `figma-implement-design`：实现时追求视觉一致性、响应式、可访问性，不做硬编码堆叠。
- `chatgpt-apps`：工具/动作边界清晰，read-only、mutation、runtime action 分离。
- `winui-app`：桌面应用应有明确 shell、响应式行为、主题资源、窗口缩放策略。

## 新信息架构

### 1. Conversation Stage

主区域，占据最大空间。

内容：

- 用户消息。
- Agent 主回答。
- 每轮回答下的 activity trace。
- 方案设计结果的紧凑摘要。

约束：

- 不再用灰色大反馈框。
- 不再把过程藏到不伦不类的卡片中。
- 过程轨迹直接作为回答的一部分展示。

### 2. Runtime Command Bar

固定在输入区上方或下方。

动作：

- Validate
- Run 1 Event
- Run 10 Events
- Viewer
- Refresh
- Reset

约束：

- Run 是正常模拟动作，不用阻断式文案恐吓用户。
- Run 前自动 validate/apply/initialize。
- 如果配置缺失，显示缺失字段和下一步建议。

### 3. Context Rail

右侧窄栏，仅放高价值状态。

内容：

- Session / model / adapter。
- 当前设计摘要。
- Runtime result 摘要。
- 配置完成度。

约束：

- 不展示大段 raw JSON。
- 不做多层卡片堆叠。

### 4. Evidence Drawer

底部或右侧可展开。

内容：

- Recommended config JSON。
- Runtime state。
- Geant4 log。
- Internal trace。

约束：

- 默认关闭。
- 用于审查，不用于普通流程。

## 前端状态模型

保留最小状态：

- `sessionId`
- `lang`
- `lastDesign`
- `lastRecommendedConfig`
- `lastRuntimeState`
- `lastRuntimeReport`
- `lastActivity`
- `sending`

## API 边界

必须保留：

- `/api/simulation/design`
- `/api/geant4/intent`
- `/api/geant4/validate`
- `/api/geant4/apply`
- `/api/geant4/initialize`
- `/api/geant4/run`
- `/api/geant4/summary`
- `/api/geant4/state`
- `/api/geant4/log`
- `/api/geant4/viewer/open`
- `/api/config/summary`
- `/api/runtime`
- `/api/reset`

不使用：

- `/api/step_async` 作为普通对话入口。

## 用户流程

### 新模拟目标

1. 用户输入自然语言目标。
2. 前端调用 `/api/geant4/intent`。
3. 对 `config_mutation` 或 `normal_chat`，默认进入 `/api/simulation/design`。
4. 显示方案、可运行性、缺失项、下一步。
5. 保存 recommended config 到前端状态。

### 询问当前配置

1. 调用 `/api/config/summary`。
2. 只读展示，不写 session。

### 询问运行结果

1. 调用 `/api/geant4/summary`。
2. 只读展示，不触发 run。

### 运行模拟

1. 从当前 config 或 latest recommended config 取 patch。
2. 调用 `/api/geant4/validate`。
3. validate 通过后调用 `/api/geant4/run`。
4. 显示 runtime smoke report 和 result explanation。

## 视觉方向

采用 `Instrument Console` 风格：

- 深色主背景。
- 高对比文字。
- 单一强调色：cyan/green for active system，amber for caution。
- 主对话清晰，不用彩色渐变堆叠。
- Activity trace 使用细线 timeline。
- Context rail 使用 compact metrics。

## 验收标准

- 输入完整 prompt 后能生成设计方案和 recommended config。
- 继续说“运行 1 个事件”能进入 runtime，不报缺字段。
- 多轮修改不丢上下文。
- 中文无乱码。
- 窗口缩放到 900px、640px 仍可用。
- Activity trace 显式可读。
- Raw JSON 不污染主对话。
- JS 语法检查通过。
- 前端静态合约测试通过。
- 全量测试通过。
- Secret scan 通过。

