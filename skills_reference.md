# 强制调用 / 显式触发 Skill 参考

## BLOCKING 级（遇到关键词必须先激活，再回答）

| Skill | 触发方式 | 作用 |
|-------|---------|------|
| **autoresearch** | `/autoresearch`, `$autoresearch plan/debug/fix/security/ship/scenario/predict/learn/reason/probe`，或在对话中提及 "autoresearch" + 任何目标/指标/任务 | 启动自主研究循环。根据子命令执行不同模式：plan(规划)、debug(调试)、fix(修复)、security(安全审计)、ship(发布)、scenario(场景分析)、predict(预测)、learn(学习)、reason(推理)、probe(探查)。**必须在生成任何其他响应前先激活** |

---

## 斜杠命令触发（需主动输入 /command）

| Skill | 触发方式 | 作用 |
|-------|---------|------|
| **loop** | `/loop 5m /foo`（默认 10 分钟间隔） | 设置定时循环任务。按指定间隔反复执行某个 prompt 或斜杠命令，适用于轮询状态、周期性检查等 |
| **init** | `/init` | 在项目根目录初始化一个 CLAUDE.md 文件，自动扫描代码库并生成项目文档，供后续会话参考 |
| **review** | `/review` | 审查当前分支的 Pull Request，给出代码评审意见 |
| **security-review** | `/security-review` | 对当前分支的待合并变更进行完整安全审查，识别潜在安全风险 |
| **simplify** | 需显式请求"简化代码"/"review for reuse" | 审查已变更的代码，检查复用性、质量和效率问题，并自动修复发现的问题 |
| **fewer-permission-prompts** | 需显式请求"减少权限提示" | 扫描对话历史中的只读 Bash/MCP 调用，生成白名单写入 .claude/settings.json，减少后续权限弹窗 |

---

## 需特定触发词的 Skill

| Skill | 触发词/方式 | 作用 |
|-------|-----------|------|
| **caveman** | `/caveman`、"caveman mode"、"talk like caveman"、"less tokens" | 超压缩沟通模式，削减约 75% token 用量（去掉填充词、冠词、客套），保留完整技术准确性 |
| **grill-me** | "grill me"、"stress-test this plan" | 对你的方案/设计进行 relentless 追问，直到达成共识，逐条理解决策树的每个分支 |
| **grill-with-docs** | "grill me against docs"、"stress-test against CONTEXT.md" | 与 grill-me 类似，但会对照项目已有的领域模型(CONTEXT.md)和决策记录(ADR)来挑战你的方案，并在决策明确时自动更新文档 |

---

## 备注

- 以上 skill 不会在日常对话中自动触发，需要你主动使用斜杠命令或说出特定触发词
- 其余 skill（如 diagnose、tdd、prototype、to-issues 等）基于自然语言匹配自动触发，无需专门调用
- document-skills:* 子系列根据文件类型或请求内容自动匹配
