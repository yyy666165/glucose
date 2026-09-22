> 结合**数据驱动神经网络的拟合能力与领域生理先验知识**，构建个性化 Neural‑ODE 血糖预测模型：
> 
> ##1
> - 不再使用全局固定 PK 参数，实现患者级个性化动力学；
> - 显式引入上下文环境特征（时间节律、睡眠、黎明现象）；
> - 将胰岛素效应、人体葡萄糖自稳态调节、运动消耗作为可解释先验约束注入微分方程；
> - 使用 ODE 积分完成连续时序演化，输出平滑生理轨迹。

## 2. Model Architecture Overview

整体模型由四大组件构成：`Context Encoder`、`Patient Personalization Module`、`Dynamics Network(FiLM‑augmented)`、`Physiological‑prior ODE System`。

### 2.1 Context Encoder 上下文编码器

**Rationale（设计理由）：**血糖变化受昼夜节律、睡眠状态、黎明现象、当前血糖区间等环境因素影响，这类信息属于环境上下文，不属于长期患者固有属性，随时刻动态变化。

输入 5 维上下文特征：
`[hour_sin, hour_cos, sleep, dawn, zone]`
网络结构：`Linear(5→64) → Linear(64→64)`，输出上下文隐向量 \(\boldsymbol \theta \in \mathbb R^{64}\)，表征当前时刻环境信息，作为动力学网络输入一部分。

### 2.2 Patient Personalization Module 患者个性化模块

**Rationale（设计理由）：**不同患者具备差异化药代动力学特性（胰岛素吸收、消除速率）；同时患者个体差异应当调制动力学网络行为，而不只是简单增加特征。

模块输入：`patient_id`

1. `Embedding(29→16)`：每个患者映射为 16 维个体嵌入向量，编码患者固有生理特质；该 embedding 会直接拼入 ODE 输入特征。
2. `FiLM head(16→256)`：从患者 embedding 生成 FiLM 参数 `gamma, beta`，对动力学网络每一层做**条件缩放偏移**，实现网络行为个性化调制。
3. `PK head(16→5)`：药代动力学头，从患者 embedding 学习个性化 PK 参数，如胰岛素吸收速率 \(k_a\)、消除速率 \(k_e\) 等，替代传统全局固定常数。

> 
> 关键点：个性化分为两条路径：①显性学习生理 PK 参数；②隐性调制神经网络动力学权重行为（FiLM）。

### 2.3 FiLM‑augmented Dynamics Network 动力学网络

**Rationale（设计理由）：**人体血糖变化存在大量未完全建模的复杂非线性交互，完全依靠手工生理公式不足以刻画全部动力学；使用神经网络拟合导数项 \(dG/dt_{NN}\)，同时通过 FiLM 引入个体条件，保证网络行为随患者不同而变化。

拼接输入特征 \([G, \boldsymbol u, \boldsymbol \theta]\)，维度 72：

- G：当前血糖标量
- \(\boldsymbol u\)：外部控制输入（胰岛素、碳水等）
- \(\boldsymbol \theta\)：context encoder 输出上下文向量

网络结构：

- Layer1: `Linear(72→64) + ReLU`，接受 FiLM (gamma,beta) 调制
- Layer2: `Linear(64→64) + ReLU`，接受 FiLM (gamma,beta) 调制
- Layer3: `Linear(64→1) + tanh * DA`
输出：\(dG/dt_{NN}\)，神经网络拟合得到血糖变化率的基础项；`DA`限定最大血糖变化幅度，tanh 做幅值约束，防止神经网络输出无界导数。

### 2.4 Physiological‑prior ODE System 带生理先验约束的 ODE 系统

**Rationale（设计理由）：**纯神经网络输出导数容易出现违背医学常识结果；将可解释生理先验作为加性约束项融合进总导数，实现黑箱数据驱动 + 白箱生理先验混合建模。

总血糖导数由神经网络项叠加多项生理先验项构成：

1. **胰岛素效应项 \(IOB \times k\)**：剩余活性胰岛素降糖作用；
2. **胰岛素净作用 net**：进一步表征胰岛素综合效应；
3. **葡萄糖自稳态调节项 \(p \times \Delta G\)**：模拟人体自身血糖反馈调节，血糖偏高趋向降糖、偏低趋向升糖；
4. **运动消耗项 \(-\gamma \times EX\)**：建模运动带来葡萄糖消耗，区分快慢两相运动状态 \(EX_{fast},EX_{slow}\)。

\(\frac{dG}{dt} = dG/dt_{NN} + \text{physiological terms}\)整个 ODE 系统状态向量共 7 维，包含：血糖 G、IOB 相关状态、COB 相关状态、快慢运动 EX 状态。
将 7 维状态送入 `odeint` 求解器做积分推演，得到下一时刻全部状态；迭代推演得到未来完整血糖时序轨迹。

## 3. Core Idea / Design Philosophy

> 
> Hybrid modeling: **Neural network for unknown nonlinear dynamics + hand‑crafted physiological prior constraints for interpretability**

- ❌ 不采用全局固定 PK 参数；通过个性化模块实现个体药代参数学习以及网络条件调制；
- ✅ Context Encoder 刻画时刻相关环境扰动（昼夜、睡眠、黎明现象）；
- ✅ FiLM 实现动力学网络行为个性化；PK head 输出显性生理参数；
- ✅ ODE 连续演化，而非离散步时序模型；生成平滑连续轨迹；
- ✅ 融合多项生理先验作为约束，缓解纯数据驱动模型的无约束问题，改善极端血糖场景预测。
