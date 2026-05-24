import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchdiffeq import odeint


class PatientPersonalization(nn.Module):
    """患者个性化模块: Patient Embedding + FiLM 调制 + 患者特异药代参数生成

    核心思想:
    - 为每个患者学习一个低维嵌入向量，捕获其生理特征
    - FiLM: 用嵌入生成 scale/shift 参数调制 ODE 网络中间层，使动力学因患者而异
    - 药代参数生成: 从嵌入生成 IOB/COB 衰减参数和 ISF 基线，替换固定值
    """
    def __init__(self, num_patients=12, embed_dim=16, dynamics_hidden_dim=64,
                 correction_hidden_dim=64):
        super().__init__()
        self.num_patients = num_patients
        self.embed_dim = embed_dim

        # 患者嵌入
        self.patient_embedding = nn.Embedding(num_patients, embed_dim)

        # FiLM 生成器: 为主动力学网络生成 scale/shift
        # dynamics_net 有两个隐藏层，需要两对 FiLM 参数
        # 每对: (gamma, beta)，维度 = hidden_dim
        self.dynamics_film_gen = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, dynamics_hidden_dim * 4)  # 2层 × (gamma + beta)
        )
        self.dynamics_hidden_dim = dynamics_hidden_dim

        # 患者特异药代参数生成器
        # 输出 5 个参数: iob_peak, iob_duration, cob_peak, cob_duration, base_isf
        self.pk_param_gen = nn.Linear(embed_dim, 5)

        # 药代参数的生理合理范围 (min, max)
        # iob_peak: [0.5, 3.0]h, iob_duration: [3.0, 10.0]h
        # cob_peak: [0.3, 2.0]h, cob_duration: [2.0, 6.0]h
        # base_isf: [0.5, 2.0]
        self.pk_ranges = torch.tensor([
            [0.5, 3.0],   # iob_peak_time
            [3.0, 10.0],  # iob_duration
            [0.3, 2.0],   # cob_peak_time
            [2.0, 6.0],   # cob_duration
            [0.5, 2.0],   # base_isf
        ])
        # 初始化偏置使默认输出接近当前固定值
        # 当前固定值: iob_peak=1.5, iob_duration=6.0, cob_peak=1.0, cob_duration=4.0, base_isf=1.0
        # sigmoid(x) = 1.5 → x ≈ 0.0 (midpoint), sigmoid(x) = 0.5*(6-3)+3=4.5 → x ≈ 0.5
        defaults = [1.5, 6.0, 1.0, 4.0, 1.0]
        for i, (lo, hi) in enumerate(self.pk_ranges):
            mid = (defaults[i] - lo) / (hi - lo)
            mid = max(0.01, min(0.99, mid))
            # sigmoid_inv(mid) = log(mid / (1-mid))
            bias = float(np.log(mid / (1 - mid)))
            self.pk_param_gen.bias.data[i] = bias
            self.pk_param_gen.weight.data[i] *= 0.01  # 小初始化，开始时接近默认值

    def forward(self, patient_ids):
        """
        Args:
            patient_ids: (batch,) 患者索引 [0, num_patients)

        Returns:
            patient_embed: (batch, embed_dim)
            dynamics_film_params: dict with gamma1, beta1, gamma2, beta2
            pk_params: dict with iob_peak, iob_duration, cob_peak, cob_duration, base_isf
        """
        patient_embed = self.patient_embedding(patient_ids)  # (batch, embed_dim)

        # 生成 FiLM 参数
        d_film = self.dynamics_film_gen(patient_embed)  # (batch, hidden*4)
        hd = self.dynamics_hidden_dim
        dynamics_film = {
            'gamma1': 1.0 + d_film[:, :hd],            # 初始化接近 1 (不调制)
            'beta1': d_film[:, hd:2*hd] * 0.01,        # 初始化接近 0
            'gamma2': 1.0 + d_film[:, 2*hd:3*hd],
            'beta2': d_film[:, 3*hd:4*hd] * 0.01,
        }

        # 生成患者特异药代参数
        pk_raw = torch.sigmoid(self.pk_param_gen(patient_embed))  # (batch, 5) ∈ [0,1]
        pk_ranges = self.pk_ranges.to(patient_embed.device)
        pk_scaled = pk_raw * (pk_ranges[:, 1] - pk_ranges[:, 0]) + pk_ranges[:, 0]  # (batch, 5)

        pk_params = {
            'iob_peak_time': pk_scaled[:, 0],      # (batch,)
            'iob_duration': pk_scaled[:, 1],
            'cob_peak_time': pk_scaled[:, 2],
            'cob_duration': pk_scaled[:, 3],
            'base_isf': pk_scaled[:, 4],
        }

        return patient_embed, dynamics_film, pk_params


class InsulinEffectNetwork(nn.Module):
    """独立的胰岛素效应子网络，增强胰岛素对血糖动力学的影响力"""
    def __init__(self, hidden_dim=32):
        super().__init__()
        # 输入: [IOB_state, basal_rate, IOB_state, ISF, glucose_level]
        # IOB_state 同时代替旧的 recent_bolus_dose（内部PK状态已经编码了剂量信息）
        self.net = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # 输出在[-1,1]范围，乘以负系数确保降糖
        )

    def forward(self, iob, basal_rate, bolus_dose, isf, glucose):
        x = torch.cat([iob, basal_rate, bolus_dose, isf, glucose], dim=1)
        return self.net(x)  # (batch, 1)


class GlucoseODEFunc(nn.Module):
    """合并的 ODE 动力学: dG/dt = f(t,G,u,theta) + gate + prior

    原主分支 + 极端值分支合并为单一 ODE 函数:
    1. 主动力学网络 (带 FiLM 患者调制)
    2. 生理约束 (insulin/carb硬约束)
    3. 极端值门控 + 生理先验 (原 ExtremeCorrectionODEFunc)
    """
    def __init__(self, hidden_dim=32, control_dim=7):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.control_dim = control_dim

        self.dynamics_layer1 = nn.Linear(1 + control_dim + hidden_dim, hidden_dim)
        self.dynamics_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.dynamics_layer3 = nn.Linear(hidden_dim, 1)
        self.dynamics_amplitude = nn.Parameter(torch.tensor(2.0))

        self.insulin_net = InsulinEffectNetwork(hidden_dim // 2)

        # 硬约束系数 (全部可学习)
        self.insulin_constraint = nn.Parameter(torch.tensor(-5e-3))    # IOB 降糖
        self.basal_constraint = nn.Parameter(torch.tensor(-2e-3))     # Basal 降糖
        self.bolus_constraint = nn.Parameter(torch.tensor(-1e-2))     # Bolus 降糖
        self.carb_constraint = nn.Parameter(torch.tensor(3e-3))      # COB 升糖

        # 碳水补偿系数 (原固定值 ×5 和 ×0.1)
        self.carb_boost_threshold = nn.Parameter(torch.tensor(5.0))   # COB超过IOB的倍数
        self.carb_boost_scale = nn.Parameter(torch.tensor(0.1))      # 补偿强度

        self.insulin_scale = nn.Parameter(torch.tensor(1.0))

        # 葡萄糖自身调节: -p × (G - G_baseline)  (Bergman 最小模型)
        self.glucose_effectiveness = nn.Parameter(torch.tensor(0.005))  # p₁
        self.G_baseline = nn.Parameter(torch.tensor(110.0))            # 血糖稳态值

        # 运动效应参数 (两室: 直接耗糖 + 胰岛素敏感度提升)
        self.exercise_uptake = nn.Parameter(torch.tensor(0.3))         # γ: 直接耗糖强度
        self.isf_boost_scale = nn.Parameter(torch.tensor(0.2))        # β: ISF提升幅度
        self.ex_fast_decay = nn.Parameter(torch.tensor(0.3))          # k_ex_fast: 快速衰减 (~2h)
        self.ex_slow_decay = nn.Parameter(torch.tensor(0.03))         # k_ex_slow: 慢速衰减 (~18h)

        # ---- 极端值门控 (原 ExtremeCorrectionODEFunc) ----
        self.hypo_onset = nn.Parameter(torch.tensor(90.0))
        self.hypo_full = nn.Parameter(torch.tensor(55.0))
        self.hyper_onset = nn.Parameter(torch.tensor(200.0))
        self.hyper_full = nn.Parameter(torch.tensor(300.0))
        self.correction_amplitude = nn.Parameter(torch.tensor(3.0))

        self.trend_gate_net = nn.Sequential(
            nn.Linear(control_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self.trend_gain = nn.Parameter(torch.tensor(1.0))
        self.rebound_scale = nn.Parameter(torch.tensor(2.0))
        self.hyper_insulin_scale = nn.Parameter(torch.tensor(0.5))

    def _apply_film(self, h, film_params, layer_idx):
        gamma = film_params[f'gamma{layer_idx}']
        beta = film_params[f'beta{layer_idx}']
        return gamma * h + beta

    def _compute_gate(self, y, u):
        g = y.squeeze(-1)
        hypo = (self.hypo_onset - g) / (self.hypo_onset - self.hypo_full + 1e-6)
        hyper = (g - self.hyper_onset) / (self.hyper_full - self.hyper_onset + 1e-6)
        level = torch.max(torch.clamp(hypo, 0, 1), torch.clamp(hyper, 0, 1))
        trend = self.trend_gate_net(u).squeeze(-1)
        return (level * (1.0 + trend * self.trend_gain)).unsqueeze(-1)

    def _physiological_prior(self, G, IOB_state):
        g = G.squeeze(-1)
        rebound = torch.relu(1.0 - g / (self.hypo_full + 1e-6)) * torch.abs(self.rebound_scale)
        hyper = (g > self.hyper_onset).float()
        enhance = hyper * IOB_state.squeeze(-1) * torch.abs(self.hyper_insulin_scale)
        return (rebound - enhance).unsqueeze(-1)

    def forward(self, t, y):
        """7 维状态: [G, IOB_abs, IOB, COB_abs, COB, EX_fast, EX_slow]
        内部做 clamp（避免 odeint 后 in-place 操作破坏 autograd）"""
        G = y[:, 0:1].clamp(min=30.0, max=500.0)
        IOB_abs = y[:, 1:2].clamp(min=0.0)
        IOB = y[:, 2:3].clamp(min=0.0)
        COB_abs = y[:, 3:4].clamp(min=0.0)
        COB = y[:, 4:5].clamp(min=0.0)
        EX_fast = y[:, 5:6].clamp(min=0.0)
        EX_slow = y[:, 6:7].clamp(min=0.0)

        u = self._u          # [batch, 7] = [bolus_event, meal_event, exercise, basal, HR, ISF, delta_G]
        theta = self._theta  # [batch, hidden_dim]
        film = getattr(self, '_film_params', None)
        pk = getattr(self, '_pk_params', None)

        # ─── PK 参数: 从患者嵌入或默认值 ───
        if pk is not None:
            k_a_iob = 1.0 / (pk['iob_peak_time'] * 12.0).clamp(min=0.5).unsqueeze(-1)
            k_e_iob = 1.0 / (pk['iob_duration'] * 12.0).clamp(min=0.5).unsqueeze(-1)
            k_a_cob = 1.0 / (pk['cob_peak_time'] * 12.0).clamp(min=0.5).unsqueeze(-1)
            k_e_cob = 1.0 / (pk['cob_duration'] * 12.0).clamp(min=0.5).unsqueeze(-1)
        else:
            k_a_iob = 1.0 / (1.5 * 12)
            k_e_iob = 1.0 / (6.0 * 12)
            k_a_cob = 1.0 / (1.0 * 12)
            k_e_cob = 1.0 / (4.0 * 12)

        # ─── 两室 PK: IOB ───
        bolus_event = u[:, 0:1]
        dIOB_abs = -k_a_iob * IOB_abs + bolus_event * k_a_iob
        dIOB = k_a_iob * IOB_abs - k_e_iob * IOB

        # ─── 两室 PK: COB ───
        meal_event = u[:, 1:2]
        dCOB_abs = -k_a_cob * COB_abs + meal_event * k_a_cob
        dCOB = k_a_cob * COB_abs - k_e_cob * COB

        # ─── 运动 PK: 两室 (直接耗糖 + 胰岛素敏感) ───
        exercise_event = u[:, 2:3]                    # 当前运动强度
        k_ex_fast = torch.abs(self.ex_fast_decay)     # 快速衰减率
        k_ex_slow = torch.abs(self.ex_slow_decay)     # 慢速衰减率
        dEX_fast = -k_ex_fast * EX_fast + exercise_event * k_ex_fast
        dEX_slow = -k_ex_slow * EX_slow + exercise_event * k_ex_slow

        # ─── 血糖动力学 (ISF 被运动慢效提升) ───
        basal_rate = u[:, 3:4]
        isf_base = u[:, 5:6]
        isf_eff = isf_base * (1.0 + self.isf_boost_scale * EX_slow)  # 运动后 ISF↑

        NN_input = torch.cat([G, u, theta], dim=1)
        h = F.relu(self.dynamics_layer1(NN_input))
        if film is not None:
            h = self._apply_film(h, film, 1)
        h = F.relu(self.dynamics_layer2(h))
        if film is not None:
            h = self._apply_film(h, film, 2)
        dG_dt = torch.tanh(self.dynamics_layer3(h)) * torch.abs(self.dynamics_amplitude)
        dG_dt = torch.clamp(dG_dt, min=-50.0, max=50.0)

        # 硬约束 (全部可学习参数)
        dG_dt += (IOB * self.insulin_constraint + basal_rate * self.basal_constraint
                  + COB * torch.abs(self.carb_constraint))

        # 胰岛素效应网络 (使用运动增强后的 ISF)
        ie = self.insulin_net(IOB, basal_rate, IOB, isf_eff, G) * (IOB + basal_rate) * self.insulin_scale
        dG_dt += torch.clamp(ie, min=-30.0, max=30.0)

        # 碳水升糖补偿 (参数可学习)
        dG_dt += torch.clamp(COB - (IOB + basal_rate) * torch.abs(self.carb_boost_threshold), min=0) * torch.abs(self.carb_boost_scale)

        # 葡萄糖自身调节 (单向: 仅在高血糖时拉回, 低血糖时不推升)
        # T1D 患者的反向调节受损, 不能依赖自调节回升
        zone_factor = torch.clamp(1.0 - (G - self.G_baseline) / 60.0, min=0.0, max=1.0)
        dG_dt += -self.glucose_effectiveness * torch.relu(G - self.G_baseline) * zone_factor

        # 运动直接耗糖 (非胰岛素介导的葡萄糖摄取)
        dG_dt += -torch.abs(self.exercise_uptake) * EX_fast

        # 极端值门控
        gate = self._compute_gate(G, u)
        prior = self._physiological_prior(G, IOB)
        dG_dt += gate * torch.tanh(dG_dt) * torch.abs(self.correction_amplitude) + prior

        return torch.cat([dG_dt, dIOB_abs, dIOB, dCOB_abs, dCOB, dEX_fast, dEX_slow], dim=1)


class NeuralODEGlucosePredictor(nn.Module):
    """单分支 Neural ODE 血糖预测模型 + 患者个性化
    - ODE 函数: GlucoseODEFunc (含主动力学+门控+生理先验)
    - 患者个性化: Patient Embedding + FiLM 调制 + 患者特异药代参数
    """
    def __init__(self, context_dim=5, hidden_dim=32, control_dim=7,
                 num_patients=12, patient_embed_dim=16,
                 integration_method='euler', rtol=1e-3, atol=1e-3):
        super().__init__()
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.control_dim = control_dim
        self.num_patients = num_patients
        self.patient_embed_dim = patient_embed_dim
        self.ode_func = GlucoseODEFunc(hidden_dim, control_dim)
        self.integration_method = integration_method
        self.rtol = rtol
        self.atol = atol

        # 上下文编码器（主分支和极端值分支共享）
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )


        # 患者个性化模块
        self.patient_personalization = PatientPersonalization(
            num_patients=num_patients,
            embed_dim=patient_embed_dim,
            dynamics_hidden_dim=hidden_dim,

        )

    def forward(self, initial_glucose, control_sequence, context, patient_ids=None,
                recursive_steps=10):
        """
        Hybrid forward: 前 recursive_steps 步递归，之后分块并行
        - 内部状态扩展为 7 维: [G, IOB_abs, IOB, COB_abs, COB, EX_fast, EX_slow]

        Args:
            initial_glucose: (batch, 1)
            control_sequence: (batch, seq_len, control_dim)
            context: (batch, context_dim)
            patient_ids: (batch,) 患者索引
            recursive_steps: 前 N 步使用递归 (默认 10)
        """
        seq_len = control_sequence.size(1)
        device = initial_glucose.device

        theta = self.context_encoder(context)
        film_params = None
        pk_params = None
        if patient_ids is not None:
            film_out = self.patient_personalization(patient_ids)
            if len(film_out) == 3:
                _, film_params, pk_params = film_out
            else:
                _, film_params = film_out, None

        # 扩展初始状态: [G, IOB_abs=0, IOB=0, COB_abs=0, COB=0, EX_fast=0, EX_slow=0]
        batch_size = initial_glucose.size(0)
        zeros = torch.zeros(batch_size, 6, device=device)
        y = torch.cat([initial_glucose, zeros], dim=1)  # [batch, 7]

        t_span = torch.tensor([0.0, 1.0], device=device)
        predictions = []
        n_rec = min(recursive_steps, seq_len)

        # ---- Phase 1: 递归 ----
        for i in range(n_rec):
            self.ode_func._u = control_sequence[:, i, :]
            self.ode_func._theta = theta
            self.ode_func._film_params = film_params
            self.ode_func._pk_params = pk_params

            y = odeint(self.ode_func, y, t_span,
                       method=self.integration_method,
                       rtol=self.rtol, atol=self.atol)[-1]
            # ODE 函数内部已做 clamp
            predictions.append(y[:, 0:1])

        # ---- Phase 2: 分块并行 (仅取 G 分量用于预测, 但需要 5 维状态继续积分) ----
        n_para = seq_len - n_rec
        if n_para > 0:
            # 继续使用完整 5 维状态
            y_base = y
            chunk_size = 2

            for cs in range(0, n_para, chunk_size):
                ce = min(cs + chunk_size, n_para)
                c = ce - cs
                ti = n_rec + cs

                y_exp = y_base.repeat_interleave(c, dim=0)
                u_exp = control_sequence[:, ti:ti+c, :].contiguous().view(-1, self.control_dim)
                theta_exp = theta.repeat_interleave(c, dim=0)

                film_exp = None
                if film_params is not None:
                    film_exp = {k: v.repeat_interleave(c, dim=0) for k, v in film_params.items()}

                # PK 参数也需要重复
                pk_exp = None
                if pk_params is not None:
                    pk_exp = {k: v.repeat_interleave(c, dim=0) for k, v in pk_params.items()}

                self.ode_func._u = u_exp
                self.ode_func._theta = theta_exp
                self.ode_func._film_params = film_exp
                self.ode_func._pk_params = pk_exp

                y_all = odeint(self.ode_func, y_exp, t_span,
                               method=self.integration_method,
                               rtol=self.rtol, atol=self.atol)[-1]
                # ODE 函数内部已做 clamp
                y_all = y_all.view(-1, c, 7)

                for j in range(c):
                    predictions.append(y_all[:, j, 0:1])

        predicted = torch.stack(predictions, dim=0).squeeze(-1).permute(1, 0)
        return predicted


    def get_patient_pk_params(self, patient_ids):
        """获取患者特异药代动力学参数"""
        _, _, pk_params = self.patient_personalization(patient_ids)
        return pk_params

    def counterfactual(self, initial_glucose, original_control_sequence, modified_control_sequence, context, patient_ids=None):
        return self.forward(initial_glucose, modified_control_sequence, context, patient_ids)


if __name__ == "__main__":
    batch_size = 32
    seq_len = 24
    context_dim = 5
    control_dim = 7  # [bolus_event, meal_event, exercise, basal, HR, ISF, delta_G]
    num_patients = 12

    model = NeuralODEGlucosePredictor(
        context_dim=context_dim, hidden_dim=32, control_dim=control_dim,
        num_patients=num_patients, patient_embed_dim=16
    )
    initial_glucose = torch.randn(batch_size, 1) * 50 + 150
    control_sequence = torch.randn(batch_size, seq_len, control_dim)
    context = torch.randn(batch_size, context_dim)
    patient_ids = torch.randint(0, num_patients, (batch_size,))

    # 无患者ID调用
    out_no_pid = model(initial_glucose, control_sequence, context)
    print(f"无患者ID输出形状: {out_no_pid.shape}")

    # 有患者ID调用
    out = model(initial_glucose, control_sequence, context, patient_ids)
    print(f"有患者ID输出形状: {out.shape}")

    # 患者特异药代参数
    pk_params = model.get_patient_pk_params(patient_ids[:4])
    print(f"患者0药代参数: IOB峰值={pk_params['iob_peak_time'][0]:.2f}h, "
          f"IOB持续={pk_params['iob_duration'][0]:.2f}h, "
          f"COB峰值={pk_params['cob_peak_time'][0]:.2f}h, "
          f"COB持续={pk_params['cob_duration'][0]:.2f}h, "
          f"ISF基线={pk_params['base_isf'][0]:.2f}")

    # 反事实：增加bolus事件，应导致血糖下降
    modified = control_sequence.clone()
    modified[:, :, 0] += 4.0  # 增加4U bolus事件 (index 0 = bolus_event)
    cf = model.counterfactual(initial_glucose, control_sequence, modified, context, patient_ids)
    diff = (cf - out).mean().item()
    print(f"增加bolus后血糖平均变化: {diff:.2f} (应为负值)")

    # 验证内部状态: 在 forward 过程中 IOB/COB 应该被正确计算
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")

    # 验证 PK 状态: 10U bolus
    single_glucose = torch.tensor([[150.0]], dtype=torch.float32)
    single_ctx = torch.randn(1, context_dim)
    single_pid = torch.tensor([0])
    single_ctrl = torch.zeros(1, 24, control_dim)
    single_ctrl[0, 0, 0] = 10.0
    with torch.no_grad():
        out_b = model(single_glucose, single_ctrl, single_ctx, single_pid)
    print(f"10U bolus: 初始=150, 最终={out_b[0, -1]:.1f}")

    # 验证运动效应: 运动 vs 不运动
    single_ctrl2 = torch.zeros(1, 24, control_dim)
    single_ctrl2[0, 0, 0] = 5.0    # 5U bolus
    single_ctrl2[0, 0, 2] = 5.0    # 运动强度5持续24步
    with torch.no_grad():
        out_noex = model(single_glucose, torch.zeros(1,24,control_dim), single_ctx, single_pid)
        out_ex = model(single_glucose, single_ctrl2, single_ctx, single_pid)
    print(f"无干预: 初始=150, 最终={out_noex[0, -1]:.1f}")
    print(f"5U+运动: 初始=150, 最终={out_ex[0, -1]:.1f}")
    diff = out_ex[0, -1].item() - out_noex[0, -1].item()
    print(f"运动降糖效果: {diff:.1f} mg/dL (应为负值)")

    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

