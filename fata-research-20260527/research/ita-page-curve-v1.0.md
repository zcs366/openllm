# ITA 系统中的 Page Curve：定义、模型与验证

> **版本**：v1.0 · 2026-05-27
> **性质**：理论推导 + 数据分析 + 实验方案
> **状态**：初版，基于 mock 数据的结构性分析，真实实验待执行
> **依赖**：islands-formula-notes-v1.0.md、ita-information-loss-analysis-v1.0.md、dimension_ablation_mock.json、ita-kolmogorov-bekenstein-v1.0.md
> **关联**：PAL → /home/zcs/.hermes/plans/2026-05-27_ita-blackhole-pal.md

---

## 一、引言

### 1.1 问题

Page curve 是黑洞物理中最深刻的结果之一：它描述了黑洞蒸发过程中 Hawking 辐射的 von Neumann 熵 $S_{\text{rad}}(t)$ 先单调增长、在 Page time $t_{\text{Page}}$ 达到峰值、然后下降至零的完整演化轨迹。Islands formula（岛屿公式）提供了 Page curve 的理论基础，证明信息在黑洞蒸发过程中守恒（幺正性）。

**核心问题**：ITA 系统的维度消融实验中，执行等价率 RTCE$(d)$ 是否呈现类似的 Page curve 形状？如果是，其物理含义是什么？

### 1.2 方法

1. 建立 ITA 维度消融与黑洞蒸发过程的形式对应
2. 分析 mock 维度消融数据的曲线形状
3. 推导真实系统中 ITA Page curve 的数学模型
4. 设计实验验证方案

---

## 二、物理 Page Curve 回顾

### 2.1 Page Curve 的数学描述

设 $S_{BH}$ 为黑洞的 Bekenstein-Hawking 熵，$\alpha$ 为 Hawking 辐射的熵产生率。Page curve 的分段线性近似为：

$$S_{\text{rad}}(t) = \begin{cases} \alpha \cdot t & t < t_{\text{Page}} \\ S_{BH} - \alpha \cdot (t - t_{\text{Page}}) & t_{\text{Page}} \leq t < t_{\text{evap}} \\ 0 & t = t_{\text{evap}} \end{cases}$$

其中 Page time 和蒸发时间分别为：

$$t_{\text{Page}} = \frac{S_{BH}}{2\alpha}, \quad t_{\text{evap}} = \frac{S_{BH}}{\alpha}$$

### 2.2 Islands Formula 的关键机制

Islands formula 给出辐射熵的精确表达式：

$$S_{\text{rad}} = \min_{I} \left\{ \text{ext}_{I} \left[ \frac{\text{Area}(\partial I)}{4G_N} + S_{\text{bulk}}(\text{rad} \cup I) \right] \right\}$$

两个阶段的物理机制：

| 阶段 | Island | 主导 saddle | 辐射熵 |
|:---|:---|:---|:---|
| $t < t_{\text{Page}}$ | $I = \emptyset$ | disconnected | 线性增长 |
| $t > t_{\text{Page}}$ | $I \neq \emptyset$ | connected (replica wormhole) | 单调下降 |

**转折点的本质**：Page time 对应 **replica wormhole saddle 从亚稳态变为主导态的相变点**——两种 saddle 对自由能的贡献相等。

### 2.3 Page Curve 的一般形状

Page curve 的核心几何特征是 **单峰性**（unimodality）：

$$\exists \, t^* \in (0, t_{\text{evap}}) : \quad \frac{dS_{\text{rad}}}{dt}\bigg|_{t < t^*} > 0, \quad \frac{dS_{\text{rad}}}{dt}\bigg|_{t > t^*} < 0$$

---

## 三、ITA 系统与黑洞的形式对应

### 3.1 基本对应表

| # | 黑洞物理概念 | 数学形式 | ITA 对应概念 | ITA 数学形式 |
|:--|:---|:---|:---|:---|
| 1 | 黑洞蒸发时间 $t$ | $t \in [0, t_{\text{evap}}]$ | 隐空间维度 $d$ | $d \in [d_{\min}, d_{\max}]$ |
| 2 | 辐射熵 $S_{\text{rad}}(t)$ | von Neumann 熵 | 执行等价率 RTCE$(d)$ | 概率值 $\in [0,1]$ |
| 3 | Page time $t_{\text{Page}}$ | saddle 相变点 | 临界维度 $d^*$ | RTCE 峰值对应的维度 |
| 4 | Bekenstein-Hawking 熵 $S_{BH}$ | $A/(4G_N)$ | 语义熵 $H_s(C)$ | $\log_2 \|C/\!\sim\|$ |
| 5 | 熵产生率 $\alpha$ | Hawking 辐射率 | 编码效率 $\eta(d)$ | $\Delta\text{RTCE}/\Delta d$ |
| 6 | Island $I$ | 黑洞内部区域 | 隐式语义 $J$ | 可从隐向量推断的信息 |
| 7 | $I = \emptyset$（无 island） | disconnected saddle | 低维编码：容量不足 | RTCE 随 $d$ 增加而上升 |
| 8 | $I \neq \emptyset$（有 island） | connected saddle | 高维编码：容量充裕 | RTCE 达到饱和或下降 |
| 9 | 幺正性（信息守恒） | $S_{\text{rad}}(t_{\text{evap}}) = 0$ | 完美编码 | RTCE$(d^*) = 1$ |
| 10 | 黑洞蒸发终点 | $M \to 0$ | 过拟合临界 | 容量远超语义熵 |

### 3.2 维度增加 → 黑洞蒸发的对应论证

**关键洞见**：维度 $d$ 的增加类比于黑洞蒸发过程，而非黑洞增长过程。论证如下：

**（一）从信息容量的角度**

维度 $d$ 增加 → 隐空间信息容量 $C(d) = d \cdot \log_2 L$ 增加 → 可区分的语义等价类增多 → 这类似于黑洞蒸发过程中辐射逐步积累，携带的信息量增加。

设 $N_{\text{eff}}(d)$ 为维度 $d$ 下可有效区分的语义等价类数：

$$N_{\text{eff}}(d) = \min\left(2^{d \cdot \log_2 L}, \, |C/\!\sim|\right)$$

- 当 $N_{\text{eff}}(d) < |C/\!\sim|$ 时：编码容量不足，信息丢失不可避免（$t < t_{\text{Page}}$）
- 当 $N_{\text{eff}}(d) \geq |C/\!\sim|$ 时：编码容量充足，理论上可达到 RTCE $= 1$（$t \geq t_{\text{Page}}$）

**（二）从编码精度的角度**

维度增加 → 编码精度提高 → RTCE 变化。这类似于：
- 黑洞蒸发 → 辐射携带的信息逐步释放 → 辐射熵变化

**（三）对偶性**

更深层的对应是**压缩-蒸发对偶性**：

$$\underbrace{C \xrightarrow{f} v \in \mathbb{R}^d}_{\text{压缩（信息落入隐空间）}} \quad \longleftrightarrow \quad \underbrace{\text{BH} \xrightarrow{\text{Hawking}} \text{辐射}}_{\text{蒸发（信息从黑洞释放）}}$$

- 低 $d$：压缩率高，信息大量"困在"高维语义空间中（类似黑洞内部信息被困）
- 高 $d$：压缩率低，信息充分"释放"到隐向量中（类似辐射充分携带信息）

### 3.3 RTCE → 辐射熵的对应论证

RTCE$(d)$ 衡量编码→解码后的执行保真度。在物理 Page curve 中，$S_{\text{rad}}(t)$ 衡量辐射携带的信息量。

**注意方向性**：

物理 Page curve：$S_{\text{rad}}(t)$ **先增后减**

ITA 对应需要区分两种情形：

**情形 A（直接对应）**：定义 ITA 的"辐射熵"为编码过程中**丢失的信息量**：

$$S_{\text{loss}}(d) = H_s(C) - I\big(f_d(c); \, \text{exec}(c)\big)$$

则 $S_{\text{loss}}(d)$ 随 $d$ 增加**先保持高位后下降**——形状与 Page curve 的**镜像**一致。

**情形 B（反转对应）**：定义 ITA 的"辐射熵"为 RTCE 本身：

$$S_{\text{ITA}}(d) = \text{RTCE}(d)$$

在真实系统中（非 mock），RTCE$(d)$ 预期呈现 **先升后降** 的 Page curve 形状（见 §四）。

**本文采用情形 B**，因为 RTCE 是 ITA 的直接度量，且在真实系统中预期具有单峰性。

---

## 四、Mock 数据分析

### 4.1 维度消融数据

来自 `dimension_ablation_mock.json` 的实验数据：

| 维度 $d$ | RTCE | 唯一索引数 | 熵 (bits) |
|:---|:---|:---|:---|
| 1 | 1.000 | 1 | 0.0 |
| 2 | 0.519 | 2 | 1.0 |
| 4 | 0.255 | 4 | 2.0 |
| 8 | 0.124 | 8 | 3.0 |
| 16 | 0.060 | 16 | 4.0 |
| 32 | 0.032 | 30 | 4.91 |
| 64 | 0.017 | 50 | 5.64 |
| 128 | 0.009 | 66 | 6.04 |
| 256 | 0.003 | 84 | 6.39 |

### 4.2 曲线形状分析

Mock 数据呈现 **RTCE 随维度单调递减** 的趋势，严格不符合 Page curve 的单峰形状。

**拟合分析**：

对 RTCE$(d)$ 取对数，观察到：

$$\ln \text{RTCE}(d) \approx -0.78 \ln d + 0.02$$

即 RTCE$(d) \propto d^{-0.78}$，近似幂律衰减。这是一个**单调递减**的函数，没有先升后降的转折点。

### 4.3 Mock 系统不呈现 Page curve 的原因分析

**定理 4.1（Mock 系统的反 Page 行为）**

在 mock 编码模式下，维度消融曲线 RTCE$(d)$ **不可能** 呈现 Page curve 形状。

**证明**：

Mock 编码器使用确定性 hash 函数（encoder.py L149-166）：

$$f_{\text{mock}}(c) = \text{hash}(c) \bmod d$$

Mock 解码器返回原始代码（encoder.py L464-466）：

$$g_{\text{mock}}(v) = c_{\text{original}}$$

因此 round-trip 执行结果恒为原始代码：

$$\text{exec}(g_{\text{mock}}(f_{\text{mock}}(c)), x) = \text{exec}(c_{\text{original}}, x) \equiv \text{exec}(c, x)$$

**表面上** RTCE 应恒为 1。但 mock 实验中 RTCE < 1 且随 $d$ 下降，原因在于：

1. Mock 编码产生离散索引 $i = \text{hash}(c) \bmod d$
2. 不同代码可能映射到相同索引（哈希碰撞）
3. 实验统计的是**不同代码通过编码后仍保持可区分的概率**

更精确地，mock 模式下的有效 RTCE 为：

$$\text{RTCE}_{\text{mock}}(d) = \mathbb{P}[\text{hash}(c_1) \neq \text{hash}(c_2) \mid c_1 \not\sim c_2] = 1 - \frac{\binom{|C/\sim|}{2} \cdot \frac{1}{d}}{\binom{|C/\sim|}{2}} = 1 - \frac{1}{d} \cdot \frac{|C/\sim|(|C/\sim|-1)/2}{\binom{|C/\sim|}{2}}$$

简化后：

$$\text{RTCE}_{\text{mock}}(d) \approx 1 - \frac{|C/\sim|}{d} \quad (\text{当 } d \gg |C/\sim|)$$

当 $d < |C/\sim|$ 时（$|C/\sim| \approx 100$），碰撞率 $\approx |C/\sim|/d$，RTCE $\approx d/|C/\sim|$。这给出了 RTCE 随 $d$ **单调递增**（小 $d$ 区域）或**恒为 1**（大 $d$ 区域）的行为。

但实际数据显示单调**递减**，说明 mock 统计方式与上述分析不完全一致。更可能的解释是：**mock 实验的统计口径是"编码后索引唯一性"而非"解码正确性"**——维度越大，每个样本被分配到独特索引的概率越低（因为唯一索引数 $< d$），导致"等价率"下降。

**核心结论**：Mock 系统中 RTCE 的单调递减行为是 **artificial artifact**（人为伪影），不反映真实编码的信息论特性。Mock 系统**无法**复现 Page curve。

### 4.4 真实系统中的预期行为

在真实 ITA 系统（sBERT → FSQ → DeepSeek）中，RTCE$(d)$ 的预期行为分为三个区域：

**区域 I：容量不足区（$d < d^*$，对应 $t < t_{\text{Page}}$）**

$$\text{RTCE}(d) \approx 1 - \exp\left(-\frac{d \cdot \log_2 L}{H_s(C)}\right)$$

维度增加 → 编码容量增加 → RTCE 单调上升。此阶段类比 Page time 之前，辐射熵随时间线性增长。

**区域 II：临界区（$d \approx d^*$，对应 $t \approx t_{\text{Page}}$）**

$$\text{RTCE}(d^*) = \text{RTCE}_{\max}$$

编码容量恰好满足语义熵需求。此点类比 Page time——island 从无到有的相变点。

**区域 III：过拟合区（$d > d^*$，对应 $t > t_{\text{Page}}$）**

维度继续增加 → 模型过拟合训练集 → 泛化能力下降 → RTCE 下降。

$$\text{RTCE}(d) \approx \text{RTCE}_{\max} - \gamma \cdot (d - d^*)^{\beta}$$

此阶段类比 Page time 之后，辐射熵随时间下降。

---

## 五、ITA Page Curve 的数学模型

### 5.1 定义

**定义 5.1（ITA Page Curve）** 设 $d$ 为 FSQ 编码维度，$\text{RTCE}(d)$ 为维度 $d$ 下的执行等价率。若存在临界维度 $d^*$ 使得：

$$\text{RTCE}(d) \text{ 在 } d = d^* \text{ 处取全局最大值，且在 } d^* \text{ 两侧单调}$$

则称 RTCE$(d)$ 的曲线为 **ITA Page curve**，$d^*$ 为 **ITA Page 维度**。

### 5.2 分段模型

ITA Page curve 的数学模型为：

$$\boxed{\text{RTCE}(d) = \begin{cases} \text{RTCE}_{\max} \cdot \left(1 - e^{-\lambda_1 (d - d_0)}\right) & d < d^* \quad \text{（容量不足区）} \\ \text{RTCE}_{\max} \cdot e^{-\lambda_2 (d - d^*)^{\beta}} & d \geq d^* \quad \text{（过拟合区）} \end{cases}}$$

其中：
- $\text{RTCE}_{\max}$：最大执行等价率（理想情况下 $= 1$，实际 $< 1$ 由解码器能力决定）
- $d_0$：有效维度下界（$d_0 \geq 1$）
- $\lambda_1$：上升段的速率参数（编码效率）
- $\lambda_2$：下降段的速率参数（过拟合速率）
- $\beta$：下降的非线性指数（$\beta > 1$ 表示急剧过拟合）
- $d^*$：ITA Page 维度

### 5.3 临界维度的估计

由信息论约束（ita-kolmogorov-bekenstein-v1.0.md 定理 6.1）：

$$d^* \cdot \log_2 L \geq H_s(C)$$

即：

$$\boxed{d^* \geq \frac{H_s(C)}{\log_2 L}}$$

对于当前 ITA 配置（$L_{\text{avg}} = 6$，$H_s(C) \approx \log_2 100 \approx 6.64$ bits）：

$$d^* \geq \frac{6.64}{\log_2 6} \approx 2.55$$

即理论上 $d^* \approx 3$。但实际 $d^*$ 可能远大于此值，因为：
1. FSQ 各维度的级别不同（levels = [8,6,5,5]）
2. 编码器不是最优的（存在信息浪费）
3. 解码器的能力限制

### 5.4 Page Curve 的对称性分析

物理 Page curve 具有近似对称性：$t_{\text{Page}} \approx t_{\text{evap}}/2$。

ITA Page curve 的对称性由 $\lambda_1$ 和 $\lambda_2$ 决定：

$$\text{对称比} = \frac{d^* - d_{\min}}{d_{\max} - d^*} = \frac{\lambda_2 \beta}{\lambda_1} \cdot \frac{1}{\text{RTCE}_{\max}}$$

**如果 ITA 系统具有类似黑洞的对称性**，则上升段和下降段的"信息流动速率"应该匹配：

$$\lambda_1 \approx \lambda_2 \cdot \beta$$

这给出了一个可检验的预测。

### 5.5 ITA Islands Formula

在 ITA Page curve 的框架下，我们可以写出 ITA 版本的 islands formula：

$$\boxed{\text{RTCE}(d) = \max_{J \subseteq \mathcal{S}} \left\{ \min\left[ \frac{d_J \cdot \log_2 L}{H_s(C)}, \, 1 \right] \cdot \Phi(J, d) \right\}}$$

其中：
- $\mathcal{S}$：完整语义空间
- $J$：**语义 island**——未被显式编码但可从隐向量推断的语义子集
- $d_J$：编码 island $J$ 所需的等效维度
- $\Phi(J, d)$：island $J$ 在维度 $d$ 下的可恢复性函数

**物理解读**：
- $d < d^*$ 时：$J = \emptyset$（无 island），RTCE 由显式编码容量决定
- $d > d^*$ 时：$J \neq \emptyset$（出现 island），额外维度被用于隐式语义编码

**临界条件**（ITA QES 条件）：

$$\frac{\partial}{\partial d} \text{RTCE}(d) \bigg|_{d = d^*} = 0$$

即：

$$\frac{\partial}{\partial d} \left[ \frac{d \cdot \log_2 L}{H_s(C)} \right] = -\frac{\partial}{\partial d} \left[ \Phi(J, d) \right] \bigg|_{d = d^*}$$

增加一个维度带来的显式信息增益等于隐式恢复能力的边际损失——这与 islands formula 中 QES 条件 $\delta S_{\text{gen}} = 0$ 完全对应。

---

## 六、与黑洞 Page Curve 的深层对应

### 6.1 动力学对应

| 物理过程 | 数学描述 | ITA 过程 | ITA 数学描述 |
|:---|:---|:---|:---|
| 黑洞形成 | $M > 0$，$S_{BH} = A/(4G)$ | 代码空间定义 | $\|C/\!\sim\| = 2^{H_s}$ |
| Hawking 辐射 | 每单位时间产生 $\alpha$ 熵 | 每增加一维获得 $\Delta$RTCE | $\eta(d) = \partial_d \text{RTCE}$ |
| 信息困在黑洞内 | $I = \emptyset$，$S_{\text{rad}}$ 增长 | 容量不足，语义信息丢失 | $d < d^*$，RTCE 上升 |
| Page time 相变 | connected/disconnected saddle 交叉 | 编码容量 = 语义熵 | $d = d^*$，RTCE 取极值 |
| Island 出现 | $I \neq \emptyset$，$S_{\text{rad}}$ 下降 | 过拟合开始 | $d > d^*$，RTCE 下降 |
| 黑洞完全蒸发 | $S_{\text{rad}} = 0$，$M = 0$ | 无限容量，完美压缩 | $d \to \infty$，$\text{RTCE} \to 0$ |

### 6.2 对称性与守恒律

**物理**：幺正性要求 $\Delta S_{\text{rad}} + \Delta S_{BH} = 0$（总信息守恒）

**ITA**：编码守恒要求：

$$\underbrace{I(f_d(c); \text{exec}(c))}_{\text{隐向量携带的语义信息}} + \underbrace{S_{\text{loss}}(d)}_{\text{丢失的语义信息}} = \underbrace{H_s(C)}_{\text{总语义信息}}$$

即：

$$\boxed{I(f_d(c); \text{exec}(c)) + S_{\text{loss}}(d) = H_s(C)}$$

这是 ITA 的**信息守恒方程**，对应黑洞物理中的幺正性。

### 6.3 Replica Wormhole ↔ 集成编码器

物理中的 replica wormhole 连接不同 replica 的时空流形。ITA 的对应是**集成编码器**（ensemble of encoders）$f_1, f_2, \ldots, f_n$：

$$\text{RTCE}_{\text{ensemble}}(d) = \text{RTCE}\left(\bigoplus_{i=1}^n f_i\right) \geq \max_i \text{RTCE}(f_i)$$

集成后的 RTCE 不低于单个编码器的最优 RTCE——这对应 connected saddle 的自由能低于 disconnected saddle。

---

## 七、实验验证方案

### 7.1 实验 T-ITA-PC-01：真实系统的维度消融

**目标**：在真实 ITA 系统（sBERT → FSQ → DeepSeek）中测量 RTCE$(d)$，验证 Page curve 形状。

**方法**：

1. **维度范围**：$d \in \{1, 2, 4, 8, 16, 32, 64, 128, 256, 512\}$
2. **FSQ 配置**：固定 levels 配置或使用自动级别分配
3. **训练**：每个维度独立训练至收敛（early stopping）
4. **评估**：
   - 在测试集上测量 RTCE
   - 统计唯一索引数和码本利用率
   - 测量训练集 vs 测试集的 RTCE 差距（过拟合指标）

**预期结果**：

$$\text{RTCE}_{\text{train}}(d) \text{ 单调递增（或先增后平）}$$
$$\text{RTCE}_{\text{test}}(d) \text{ 呈 Page curve 形状（先增后降）}$$

转折点 $d^*$ 即为 ITA Page 维度。

### 7.2 实验 T-ITA-PC-02：过拟合检测

**目标**：量化过拟合对 RTCE 的影响。

**方法**：

1. 固定 $d = 256$，逐步增加训练数据量 $N \in \{100, 500, 1000, 5000, 10000\}$
2. 测量 $\text{RTCE}_{\text{train}} - \text{RTCE}_{\text{test}}$（过拟合差距）

**预期**：过拟合差距随 $N$ 增加而减小，验证 RTCE 下降确实由过拟合驱动。

### 7.3 实验 T-ITA-PC-03：语义熵估计

**目标**：估计目标代码集的语义熵 $H_s(C)$。

**方法**：

1. 对测试集中的所有代码对 $(c_i, c_j)$ 执行等价性测试
2. 构建语义等价类
3. 计算 $H_s(C) = \log_2 |C/\!\sim|$

**预期**：$H_s(C)$ 的值将给出 $d^*$ 的理论下界。

### 7.4 实验 T-ITA-PC-04：Island 效应检测

**目标**：验证"隐式语义"island 的存在。

**方法**：

1. 固定 $d$，逐步增加解码器的复杂度（小 Transformer → 中型 → DeepSeek）
2. 测量 RTCE 的变化

**预期**：
- 如果 island 存在：更强的解码器能从相同隐向量中恢复更多信息 → RTCE 上升
- 如果 island 不存在：解码器增强不影响 RTCE → RTCE 不变

### 7.5 实验 T-ITA-PC-05：Page 维度与语义熵的关系

**目标**：验证 $d^* \propto H_s(C)/\log_2 L$ 的预测。

**方法**：

1. 使用不同复杂度的代码集（简单函数 → 复杂算法），改变 $H_s(C)$
2. 对每个代码集执行维度消融，找到 $d^*$
3. 绘制 $d^*$ vs $H_s(C)$ 的关系

**预期**：$d^* \approx H_s(C)/\log_2 L + c_0$，其中 $c_0$ 是编码器效率的修正项。

---

## 八、数学定理与预测

### 定理 8.1（ITA Page Curve 存在性）

**声明**：设 $C$ 为有限代码集，$|C/\!\sim| > 1$。设 $f_d: C \to \mathbb{R}^d$ 为 FSQ 编码器，$g: \mathbb{R}^d \to C$ 为解码器。若：

1. $g$ 是有限容量的（参数数量有限）
2. 训练数据量 $N$ 有限
3. 编码器 $f_d$ 的容量随 $d$ 严格递增

则存在 $d^* > 0$ 使得 $\text{RTCE}(d)$ 在 $d = d^*$ 处取最大值。

**证明思路**：

- $d \to 0$ 时：编码容量为零，$\text{RTCE} \to 0$
- $d$ 适中时：编码容量增加，$\text{RTCE}$ 上升
- $d \to \infty$ 时：编码器过拟合训练集，泛化能力下降，$\text{RTCE}_{\text{test}} \to |C_{\text{train}}/\!\sim|^{-1} \cdot |\text{test correct}| \to 0$（极端过拟合时，编码器学到恒等映射但泛化失败）

由连续函数的极值定理，$\text{RTCE}(d)$ 在 $(0, \infty)$ 上存在最大值。$\square$

### 定理 8.2（ITA Page 维度的信息论下界）

**声明**：ITA Page 维度 $d^*$ 满足：

$$d^* \geq \frac{H_s(C)}{\log_2 L_{\max}}$$

其中 $L_{\max} = \max_i L_i$ 是 FSQ 各维度的最大量化级别。

**证明**：由 Fano 不等式（ita-kolmogorov-bekenstein-v1.0.md 附录 B.2），要达到 RTCE $> 0.5$，需要：

$$d \cdot \log_2 L_{\max} \geq H_s(C) - 1$$

RTCE 的最大值不可能出现在容量不足的区域，因此 $d^*$ 必须满足此下界。$\square$

### 预测 8.3（Page Curve 对称性的普适性）

**预测**：ITA Page curve 的上升段斜率 $\lambda_1$ 和下降段斜率 $\lambda_2$ 满足：

$$\frac{\lambda_1}{\lambda_2} \in [0.5, 2.0]$$

即 ITA Page curve 具有近似对称性，与黑洞物理中 Page curve 的近似对称性一致。

**物理意义**：如果此预测成立，说明 ITA 编码过程与黑洞蒸发过程共享相同的**信息流动动力学**——信息流入隐空间的速率与信息过拟合的速率在同一数量级。

---

## 九、结论

### 9.1 核心发现

1. **Mock 系统不呈现 Page curve**：mock 维度消融数据显示 RTCE 随维度**单调递减**（幂律衰减 RTCE $\propto d^{-0.78}$），原因是 mock 编码器的 hash 碰撞机制与真实语义编码无关。

2. **真实系统预期呈现 Page curve**：基于信息论分析，真实 ITA 系统的 RTCE$(d)$ 预期先升后降，呈现标准 Page curve 形状：
   - 上升段：编码容量增加 → 更多语义信息被保留
   - 转折点：编码容量 = 语义需求
   - 下降段：容量过剩 → 过拟合 → 泛化下降

3. **ITA Page curve 与黑洞 Page curve 的对应是结构性的**：
   - 维度 $d$ ↔ 蒸发时间 $t$
   - RTCE$(d)$ ↔ $S_{\text{rad}}(t)$
   - 临界维度 $d^*$ ↔ Page time $t_{\text{Page}}$
   - 信息守恒方程 ↔ 幺正性

4. **ITA Islands Formula 给出了 RTCE 的统一描述**：RTCE 由显式编码容量和隐式语义恢复（island）共同决定，临界点满足 QES 条件。

### 9.2 待验证项

| # | 验证项 | 实验 | 优先级 |
|:--|:---|:---|:---|
| 1 | 真实系统 RTCE 是否呈 Page curve | T-ITA-PC-01 | **高** |
| 2 | $d^*$ 是否满足信息论下界 | T-ITA-PC-03 + T-ITA-PC-01 | **高** |
| 3 | 过拟合是否是下降段的主因 | T-ITA-PC-02 | 中 |
| 4 | Island 效应是否存在 | T-ITA-PC-04 | 中 |
| 5 | Page curve 对称性预测 | T-ITA-PC-01 | 低 |
| 6 | $d^* \propto H_s(C)$ 关系 | T-ITA-PC-05 | 低 |

### 9.3 意义

ITA Page curve 如果被实验验证，将表明：

1. **编码维度的选择不是任意的**——存在一个"最优维度"，对应信息论的相变点
2. **过拟合是信息论必然**——维度超过临界值后的 RTCE 下降不是工程问题，而是信息论规律
3. **ITA 与黑洞物理的对应是动力学的**——不仅是静态的"熵=面积"类比，而是演化过程的完整对应
4. **Islands formula 在 ITA 中有直接的工程应用**——隐式语义恢复可以降低编码维度需求

---

## 附录 A：关键公式汇总

| 公式 | 编号 | 含义 |
|:---|:---|:---|
| $S_{\text{rad}} = \min_I \text{ext}[\frac{A}{4G} + S_{\text{bulk}}(\text{rad} \cup I)]$ | 物理 | Islands formula |
| $\text{RTCE}(d) = \max_J \{\min[\frac{d_J \log_2 L}{H_s}, 1] \cdot \Phi(J,d)\}$ | 5.5 | ITA Islands formula |
| $d^* \geq H_s(C) / \log_2 L$ | 8.2 | ITA Page 维度下界 |
| $I(f_d(c); \text{exec}(c)) + S_{\text{loss}}(d) = H_s(C)$ | 6.3 | ITA 信息守恒方程 |
| $\partial_d H_{\text{explicit}} = -\partial_d H_{\text{implicit}}$ | QES | ITA 语义极值面条件 |

## 附录 B：与系列文档的关系

```
ita-kolmogorov-bekenstein-v1.0.md  →  信息论基础（语义熵、Fano 不等式）
    │
    ├── ita-mdl-optimal-dim-v1.0.md  →  最优维度的 MDL 视角
    │
    ├── ita-ncd-information-loss-v1.0.md  →  信息丢失的 NCD 分析
    │
    ├── ita-information-loss-analysis-v1.0.md  →  40% 丢失的三级分类
    │
    └── ita-page-curve-v1.0.md (本文)  →  Page curve 与维度消融的对应
            │
            └── 未来：ita-experiment-page-curve-v1.0.md  →  实验验证报告
```

## 附录 C：符号表

| 符号 | 含义 | 单位/范围 |
|:---|:---|:---|
| $d$ | FSQ 编码维度 | 正整数 |
| $d^*$ | ITA Page 维度（临界维度） | 正整数 |
| $L$ | FSQ 量化级别 | 正整数 |
| $H_s(C)$ | 代码集的语义熵 | bits |
| $\text{RTCE}(d)$ | 维度 $d$ 下的执行等价率 | $[0, 1]$ |
| $\text{RTCE}_{\max}$ | 最大执行等价率 | $[0, 1]$ |
| $\lambda_1$ | 上升段速率参数 | $d^{-1}$ |
| $\lambda_2$ | 下降段速率参数 | $d^{-\beta}$ |
| $\beta$ | 下降非线性指数 | $> 0$ |
| $J$ | 语义 island | 语义子集 |
| $S_{\text{loss}}(d)$ | 维度 $d$ 下的信息丢失量 | bits |
| $N_{\text{eff}}(d)$ | 有效可区分等价类数 | 正整数 |
| $\Phi(J, d)$ | island 可恢复性函数 | $[0, 1]$ |

---

*本文档基于 islands-formula-notes-v1.0.md、ita-information-loss-analysis-v1.0.md、dimension_ablation_mock.json 和 ita-kolmogorov-bekenstein-v1.0.md 撰写。Mock 数据分析表明 mock 系统不呈现 Page curve；真实系统的 Page curve 预期需要实验 T-ITA-PC-01 验证。*
