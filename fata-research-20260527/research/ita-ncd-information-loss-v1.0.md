# ITA 编码的 NCD 信息损失度量

> **版本**：v1.0 · 2026-05-27
> **性质**：理论推导文档
> **前置**：T-IAT-01（Kolmogorov↔Bekenstein）、T-IAT-02（MDL 最优维度）

---

## 一、NCD 的定义

**归一化压缩距离（NCD）** 基于 Kolmogorov 复杂度定义：

$$\text{NCD}(x, y) = \frac{K(xy) - \min(K(x), K(y))}{\max(K(x), K(y))}$$

其中：
- $K(x)$ = x 的 Kolmogorov 复杂度
- $xy$ = x 和 y 的拼接
- NCD ∈ [0, 1]：0 表示完全相同，1 表示完全不同

**实际计算**：用压缩算法（如 gzip）近似 K：

$$\text{NCD}(x, y) \approx \frac{C(xy) - \min(C(x), C(y))}{\max(C(x), C(y))}$$

其中 C(x) 是 x 的压缩后长度。

---

## 二、ITA 中的 NCD 度量

### 2.1 代码-压缩 NCD

定义代码 c 与其压缩表示 f(c) 之间的 NCD：

$$\text{NCD}_{\text{compress}}(c) = \text{NCD}(c, f(c))$$

**物理含义**：代码被压缩后，丢失了多少信息？

- $\text{NCD}_{\text{compress}} \to 0$：信息完全保存（编码完美）
- $\text{NCD}_{\text{compress}} \to 1$：信息完全丢失（编码失败）

### 2.2 行为-恢复 NCD

定义原始代码 c 与恢复代码 g(f(c)) 之间的 NCD：

$$\text{NCD}_{\text{recover}}(c) = \text{NCD}(\text{exec}(c), \text{exec}(g(f(c))))$$

**物理含义**：解码后的行为与原始行为差多少？

- $\text{NCD}_{\text{recover}} \to 0$：行为完全恢复（信息守恒）
- $\text{NCD}_{\text{recover}} \to 1$：行为完全丢失（信息丢失）

### 2.3 Round-trip NCD

定义 round-trip 信息损失：

$$\text{NCD}_{\text{RT}}(c) = \frac{\text{NCD}_{\text{compress}}(c) + \text{NCD}_{\text{recover}}(c)}{2}$$

**物理含义**：编码→解码全过程中丢失的总信息量。

---

## 三、NCD 与黑洞信息守恒

### 3.1 黑洞的 NCD

在黑洞物理中，NCD 的对应：

$$\text{NCD}_{\text{BH}} = \text{NCD}(\text{物质态}, \text{辐射态})$$

- $\text{NCD}_{\text{BH}} \to 0$：信息守恒（物质态的信息完全在辐射态中）
- $\text{NCD}_{\text{BH}} \to 1$：信息丢失（违反量子力学）

### 3.2 对应表

| ITA | 黑洞物理 | 数学结构 |
|-----|---------|---------|
| $\text{NCD}_{\text{compress}}(c)$ | $\text{NCD}(\text{物质}, \text{黑洞})$ | 压缩过程的信息损失 |
| $\text{NCD}_{\text{recover}}(c)$ | $\text{NCD}(\text{黑洞}, \text{辐射})$ | 解压过程的信息损失 |
| $\text{NCD}_{\text{RT}}(c)$ | $\text{NCD}(\text{物质}, \text{辐射})$ | 全过程的信息损失 |
| $\text{NCD}_{\text{RT}} = 0$ | 信息守恒 | 幺正演化 |
| $\text{NCD}_{\text{RT}} = 1$ | 信息丢失 | 违反量子力学 |

---

## 四、NCD 的维度依赖性

### 4.1 NCD(d) 的理论预测

由 T-IAT-01 和 T-IAT-02 的结果：

$$\text{NCD}_{\text{RT}}(d) \approx 1 - \frac{I(d)}{H_s(C)}$$

其中 I(d) 是维度 d 下的互信息。

代入 I(d) 的饱和曲线：

$$\text{NCD}_{\text{RT}}(d) \approx 1 - \frac{I_{\max}}{H_s(C)} \cdot (1 - e^{-d/d_0})$$

### 4.2 临界维度

当 $\text{NCD}_{\text{RT}}(d^*) = \epsilon$（可容忍的信息损失阈值）时：

$$d^* = d_0 \cdot \ln\left(\frac{I_{\max}}{H_s(C) \cdot (1 - \epsilon)}\right)$$

**物理含义**：给定可容忍的信息损失 $\epsilon$，所需的最小编码维度 $d^*$。

### 4.3 与 Bekenstein 界的对应

在黑洞物理中：

$$\text{NCD}_{\text{BH}} = 0 \iff S_{\text{辐射}} = S_{\text{黑洞}} = A / (4 l_P^2)$$

即信息守恒要求辐射熵等于黑洞熵。

在 ITA 中：

$$\text{NCD}_{\text{RT}} = 0 \iff I(d) = H_s(C)$$

即信息守恒要求互信息等于语义熵。

**两者是同一个条件的两种表述。**

---

## 五、NCD 作为实验度量

### 5.1 实验方案

1. 固定编码器架构，改变维度 d = 1, 2, 4, 8, 16, 32, 64, 128, 256
2. 对每个维度：
   a. 编码代码 c → f(c)
   b. 解码 f(c) → g(f(c))
   c. 计算 NCD_compress(c) = NCD(c, f(c))
   d. 计算 NCD_recover(c) = NCD(exec(c), exec(g(f(c))))
   e. 计算 NCD_RT(c) = (NCD_compress + NCD_recover) / 2
3. 绘制 NCD_RT(d) 曲线

### 5.2 预期结果

```
NCD_RT
  │
1 ┤ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ 信息完全丢失
  │
  │     ╲
  │      ╲
  │       ╲
  │        ╲___________
  │                     ─ ─ ─ ─ ─   信息完全保存
0 ┤
  └──┬──┬──┬──┬──┬──┬──┬──┬──┬──→ d
     1  2  4  8 16 32 64 128 256
              ↑
              d* (临界维度)
```

### 5.3 与 Page curve 的对应

NCD_RT(d) 曲线的形状应该类似 Page curve 的"辐射熵"曲线：

- **早期（d 小）**：NCD_RT 高（信息大量丢失）
- **临界点（d = d*）**：NCD_RT 降至阈值（信息开始守恒）
- **后期（d 大）**：NCD_RT 趋近 0（信息完全守恒）

**这就是 ITA 版的 Page curve。**

---

## 六、NCD 的普适性

### 6.1 不依赖具体编码器

NCD 的优势是**普适性**——它不依赖具体的编码器架构，只依赖输入和输出的压缩复杂度。

这意味着：
- 不同编码器（sBERT、CodeT5+、StarCoder）可以用同一个 NCD 度量比较
- 不同任务（函数克隆检测、代码翻译）可以用同一个 NCD 度量比较
- 不同数据集（BigVul、BCB、OJClone）可以用同一个 NCD 度量比较

### 6.2 与黑洞物理的对应

在黑洞物理中，NCD 的普适性对应：

- 不同黑洞（质量、电荷、自旋不同）可以用同一个 NCD 度量信息守恒
- 不同蒸发阶段（早期、中期、晚期）可以用同一个 NCD 度量信息守恒
- 不同量子引力理论（弦论、圈量子引力）可以用同一个 NCD 度量信息守恒

**NCD 是信息守恒的普适度量。**

---

## 七、总结

| 发现 | 含义 |
|------|------|
| $\text{NCD}_{\text{RT}}(d) \approx 1 - I(d)/H_s(C)$ | NCD 与互信息的关系 |
| $\text{NCD}_{\text{RT}} = 0 \iff I(d) = H_s(C)$ | 信息守恒条件 |
| NCD_RT(d) 曲线 = Page curve | 代码系统复现黑洞物理 |
| NCD 是普适度量 | 不依赖具体编码器/任务/数据集 |

---

## 八、待验证

1. NCD 的实际计算：gzip 近似是否足够精确？
2. NCD_RT(d) 曲线是否真的呈现 Page curve 形状？
3. 不同编码器的 NCD 曲线是否具有普适性？

---

*ITA NCD 信息损失度量 v1.0 — 普适的信息守恒度量*
