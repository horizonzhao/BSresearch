# 离散 Delta 对冲误差的来源：从伊藤引理到模型误设与交易成本

## 1. 引言

Black-Scholes 理论告诉我们：在 GBM、连续交易、无摩擦的假设下，Delta 对冲可以完全复制期权。现实中这三条假设都不成立：只能离散地再平衡，波动率本身随机，每次交易都有成本。本文用一套从零实现、每一步都有独立对照的数值平台回答一个问题：**按 BS Delta 对冲时，误差有多大、从哪里来、怎样随再平衡频率、波动率模型和交易成本变化？**

主要发现可以概括为一个分解：


$$\text{对冲误差}=\underbrace{\text{离散化误差}}_{\propto N^{-1/2}}+\underbrace{\text{模型误设误差}}_{\text{与}~N~\text{基本无关}}+\underbrace{\text{交易成本}}_{\propto k\sqrt{N}}$$

第 3、4 节先验证所用的数值工具（蒙特卡洛与离散化格式）本身是可靠的，第 5、6 节依次测量这三项。

基准参数：$S_0=K=100,\ T=1,\ r=0.05,\ \sigma=0.2$。Heston 参数：$\kappa=1,\ \theta=0.04,\ \xi=0.8,\ \rho=-0.7,\ v_0=0.04$，Feller 比值 $2\kappa\theta/\xi^2=0.125 \lt 1$。全部实验使用固定随机种子（默认 42）。

---

## 2. 模型与方法

### 2.1 Black-Scholes 与 Greeks


$$d_1=\frac{\ln(S/K)+(r+\sigma^2/2)T}{\sigma\sqrt{T}},\quad d_2=d_1-\sigma\sqrt{T},\qquad C=S \Phi(d_1)-Ke^{-rT}\Phi(d_2)$$

$$\Delta=\Phi(d_1),\quad \Gamma=\frac{\varphi(d_1)}{S\sigma\sqrt{T}},\quad \text{Vega}=S\varphi(d_1)\sqrt{T}$$

$T\to0$ 时价格取内在价值；$\sigma\to0$ 时取远期确定性 payoff 的贴现值。

### 2.2 蒙特卡洛与方差缩减

风险中性下 $S_T=S_0\exp\big((r-\sigma^2/2)T+\sigma\sqrt{T} Z\big)$，估计量为 $e^{-rT}(S_T-K)^+$ 的样本均值，标准误 $\text{SE}=s/\sqrt{n}$。

- **对偶变量**：取 $Y=\tfrac{1}{2}\big[f(Z)+f(-Z)\big]$。看涨 payoff 关于 $Z$ 单调，$f(Z)$ 与 $f(-Z)$ 负相关，方差下降。
- **控制变量**：以 $X=S_T$（$\mathbb{E}[X]=S_0e^{rT}$ 已知）为控制，$Y_{cv}=Y-b^{\star}(X-\mathbb{E} X)$，$b^{\star}=\text{Cov}(Y,X)/\text{Var}(X)$。方差缩减比例为 $1/(1-\rho_{XY}^2)$。

### 2.3 三种 Greeks 估计量

记 $g(x;S_0)$ 为 $S_T$ 的对数正态密度。

- **有限差分**：$\big[V(S_0+h)-V(S_0-h)\big]/2h$，两次定价使用同一组 $Z$（公共随机数）。
- **路径微分**：交换求导与期望，$\partial_{S_0}(S_T-K)^+=\mathbf{1}_{\\{S_T>K\\}}~\partial_{S_0}S_T=\mathbf{1}_{\\{S_T>K\\}}S_T/S_0$。要求 payoff 几乎处处可微且 Lipschitz。
- **似然比**：把参数从 payoff 移到密度上，$\partial_{S_0}\mathbb{E}[f(S_T)]=\mathbb{E}\big[f(S_T)~\partial_{S_0}\ln g\big]$。对数正态下 $\partial_{S_0}\ln g=Z/(S_0\sigma\sqrt{T})$。不要求 payoff 光滑。

**数字期权为什么能区分三者**：payoff $\mathbf{1}_{\\{S_T>K\\}}$ 的导数几乎处处为 0，但在 $S_T=K$ 处有一个 Dirac 质量。路径微分只看到“几乎处处”的部分，所以期望恒为 0，是**有偏**的，而不只是方差大。有限差分只有落在 $(K\text{ 附近宽度}\propto h)$ 的路径才贡献 $\pm1/2h$，方差 $\propto 1/h$。似然比完全不对 payoff 求导，因此依然无偏。

### 2.4 离散化格式

对 $dS=\mu S~dt+\sigma S~dW$，伊藤-泰勒展开到 $\Delta W$ 的二阶：


$$S_{n+1}=S_n+\mu S_n\Delta t+\sigma S_n\Delta W_n+\sigma^2S_n\int_{t_n}^{t_{n+1}}\int_{t_n}^{s}dW_u~dW_s+\cdots,\qquad \iint dW~dW=\tfrac{1}{2}(\Delta W_n^2-\Delta t)$$

Euler 丢掉最后一项，Milstein 保留它。这一项均值为 0、标准差为 $O(\Delta t)$。在 $N=T/\Delta t$ 步上，各步的这一误差独立累加，总方差为 $N\cdot O(\Delta t^2)=O(\Delta t)$，所以 Euler 的强误差为 $O(\Delta t^{1/2})$；Milstein 消掉它后，剩下的局部误差是 $O(\Delta t^{3/2})$，强收敛阶升到 1。**弱收敛**只关心期望，而这一项均值为 0，在期望的主阶上不起作用，所以两种格式的弱收敛阶都是 1。

强误差必须在**同一条布朗路径**上比较精确解 $S_0\exp((\mu-\sigma^2/2)T+\sigma W_T)$ 与离散解；本实现对精确解和离散解使用同一组 $\Delta W$。

### 2.5 Heston 模型与傅里叶定价


$$dS=rS~dt+\sqrt{v}~S~dW^S,\qquad dv=\kappa(\theta-v)~dt+\xi\sqrt{v}~dW^v,\qquad d\langle W^S,W^v\rangle=\rho~dt$$

当 $2\kappa\theta \lt \xi^2$ 时 $v$ 可以触及 0，离散后会变负。实现了三种处理：

| 格式 | 漂移项 | 扩散项 |
|---|---|---|
| Euler（只在扩散项截断） | $v$ | $\sqrt{v^+}$ |
| Reflection | $v$，更新后取 $\lvert v\rvert$ | $\sqrt{\lvert v\rvert}$ |
| Full truncation | $v^+$ | $\sqrt{v^+}$ |

**参考值**用 Heston (1993) 的半解析公式，而不是另一个离散格式：


$$C=S_0P_1-Ke^{-rT}P_2,\qquad P_j=\tfrac{1}{2}+\tfrac{1}{\pi}\int_0^\infty\text{Re}\Big[\tfrac{e^{-iu\ln K}~\psi_j(u)}{iu}\Big]du$$

其中 $\psi_2(u)=\phi(u)$，$\psi_1(u)=\phi(u-i)/\phi(-i)$，$\phi$ 是 $\ln S_T$ 的特征函数。实现采用 Albrecher 等 (2007) 的 “little trap” 形式，即 $g=(b-d)/(b+d)$ 配合 $e^{-dT}$，避免复对数在长期限下跨越分支切割。

验证：$\xi=10^{-4}$、$v_0=\theta=0.04$ 时与 BS 价格相差 $2.6\times10^{-8}$；Fang-Oosterlee (2008) 标准测试例（$\kappa=1.5768,\theta=0.0398,\xi=0.5751,\rho=-0.5711,v_0=0.0175,r=0$）得到 5.785155，与文献值 5.785155450 一致；本文参数下傅里叶价格 9.2455，20 万条路径、1024 步的 full truncation MC 为 $9.2331\pm0.0202$，相差 0.6 个标准误。

### 2.6 离散 Delta 对冲

卖出一份看涨，收取 $V_0$。在 $t_i=i\Delta t\ (i=0,\dots,N-1)$ 用当前 $S_{t_i}$ 与剩余期限计算 BS Delta 并调仓，现金按 $e^{r\Delta t}$ 计息。到期 $\text{PnL}=\Delta_{N-1}S_T+B_T-(S_T-K)^+$。Delta 只依赖 $t_i$ 时刻的信息，不使用未来信息。比例交易成本为每次 $k~\lvert\Delta\text{shares}\rvert~S_{t_i}$。

---

## 3. 实验一：蒙特卡洛定价与 Greeks

**定价**（BS 真值 10.4506，$10^5$ 个 $Z$）：

| 估计器 | 价格 | SE | 方差缩减比例 |
|---|---|---|---|
| 基础 MC | 10.4205 | 0.0468 | 1 |
| 对偶 | 10.4763 | 0.0234 | 3.98 |
| 对偶 + 控制变量 | 10.4388 | 0.0062 | 56.4 |

方差缩减比例按相同个数的 $Z$ 计算；对偶法每个 $Z$ 计算两次 payoff，按 payoff 计算次数折算约为 2 倍。基础 MC 的 SE 对 $N\in[10^2,10^6]$ 的 log-log 斜率为 $-0.47$；1000 次独立重复中 95% 置信区间覆盖真值 951 次（二项标准误约 0.7%）。

**Greeks**（20 万条路径）：

| | 真值 | 有限差分 | 路径微分 | 似然比 |
|---|---|---|---|---|
| 看涨 Delta | 0.63683 | $0.63453\pm0.00129$ | $0.63453\pm0.00129$ | $0.63288\pm0.00326$ |
| 看涨 Vega | 37.524 | $37.305\pm0.169$ | $37.305\pm0.169$ | $37.018\pm0.620$ |
| 数字 Delta | 0.018762 | $0.0185\pm0.0021$（$h=0.01$） | 0（SE = 0） | $0.018765\pm0.000062$ |

三种方法在看涨上都在 2 个标准误内无偏；公共随机数下的有限差分与路径微分几乎逐位相同，因为看涨 payoff 是分段线性的，小 $h$ 的中心差分基本就是路径导数。似然比的 SE 大 2.5~3.7 倍，是它适用范围更广的代价。

数字期权完全符合 2.3 节的分析：路径微分**恒为 0**，有限差分 $h=0.1$ 时 SE 为 0.00065，$h=0.01$ 时为 0.0021（约 $\sqrt{10}$ 倍，对应方差 $\propto 1/h$），似然比的方差约为后者的 1/1100。

对欧式看涨做步长扫描（$h\in[10^{-4},10^{-1}]$）时，批次间标准差约 $4\times10^{-3}$ 且几乎不随 $h$ 变化，测得的偏差在 $10^{-5}\sim 10^{-3}$ 之间无规律波动、始终低于噪声水平。这说明**在公共随机数下，光滑 payoff 的有限差分不存在明显的偏差-方差权衡**，权衡只在不连续 payoff 上显著。

---

## 4. 实验二：离散化格式

**GBM 强/弱收敛**（$\Delta t=T/2^k,\ k=2,\dots,8$；强误差 2.5 万条路径，弱误差 6 万条路径）：

| | Euler | Milstein | 理论 |
|---|---|---|---|
| 强收敛阶 | 0.49 | 0.98 | 0.5 / 1.0 |
| 弱收敛阶 | 1.28 | 0.99 | 1.0 / 1.0 |

强收敛阶与理论高度一致，Milstein 的强误差在 $\Delta t=1/256$ 时为 0.0034，比 Euler（0.147）小 40 倍以上。弱误差方面，Euler 在 $\Delta t \le 1/32$ 时已降到 $10^{-3}\sim 10^{-4}$，接近 MC 分辨率，拟合斜率对样本量敏感（测试中 2.5 万条路径时为 0.70），只能说与理论值 1 “大致相符”。有意思的是，这组参数下 Euler 的弱误差绝对值反而比 Milstein 小，说明收敛**阶**相同时常数可以差别很大。

**Heston 负方差与定价偏差**（以傅里叶价 9.2455 为基准）：

| $\Delta t$ | 负方差步占比（Euler） | Reflection 偏差 | Full truncation 偏差 |
|---|---|---|---|
| 1/8 | 42.6% | +3.25 | +0.91 |
| 1/32 | 37.3% | +2.19 | +0.15 |
| 1/256 | 28.4% | +1.10 | $-0.04\pm0.10$ |

$\Delta t=1/32$ 时用 20 万条路径复核：reflection $11.433\pm0.029$（偏差 74 个标准误），full truncation $9.392\pm0.021$（7 个标准误）。负方差比例随 $\Delta t$ 下降得很慢，因为 Feller 比值只有 0.125，方差过程大量时间贴近 0。Reflection 把负的 $v$ 翻成正值，等于在零点附近不断注入额外方差，系统性地高估期权价格；full truncation 在漂移项中也使用 $v^+$，让均值回复把 $v$ 拉回正区间，偏差小得多且随 $\Delta t$ 快速下降。**如果没有傅里叶基准，而是用细网格 full truncation 的 MC 价格当“真值”，就无法发现 full truncation 自身在粗网格下的显著偏差。**

---

## 5. 实验三：GBM 下的离散 Delta 对冲

**理论**。对冲组合在 $[t_i,t_{i+1}]$ 上的损益，减去期权价值的变化，用伊藤引理展开后一阶项相互抵消，剩下二阶项。对卖出期权的一方：


$$\text{PnL}\approx-\sum_{i=0}^{N-1}e^{r(T-t_i)}\cdot\tfrac{1}{2}\Gamma_iS_i^2\sigma^2\big(\Delta W_i^2-\Delta t\big)$$

其中 $\Delta W_i\sim N(0,\Delta t)$。每一项均值为 0、方差为 $O(\Delta t^2)$，且彼此不相关，总方差为 $O(\Delta t)$，所以 $\text{std}\propto N^{-1/2}$。Kamal 与 Derman (1999) 对平值期权给出近似 $\text{std}\approx\sqrt{\pi/4}~\text{Vega}~\sigma/\sqrt{N}=6.65/\sqrt{N}$。

**结果**（每个 $N$ 2 万条路径，风险中性漂移）：

| $N$ | 4 | 16 | 64 | 256 | 512 |
|---|---|---|---|---|---|
| 均值 | $-0.040\pm0.023$ | $0.003\pm0.012$ | $0.009\pm0.006$ | $-0.006\pm0.003$ | $0.002\pm0.002$ |
| 标准差 | 3.224 | 1.682 | 0.867 | 0.433 | 0.308 |
| $\text{std}\cdot\sqrt{N}$ | 6.45 | 6.73 | 6.94 | 6.92 | 6.98 |
| 5% 分位数 | $-5.56$ | $-2.75$ | $-1.41$ | $-0.72$ | $-0.49$ |

- 标准差的 log-log 斜率为 $-0.484$；$\text{std}\cdot\sqrt{N}$ 稳定在 6.4~7.0，与 Kamal-Derman 的 6.65 一致。
- **直接验证误差来源**：在同一批路径上把上式逐步累加，与实际模拟的 P&L 相关系数为 0.976（$N=32$）和 0.987（$N=128$），两者标准差分别为 1.17 对 1.20、0.61 对 0.61。离散对冲误差几乎完全由伊藤二阶项解释。
- **Gamma 的作用**：平值附近 $\Gamma$ 最大，因此误差集中在标的价格靠近行权价的时段。按时间步平均时这一效应被路径分散稀释：最后 4 步的平均绝对贡献比最初 4 步大约 7%，路径平均的 $\Gamma$ 从 0.0188 升到 0.0201。
- **漂移测度**：用真实漂移 $\mu=0.12$（$N=64$）时均值为 $-0.004\pm0.006$，与风险中性下无可检测的差别。原因是 Delta 对冲在一阶上完全消去了对漂移的暴露，漂移只通过高阶的离散化效应影响 P&L，量级低于 MC 分辨率。

---

## 6. 实验四：模型误设与交易成本

### 6.1 模型误设

真实路径由 Heston（full truncation，$N$ 步）生成，对冲者用 BS Delta、波动率 $\sigma_h$，并按 $V_{BS}(\sigma_h)$ 收取权利金。

**理论**。按 $V_{BS}(\sigma_h)$ 收费并以 $\sigma_h$ 的 Delta 连续对冲时，卖方的 P&L 逐路径等于


$$\text{PnL}=\tfrac{1}{2}\int_0^T e^{r(T-t)}~{\Gamma^{(h)}_t} S_t^2~\big(\sigma_h^2-v_t\big)~dt$$

其中 $\Gamma^{(h)}$ 为以 $\sigma_h$ 计算的 BS Gamma。它说明误差由“对冲波动率与真实瞬时方差之差”以 Gamma 加权积累而成。均值还可以更直接地得到：风险中性下折现的自融资交易收益期望为 0，因此有**精确**的恒等式：


$$\mathbb{E}\big[\text{PnL}_T\big]=e^{rT}\big(V_{BS}(\sigma_h)-C_{\text{true}}\big)$$

其中 $C_{\text{true}}$ 应取生成路径所用离散格式下的价格。

**结果**（$N=64$，每组 2 万条路径）：

| $\rho$ | $\sigma_h$ | 均值（实测） | 均值（预测） | 标准差 | 偏度 | 超额峰度 | $q_{05}$ | $q_{01}$ |
|---|---|---|---|---|---|---|---|---|
| $-0.7$ | 0.200 | $1.218\pm0.028$ | 1.207 | 3.94 | $-2.20$ | 8.5 | $-6.13$ | $-13.70$ |
| $-0.7$ | 0.168（隐含） | $-0.053\pm0.029$ | $-0.060$ | 4.04 | $-2.46$ | 10.3 | $-7.71$ | $-15.98$ |
| $0$ | 0.200 | $1.293\pm0.038$ | 1.293 | 5.31 | $-2.79$ | 14.6 | $-8.32$ | $-19.25$ |
| $0$ | 0.165（隐含） | $-0.085\pm0.038$ | $-0.086$ | 5.42 | $-3.06$ | 16.7 | $-10.04$ | $-21.71$ |
| $+0.7$ | 0.200 | $2.140\pm0.032$ | 2.167 | 4.54 | $-2.02$ | 7.2 | $-6.27$ | $-14.27$ |
| $+0.7$ | 0.142（隐含） | $-0.136\pm0.032$ | $-0.107$ | 4.53 | $-2.51$ | 10.7 | $-8.69$ | $-17.82$ |
| GBM 基准 | 0.200 | $\approx0$ | 0 | 0.86 | $-0.23$ | — | $-1.43$ | — |

“预测”一列把 40 万条路径、64 步 full truncation 的 MC 价格作为 $C_{\text{true}}$ 代入上式。隐含波动率由傅里叶价格反推；$v_0=\theta$ 使 $\sqrt{v_0}$ 与 $\sqrt{\theta}$ 两种选择都等于 0.2，表中合并为一行。

**结论**：

1. **均值完全由定价决定**。六组实验中预测值与实测值都在约 1 个标准误内吻合。用 $\sigma_h=0.2$ 时卖方“多收”了权利金，因为高 vol-of-vol 下平值 Heston 价格低于 $\sqrt{\theta}$ 对应的 BS 价格（隐含波动率 0.14~0.17 小于 0.2），这正是 $\mathbb{E}[\sqrt{\bar{v}}] \lt \sqrt{\mathbb{E}[\bar{v}]}$ 的凹性效应。用隐含波动率对冲时均值约为 0，剩下的小幅负值正是 64 步 full truncation 路径相对傅里叶价格的离散偏差。
2. **误差不随 $N$ 消失**。隐含波动率、$\rho=-0.7$ 下，$N=16,64,256$ 时标准差为 4.53、4.04、3.94；GBM 世界对应为 1.68、0.87、0.43。误设误差由 $\int\Gamma S^2(\sigma_h^2-v_t)dt$ 驱动，它是 $O(1)$ 的，与再平衡频率无关。
3. **左偏厚尾**。所有 Heston 组的偏度都在 $-2.0\sim -3.1$，超额峰度 7~17。卖方持有空头 Gamma，在 $v_t$ 飙升时大幅亏损；$\xi=0.8$ 下 $v_t$ 的分布强烈右偏，映射为 P&L 的左偏。
4. **$\rho$ 的影响与直觉不同**。直觉上 $\rho$ 应主要改变偏度；实测中三个 $\rho$ 的偏度符号相同、差别不大，$\rho=0$ 时标准差与峰度反而最大。在本参数下，尾部主要由 vol-of-vol 决定，$\rho$ 更多通过改变 Heston 价格（从而改变公允的 $\sigma_h$）影响结果。
5. **公允定价不等于更安全**。改用隐含波动率后均值从约 $+1.2$ 降到约 0，但 $q_{01}$ 从 $-13.7$ 恶化到 $-16.0$：此前多收的权利金其实在为尾部风险提供缓冲。

### 6.2 交易成本与最优再平衡频率

**理论**。每次调仓的股数变化约为 $\Gamma S\sigma\sqrt{\Delta t}~\lvert Z\rvert$，累计成本的期望约为


$$k\sum_i S_i\cdot\Gamma_iS_i\sigma\sqrt{\Delta t}~\mathbb{E}\lvert Z\rvert~\propto~k\sqrt{N}$$

而离散化误差的标准差 $\propto 1/\sqrt{N}$。若以 $c_1k\sqrt{N}+c_2/\sqrt{N}$ 衡量左尾，其最小点 $N^*=c_2/(c_1k)\propto 1/k$：费率减半，最优频率翻倍。这与 Leland (1985) 的思路一致。

**结果**（GBM 路径，$\sigma=0.2$，每个点 1.2 万条路径；$k$ 为假设比例费率，表中 $k=0.001$ 即 0.1%、$k=0.0005$ 即 0.05%）：

| $N$ | 16 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| $-q_{05}$，$k=0$ | 2.73 | 1.41 | 1.02 | 0.72 | 0.50 |
| $-q_{05}$，$k=0.0005$ | 2.84 | 1.61 | 1.29 | 1.09 | 1.00 |
| $-q_{05}$，$k=0.001$ | 2.96 | 1.81 | 1.58 | **1.48** | 1.54 |
| 标准差，$k=0.001$ | 1.70 | 0.88 | 0.64 | 0.48 | 0.42 |

- $k=0.001$ 时左尾在 $N=256$ 最小，呈 U 形；$k=0.0005$ 时到 512 仍在下降，与 $N^*\propto 1/k$ 的预测一致（预期最优约在 512 附近或更高）。
- **成本项的标度可以直接核对**：成本带来的左尾增量（$k=0.001$ 减去 $k=0$）在 $N=256$ 和 512 时分别为 0.75 和 1.05，比值 1.40 $\approx\sqrt{2}$；$N=512$ 时 $k=0.0005$ 的增量为 0.50，恰为 $k=0.001$ 的一半。成本确实 $\propto k\sqrt{N}$。
- **标准差看不到 U 形**。所有 $k$ 下标准差都单调下降，因为成本在各路径之间变化不大，主要是让整个分布向左平移。衡量“总风险”时应使用包含均值的指标（分位数或期望损失），而不是标准差。
- **价内外敏感性**（$k=0.001$）：$K=90$ 与 $K=110$ 时最优频率也都在 $N=256$（左尾 1.23 与 1.55），虚值时整体风险更高。

---

## 7. 结论与局限性

**结论**。离散 Delta 对冲误差可以清楚地拆成三部分：

1. **离散化误差**来自伊藤二阶项，按 $N^{-1/2}$ 衰减，Kamal-Derman 近似给出了正确的常数，逐步分解与实际 P&L 相关性达 0.98 以上；
2. **模型误设误差**不随 $N$ 消失，其均值由一个精确恒等式 $e^{rT}(V_{BS}(\sigma_h)-C_{\text{true}})$ 决定，其左偏厚尾来自空头 Gamma 与随机波动率的相互作用；
3. **交易成本**按 $k\sqrt{N}$ 增长，与离散化误差权衡后得到 $N^*\propto 1/k$ 的最优频率。

数值层面，傅里叶半解析基准揭示了 Heston 离散格式在 Feller 条件不满足时的显著偏差，其中 reflection 不宜使用。

**局限性**：

- 所有参数都是假设值，没有校准到真实期权数据；利率为常数，无股息，无跳跃。
- 模型误设实验默认 $v_0=\theta$，未能区分“初始波动率”与“长期波动率”两种对冲选择。
- Heston 路径用 full truncation，粗网格下存在已量化的偏差；没有实现 QE 格式。
- 交易成本实验只在 GBM 世界中进行，没有与模型误设叠加；没有比较无交易带（Whalley-Wilmott）等更优的对冲策略。
- `plot_cost_tradeoff` 函数返回的“最优 N”使用了权重为 0.2 的启发式组合指标，本文的最优频率直接按 $-q_{05}$ 读取。
- 欧式看涨的 Euler 弱收敛阶测量受 MC 噪声影响较大。

---

## 8. 复现说明

- 环境：Python 3.13，numpy 2.5，scipy 1.18，matplotlib 3.11，pytest。依赖见 `requirements.txt`。
- 测试：`.\.venv\Scripts\pytest -q`，共 45 项，同时以较小样本重新生成 `figures/` 下全部图片。
- 本文数字：各绘图与统计函数的默认参数，全局种子 `DEFAULT_SEED = 42`（`src/utils.py`）；实验一由 `notebooks/01_pricing_and_greeks.ipynb` 复现，实验二由 `02_discretization_convergence.ipynb` 复现，实验三与实验四由 `03_hedging_experiments.ipynb` 复现。

**参考文献**

- Albrecher, H., Mayer, P., Schoutens, W., Tistaert, J. (2007). The little Heston trap. *Wilmott Magazine*.
- Andersen, L. (2008). Simple and efficient simulation of the Heston stochastic volatility model. *Journal of Computational Finance*.
- Fang, F., Oosterlee, C. W. (2008). A novel pricing method for European options based on Fourier-cosine series expansions. *SIAM Journal on Scientific Computing*.
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer.
- Heston, S. L. (1993). A closed-form solution for options with stochastic volatility. *Review of Financial Studies*.
- Kamal, M., Derman, E. (1999). When you cannot hedge continuously: the corrections to Black-Scholes. *Risk*.
- Leland, H. E. (1985). Option pricing and replication with transaction costs. *Journal of Finance*.
- Lord, R., Koekkoek, R., van Dijk, D. (2010). A comparison of biased simulation schemes for stochastic volatility models. *Quantitative Finance*.
