# Spherical-Harmonic Coefficients and Parameter Sampling

本文档整理项目中 spherical-harmonic coefficient、几何参数采样，以及
`isotropically normalized` 实现的代码核对结果。

主要代码位置：

- [`representation/SHPSG.py`](representation/SHPSG.py)
- [`representation/sampling.py`](representation/sampling.py)
- [`config.py`](config.py)

## 1. Spherical-harmonic coefficient 的定义

入口函数为：

```python
SHPSG(Ar, D2_8, D9_15)
```

最高阶数为 `N=15`，因此系数数组大小为：

\[
(N+1)^2=16^2=256.
\]

每一个系数有三个复数分量，分别对应最终表面坐标的 `x/y/z` 分量。

### 1.1 基础椭球

代码令初始主轴为：

\[
(a,b,c)=(Ar,1,1).
\]

随后使用统一缩放因子：

\[
s=Ar^{-1/3},
\]

得到：

\[
a'=Ar^{2/3},\qquad b'=c'=Ar^{-1/3}.
\]

因此：

\[
a'b'c'=1.
\]

这使不同 `Ar` 的基础椭球保持相同的轴乘积/参考体积，只改变主轴比例。`Ar=1`
时为球，`Ar>1` 时沿 `x` 方向拉长，`Ar<1` 时沿 `x` 方向压缩。

基础椭球的 (n=0,1) 系数来自固定的单位球模板 `fvec_sphere`，再分别用
`a_norm`、`b_norm`、`b_norm` 缩放 `x/y/z` 方向；这部分不是随机系数。

### 1.2 原始随机系数分布

对每个阶数 (n=1,\ldots,15)，代码生成：

```python
J = 1 - 2 * np.random.rand(n + 1, 3)
M = 1 - 2 * np.random.rand(n, 3)
```

所以基本随机变量满足：

\[
J_{ij},M_{ij}\sim U(-1,1).
\]

负 (m) 分量不是重新独立采样，而是利用球谐实值约束补齐：

\[
c_n^{-m}=(-1)^m c_n^{m*}.
\]

随后构造：

\[
P=L+iN.
\]

因此最终的原始系数并不是 256 个完全独立的复数均匀分布，而是“部分分量
独立采样 + 共轭对称补齐”。

### 1.3 按阶数归一化并施加谱带幅值

对每个阶数 (n) 和坐标方向 (j\in\{x,y,z\})，代码先计算：

\[
R_n^{(j)}=
\sqrt{\sum_m |P_{nm}^{(j)}|^2}.
\]

然后使用：

\[
f_{nm}^{(j)}=
\frac{P_{nm}^{(j)}}{R_n^{(j)}}I_n^{(j)}.
\]

这一步保留随机系数的方向/相位，但把每个阶数的能量强制设为目标值
(I_n^{(j)})。

低阶谱带幅值：

\[
D_2=
\frac{D2\_8}{\sum_{k=2}^{8}(2/k)^{1.387}}d_1,
\]

高阶谱带幅值：

\[
D_9=
\frac{D9\_{15}}{\sum_{k=9}^{15}(9/k)^{1.426}}d_1,
\]

其中 (d_1) 是基础椭球低阶系数的范数。

对 (n=2,\ldots,8)：

\[
I_n^{(x)}=I_n^{(y)}=I_n^{(z)}=
D_2\left(\frac{n-1}{2}\right)^{-1.387}\frac{1}{\sqrt3}.
\]

对 (n=9,\ldots,14)：

\[
I_n^{(x)}=I_n^{(y)}=I_n^{(z)}=
D_9\left(\frac{n-1}{9}\right)^{-1.426}\frac{1}{\sqrt3}.
\]

`d2` 控制低阶、大尺度形变；`d9` 控制高阶、小尺度纹理。虽然函数参数名
为 `D9_15`，当前代码实际只给 (n=9,\ldots,14) 赋值，第 15 阶保持为零。

## 2. `Ar`、`dL`、`dH` 等参数的 sampling rule

代码中没有直接命名为 `dL`、`dH` 的变量。按当前项目语义可对应为：

```text
dL ≈ d2  : low-order band, d2_8
dH ≈ d9  : high-order band, d9_15
```

当前 [`config.py`](config.py) 中的几何范围为：

\[
Ar\in[0.25,2.5],
\]

\[
dL=d2\in[0.1,0.4],
\]

\[
dH=d9\in[0,0.3].
\]

连续范围采样统一使用：

\[
q=q_{\min}+u(q_{\max}-q_{\min}),\qquad u\in[0,1].
\]

因此：

\[
Ar=0.25+2.25u_{Ar},
\]

\[
d2=0.1+0.3u_{d2},
\]

\[
d9=0.3u_{d9}.
\]

支持的模式如下：

| 模式             | 规则                                                                                     |
| -------------- | -------------------------------------------------------------------------------------- |
| `random`       | 每个参数独立采样 (u\sim U(0,1))，然后线性缩放                                                         |
| `lhs`          | 在单位超立方体使用 Latin Hypercube Sampling，每个维度的每个分层恰好取一个样本；默认 `maximin`                       |
| `low_re_dense` | 几何参数使用 LHS；流动参数使用 LHS，并将 (u_{Re}) 替换为 (u_{Re}^{\alpha})，默认 \(\alpha=2.5\)，以增加低 Re 区域密度 |
| `grid`         | 直接读取 `grid_values`，对各参数列表做笛卡尔积，不进行连续范围缩放                                               |

当前的training set都采用lhs的方法生成参数，testing set采用grid方式手工指定分布

## 3. `isotropically normalized` 的实现确认

代码中没有名为 `isotropically_normalized` 的独立函数。实际存在两种不同的
归一化，应分别理解。

### 3.1 基础椭球的统一等体积缩放

```python
scaler = (1.0 / Ar) ** (1 / 3)
a_norm = Ar * scaler
b_norm = 1.0 * scaler
```

其规则是：

\[
s=Ar^{-1/3},
\qquad
(a',b',c')=s(Ar,1,1).
\]

这是对三个空间方向使用相同标量的 isotropic/uniform scaling，目的却是保持
轴乘积为 1，即做等体积归一化。它不会把椭球变成球，也不会消除 `Ar` 的各向异性；
它只消除整体尺寸/体积变化。

### 3.2 扰动谱带的三方向均分归一化

对随机系数，代码把每个阶数的目标幅值在 `x/y/z` 三个方向设置成相同值，
并除以 \(\sqrt3\)：

\[
I_n^{(x)}=I_n^{(y)}=I_n^{(z)}=\frac{A_n}{\sqrt3}.
\]

因此三方向合成后的总幅值满足：

\[
\sqrt{(A_n/\sqrt3)^2+(A_n/\sqrt3)^2+(A_n/\sqrt3)^2}=A_n.
\]

再结合每个方向的系数范数归一化：

\[
\left\|\frac{P_n^{(j)}}{R_n^{(j)}}I_n^{(j)}\right\|_2
=I_n^{(j)},
\]

可以得到明确 rule：

1. 对每个阶数、每个坐标方向计算随机系数范数 (R_n^{(j)})；
2. 用 (R_n^{(j)}) 除掉原始随机系数的幅值；
3. 将该方向的目标幅值设为 (A_n/\sqrt3)；
4. 三个方向的总能量恢复为 (A_n)。

这才是代码中最接近“isotropic normalization”的部分：它保证扰动能量在
`x/y/z` 三个坐标分量上等额分配。

### 3.3 重要限制

当前实现并没有在最终生成的三维网格上再次计算体积，并将完整颗粒重新缩放到
单位体积。因此：

- 基础椭球进行了等体积归一化；
- 随机扰动系数进行了按阶数、按方向的范数归一化；
- 但叠加扰动后的最终颗粒不保证严格具有相同体积；
- `Ar` 造成的主轴各向异性仍然保留。

因此更准确的表述是：

> 当前代码实现了基础椭球的统一等体积缩放，以及扰动球谐系数在三个坐标方向上的等能量归一化；它没有对最终颗粒做严格的全局 isotropic volume normalization。
