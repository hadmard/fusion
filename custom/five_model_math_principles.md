# 五个模型的数学原理与推导说明

## 1. 文档目的

本文档的目标不是重复结果表，而是把当前对比中的五个模型放到同一套数学符号下，严谨说明：

- 每个模型的输入输出是什么
- 每个模型在 backbone 后到底做了什么变换
- 五个模型之间的结构差异可以写成什么公式
- 哪个模型比前一个模型多了什么自由度，少了什么约束

本文档覆盖的五个模型是：

1. `uv_single.pth`
2. `menkong.pth`
3. `kimi.pth`
4. `multi_feature.pth`
5. `high_resolution.pth`

其中：

- `uv_single` 是单模态基线
- `menkong` 是旧门控 cross-attention 融合
- `kimi` 是 `fusion_layers` 多层 AttnResidual 融合
- `multi_feature` 是 same-grid 多深度顺序读取
- `high_resolution` 与 `multi_feature` 的数学结构一致，但分辨率更高

---

## 2. 统一符号定义

### 2.1 输入与特征

记：

- 紫外图像为 `x^u`
- 白光图像为 `x^w`

backbone encoder 提取出的第 `l` 层特征记为：

- `U_l = E_l(x^u)`
- `W_l = E_l(x^w)`

其中 `l in {1, 2, 3, 4}` 对应当前工程里使用的 4 个 encoder feature levels。  
每个特征都可以写成二维网格张量：

`U_l in R^{B x C_l x H_l x W_l}`, `W_l in R^{B x C_l x H_l x W_l}`

为了书写 attention，更方便把它们拉平成 token 序列。定义：

`T(A) = Flatten(A) in R^{B x N_l x C_l}`, 其中 `N_l = H_l W_l`

若需要统一通道数到融合维度 `d`，记线性投影为：

`P_l : R^{C_l} -> R^d`

于是：

- `Q_l = P_l(T(U_l)) in R^{B x N_l x d}`
- `K_l = P_l(T(W_l)) in R^{B x N_l x d}`

---

### 2.2 通用 cross-attention 记号

对任意 query `Q`、key `K`、value `V`，标准多头交叉注意力记为：

`CrossAttn(Q, K, V) = MHA(Q, K, V)`

若展开到单头写法，可以表示为：

`Attn(Q, K, V) = Softmax(QK^T / sqrt(d_k)) V`

多头情形只是对多个头分别计算，再做拼接与线性映射：

`MHA(Q, K, V) = Concat(head_1, ..., head_h) W^O`

其中：

`head_i = Attn(QW_i^Q, KW_i^K, VW_i^V)`

---

### 2.3 通用检测头

五个模型的差异主要发生在“融合后特征如何构造”。  
一旦进入 RF-DETR 主干的 projector + transformer + detection head，它们共享同一类检测形式。

记融合后的多尺度特征为：

`F = {F_1, F_2, F_3, F_4}`

经过 projector 和 transformer decoder 后得到 decoder 隐状态：

`H = Decoder(F)`

分类头与框回归头分别为：

- `y_cls = W_cls H + b_cls`
- `y_box = sigma(W_box H + b_box + r)`

其中：

- `r` 表示 reference points / unsigmoid reference 的补偿项
- `sigma` 为 sigmoid

因此，五个模型的真正区别都可以归结为：  
它们如何从 `{U_l}` 与 `{W_l}` 构造出最终送入检测器的融合特征 `{F_l}`。

---

## 3. 模型一：uv_single

### 3.1 数学定义

`uv_single` 完全不使用白光分支，因此：

- `F_l = U_l`

或者写成：

`F = E(x^u)`

后续检测为：

`H = Decoder(Projector(F))`

`y_cls, y_box = Head(H)`

---

### 3.2 本质解释

`uv_single` 的数学本质最简单：

- 没有跨模态对齐问题
- 没有融合误差传播
- 没有额外融合参数

它提供的是一个最纯粹的单模态映射：

`x^u -> detection`

它的优点是优化稳定、计算代价低。  
它的缺点是任何白光补充信息都不可能被利用。

---

## 4. 模型二：menkong

### 4.1 基本结构

`menkong` 的每个 level 都有一个旧式门控融合块。  
对某个 level 的 UV/White token，记输入为：

- `Q_l = T(U_l)`
- `M_l = T(W_l)`

先做一次 UV <- White cross-attention：

`A_l = CrossAttn(LN_u(Q_l), LN_w(M_l), LN_w(M_l))`

然后用可学习标量门控 `alpha_attn` 控制残差注入：

`H_l^{(1)} = Q_l + alpha_attn * A_l`

接着再做一个 FFN 残差：

`G_l = FFN(LN_f(H_l^{(1)}))`

`H_l^{(2)} = H_l^{(1)} + alpha_ffn * G_l`

最后输出：

`F_l = H_l^{(2)}`

若有多层堆叠，旧实现是简单串行传播：

`H_l^{(t+1)} = Block_t(H_l^{(t)}, M_l)`

最终：

`F_l = H_l^{(T)}`

---

### 4.2 数学特点

`menkong` 的核心不是“做了 cross-attention”，而是：

它对 cross-attention 增益和 FFN 增益都引入了显式标量门控。

因此它学习的是：

- “白光信息是否该注入”
- “注入之后非线性重整形要开多大”

如果写成更抽象的形式：

`F_l = Q_l + g_attn * Phi_attn(Q_l, M_l) + g_ffn * Phi_ffn(Q_l, M_l)`

其中：

- `g_attn = alpha_attn`
- `g_ffn = alpha_ffn`

这意味着 `menkong` 的控制权在“层内残差幅度”上。

---

### 4.3 相比 uv_single 多了什么

`uv_single` 是：

`F_l = Q_l`

`menkong` 是：

`F_l = Q_l + controlled_residual(Q_l, M_l)`

因此从数学上说，`menkong` 比 `uv_single` 多了一个“受门控约束的跨模态残差校正项”。

---

## 5. 模型三：kimi

### 5.1 结构动机

`kimi` 对应的是早期 `fusion_layers` 路线。  
它不再使用 `menkong` 那种显式标量门控，而是引入“多层读取 + 深度方向的历史状态聚合”。

对于某个 level，记初始状态：

`H_l^{(0)} = Q_l`

每一层先从 White memory 读取：

`Z_l^{(t)} = CrossAttn(LN(H_l^{(t-1)}), LN(M_l), M_l)`

但关键点在于，下一层不是简单只吃上一层输出，而是先把历史状态集合：

`S_l^{(t)} = {H_l^{(0)}, Z_l^{(1)}, ..., Z_l^{(t)}}`

通过一个 depth-attention residual 聚合器压成新的状态。

---

### 5.2 深度聚合公式

定义历史状态堆叠为：

`V = [s_1, s_2, ..., s_t]`, `s_i in R^{B x N x d}`

再定义可学习伪查询向量：

`q_t in R^d`

代码中的聚合本质可以写成：

`k_i = RMSNorm(s_i)`

`a_i = <q_t, k_i>`

`omega_i = exp(a_i) / sum_j exp(a_j)`

于是聚合结果：

`H_l^{(t)} = sum_i omega_i s_i`

最终经过最后一次 FFN：

`F_l = H_l^{(T)} + FFN(LN(H_l^{(T)}))`

---

### 5.3 数学特点

`kimi` 的关键不是门控，而是把“控制权”从层内标量门控转移到了层间历史状态选择：

- `menkong` 决定的是“这一层的 residual 放大多少”
- `kimi` 决定的是“多次读取后，到底信哪几个历史状态”

因此它可被视为：

`F_l = Aggregate({Q_l, Read_1(Q_l, M_l), ..., Read_T(Q_l, M_l)})`

相比 `menkong`，这里的控制变量从：

- 标量 `alpha_attn, alpha_ffn`

转成了：

- 深度方向 soft selection 权重 `omega_i`

---

### 5.4 相比 menkong 多了什么

`menkong` 的控制是“单层内局部门控”。  
`kimi` 的控制是“多层读取后的全局历史聚合”。

如果抽象写：

- `menkong`: `F = Q + g * Phi(Q, M)`
- `kimi`: `F = Psi({Q, Phi_1(Q, M), ..., Phi_T(Q, M)})`

其中 `Psi` 是一个深度方向注意力聚合算子，而不是简单残差和。

---

## 6. 模型四：multi_feature

### 6.1 结构动机

`multi_feature` 不再假设某个 UV level 只读某个对应 White level。  
它的思想是：

- 对一个 UV level
- 顺序读取 4 个不同深度的 White memory
- 每次读取后都把结果放入历史状态集合
- 再用 AttnResidual 聚合

因此，它比 `kimi` 多的不是“更多层”，而是“更多 White memory 源”。

---

### 6.2 通道统一

不同深度特征的通道数可能不同，所以先统一到融合维度 `d`：

`Q_l^{(0)} = P_l(T(U_l))`

对白光各层：

`M_j = P_j(T(W_j)), j in {1,2,3,4}`

---

### 6.3 顺序读取公式

设对第 `l` 个 UV level，预定义读取顺序为：

`pi_l = (pi_l(1), pi_l(2), pi_l(3), pi_l(4))`

这是 same-level-first 顺序，例如某层优先读同层 White，再读其它深度。

定义历史状态：

`H_l^{(0)} = Q_l^{(0)}`

第 `t` 次读取前，如果 `t = 1`：

`R_l^{(1)} = H_l^{(0)}`

如果 `t > 1`：

`R_l^{(t)} = AttnResidual({H_l^{(0)}, Z_l^{(1)}, ..., Z_l^{(t-1)}})`

然后从指定 White depth 读取：

`Z_l^{(t)} = CrossAttn(R_l^{(t)}, M_{pi_l(t)}, M_{pi_l(t)})`

最终聚合：

`H_l^{(*)} = AttnResidual({H_l^{(0)}, Z_l^{(1)}, ..., Z_l^{(4)}})`

再加一个最终 FFN：

`\tilde{H}_l = H_l^{(*)} + FFN(LN(H_l^{(*)}))`

最后投回原通道维度：

`F_l = P_l^{-1}(\tilde{H}_l)`

---

### 6.4 数学特点

`multi_feature` 相比 `kimi`，最本质的变化是 memory 不再单一：

- `kimi` 的某个 level 只读本 level 对应 White memory
- `multi_feature` 的某个 level 依次读 4 个 White depths

因此其数学形式更接近：

`F_l = Psi({Q_l, Phi(Q_l, M_1), Phi(Q_l, M_2), Phi(Q_l, M_3), Phi(Q_l, M_4)})`

而不是：

`F_l = Psi({Q_l, Phi_1(Q_l, M_l), ..., Phi_T(Q_l, M_l)})`

也就是说，变化发生在：

- “读几次”之外
- 更重要的是“每次读谁”

---

## 7. 模型五：high_resolution

### 7.1 数学结构

`high_resolution` 与 `multi_feature` 的融合公式本质相同。  
差异主要不在融合算子，而在输入分辨率更高：

- `multi_feature`: `x^u, x^w in R^{560 x 560}`
- `high_resolution`: `x^u, x^w in R^{672 x 672}`

于是：

- token 数 `N_l = H_l W_l` 更大
- 小目标在 encoder feature 上的可分辨性更高
- cross-attention / 顺序读取看到的局部空间细节更多

因此从数学上讲，`high_resolution` 可以写成：

`F_l^{HR} = MultiFeatureFusion(E_l(x^u_{672}), E_1(x^w_{672}), ..., E_4(x^w_{672}))`

而 `multi_feature` 是：

`F_l^{MF} = MultiFeatureFusion(E_l(x^u_{560}), E_1(x^w_{560}), ..., E_4(x^w_{560}))`

两者的 fusion operator 相同，差别在输入采样分辨率和由此诱导的 token 网格密度不同。

---

### 7.2 数学影响

高分辨率带来的不是一个新算子，而是：

把同一个算子作用在更细的离散网格上。

如果把连续目标边界记为区域 `Omega`，离散采样后的误差可以抽象写成：

`epsilon(h) = |Omega - Omega_h|`

当网格步长 `h` 下降时，通常有更好的小目标近似。  
`high_resolution` 的优势可以理解为：

- 更小的 `h`
- 更大的 `N_l`
- 更高密度的空间离散

因此，`high_resolution` 不是“新融合方式”，而是“同一融合方式在更细空间离散下的实现”。

---

## 8. 五个模型的统一递进关系

把五个模型放在一条演化链上，可以写成：

### 8.1 uv_single

`F_l = U_l`

只有主模态，没有融合。

### 8.2 menkong

`F_l = U_l + gated_residual(U_l, W_l)`

引入单层受门控约束的 White 残差。

### 8.3 kimi

`F_l = DepthAggregate({U_l, Read_1(U_l, W_l), ..., Read_T(U_l, W_l)})`

把控制权从标量门控改为深度方向聚合。

### 8.4 multi_feature

`F_l = DepthAggregate({U_l, Read(U_l, W_1), ..., Read(U_l, W_4)})`

把“同一 White depth 多次读”推进到“多 White depths 顺序读”。

### 8.5 high_resolution

`F_l = MultiFeatureFusion on finer spatial grid`

保持与 `multi_feature` 相同的融合结构，但提升输入分辨率。

---

## 9. 从数学自由度角度看五个模型

还可以从“模型拥有哪类可学习自由度”来理解：

### 9.1 uv_single

自由度只来自检测主干本身。

### 9.2 menkong

新增自由度：

- `alpha_attn`
- `alpha_ffn`

属于显式幅度控制自由度。

### 9.3 kimi

新增自由度：

- 多层 cross-attention 参数
- depth attention 的伪查询向量

属于历史状态选择自由度。

### 9.4 multi_feature

新增自由度：

- 多 White depth 的顺序读取
- 各 depth 统一投影参数

属于 memory source 选择自由度。

### 9.5 high_resolution

算子自由度未增加太多，但离散表示能力增强。  
它增加的主要是：

- 更细粒度空间采样自由度

因此它更像“表示精度增强”，不是“融合规则增强”。

---

## 10. 最终总结

如果只用一句数学化的话概括五个模型：

- `uv_single`：`只看 U`
- `menkong`：`看 W，但通过标量门控决定注入多少`
- `kimi`：`反复看同层 W，并在深度方向决定保留哪些历史状态`
- `multi_feature`：`反复看不同深度 W，并在深度方向决定保留哪些历史状态`
- `high_resolution`：`用与 multi_feature 相同的融合规则，但在更细的空间网格上执行`

因此，这五个模型并不是五种完全无关的算法，而是围绕同一个问题逐步演化：

`如何让 UV 主分支从 White 分支中读到真正有用的信息，同时尽量不把噪声一起注入。`

如果从推导主线看，最本质的三次变化是：

1. `uv_single -> menkong`
   - 从“完全不融合”变成“受门控约束的单层融合”

2. `menkong -> kimi`
   - 从“层内门控”变成“层间历史聚合”

3. `kimi -> multi_feature/high_resolution`
   - 从“同一 White 源多次读取”变成“多 White 深度顺序读取”

这就是当前五个模型在数学原理上的主线关系。
