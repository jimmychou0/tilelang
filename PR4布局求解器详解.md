# PR-4 布局求解器详解（设计核心）

> 本文把 PR-4 最核心、最难的部分——**fragment 布局联合求解器**——单独讲透：它解决什么问题、每一步怎么判定、为什么这样设计、哪里出过什么 bug。自包含，不需要先读其他文档。代码位置以 `src/ascend/transform/pto_fragment_layout_solver.cc`（重构后未变动）为准，配合真实 dump 与报文。

---

## 0. 一分钟版

一块 `T.alloc_fragment((8, 128))` 的缓冲，向量指令一次吃 128 个数——**哪一维放进这 128 条 lane？** 这个选择不能每块 buffer 自己拍板（转置访问会把两块 buffer 的选择锁死在一起），必须全局联立求解。求解器做的事：

```
收集所有并行单元的访存 → 每个单元给"内维当 lane / 外维当 lane"两个方向投票
→ 投票全过的方法定（有的钉死方向、有的产出"两块 buffer 必须同向/反向"的锁链关系）
→ 带权并查集求解 → 解不唯一取占内存小的 → 注入最终布局
```

一句话：**这是一道带约束的全局选择题，投票筛掉不可行的选项，奇偶并查集解出可行的组合，字节最小破并列。**

---

## 1. 问题：哪一维放进 lane，为什么不能各自贪心

### 1.1 两种摆法，代价悬殊

```
选法 A：j 维（128）当 lane —— 8 个 entry（紧凑）   选法 B：i 维（8）当 lane —— 需补齐
        j→ 0……………128                                  j→ 0……………128
    i=0 ┌────────────┐  entry 0                 i=0 ┌──┐
    i=1 │  一整行     │  entry 1                 i=1 │  │  entry 0 由 i=0..7
    i=2 │            │                          i=2 │  │  的 128 个采样点
    ⋮   ⋮            ⋮                          ⋮   │  │  组成（含 padding）
    i=7 └────────────┘  entry 7                 i=7 └──┘
        一次 vload = 一整行                            一次 vload = 一列采样
```

- 选 A：`vload(handle, entry=i)`，8 个 entry，**省**；
- 选 B：entry = j，128 个 entry，还要把 i 维补齐到 128——最多 **32× 膨胀**（`[256,4]` 形状）。

直觉都想选 A。但布局是**逻辑对象到 (entry, lane) 的映射定义**，不是描述数据现在存在哪——选谁取决于**访问它的程序**。

### 1.2 为什么不能贪心：转置把选择锁死

```python
f = T.alloc_fragment((128, 8))     # f 的 dim1 只有 8
g = T.alloc_fragment((8, 128))     # g 的 dim0 只有 8
for i, j in T.Parallel(128, 8):    # 单元范围与两侧形状对齐：i<128、j<8
    g[j, i] = f[i, j] + 1.0        # 转置搬运，无其他访存
```

（单元范围必须是 `128×8` 而不是 `128×128`：j 要同时落在 f 的 dim1（长 8）和 g 的 dim0（长 8）里，取大就越界——§13 例 2 的实测程序就是这个形状。）

单元的 lane 维必须是某个循环变量。假设 j 当 lane：`f[i,j]` 要求 **f 的 lane 维 = dim1**（j 是 f 的第二维），同时 `g[j,i]` 要求 **g 的 lane 维 = dim0**（j 是 g 的第一维）。换 i 当 lane，两个都翻过来。也就是说：

```
方向 j ⟹ f=dim1, g=dim0          方向 i ⟹ f=dim0, g=dim1
        两种切法下 f、g 永远相反 —— 锁死
```

f 单独挑"最省"、g 单独挑"最省"，会挑出矛盾组合。所以这题的数学形态是：**一组 0/1 变量 + 相同/相反关系约束 + 字节最小目标**——正是带权并查集的形状。

---

## 2. 数据模型：两种 0/1 变量

求解器（[Run :514](src/ascend/transform/pto_fragment_layout_solver.cc#L514) 入口）先用自带的 [AccessScanner :85](src/ascend/transform/pto_fragment_layout_solver.cc#L85)（bind 环境敏感）把每个 SIMD_VF 区域拆成单元（[UnitInfo :116](src/ascend/transform/pto_fragment_layout_solver.cc#L116)：两个循环变量 + 访存列表），同时给每个单元收集**外层串行循环范围**（含 kernel 级、穿过 if——供证明用）。

待定变量就两种，全部取值 0/1（[VarKind :847 附近](src/ascend/transform/pto_fragment_layout_solver.cc#L847)只为 2D 单元/2D fragment 建变量——1D 没有选择，也就没有这道题）：

| 变量 | 一个一 | 0 | 1 |
|---|---|---|---|
| **单元方向变量 u** | 每个 2D 并行单元 | 内维 j 当 lane | 外维 i 当 lane |
| **fragment 变量 f** | 每块 2D fragment | dim0 当 lane 维 | dim1 当 lane 维 |

**变量不是什么高级东西，就是一道还没填答案的是非题。** 题 1 问"这个循环巢横着切还是竖着切"（128 条 lane 吃掉哪个循环变量）；题 2 问"这块 buffer 的哪一维摆进存储 lane"（决定 entry 数量与紧凑度）。求解 = 给所有是非题填答案。

### 2.1 为什么必须分两种：切法与摆法是两个侧面的属性

先立住一个容易混的点：**"按循环切完 lane，摆法不就定了么"——对一半**。单单元访问单 buffer 时确实如此；但切法和摆法活在两个不同的空间里，一比一的咬合只在最简单的场景成立，一般场景下它们是**多对多**的牵连。必须分成两种变量才能把这个牵连表达清楚：

```
切法（u）＝ 执行侧：哪 128 个【迭代】进一条向量指令     —— 每个【单元】的属性
摆法（f）＝ 数据侧：哪条逻辑线在内存里【连续】         —— 每块【buffer】的属性
咬合律：lane 循环变量所索引的那一维，必须是连续维
```

直观图（迭代空间 ≠ 数据空间）：

```
迭代空间（u 的领地）                    数据空间（f 的领地，backing 线性内存）
      j→ 0………128                     摆A(dim1连续)：第i行占连续128格
  i=0 ┌██████████┐ ←一条指令            0│f[0,0] f[0,1] … f[0,127]│128
  i=1 │          │ ←下一条              128│f[1,0] f[1,1] …        │…
  ⋮   └──────────┘   指令间串行        摆B(dim0连续)：第j列占连续128格（含补齐）
                                        0│f[0,0] f[1,0] … f[7,0] pad│128
同一个逻辑元素 f[3,5]：摆A住在地址 3×128+5=389；摆B住在 5×128+3=643。
```

**为什么单靠"选切法"不够——三种情形各缺一样东西：**

**情形 ①：一个切法拖多块 buffer（转置）。** 一条指令同时读写 `f` 的一行和 `g` 的一列——切法只有**一个**（单元的），摆法却有**两套**（每块 buffer 各一份）。切法变量表达"怎么干活"，摆法变量表达"每块货各自怎么码"——一个 u 配多个 f，不分成两种变量就没法表达这个一对多。

**情形 ②：一块 buffer 被多个单元拉扯（不需要第二块 buffer！）。** 每个单元有自己的切法（可能被各自的 UB 逼向不同方向），但这块 buffer 的物理内存只有一份、摆法只有一个：

```
单元1（被 a_ub 逼成 j=lane）──┐
                              ├→ f 只有一份摆法，两个要求矛盾 ⟹ 冲突
单元2（被 b_ub 逼成 i=lane）──┘
```

没有独立的 f 变量，"两个 u 对同一个 f 提出不同要求"这件事根本无法表达。

**情形 ③：没人访问也要选。** floating fragment（分配了没人碰）没有任何单元、任何切法，仍然需要一个摆法（字节最小直接选）。

### 2.1.1 传动杆：咬合律在下标里的体现

分了两种变量之后，**每条访存下标在 u 和 f 之间架起一根传动杆**——咬合律的具体形式。看 `acc[i, j]`：

```
若 u=0（j 当 lane）：lane 坐标 = j → j 落在 acc 的 dim1 上 → 逼着 f=1
若 u=1（i 当 lane）：lane 坐标 = i → i 落在 acc 的 dim0 上 → 逼着 f=0
```

完整例子（转置对）：

```python
f = T.alloc_fragment((128, 8))
g = T.alloc_fragment((8, 128))
for i, j in T.Parallel(128, 8):   # 一道 u 题
    g[j, i] = f[i, j] + 1.0         # 架起 u↔f、u↔g 两根杆
```

逐格填表看传动方向：

| u 的答案 | `f[i,j]` 要求 | `g[j,i]` 要求 |
|---|---|---|
| u=0（j当lane）| j 是 f 的 dim1 → **f=1** | j 是 g 的 dim0 → **g=0** |
| u=1（i当lane）| i 是 f 的 dim0 → **f=0** | i 是 g 的 dim1 → **g=1** |

规律：f 永远与 u 相反、g 永远与 u 相同。翻译成约束语言就是两条边（[Unite(a,b,relation)](src/ascend/transform/pto_fragment_layout_solver.cc#L406) 的 relation 即此）：

```
f XOR u = 1        （异或=1：必相反）
g XOR u = 0        （异或=0：必相同）
```

两式一拼，u 被消掉：**f XOR g = 1**——f、g 必一横一竖，而 u 自己自由（这正是 §7.2 奇偶边的来历）。

注意消元的方向说明**两种变量都不多余**：上式消掉 u 留下 f⊕g（情形①的视角——布局侧的锁链）；反过来单看一块 buffer，多个 u 的要求汇到同一个 f 上（情形②的视角——执行侧的拉扯）。系统里谁也不依附谁，两种变量各管一个侧面，靠传动杆锁在一起。

### 2.2 三种约束在这个模型里各是什么

| 约束 | 模型语言 | 例子 |
|---|---|---|
| 一元约束 | `某变量 = 常数`（钉死） | UB 梳子毙掉方向 0 → u=1；用户注解 → f=0 |
| 奇偶边 | `变量A XOR 变量B = 常数` | 上面的 f⊕u=1、g⊕u=0 |
| 无约束 | 所在分量没人钉 | 交给字节最小选 |

求解 = 沿边传播异或值；绕一圈算出"f XOR f = 1"（自己与自己相反）= 矛盾 = 程序无解，报冲突。

### 2.3 为什么这个编码划算

本问题的所有约束**恰好**都能写成"单变量定值"或"两变量异或为常数"——正是带权并查集的形状：近线性求解、矛盾检测（奇环）免费。若约束出现三变量关系或非异或关系（如 "f ≥ u"），这套编码就不够用，得换 CSP/SAT。本问题的约束恰好全部是两两相同/相反——这正是**二染色（二部图）判定**：任意拓扑的环都能装下（偶环可解、奇环矛盾），带权并查集对它**精确完备**，不是近似——所以循环依赖不需要在准入端挡（§13 例 3 的链正常解出传递，例 4 的奇环被求解器带证据拒绝）。准入真正收窄的是另一类东西：rank>2 fragment（3D 累加器需摊平）、同 data Var 视图与跨区域共享（别名/所有权问题，前扫描拒）、非规范 pinned 形态（PR-6 广播语义）。另有一族形态会被布局侧拒绝但**其源程序本身即非法**：同一单元内原地转置访问（`f[i,j] = f[j,i] + 1`——Parallel 点间读-写竞态，违反 m2 §11.2 输入契约）；求解器因两方向 dep-dim 互斥而拒绝（报错是布局文案而非竞态文案），属双重保险的措辞瑕疵，非支持缺口。合法的对称操作应 out-of-place（`g[j,i] = f[i,j]`，即例 2 形态）。

---

## 3. 前置工具：代换

### 3.1 原理：向量化到底对循环做了什么

一段 `for j in T.parallel(300)` 的循环，机器并没有真的跑 300 次——**向量硬件把 300 次迭代切成若干轮，每轮 128 条 lane 同时各干一次**：

```
VL = 128（T.SimdVF 声明的向量宽度，64/128/256 三档）

迭代 j:  0    1    2   …  127 │ 128  129 … 255 │ 256 … 299
        └────── 第 0 轮 ──────┘└── 第 1 轮 ───┘└ 第 2 轮┘
           lane: 0    1   … 127 │  0    1  … 127│  0  … 43
            q  = 0              │  q = 1        │  q = 2
```

于是**每个迭代编号 j 都唯一分解成两个数**：

```
j = q × VL + lane      q（chunk/轮次编号）= j ÷ VL 的商
                       lane（第几条 lane）= j mod VL
```

- `q` 回答"这是第几轮"——**轮与轮之间是串行的**（loop 一次次转）；
- `lane` 回答"这一轮里我是第几条向量通道"——**一轮之内 128 条 lane 同时发生**。

这就是 `ceil(300/128) = 3` 轮、末轮只有 44 条有效 lane（lane 44–127 是补齐的 padding，由 loop mask 关掉）的由来。

**代换做的事**：既然向量化的语义就是"j 被拆成 (q, lane)"，那就把程序里所有下标里的 j **提前改写成 q×VL+lane**——改写后的表达式里，"哪些部分随 lane 变、哪些不随 lane 变"一目了然，而方向合不合法恰好只取决于这一点。

### 3.2 两个具体例子

**例 1（UB 连续性，§4 的伏笔）**：`a_ub[i*128 + j]`，方向 j 当 lane，j 有 256 个（两层 IV 不再展开）：

```
代入 j = q×128 + lane：
    地址 = i*128 + (q×128 + lane) = (i+q)×128 + lane
                          ↑______________↑  随 lane 的部分恰好是 "+lane"
    → 地址 = 起点 + lane：128 条 lane 读一段连续地址 ✅ 一次 vload 搞定
```

**例 2（lane 匹配，§5.1 的伏笔）**：`f[i, j*2]`，方向 j 当 lane：

```
代入 j = q×128 + lane：
    下标 dim1 = 2×(q×128 + lane) = q×256 + 2×lane
    存储位置 = 下标 mod 128 = 2×lane mod 128     ← 是 lane 的 2 倍，不是 lane！
    → lane=64 的那条通道算出的存储位置是 128 mod 128 = 0 ≠ 64 ❌ 错位
```

### 3.3 为什么"方向"决定谁被代换

方向 j 当 lane → 代换 j；方向 i 当 lane → 代换 i。**代换谁，谁的时间结构（轮次×通道）就暴露出来**；没被选中的那个变量原样保留，在证明里叫"保留变量"（每轮固定一个值，类似串行下标）。同一个程序、两个方向 = 两次不同的代换 = 两套可判定性检查——这正是投票要对每个方向各做一遍的原因。

代换后所有下标都是 `(q, lane, 保留变量)` 的函数，"方向合不合法"从"猜"变成了**可判定的代数问题**（证明器能算恒等式和范围）。两道投票（§4、§5）都建立在这一步之上。

---

## 4. 投票 A：UB 访存的连续性

**问：一轮之内，128 条 lane 各自要读的地址，是不是 `起点 + lane`？**（向量 load 的物理形态就是 128 条 lane 拿一段相邻地址。）

实现在 [ProbeDirection :153](src/ascend/transform/pto_fragment_layout_solver.cc#L153)。拿 sv3 的 `a_ub[k*1024 + i*128 + j]` 代入：

```
方向 j（j = lane）：地址 = k*1024 + i*128 + lane
  lane: 0    1    2   …… 127
  地址: base base+1 base+2 …… base+127      ← 连续斜带 ✅ 一次 vload

方向 i（i = lane）：地址 = k*1024 + lane*128 + j
  lane: 0      1       2      …… 127
  地址: base base+128 base+256 …… base+16256 ← 步长 128 的梳子 ❌
```

梳子既不连续也不"全 lane 同址"（lane-uniform 是另一种合法形态，如 `x_ub[k]`），方向 i 出局。**sv3 的求解判词 `forward_thread = _j` 完全来自这一票**——不是偏好，是 UB 访存只有一种切法喂得饱向量访存。

---

## 5. 投票 B：fragment 访存的三件事证明

方向活过投票 A，还要问 fragment："按这个方向摆，你的每个访问表达得出来吗？"实现在 [FragmentDirectionFeasible :251](src/ascend/transform/pto_fragment_layout_solver.cc#L251)：在**候选规范布局**下代换，然后证三件事。它与 Verify 共用同一套证明思路（见 §10），域也相同。

### 5.0 entry 和 storage_lane 是怎么算出来的（三件事的公共前置）

5.1–5.3 证的两个对象——`storage_lane` 和 `entry`——不是凭空出现的，每个访存都走同一条三步流水线（[代码 :296-312](src/ascend/transform/pto_fragment_layout_solver.cc#L296)）：

**第 1 步：按候选方向造布局公式**（[BuildCanonicalLayout :450](src/ascend/transform/pto_fragment_layout_solver.cc#L483)）。布局是两条关于"下标占位符"的公式模板（ph0/ph1 代表访问下标的第 0/1 维）：

```
storage_lane = ph[lane维] mod VL
entry        = ph[lane维] ÷ VL  +  ph[其余维] × (lane维一侧的格子数 per_dim_entries)
              └── lane 维内的格子号 ──┘ └── 其余维换算成格子 ──────────┘
```

**第 2 步：代换**（§3 的工具）：把访问下标里的循环变量换成 `q×VL + lane`。

**第 3 步：套模板**：把代换后的下标填进占位符（coord_map + Substitute），化简得到这个访存在这个方向下的两条**具体表达式**——三件事证明对着的正是这两条式子。

**完整算例**（sv3 的 `acc[i, j]`，acc = (8,128)，VL=128，候选方向 j 当 lane、即 lane 维=dim1，per_dim_entries = ceil(128/128) = 1）：

```
模板:  storage_lane = ph1 mod 128
       entry        = ph1 ÷ 128 + ph0 × 1
代换:  j = 0×128 + lane（j 只有 128 个，q 恒 0）
填入:  ph0 = i, ph1 = lane
结果:  storage_lane = lane mod 128 = lane          ← 5.1 证的就是它 == lane
       entry        = lane ÷ 128 + i = 0 + i = i    ← 5.2/5.3 证的就是它
```

对照 §7 转换产物：Vectorize 发射的 `vstore(handle, i*128)` 里那个偏移 `i*128` 正是 `entry × VL`——**证明阶段算出的 entry 表达式，就是转换阶段直接拿来用的偏移公式**，这也是"证明在前、转换只填结果"的体现。

反例同管线走一遍（`f[i, j*2]`，f=(8,256)，方向 j 当 lane、lane 维=dim1，per_dim_entries=ceil(256/128)=2）：

```
代换:  j*2 = 2×(q×128 + lane)
填入:  storage_lane = (2q×128 + 2×lane) mod 128 = 2×lane mod 128   ← ≠ lane，5.1 立刻毙
       entry = (2q×128+2×lane) ÷ 128 + i×2 = 2q + i×2（若 lane 部分除不尽则更糟）
```


### 5.1 证明 (a)：存储位置 == 执行位置（lane 匹配）

布局的正向映射给出：存储 lane = `下标[lane维] mod VL`。要求它恒等于执行 lane。

**双向推演 `f[i, j*2]`（buffer `[8,256]`）——本求解器存在的头号理由：**

| | 方向 j（j=lane，f 的 lane 维=dim1） | 方向 i（i=lane，f 的 lane 维=dim0） |
|---|---|---|
| 存储 lane | `2j mod 128` | `i mod 128 = lane` ✅ |
| 与执行 lane 比较 | j=64 时 128 mod 128=0 ≠ 64 ❌ | ✅ |
| entry | —（已死） | `2j`，与 lane 无关 ✅，`< 256` ✅ |
| 结论 | **死** | **活** |

如果只按语法挑（"下标里有 j 就选 j"），会选中方向 j，到 Verify 才发现证不过——一个**换个切法就能跑的程序被误拒**。这就是可行性必须在**选择时**用证明筛的原因：fail-closed 不能变成 fail-wrong。

同类死法：`keep[i, j+1]`（j 方向存储 lane 平移一格）、`keep[i, 127-j]`（置换）。

### 5.2 证明 (b)：entry 与 lane 无关

对 §5.0 算出的 entry 表达式（如 sv3 的 `entry = i`）：entry 允许随**轮次**变（q、串行变量——每轮做不同 entry 天经地义），但一轮之内 128 条 lane 必须同属一个 entry：

```
entry = i     ┌──────────────────┐  128 lane 同住 entry 0 ✅
              └──────────────────┘
entry 依赖 lane ┌────┬────┬────┐   一轮内被劈到两个 entry，
              └────┴────┴────┘   一次 vload 表达不了 ❌
```

用户程序很难直接写出违例（多数在 (a) 先死），这条是挡奇异手写 TIR 的保险丝。

### 5.3 证明 (c)：entry 双侧区间（entry 落在 fragment 的格子范围内）

**先讲 count 是什么。** fragment 的物理空间按 entry 划分成格子（§1.1 的图），count 是这块 fragment **总共有的格子数**。公式里两个成分对准 §1.1 的例子（`f = (8, 128)`，即 dim0 长 8、dim1 长 128，VL=128）：

| | 选法 A（j 维当 lane，即 dim1 是 lane 维） | 选法 B（i 维当 lane，即 dim0 是 lane 维） |
|---|---|---|
| **lane 维**（被向量宽度吃掉的那一维） | dim1，长 128 | dim0，长 8 |
| **其余维**（没被吃掉的维，之积） | dim0，长 8 | dim1，长 128 |
| `ceil(lane维长 / VL)`（lane 维占几个格子宽） | ceil(128/128) = **1** | ceil(8/128) = **1**（8 个采样补齐到 128） |
| **count = 其余维积 × ceil(lane维/VL)** | 8 × 1 = **8 个格子** | 128 × 1 = **128 个格子** |

公式的直观读法：**一格 = 沿 lane 维连续 VL 个元素**；lane 维长不够 VL 整除时，尾块也占一整格（ceil 的来历）；其余维的每一种下标组合各需要自己的一套格子，所以乘其余维积。ragged 例：`f = (4, 130)`@VL=64、lane 维=dim1：count = 4 × ceil(130/64) = 4 × **3** = 12（第 3 格只用 8 条 lane 也占整格）。

代换后每个访存算出的 entry 表达式回答"这次访问落在第几格"。证明 (c) 要证的是：

```
0 ≤ entry < count
```

**为什么必须证这个：** entry 是数组下标——`vload/vstore(handle, entry×VL)` 会按 entry 索引 backing 存储。entry 越界就是数组越界，但它砸的不是自己：backing 排在统一 UB 地址空间里，**前面一格是别的 buffer，后面一格也可能是别的 buffer**——越界写 = 静默污染邻居的数据，不崩溃、无报错。所以这一证是纯安全检查，和"表达得出来"（证明 a/b）不同层。

**怎么证明：** entry 是关于 `(q, lane, 保留变量, 外层串行变量)` 的表达式。lane 和 q 的范围求解器自己知道（`lane ∈ [0,VL)`、`q ∈ [0,轮数)`）；**外层串行变量（如 `for k in serial(K)` 的 k）的范围必须由收集器补上**（`CollectEnclosingSerialRanges`：沿函数体收集全部非常量界非 parallel 循环的范围，含 kernel 级、穿过 if/while）。范围齐了之后交给算术证明器算两个不等式：

- 下界：`entry ≥ 0` —— 挡 `f[j - VL]` 这类负偏移（j 从 0 起时 entry = -1）；
- 上界：`entry < count` —— 挡访问越过最后一格。

**两个完整算例（正反各一）：**

```
正例：f = [4, 128]，方向 j=lane，访问 f[k, j]，k ∈ [0, 4)（范围已绑）
    entry 表达式 = k
    下界：k ≥ 0 ✅    上界：k < 4 = count ✅    → 放行

反例：同一块 f，但 k ∈ [0, 8)（k 跑 8 次、f 只有 4 行）
    entry = k，k=4..7 时 entry ≥ count
    上界证不过 → 拒绝，报文原样带出表达式：
    "the slot entry index (k) is not provably within [0, 4)"
```

```
反例 2（多 chunk 上界）：f = [256]，VL=128，访问 f[i + 128]，i = q×128+lane
    entry = (i+128) ÷ 128 = q + 1
    q 的范围 [0, 2) 绑上后：q=1 时 entry = 2 = count → 上界失败 ❌
    （q 范围不绑的话这个反例反而"证明通过"——漏判。所以
      chunk 维的范围也必须绑全，这正是 1D 多 chunk 场景的坑）
```

**证不出 = 拒绝**（fail-closed）：证明器算不出来不等同于不越界，一律当不安全处理。代价是把个别实际安全的程序也拒了——如实登记（例如 `for k in serial(8): if k<4: f[k,j]`、f 只有 4 行：if 条件能推出 k<4，但范围收集不带条件分析，两侧一致拒绝，属保守行为非判定分裂）。

实测报文（238）均含 entry 表达式原文，可直接对拍：

```text
keep[k, j]（f 4 行，k∈[0,8)）→ "the slot entry index (k) is not provably within [0, 4)"
f[i+128]（f [256]）          → "the slot entry index (solver_lane//128 + solver_q + 1)
                                is not provably within [0, 2)"
```

---

## 6. 选票表与 continue-not-break

每个 2D 单元有一张 **2 方向 × N 访存** 的选票表，任一格 ❌ 该列作废（mask 摘位）：

```
                方向0(j=lane)   方向1(i=lane)
a_ub[...]           ✅              ❌(梳子)
acc[i,j]            ✅              —(方向已死)
x_ub[k]             ✅(与lane无关)   —
                   ────────       ────────
     幸存:          ✓活             ✗死 → 单元被钉死方向0
```

两列全死 → "无可行方向"分类拒绝（真实报文见过：`keep[i,j+1]` 在 j 列死于 lane 匹配、i 列死于 entry 区间，各死一次后报组合冲突）。

### 6.1 选票之前的预备动作：dep-dim 怎么算

§7 产物里反复用到的 dep-dim（及 l_inner/l_outer——方向 0/1 下的 dep-dim），是在这里算的（[ConstrainUnit :902](src/ascend/transform/pto_fragment_layout_solver.cc#L902)）：对每个访存 × 每个方向，**逐维扫下标，看 lane 循环变量出现在哪一维**：

```cpp
const Var &sel_var = select_inner ? unit.jv : unit.iv;  // 该方向下谁当 lane
int dep_dim = -1, deps = 0;
for (size_t d = 0; d < access.indices.size(); ++d)      // 逐维扫
  if (下标[d] 用到 sel_var) { ++deps; dep_dim = d; }
// deps 必须恰好 == 1，否则该方向直接作废
```

`f[i, j]` 方向 0（j=lane）：dim0 的下标 `i` 不含 j；dim1 的下标 `j` 含 j → **dep_dim = 1**。`g[j, i]` 同方向：j 出现在 dim0 → **dep_dim = 0**——§2.1 填表那两列的数字就是这条扫描的产出。

注意两件事：①这是**纯语法找位置**（UsesVar 扫描），不是证明——`f[i, j*2]` 的 dep_dim 也是 1（j*2 含 j），合法性要到下一步 `FragmentDirectionFeasible` 才判；②`deps` 计数同时把关：=0（访问与 lane 无关）或 ≥2（lane 变量跨维混合，如 `f[i+j, 0]`）都不合咬合律，方向作废。

**关键细节：某格打 ❌ 的处理是 `continue` 不是 `break`**（[ConstrainUnit :877](src/ascend/transform/pto_fragment_layout_solver.cc#L877)）——即使方向 0 在第一个访存就死了，方向 1 对该访存的检查照做，把"方向 1 下这块 fragment 的 lane 维是 dim 几"记录下来。为什么较真：

> 漏记的后果是 fragment 从这个单元**得不到任何约束**、变成自由分量，字节最小替它挑一个"更省"的维——而这个维与单元实际方向矛盾，程序到下游被拒。`f[i,j*2]`（无 UB 版）是这个 bug 的标准触发器，现在是回归用例（断言 `forward_thread == "_i"`）。

---

## 7. 三种投票产物

投票结束，每个单元产出下列之一（[ConstrainUnit 建边段 :991 起](src/ascend/transform/pto_fragment_layout_solver.cc#L991)）：

```
单元投票后状态                     产物
─────────────────────────────    ─────────────────────────────
只剩一个方向活 .................. 一元约束（钉单元 + 钉 fragment）
两方向都活、钉出同一 lane 维 .... 一元约束（只钉 fragment，单元自由）
两方向都活、钉出不同 lane 维 .... 奇偶边（谁也不钉，记锁链）
该单元根本不碰 fragment ......... 无 fragment 产物（只有 UB 侧方向约束）
```

### 7.1 一元约束（直接钉死）

三种来源，各配算例：

**来源 ①：单元被投票钉死**（UB 梳子/三件事毙掉一个方向）——单元和它碰到的 fragment 一起钉：

```
sv3 的 k 循环单元：
  方向0(j=lane) 全票 ✅    方向1(i=lane) 死于 UB 梳子 ❌
  ⟹ AddUnary(u=0)；且方向0 下 acc 的 lane 维=dim1 ⟹ AddUnary(acc=1)
  （这也解释了 sv3 判词 forward_thread=_j）
```

**来源 ②：两方向殊途同归**（两个方向钉出**同一个** lane 维）——只钉 fragment，单元保持自由：

```
keep = [256]，单元访问 keep[i]（1D fragment 只有一维）：
  方向0 下 lane 维 = dim0；方向1 下 lane 维 = dim0（1D 没得选）
  ⟹ 只 AddUnary(keep=0)；u 不钉——单元切哪边这块 buffer 的摆法都一样
```

**来源 ③：用户 pinned**——`T.annotate_layout` 规范形态校验通过，直接作为一元约束进系统（校验细节见 §9）。

### 7.2 奇偶边（变量间的锁链）——含传递的完整机制

**触发条件**：单元两方向都活（UB 没约束它），且两个方向把同一块 2D fragment 钉到**不同** lane 维。这时谁也不能钉——钉哪个都等于替用户做了选择——只能记"这俩变量绑在一起变"。

**边权（relation）怎么来的**：代码里就是 `Unite(u, f, relation = l_inner)`——`l_inner` 是**方向 0 下这块 fragment 的 dep-dim**（§6 选票表 continue 保证它被记录）。含义：`f 的取值 = u 的取值 XOR l_inner`。用 §2.1 转置例填一遍（`g[j,i] = f[i,j] + 1`）：

| | l_inner（方向0 下的 dep-dim） | 边 |
|---|---|---|
| f 的访问 `f[i,j]` | j 是 f 的 dim1 → l_inner(f)=1 | `f = u XOR 1`（u=0→f=1，u=1→f=0 ✅ 对上 §2.1 的表） |
| g 的访问 `g[j,i]` | j 是 g 的 dim0 → l_inner(g)=0 | `g = u XOR 0`（g 与 u 同向） |

**消元看连锁**：两条边拼起来 u 被消掉——

```
        f=1               g=0
   f ─────── u(方向) ─────── g        （边上标的是 u=0 时的取值）

   u 随便选哪个，f⊕g 恒 = 1 ⟹ 两块 buffer 必须一横一竖，u 自由
```

#### 传递的代数：为什么链上能"接力"

**核心等式**：沿任意路径，两端的关系 = 各边权异或。推导只用了"每个变量出现两次就消掉"：

```
已知 f⊕g = r1，g⊕h = r2，求 f⊕h：
   f⊕h = (f⊕g) ⊕ (g⊕h)     ← 把两个已知等式两边分别异或
        = f ⊕ (g⊕g) ⊕ h     ← 结合律重排
        = f ⊕ 0 ⊕ h          ← g⊕g = 0（任何数异或自己是 0）
        = f ⊕ h
   所以 f⊕h = r1 ⊕ r2        ← 中间变量 g 被消掉，只剩边权
```

路径上不管串多少个变量，同样的手法都成立——**中间变量全部消掉，只剩边权异或**。这就是"传递"的全部数学。

**为什么"同向/反向"恰好配 XOR**：把同向记 0、反向记 1，关系合成表正好是异或：

```
f 与 g 同向(0)，g 与 h 同向(0)  ⟹ f 与 h 同向(0)      0⊕0=0
f 与 g 同向(0)，g 与 h 反向(1)  ⟹ f 与 h 反向(1)      0⊕1=1
f 与 g 反向(1)，g 与 h 反向(1)  ⟹ f 与 h 同向(0)      1⊕1=0  ← 负负得正！
```

最后一行是关键直觉：**转置两次 = 原样**（双重翻转回到起点）。所以"同向/反向"这个关系系统天然就是 mod-2 加法（Z₂ 群），XOR 是它唯一的运算——不是我们选了 XOR，是这个问题的代数结构本来就是 XOR。

#### 传递的物理：三块 buffer 的接力实例

```python
f = T.alloc_fragment((128, 8))
g = T.alloc_fragment((8, 128))
h = T.alloc_fragment((128, 8))

for i, j in T.Parallel(128, 8):     # 单元1
    g[j, i] = f[i, j] + 1.0           # 转置搬运 ⟹ f⊕g = 1
for i, j in T.Parallel(128, 8):     # 单元2
    h[i, j] = g[j, i] + 1.0           # 又一次转置 ⟹ g⊕h = 1
```

两条奇偶边进系统（各自的单元变量 u₁、u₂ 被消掉后）：

```
   f ────1──── g ────1──── h

传递：f⊕h = 1⊕1 = 0 ⟹ f、h 必同向
物理：g 是 f 的转置，h 是 g 的转置 —— 转置两次 = 原样，h 的摆法和 f 一样
```

用具体取值走一遍两种候选解（各 buffer 只有 dim0/dim1 两种摆法）：

```
解一：f=dim0(横) → g 必 dim1(竖) → h 必 dim0(横)     ✓ 与 f 同向，自洽
解二：f=dim1(竖) → g 必 dim0(横) → h 必 dim1(竖)     ✓ 同样自洽
两个解都合法 ⟹ 无锚自由分量 ⟹ 字节最小选（§7.3）
```

一条 pinned 能把整条链拖住：若用户注解 `h` 必须横（h=0），沿链反推 g=1、f=0——**一个变量定值，全链连锁确定**（§8 的 AddUnary 落在根上、成员靠 parity 换算，就是这条链的实现）。

#### 传递的另一面：矛盾（奇环）

往上面的程序再加一个单元，把 h 直接连回 f：

```python
for i, j in T.Parallel(128, 8):     # 单元3
    f[j, i] = h[i, j] + 1.0           # 第三次转置 ⟹ h⊕f = 1
```

现在图上多了一条 h→f 的边，权重 1：

```
   f ────1──── g ────1──── h
   ↑_________________________│  ← 第三条边，权重 1

沿环一圈：f⊕f = 1⊕1⊕1 = 1   ← 但 f⊕f 恒等于 0！
```

矛盾了：链上传递说 f⊕h=0（同向），直边说 f⊕h=1（反向）——**两个来源对同一对变量给出相反的结论**。物理上就是"转置三次还得等于原样"，不可能。数学上这叫**奇环**（odd cycle）：每块 buffer 自己都有合法摆法，但整个系统无解——等价于"给环上节点染两色、每条边两端必须异色"办不到。`Unite` 在合并时发现两端已同组且新边权与链上关系不符，返回 false，Reject 并带出每条边的来源单元。

#### 为什么"两方向都活"才出边

单元已被钉死的话 fragment 跟着定，发来源①的一元约束即可。奇偶边是"单元自己没想好、但不管它怎么想，别的变量之间已有确定关系"的记账方式——单元变量 u₁、u₂ 建边后即被消掉，留在系统里的只有 buffer 之间的关系。

### 7.3 自由分量与字节最小

**分量是什么**：并查集传播完，图分成若干连通块；被一元约束锚住的块**唯一定值**；**没有任何一元约束碰过的块 = 自由分量**——可能是一个孤立 fragment（如只写不读、或 floating 未访问），也可能是一串只靠奇偶边相连的 fragment。

**自由 ≠ 随便**：每个分量恰有两个候选解（整组翻转——选一个根值，链上其余变量由边权连锁确定）。比较两个解的**总字节数**：

```
单块字节数 = count × VL × 元素字节（32B 对齐）
count     = 其余维积 × ceil(lane维/VL)        （§5.3 的公式）

算例：f = [128, 4]@VL=128，f32（4 字节）
  根值 f=0（dim0 当 lane）：count = 4×ceil(128/128) = 4  → 4×128×4B  = 2 KiB
  根值 f=1（dim1 当 lane）：count = 128×ceil(4/128) = 128 → 128×128×4B = 64 KiB
  取 f=0。32× 差距。
```

**级联**：分量内一块定了，链上其余跟着翻——

```
分量 {f, g}（f⊕g=1）：根选 f=0 ⟹ g=1；根选 f=1 ⟹ g=0。
两个解的总字节 = bytes(f=0)+bytes(g=1)  vs  bytes(f=1)+bytes(g=0)，取小。
```

**字节最小的双重身份**：除了紧凑，它还是**容量侧防线**——`[256,4]` 落错维的 32× 膨胀布局从源头不再产生。否则膨胀布局导致预留失败，报错是"容量不足"、根因是"布局取向"，用户无从下手（报错面与根因错配）。

**仍并列怎么办**：两个解总字节相同（典型：两个同形状 buffer 的转置对，怎么翻都是一横一竖、总字节不变）→ **源码序第一个含自由方向的单元取内维（方向 0）**，整组随之确定：

```
f = [128,8]、g = [8,128]、g[j,i]=f[i,j]：
  解A: f=1(8格), g=0(8格) → 总 16 格
  解B: f=0(128格), g=1(128格) → 总 256 格   ← 取解A；不并列，字节最小直接定
并列时（两解同字节）：源码序规则上场，输出恒定
```

这条规则 + §8 的 DSU 值迁移，是"构建可复现"的两根桩——同样输入永远得到同样布局（有专门的确定性回归用例逐值断言 entry 数）。

---

## 8. Solve：带权并查集

[Solve :1080](src/ascend/transform/pto_fragment_layout_solver.cc#L1080)。§7 的产物汇到这里：一堆变量（节点）、一堆 XOR 边（关系）、少量钉死值（锚）。Solve 回答三个问题：**有没有矛盾？锚住的块定值是多少？没锚的块选哪个根？**

### 8.1 先懂并查集本身（30 秒）

并查集是管理"分组"的数据结构，只有两个动作：

- **Unite(a, b)**：把 a、b 所在的两组合并成一组；
- **Find(a)**：a 在哪组（返回组代表，即根）。

实现是森林：每组的根当代表，每个节点记一个父指针。它不存任何组员名单——"同组"就是"Find 结果相同"。**带权（奇偶）并查集**在每个节点的父指针上多存一个数：**本节点取值 XOR 父节点取值**（[parity_ 数组](src/ascend/transform/pto_fragment_layout_solver.cc#L435)）。于是沿父链一路异或到根，就得到"任意节点与根的取值关系"——[Find :380](src/ascend/transform/pto_fragment_layout_solver.cc#L380) 返回的第二个出参就是它。

**正确的建立时序：先有点，边来了才成为森林。** 节点不是随边创建的——收集阶段每登记一个 2D 单元 / 2D fragment，节点就进架子（[:852/:860](src/ascend/transform/pto_fragment_layout_solver.cc#L852)）；初始每个节点自成单节点树。**如果从头到尾没有奇偶边**（所有产物都是一元约束，如 sv3），并查集退化成一个**字典**：

```
sv3 的 DSU 全貌（全程没调过一次 Unite）：
   [u₁]    [u₂]    [acc]        ← 三个孤岛，无边
   root_value_: { u₁:0,  u₂:0,  acc:1 }
```

无边时所有操作照常、语义不变：`Find` 立即返回自己（奇偶 0，没有链可走）；`AddUnary` 就是往字典里直接放一条；矛盾检测照常（sv3 里单元 1、单元 2 都对 acc 发 `AddUnary(acc,1)`，同值通过——若一个 1 一个 0 则字典旧值 ≠ 新隐含值，冲突）。两种形态还能**共存**于同一个 DSU 实例（一个函数里 sv3 型孤岛单元和转置对连成的树同时在场）。所以实现上不需要"有没有边"两条代码路径——**无边 = 链长为 0 的特例**；反过来说，奇偶边才是"额外"的东西，它给字典加上"组内连锁"的能力，没有它 DSU 依然完整可用，只是表达不了"绑在一起变"。

### 8.2 三种产物怎么落进这棵树

| §7 的产物 | DSU 操作 | 效果 |
|---|---|---|
| 奇偶边 `f⊕u = r` | `Unite(u, f, r)`：合并两组，且旧根到新根的边权设成让 `f⊕u` 恰好 = r 的值 | 同组 + 关系可查 |
| 一元约束 `f = v` | [AddUnary :1050](src/ascend/transform/pto_fragment_layout_solver.cc#L1050)：`root_value_[Find(f) 的根] = v ⊕ (f 到根的奇偶)` | 整组被锚定 |
| （查询）`f 到底是几` | `value(f) = root_value_[根] ⊕ parity_to_root` | 链上 XOR 一路算回来 |

注意 AddUnary 的落点：**值记在根上，不是记在 f 上**——同一组共享一个锚值，各成员通过各自的 parity 换算出自己的值。这就是"钉死一块，全组连锁确定"的机制实现。

### 8.3 完整走一遍转置例（含并查集演化图）

变量 u（单元）、f、g（两块 buffer）；§7.2 已推出边 `f⊕u=1`、`g⊕u=0`。

先明确 **AddUnary 的含义：'把某变量钉成某值'的落地动作**，内部三步——①Find 找根和到根奇偶 p；②换算坐标系 `根值 = value ⊕ p`（因 var值 = 根值⊕p，反解）；③根上没值就写入，有值且不等就 Reject（报两条来源）。**值记在根上、不记在变量上**：一个分量只需一个锚，任何成员用 `根值 ⊕ 到根奇偶` 反查；合并时只需迁移根值（§8.5 的坑正源于此）。

逐步演化（箭头 = 父指针，边上数字 = parity_ 即'与父的 XOR'）：

```
初始——三个孤岛（各自成根，均无锚值）：
   [u]        [f]        [g]

Unite(u,f,1)：u 挂到 f，边权 1：
   u ──1──► f                            根 = f

Unite(u,g,0)：Find(u) 沿 u→f 得奇偶1、根f；合并 f 挂到 g，边权 = 0⊕1⊕0 = 1：
   u ──1──► f ──1──► g                   根 = g

   自查（沿父链异或到根）：
      u: 1⊕1=0 → u⊕g=0 ✓ 同向
      f:    1   → f⊕g=1 ✓ 相反（一横一竖）
      u⊕f = 0⊕1 = 1 ✓ 两条约束都满足

无 AddUnary → 整组无锚 → 自由分量 → 字节最小选根（§7.3）
若 AddUnary(f, 1)（用户 pin）：Find(f) 奇偶1、根g → root_value[g] = 1⊕1 = 0
   反查全组：g = 0；f = 0⊕1 = 1 ✓（pin 值）；u = 0⊕0 = 0
   —— 一个钉子，整组三块全部连锁定值

矛盾样貌：若再 AddUnary(g, 1) → 隐含根值 1⊕0 = 1 ≠ 已存 0 → Reject（两条来源）
```

### 8.4 矛盾怎么被发现（两条路）

**路 1：AddUnary 撞值。** 对同一组先后钉两个值，换算到根上不等——第二个 [AddUnary 的 else 分支](src/ascend/transform/pto_fragment_layout_solver.cc#L1064)直接 Reject，报文带两条约束的来源（"『来源A』与『来源B』在同一组上不一致"）。典型成因：UB 把 u 钉成 0、用户注解把同一组的 f 钉成"只在 u=1 时才成立"的值。

**路 2：Unite 出奇环。** 合并前两节点已在同组——`Unite` 不改树，只**校验**：把新边权和链上已算出的关系比对，不等即矛盾（[ra==rb 分支](src/ascend/transform/pto_fragment_layout_solver.cc#L413)返回 false）。图论上就是"绕一圈 XOR 和 = 1"的奇环。三个及以上 fragment 互相转置锁死时走这条。

两条路都归为**已证明的布局冲突**（程序无解，不是求解器不行），分类拒绝。

### 8.5 一个实现细节：根值迁移（确定性的关键）

[Unite :406](src/ascend/transform/pto_fragment_layout_solver.cc#L406) 合并两棵树时，若**旧根上记着锚值**而它被摘下根位，值必须换算后搬到新根（`迁移值 = 旧值 ⊕ 新边权`）；不搬的话那个锚就孤儿化——组员看起来毫无约束，会被当成自由分量重新选根，**同样的程序两次编译选出不同布局**（历史上真实发生过，P1-2）。现在的实现迁移值、并在"两根都带值且不等"时置 conflicting_ 标志由下一次 AddUnary/Solve 兜底拒绝。它与源码序规则（§7.3）一起构成构建可复现性，有逐值断言 entry 数的确定性回归用例钉住。

---

## 9. PreScan 与 pinned 校验

### 9.1 PreScan 是什么、为什么在求解**之前**

PreScan（[:624](src/ascend/transform/pto_fragment_layout_solver.cc#L624)）是求解前的**资格预审**：在开始建变量、投票之前，先把"求解器处理不了、或处理了也必然错"的程序形态挑出去，并把用户给的布局注解转成求解器的输入。它跑在 CollectRegions 之后、BuildConstraints 之前，理由很实际——

- 这些检查**与方向选择无关**（不管怎么摆布局都非法），放在投票里逐方向重复查是浪费，而且报错会淹没在"无可行方向"里看不见真因；
- pinned 注解必须在建约束**之前**变成一元约束（§8.2），否则求解器无从知道用户的意图。

一句话：**PreScan 管"这道题该不该进求解器、带什么前提进"，投票管"进了之后选哪个答案"。**

### 9.2 五项检查逐个讲

**① kernel 级注解 → 规范形态校验 → pinned。** 用户在 kernel 作用域 `T.annotate_layout({f: 布局})` 明说"f 就这么摆"。求解器不直接信任——先校验布局是**规范形态**，三条各挡一种错：

| 规范形态要求 | 挡什么 |
|---|---|
| 恰一个 entry 维（forward_index 恰一条） | 多维 entry 的布局没有对应的格子模型（§5.3 的 count 算不出来） |
| replicate == 1 | 复制型布局（每线程一份）是 PR-6 的广播语义，本求解器不建模型 |
| lane 是 floor-mod 形态（`forward_thread = 下标 mod VL`） | 任意映射（如 hash）没有"lane 维"可言，§2 变量模型失效 |

通过 → 作为一元约束进 §8（这就是 §7.1 来源③）；不通过 → 分类拒绝并提示"改注解或去掉"。

**② SimdVF 作用域内的注解 → 拒绝。** 写在 `with T.SimdVF:` 里的注解会被前端挂到 kernel 块上（引擎对区域级注解静默忽略），用户以为"只对这个区域生效"，实际语义不明。直接拒，报错说明注解位置错了。

**③ 区域外访问 → 拒绝。** fragment 的整套机制（布局、backing、转换）都以**转换区域**为单位；在区域外读 fragment（kernel 作用域、无 Parallel 的 SimdVF 里）意味着布局管不到它——值可能没初始化、backing 可能没声明。报错原文："every fragment access of a PTO kernel must live inside a converting T.Parallel body"。

**④ 跨区域使用 → 拒绝。** 同一块 fragment 被两个 SimdVF 区域访问：两个区域各自声明 handle、各自转换，但 backing 是一份——两边的写入顺序、entry 约定都无法跨区域保证。报错点名两个区域的序号。

**⑤ 同 data Var 多视图 → 拒绝。** 两块 fragment Buffer 共享同一底层分配（视图）：`g = f.view(...)` 类。两块 buffer 会被当成独立变量各选布局（§2），但对同一块内存——一个选横一个选竖就是静默数据破坏。当前按对象身份拒绝（结构完全相同的重复对象视为同一）；"结构归一"（形状/stride/offset 全同即同一）登记为后续增强。

（另有**无契约 TileOp 碰 fragment → 拒绝**：fill/copy 已被 §3 展开、reducer 走 PR-6 契约，剩下到这里的 TileOp 一律没有与 fragment 存储的配合协议。）

### 9.3 pinned 校验怎么比对（机制，非历史）

注解布局要和**该 buffer 的规范形态**比对——不是比"合法"，是比"和求解器自己会选的形状完全一致"。分两步（[:786 起](src/ascend/transform/pto_fragment_layout_solver.cc#L786)）：

```
第 1 步（前置）：InputShape 逐维 CanProveEqual 相等
第 2 步：调用 FragmentNode::IsEqual —— 比 thread_range / OutputShape /
         replicate / ThreadExtent / forward 映射内容（用公共变量替换后
         逐表达式比较，防 AST 形状不同但数学相同被误判）
```

为什么两步缺一不可：`IsEqual` 对 **shape 不等**的输入**故意返回 true**（SIMT 世界里允许广播复用不同形状的布局）——单用第 2 步，`[128,4]` 的注解能钉到 `[128,8]` 的 buffer 上，count 全错。反过来只做逐字段自写比较（历史实现）会漏比 forward 映射的**内容**——`forward_index = [i//2]` 这种**非单射**映射（两个逻辑元素映射进同一格 = 静默别名）长度对、形状对、就是映射本身非法。两步合起来：形状先卡死，再比全部字段内容。

---

## 10. 一个证明，两个消费者（最重要的不变量）

### 10.1 为什么会有两个消费者

同一个"访存合法性"问题，编译流程里有**两个时刻**要问它：

| 时刻 | 谁在问 | 问法 | 用答案干什么 |
|---|---|---|---|
| 布局求解时（§5） | 求解器投票 | "**如果**按这个候选方向摆，这个访存证得过吗？"（试探——布局还没定，两方向各问一遍） | 筛掉不可行方向（§6 选票表） |
| 转换前（Vectorize 之前） | Verify 安检门 | 求解器选出的**最终布局**下，每个访存证得过吗？（复核——此刻布局已注入、程序即将被改写） | 全过才放行到 Vectorize |

两个时刻问的是**同一道题**（同一个访存、同一条 entry/lane 公式、同三件事），只是前提（候选 vs 最终布局）不同。既然是同一道题，就该只有一个答案函数——于是两边调用**同一个** `AnalyzeFragmentAccess`（[utils :639](src/ascend/transform/parallel_to_pto_utils.cc#L639)；求解器侧经 `FragmentDirectionFeasible` 传候选布局，Verify 侧 [:1105](src/ascend/transform/verify_parallel_to_pto.cc#L1105) 传注入后的布局）：

```
                 AnalyzeFragmentAccess（唯一实现）
                   ↑ 传候选布局            ↑ 传最终布局
        求解器 FragmentDirectionFeasible    VerifyParallelToPTO
        "选方向时筛不可行项"                 "放行前复核"
```

### 10.2 "同一套范围集"为什么也是不变量的一半

证明 (c) 要绑外层串行变量的范围（§5.3）。两边绑的范围必须**逐个相同**——否则即使证明函数只有一个，输入不同照样分叉。所以范围收集也只有一份实现（`CollectEnclosingSerialRanges`，utils :93），两边都调它：求解器在 AddRegion 时给每个单元收集（[unit.outer_ranges](src/ascend/transform/pto_fragment_layout_solver.cc#L122)），Verify 的 visitor 自己遍历时同样收集（[:510 起](src/ascend/transform/verify_parallel_to_pto.cc#L510)——进一个串行循环压一层范围、出来弹掉）。分叉会是什么样，拿 `f[k, j]` 演示：

```
假设（反例）求解器绑了 k∈[0,4)、Verify 没绑：
  求解器：entry=k ∈ [0,4) ✅ 选定方向
  Verify：entry=k 无范围 → 证不出 < count ❌ 拒绝
  → 求解器放行、Verify 拒 = 同一程序时好时坏（误拒）
反方向的不对称更危险：求解器没绑（拒绝）、Verify 绑了（通过）
不会发生——求解器在流水线前面，它拒了就到不了 Verify；
真正要防的是上面那种"前面放行、后面拒绝"，以及更糟的
"两边证明函数不同导致某访问一边证得出一边证不出"
```

### 10.3 分叉的实际危害与"求解器先报"的好处

历史上两边各写过一套证明，出现过"求解器放行、Verify 拒"（用户程序时灵时不灵）——这是误拒；反过来"求解器拒、Verify 本该放"会错杀合法程序；最危险的是两边对**同一个访问**得出不同 entry 表达式却都各自通过——那是静默算错的温床。统一实现后这些类别整体消失。

附带一个工程好处：求解器在流水线上先执行，同类非法程序的报错点**固定在求解器**、文案稳定——测试可以直接断言报错原文（本求解器的负例测试全部如此），不会因为某次重构挪动了 pass 顺序就变成 Verify 的另一套文案。

### 10.4 Verify 不是求解器的复读机

Verify 复核之外还有自己的独立职责（求解器不做的）：跨 SIMD_VF 共享 fragment 的拒绝、lane 偏移/区域级提取的拒绝、UB 侧访存的连续性分类（`AnalyzeBufferAccess`，同一文件里证明的 UB 版）。所以它是一道完整的安检门，"共用证明"只是两者重叠的那部分——重叠部分共用，专属部分各管各的。

---

## 11. 全流程图

```
Run(:514)
 ├─ ① CollectRegions(:557 AddRegion)  拆单元、收集访存(含多下标,AccessScanner:85)、
 │                                    记录 fragment、收集外层范围(含kernel级、穿if)
 ├─ ② PreScan(:624)                   注解→pinned / 区域外访问拒 / 同data拒 /
 │                                    跨区域拒 / 无契约TileOp拒
 ├─ ③ BuildConstraints(ConstrainUnit:877)  逐单元投票：
 │      ┌─────────────────────────────────────────────┐
 │      │ 初始 mask=两方向都活（1D 单元只有一个）          │
 │      │ (a) UB 连续性(ProbeDirection:153)：梳子→摘位    │
 │      │ (b) 三件事(FragmentDirectionFeasible:251)：      │
 │      │     lane匹配/entry无关lane/双侧区间 → 摘位        │
 │      │     摘位用 continue（另一方向继续记录约束）        │
 │      │ 两方向全死→"无可行方向"拒                        │
 │      │ 幸存→ 只剩一个: AddUnary(:1050) 钉单元           │
 │      │        两方向钉出同一 lane维: AddUnary 钉 fragment │
 │      │        钉出不同 lane维: 奇偶边 Unite(:406)        │
 │      └─────────────────────────────────────────────┘
 ├─ ④ Solve(:1080)   XOR传播；奇环/双钉→冲突拒(带每边来源)；
 │                   自由分量→字节最小(联动容量:32×膨胀不产生)
 │                   并列→源码序首自由单元取内维（确定性）
 └─ ⑤ EmitLayouts    BuildCanonicalLayout(:450) 造规范布局
                     → 注入 layout_map（注入即最终答案，引擎 step0 跳过）
```

---

## 12. 真实材料（全部 238 实测，可复现）

**sv3 判词**（`_lower_capture(kernel, "AscendLayoutInference")` 后 `_region_layouts(mod)`）：

```text
fragment `acc`: forward_thread = _j  (lane dim), forward_index = (_i,)
```

**三个错误报文**：见 §5.3 与 §6（区间越界两例、无可行方向一例）。

**复现模板**：

```python
from test_pto_slot_plan_structure import _lower_capture, _region_layouts
mod, err = _lower_capture(kernel(), "AscendLayoutInference")
print(_region_layouts(mod))    # 每块 fragment 的 forward_thread / forward_index
# 负例: _lower_capture(kernel(), "PlanPtoSlotStorage", expect_error=True) 看 err
```

**相关回归用例**（`test_pto_slot_plan_structure.py`）：`f[i,j*2]` 求解器取解 + Vectorize 端到端两个用例（钉 continue 修复）；转置对 `g[j,i]=f[i,j]` 字节最小 + 源码序破并列逐值断言；pinned 翻转与形状拒绝；非单射注解拒绝；if 包裹串行 1D/2D（钉范围收集穿 if）；1D 多 chunk 区间拒绝；U3 双 root 拒绝。

---

## 13. 六个端到端例子：从源码到求解判词（每层判断全程可见）

以下六个程序全部在 238 实测（捕获点 `AscendLayoutInference`，例 5 为 `PlanPtoSlotStorage`），每个例子给：**源码 → 收集到的单元/访存 → 逐方向投票表 → 投票产物 → 求解过程 → 实测判词**。六个例子合起来覆盖三种产物的全部产生路径和求解的全部出口（钉死 / 奇偶边+字节最小 / 传递 / 奇环拒绝 / 并列破缺）。

公共记号：`f=k` 表示 fragment 变量 f 取值 k（k=0 → dim0 当 lane 维，k=1 → dim1）；`u=n` 表示单元方向变量取值 n（0=内维 j 当 lane，1=外维 i 当 lane）。

### 例 1（sv3）：UB 投票毙掉一个方向 → 一元约束钉死

```python
acc = T.alloc_fragment((8, 128), "float32")
for k in T.serial(4):
    for i, j in T.Parallel(8, 128):
        acc[i, j] = acc[i, j] + a_ub[k*1024 + i*128 + j]   # 单元1
for i, j in T.Parallel(8, 128):
    out_ub[i*128 + j] = acc[i, j]                           # 单元2
```

**收集**：单元1（i,j）∈(8,128)，访存 `acc[i,j]`（读写）、`a_ub[k*1024+i*128+j]`（读）；单元2 同形，访存 `acc[i,j]`、`out_ub[i*128+j]`。

**单元1 投票表**：

| | 方向 0（j=lane） | 方向 1（i=lane） |
|---|---|---|
| UB 连续性：a_ub 地址 | `k*1024 + i*128 + lane` = 起点+lane ✅ | `k*1024 + lane*128 + j` 步长 128 梳子 ❌ |
| acc 三件事 | lane=j ✓、entry=i 与 lane 无关 ✓、i<8 ✓ | 方向已死 |
| **结论** | **活** | **死** |

**产物**：只剩方向 0 → `AddUnary(u₁=0)`；方向 0 下 acc 的 dep-dim = dim1 → `AddUnary(acc=1)`。单元2 同样投票、同样结论（一致，不冲突）。

**求解**：acc=1 已锚定，无自由分量。

```
实测判词：fragment `acc`: forward_thread = _j   forward_index = (_i,)
```

### 例 2（不等形转置对）：奇偶边 + 字节最小

```python
f = T.alloc_fragment((128, 8), "float32")
g = T.alloc_fragment((8, 128), "float32")
for i in T.Parallel(128):              # 单元1（filler，不碰 f/g）
    b_ub[i] = a_ub[i]
for i, j in T.Parallel(128, 8):        # 单元2（转置）
    g[j, i] = f[i, j] + 1.0
```

**单元2 投票表**（无 UB 访存 → 投票 A 全票通过，两方向都活）：

| | 方向 0（j=lane） | 方向 1（i=lane） |
|---|---|---|
| f[i,j] 三件事 | lane=j ✓、entry=i ✓（<128）、i<128 ✓ | lane=i ✓、entry=j ✓（<8）、j<8 ✓ |
| g[j,i] 三件事 | lane=j ✓、entry=i ✓ | lane=i ✓、entry=j ✓ |
| **结论** | **活** | **活** |

**产物**：两方向都活且钉出**不同** dep-dim（f 在方向 0/1 下 dep-dim = 1/0，g = 0/1）→ 各发一条奇偶边：

```
f = u₂ XOR 1        （l_inner(f) = 1：j 是 f 的 dim1）
g = u₂ XOR 0        （l_inner(g) = 0：j 是 g 的 dim0）
```

**求解**：分量 {u₂, f, g} 无锚 → 自由 → 字节最小。两个候选解各算总 entry：

```
解A（u₂=0）：f=1 → count=128×ceil(8/128)=128；g=0 → 128×ceil(8/128)=128   总 256
解B（u₂=1）：f=0 →   8×ceil(128/128)=8；  g=1 →   8×ceil(128/128)=8      总 16  ← 取小
```

```
实测判词：f: forward_thread = _i（dim0）；g: forward_thread = _j（dim1）
        ⟺ f=0, g=1 —— 与解B逐位一致 ✓
```

### 例 3（三块接力）：奇偶边的传递

在例 2 基础上加一块 h 和第二个转置单元：

```python
h = T.alloc_fragment((128, 8), "float32")
for i, j in T.Parallel(128, 8):
    g[j, i] = f[i, j] + 1.0            # 单元2：f⊕g = 1（同上）
for i, j in T.Parallel(128, 8):
    h[i, j] = g[j, i] + 2.0            # 单元3：g⊕h = 1
```

**求解**：分量 {f, g, h, u₂, u₃}，链 `f──1──g──1──h`。传递（§7.2 的核心等式）：`f⊕h = 1⊕1 = 0`——f、h 必同向（转置两次=原样）。字节最小两个解：`f=0,g=1,h=0`（8+8+8=24 entry）vs `f=1,g=0,h=1`（128×3=384）→ 取前者。

```
实测判词：f: _i   g: _j   h: _i     ⟺ f=0, g=1, h=0
        f 与 h 同向（f⊕h=0）✓，g 与两者皆反 ✓
```

### 例 4（奇环）：传递的另一面——矛盾拒绝

在例 3 基础上再加第三个转置把环闭合：

```python
for i, j in T.Parallel(128, 8):
    f[j, i] = h[i, j] + 3.0            # 单元4：h⊕f = 1 ← 闭合
```

**求解**：链上传递说 f⊕h=0（例 3），新边说 f⊕h=1——同一对变量两个相反结论。绕环一圈：`1⊕1⊕1 = 1`，但任何变量与自身的 XOR 恒为 0——**奇环，系统无解**。物理：转置三次还要求等于原样，不可能。

```
实测：REJECTED —— "proven layout conflict: contradictory lane-direction
constraints reach the same unit/buffer group — 'the fragment access of
T.Parallel unit (i, j)' disagrees with 'the fragment access of T.Parallel
unit (i, j)' on the same group"（两条来源边各是哪个单元的哪个访问）
```

### 例 5（`f[i, j*2]`）：三件事证明毙掉一个方向 → 另一向钉死

```python
f = T.alloc_fragment((8, 256), "float32")
for i, j in T.Parallel(8, 128):        # 单元1：无 UB 访存
    f[i, j*2] = 1.0
for i, j in T.Parallel(8, 128):        # 单元2：RMW，同形
    f[i, j*2] = f[i, j*2] + 1.0
```

**单元1 投票表**：

| | 方向 0（j=lane） | 方向 1（i=lane） |
|---|---|---|
| 证明 (a) lane 匹配 | 存储位置 = 2j mod 128 ≠ j（j=64 时 =0≠64）❌ | 存储位置 = i mod 128 = lane ✓ |
| 证明 (b)/(c) | 方向已死 | entry = j*2 < 256 = count ✓ |
| **结论** | **死** | **活** |

**产物**：`AddUnary(u₁=1)`；方向 1 下 f 的 dep-dim = dim0 → `AddUnary(f=0)`。

```
实测判词：f: forward_thread = _i（dim0）
```

对照例 1：同样是"死一个方向活一个"，**死因不同**（例 1 死于 UB 梳子、例 5 死于 lane 匹配）——同一套产物路径，两条独立的死因通道。

### 例 6（同形转置对）：真并列 → 源码序破缺

```python
f = T.alloc_fragment((64, 64), "float32")    # lanes=64
g = T.alloc_fragment((64, 64), "float32")
for i in T.Parallel(64):                     # filler
    b_ub[i] = a_ub[i]
for i, j in T.Parallel(64, 64):
    g[j, i] = f[i, j] + 1.0
```

**求解**：与例 2 同构的奇偶边（f=u⊕1、g=u⊕0），但方形 → 两个解的总 entry 都是 64+64=128——**真并列**，字节最小分不出。破缺规则：**源码序第一个含自由方向的单元取内维**——filler 是 1D 无方向变量，轮到转置单元 → u₂=0（j=lane）→ f=1（dim1）、g=0（dim0）。

```
实测判词：f: forward_thread = _j   g: forward_thread = _i
        ⟺ f=1, g=0，与源码序规则推出的逐位一致 ✓
```

### 六例覆盖矩阵

| 例 | 投票A(UB) | 投票B(三件事) | 一元约束 | 奇偶边 | 传递 | 字节最小 | 并列破缺 | 拒绝 |
|---|---|---|---|---|---|---|---|---|
| 1 | ✓毙向 | ✓放行 | ✓钉死 | | | | | |
| 2 | 全过 | ✓放行 | | ✓ | | ✓ | | |
| 3 | 全过 | ✓放行 | | ✓ | ✓ | ✓ | | |
| 4 | 全过 | ✓放行 | | ✓ | ✓矛盾 | | | ✓奇环 |
| 5 | 无UB | ✓毙向 | ✓钉死(另一向) | | | | | |
| 6 | 全过 | ✓放行 | | ✓ | | 并列 | ✓ | |

---

## 14. 设计要点回顾

| 决策 | 理由 | 代码 |
|---|---|---|
| 全局联立而非逐 buffer 贪心 | 转置等访问锁死多块选择，贪心出矛盾组合 | Run :514 |
| 方向可行性=证明而非语法 | `f[i,j*2]` 语法两向都沾边，只有证明筛得掉不可行向，避免误拒 | FragmentDirectionFeasible :251 |
| 证明域绑全部外层范围 | `f[k,j]` 不绑 k 证不出；范围收集穿 if | utils :93 |
| 摘位 continue 不 break | 保住另一方向的约束记录，防字节最小选出矛盾方向 | ConstrainUnit :877 |
| 奇偶边只在"两方向都活"时产生 | 单元被钉时一元约束足够；奇偶边记"变量间确定关系" | Unite :406 |
| 字节最小破自由分量 | 紧凑 + 从源头消除 32× 膨胀（容量报错面不错配） | Solve :1080 |
| 源码序破并列 + DSU 值迁移 | 构建可复现（指针序曾导致两次编译不同结果） | :406 / Solve |
| pinned = InputShape 前置 + 完整 IsEqual | 非单射映射会静默别名；IsEqual 单用放过错形 | LoadPinnedAnnotations |
| 与 Verify 共用一个证明函数 | 判定永不分叉；求解器先报、文案可测 | utils :639 双消费方 |
| 注入即最终答案 | 跳过引擎 SIMT 推导，布局歧义为零 | EmitLayouts / :450 |

## 15. 快问快答

- **为什么不简单地"维度大的当 lane"？** 转置访问会锁死两块 buffer 的选择必须相反；且"哪维连续"取决于 UB 访存形态（梳子问题），不是形状大小。
- **1D fragment 需要求解吗？** 走同一条路，但没有方向选择（唯一维就是 lane 维），约束退化为区间证明。
- **从没被访问的 fragment（floating）怎么办？** 也进求解（B5：step 0 会让它和引擎冲突），作为单节点自由分量拿字节最小布局；rank>2 的死分配不阻塞整个函数。
- **用户能干预吗？** `T.annotate_layout` 提供规范形态即作为 pinned 参与；非规范/非单射分类拒绝。
- **解出来的布局谁消费？** Plan 算 count/bytes（容量）、Vectorize 决定 entry×lanes 偏移、Verify 复核——三者的共同输入。
