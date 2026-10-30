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
f = T.alloc_fragment((128, 8))
g = T.alloc_fragment((8, 128))
for i, j in T.Parallel(128, 128):
    g[j, i] = f[i, j] + 1.0        # 转置搬运，无其他访存
```

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

### 2.1 为什么必须分两种：访存下标里的"传动杆"

单元的切法（u）和 buffer 的摆法（f）是**两件事**，但每条访存下标都在它们之间架了一根传动杆。看 `acc[i, j]`：

```
若 u=0（j 当 lane）：lane 坐标 = j → j 落在 acc 的 dim1 上 → 逼着 f=1
若 u=1（i 当 lane）：lane 坐标 = i → i 落在 acc 的 dim0 上 → 逼着 f=0
```

完整例子（转置对）：

```python
f = T.alloc_fragment((128, 8))
g = T.alloc_fragment((8, 128))
for i, j in T.Parallel(128, 128):   # 一道 u 题
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

### 2.2 三种约束在这个模型里各是什么

| 约束 | 模型语言 | 例子 |
|---|---|---|
| 一元约束 | `某变量 = 常数`（钉死） | UB 梳子毙掉方向 0 → u=1；用户注解 → f=0 |
| 奇偶边 | `变量A XOR 变量B = 常数` | 上面的 f⊕u=1、g⊕u=0 |
| 无约束 | 所在分量没人钉 | 交给字节最小选 |

求解 = 沿边传播异或值；绕一圈算出"f XOR f = 1"（自己与自己相反）= 矛盾 = 程序无解，报冲突。

### 2.3 为什么这个编码划算

本问题的所有约束**恰好**都能写成"单变量定值"或"两变量异或为常数"——正是带权并查集的形状：近线性求解、矛盾检测（奇环）免费。若约束出现三变量关系或非异或关系（如 "f ≥ u"），这套编码就不够用，得换 CSP/SAT——所以准入刻意收窄，让每条约束都落进这两种形态（例如三块 fragment 的循环依赖在前扫描就被拒，不进求解器）。

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

entry 允许随**轮次**变（q、串行变量——每轮做不同 entry 天经地义），但一轮之内 128 条 lane 必须同属一个 entry：

```
entry = i     ┌──────────────────┐  128 lane 同住 entry 0 ✅
              └──────────────────┘
entry 依赖 lane ┌────┬────┬────┐   一轮内被劈到两个 entry，
              └────┴────┴────┘   一次 vload 表达不了 ❌
```

用户程序很难直接写出违例（多数在 (a) 先死），这条是挡奇异手写 TIR 的保险丝。

### 5.3 证明 (c)：entry 双侧区间

绑完全部范围后证 `0 ≤ entry < count`。**必须绑外层串行循环的范围**（`CollectEnclosingSerialRanges`，含 kernel 级、穿 if/while）——否则 `f[k,j]` 的 `entry = k < count` 永远证不出，合法程序被误拒。历史上两个真 bug 都在这：不绑 kernel 级循环（A4）、范围收集不穿 if（`for k: if…: f[k,j]`，P2-b）。

**为什么双侧**：只查上界，`f[j-VL]` 负偏移写到 backing **前面**的邻居 buffer；只查下界，`f[i+128]` 写到**后面**。都是静默内存破坏。

实测报文（238）：

```text
keep[k, j]，f 只有 4 行，k ∈ [0,8)：
  "fragment buffer `keep` access (k, j) is not provable under the
   vectorized coordinate `j`: the slot entry index (k) is not provably
   within [0, 4)"                        ← 报文连 entry 表达式原样带出

f[i+128]，f 是 [256]，i = q×128+lane：
  entry = q+1，q=1 时 = count=2           ← "the slot entry index
                                            (solver_lane // 128 + solver_q + 1)
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

**关键细节：某格打 ❌ 的处理是 `continue` 不是 `break`**（[ConstrainUnit :877](src/ascend/transform/pto_fragment_layout_solver.cc#L877)）——即使方向 0 在第一个访存就死了，方向 1 对该访存的检查照做，把"方向 1 下这块 fragment 的 lane 维是 dim 几"记录下来。为什么较真：

> 漏记的后果是 fragment 从这个单元**得不到任何约束**、变成自由分量，字节最小替它挑一个"更省"的维——而这个维与单元实际方向矛盾，程序到下游被拒。`f[i,j*2]`（无 UB 版）是这个 bug 的标准触发器，现在是回归用例（断言 `forward_thread == "_i"`）。

---

## 7. 三种投票产物

### 7.1 一元约束（直接钉死）

三种来源：只剩一个方向活（UB/证明毙了另一个）→ 钉单元；两方向殊途同归（1D fragment 只有一维）→ 只钉 fragment、单元保持自由；用户 `T.annotate_layout` 校验通过 → pinned。

### 7.2 奇偶边（变量间的锁链）

**触发条件精确**：单元两方向都活（UB 没约束它），且两个方向把同一块 fragment 钉到**不同** lane 维。这时谁也不能钉——钉哪个都等于替用户做了选择——只能记"这俩变量绑在一起变"。

§1.2 的转置例：`f⊕u = 1`、`g⊕u = 0`（XOR 关系，`u` 被消掉后 **f⊕g = 1，f、g 必反向**，而 u 保持自由——单元切哪边都行，两块 buffer 必须一横一竖）。

**传递**是它的威力（三单元链：`f⊕g=1`、`g⊕h=1` ⟹ `f⊕h=0` 必同向）；**奇环**是它的矛盾检测（绕一圈 XOR 累加 = 1 ⟹ "f 与自己反向"⟹ 已证明布局冲突，报错带每条边的来源单元）。

**为什么"两方向都活"才出边**：单元被钉死的话 fragment 直接跟着定，发一元约束即可，不需要绕。奇偶边是"单元自己没想好、但不管它怎么想别的变量已有确定关系"的记账方式。

### 7.3 自由分量与字节最小

没有被任何一元约束锚定的连通分量（含 floating 未访问 fragment）：每个分量只有"整组翻转"两个候选解，**按总字节数取小**：

```
count = 其余维积 × ceil(lane维/VL)
[128,4]@VL=128：dim0 当 lane → 4 entry；dim1 → 128 entry。取 dim0。
```

字节最小同时是**容量侧防线**：`[256,4]` 落错维的 32× 膨胀布局从源头不再产生——否则预留失败的报错（"容量不足"）与根因（布局取向）错配，用户无从下手。

**仍并列**（如两个同形状 buffer 的转置对）→ **源码序第一个含自由方向的单元取内维**，整组随之确定。这条规则保证同样输入永远得到同样输出——它和 DSU 的值迁移修复（见 §8）一起，是"构建可复现"的两根桩。

---

## 8. Solve：带权并查集

[Solve :1080](src/ascend/transform/pto_fragment_layout_solver.cc#L1080)：沿奇偶边做 XOR 传播 + 一元钉死；矛盾（节点被钉成两个值 / 奇环）→ 分类拒绝并附每边来源；自由分量按 §7.3 收尾。

实现是经典带权并查集（[ParityDSU::Unite :406](src/ascend/transform/pto_fragment_layout_solver.cc#L406)），但有一个非显然的坑（修过的 P1-2，值得知道）：**合并两棵树时，旧根上钉的值必须迁移到新根**，否则那个值孤儿化——组件看起来无约束，构建结果取决于指针序（同样的程序两次编译选不同布局）。现在 `Unite` 迁移值并记录冲突来源，配合源码序规则，确定性有专门的回归用例钉住。

---

## 9. PreScan 与 pinned 校验

求解前的前扫描（[PreScan :624](src/ascend/transform/pto_fragment_layout_solver.cc#L624)），全部为分类拒绝或钉死：

| 检查 | 处理 |
|---|---|
| kernel 级 `T.annotate_layout` 规范形态（恰一 entry 维、replicate==1、lane 为 floor-mod） | 校验通过 → pinned 一元约束 |
| SimdVF 作用域内注解 | 会挂 kernel 块按 kernel 级 pinned 接受（DSL 构造不出专属形态，登记差异） |
| fragment 在转换区域之外被访问 / 被两个区域使用 / 多 fragment 共享 data Var | 分类拒绝 |
| 无 PTO 契约的 TileOp 碰 fragment（含 fragment 上的 direct reduce） | 拒绝 |

pinned 的字段校验有一段历史教训：曾用自写比较只查 InputShape/forward_thread/forward_index **长度**，`forward_index=[i//2]` 这类**非单射映射**能混过、静默元素别名。现在 InputShape 逐维相等**前置** + 调用完整 `FragmentNode::IsEqual`（比 thread_range/OutputShape/replicate/forward 映射**内容**）——前置的原因：IsEqual 对 shape 不等的输入故意返回 true（SIMT 广播复用），单用会放过错形注解。

---

## 10. 一个证明，两个消费者（最重要的不变量）

求解器投票（候选布局下试探"这个方向证得过吗"）与 Verify 放行（最终布局下复核"选出来的方向真的证得过吗"）调用**同一个** `AnalyzeFragmentAccess`（[parallel_to_pto_utils.cc:639](src/ascend/transform/parallel_to_pto_utils.cc#L639)）、绑**同一套**范围集（`CollectEnclosingSerialRanges` :93 共享）：

```
                 AnalyzeFragmentAccess（唯一实现）
                   ↑                    ↑
        求解器 FragmentDirectionFeasible    VerifyParallelToPTO
        "选方向时筛不可行项"                 "放行前复核"
```

**判定不可能分叉**。若各写一套（历史上出现过），"求解器放行 Verify 拒"（误拒）或反过来（静默算错温床）都可能出现。附带好处：求解器先执行 → 同类非法程序的报错点固定在求解器、文案稳定可测试（改测试断言时就靠这个确定性）。

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

## 13. 设计要点回顾

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

## 14. 快问快答

- **为什么不简单地"维度大的当 lane"？** 转置访问会锁死两块 buffer 的选择必须相反；且"哪维连续"取决于 UB 访存形态（梳子问题），不是形状大小。
- **1D fragment 需要求解吗？** 走同一条路，但没有方向选择（唯一维就是 lane 维），约束退化为区间证明。
- **从没被访问的 fragment（floating）怎么办？** 也进求解（B5：step 0 会让它和引擎冲突），作为单节点自由分量拿字节最小布局；rank>2 的死分配不阻塞整个函数。
- **用户能干预吗？** `T.annotate_layout` 提供规范形态即作为 pinned 参与；非规范/非单射分类拒绝。
- **解出来的布局谁消费？** Plan 算 count/bytes（容量）、Vectorize 决定 entry×lanes 偏移、Verify 复核——三者的共同输入。
