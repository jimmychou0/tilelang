# TileLang Parallel 到 PTO：PR4-R4 Fragment 布局联合推导设计方案（v5）

> **实现状态（2026-10-23 更新）**：本设计已合入 PR-4（提交 `bdb70e3c` + 后续修复 `21a4d769`）——前扫描（A2 区域外拒绝、同 data 拒绝、跨区域拒绝、SimdVF 级注解拒绝、kernel 级注解校验为 pinned、无契约 TileOp 拒绝）、奇偶图（每访问单边编码：l = d XOR c，单侧可行转一元约束）、带奇偶并查集求解、字节最小选解 + 源码序首单元取内维破并列、注入为既定事实（step 0 跳过已布局缓冲，B5 关闭）、fragment 退出选维证明并按解布局约束方向（B1 关闭：`f[256,4]` 转置访问解出 lane dim 0 / 8 entry，结构与设备数值均通过）、`AnalyzeLaneUse` 值分类分派（带同一外层范围集）。**finalize 以 pinned 参与的接口（§3.3 末行）仍留待 PR-6**；广播布局读写判据与 count 规则同前（§3.4）。首轮实现（曾撤出）的三处致命错误已在重写中逐项修正：入口以 collector 传入的 target 判定、每访问恰一条边、节点 0 不再充当虚拟零点。**十五轮修复（`21a4d976`）**：DSU 重挂根时迁移已钉值（此前孤儿化导致构建结果依赖指针序，违反 §3.9/3.10 确定性）；AddUnary 冲突带两侧约束来源；未访问的 fragment 分配不再进入 rank/同 data 门（死缓冲不阻塞）。**终审更正（2026-10-23）**：§1.3 的 UB 多下标崩溃描述有误——收集 visitor 只记录 fragment，`B[j]+B[j+VL]` 从未崩溃且实测跑通（正文已更正）；§3.7 的共享 helper/② 兜底按实现口径重述（等价偏差，注入即跳过重提议）。

> 本文是 PR-4 的布局推导子设计（PR-4 总设计见《PR4 Fragment-slot 存储详细设计》）。核心：LayoutInference 前置的奇偶图区域求解，修复"2D fragment 的 lane 只能落末维"等布局层缺陷，并顺带关闭 PR4 的 A2/A4/B4/B5/A6。基线：`pto-tparallel-control-flow` @ `bc715beb`。历次设计评审的意见与采纳记录归档于《PR4 设计评审历程归档》，不占本文篇幅。

## 1. 问题背景

### 1.1 Fragment 布局与消费方

`local.fragment` Buffer 的逻辑布局（每个逻辑元素落到哪个 (entry, lane)）写入 `layout_map` 后被三处消费：`PlanPtoSlotStorage` 按 `OutputShape[0]` 算 entry 数与预留字节；Verify/Vectorize 共用的两证明（`AnalyzeFragmentAccess`）；Vectorize 发射 `vstore/vload(handle, entry×VL, ...)` 的 offset。布局写定后下游只能服从——写错轻则容量膨胀，重则程序被拒。

### 1.2 现状机制与两个真实阻塞点

`ParallelOpNode::InferPtoSimdLayout` 先由 `PtoRequiredVectorDim` 选维：`PtoAccessCollector` 收集全部访存（仅跳过单元素 local.var，**含 fragment**），对每个访问按**逻辑行主序 `ElemOffset`** 在补齐域上做连续性/lane-uniform 证明（parallel.cc:1030-1050、1150、1197-1226），双合法内维优先，双失败报 "no vectorizable dimension"（:1362-1371）。

两个**互相独立**的阻塞点：

**阻塞点一（布局推导选维）**：fragment 以逻辑行主序 stride 参与地址证明，而该 stride 不代表 backing 的保存方式——沿非末维取向时逻辑步长 ≠ 1，访问被判不连续。后果：**2D 单元的 lane 只能落在 fragment 的最后一个逻辑维**，取向完全一致的程序也被拒，与顺序无关。例（`f` 为 `[256,4]`，两单元取向一致沿 i）：

```python
for i, j in T.Parallel(256, 4):          # 生产：in_ub[j,i] 地址 j*256+i，沿 i 连续
    f[i, j] = in_ub[j, i]
for i, j in T.Parallel(256, 4):          # 消费：同取向
    out_ub[j, i] = f[i, j]
```
沿 i 时 `f[i,j]` 逻辑步长 4 → 不连续；沿 j 时 UB 不连续 → 第一单元即报 "no vectorizable dimension"，**到不了布局合并冲突**。

**阻塞点二（Verify 值分类）**：`CheckComputeValue` → `AnalyzeLaneUse`（verify_parallel_to_pto.cc:745-760）的 BufferLoad 分支不按 scope 分派，直接进 `AnalyzeBufferAccess`（parallel_to_pto_utils.cc:1093-1107），非连续访问抛 "cannot be classified"（:519-526）。修完阻塞点一，沿 i 读 `f[i,j]` 仍死在 Verify。

**现况唯一走到"布局冲突"的路径**：1D 单元（如 `f[i, 0]`）定 dim0 布局 + 2D 单元沿 j 提议 → layout_inference.cc:482 的**裸 `LOG(FATAL)`**。

### 1.3 受影响输入形态

1. **取向一致、但 lane 不在 fragment 末维**（主打击面，顺序无关被拒）；
2. **推导顺序型冲突**（1D+2D 混合 pin）——不可捕获 FATAL；
3. **多下标单元崩溃（终审更正：仅 fragment 受影响）**：`RecordBufferAccess` 的结构相等 ICHECK 只作用于 **fragment** 缓冲——收集 visitor（`ParallelLoopNestVisitor`）本身只对 `local.fragment` 访问调用记录，UB 缓冲从不进入该路径，`B[j]+B[j+VL]` 类 UB 多下标**从未在收集阶段崩溃**（终审实测：入口平移读正确发射两个 vload + vadd，完整跑通）。真实受影响的是 fragment 三形态：`f[i]=f[i]+f[i+VL]`、G2 展开的带偏移自拷贝 `f[i+128]=f[i]`——修复按 §3.1 的多组容器落地（限 fragment）。**注**：原地写法 `f[i]=f[i]+f[i+VL]` 仅在循环 extent ≤ VL 时是合法 Parallel 程序（extent 更大时点 i 读 `f[i+VL]` 与点 i+VL 写同一元素构成数据竞争，违反 m2 §11.2 输入契约）；测试用例一律改为 **out-of-place**（`out[j]=B[j]+B[j+VL]`）或显式限定 extent ≤ VL。带偏移自拷贝合法的前提是源/目的区间不重叠；重叠的 `T.copy` 经 G2 展开后同样竞争——**检查定在 `ExpandPtoFragmentFillCopy`**（展开时源/目的区间均已知；前扫描只见展开后的 Parallel 单元，跨逻辑点依赖分析 m2 §5.2 明确不做），重叠即分类拒绝。

### 1.4 时机

PR-6 的 materializer/finalize 将生成部分和/结果/seed Fragment 并被普通单元读写；PR-5 引入更多取向混合访问。缺陷留到那时将以"no vectorizable dimension / FATAL / 崩溃"三种面目在合法程序上复发。

## 2. 方案背景

### 2.1 目标设计要求（m2 §4.6/§8.6，不变）

生产消费共同约束同一布局；entry 各 lane 一致、存储 lane 与执行 lane 对应；需重排时诊断；Buffer 布局输入空间保留逻辑 shape E、循环求逆用补齐 P。

### 2.2 约束结构：奇偶图

每 2D 单元一个二值方向 `d`；每 2D fragment 一个二值 lane 维 `l`；**规范布局由 shape、l、VL 唯一确定**。UB 访问给 `d` 一元约束（现有连续性证明仅对 UB 成立）；每个（单元, fragment）访问给出 `d→l` 部分函数（双射/单边/不可行；**下标不依赖向量化变量 = 该方向不可行**——m2 §8.6 的单元素提取，现为 `continue` 放过）。传播后每个连通分量**要么奇环无解、要么恰两互为翻转的解**——带奇偶并查集一次求解，顺序无关；试探/回滚/"搜索未完成"永不需要。**PR-6 护栏**：该结论依赖"每条约束关系只是双射或一元"——PR-6 引入非规范布局（广播等）时，必须确认其约束边仍是双射/一元；一旦出现更复杂的关系（如随方向非线性变化的布局），"一次求解、恰两解"不再成立，需回到有界试探或分流求解。

### 2.3 路线决策与改动面（诚实口径）

路线维持**区域级求解**（引擎接入的三处硬冲突：step 1 strict 全量定死且不唤醒、合并点 ProveFragmentContains 替换分支、free-mode root 枚举 + cost model）。改动面不再声称"全部位于 PTO 分派内"：

| 改动 | 位置 | 性质 |
|---|---|---|
| 前置求解器、物化、诊断 | `parallel.cc` PTO 分派内 | 本设计主体 |
| 访问收集放宽（多下标，**opt-in 默认严格**） | `parallel.cc` `RecordBufferAccess`/ParallelOp 构造 | **上游 API 扩展，默认行为不变**见 §3.1 |
| 前扫描的区域分组与注解发现 | 前扫描自遍历函数体（或扩展引擎收集器，二选一） | 新增遍历（"单点挂载"只是调用位置，不含此项） |
| `AnalyzeLaneUse` scope 分派 + `LaneUseContext` 布局字段 | `parallel_to_pto_utils.{h,cc}`、Verify 的 `MakeLaneUseContext` | PTO 新路径 + 上下文扩展 |
| `Collect(f)` 与 `collector.Run()` 之间的前置调用点 | `layout_inference.cc` | 单点挂载 |
| **合并点 / propagate_alias / step 0 种子** | **不改** | — |

### 2.4 非目标

- 不修改 SIMT/CUDA/AscendC 推导语义；
- **R4 只接受规范注解**：非规范注解（含广播布局 lane=rep）一律"注解形态拒绝"；广播布局的读写判据（读：lane==执行 lane ∨ replicate；写侧规则**留给 PR-6 定义**（曾考虑"仅 lane==执行 lane"，会禁止 m2 §8.5 的 finalize 整体写回与 acc→result 拷贝；复制型布局的正确条件是"写入值各 lane 相等"或"同构广播 buffer 间逐 lane 拷贝"，此处不预固化）与 count 规则归 PR-6，R4 只留参与接口（§3.4）；
- 不做 cost model——仅"恰两解比较 backing 字节数取小，并列按程序序"（§3.2）；
- reducer v2 布局生成归 PR-6；不改公共 `IsEqual`。

## 3. 设计方案

### 3.0 求解时机与注入方式

求解器是 **LayoutInference 的前置步骤**，调用位置在 `collector.Collect(f)` 与 `collector.Run()` 之间（layout_inference.cc:1804-1805；`Run()` 是收集器方法、拿不到函数体——前置步骤从收集结果取 `infer_list` 与区域结构、对函数体的自遍历在其内部完成）：在**整个函数的推导对象收集完成之后**（`infer_list` 建好；kernel 级 `T.annotate_layout` 在外层 block body 访问完毕时才加载、晚于内层 SIMD_VF 的算子登记，故必须等收集全部完成——按区域即时求解会漏注解 pinned 约束）、**step 0 之前**运行：

```
LayoutInference 入口（Collect(f) 与 collector.Run() 之间挂前置步骤，:1804-1805）：
  收集全部推导对象（kernel 级注解加载完成）              # 引擎既有
  【新增】前扫描自遍历函数体：区域分组 + SimdVF 级注解发现   # 引擎只记 bool 型
                                                            # in_pto_simd_vf_vec_/pto_lane_bounds_，
                                                            # 无区域标识、不加载块注解
  【新增】PTO 前置：对每个转换中 SIMD_VF 区域
     前扫描（§3.3：注解/同 data/区域外访问/TileOp 契约检查）
     奇偶求解 → 每 fragment 规范布局
     连同注解 pinned 布局写入初始 layout_map              # 标准通道注入
  step 0 种子 / strict / common / free                    # 引擎既有，照常
```

- 注入走标准初始 layout_map：step 0 对 floating fragment 的 FullyReplicated 种子**本就只对无布局者下种**（layout_inference.cc:599-601 的 `if (layout_map.count(buffer)) continue;`)，已解 fragment 天然跳过，无需占位兜底。
- **SimdVF 作用域内注解**（挂在 SIMD_VF block 上）：引擎 VF 注解分支只对 SIMT_VF 加载（vf_regions.h:46-50），这类注解今天被静默忽略且随后被写回覆盖（:1844-1847）。本设计**不加载、保持拒绝**（终审实测：DSL 可构造的形态落在区域内的嵌套注解块上，经 pinned 约束与访问约束的矛盾路径 fail-closed 拒绝——拒绝本身可靠，专属报文为登记的形态差异项）（本仓与相邻仓中与 SimdVF 同文件的 annotate_layout 均为 kernel 级 UB/L1 NZ 布局；唯一 fragment 注解例子在 SimtVF 内且为 SIMT 线程布局——不存在"在 SimdVF 内手写规范 PTO 布局"的用户，报错与加载无实际差异；ST 套件若带 SIMT 线程布局注解迁入，加载后也会被规范校验拒掉）。三点细化：① 拒绝范围限定 **PTO 目标 + 转换区域内 + 注解指向区域 fragment**（AscendC 的 SIMD_VF 与 PTO 的 SIMT_VF 不受影响）；② 诊断给迁移指引（"PTO SIMD 区域内 fragment 布局由推导决定，SimtVF 时代的线程布局注解不适用，迁移时请删除"）；③ **ST 套件 grep 已完成（2026-10-21，PR head `peanutchan/tilelang-ai`@`593a8f0c`）**：覆盖全部 21 个 kernel + vf_mode.py + 两个 harness（从 PR diff 取完整文件，逐文件新增行数与 GitHub additions 一致），`annotate_layout`、`T.Fragment(`、`make_*_layout`、`layout_map` 均 **0 命中**——迁移删除清单为空，"SimdVF 级注解报错"不影响任何 ST kernel（kernel 经 vf_mode.py 在 SimtVF/SimdVF 间切换，simd-pto 即 M4 验收路径）。后续 PR head 变化时按同方法复查。kernel 级注解正常参与。
- 注入条目在 step 1 被当作 strict 事实（`strict_layout_map` 全量拷贝正是期望语义——已定布局 = 事实）；物化在各级调用只做与解集一致的生成，不产生需唤醒传播的新布局；`RunOneAttempt`/`InferInFreeMode` 对 PTO 分量不参与。
- 物化与求解看同一访问集合。

### 3.1 约束模型与收集

- 可行性判据 = **三项证明**：lane 匹配（存储 lane == 执行 lane）、entry 与 lane 无关、**entry 区间（双侧 + 全证明域）**——`min(entry) ≥ 0 且 max(entry) < count`（缺下界时 `f[j-VL]` 类负偏移写到 backing 前面的对象；count 按拟议规范布局算出）。证明域（两处，缺一不可）：① `AnalyzeFragmentAccess` 现的新建 analyzer 只绑定 lane/chunk/保留维（parallel_to_pto_utils.cc:603-620），**外层串行变量是自由的**——entry 依赖外层变量时（`for k: … f[k,j]`）证不出上界，"证不出就拒"会误拒今天能过的程序；② LayoutInference 收集器本身也不绑定（layout_inference.cc:1073-1074）——若只修 ①，`f[k,j]` 会在**求解器里被拒、在 Verify 里却能通过**，两处口径分裂。因此区间证明在两处都**绑定访问点的全部外层循环范围（含区域外的 kernel 级循环）**（Verify visitor 的 analyzer 已有这些绑定，复用其口径；前扫描求解侧自行 Bind，不依赖收集器）——只绑区域内循环的话，entry 依赖 kernel 级循环变量时求解器与 Verify 又会得出不同结论。证不出仍 → 分类拒绝。**区间证明与多下标同一提交落地**（多下标放开而区间不查会静默越界——`j` 覆盖全域时 `B[j]+B[j+VL]` 读写到 backing 之外，而 `AnalyzeFragmentAccess` 与 `CheckEntryOffset` 均不查动态区间，vectorize:1078-1112 只查粒度）。
- **多下标收集机制**：`RecordBufferAccess` 在 ParallelOp 构造时执行。把结构相等 ICHECK **全局放宽**为"存首组"不可行：该 ICHECK 守护**全部后端**（`ParallelOp(` 共 18 处构造点：CUDA/ROCm/Metal 的 copy/fill/transpose/atomic、通用 copy/fill、layout_inference 等），而 SIMT 下游只消费首组下标（`ValidateCandidateAgainstFragments`、layout_cost_model.cc:940 的 `GetIndiceMap()`、LowerTileOp 重映射）——"现可通过程序必为单组"只覆盖今天能通过的程序，**今天被 ICHECK 拦下的 SIMT 多下标程序**（如 CUDA 的 `frag[i]+frag[(i+1)%N]`）放宽后会按首组推导布局、第二组读到本线程另一个寄存器，**静默算错**。修正为 **opt-in**：开关加在 `ParallelOp(For)` **构造函数**参数上（默认严格），并**存为 ParallelOpNode 成员**——`ExpandBindValues` 在推导期补录访问（parallel.cc:364-368）时读同一成员，否则补录访问仍走单组路径、求解器与物化所见不一致。layout_inference.cc:980 传入的必须是**逐循环计算的 `InsidePtoSimdVF()`**（layout_inference.cc:823）——该收集器同时为 Ascend SimtVF 与 AscendC 的循环构造 ParallelOp，传常量 true 会让 SimtVF 路径出现静默错算。PTO 专用多组容器（`std::vector<BufferAccessInfo>`）随开关注入，求解器与 `InferPtoSimdLayout` 遍历全部组、组间合取——对上游是"API 扩展，默认行为不变"。补回归：SIMT 多下标程序仍报 ICHECK 原文、**另加 Ascend SimtVF 形态多下标回归**（§4-15 两项，防常量 true 误用）。
- 对齐偏移：`j+VL`/`i+128` 类在三项证明下是 entry 平移（lane 不变、区间可证）；`j+1` 类 lane 偏移不可行。
- UB 连续性证明仅施加于 shared/shared.dyn（fragment、local.var 退出）。

### 3.2 求解与并列规则

- 带奇偶并查集：一元约束（UB、注解 pinned）先传播；分量奇偶矛盾 → **已证明冲突**（诊断含每边来源单元与两端拟议布局）；
- 恰两解时**比较 backing 字节数取小**（`count = 其余维积 × ceil(lane维/VL)`）；**并列的次序由程序决定**（多 fragment 反向相连的分量里"内维优先"无定义——`g[j,i]=f[i,j]` 的两解恰好互换 f/g 的 lane 维）：**源码序第一个含自由方向的单元取内维**（其向量化坐标落在循环内维），分量内其余随之确定——总存在且唯一；
- 求解产物为最终布局，Plan 据此算容量（1.2 反例得 8 entry/4 KiB）。

### 3.3 前扫描：外部来源与确定性拒绝

前扫描在求解前依次执行，全部为**分类诊断**；因位于 LayoutInference 内、早于 Plan（pipeline.py 的 PlanPtoSlotStorage 挂点）与 Verify（Vectorize 前的验证 pass），天然消除三轮评审提出的"Plan 先于 Verify 报错"问题：

| 检查 | 处理 |
|---|---|
| kernel 级 `T.annotate_layout` 目标为区域内 fragment | 校验 PTO 规范形态（恰一维 forward_index、replicate==1、**lane 为 floor-mod 形态**；VL 一致性按 §3.9 口径比较——`ThreadExtent()==VL` 或先 `BindThreadRange` 再比较，用户注解可能未绑定 thread range）→ pinned 一元约束；**非规范一律"注解形态拒绝"**（广播布局等留 PR-6；"∨ replicate 通用判据"与 §3.8 的 Plan 断言、§3.5 判据互斥，不采用） |
| SimdVF 作用域内注解目标为区域内 fragment | 分类拒绝（现被引擎静默忽略，§3.0） |
| 区域内多个 fragment Buffer 共享同一 data Var | **确定性拒绝**："fragment 存储视图共享不支持"（现况三口径不一致，view 被拒只是处理顺序巧合且报文误导）。**判定口径**：只统计**区域里实际被访问**的 Buffer 对象；结构完全相同的对象（shape、strides、elem_offset、dtype 一致）视为同一个，避免把同一分配的重复对象误判为 view |
| 区域内 fragment 同时在转换区域之外被访问（kernel 级/区域级语句） | **确定性拒绝**（Verify 只查转换区域，"一个 VF 内用 + VF 外访问"今天无拒绝路径，不能声称"跨 VF 拒绝兜底"；本检查同时关闭 PR4 清单 A2 缺口，不依赖"两个 VF 都用才失败"） |
| 同一 fragment 被**两个转换区域**使用 | **并入前扫描诊断**：给出"fragment `f` 被区域 #a 与 #b 同时访问，跨 VF 使用不支持"——今天依赖 Plan/Verify 的"no plan in scope"报错，文案与根因不符 |
| **TileOp 参与方式** | **有 PTO 契约的 reducer TileOp 以方向依赖节点参与**：`finalize_reducer`（由 CanonicalizeLegacyReducer 在 LayoutInference 之前生成、作为 TileOp 参与推导）**不展开成 Parallel**（跨 lane 规约、`[1]` 累加器 fill 须覆盖全部 lane，均不可 Parallel 化；materializer 在 LayoutInference 之后，"先展开再求解"时序排不出）。其结果/累加器布局按 m2 §8.3 **由更新单元的向量化方向决定**（"选择另一条向量化维时，两类处理随之交换"）——更新语句读 2D fragment 时该方向是求解变量，故 reducer 布局建模为**与更新单元方向变量之间的双射边**（每个方向对应的布局与读写判据由 PR-6 提供）；仅当方向被其他约束钉死时才退化为 pinned。**TileOp 自带 InferLayout 的抑制**：finalize 等自带 InferLayout 的 TileOp 在 PTO 区域不得按 SIMT 逻辑提议布局（只可不提议、或只重申已注入布局），否则 §3.5"合并点只看到相等提议"的前提破坏、:482 FATAL 复活。**无 PTO 契约的 TileOp 拒绝，范围限定为访问区域内 fragment 的 TileOp**（照字面全局实现会误拒 PR-3 在 UB 上做的 direct reduce；fragment 上的 direct reduce 拒绝正确，m2 §7 只支持 UB 输入）。普通 fragment fill/copy 维持 G2 先行展开 |

### 3.4 PR-6 扩展点

接口：materializer/finalize 生成的 Buffer 布局作为**方向依赖节点**（§3.3：与更新单元方向变量的双射边；方向被钉死时退化 pinned）经注入通道进入求解。PR-6 需自带：(a) 该形态的**读写判据**扩展（`AnalyzeFragmentAccess`：读 lane==执行 lane ∨ replicate；写侧规则由 PR-6 定义——复制型布局的正确条件是"写入值各 lane 相等"或同构广播 buffer 间逐 lane 拷贝，不是"仅 lane==执行 lane"——但**同构性前提**（各 lane 值相等的不变式）由写侧维护，规则细节不在此预固化）；(b) 该形态的 **count 规则**。R4 不实现，且 R4 的 Plan 断言对 PR-6 形态留有按契约放行的开关。

### 3.5 合并点与别名：不改动

- **可达性论证**：解集在 step 0 前注入初始 layout_map；物化在返回前与解集核对（②，不等即分类诊断，先于任何合并）；因此**合并点只会看到相等的 PTO fragment 提议**，step 1 后布局进入 strict_layout_map，`ProveFragmentContains` 替换分支对 PTO 条目不可达，:482 裸 FATAL 随之不可达（回归钉死）。
- propagate_alias 的 `ICHECK(IsEqual)`（:362-367）**不再需要屏蔽**：同 data 共享已在前扫描确定性拒绝，PTO fragment 不会与任何其他 Buffer 共享 data 进入别名传播。
- 保留 ② 作为防御断言（PTO 分派内），不改通用代码。
- **判据修正**：`PtoFragmentLayoutEquivalent` 比较逻辑 shape、forward_thread、forward_index、replicate、thread_range——**不含 dtype**（dtype 非 Layout 字段；dtype 一致性在 Plan/Verify 侧按 Buffer 检查）。生效点仅两处：注入时与注解 pinned 的核对、物化 ②。

### 3.6 Verify 值分类分派

`AnalyzeLaneUse` 的 BufferLoad 分支按 scope 分派：fragment → `AnalyzeFragmentAccess`（按已定布局跑三项证明；**判据维持 lane 相等**——"∨ replicate"属 PR-6，R4 不加新语义）；UB → 现行路径。`LaneUseContext` 需新增布局查询字段（现无此字段，parallel_to_pto_utils.h:323-340），Verify 的 `MakeLaneUseContext` 相应构造——§3.8 如实列入（"Verify 无结构性改动"作废）。lane 偏移 fragment 读出现在值位置时走本路径报分类诊断（测试钉死）。

### 3.7 方向存储与访问集一致性

- **方向不存节点成员**（opt-in 开关与多下标容器是 ParallelOpNode 新成员，此处指**求解变量**不落节点）：单元方向 `d` 由物化阶段**从解集反推**——单元 fragment 访问在 pinned 布局下的依赖维即方向，UB 一元约束同向核对；解在分量内唯一（§3.2 并列规则定死），反推确定。无 fragment 访问且 UB 双向合法的单元，方向不影响任何 fragment 布局，维持现有内维偏好。
- **访问集一致（终审按实现口径表述）**：求解器自带与消费方同语义的访问扫描器（平坦序列 bind + AttrStmt 作用域，同 `ExpandBindValues` 的词法规则），保证求解器所见 = 物化所见；两者为**独立实现而非共享 helper**（登记的等价偏差：行为一致由两侧同规则维护，无 ② ICHECK 兜底——注入条目在 `InferPtoSimdLayout` 中直接跳过重提议，比"物化后核对相等"更强，物化与解集分歧的路径不可达）。

### 3.8 代码接入点

| 文件 | 改动 |
|---|---|
| `src/op/parallel.cc` | 访问收集放宽 + PTO 多组容器（§3.1，限 fragment）；fragment 退出选维地址证明；`InferPtoSimdLayout` 按注入跳过重提议（终审口径，见 §3.7） |
| `src/ascend/transform/pto_fragment_layout_solver.{h,cc}` | 前置求解器（前扫描 + 奇偶 + 注入，独立文件——设计早期写 parallel.cc，实现落在专用文件） |
| `src/ascend/transform/layout_inference.cc` | 前置调用点位于 `Collect(f)` 与 `Run()` 之间（:1804-1805）；前扫描的区域分组/注解发现由**前置步骤自遍历函数体**承担（引擎无区域标识、不加载块注解，扩展收集器为备选）；**不改**合并点/propagate_alias/step 0 |
| `src/ascend/transform/parallel_to_pto_utils.{h,cc}` | `AnalyzeLaneUse` scope 分派；`LaneUseContext` 布局字段；entry 区间证明并入 `AnalyzeFragmentAccess` |
| `src/ascend/transform/verify_parallel_to_pto.cc` | `MakeLaneUseContext` 构造扩展 |
| `plan_pto_slot_storage.cc` | Plan 规范形态断言（仅对被 Parallel 单元访问的 fragment；报**分类诊断**且晚于前扫描） |
| 测试 | §4 |

### 3.9 诊断与来源记录

每条约束边携带 `{来源单元 id, 访问文本}`；求解产物布局携带提交来源（求解器/注解 pinned/方向依赖节点）。诊断分类：

| 类别 | 触发 | 内容 |
|---|---|---|
| 已证明冲突 | 分量奇偶矛盾 / 注入核对不等 | 双方布局 DebugOutput + 各自来源单元 |
| 访问不可行 | 三项证明失败（lane 偏移、跨 entry 拼接、单元素提取、需要重排） | 具体访问 + 拟议布局 |
| entry 区间违例 | `min(entry)<0` 或 `max(entry)≥count` | 指出越界方向与界值（双侧） |
| 注解形态拒绝 | 非规范 annotate / SimdVF 级注解 | 求解器路径给出注解内容 + 期望形态（§3.3）；引擎先命中时为引擎的形态检查/冲突报文（终审登记的形态差异，拒绝本身 fail-closed） |
| 同 data 共享 / 跨区域使用 / 区域外访问 | 前扫描 | 涉及对象与区域标识 |
| 选维无解（仅 UB） | UB 两方向均不连续 | 现有文案（原因串保留） |

注解规范形态校验的比较口径：`ThreadExtent()==VL`，或先 `BindThreadRange` 再比较——用户手写的 `T.Fragment` 注解可能未绑定 thread range)。

### 3.10 确定性

结构保证：单次求解 + 程序序并列规则，无遍历依赖、无 cost model、无试探。验收用**固定解集断言**（同一输入断言布局与 count 逐值）。

## 4. 测试与验收

结构测试（144）：

1. **取向一致、lane 不在末维**（主修面）：共同沿 i、count=8、offset 一致；
2. **1D 单元 + 2D fragment**（`f[i,0]` 类 pin dim0）；
3. 同单元 out-of-place `out[j]=B[j]+B[j+VL]`（UB 多下标——**终审实测本就通过**，UB 不进收集 ICHECK，无需修复即有正例）、`out[i]=f[i]+f[i+VL]`（fragment 多下标 + entry 平移 + **区间证明**；原地写法仅 extent ≤ VL 时附带一例并注明契约依据）；
4. 带偏移自拷贝（G2 展开 `f[i+128]=f[i]`，**区间不重叠**为前提；重叠自拷贝负例 → ExpandPtoFragmentFillCopy 拒绝）；
5. **`+VL` 越界负例**：`j` 全域覆盖时触发区间拒绝）；
6. `f[i+1]` 负例：lane 偏移不可行；
7. **注解**：kernel 级规范注解作为 pinned 生效（已测：pin 翻转等字节并列定解）；kernel 级非规范/形态不符拒绝（已测：fail-closed，报文形态见 §3.3）；**SimdVF 级注解拒绝**（已测：DSL 可构造形态经冲突路径 fail-closed）；
8. **alias/view 负例**：同 data 两 fragment Buffer → 前扫描确定性拒绝、报文准确；
9. **跨边界负例**：区域内 fragment 在 VF 外被访问 → 前扫描拒绝；
10. **退化维单元连 fragment**：`M==1`/`N==1` 与两解取小的一致性（对照 m2 §4.6.1）；
11. **多 fragment 反向分量并列规则**：`g[j,i]=f[i,j]` 无约束分量按源码序第一单元取内维，断言具体解；
12. 冲突诊断：UB 双向强制 → "已证明冲突"含来源单元；:482 不可达回归；
13. **lane 偏移读在值位置** → Verify 值分类路径报分类诊断；
14. 非整除补齐、解集固定断言、SIMT/AscendC/slot 回归 46/46 不降；
15. **SIMT 多下标仍报 ICHECK 原文**（opt-in 默认侧）：CUDA 形态 `frag[i]+frag[(i+1)%N]` 与 **Ascend SimtVF 形态**各一例；
16. **区间证明正/负例**)：外层串行变量下标 `f[k,j]` 编译通过且区间证明成立；`f[j-VL]` 负偏移被拒并指出下界违例；
17. **UB-only TileOp 不误拒**：转换区域内只访问 UB 的 TileOp（PR-3 direct reduce 形态）不被"无契约 TileOp"拒绝；
18. **跨区域 fragment 诊断**)：一个 fragment 被两个转换区域使用 → 前扫描诊断含两个区域标识（不再出现误导性 no-plan 报文）；
19. **ST 套件 grep（已完成，见 §3.0 决策③）**：`peanutchan/tilelang-ai`@`593a8f0c` 全部 21 个 kernel + harness，`annotate_layout`/`T.Fragment(`/`make_*_layout`/`layout_map` 零命中；head 变化时复查。

设备测试（238）：沿 dim0 数值用例（现 2D fragment 用例全为 `[M,VL]` 沿 j）；out-of-place `out[j]=B[j]+B[j+VL]` 数值（extent > VL，合法并行形态）；不重叠自拷贝数值。

## 5. 风险与开放点

| # | 事项 | 应对 |
|---|---|---|
| 1 | 访问收集放宽对 SIMT 的影响 | 为 opt-in：非 PTO 构造点默认严格、行为逐字节不变；SIMT 多下标回归钉死（§4-15） |
| 2 | floating fragment 种子与已解 fragment 的互不干扰 | 种子有条件（:599-601 已核实）；测试钉死 |
| 3 | 求解器访问扫描器与 `ExpandBindValues` 词法规则漂移 | 独立实现同规则维护（等价偏差，§3.7）；漂移由 Verify 的逐访问证明兜住（不一致的访问在 Verify 处分类拒绝，不会静默） |
| 4 | finalize 方向依赖节点参与接口与 PR-6 §8.3 最终形态偏差 | 接口先行，PR-6 对齐补测 |
| 5 | 上游改动收敛清单（收集放宽、`Collect(f)` 与 `Run()` 之间的单点挂载、utils/verify 上下文扩展） | 合入前与 base owner 对齐，附固定解集断言 |

## 6. 实施切分

1. **提交 1（收集与证明，行为保持/扩展）**：多下标收集（**opt-in 参数 + 节点成员 + ExpandBindValues 同源**)+ **entry 区间证明（双侧 + 两处证明域绑定)** + `AnalyzeLaneUse`/`LaneUseContext` scope 分派——区间与多下标同提交)；非 PTO 构造点行为不变。**本提交只把"多下标崩溃"转为分类诊断；layout_inference.cc:482 的裸 FATAL 到提交 2（求解 + 物化核对）才消失**；附 §4-15/16 正负例；
2. **提交 2（求解，原子）**：fragment 退出选维证明 + 前置求解器（前扫描 + 奇偶 + 注入）+ 物化/反推/②——证明退出与求解同提交；
3. **提交 3（收尾）**：Plan 规范断言（分类诊断）、诊断来源记录、负例补齐；
4. **提交 4（设备验收）**：沿 dim0 数值用例 + 全量回归。
