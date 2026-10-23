# TileLang Parallel 到 PTO：PR-4 Fragment/slot 存储详细设计

> **当前状态（2026-10-23，十五轮后；历史评审轮次细节见《PR4 设计评审历程归档》）**
>
> **分支**：`zjm/pto-slot-clean`（PR jimmychou0/tilelang#2，base `pto-tparallel-control-flow`@`bc715beb` 之上 18 提交；HEAD `21a4d976`）。
>
> **已落地**（按设计章节）：Fragment/slot 存储全链路——布局联合求解器（R4 重写合入：前扫描 A2/同 data/跨区域/注解/TileOp 拒绝、奇偶求解、字节最小定解 + 源码序破并列、注入为既定事实，B1/B4/B5 关闭）、PlanPtoSlotStorage + AutoSchedule 预算扣减 + Merge 容量终检（含 no-contract 路径）+ 逐 backing 字节区间不相交两调度路径断言（D2/D3 关闭）、Vectorize handle 化转换（declare_fragment_buffer + vload/vstore、整除粒度证明、kernel 作用域清理、predicate 声明收集，D5/D7 关闭）、G2 fill/copy 展开（未对齐 1D 谓词化、**fragment 侧门控**、跨 dtype Cast、重叠/同 data 拒绝）、G3 谓词合成（base Legalize）、A4 双侧区间证明（区域级/kernel 级/split 提升循环范围两侧同集）、J3 组合发射（brc 含 group、dintlv 两连读+vdintlv、intlv 写 vintlv+两连写、continuous 静默、无 mask 合成全真 mask）、Q4 投影拒绝、U3 单 root 断言、A6 分类诊断、诊断批。**确定性**（R4 §3.9/3.10）：DSU 约束迁移修复 + 冲突来源 + 逐值断言用例。
>
> **保序（G4/A1，当前口径——待 P1-1 决策更新）**：TileLang 侧保守 mem_bar 已全部移除，backing 访问保序依赖 PR307 自动 vecscope bar；**已知风险**（评审 P1-1）：PR307 对 >2 层循环嵌套与 offset 依赖 vecscope 外变量的形态只记 Unknown 不插 bar，而本 PR 新增的 k×multi-chunk 2D、kernel 级 `f[b,j]`、split `f[t,i]` 恰好命中；现有设备全绿是幸存者偏差（测试用例都在建模包络内，shell 累加器 offset=0 可能被 mem-to-value 直接转发）。**修复方案待定**（区域门控 vs 全量恢复 vs 拒绝），定案后本节与 §6 决策表同步。
>
> **工具链下限**（唯一口径）：PTOAS master ≥ `d30269595`（0.69，PR307 已上游）+ MR274 补丁（memref/mem-to-value，未上游）；CANN 编译器 ≥ weekly-0909 系（fmaximum 标量选择）；MR274 合入后收敛为纯 master。
>
> **验证（2026-10-23，238，干净构建）**：结构 48/48（含求解器确定性/注解 pinned/unroll/死缓冲容忍/UB 多下标/U3 双 kernel）；全 pto 目录（设备含）上轮 628/628（本轮修复后全量复验在跑）；中间提交 4e649c5b/aeb9a140/bdb70e3c 的逐提交验证待跑。
>
> **明确不属本 PR**（按裁定）：shell 累加器 vreg 化（m2 归 PTOAS mem-to-value 动态 offset 转发）、单元素提取（m2 §8.6）、2D 未对齐 fill/copy、投影改写 plan（base 机制）、广播布局与 finalize pinned（PR-6）。
>
> **登记的形态差异**（非功能缺口）：非规范注解报引擎内部 Check；SimdVF 级注解经冲突路径 fail-closed；A6/预算 LOG(FATAL) 形态；同 data 视图防御路径（DSL 不可构造）；M9 等价偏差（求解器自带同语义扫描器；注入即跳过重提议）。
>
> **PTOAS 问题清单（当前有效项）**：①fmaximum 标量 max 在 CANN GA 上无选择（weekly-0909 已修，声明版本下限即关闭）；②mem-to-value 动态 offset 转发（性能增强）；③PR307 对三类形态的覆盖保证确认（形式确认，设备已实证建模包络内）；④MR274 合入计划；⑤跨 vecscope 保序语义（连续 vecscope 间无同步指令——源码证据已取，需设备 A/B + 答复）。

## 1. 概述

### 1.1 背景与问题

`target="pto"` 下，PR-1/M1 已能把 `T.SimdVF(lanes=...)` 内的标量 `T.Parallel` 程序自动向量化为 PTO VMI，但只支持 UB（`shared`/`shared.dyn`）上的访存。用户用 `T.alloc_fragment` 分配的 Buffer（TIR scope 为 `local.fragment`）原本被 `VerifyParallelToPTO` 直接拒绝。典型受影响的写法是"值要跨循环存活"，以 sv3（GEMV 部分和，gym `benches/sv3` 验收用例）为例：

```python
acc = T.alloc_fragment((CM, VL), "float32")     # 累加器，比如 (24, 128)
for i, j in T.Parallel(CM, VL):                 # 置零
    acc[i, j] = 0.0
for k in T.serial(K):                           # 外层串行循环
    xk = ...
    for i, j in T.Parallel(CM, VL):             # 每轮乘加，acc 保存上一轮结果
        acc[i, j] = acc[i, j] + cast(a_ub[k, i, j]) * xk
for i, j in T.Parallel(CM, VL):                 # 最终写回
    out_ub[i, j] = acc[i, j]
```

打通这类程序需要同时解决三个问题，它们决定了本设计的形状：

1. **存储空间要在调度之前确定。** `acc` 的理想物理位置是向量寄存器（KEEP-in-RF），寄存器不够时落到 UB；选择权在 PTOAS。但 AutoSchedule 要按剩余 UB 容量决定普通 Buffer 能否双缓冲，所以 TileLang 必须在调度之前、在还不知道这段空间最终是否被真正读写的时候，就按"会被读写"预留好容量。
2. **读写需要一种不预设物理实现的指令表达。** "`acc[i,j]` 的读"在向量语义下是"读取第 i 个 entry 的整条 VL-lane 向量"。表达是：每个 Fragment backing 一次 `declare_fragment_buffer` 声明返回 handle，读写复用普通 `vload/vstore`，以 handle 为基址、`element_offset = entry_index × lanes` 为偏移；物理上走寄存器还是走 UB 由 PTOAS 的 mem-to-value 决定（现实限定：mem-to-value 仅转发常量 offset，slot 的 `entry_var×VL` 动态访存全部保留在 UB——寄存器驻留目前只对常量 entry 用例成立，登记 A1/§8.2）。
3. **地址必须确定且不与其他数据重叠。** 若 PTOAS 保留实际访存，entry 到字节的映射由 TileLang 唯一确定，且这段 UB 不与任何其他对象（包括其他 Fragment 存储）复用地址。

### 1.2 目标与交付物

目标一句话：`local.fragment` Buffer 在调度前被规划为带固定 UB backing 的 Fragment 存储对象，Verify 按 `layout_map` 契约准入其访问，Vectorize 在所属 VF 内生成一次声明并把它全部读写转换为 handle 基址的普通 `vload/vstore`，地址由既有 lowering/合并 pass 演进为最终 UB 字节偏移并输出 `pto.memref_declare`，PTOAS 决定寄存器驻留或实际访存。

| # | 交付物 | 内容 | 状态 |
|---|---|---|---|
| D1 | Fragment Buffer 布局契约 | 联合求解器为 Fragment Buffer 推导 entry/lane 映射并写回 `layout_map` | ✅ 已实现（B1 关闭：lane 可落任意逻辑维，字节最小定解） |
| D2 | `PlanPtoSlotStorage` pass | 调度前声明 backing、记录规划信息（含预建 `fragment_mem_var` 与 `vector_type_name`） | ✅ 已实现 |
| D3 | UB 容量预算 | AutoSchedule 扣减预留总量；无调度路径做总量检查 | 🔶 部分实现（生效范围与缺口见 §5.3） |
| D4 | 独占地址分配 | backing 与普通 UB、其他 Fragment 存储互不重叠 | ✅ 已实现（直接证据待补，§8.2） |
| D5 | Fragment 存储接口 | `tl.vmi.declare_fragment_buffer` 注册、handle 基址 vload/vstore 发射、codegen `pto.memref_declare` | ✅ 已实现（合并栈设备复验通过） |
| D6 | Verify 准入扩展 | `AnalyzeFragmentAccess` 两证明 + 双侧 entry 区间；放行合法访问，分类拒绝非法形态 | ✅ 已实现（A2/A3 关闭；值分类分派带同一外层范围集） |
| D7 | Vectorize Storage 模块 | 声明发射、读写/fill/copy/条件写转换 | ✅ 已实现（G2 未对齐 1D 谓词化、fragment 侧门控、跨 dtype Cast；B2/B3 关闭；A5 广播布局归 PR-6） |
| D8 | 测试与验收 | 结构/容量/负例契约测试、设备用例、兼容性回归 | 🔶 部分（断言缺陷与覆盖缺口见 §8.2） |

### 1.3 基线与前提（当前状态）

- **实现基线**：`pto-tparallel-control-flow` @ `bc715beb`（含 M1/M2/M3 全部：控制流、vecscope 分裂/carry、**单侧条件写谓词化 store**（Legalize 直接产出 predicate，嵌套 And/Or 合取，Vectorize 降为 mask_and/or/not——取代本文此前的 G3 Select 模式匹配）；split 分配提升、carry 决策对齐、per-lane alloc_var 提升；**独立 AIV AssignCore 前移至 Plan 之后**（pipeline.py:120/126/129）——引入 SubcoreProjector 与 slot plan 的交互，见 §8.1 Q4）+ 本文提交（见 §9）。
- **语义规范**：新设计（m2 版）§4.5、§4.6、§5、§6、§9；旧版与 m2 版的差异以 m2 为准。里程碑与验收：路线图 §6、§10.1、§11.3。
- **PTOAS 前提（已定版，2026-10-22）**：**工具链以当前 `pr274-pr307-merge` lineage 为唯一验证目标**——MR274 与 PR307 均已确定终将合入上游，本 PR 不以它们的上游时点为阻塞；最低版本声明为"含 MR274（memref_declare + mem-to-value）与 PR307（vecscope 自动 mem_bar）及 dist_mode 移除（PTOAS !293）的第一个发布版本"。此前的 `vmi-0.1.9` wheel 无 memref_declare，G1 后本就不适用；组合发射（J3）也只在该 lineage 上验证，旧 wheel 不再作为兼容目标。mem-to-value 仅转发常量 offset（动态 entry 访存不被消除）；`memref_declare` 位于 vecscope 内部的用法由我方设备验证覆盖。
- **工具链合入状态（决策后）**：MR274 与 PR307 均将合入上游（已确认），本 PR 不阻塞于其时点；设备验证以 `pr274-pr307-merge` 为准并将在其合入后的第一个发布版本上复验一次（声明为最低版本）。

### 1.4 与新设计的差距收敛状态

| # | 差距 | 新设计出处 | 状态 |
|---|---|---|---|
| G1 | `declare_fragment_buffer` 声明 + handle 基址访存 + codegen `pto.memref_declare` + Plan 恢复 `fragment_mem_var`/`vector_type_name` + 后置检查改为从声明恢复 | §6.2–6.4、§5.4、§6.5、§12.2 | ✅ 已实现并设备复验通过 |
| G2 | fill/copy 纳入支持：region min/extent、部分 fill 目标 mask、copy 源/目的独立 offset | §6.2 末段、§8.5 | ✅ 已关闭（未对齐 1D 子区间经"基址对齐下取 + 谓词化"展开；fragment→fragment 需模 VL 同余；UB↔fragment 未对齐拒绝；注解化调用拒绝；正负例 + 设备数值通过） |
| G3 | 条件写回收敛为 masked vstore：predicate 与 parent mask 合成（普通 UB 与 Fragment 两路），不读旧值/vsel | §5.3、§9.1 第 2/5 类、§9.4 | ✅ 由 base 谓词化 Legalize + Vectorize 统一 mask_and 实现（base 先产出 predicate store，本文 fragment 分支消费同一 predicate；嵌套 And/Or 由 base 降为 mask 合取——早期 G3 Select 模式匹配方案随 base 演进删除）；合并栈设备复验通过 |
| G4 | mem_bar 责任：新设计把 VF 内保序完全划给 PTOAS | §9.5 | ✅ 已按 m2 划界落地（十二轮）：TileLang 侧保守 bar 移除，保序归 PR307（工具链下限 = 合并 lineage） |

架构层（容量预算、独占分配、backing 单份/不版本化、entry×lanes 偏移、chunk mask 写回）吻合新设计；**布局契约除外**——现实现中 2D fragment 的 lane 只能落在最后一个逻辑维（选维证明与 Verify 值分类均按逻辑行主序判连续，取向一致的程序也被拒，登记 B1），由 R4 v5 子设计解决（LayoutInference 前置的奇偶图区域求解，并顺带关闭 A2/A4/B4/B5/A6）。当前实现为"查表推导 + 已有条目校验 + 冲突即诊断"，是 R4 落地前的过渡状态。

## 2. 范围与非目标

### 2.1 非目标（明确归属）

| 内容 | 归属 |
|---|---|
| reducer v2（PartialFragment、部分和/结果/seed Buffer 的布局与 materialize） | PR-6。本文 layout_map 契约只覆盖普通 Fragment Buffer。口径（依 R4 v5 §3.3）：G2 的前置展开只覆盖**用户源码中的** fragment fill/copy（时序在推断前、可 Parallel 化）；materializer 生成的 fill/finalize/copy **不展开**——由 Storage/Reduce 模块**直接消费**：`finalize_reducer` 作为有 PTO 契约的 TileOp 以**方向依赖节点**参与 R4 求解（布局由更新单元方向决定，PR-6 提供各方向布局与读写判据）；`[1]` 累加器 fill 与跨 lane 规约不可 Parallel 化，同为直接消费（登记 A3） |
| gather/scatter、打包类型（sv4、sp1–sp6） | PR-5 |
| Parallel 体内串行循环的通用处理 | m2-on-m1 已落地常量界 shell 保留 + carry 提升 + vecscope 分裂（原分裂地址断裂 bug J1 已由 base 修复关闭）；**不含 fragment 读改写**——[VL] fragment 串行累加器会触发 carry/fragment 分派不一致（登记 B2） |
| `local.var` 标量（`T.alloc_var`） | m2-on-m1 已放行 |
| slot/Fragment 存储驻留决策、物理寄存器拆分 | PTOAS P2 |
| vmi.py 的手写 declare 接口 | Fragment 存储是编译器管理对象（新设计 §6.2），TIR 打印走 `TScriptPrinterName` 自动注册；手写面待确有需求另行评审 |
| `memref_view` 静态分区命名 | 新设计 §6.5 明确"默认转换无需新增 TIR view 接口"，不归 TileLang |

### 2.2 依赖

- 硬依赖：M1/M2 基线（`m2-on-m1`，含控制流与新的区域结构 `ConvertRegionBody`/carry/split/roundtrip-elimination）——Storage 模块已集成进该结构。
- PTOAS：MR274 及以上为下限（`pto.memref_declare`）；验证基线为本地合并分支 `pr274-pr307-merge`（含 PR307 自动 mem_bar）——两个 MR 均未合入上游 master，见 §1.3。`vmi-0.1.9` 为换形前的历史基线，G1 后不再适用。
- 联合验收依赖（不阻塞代码合入）：PR-3（direct reduce，sv9）；PR-2 侧待办（§8 J2；J1 已关闭）。

## 3. 总体设计

### 3.1 核心概念：Fragment 存储与 memref

一个 Fragment 存储是编译器管理的向量数组（术语沿用新设计 §1.2：Fragment backing = 预留 UB Buffer；Fragment 存储句柄 = 声明返回的 Handle Var；Fragment memref = PTOAS 中对该 UB 地址的类型化视图）：

- `count` 个 entry，每个 entry 保存一个完整的 VL-lane 向量（`vector_type`，如 `vector<128xf32>`）；
- 绑定一块固定预留的 `shared.dyn` backing，容量 `align_up(count × VL × 元素字节数, 32)`；
- **声明一次/VF**：`Bind(handle, declare_fragment_buffer(backing_ptr, count, lanes))`，在所属 VF 内定义并支配全部使用，跨顺序 Parallel 共用同一 handle；handle 为每区域一次的全新 Var（命名取自 plan 的 `fragment_mem_var`）；
- 读写复用普通 VMI：`vload(fragment_mem, entry_index × lanes, size=lanes)` 读完整 entry；`vstore(value, fragment_mem, entry_index × lanes, active_mask)` 更新 active lane、inactive lane 保留目标旧值；
- PTOAS 可将读写对替换为寄存器值传递（mem-to-value 消除访存），也可保留对 backing 的实际读写；消除后预留地址不回收。现实限定：mem-to-value 仅转发常量 offset——slot 的动态 entry（`entry_var×VL`）访存全部保留在 UB 并由 mem_bar 保序。

只有 Vectorize 输入 TIR 中已分配为 `local.fragment` 的 Buffer 才映射为 Fragment 存储；普通算术、广播、cast 的中间值走向量 SSA 值，不建存储。

### 3.2 职责划分

| 事项 | TileLang（本 PR） | PTOAS（P2/MR274） |
|---|---|---|
| entry 数量、向量类型、访问索引 | 确定并写入声明与读写 | 消费 |
| UB 容量 | 调度前固定预留；合并时分配独占地址 | 使用已预留地址，不可返还 |
| 物理实现 | 不做 liveness、不决定驻留 | 寄存器驻留或保留访存的决策、物理寄存器拆分 |
| 同步 | 保留既有外层任务同步；（有意偏差）写 backing 的单元后自插 mem_bar | VF 内保留访存的依赖分析与 mem_bar；mem-to-value 删除被转发抵消的 bar |

### 3.3 数据流总览

```text
（前端 → Simplify）
ExpandPtoFragmentFillCopy               [新] fragment 触碰的 T.fill/T.copy 就地展开为 T.Parallel 循环
        ↓
LayoutInference                         [R4 v5 前置，落地后新增] 前扫描（注解/同data/区域外/TileOp 契约）
        ↓                                 → 奇偶求解 → 注入；引擎阶段照常：物化 + kLayoutMap 写回
AscendInsertOOBPadding
        ↓
PlanPtoSlotStorage                      读布局 → 计算 count/vector_type/bytes → 声明 shared.dyn backing
        ↓                                 → 记录 PtoSlotPlan（含预建 fragment_mem_var 与 vector_type_name）
AssignCore（SubcoreProjector）              [base] 独立 AIV 投影：为需子核局部值的 buffer 生成 _aiv{p} 副本（不认识 slot plan，见 §8.1 Q4）
        ↓
AutoSchedule                            shared 容量上限扣减预留总量
  （或关闭调度 → InferBufferAliases   显式排除 backing 复用）
        ↓
NormalizeBufferVersion → VerifyParallelToPTO → Legalize → EliminateSimdVFRoundTrip → VectorizeParallelToPTO
                                      Verify 放行并按两证明检查 Fragment 访问；
                                           Vectorize 在 VF 内 Bind 一次声明（全新同名 handle），
                                           读写 → handle 基址 vload/vstore（offset = entry×lanes），
                                           单侧条件写恢复谓词合成 store mask，
                                           写 backing 的单元后跟 vecscope mem_bar
        ↓
AscendLowerTileOp / LowerAccessPtr / FlattenBuffer / MergeUBAllocations
                                      改写声明的 backing 地址操作数（VMI 继续引用同一 handle）；
                                          backing 独占偏移
        ↓
PTO codegen                             声明特判 → pto.memref_declare(字节地址, (count*lanes,), dtype)；
                                          handle 打印为 memref 名，读写输出 pto.vmi.vload/vstore
        ↓
PTOAS（mem-to-value）                    驻留/访存决策、mem_bar
```

字节地址口径：`shared.dyn` 指针在 PTODSL 中即 `castptr(const(0, i64), T*)`，UB arena 基址为 0，因此声明的字节地址 = backing 元素偏移 × 元素字节数，与换形前 `addptr(buf_dyn_shmem, O + entry×lanes)` 形态数值等价（据 PTOAS 设计文档 §3.3 快照，两形态逐条一致）。

### 3.4 已核实的前提事实（按固定基线 pto-tparallel-control-flow@bc715beb 核对；基线变化时复验）

1. AutoSchedule 容量预算**只从任务读写 region 枚举 buffer**（`auto_schedule.cc:815-828`），backing 在 Vectorize 前无读写——不重复计数，但必须显式注入预算。
2. 唯一"storage Var 无引用即删分配"的清理是 `AscendRemoveNoOp`，位于 Vectorize **之后**；但 m2-on-m1 新增的 `EliminateSimdVFRoundTrip` 在 Vectorize **之前**运行且会按访问剪枝分配——**backing 必须豁免**（已实现：剪枝豁免 `tl.pto_slot_plan` 登记的 backing Var，见 §5.4）。
3. m2-on-m1 的 vecscope 分裂（Plan A）对任何 RMW 单元触发；base 分裂把区域局部分配提升到外层共享（新基线 d1a2e96f）（vectorize:723-729），backing 不再拿多个地址，但 fragment carry 走分裂仍改变保序结构——vecscope-split 一侧已限 UB scope（§5.6-9），区域级 carry 一侧未过滤（登记 B2；修复时可评估 fragment 串行 RMW 改走分裂：vecscope 切换是完整同步点）。
4. VPTO 拒绝 vecscope 边界穿越的 pset 值——声明与全量 mask 的 Bind 都在 `tl.simdvf_scope` 之内。
5. SBlock 是调度任务边界，注解随任务载体在内存 IR 变换中存活（`tl.simdvf_lanes` 在调度链后仍可读为证）。

## 4. 数据结构与接口定义

### 4.1 Fragment Buffer 布局（layout_map 契约，D1）

- **注解键**：复用 `attr::kLayoutMap`（`src/layout/layout.h:465`），值为 `Map<Buffer, Layout>`；条目是普通 Fragment Buffer → `Fragment` 对象。
- **entry/lane 映射**（`VL` 为 lanes；向量化维由 UB store 连续性等约束决定）：

| Buffer 形态 | 元素 → (entry, lane) | entry 数 count |
|---|---|---|
| 一维 `B[N]` | entry `j//VL`，lane `j%VL` | `ceil(N/VL)` |
| 二维 `B[M,N]`，j 向量化 | entry `i*Q_N + j//VL`（`Q_N=ceil(N/VL)`），lane `j%VL` | `M * Q_N` |
| 二维，i 向量化 | entry `j*Q_M + i//VL`，lane `i%VL` | `N * Q_M` |

  **现实现限制（B1）**：2D 单元的 lane 只能落在 Buffer 的最后一个逻辑维——fragment 访问目前仍以逻辑行主序参与选维连续性证明与 Verify 值分类，取向一致但 lane 不在末维的程序被拒（R4 v5 方案修复）。

  `Fragment` 字段：`forward_thread` = lane 表达式（`thread_range=[0,VL)`）、`forward_index = [entry 表达式]`、`replicate_size = 1`、输入空间为 Buffer 的**逻辑 shape（不补齐）**——与循环布局的 P 是有意差异（新设计 §4.5）。

- **count 禁止用"元素总数 ÷ VL"计算**。反例：`B[4,130]`、VL=128，元素总数 520，`ceil(520/128)=5`；正确值为 `4 × ceil(130/128) = 8`。行尾不满一个向量时，每个逻辑行仍需完整 entry 空间。
- **一致性规则**：同一 Fragment Buffer 被多个 Parallel 整体访问时必须同一映射；已有条目只检查兼容、不覆盖；不一致在推导处直接诊断。PTO 路径跳过 SIMT 的 `ValidateCandidateAgainstFragments` 再校验（padded 执行空间 vs 逻辑 Buffer 空间是有意不同）。
- **写回**：`layout_inference.cc` SIMD_VF 分支执行 `kLayoutMap` 写回；**实际触发条件是函数级 `layout_map` 非空**（layout_inference.cc:1844-1847），非按区域判断——无 Fragment 布局的函数注解集不变（AscendC/手写 VMI 零影响），但"按区域精确写回"是对实现的简化表述。

### 4.2 `PtoSlotPlan` 规划记录（D2）

新增 TIR Object（`src/ascend/transform/pto_slot_plan.h`）：

```text
PtoSlotPlan {
  fragment:          Buffer      // 输入 TIR 中的 local.fragment Buffer
  backing:           Buffer      // 隐藏 shared.dyn Buffer
  fragment_mem_var:  Var         // 预建 Fragment 存储句柄（Handle）：命名与规划身份；
                                 //   定义由 Vectorize 插入——每区域一次 Bind，用全新同名 Var
                                 //   （vecscope 分裂时各部分各自声明，同一 Var 不会 Bind 两次）
  count:             IntImm      // entry 数（来自布局，见 4.1）
  vector_lanes:      IntImm      // VL（= tl.simdvf_lanes）
  vector_type_name:  String      // 完整 entry 向量类型名 "float32x128"（供核对）
  elem_dtype_name:   String      // 标量元素类型名（== fragment->dtype）
  bytes:             IntImm      // align_up(count * VL * sizeof(elem), 32)
}
```

- 挂载：注解键 `tl.pto_slot_plan`，值 `Array<PtoSlotPlan>`。**实际挂点为发起规划的转换 SIMD_VF 块**（无论 Fragment 分配在何处），引用实际 Buffer/Var 对象，不按名称查找。
- 消费方：AutoSchedule 扣减、`EliminateSimdVFRoundTrip` 剪枝豁免、Verify 一致性、Vectorize 建存储映射。
- 对齐 32 B；entry 紧邻排列；末 entry 已含完整 VL 容量。已知待办：m2 §6.1 要求的 `alignment` 字段未单列——常量在 `kPtoSlotAlignmentBytes`（pto_slot_plan.h）与管线 `_MERGE_UB_ALIGNMENT`（pipeline.py）两处各写一份，靠注释约定同步，待统一为单一来源（当前准入面内预留量必为 32 的倍数，实际风险低）。
- `PlanPtoSlotStorage` 不插入 `fragment_mem_var` 的 Bind——声明在所属 VF 内定义并支配全部访问（新设计 §6.1/§6.3）。

### 4.3 声明 intrinsic 与读写（D5，已实现）

TIR 层（`src/op/builtin.cc` 注册，Opaque）：

```text
fragment_mem = tl.vmi.declare_fragment_buffer(backing_ptr, count, lanes)
    // backing_ptr = tl.access_ptr(BufferLoad(backing,[0,…]), count*VL, rw=3)
    //               必须用地址 pass 能改写的标准地址操作数（新设计 §6.5）
    // count/lanes: 正的编译期整数常量；返回 Handle；声明只绑定地址与存储属性，不访存、不初始化

value = tl.vmi.vload(fragment_mem, element_offset, size=lanes)
    // element_offset = entry_index * lanes；entry 可为常量或循环变量，接口契约要求 [0, count) 内
    // （当前实现：常量 offset 后置检查强制；动态 offset 仅查 entry×lanes 粒度，区间证明待补，登记 A4）
    // 读取完整 entry；vload 读可变内存，不得注册为 pure

tl.vmi.vstore(value, fragment_mem, element_offset, active_mask)
    // 更新 active lane；inactive lane 保留目标 backing 旧值；写效果必须保留
```

- 节点编码：`Bind(fragment_mem_var, Call(Handle, op, [backing_ptr, count, lanes]))`；元素 dtype 从 backing 的有类型地址取得，不从 Handle 推断。
- 两类基址共用 VMI：普通 UB 用现有 `access_ptr`，Fragment 用声明返回的 handle（新设计 §6.3）。
- 声明在所属 VF 内定义、支配使用；不同 VF 不共享句柄；声明不承诺 noalias（PTOAS 按普通 memref 别名语义优化；若实测别名过保守成为性能问题，才考虑专用接口——新设计 §10.2 明示这不是第一版前置条件）。

PTODSL 层（MR274 已支持）：

```python
x_frag_mem = pto.memref_declare(<byte_address>, (4096,), pto.f32)   # shape 为 count*lanes 个标量元素
x = pto.vmi.vload(x_frag_mem, q * 128, size=128)
pto.vmi.vstore(x, x_frag_mem, q * 128, active)
```

codegen 硬约束（新设计 §6.5 末段，均已落实在 `PrintPtoMemrefDeclare_`）：

1. 地址从标准地址操作数提取**最终 UB 字节地址**，输出 signless i64（`shared.dyn` 指针即 `castptr(const(0))`，字节地址 = 元素偏移 × 元素字节数）；
2. dtype 保持原 backing 的**标量元素类型**——从有类型地址操作数取得（`GetAnnotatedPointerDtype`），不被合并 arena 改成字节类型；
3. memref shape 是 `count*lanes` 个元素，vload/vstore offset 是元素单位；
4. **声明已计入 backing 偏移，访存不得重复相加**（entry q 的字节地址 = 声明地址 + q × lanes × sizeof(elem)）。
5. 待补：codegen 目前允许非常量的声明地址（走表达式分支），PTOAS 会将其依赖分析降级为 Unknown——应补常量断言。

**fill/copy（G2，已实现，限定：仅支持起点按 VL 对齐的子区间）**：布局推导前由 `ExpandPtoFragmentFillCopy` 把触碰 fragment 的 `T.fill`/`T.copy` 就地展开为 `MakeSIMTLoop` 等价的 T.Parallel 循环。对齐子区间语义成立：fill 按 region min/extent 迭代；copy 源/目的下标各自映射；部分覆盖由 chunk mask 收敛。**未对齐起点不支持**：展开循环下标从 0 起、存储 lane 为 `(start+i)%VL`，lane 证明失败——`T.fill(f[64:192],0)`、`T.fill(f[5:6],v)`、`T.copy(src_ub[0:128], f[64:192])` 均被拒。**修法（待实施）**：展开时循环基址向下对齐到 VL、原区间作谓词（`start ≤ base+i < end`），存储 lane 即等于执行 lane，G3 将谓词变 masked store——fill 与偏移模 VL 同余的 fragment→fragment copy 可处理；UB↔fragment copy 的 fragment 侧未对齐时 UB 侧 vload 读到区间起点之前（超出"尾部可读"约定），拒绝或要求调用方补齐。重叠自拷贝展开后是数据竞争——**检查定在 `ExpandPtoFragmentFillCopy`**（展开时源/目的区间均已知；前扫描只见展开后的 Parallel 单元，跨逻辑点依赖分析 m2 §5.2 明确不做），重叠即分类拒绝。展开后走与手写 Parallel 完全相同的布局推导 → Plan → Verify → Vectorize 链路，tileop 调用不进调度（fragment 不产生 fill/copy 任务）；Verify 中的拒绝保留为防御性检查（命中即展开遗漏）。

**条件写（G3，由 base + 本文共同实现）**：base Legalize（`bc715beb` 及其前驱）把单侧条件写直接降为**带 predicate 的 BufferStore**（不分 scope；嵌套条件用 And 合取），Vectorize 把 `predicate` 表达式降为 lane mask 后统一合成 `store_mask = mask_and(chunk_mask, pred_mask)`——fragment 与 UB 两条路径共用同一谓词消费代码（vectorize 的 store 分支在 scope 分派之前做 mask 合成，不再有各自独立路径）。表达式 Select（双侧独立值）仍生成 vsel；`And/Or/Not` 由 base 的 Vectorize 降为 mask_and/mask_or/mask_not。本文早期的 G3 Select 模式匹配方案（从 Legalize 物化的 `Select(cond, new, Load(old))` 恢复谓词）随 base 谓词化的落地而删除——两套机制叠加会在同一作用域引入同名变量冲突。

### 4.4 注解键汇总

| 键 | 载体 | 值 | 写入方 / 消费方 |
|---|---|---|---|
| `layout_map`（复用） | SBlock | `Map<Buffer, Layout>` | 布局推导 / Verify、Vectorize、Plan |
| `tl.pto_slot_plan`（新） | 转换 SIMD_VF 块（实际挂点，见 §4.2） | `Array<PtoSlotPlan>` | Plan / AutoSchedule、Verify、Vectorize、RoundtripEliminate 剪枝豁免 |

## 5. 详细设计

### 5.1 布局推导（`parallel.cc` + `layout_inference.cc`）

**目标实现 = R4 v5 子设计**（前扫描 + 奇偶求解 + 注入 + 物化，见《PR4-R4 Fragment 布局联合推导设计方案》§3）；以下五条描述**当前过渡实现**的行为：

`InferPtoSimdLayout` 在构造循环 lane 布局后追加 Fragment 布局推导：

1. **收集**：`PtoAccessCollector` 收集单元内全部访存（含 fragment，仅跳过单元素 local.var）。**限制（B1 根源）**：fragment 访问目前仍以逻辑行主序 `ElemOffset` 参与选维连续性证明（其逻辑 stride 不代表 backing 保存方式）——这是"lane 只能落末维"限制的来源，R4 v5 修复①为让 fragment 退出该证明。
2. **推导**：按 §4.1 表以 Buffer 逻辑下标构造 entry/lane 表达式，组装 `Fragment` 对象。
3. **兼容性**：比较发生在引擎合并点，走 `LayoutNode::IsEqual` 重载——**只比 shape 与 forward_index、忽略 forward_thread/thread_range/replicate**（同 entry 异 lane 的两个布局会被误判相等，PR-6 entry=0/lane=rep 形态的风险）。**目标口径（R4 v5 §3.5）**：求解器注入后合并点只会看到相等的 PTO 提议（替换分支不可达），PTO 专用判据仅在两处生效——注入时与注解 pinned 的核对、物化后的核对；**不改合并点**。
4. **形态检查**：实现按"依赖向量化坐标的维数"计数——零维的访问当前被跳过（R4 v5 判为不可行，m2 §8.6 的单元素提取）、多于一代数即拒；细粒度的规范形式检查由 Verify 两证明兜底。
5. **写回**：执行 `kLayoutMap` 写回（实际触发条件为函数级 layout_map 非空，见 §4.1）。

PTO 路径在 `InferLayout` 早退分支跳过 `ValidateCandidateAgainstFragments`（`in_pto_simd_vf` 守卫）。

### 5.2 `PlanPtoSlotStorage`（新 pass）

- 形式：`PrimFuncPass`（`src/ascend/transform/plan_pto_slot_storage.cc`），`TargetIsPTO` 早退；pipeline 接在 `AscendInsertOOBPadding` 之后、调度分支之前，双路径执行。
- 对每个含待转换 Parallel 的 `SIMD_VF` SBlock：
  1. 读 `kLayoutMap`，筛出区域内有访问的 `local.fragment` 条目；缺失则报"前置布局推导输出不完整"。
  2. 计算 count（来自布局）、`vector_type = elem.with_lanes(VL)`、`bytes = align_up(count*VL*elem_bytes, 32)`（int64 溢出检查）。
  3. 创建 backing：`decl_buffer({bytes/elem_bytes}, elem_dtype, name + "_slot_backing", "shared.dyn")`，append 到**转换 SIMD_VF 块**的 `alloc_buffers`。
  4. 预建 `fragment_mem_var = Var(name + "_fragment_mem", Handle)`、`vector_type_name = "<elem>x<VL>"`；组装 `PtoSlotPlan` 写入 `tl.pto_slot_plan`。Plan 不插 Bind。
  5. **其余不动**：保留原 Fragment Buffer 与标量计算；backing 零读写、无 DMA/fill/copy 任务、无虚假调度依赖。

### 5.3 UB 容量预算（D3）

`ScheduleSingleKernel` 在 `SetMemoryLimit("shared", limit)` 之前，**从 `metadata.kernel_root` 起扫描（含 root 自身注解）**，按 backing data Var 去重汇总 `Σ bytes` 从上限扣减（扣减前断言余额为正）；超限报错（当前为 ICHECK 形态，应改用户错误——见 U5；`T.annotate_unlimit_memory` 跳过。无调度路径在 `MergeUBAllocations` 末尾做带门槛的合并总量断言——门槛与识别均为**声明基**（`CollectAllSlotBackingVars`：plan 注解 ∪ `declare_fragment_buffer` 地址操作数），因为该检查运行于 Vectorize 消费 plan 之后。

**预算的生效范围与边界（U1–U6）**：

1. **预算的约束面（U1）**：`memory_limits_` 仅在 `Z3SchedulePythonLoop` 内使用（auto_schedule.cc:820-826），约束循环内多缓冲的版本选择；直线任务列表没有版本选择，预算没有可约束的决策——**这不是有效性缺陷**，物理容量由 Merge 终检保证。剩余问题：超限要到 Merge 阶段才发现且报 ICHECK；无 contract 时终检跳过（并入 U5/R3 处理）。
2. **对齐口径（U2）**：调度器 `get_buffer_size` 为 `ceil(elements×bits/8)`（:800-817），**不含对齐**；Merge 对每个分配对齐到 `max(32, scope_align)`（shared.dyn 的 scope_align 为 0 故取 32）再总量对齐 32（merge_ub_common.h:195-202、:77-85）——N 个 shared.dyn 分配最多多出 **31N 字节不在预算内**。当前准入面内（lanes∈{64,128,256} × elem_bytes∈{1,2,4}）预留量本身必为 32 的倍数，对齐对 backing 恒为无操作——但该逻辑从未被用例触发。
3. **资源域前提（U3，合入门槛）**：`RewriteTilelangKernels` 遍历整个 PrimFunc、**每个 tilelang_root 独立改写**（kernel_rewriter.h:54）——一个 PrimFunc 可含多个 kernel root；AutoSchedule 对每个 root **各自按 248KB 起算并扣减**，而 Merge 的最终校验按 func.body **累加所有** shared.dyn 与单一 248KB 比较（merge_ub_allocations.cc:441-468）。多 kernel root 时两者必然不一致（预算偏松或校验偏严）。需声明"一个 device function 一个 kernel root"前提，或把校验改为按 root 分组。
4. **诊断形态与指引（U5）**：预留超限用 ICHECK（AutoSchedule 预算函数内，内部错误形态），而基线容量不足用 LOG(FATAL) + 可操作建议（:876-893）——m2 §6.5 要求这是**面向用户的编译错误**；且文案建议 `T.annotate_unlimit_memory("shared")` 会让预留**整体跳过**（:1352-1359，limit 设 1<<40），用户照做后会在更晚、更难理解的位置（Merge 终检，用真实物理上限）再次 ICHECK——该建议应删除或注明仅调试用、不保证设备可运行。
5. **B1 的容量后果（U4）**：`[256,4]` fragment 被 B1 迫使 lane 落 dim1 → count=256、预留 128KB（若 lane 落 dim0 则 8 entry/4KB，**32 倍膨胀**）；128KB 占 248KB 一半，配任何双缓冲即触发预留相关失败——用户看到的是容量不足消息，**根因是布局取向限制**（根因与报错面错配）。R4 v5 落地后两解取小自动缓解；R4 之前的诊断文案宜提示"布局取向可能使 entry 数膨胀"。
6. **SIMT_VF 探测器双份（U6，已降级）**：UB 上限常量之外，`SimtVFDetector` 类在 Merge 侧另写了一份。评审核实：Merge 之后 `SIMD_VF` 块名仍在（`thread_storage_sync` 也按名匹配），"中途改名导致放松 32KB" 的担忧当前不成立——降级为两侧共用一个 helper 的整洁性问题。

**其余已知缺口**：预留量求和/对齐无溢出检查；预留恰等于上限也会被拒；固定分配、无 eligible buffer 的最低需求检查依赖调度器既有行为；最终容量断言位于 Merge 的"无 contract 即返回"之后——无 contract 的 kernel 不执行该断言；UB 上限常量为镜像另抄（未复用 AutoSchedule helper）；合并后逐 backing 的地址区间核验未做。

### 5.4 独占地址分配（D4）

- 自动调度路径：alias contract 由 InsertSync 基于访问寿命产出，backing 无访问即不在 contract 中——与所有对象天然互斥。这是一个**隐含性质**（依赖"不在 contract 即互斥"的分配器语义），代码中无显式注释与专项测试。
- 手动调度路径：`InferBufferAliases` 显式排除 slot backing——该 pass 运行于 Vectorize 之后（plan 已被消费），识别同样走 `CollectAllSlotBackingVars`（声明基，plan 注解作为转换前兜底）。结构测试断言 backing 在 `tl.buffer_alias_map` 中为空别名表。
- 版本链防御：**未实现**专项 ICHECK——当前安全性来自 backing 在调度期零读写、不进入任何任务的访问集，因而不会被版本化 pass 选中；`MaterializeMultiBuffer`/`RewriteAscendBufferVersionLayout` 中无 pto_slot 相关代码。
- `EliminateSimdVFRoundTrip`（m2-on-m1 新增，位于 Verify 之后、Vectorize 之前）的 `PruneAllocBuffers` 豁免 plan 登记的 backing Var（该 pass 按访问剪枝，而 backing 在 Vectorize 前零访问——不豁免会删掉后续访存要引用的分配）。

### 5.5 VerifyParallelToPTO 扩展（D6）

1. **CheckScope**：`local.fragment` 放行条件 = 区域 Parallel 体内 + `layout_map` 条目 + plan 记录在作用域链上；区域级单元素提取、跨 SIMD_VF、plan/布局不一致分类拒绝。fill/copy on fragment 在布局推导前已被展开为 Parallel 单元，Verify 中的同名拒绝保留为防御性检查（**该防御门目前漏 `tl.tileop.ascend_copy`**，登记 A3）。**覆盖边界（A2）**：以上检查仅在转换区域内生效——Verify 顶层遍历对转换区域之外的 fragment 访问（kernel 作用域、无 Parallel 的 SIMD_VF）**没有任何检查**，会掉入通用 lowering 与 backing 读写不一致。**修复归属：R4 v5 §3.3 前扫描**（LayoutInference 内、早于 Plan/Verify 的确定性拒绝，覆盖 kernel 级/区域级/跨区域三类），在 R4 合并提交计划的**步骤 ②** 关闭。
2. **`AnalyzeFragmentAccess`**（`parallel_to_pto_utils`，Verify/Vectorize 共用）：逆映射代入 Buffer 下标 → 布局 `forward_thread`，证明 `== 执行 lane`（全域范围）；代入 `forward_index[0]` 得 entry，证明 lane-free。负例分类：lane 偏移（`B[i,j+1]`）、跨 lane 置换（`B[VL-1-lane]`）、生产消费方向不兼容、`x_frag[7]` 区域级提取、VL 不一致、entry 越界（**仅常量 entry 可查**，动态越界待补区间证明，登记 A4）。
3. **谓词形态**（G3，由 base 实现）：单侧条件写在 Legalize 后以 predicate store 形态到达（Verify 对 predicate 的检查沿用 base 的条件规则）；Vectorize 统一将 predicate 降为 mask 并合成 store_mask（见 §5.6-6）。

### 5.6 VectorizeParallelToPTO：Storage 模块（D7）

已集成进 m2-on-m1 的 `ConvertRegionBody`/carry/vecscope-split 区域结构：

1. **状态模型**：VF 级 `slot_map_`（Buffer → {plan, layout, handle}），按区域在 `ConvertRegionBody` 入口构建（`BuildSlotState`，沿 `block_stack_` 注解链查找）；单元级 `unit_wrote_backing_` 随 `ConvertUnit` 重置。
2. **声明发射**（G1，已实现）：在 `tl.simdvf_scope` 内、首个单元之前，对本区域实际访问的每个 plan 发射一次 `Bind(handle, declare_fragment_buffer(access_ptr(backing 全量, rw=3), count, lanes))`（按首次访问的文档序）；handle 为全新同名 Var，所有顺序单元共用。声明的访问操作数覆盖全部 entry，供地址 pass 链改写为最终 arena 偏移。
3. **读转换**：`AnalyzeFragmentAccess` 得 entry；发射 `vload(handle, entry×lanes, size=VL)`。同一语句两次读生成两次 vload，不做 CSE。**勘误（评审）**：`vload` 在 `DeadBindElim` 白名单内——值位置未被使用的 fragment 读连同其 Bind 会被删除。这对语义无害（读无副作用、backing 保序由写侧 bar 承担），但与本文早先"白名单不含 slot 族"的表述不符；删除仅发生在结果确实无人消费时。
4. **写转换**：值先转向量（标量 RHS 先 `vbrc`）；发射 `vstore(value, handle, entry×lanes, store_mask)`，store_mask 为 chunk mask 或（条件写时）`mask_and(chunk, cond)`。
5. **fill/copy**（G2，已实现，**限定：起点按 VL 对齐的子区间**，§4.3）：不在 Vectorize 内特判——`ExpandPtoFragmentFillCopy` 在布局推导前已将其展开为 T.Parallel 单元（见 §4.3），本模块按普通单元转换，region 语义（min/extent、部分 fill 目标 mask、copy 源/目的独立 offset）由展开形态 + chunk mask 承载；未对齐起点与重叠自拷贝在展开 pass 拒绝（§4.3）。
6. **条件写**（G3，由 base + 本文统一实现）：Legalize 直接产出 predicate store（嵌套条件 And 合取）；Vectorize 在 store 分支的 scope 分派**之前**统一将 predicate 降为 mask 并合成 `store_mask = mask_and(mask_var_, pred_mask)`——fragment 和 UB 两条路径消费同一合成结果，不再有独立的谓词恢复逻辑（本文早期的 Select 模式匹配已删除，其存在会与 base 的 `predicated` 变量产生同名冲突）。
7. **mem_bar**：每个写 backing 的单元后发射 `tl.simd.mem_bar("VST_VLD")`（有意偏差，见 §6/§8）。
8. **收尾**：从 SBlock `alloc_buffers`/`reads`/`writes`/`layout_map`/`tl.pto_slot_plan` 移除已消费的 Fragment 条目，backing 保留；两个后置检查——残留 Fragment 访问拒绝，以及从声明恢复 backing/count/dtype/VL 核对（handle 仅可作 vload/vstore 基址、元素 offset 为 entry×lanes 粒度且在 [0, count×lanes) 内（**仅常量 offset 强制**，动态只查粒度，区间证明随 R4 v5 §3.1）、值 dtype/lanes 与 plan 一致）。
9. **与 base 重写的协同**：vecscope-split 一侧的 `UnitUbCarry` 已限 `shared/shared.dyn` scope（fragment carry 不触发分裂；分裂发生时各部分经入口 2 重新声明各自的同名 handle，同一 Var 不重复 Bind）；**区域级 carry 提升（`TryRegionCarry`）未过滤 scope，且 store 先判 carry 后判 fragment、load 顺序相反——[VL] fragment 串行累加器会写 vreg 读 backing，终至 ICHECK（登记 B2，待修）**；roundtrip 消除的剪枝豁免 backing（见 §5.4），但该 pass 对 fragment 本体的 store→load 消除未关闭（登记 B3）。

### 5.7 地址演进与 PTO codegen（D5）

地址 pass（`LowerAccessPtr`/`FlattenBuffer`/`MergeUBAllocations`）改写**声明的 backing 操作数**，VMI 读写继续引用同一 handle；backing 获得独占偏移 `O`。

- **codegen**（G1 后，已实现）：`PrintPtoMemrefDeclare_` 特判声明，输出 `pto.memref_declare(字节地址, (count*lanes,), dtype)`（四条硬约束见 §4.3）；handle 经通用 Bind 路径打印为 memref 名；读写走既有 vmi 打印输出 `pto.vmi.vload/vstore(handle, offset, size/mask)`。
- **对齐检查旁路**：`VerifyPTOCodegenLegality` 的 store 对齐证明对 declare 句柄基址直接按构造对齐放行（backing 32B 对齐 + 整 entry 访问）；`ClassifyVmiCallResult_` 把声明结果归类为 Handle（仅可作访存基址）。

### 5.8 逐文件改动清单（当前已实现部分）

| 文件 | 改动 |
|---|---|
| `src/op/parallel.cc` | Fragment 布局推导 + PTO 跳过 SIMT 再校验 |
| `src/op/builtin.{h,cc}` | `tl.vmi.declare_fragment_buffer` 注册（Opaque，3 输入） |
| `src/ascend/transform/layout_inference.cc` | SIMD_VF 分支 `kLayoutMap` 写回 |
| `src/ascend/transform/pto_slot_plan.h` | `PtoSlotPlan`（含 `fragment_mem_var`/`vector_type_name`）+ 注解键 + 收集工具 |
| `src/ascend/transform/plan_pto_slot_storage.cc` | Plan pass + FFI + pipeline 接入 |
| `src/ascend/transform/expand_pto_fragment_fill_copy.cc` | G2 展开 pass + FFI + pipeline 接入（layout inference 之前） |
| `src/ascend/transform/auto_schedule.cc` | 容量扣减 |
| `src/ascend/transform/merge_ub_allocations.cc` | 带门槛总量断言 |
| `src/ascend/transform/infer_buffer_aliases.cc` | backing 复用排除（声明基识别） |
| `src/ascend/transform/eliminate_simdvf_roundtrip.cc`（base 新增，本文修补） | 剪枝豁免 plan 登记的 backing |
| `src/op/parallel.{h,cc}` | 多下标收集 opt-in（fragment 限定；逐循环 `InsidePtoSimdVF` 传入，节点成员使 `ExpandBindValues` 同源） |
| `src/ascend/transform/parallel_to_pto_utils.{h,cc}` | `AnalyzeFragmentAccess`（两证明 + 双侧区间 + `outer_loop_ranges` 参数） |
| `src/ascend/transform/vectorize_parallel_to_pto.cc` | Storage 模块（声明/handle 读写/谓词合成/后置检查，整除粒度证明）+ carry/split 的 UB-scope 过滤（区域/分派/shell 三处，B2）+ `EnclosingSerialRanges` + mem_bar |
| `src/ascend/transform/verify_parallel_to_pto.cc` | CheckScope 放行 + `AnalyzeFragmentAccess` 接入（含外层范围绑定）+ fill/copy/ascend_copy 防御性拒绝 |
| `src/ascend/transform/verify_pto_codegen_legality.cc` | declare 句柄基址的 store 对齐旁路 |
| `src/ascend/codegen/codegen_pto.{h,cc}` | `pto.memref_declare` 特判 + 声明结果 Handle 分类 |
| `testing/ascend/language/pto/` | 结构测试 + 设备测试 + 评审修复回归 |
| ~~R4 v5 求解器（子设计 §3.8）~~ | **已撤出**（2026-10-22 评审：入口短路恒不运行 + 边编码矛盾 + 前扫描契约未实现）。收集/证明半部已随 `parallel.{h,cc}`/utils/verify/vectorize 落地（上表）；求解器按子设计重写后在 `layout_inference.cc` 的 `Collect()`/`Run()` 之间单独挂载（挂点注释已留在代码中） |

## 6. 关键设计决策与理由

| 决策 | 理由 |
|---|---|
| 读写复用普通 vload/vstore + handle 基址（不新增 slot_load/store） | 新设计 §6.2/§10.2：memref 已携带全部语义（count 是 shape、entry 类型是 vreg 类型、index 是 offset）；PTOAS 的 load/store 已接受 memref，mem-to-value 直接处理；专用句柄方案成本高收益低。声明 + handle 使对象身份与地址关联在 IR 中显式化 |
| backing 选 `shared.dyn` 且全程独占 | 只有 shared.dyn 进入 MergeUB 统一合并与容量核算；TileLang 无法用访问分析证明复用安全，独占最多浪费容量、不会地址冲突；消除访存后地址不回收 |
| 容量在调度前按最坏情况预留 | AutoSchedule 按剩余容量决定普通 Buffer 双缓冲；不得假定 PTOAS 会将 Fragment 值提升到寄存器 |
| entry q 放在 VMI offset（`entry×lanes`），不为每个 q 创建新 memref | 新设计 §6.2；声明一次/VF，动态寻址由 offset 表达 |
| 声明已计入 backing 偏移，访存不再叠加 | 新设计 §6.5；防双重偏移 |
| count 来自布局 entry 数，禁用 元素总数/VL | 行尾不满整向量时两者不等（`B[4,130]` → 8 不是 5）；每个逻辑行需完整 entry 空间 |
| Fragment vstore 用 active mask，inactive lane 保留旧值 | 五类 mask 有意区分（新设计 §9.1）；"条件成立才更新"由 masked store 承担 |
| 条件写回 G3：base Legalize 谓词化 + Vectorize 统一 mask_and 合成 | 新设计 §5.3/§9.4 明确"不补读目标旧值构造普通 Select""无需额外 vsel"。base `bc715beb` 已实现谓词化 Legalize（含嵌套 And/Or 合取），本文 fragment 分支消费同一 predicate——不再需要独立的 Select 模式匹配恢复 |
| fill/copy 布局推导前展开为 T.Parallel 复用既有单元转换，不在 Vectorize 内做专用 lowering（**仅起点按 VL 对齐**） | 对齐子区间的 region 语义由循环形态天然承载（min/extent 迭代、源/目的独立映射、chunk mask 收敛部分覆盖）；fill/copy 调用不进调度；展开后与手写 Parallel 走完全相同的推导/准入/发射链。未对齐起点与重叠自拷贝的检查定在 `ExpandPtoFragmentFillCopy`（展开时源/目的区间均已知；前扫描只见展开后的 Parallel 单元、无法做跨逻辑点依赖分析，m2 §5.2 明确不做） |
| 每区域一次全新同名 handle Var（plan 预建 Var 只作命名/身份） | vecscope 分裂使一个 plan 对应多个区域，各部分必须各自声明；同一 Var Bind 两次会触犯 SSA 单定义（codegen 有重复定义检查） |
| vload 不做 CSE（DCE 白名单含 vload：无人消费的读会被删除，语义无害——见 §5.6-3 勘误） | 内存语义；合并/消除是 PTOAS 职权；两次读保留两次 vload 是验收要求（前提是两次读的结果均被使用） |
| （保序口径见 §6 决策表——当前无 TileLang 侧 bar，P1-1 决策可能恢复部分形态） | **覆盖矩阵（登记 A1）**：该 bar 只保护跨单元 RAW；单元内 store→load 同 entry、跨单元 WAR（需 VLD_VST）无 TileLang 侧保护；PR307 自动插 bar 对 Unknown 依赖只记录不插、且 mem-to-value 仅转发常量 offset（动态 entry 的 bar 不会被转发删除）——故保守 bar 保留为默认，移除决策待 Unknown 形态用例补验 |
| 声明的 Bind 在 `tl.simdvf_scope` 之内 | VPTO 将 vecscope 数据视为 region-local，跨界引用被拒 |
| `declare_fragment_buffer` 的 backing 操作数用 `tl.access_ptr` 表达而非旁表名字 | 地址 pass 只能改写 IR 中的地址表达 |
| carry/vecscope-split 检测限 UB scope（split 侧、region carry 侧、单元 shell 侧三处均已排除 fragment，B2 关闭） | fragment carry 走分裂/晋升改变保序结构；现由 slot 路径 + mem_bar 保序 |

## 7. 测试与验收计划

> G1–G3 换形已在合并工具链（PTOAS `pr274-pr307-merge` = MR274 memref/mem-to-value + PR307 自动 mem_bar，ptoas 0.68，238 全栈自建）完成设备复验；历史双工具链数字见 §7.3 末。

### 7.1 结构测试（`test_pto_slot_plan_structure.py`）

覆盖：Plan 记账（backing/plan 字段/count/bytes/handle Var/vector_type、ragged ceil）；**R4 落地后并入其 §4 全部 19 项用例**（取向一致非末维、1D+2D、out-of-place +VL 多下标、区间正负例、SimdVF 注解拒绝、同 data/跨区域诊断、并列规则、SIMT 多下标回归、沿 dim0 设备用例等）、Vectorize 发射（每区域一次 declare、entry×lanes 偏移、chunk mask、两次读不合并、mem_bar 计数与种类、alloc/注解清理、handle 无残留用法）、条件 kill 谓词合成（mask_and 进 store mask、无旧值读/无 vsel）、fill/copy 展开正例（tileop 调用不残留、走 handle 读写、mem_bar 计数；消费者为 RMW 形态——纯 store→单载形态会被 base 往返消除合法前向；该消除对 fragment 同样生效，登记为偏差 B3）、负例（区域级提取、lane 偏移、跨 VF）、容量超限诊断。

**结果：144 与 238 均 13/13 通过**（含 kernel 作用域分配预算、手动调度独占性两用例）；144 另过四个源级套件共 46/46（cast_matrix 与 masked_store_alignment 因需 torch_npu/CANN 导入未在 144 运行，由 238 全量回归覆盖）。

### 7.2 设备验收（`test_vectorize_slot_parallel_run.py`）

**238 合并栈 13/13 通过**（旧基线 m2-on-m1@5077f31d 上的结果）：sv3 keep（64/128/256）、条件 kill（cf1 形态，谓词合成）、ragged 1D（64/128/256）、fill_then_mask（64/128）、copy_out_of_fragment（64/128/256）。源码断言含 `pto.memref_declare`/vecscope/mem_bar。

### 7.3 回归与工具链

- **238 全量 PTO 目录回归（合并栈，rebase 后）：506 通过 / 4 失败**。4 个失败全部为工具链侧联合事项（J3/J4），与本 PR 改动无关：
  - 3× `test_ordinary_broadcast_keeps_conservative_extent[b8/b16/b32]`：`dist_mode` 由 PTOAS 主线提交 `2729bb556` 删除。**J3 为合入前置**（不修则 PTOAS 主线上线后，TileLang 中使用 dist_mode 发射路径的 kernel 与相关测试即失败——非整个构建）：修法为 TileLang 改发射（两次连续 vload + vdintlv）；slot 路径不受影响。
  - 1× `test_scalar_float_cast_rejected_by_device_compiler`：负例期望的 VPTO 报错文案未出现——MR274 系 ptoas 对该源返回 0 但不产出 fatobj（静默无产物），报错面错配；该用例此前从未进入历史回归子集，非本次回归。
- **gym 数值矩阵（合并栈，rebase 后）：9/9 全部 PASS**——sv1、sv3v×2、cf1v–cf6v；原 J1（base vecscope 分裂地址断裂）已被 base 修复（split 分配提升 + carry 决策对齐，现在新基线中的 `d1a2e96f` + `1e2cdfa6`），J1 关闭。**另 sv2v 编译+数值通过**（base 的 per-lane alloc_var 提升（新基线 `4c70bec7`） 解锁）——simd-pto 验收面 15 家族中 **10 个数值通过**；剩余 5 个拒绝均有明确归属：sv5v/sv8v（local.var 数组累加器，PR-2 路线）、sv9v（PR-3 直接规约）、sv4v（PR-5 gather）、sv6v/sv7v（J2/J4 工具链）。
- **sv9v 重估**：G2 移除了 fragment-copy 屏障；三个 arm 现统一拒绝于 `T.reduce_max/min` 直接规约——干净的 PR-3 依赖，符合路线图。
- **G4 实验**：临时移除 TileLang 侧 slot mem_bar 后 slot 套件仍 **13/13 通过**。**证据面限定（A1）**：该实验覆盖的是仿射地址、常量界、两层以内循环形态的跨单元 RAW；PR307 对 Unknown 依赖只记录不插 bar，单元内 RAW 与跨单元 WAR 未验证——"自动插 bar 完全覆盖"的结论以此为准，保守基线已恢复并复验（13/13）。
- 历史基线（换形前 backing 直址形态）：144 结构 7 套件 69/70；238 设备全量 143/143（文件子集）；vmi-0.1.9 与 MR274 双栈全绿。G1 后工具链下限为 MR274+（memref_declare）。

### 7.4 兼容性

AscendC 注解集不变；PTO SIMT、手写 VMI 区域照旧跳过；存量 kernel 的 MergeUB/InferBufferAliases 行为不变（改动以 plan/声明为门槛）。**一处例外**：G2 展开只作用于 PTO 目标下触碰 fragment 的 fill/copy 调用。（原"G3 谓词合成作用于所有连续写"的例外已随 base 谓词化而消解——masked store 现在是 base 自身的行为，不再是本文引入的偏差。）

## 8. 已知限制、偏差与开放问题

### 8.1 问题登记表

状态取值：✅ 已关闭 / 🔄 由 R4 子设计关闭（标注步骤）/ ⚠ 待修复 / 📌 有意偏差（明示保留）。

| # | 类别 | 事项 | 状态 / 归属 |
|---|---|---|---|
| ~~J1~~ | base bug | m2-on-m1 vecscope 分裂的 shared-carry 地址断裂 | ✅ base d1a2e96f + 1e2cdfa6 修复关闭 |
| J2 | 工具链（**十五轮定案：版本要求，非缺陷**） | ~~gym sv6v/sv7v 的 `fmaximum` 标量 max 后端不可选~~——标量 `pto.maximum` 降级为 `fmaximum` ISD 节点，**CANN 9.1.0/9.2.0 GA 的 bisheng 无选择模式，weekly-0909 已支持**（矩阵实证）；向量路径 vcmp+vsel 各版本均正常。最新 CANN 下 sv6v 完整编译通过 | ✅ 按"最低 CANN 编译器版本 = 含 fmaximum 选择的版本（weekly-0909 系）"声明即关闭；GA 环境需混用新编译器或等 GA 化 | ⚠ 建议 hotfix 并记入路线图 |
| J3 | 工具链 | PTOAS !293 删除了 vload/vstore 的 `dist_mode` kwarg | ✅ 已关闭：组合发射（brc 含 group、dintlv、intlv 写、continuous 静默、无 mask 合成） |
| J4 | 工具链 | ~~vecscope 内标量指令 fatobj 静默缺失~~ **定案：环境侧 wrapper bug**（scratch wrapper 丢退出码；ptoas 本身正确） | ✅ 已关闭（wrapper 修复，负例原样通过） | ⚠ ptoas 侧：硬拒绝或正确产出；TileLang 侧负例断言待报错面稳定后更新 |
| B1 | 布局限制 | 2D fragment 的 lane 只能落最后一个逻辑维（选维证明与 Verify 值分类均按逻辑行主序判连续），取向一致但 lane 不在末维的程序被拒；高瘦 fragment（[256,4]）编译过则 32× entry 膨胀（U4 的根因） | ✅ 已关闭（十一轮：求解器重写合入，fragment 退出选维证明 + 奇偶联合求解；旗舰用例结构+设备通过） |
| A2 | Verify 缺口 | 转换区域之外的 fragment 访问无拒绝路径（kernel 作用域、无 Parallel 的 SIMD_VF）——顶层 Verify 无任何访问检查 | ✅ 已关闭（前扫描对区域外/区域级 fragment 访问统一拒绝，负例入套件） |
| A4 | Verify 缺口 | 动态 entry 只查粒度不查范围——越界写到 backing 外相邻 UB 对象且不报错 | ✅ 已关闭（双侧区间 + 外层范围绑定：区域级 serial wrapper / 单元内 shell / split 提升循环；**窄范围**：不绑 region 之外的 kernel 级循环，依赖其的访问两侧一致拒绝——扩展为后续小项） |
| B4 | 崩溃 | fragment 的同单元不同下标在收集阶段 ICHECK 崩溃（`f[i]=f[i]+f[i+VL]`、自拷贝 `f[i+128]=f[i]`——原地写法仅 extent≤VL 是合法并行程序，测试一律 out-of-place）。**终审更正**：登记时的 UB 形态 `B[j]+B[j+VL]` 从未受影响——收集 visitor 只记录 fragment，UB 不进 ICHECK 路径，实测完整跑通 | ✅ 已关闭（多下标收集，**限 local.fragment**——UB 本就不需收集；非 fragment 保持严格 ICHECK） |
| B5 | 崩溃 | floating fragment 被 step 0 种成 SIMT FullyReplicated 后与 PTO 提议在合并点冲突 → layout_inference.cc:482 裸 FATAL | ✅ 已关闭（求解器为 floating fragment 前置产出字节最小布局，step 0 跳过已布局缓冲；正例入套件） |
| A6 | Plan 缺口 | Plan 只查 forward_index 维数即用 OutputShape()[0] 算 count，不校验规范形态；注解/floating 布局进入会算错预留量 | ✅ 已关闭（replicate==1 + 常量正 thread extent 断言；报错仍为 ICHECK 形态，分类化登记为后续小项） |
| U1 | 预算时点 | 预算只在 kernel 含调度循环时才约束多缓冲版本选择（memory_limits_ 仅在 `Z3SchedulePythonLoop` 内使用）；直线任务列表没有版本选择决策，预算本来就没有可约束的对象——**物理容量始终由 Merge 终检保证**（base :921 注释同此立场）。剩余问题是 Merge 阶段才发现超限且报 ICHECK、以及无 contract 时终检跳过 | ⚠ 并入 U5/R3：改用户错误形态、保无 contract 路径终检（**原 P1 升门槛已降为 P2**——直线 kernel 无决策可约束，非有效性缺陷） |
| U3 | 预算口径 | 多 kernel root 时预算与终检口径不同 | ✅ 单 root 断言 + 负例用例；按 root 分组校验维持登记 |
| U2 | 预算口径 | 预算不含对齐 padding、最终布局含（差额上界 31N/分配；当前准入面内对 backing 恒为无操作但未触发过用例） | ⚠ 随 U1 一并处理 |
| U4 | 诊断错配 | B1→entry 32× 膨胀→预留失败，报错面（容量不足）与根因（布局取向）错配 | ✅ 已关闭（求解器两解取小，膨胀布局不再产生） |
| U5 | 诊断形态 | 预留超限报文 | ✅ slot 预算报文已改分类形态（去 unlimit 建议）；Z3 通用报文保留 base 建议（波及非 PTO 用户属回归，已还原）；LOG(FATAL) 形态归登记 |
| U6 | 校验口径 | HasSimtVF 探测器双份（Merge 后块名仍在，按名匹配成立——降级） | 🔧 整洁性：两侧共用 helper |
| A1 | 同步（十二轮决策） | **TileLang 侧保守 mem_bar 已全部移除**（单元级/WAR/shell 迭代）：backing 访问保序整体移交 PR307 自动 vecscope bar；工具链下限 = `pr274-pr307-merge` lineage（PR307 对建模依赖自动插 bar；Unknown 形态的覆盖请求已列入 PTOAS 问题清单）；shell 累加器无 bar 设备数值为委派的直接证据 | ✅ 责任移交（G4 按 m2 §9.5 的原始划界落地）；PR307 Unknown 形态与 mem-to-value 动态 offset 转发为 PTOAS 侧请求 |
| P-声明盲区 | Vectorize | 只出现在 store predicate 里的 fragment 读不会被声明收集 | ✅ 已关闭（声明收集与残留检查显式遍历 store predicate） |
| 报文误导 | Verify | 跨 SIMD_VF 共享 fragment 时的报文把用户错误说成 pass 顺序问题 | ✅ 已关闭（前扫描以 ownership 报文拒绝 + Verify 报文如实化） |
| A3 | 偏差 + 缺口 | G2 前置展开在 ReducerPlanAndMaterialize 之前，materializer 的 fill/finalize/copy 不会被展开（口径：Storage/Reduce 直接消费，见 §2.1）；[1] 形状的逻辑 fill 只写 lane0（当前准入面内无害，PR-6 广播初始化需另行处理） | ✅ ascend_copy 补门已关闭；📌 materializer 偏差与 [1] fill 归 PR-6 |
| A5 | 偏差 + 缺口 | ~~G3 谓词恢复作用于所有 store~~ **已随 base 谓词化消解**：base Legalize 自身对 UB 做谓词化 masked store（不再有“恢复 vsel”的对象），嵌套 And/Or 由 base 降为 mask 合取——本文的 fragment 分支消费同一机制，无独立范围可言 | ✅ 已关闭（base bc715beb + 本文 fragment 接入） |
| B2 | 崩溃 | [VL] fragment 串行累加器：`TryRegionCarry` 不查 scope + store/load 分派顺序相反 → 写 vreg 读 backing，终至 ICHECK | ⚠ carry 分派统一 scope 检查；支持面如实为"2D fragment 累加器"；修复时可评估 fragment 串行 RMW 改走 vecscope 分裂（base 分配提升后不再多地址，vecscope 切换是完整同步点） |
| B3 | 偏差 | base 的 roundtrip 消除对 VF 内 1D fragment 同样消除 store→load（偏离 m2 §5.5.4/§10.2 访存消除归 PTOAS），被消除 fragment 的 backing 仍被预留且不再声明 | ⚠ **决策已定待实施**：对 local.fragment 关闭该消除（关闭后保留 UB 往返 + mem_bar，语义正确） |
| C5 | 提交结构 | 已重构：builtin 前置、Verify 放开与转换同提交、FFI 注册与调用同提交、逐提交可编译（三个点名中间提交的运行验证待跑） | ✅（运行级验证待跑） |
| Q4 | 调度交互 | SubcoreProjector 不认识 slot plan（投影 kernel 复制 fragment 后 plan 仍指向原 buffer） | ✅ 已关闭（分类拒绝；投影改写 plan 等真实需求出现再做）；带 `get_subblock_idx` 的构造用例未写（API 复杂度中等，登记） |

### 8.2 测试与验收缺口

**容量与独占为预留设计的核心验收面（第一优先）——已补齐（十一轮，`test_pto_slot_capacity.py`）**：①ping/pong 多轮 backing 单份（最终模块恰一次声明）；②两条调度路径下从最终合并模块的声明地址（access_ptr 的 offset/extent）直接断言逐 backing 字节区间两两不相交；③非零 backing 基址（普通 UB 缓冲在前时偏移 > 0）；④合并总量不超物理上限。双缓冲退化场景由预算扣减机制承载（超限负例）+ 容量断言。

其余缺口：哨兵检查、非 fp32 dtype；非对齐子区间 fill/copy（§4.3）、UB→fragment 反向 copy、2D region；访存保留/寄存器驻留结构对照、优化开/关数值对照；[VL] 累加器 + 串行循环（B2 修复后）；R5/R6/R8 的"设备验证"限定为已覆盖用例。

**既有测试的断言缺陷（已随 2026-10-22 重构修复）**：S:341 改为整参数匹配；S:500 注释与断言对齐（两个写 backing 单元两个 bar）；R:62-67 参数清理、docstring 如实化；S:485 残留 tileop 过滤补 `ascend_copy`；S:671 docstring 改为如实描述（kernel 作用域分配仍经区域 plan 进入预算）。S:449 的双分支文案保留（前端包装使两种拒绝路径均可达，注释已说明）。~~R4 落地时的负例报文同步~~（求解器撤出，报错点不迁移）。

### 8.3 验收口径

M4 新解锁家族为 6 个（sv2、sv3、cf1、cf3、cf4、cf6；sv1 属 M1、cf2/cf5 属 M2）；sv5/sv8 当前归"PR-2 路线"与路线图 F-FRAG 归属矛盾、sv6 归 J2、sv9 归 PR-3——需与路线图 owner 同步归属或补齐能力；本文验证使用 gym 的 *v twin，其与路线图 §11 ST 套件的对应关系待说明。

## 9. 实施计划与当前状态

**提交序列（2026-10-23，HEAD `21a4d976`，18 提交，base `bc715beb`；每个提交独立可编译，Verify 放开与转换实现同提交，pipeline 注册与 FFI/C++ 实现同提交，路由断言随注册同提交翻转）**：

| # | 提交 | 内容 |
|---|---|---|
| 1 | `db3f8b45` | builtin declare_fragment_buffer 前置 |
| 2 | `6646a426` | 布局推导 + PlanPtoSlotStorage + Plan 的 pipeline/路由 |
| 3 | `1e0e59c2` | 预算扣减 + Merge 容量断言 + 别名独占 + 往返消除保 backing |
| 4 | `c8a08830` | 展开 pass + 重叠自拷贝分类拒绝 + `AnalyzeFragmentAccess` 两证明 |
| 5 | `f5749a62` | Vectorize Storage 模块（整除粒度证明）+ Verify 放开（同提交）+ codegen memref_declare + carry 全链排除 fragment（B2） |
| 6 | `09fb685e` | 结构/设备套件（含断言缺陷修复） |
| 7 | `931ada08` | 多下标收集（fragment 限定）+ 双侧区间证明 + 外层范围绑定 |
| 8 | `b6fcb599` | per-buffer 往返门控 + ascend_copy 补门 + A6 规范断言 |
| 9 | `539ff03d` | 评审回归用例 |
| 10 | `bdb70e3c` | 联合布局求解器（R4 重写） |
| 11 | `b685a8ca` | kernel 级 entry 范围 + 跨单元 WAR（后被 `fb42b4b0` 随 bar 移除一并删除——历史保留 add+delete，P1-1 决策可能再引入） |
| 12 | `4e649c5b` | 组合 dist_mode 发射 + 分类诊断 + 后置检查加固（提交说明已按实际 diff 校正） |
| 13 | `aeb9a140` | 未对齐展开（fragment 侧门控）+ 投影门 |
| 14 | `9eba46e7` | 求解器/容量/新形态用例 |
| 15 | `581eb9c3`→`f151c416` | PTODSL (source, offset) ABI 修正 |
| 16 | `fb42b4b0`→`7f135dcb` | 保序移交工具链 + 确定性用例 + 单 root 前提 |
| 17 | `fa146932`→`3b6f4a9e` | 注解 pinned 参与 + 退化维用例 |
| 18 | `21a4d976` | 十五轮评审修复（DSU 确定性、group 组合、G2 fragment 侧门控 + Cast、unroll 绑定、死缓冲容忍、显式字段比较、no-contract 终检、死代码清理、7 个新结构用例 + 设备用例） |

**验证记录（2026-10-23，238，PTOAS master `d302630695` + MR274，干净构建）**：
- 结构 + 容量：**48/48**（含本轮新增：确定性合并触发、unroll 包裹、死 rank-3 容忍、UB 多下标正例、U3 双 kernel 负例、G2 UB 侧放行、注解 pinned 翻转）；
- 全 pto 目录（设备含）：上一轮 628/628；本轮修复后的全量复验在跑（结果落地后更新此行）；
- 三个点名中间提交（`4e649c5b`/`aeb9a140`/`bdb70e3c`）的编译 + 结构套件验证待跑。

**需要撤回的历史声明**（原文已随本轮重写删除，此处留痕）："保序整体移交 PR307"（P1-1 风险待决策）、"SimdVF 级注解已测 fail-closed"（实为冲突路径，专属拒绝不可达）、"功能缺口已清零"（本轮又修了 DSU 确定性等）、"U5 已分类化"（slot 报文是、Z3 已还原）、"容量四用例已补齐"（ping/pong 已按真多轮语义修正）。

**合入前待办**：①P1-1 保序决策（区域门控 / 全量恢复 / 拒绝）+ 对应设备用例；②跨 vecscope 保序的设备 A/B + PTOAS 答复；③中间提交验证；④全量复验数字回填。
