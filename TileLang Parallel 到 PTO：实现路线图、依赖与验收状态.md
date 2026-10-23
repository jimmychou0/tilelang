# TileLang Parallel 到 PTO：实现路线图、依赖与验收状态

> 本文是《TileLang Parallel 到 PTO：目标设计方案》的配套**实现路线图**：记录 PR 拆分、依赖、阻塞和验证进度，用于开发执行和项目跟踪。语义、接口和职责边界以目标设计方案为准，本文只引用其章节，不重复定义 slot、mask 或 reducer 语义。
> 
> 编号约定：PR\-1～PR\-6 是上游（tile\-ai 主线）路线图路标，除已合入的 tile\-ai\#3308/\#3310 外均尚未发起；PR \#262、\#269 是 fork（WenboCodes/tilelang\-deepseek）侧的原始特性实现参照，两套编号不混用。
> 
> 

## 1\. Roadmap 总览

### 1\.1 背景与目标

**现状**：

- `target="ascend"` 后端已合入 tile\-ai 主线（[tile\-ai/tilelang\#3308](https://github.com/tile-ai/tilelang/pull/3308)），PTO driver/codegen 已合入主线（[tile\-ai/tilelang\#3310](https://github.com/tile-ai/tilelang/pull/3310)）。

- VMI 语言面（`tl.vmi.*`、VMI codegen）还没有合入主线，只在 fork 的 asc 分支（[deepseek\-ai/tilelang\#250](https://github.com/deepseek-ai/tilelang/pull/250)）。

- `T.Parallel` 对接 PTO 的完整实现目前只在 fork 侧：[PR \#269](https://github.com/WenboCodes/tilelang-deepseek/pull/269)（T\.Parallel→PTO VMI 自动向量化）、[PR \#262](https://github.com/WenboCodes/tilelang-deepseek/pull/262)（控制流 \+ reduce\_max/min/sum），均为 OPEN。

- ST 验收套件已移植到 [PTO\-ISA/tilelang](https://github.com/PTO-ISA/tilelang) 的 pto\-dev 分支：[PR \#2](https://github.com/PTO-ISA/tilelang/pull/2)，`examples/ascend/t_parallel_st_suite/`（21 个 SimtVF `T.Parallel` kernel \+ 21 个手写 PTODSL 对照版）。

**目标**：用户在 `T.SimdVF(lanes=L)` 内写标量并行的 `T.Parallel` 程序（element\-wise、控制流、reduce），编译器自动向量化为 PTO VMI，经 PTOAS 编译在 Ascend 950 上正确运行。全部能力分若干 PR 合入 tile\-ai 主线，每个 PR 的发起是一个路标。

### 1\.2 分工原则

||TileLang 侧|PTOAS / 工具链侧|
|---|---|---|
|职责|逻辑 lane 布局（E/P/Q/remaining、tail mask）、pass pipeline（Verify→Legalize→Vectorize）、VMI 指令生成、Fragment→slot 逻辑映射、UB 容量预算|物理寄存器拆分、mask 物理化、slot 物理实现（vreg 驻留 / slot 访存消除或保留对已绑定 backing 的读写决策）、依赖分析与同步插入（mem\_bar）、CSE/LICM 等优化|

职责边界的规范性定义见目标设计 §10（TileLang / PTOAS 职责边界），本文不重复展开。

### 1\.3 上游 PR 发起时间线

```text
gantt
    dateFormat YYYY-MM-DD
    axisFormat %m-%d
    section PR 发起节奏
    M1 PR-1 T.Parallel 核心向量化 含 VMI 语言面 :milestone, 2026-10-12, 0d
    M2 PR-2 控制流 CF to Select :milestone, 2026-10-19, 0d
    M3 PR-3 direct Reduce 与 TopK :milestone, 2026-10-26, 0d
    M4 PR-4 Fragment 与 slot 存储 :milestone, 2026-11-02, 0d
    M5 PR-5 gather scatter 与打包类型扩展 :milestone, 2026-11-09, 0d
    M6 PR-6 reducer v2 与 RMSNorm 验收 :milestone, 2026-11-16, 0d
```

> - 里程碑均为 PR 发起日期。发起节奏可能快于社区 review 与合入速度：视 review 进展，后续 PR 采用 stacked PR 或等前序 PR 合入后再发起。
> 
> - M3 的主线 numeric 验收在 PTOAS P1（mem\_bar，见 §9\.1）合入后闭环；**P1 需要修复后合入 PTOAS 主线**。
> 
> 

### 1\.4 里程碑表

|路标|上游 PR|内容|特性|依赖|预计发起|新增可打通 ST 用例（累计）|状态|
|---|---|---|---|---|---|---|---|
|M1|PR\-1|**T\.Parallel 核心向量化（含 VMI 语言面）**：`tl.vmi.*` builtin \+ VMI codegen、LayoutInference PTO 分支、Verify/Legalize/Vectorize pipeline、1D/2D 映射、E/P/Q tail mask、dtype/cast（含 FP8 转换矩阵）、element\-wise|F\-EW|无|2026\-10\-12|sv1（1/21）|待拆分代码验证，准备分支|
|M2|PR\-2|控制流：IfThenElse/Min/Max→Select、布尔表达式展开、vcmp/vsel、mask\_and/or/not|F\-CF|M1|2026\-10\-19|sv7、cf2、cf5、sp4（5/21）|待拆分代码验证，准备分支|
|M3|PR\-3|direct `tl.tileop.reduce`（max/min/sum，单元素结果以 `1PT` 写回）、TopK example（index 全链路 int32）|F\-R1|M2 \+ PTOAS P1|2026\-10\-26|本路标不新增（sv9 还依赖 M4 的 F\-FRAG）|实现中，numeric 闭环仍受 P1 阻塞|
|M4|PR\-4|Fragment/slot 存储：PlanPtoSlotStorage、layout\_map contract、UB 容量预算、MergeUBAllocations 独占分配、slot\_alloc/load/store codegen|F\-FRAG|M1 \+ PTOAS P2|2026\-11\-02|sv2、sv3、sv5、sv6、sv8、sv9、cf1、cf3、cf4、cf6（15/21，其中 14 个用例的实现路径已明确，cf6 已纳入但实现路线待定，见 §12\.4）|Todo|
|M5|PR\-5|间接访存与打包类型扩展：单侧 store predicate、uniform scatter 及通用 gather/scatter 向量化（F\-GS）、ue8m0/fp4 位操作与 reinterpret（F\-PACK）、stride\-2/interleave 布局|F\-GS / F\-PACK|M4 \+ PTOAS P3|2026\-11\-09|sv4、sp1、sp2、sp3、sp5、sp6（21/21）|store 语义已定；通用 gather/pack 待设计|
|M6|PR\-6|reducer v2：PTO SIMD LayoutInference、ReducerPlanAndMaterialize 分支、finalize/seed/inactive\-tail、RMSNorm numeric|F\-R2|M3 \+ M4；RMSNorm uniform 写回另依赖 M5|2026\-11\-16|ST 套件无直接消费者，由 RMSNorm 产品用例验收（21/21）|Todo|

> **PR\-1 拆分方式（待定）**：当前倾向把 VMI 语言面并入 PR\-1 一次合入；备选是 VMI 语言面单独先行一个 PR（review 工作量更小）。视社区反馈决定。详见 §12\.4 待定事项。
> 
> **CF6 路线待定**：cf6 的 Parallel 体内含常数界串行循环（Newton 迭代），已纳入本期验收范围并计入 M4 的目标覆盖（15/21）；打通依赖三条候选路线之一（见 §12\.4），待讨论确定。
> 
> 

## 2\. 设计章节到实现任务的映射

目标设计文档的章节编号按合并计划 §3\.1 的 12 章骨架引用：§1 背景、目标与术语；§2 输入、输出与贯穿示例；§3 目标 pass pipeline 与阶段契约；§4 循环布局、Buffer 布局和 E/P/Q/remaining；§5 Verify、Legalize、Vectorize 的职责；§6 Fragment Buffer 与 slot 存储；§7 direct reduce 语义；§8 reducer v2、finalize、seed 与 inactive\-tail；§9 mask、条件更新、访存和同步边界；§10 TileLang / PTOAS 职责边界；§11 支持边界与正确性条件；§12 目标输出示例。

路线图只引用设计章节，不复制完整技术内容：

|目标设计章节|路线图任务|
|---|---|
|§3 目标 pass pipeline 与阶段契约|T0、PR\-1/PR\-2 pipeline 接入、PR\-4 PlanPtoSlotStorage、PR\-6 materializer 分支|
|§4 循环布局、Buffer 布局和 E/P/Q/remaining|PR\-1 基础布局与 tail；PR\-4 layout\_map contract；PR\-6 reducer layout|
|§5 Verify、Legalize、Vectorize 的职责|PR\-1/PR\-2 基础路径；PR\-3 direct Reduce；PR\-4 Storage；PR\-5 store predicate/scatter；PR\-6 Reduce v2|
|§6 Fragment Buffer 与 slot 存储|T4 / PR\-4 / PTOAS P2|
|§7 direct reduce 语义|T1/T2/T3 / PR\-3 / PTOAS P1|
|§8 reducer v2、finalize、seed 与 inactive\-tail|T6/T7 / PR\-6（依赖 PR\-3、PR\-4；RMSNorm uniform 写回依赖 PR\-5）|
|§9 mask、条件更新、访存和同步边界|PR\-2～PR\-6；PTOAS P1/P2/P3|
|§10 TileLang / PTOAS 职责边界|本文 §1\.2 分工原则、§9 PTOAS P1/P2/P3 依赖|
|§11 支持边界与正确性条件|本文 §11 验收矩阵（ST 21 kernels、TopK、RMSNorm、回归）|

> 注：合并计划 §6 原映射表最后一行写作「§10 验收」，与计划 §3\.1 的 12 章编号（§10 为「TileLang / PTOAS 职责边界」，无验收章节）不一致；本表已按 §3\.1 骨架修正为 §11（支持边界与正确性条件），并补 §10 一行。
> 
> 

### 2\.1 Feature—依赖—验收表

|Feature|TileLang 路标|PTOAS 依赖|主要验收|
|---|---|---|---|
|F\-EW|PR\-1|无|vecadd、连续 load/store、dtype、tail|
|F\-CF|PR\-2|无|Select、mask 逻辑、条件更新|
|F\-R1|PR\-3|P1|direct reduce、`1PT` 单元素写回、TopK|
|F\-FRAG|PR\-4|P2|slot、UB budget、跨循环保存|
|F\-R2|PR\-6|—（本体依赖 PR\-3、PR\-4；RMSNorm uniform 写回还依赖 PR\-5/P3）|finalize、seed、inactive\-tail、RMSNorm|
|F\-GS/F\-PACK|PR\-5|P3|store predicate、uniform scatter、通用 gather/scatter、FP4/UE8M0|

> 注：合并计划 §6\.2 原表 F\-R2 行的「PTOAS 依赖」列写作「PR\-3 \+ PR\-4」，PR\-3/PR\-4 是 TileLang 侧路标而非 PTOAS 依赖，本表已修正。
> 
> 

## 3\. PR\-1：基础 element\-wise 向量化（M1，F\-EW）

- **内容**：T\.Parallel 核心向量化（含 VMI 语言面）：`tl.vmi.*` builtin \+ VMI codegen、LayoutInference PTO 分支、Verify/Legalize/Vectorize pipeline、1D/2D 映射、E/P/Q tail mask、dtype/cast（含 FP8 转换矩阵）、element\-wise。

- **对应设计章节**：§3（pipeline）、§4（循环布局与 E/P/Q/remaining）、§5（Verify/Legalize/Vectorize）。

- **依赖**：无。

- **预计发起**：2026\-10\-12。

- **新增可打通 ST 用例**：sv1（累计 1/21）。

- **状态**：待拆分代码验证，准备分支。

- **工作项**：T0（PR\-1/PR\-2 上游准备，见 §10）。

- **待定**：VMI 语言面并入 PR\-1 一次合入（当前倾向）还是单独先行一个 PR（备选），视社区反馈决定。

- **实现参照**：fork 侧 [PR \#269](https://github.com/WenboCodes/tilelang-deepseek/pull/269)（OPEN）；VMI 语言面现状见 fork asc 分支 [deepseek\-ai/tilelang\#250](https://github.com/deepseek-ai/tilelang/pull/250)。pipeline 骨架与核心向量化已在 pto\-dev 基线通过编译/设备验证（见 §10 与 §12\.5），但拆分后分支的代码验证尚未执行。

## 4\. PR\-2：控制流（M2，F\-CF）

- **内容**：IfThenElse/Min/Max→Select、布尔表达式展开、vcmp/vsel、mask\_and/or/not。

- **对应设计章节**：§5（Legalize 的 If→Select / Min/Max→Select 与布尔表达式规范化；Vectorize ControlFlow 模块）、§9（mask 与条件更新）。

- **本阶段边界**：保留现有 continuous store 的 CF 实现与验收；目标设计中的单侧 predicate 统一改造和 uniform scatter 由 PR\-5 承接，不将当前 CF 测试通过视为这些能力已经实现。

- **依赖**：M1。

- **预计发起**：2026\-10\-19。

- **新增可打通 ST 用例**：sv7、cf2、cf5、sp4（累计 5/21）。cf4 的验收形态保留 Fragment Buffer（用例形态不变），需要 F\-FRAG，移至 M4。

- **状态**：待拆分代码验证，准备分支。

- **工作项**：T0（分支准备）、T3（控制流用例集成验证，见 §10）。

- **实现参照**：fork 侧 [PR \#262](https://github.com/WenboCodes/tilelang-deepseek/pull/262)（OPEN，控制流部分）。控制流已在 pto\-dev 基线通过编译/设备验证（见 §10），拆分后分支的代码验证尚未执行。

## 5\. PR\-3：direct reduce 与 TopK（M3，F\-R1）

- **内容**：direct `tl.tileop.reduce`（max/min/sum，单元素结果以 `1PT` 写回）、TopK example（index 全链路 int32）。

- **对应设计章节**：§7（direct reduce 语义及单元素写回）；TopK 的 `idx_ub[k] = best[0]` 按普通标量读写处理。

- **依赖**：M2 \+ **PTOAS P1**。

- **预计发起**：2026\-10\-26。

- **新增可打通 ST 用例**：本路标不新增（sv9 还依赖 M4 的 F\-FRAG）。

- **状态**：实现中；**numeric 闭环被 PTOAS P1 阻塞；临时分支可继续本地验证，主线验收待 P1 修复合入**（问题说明见 §9\.1）。

- **工作项**：

    - T1：direct Reduce 收尾——device numeric 闭环（依赖 PTOAS P1；本地临时分支验证与主线验收分开记录）。状态：**Blocked（PTOAS P1）**。

    - T2：direct Reduce 单元素 `1PT` 写回 \+ TopK 适配（E=161/L=64/P=192 尾部语义、index 全链路 int32）。状态：Todo。

    - T3：控制流与 [PR \#262](https://github.com/WenboCodes/tilelang-deepseek/pull/262) 用例集成验证：CF 设备矩阵、TopK 重跑、vecadd/vintlv/dtype/SIMT/手写 VMI corpus。状态：Todo。

- **TopK 适配与验收**：参照 PTO\-ISA PR \#2，candidate/best/idx\_ub/GM index 输出及测试回读统一 int32，记录与 PR262 原文件的 diff。staging `s` 分配 P、只 copy E，`s[E:P]` 只要求可访问；`scores[E:P]` 填 `-inf`，`idx_cand[E:P]` 填 `INT_MAX`，候选条件包含 `i<E`。两个 frontend example 的 `assert_lowered_contract` 更新为 `Verify → Legalize → Vectorize`。覆盖整除、非整除、tie\-break 和有效 score 为 `-inf`；旧非 frontend TopK 的 `scalar.cast(vreg)` 工具链风险单独探测。

- **与本地实施计划的对应**：T1 对应 5A/Gate E1；T2 对应 5B/Gate E2，首次 TopK 设备证据在此交付；T3 对应 5C/Gate E3，运行 CF 简单到复杂设备矩阵及 PR262 examples 集成验证，均在通用 Fragment/slot 之前。Gate E1 保持 BLOCKED 并保留 consume 失败证据；用户已允许继续 5B/5C 和提交经检视的代码检查点，不把继续开发记为 E1 通过。

- **与 PR\-6 的边界**：PR\-3 的 direct reduce 支持 `tl.tileop.reduce` 等直接规约操作，输入通常是已存在的向量/UB region，不经过 legacy reducer 的 canonicalization 与 materialization；PR\-6 的 reducer v2 是前端 legacy reducer 经规范化、计划和 materialization 后的 finalize 路径。两者输入 TIR、布局来源、mask 语义和验收方式都不同，不可混写（设计 §7 与 §8 分别定义）。

## 6\. PR\-4：Fragment / slot（M4，F\-FRAG）

- **内容**：Fragment/slot 存储：PlanPtoSlotStorage、layout\_map contract、UB 容量预算、MergeUBAllocations 独占分配、slot\_alloc/load/store codegen。

- **对应设计章节**：§6（Fragment Buffer 与 slot 存储）；§4（Buffer layout 与 layout\_map contract）；§5（Vectorize Storage 模块）。

- **依赖**：M1 \+ PTOAS P2。注意：slot 的**物理**实现（vreg 驻留 / slot 访存消除或保留对已绑定 backing 的读写决策）依赖 PTOAS P2，但 logical slot contract、UB 容量预算和 backing 地址分配仍由 TileLang 负责（设计 §10）。

- **预计发起**：2026\-11\-02。

- **新增可打通 ST 用例**：sv2、sv3、sv5、sv6、sv8、sv9、cf1、cf3、cf4、cf6（15/21，其中 14 个用例的实现路径已明确，cf6 已纳入但实现路线待定）——ST 用例解锁数最多（10 个）的路标。cf4 从 M2 移入（验收形态含 Fragment Buffer）。

- **状态**：Todo。

- **工作项**：T4：Fragment/slot 存储体系——通用 layout\_map contract、PlanPtoSlotStorage、AutoSchedule 容量扣除、MergeUBAllocations、slot codegen。状态：PR-4 实现完成（`zjm/pto-slot-clean`，2026-10-22 评审后重构为 9 提交：布局/Plan/预算/展开/转换/测试/多下标+区间/门控/回归用例；R4 布局求解器撤出待重写独立合入，B1/B5/A2 顺延）；复验与合入待跑（见 PR4 设计 §9 复验矩阵）。

- **风险**：Fragment/slot 是新子系统，无现成迁移来源，工作量不确定性最高（见 §12\.3）。

## 7\. PR\-5：gather、scatter 与打包类型（M5，F\-GS / F\-PACK）

- **内容**：间接访存与打包类型扩展：单侧 store predicate、uniform scatter 及通用 gather/scatter 向量化（F\-GS）、ue8m0/fp4 位操作与 reinterpret（F\-PACK）、stride\-2/interleave 布局。

- **对应设计章节**：§4\.7、§5\.2/§5\.3、§9 已定义 uniform scatter、单侧 store predicate 和双侧同址 Select 的目标规则；通用间接 gather/scatter 与 F\-PACK 仍需细化设计。

- **依赖**：M4 \+ PTOAS P3。

- **预计发起**：2026\-11\-09。

- **新增可打通 ST 用例**：sv4、sp1、sp2、sp3、sp5、sp6（累计 21/21）。

- **状态**：store 语义已确定、实现待启动；通用 gather/pack 方案待设计。

- **工作项**：T5：间接访存与打包类型扩展——单侧 store predicate、uniform scatter、Verify 准入与 mask 消费、通用 gather/scatter 向量化、ue8m0/fp4 位操作与 reinterpret、stride\-2/interleave 布局。状态：Todo。

- **已确定的 store 工作项**：Legalize 对所有单侧条件 store 保留 predicate；Vectorize 为 continuous store 和 uniform scatter 消费 `parent_mask & condition`；双侧同址 store 在可安全求值时合并 Select。Verify 检查支持的形式、类型和地址，不证明唯一写者或数据竞争。

- **测试**：无条件 uniform scatter、单 lane 条件命中/全假、嵌套条件、单侧 continuous store、双侧同址 Select、tail 与相邻地址哨兵。多 lane 同址冲突只检查编译与结构，不断言胜者。

- **阶段边界**：这些改造在 PR\-5 实施。PR\-3/本地 5B 不修改现有 CF，也不开放尚未实现的 uniform scatter；当前 unsupported 拒绝只反映阶段能力。通用 gather/pack 的方案仍需单独完成。

## 8\. PR\-6：reducer v2 与 RMSNorm（M6，F\-R2）

- **内容**：reducer v2：PTO SIMD LayoutInference、ReducerPlanAndMaterialize 分支、finalize/seed/inactive\-tail、RMSNorm numeric。

- **对应设计章节**：§8（reducer v2、finalize、seed 与 inactive\-tail）；§9（finalize participant mask 与 inactive\-tail 的 mask 规则）。

- **依赖**：reducer v2 本体为 M3 \+ M4；RMSNorm 的 uniform 向量写回还依赖 M5（见下方写回依赖）。

- **预计发起**：2026\-11\-16。

- **新增可打通 ST 用例**：ST 套件无直接消费者，由 RMSNorm 产品用例验收（累计 21/21）。

- **状态**：Todo。

- **工作项**：

    - T6：reducer 特性——PTO SIMD LayoutInference、ReducerPlanAndMaterialize PTO 分支、Vectorize reducer v2（sum/max/min、finalize、seed、inactive\-tail accumulator）。状态：Todo。

    - T7：产品验证——RMSNorm compile/numeric、ST 套件 21 个 kernel 验收（见 §11）。状态：Todo。

- **写回依赖**：reducer v2 本体仍依赖 M3/M4；RMSNorm 若采用目标设计中的向量 RHS → uniform UB 写回，还依赖 M5 的 uniform scatter。本地实施计划的任务 7 启动前核对该能力，未完成时先记录对应数值用例的依赖阻塞。

- **与 PR\-4 的边界**：PR\-6 不是 PR\-4 的同义词。PR\-4 交付 Fragment/slot 存储体系；PR\-6 在其上交付 reducer v2 语义与 RMSNorm 验收。没有 slot contract 就不能宣称 RMSNorm 完成。

- **实现注记**：reducer v2 的 inactive\-lane 保留语义（inactive lane 保留历史部分和，finalize 使用完整参与组 mask）是正确性要求，早期实现曾缺失该行为；PR\-6 验收需覆盖 inactive\-tail 用例（语义见目标设计 §8）。

## 9\. PTOAS P1/P2/P3 依赖

|\#|工作项|阻塞关系|影响路标|
|---|---|---|---|
|P1|**vecscope 内 V\-pipe 依赖分析 \+ ****`pto.mem_bar`**** 自动插入**（见 §9\.1）|阻塞 direct Reduce 的 numeric 验证|M3|
|P2|slot 物理实现：vreg 驻留 / slot 访存消除或保留（对已绑定 backing 的读写）决策，liveness 与物理寄存器压力分析，逻辑向量→物理寄存器拆分；为保留的 UB 访问插 mem\_bar|Fragment/slot 阶段配套|M4|
|P3|gather/scatter 与打包类型的 PTO 侧指令配套；尚未完成方案分析与设计，待细化|M5 配套|M5|

### 9\.1 P1 问题说明（供 PTOAS 对接）

**最小复现**（TileLang 源码）：

```python
@T.prim_func
def main(A: T.Tensor((128,), "float32"), B: T.Tensor((1,), "float32")):
    with T.Kernel(1):
        a = T.alloc_shared((128,), "float32")
        best = T.alloc_shared((1,), "float32")
        T.copy(A, a)
        with T.SimdVF(lanes=64):
            for i in T.Parallel(128):
                a[i] = a[i] * 2                       # Parallel 写 UB（in-place）
            T.reduce_max(a, best, dim=0, clear=True)  # 同 region direct reduce 读同一 UB
        T.copy(best, B)
```

**历史复现中 TileLang 生成的 PTODSL**（关键片段，保留当时输出，不代表当前目标写回形态）：

```python
with pto.vecscope():
    mask = pto.vmi.create_mask(64, size=64)
    for q in range(0, 2, 1):                       # unit 循环：vload → vmul → vstore 写 UB
        v = pto.vmi.vload(pto.addptr(a_ptr, q * 64), 0, size=64)
        pto.vmi.vstore(pto.vmi.vmul(v, two, mask), pto.addptr(a_ptr, q * 64), 0, mask)
    # ← unit 的 vstore 与 reduce 的 vload 之间没有任何保序
    r0 = pto.vmi.vload(a_ptr, 0, size=64)          # reduce：读同一 UB 地址
    r1 = pto.vmi.vload(a_ptr, 64, size=64)
    red = pto.vmi.vcmax(pto.vmi.vsel(mask, pto.vmi.vmax(r0, r1, mask), r0), mask)
    pto.vmi.vstore(pto.vmi.vbrc(red, size=64), best_ptr, 0, pto.vmi.create_mask(1, size=64))
```

**问题**：同一 SIMD VF 内部、同一 UB 地址的 `vstore`（Parallel 写）→ `vload`（reduce 读）RAW 依赖没有被保序，设备上 reduce 读到写入前的旧值。确定性复现（非竞态）；9 用例数值矩阵 6 失败 3 通过；region 之后 copy\-out 能读到新值（store 本身已落盘）。

**需要 PTOAS 做的工作**：在 VPTO 后端的 vecscope 内做 load/store 依赖分析，对真实依赖边按需插入 `pto.mem_bar`（向量 store/load 之间 RAW 用 `VST_VLD`、WAR 用 `VLD_VST`）。`pto.mem_bar` 是 vecscope 内合法原语（16 种 kind，TileOps a5 生产模板已有 `vsts`→`vlds` 同 tile 插 `VST_VLD` 的同构先例）；注入单行 `pto.mem_bar "VST_VLD"` 后 9 用例矩阵 9/9 全部通过（手工注入的实验性验证，不代表 PTOAS 自动插入已落地）。同一机制还需按真实指令队列覆盖 direct Reduce 写回后的读取：vector store → vector load 使用 `VST_VLD`，vector store → auxiliary scalar load 使用 `VST_LD`，不能将两者混用。TileLang 侧不实现该分析。mem\_bar 有运行时开销，应按需而非无条件插入。

> 上述为 PTOAS master `b9da12032` 的历史定位证据。后续临时 PTOAS 分支已有验证记录，但 consume 失败仍未闭环，Gate E1 继续 BLOCKED；本次文档更新不重写历史产物或将其改为通过。完整证据链随 PR\-3 提供。
> 
> 

## 10\. 代码模块、pass 拆分和分支

### 10\.1 TileLang 侧工作分解（T0–T8）

|\#|工作项|状态|归入上游 PR|
|---|---|---|---|
|T0|PR\-1/PR\-2 上游准备：基于 PTO\-ISA/tilelang（fork 自 tile\-ai main）整理发起分支，剔除 pto\-dev 中 ci/example 等与 PTO/VMI 无关的改动（见 [PTO\-ISA/tilelang\#1](https://github.com/PTO-ISA/tilelang/pull/1)），拆分 PR\-1/PR\-2 分支并完成代码验证|Todo|PR\-1 / PR\-2|
|T1|direct Reduce 收尾：device numeric 闭环（依赖 PTOAS P1；本地临时分支验证与主线验收分开记录）|Blocked（PTOAS P1）|PR\-3|
|T2|direct Reduce 单元素 `1PT` 写回 \+ TopK 适配（E=161/L=64/P=192 尾部语义、index 全链路 int32）|Todo|PR\-3|
|T3|控制流与 [PR \#262](https://github.com/WenboCodes/tilelang-deepseek/pull/262) 用例集成验证：CF 设备矩阵、TopK 重跑、vecadd/vintlv/dtype/SIMT/手写 VMI corpus|Todo|PR\-2 / PR\-3|
|T4|Fragment/slot 存储体系：通用 layout\_map contract、PlanPtoSlotStorage、AutoSchedule 容量扣除、MergeUBAllocations、slot codegen|实现完成，待复验合入（R4 求解器除外）|PR\-4|
|T5|间接访存与打包类型扩展：单侧 store predicate、uniform scatter、Verify 准入与 mask 消费、通用 gather/scatter 向量化、ue8m0/fp4 位操作与 reinterpret、stride\-2/interleave 布局|Todo|PR\-5|
|T6|reducer 特性：PTO SIMD LayoutInference、ReducerPlanAndMaterialize PTO 分支、Vectorize reducer v2（sum/max/min、finalize、seed、inactive\-tail accumulator）|Todo|PR\-6|
|T7|产品验证：RMSNorm compile/numeric、ST 套件 21 个 kernel 验收（见 §11）|Todo|PR\-5 / PR\-6|
|T8|最终回归（[PR \#269](https://github.com/WenboCodes/tilelang-deepseek/pull/269)/[PR \#262](https://github.com/WenboCodes/tilelang-deepseek/pull/262)/dtype/SIMT/AscendC/手写 VMI/slot/reducer 全量）与上游 PR 拆分、review、合入|Todo|PR\-1 \~ PR\-6|

已完成部分（不占用路标）：pipeline 骨架、核心向量化与控制流已在 pto\-dev 基线通过编译/设备验证，对应 PR\-1/PR\-2 的内容。

### 10\.2 分支与 PR 编号对照

- **上游（tile\-ai/tilelang）**：`target="ascend"` 后端（\#3308）、PTO driver/codegen（\#3310）已合入主线；PR\-1～PR\-6 为待发起的上游路标，发起分支基于 PTO\-ISA/tilelang（fork 自 tile\-ai main）整理（T0）。

- **PTO\-ISA/tilelang**：pto\-dev 分支为当前开发基线；[PR \#1](https://github.com/PTO-ISA/tilelang/pull/1) 为发起分支整理参照，[PR \#2](https://github.com/PTO-ISA/tilelang/pull/2) 为 ST 验收套件。

- **fork（WenboCodes/tilelang\-deepseek）原始特性实现参照**：[PR \#269](https://github.com/WenboCodes/tilelang-deepseek/pull/269)（T\.Parallel→PTO VMI 自动向量化）、[PR \#262](https://github.com/WenboCodes/tilelang-deepseek/pull/262)（控制流 \+ reduce\_max/min/sum）。截至合并计划编写时两者均为 OPEN。fork PR 编号与上游 PR\-1～PR\-6 路标编号分开使用，不互相指代。

- **fork（deepseek\-ai/tilelang）**：VMI 语言面在 asc 分支（[\#250](https://github.com/deepseek-ai/tilelang/pull/250)），未合入主线。

### 10\.3 源码核对入口

|文件或模块|核对内容|
|---|---|
|`tilelang/ascend/pipeline.py`|Ascend pipeline 和 pass 接入|
|`tilelang/ascend/target.py`|PTO 与 AscendC 的 target 识别|
|`tilelang/ascend/language/frame.py`|`T.SimdVF(lanes=...)` 前端接口|
|`src/ir.cc`|SimdVF FFI、frame 状态和 TIR block 注解|
|`src/transform/layout_inference/layout_inference.cc`|布局推导和注解写回|
|`src/op/parallel.cc`|Parallel 布局候选|
|`src/transform/loop_partition.cc`|迭代点划分和布局求逆|
|`src/layout/layout.h`|Layout / Fragment 表示|
|`3rdparty/tvm/src/tirx/ir/buffer.cc`|`ElemOffset` 的多维元素偏移计算|
|`src/transform/lower_access_ptr.cc`|`tl.access_ptr` 到 `tvm_access_ptr` 的区间描述|
|`src/ascend/transform/ascend_simdvf_lower_parallel.cc`|现有 Ascend SIMD lowering|
|`tilelang/ascend/analysis/vf_checker.py`|通用 VF 检查参考|
|`tilelang/ascend/language/vmi.py`|VMI API、逻辑类型和地址约定|
|`src/ascend/codegen/codegen_pto.cc`|VMI 到 PTODSL 的 codegen|

其他核对入口：reducer 前端接口、epoch 校验和 reducer v2 测试（含共同 seed 和初始化捕获用例）；Parallel 布局与 reducer proposal、Ascend 布局推导及注解写回、部分和分析与 finalize 接口和 PartialFragment 定义；RMSNorm 源程序与 TIR 生成记录。fork 侧 [PR \#269](https://github.com/WenboCodes/tilelang-deepseek/pull/269)（向量化）与 [PR \#262](https://github.com/WenboCodes/tilelang-deepseek/pull/262)（控制流与 direct reduce）中的对应实现为迁移参照（见 §10\.2）。

## 11\. ST、TopK、RMSNorm 验收矩阵

- **ST 套件**（[PTO\-ISA/tilelang\#2](https://github.com/PTO-ISA/tilelang/pull/2)，`examples/ascend/t_parallel_st_suite/`）：`kernels/` 21 个 kernel 全部纳入本期验收，验收路径为 **SimdVF→PTO 向量化**，暂不设置 opsim 性能门槛（`kernels_ptodsl/` 21 个手写 PTODSL 版作正确性与性能参照）。

- **产品用例**：TopK（非整除、tie\-break、\-inf 边界）、RMSNorm（compile \+ numeric）；回归：vecadd/vintlv/PTO dtype/PTO SIMT/手写 VMI、AscendC 不回归。

### 11\.1 逐 kernel 特性需求矩阵（21 个 kernel 逐个分析的结论）

|特性|含义|归属路标|
|---|---|---|
|F\-EW|element\-wise 基础向量化：连续 load/store、算术、broadcast、cast、tail mask|M1|
|F\-CF|lane\-varying 控制流：IfThenElse/Min/Max→Select、mask 逻辑、条件更新|M2|
|F\-R1|direct reduce：reduce\_max/min/sum 向量化、`1PT` 单元素写回与 TopK|M3|
|F\-FRAG|Fragment/slot 存储：值驻留寄存器（KEEP\-in\-RF）、accumulator 跨循环驻留、layout\_map/slot|M4|
|F\-GS|uniform scatter、store predicate 与通用 gather/scatter 间接访存|M5|
|F\-PACK|打包类型/特殊位操作：ue8m0/fp4、bit reinterpret、nibble 拆包|M5|
|F\-R2|reducer v2：reducer LayoutInference/materialize、finalize/seed、inactive\-tail accumulator|M6|

逐 kernel 需求（"读/写"标注间接访存方向）：

|用例|计算模式|F\-EW|F\-CF|F\-R1|F\-FRAG|F\-GS|F\-PACK|可满足路标|
|---|---|---|---|---|---|---|---|---|
|sv1|流式 eltwise|✓||||||**M1**|
|sv2|absmax scale \+ 广播应用（KEEP/reload 臂）|✓|✓||✓|||M4|
|sv3|GEMV partial 累加（acc 跨 K 驻留；split 臂有 guard）|✓|✓||✓|||M4|
|sv4|index gather \+ psum（`w_ub[idx[i]]` 读侧 gather）|✓|||✓|✓读||M5|
|sv5|小组 absmax scale（G16）|✓|✓||✓|||M4|
|sv6|中组 absmax（G32，warp 尺度驻留）|✓|✓||✓|||M4|
|sv7|大组 absmax（G128，partial 经 UB round\-trip）|✓|✓|||||**M2**|
|sv8|quant e2e：scale 广播回全网格|✓|✓||✓|||M4|
|sv9|topk e2e（reduce\_max/min \+ kill 迭代）|✓|✓|✓|✓|||M4|
|cf1|阈值 kill（scores KEEP 跨 serial K）|✓|✓||✓|||M4|
|cf2|阈值 kill remat（条件写 shared，跨 region 依赖）|✓|✓|||||**M2**|
|cf3|victim idx kill \+ lane index remat|✓|✓||✓|||M4|
|cf4|嵌套 if（clamp\-hi/clamp\-lo/fat\-mul 三路 select；验收形态保留 Fragment staging）|✓|✓||✓|||M4|
|cf5|除法近零保护分支（fp32 div）|✓|✓|||||**M2**|
|cf6|Newton 迭代分支（Parallel body 内嵌常数界 serial，路线待定，见 §12\.4）|✓|✓||✓（轻）|||M4（路线待定）|
|sp1|双 payload 行 scatter（pos 间接行写）|✓|✓||✓|✓写||M5|
|sp2|行 gather \+ 手动加权累加（acc KEEP）|✓|||✓|✓读||M5|
|sp3|ue8m0 scale 位合成（reinterpret/shift）|✓|||||✓|M5|
|sp4|pad 掩膜双分支拷贝（纯 vsel）|✓|✓|||||**M2**|
|sp5|sideband/interleave 行 gather|✓||||✓读||M5|
|sp6|fp4 LUT 解包（lane\-varying LUT gather \+ stride\-2 写 \+ nibble 位操作）|✓||||✓读|✓|M5|

**路标解锁汇总**：

|路标|新解锁用例|累计|
|---|---|---|
|M1（F\-EW）|sv1|1/21|
|M2（\+F\-CF）|sv7、cf2、cf5、sp4|5/21|
|M3（\+F\-R1）|—（sv9 同时依赖 M4 的 F\-FRAG）|5/21|
|M4（\+F\-FRAG）|sv2、sv3、sv5、sv6、sv8、sv9、cf1、cf3、cf4、cf6（cf6 路线待定）|15/21（14 个用例路径已明确，cf6 已纳入、路线待定）|
|M5（\+F\-GS/F\-PACK）|sv4、sp1、sp2、sp3、sp5、sp6|**21/21**|
|M6（\+F\-R2）|ST 套件无直接消费者（F\-R2 由 RMSNorm 产品用例验收）|21/21|

三点结论：

1. **SV4 需要 F\-GS**：核心是读侧 lane\-varying gather `w_ub[idx[i]]`，仅靠 F\-EW/CF/R1/FRAG 覆盖不了。

2. **SP4 只需 F\-EW\+F\-CF**：名字带 gather 但索引恒等，是纯 lane\-varying if/else（vsel）用例，M2 即可满足。

3. **F\-R2 在 ST 套件没有消费者**：套件内的归约要么是手工 `alloc_var`\+`T.max` 的 per\-lane serial 折叠（SV2/5/6/7/8），要么是 direct `T.reduce_max/min`（SV9，归 F\-R1）；F\-R2 的验收挂在 RMSNorm 上。

### 11\.2 分层验收清单（基础向量化与控制流）

以下清单承接源方案文档1的分层测试与验收要求，按最终设计修正：术语统一为 VL；不再区分「第一/第二阶段」。验收归属：基础向量化与控制流归 PR\-1/PR\-2；direct reduce 相关用例归 PR\-3（含数值与 mem\_bar 验收，依赖 PTOAS P1，见 §9\.1）；slot 与 reducer v2 相关验收见 §11\.3。

**布局和 TIR 结构：**

|用例|检查重点|
|---|---|
|一维 vecadd|Parallel 替换、chunk 范围、SIMD\_VF 和 lanes 注解保留、布局注解删除|
|一维常量非整除|原始 For extent 为 E、布局 shape 为 P；补齐域的前向/逆向映射往返一致，Q 和每轮 mask 正确|
|二维 Parallel|接受 `T.Parallel(M, N)` 及等价的两层紧邻 For，只消费最外层注解；分别覆盖 j 维和 i 维连续，确认只补齐选中维，全局 p 与行内 q 正确对应|
|退化二维选维|紧凑 `(127,1)` 选 i、紧凑 `(1,127)` 选 j；核对布局 shape、chunk 数和有效 lane 数|
|向量地址|覆盖非零 chunk、双缓冲版本槽位、显式 stride 和非零 `elem_offset`；检查 `access_ptr` 的实际起点、extent=VL、VMI offset=0，偏移只计入一次|
|标量和绑定|覆盖标量 load \+ `vbrc`、已有 Bind 及多次 Var 引用；整数标量求和后 store 时，加法保持标量，只广播结果；浮点标量按向量计算（设计 §8\.5）|
|下标作为值|覆盖仅用于地址、`a[i] + i`、下标直接输出和 `2 * i`；检查 `vci` 按需生成、整数 dtype 保留、二维外层坐标仍为标量|
|多语句和元数据|保持多条 store 的顺序，保留有效元数据，清理旧循环变量引用；无 Parallel 的纯手写 VMI 区域跳过|
|remaining 作用域|每行或列重置、每个顺序 Parallel 独立初始化、不同 SIMD\_VF 区域互不串扰；局部 DCE 保留初始化和更新|
|串行包裹整体|区域内串行 For 包住 Parallel 整体（含多层串行、串行变量进地址、整体间夹标量语句）：检查接受且顺序保持；非整除整体保留嵌套循环和 remaining 初始化位置；Parallel 体内串行仍拒绝|
|循环种类|`Unrolled` 包裹整体接受；`Vectorized` 包裹整体拒绝（并列不扩为拒绝）；`ThreadBinding` 区域级拒绝，并列与包裹同等，诊断带 `ForKind`（线程绑定还带绑定名）|
|区域级 Assert|条件引用已定义变量的 `AssertStmt` 接受；引用未定义变量的拒绝（`AssertStmt` 无 body，只检查条件）|
|后端不变|AscendC 的 `SIMD_VF` 区域内 Parallel 不新增任何布局注解（无 PTO key，也无 `parallel_loop_layout` / padding guard）；PTO 的无 Parallel 手写 VMI 区域同样不新增注解|
|Cast|检查转换属性和数值边界，不引入中间转换或 masked vcvt；混合位宽的结果由同一逻辑 mask 限定写回|

Vectorize 局部 DCE 后检查直接输出：仅用于 continuous 地址的下标不应留下无用 `vci` 链；对设计 §2\.1/§12\.1 的 vecadd 输入，Vectorize 直接输出必须包含三次 VMI load（这些 load 都有使用，PTOAS 合并重复 load 的结果另行记录）。后续迁移清理位置时，按设计 §5\.5\.5 调整死代码清理的验收层次。

**Verify 负例：**

负例覆盖非法 lanes、缺少或不可逆布局、非零 loop min、非法 step、非正或非常量 extent、非连续地址、GM 访问、不受支持的跨 lane 归约形式、不支持的 Cast、混合手写 VMI/Parallel 和向量化维不一致。布局负例还包括未按选中维补齐、补齐过大、误改保留维、非规范映射；一维 `Parallel(1)` 和二维 `(1,1)` 的单点形态为分类拒绝负例。

布局、区域级断言条件和循环种类用例直接构造 TIR 并调用 Verify，避免前端先拒绝而测不到 pass 内的检查位置。断言分类诊断（`[VerifyParallelToPTO]` 前缀、具体原因及位置）；已读取本区域 lanes 时，诊断打印正确值，尚未读取时省略该字段；不能泄漏裸 TVM 断言。

extent 非整除为正例；同一逻辑长度搭配未补齐或错误的布局仍应拒绝。`M=2, N=64, VL=128` 为按行 tail 的正例，检查两行各用一个 64\-active mask，没有把逻辑点合并到同一 chunk。

循环结构负例仍包括三维及以上 Parallel 嵌套、Parallel 体内的串行/流水/While 循环和不可规范化的条件分支。区域内串行 For 包住 Parallel 整体、整体之间夹标量语句仍接受。类型负例保留待转换 Parallel 中的 FP4 load/store 和涉及 FP4 的 Cast；已有手写 VMI 的对应类型沿原路径验证。

跨逻辑点依赖、未知别名、无法证明 padding 安全的情形，都不作为这两个 pass 必须报错的负例。设备正例必须满足调用方契约，不执行已知越界输入来验证拒绝行为，也不以串行执行结果作为非法 Parallel 的验收标准。

**Pipeline、codegen 和后端优化：**

检查 `LowerAccessPtr` 输出的元素 dtype、每 chunk 起点、extent=VL 和读写掩码，并核对后续版本布局改写和 UB 合并后的地址。FP8 的 extent 按元素计数。保留现有同步和版本关系。

目标分流测试确认 PTO SIMD 使用独立布局 key。AscendC 的 `SIMD_VF` Parallel 注解集合应与基线一致，不新增 PTO key、`parallel_loop_layout` 或 padding guard；PTO 无 Parallel 的手写 VMI 区域同样不新增注解。

生成的 TIR 必须能够转换为 PTODSL，通过 PTOAS VMI verifier 并完成编译。分别保存 Vectorize 直接输出和 PTOAS 优化结果，记录重复 vload 是否被 MLIR CSE 消除、标量 load 是否被 LICM 提升，并检查优化没有跨 store 或 barrier 错误复用值。未发生 CSE/LICM 不自动判为向量化失败，错误复用导致的语义变化必须修复。

非整除路径还检查以下关系：

- 非整除路径按单个 chunk 循环生成，循环前初始化 remaining，循环内先生成 mask、再计算和 store、最后更新 remaining。允许后续等价简化；例如 `Q=1` 时可以消除只迭代一次的 chunk 循环及无用的状态更新。

- 对仍保留跨迭代标量递推的循环，PTODSL AST rewrite 生成 `scf.for iter_args` / `scf.yield`，初始化与更新的类型一致；TileLang 不直接输出这些 MLIR 节点。循环或递推已被等价简化时，不要求产物中仍出现这些节点，验收以有效 lane、写回范围和每次外层迭代的重置语义为准。

- 在实际使用的 PTOAS 版本上确认动态 `create_mask` 能完成物理化。VPTO/LLVM 中可出现 `plt_b8/b16/b32` 或被优化后的等价形式；多物理 part、混合位宽的 predicate 均覆盖正确的逻辑 lane。

- 在实际使用的 PTOAS 版本上确认 masked store 的部分写回行为，覆盖受支持 dtype；编译和 VPTO 观察与设备数值分别记录，不以属性名推断写回语义。

- 生成的局部标量状态没有变成 UB 或栈上的额外读写；Buffer 地址、版本槽位维和 masked store 的范围仍正确。不能只按 PTODSL 文本是否出现 `remaining` 判断整条路径通过。

**数值正确性：**

覆盖 `lanes=64/128/256`、一维 vecadd、二维连续维 Parallel、多个 BufferStore、标量常量、标量 UB load 和受支持的 Cast。以独立参考计算为基准，AscendC 结果作为补充对照；两个后端结果相同并不足以确认正确性。

依赖相关用例使用合法 Parallel 输入，覆盖各点原地更新、同一点内的多语句依赖，以及顺序 Parallel 之间经 Buffer 传递结果。Cast 用例同时验证转换属性和数值边界；无法直接映射的组合必须在 Verify 中报错。

对每个 `VL=64/128/256` 覆盖以下长度；`E=1` 属于当前不支持的边界负例。

|长度类别|代表 E|检查重点|
|---|---|---|
|`1<E<VL`|`VL-1`|只有一个 chunk，所有有效结果正确|
|`E=VL`|`VL`|单个完整 chunk，与现有结果一致|
|`E>VL` 且非整除|`VL+1, 2*VL-1, 2*VL+17`|一个或多个完整 chunk 后正确处理 tail|
|`E>VL` 且整除|`2*VL, 3*VL`|多 chunk 整除路径保持原行为|

另保留 `E=150, VL=128` 的示例。二维分别构造沿 j 和沿 i 连续的输入，至少含两个保留坐标，并使用不同的行/列数据，检查 remaining 重置以及真实 Buffer stride 未被布局补齐改写。

连续输入使用可容纳完整 tail load 读取的存储，并提前初始化 padding。输出预填 sentinel，逐元素比较有效范围，确认 `[E,P)` 及相邻行/列的哨兵值没有被无效 lane 写入。混合位宽覆盖 FP16/BF16→FP32、整数下标和已支持的 FP32↔FP8 方向，确认 VL 按元素计数。再用满足后端对齐约束的紧凑输出 Buffer，验证输出侧不要求补齐到 P。

**验收要求：**

|类别|要求|
|---|---|
|范围|编译期常量 extent（整除与非整除），及目标设计规定的一维、二维、表达式和类型边界；非整除路径仅选中维补齐、单个 chunk 循环、remaining 和动态 prefix mask|
|必须通过|TIR 结构、Verify 负例、PTO codegen、PTOAS 编译和数值正确性测试|
|兼容性验证|选取与本次改动有关的 AscendC、PTO SIMT 和手写 VMI 已有代表用例，确认原编译路径和行为保持正常|
|必须检查并记录|重复 load 的 CSE、标量 load 的 LICM；未被优化不自动判为向量化失败|
|本期不实现|符号 extent 或 guard、通用 masked load、自动 padding、额外边界/同步证明、特殊打包类型，以及后续访存和性能优化|

兼容性验证分别确认：AscendC 仍走现有 `AscendSimdVFLowerParallel`；PTO SIMT 仍使用原线程布局；没有待转换 Parallel 的手写 VMI 区域由本路径跳过，继续正常编译。用例选择围绕本次涉及的共用前端、LayoutInference 和 Ascend pipeline。

数值比较方法应与参考计算和被测 dtype 的语义一致，并记录所用容差或精确比较条件。环境不具备、未执行和测试失败分别记录，未执行的检查不能列为通过。本期不设指定性能提升指标；递推写法的指令数量按编译结果记录。

### 11\.3 slot 存储与 pipeline 验收（PR\-4）及 RMSNorm 基线（PR\-6）

承接源方案文档2的实现检查要求，按最终设计修正。实现时应保存 Plan 输出、调度后、Verify 输入、Legalize 输出、Vectorize 输出及 UB 合并后的 IR，分别检查容量规划、源语义、SIMD 表达和最终地址。reducer 前置阶段与准入用例见设计 §8\.6/§8\.7，单 update 按静态更新语句计数。

存储与 pipeline 的验收要求：

|检查位置|预期结果|
|---|---|
|PlanPtoSlotStorage|在 AutoSchedule 准备阶段前执行；仅为待转换的 Fragment Buffer 声明 backing 并预建句柄，保留标量计算；无 seed 的 RMSNorm 预留 17 KiB|
|AutoSchedule 容量预算|slot 预留按资源域计入一次，不因尚无 backing 读写漏算，也不重复计入普通 Buffer；覆盖加入预留后双缓冲不可行、单缓冲可行，以及最低需求仍超限的用例|
|多缓冲、同步和清理|backing 保持一份，不产生虚构的 DMA/fill 任务；分配、规划和 Buffer 替换保持一致；Vectorize 前不丢失尚无数据访问的 backing|
|双缓冲与连续 VF 调用|输入、输出 UB 使用 ping/pong 时，slot 仍为单份且无版本槽位索引；连续多轮使用不同输入，结果符合参考，slot 按源初始化和写入产生本轮值|
|Vectorize|每个 slot\_alloc 关联规划中的 backing 和预建句柄；没有新增未经预算的空间；count、类型和索引与布局一致|
|地址 lowering 与 codegen|slot\_alloc 的 access\_ptr 随展平和合并改为最终 UB 偏移；slot\_load/store 保留到 PTOAS 输入，没有提前变成普通 vload/vstore，也不重复分配空间|
|MergeUBAllocations|自动和手动调度路径均为所有 slot backing 分配独占空间；与普通 UB、其他 slot 不重叠；包含对齐的最终总量不超出容量|
|PTOAS|分别验证已预留 UB 上的 slot 读写、寄存器驻留，以及保留多次 slot\_load 的结果；对实际 UB 读写生成必要的 mem\_bar；不新增 UB 分配，消除访存后 TileLang 预留容量保持不变|

RMSNorm 的计算基线断言（PR\-6 执行，与设计 §8\.7 的 reducer 用例表互补）：

- 前置 PTO 阶段已确认只有一个静态 update，完成部分和推导及初始化、更新展开；Legalize 保留 x 的逻辑分配、两个消费循环、普通累加表达式、finalize 参数和结果 copy。

- VL=128 且无 seed 时，`x_frag`、`sum_sq_1`、`sum_sq_result` 分别对应 `x_slot`、`acc_slot`、`result_slot`，count 为 32、1、1。共三个 slot 对象、34 个 entry；两处 x 消费使用相同 slot 映射。

- Vectorize 后，原 Parallel 被 q 循环替代，更新包含向量乘加，finalize 包含跨 lane 求和、广播和写回；每条参与 lane 都获得完整结果，没有残留待处理的 reducer 或 SIMT AllReduce。

- `var/rstd_val` 随向量操作数提升并保持 SSA 定义使用，标量常量被广播，不生成标量浮点 ALU 计算。UB 写回按实际 RHS 类型、地址和执行 mask 核对；向量 RHS 写 uniform 地址依赖 PR\-5 的 scatter，并验证相邻地址不被改写。RMSNorm 的这条写回路径不能由 direct Reduce 的 `1PT` 测试代替。

- seed 为共同标量初值，PTO materializer 在初始化位置用 `seed_frag` 捕获，Vectorize 将这个 `local.fragment` Buffer 映射为 `seed_slot`；RMSNorm 的 finalize 先广播规约结果，再做 seed 的向量 combine，每个逻辑输出只计入一次。逐输出 seed 输入与未支持的 Fragment 单元素提取明确报错。

- slot 操作不带 mask，读写类型一致，变量 q 的索引范围合法，条件更新保留不活跃 lane 的旧部分和。

- 每个 TileLang 生成的 slot\_alloc 都能追溯到 Vectorize 输入中的 `local.fragment` 分配。平方、规约、广播、rstd 和权重加载等普通计算中间值不额外分配 slot；同一向量 SSA 值的使用直接引用其定义。

- Fragment Buffer 的读写按源语义保留；没有因相同表达式或相同地址而提前共享可变内存的读取。两次 `x_frag[i]` 的 BufferLoad 仍生成两次 slot\_load。

- 覆盖完整 VL、常量非整除、显式条件和 max/min，按对应的浮点语义检查结果。单独检查循环内及跨循环使用的 SSA 值不创建 slot，以及 Fragment Buffer 的 entry 数量按 shape 和布局保留，不因消费紧邻生产而缩小。

- PTOAS 分别验证已预留 UB 上的保存与读取、寄存器驻留，以及保留多次 slot\_load 的合法结果；性能与寄存器压力需要单独测量，不能由 TIR 结构检查代替。

## 12\. 当前状态、阻塞项与证据

### 12\.1 状态一览

|路标|状态|
|---|---|
|PR\-1 / PR\-2|待拆分代码验证，准备分支（对应内容已在 pto\-dev 基线通过编译/设备验证，见 §10\.1）|
|PR\-3|实现中；numeric 验证被 PTOAS P1 阻塞|
|PR\-4|Todo|
|PR\-5|store 语义已定；通用 gather/pack 待设计|
|PR\-6|Todo|

### 12\.2 阻塞项与前置核对项

- **T1 / PR\-3 numeric 验证：Blocked（PTOAS P1）**。direct reduce 的 device numeric 验收依赖 PTOAS P1（vecscope 依赖分析 \+ 自动 `mem_bar`，见 §9\.1），主线验收待 P1 修复合入；已批准的临时分支本地验证可以继续。

- **PR\-6 实现注记**：reducer v2 的 inactive\-lane 保留语义为正确性要求，验收需覆盖 inactive\-tail 用例（见 §8）。

- **P3 未细化**：gather/scatter 与打包类型的 PTO 侧配套尚未完成方案分析与设计。

### 12\.3 风险与依赖

1. **P1（mem\_bar）需在 PR\-3 发起（2026\-10\-26）前修复并合入 PTOAS 主线**；若滑期则 M3 顺延。

2. **PR\-1 拆分方式需社区反馈**：VMI 语言面并入首个 T\.Parallel PR（当前倾向）vs 单独先行（备选）。

3. **TileLang 与 PTOAS 的版本配套**：TileLang 跟随 PTOAS/CANN 主线演进，上游合入时需声明依赖的 PTOAS 最低版本（P1 落地后 M3 起有硬性版本依赖）。

4. **Fragment/slot 是新子系统**，无现成迁移来源，工作量不确定性最高；且 M4 是 ST 用例解锁数最多（10 个）的路标，其延期对验收面影响最大。

5. **ST 套件验收需经 SimdVF→PTO 向量化路径**：现有 `kernels/` 为 SimtVF 形态，验收时需以 SimdVF 形态编译通过并数值正确。

### 12\.4 待定事项

* [ ] **PR\-1 拆分方式**：VMI 并入首个 T\.Parallel PR（倾向）还是单独先行（备选）——视社区反馈。

* [ ] **CF6 内嵌串行循环的路线选择**：cf6 的 Parallel 体内含常数界 `T.serial(3)` Newton 迭代。cf6 已纳入本期验收范围，以下路线均以覆盖 cf6 为前提：\(1\) Legalize 展开常数界串行循环，并合并分支臂内同位置写链；\(2\) Vectorize 直接支持 Parallel 体内 uniform 串行循环；\(3\) 验收层改写用例形态。待讨论确定。

### 12\.5 证据与验证记录

- **已执行**：pipeline 骨架、核心向量化与控制流已在 pto\-dev 基线通过编译/设备验证（对应 PR\-1/PR\-2 内容，见 §10\.1）。

- **已执行（实验性）**：针对 §9\.1 的 P1 问题，手工注入单行 `pto.mem_bar "VST_VLD"` 后 9 用例数值矩阵 9/9 通过（基于 PTOAS master `b9da12032`）；这是问题定位与修复方向的证据，不代表 PTOAS 自动插入已落地。完整证据链随 PR\-3 提供。

- **已执行（失败记录）**：P1 问题最小复现为确定性复现，9 用例数值矩阵 6 失败 3 通过（见 §9\.1）。

- **证据层级（历史证据，不等于编译器或设备验收通过）**：seed 输入探针只执行了真实前端和 `VerifyReducerEpoch`（浮点标量常量与 `seed_ub[0]` 通过；整个 Buffer、切片、显式 Broadcast 和 Python 列表被拒绝），仅确认前端输入契约；row/column 归约的整数模型检查（VL=64/128/256，N=1、VL\-1、VL、VL\+1、2\*VL\+17、4096，M=1、3，有无 seed 的 sum/max/min，共 432 组通过）只检查数学分解，不涉及 TIR 变换、浮点重结合、物理 load 或设备执行。

- **未闭环/未落地（不得视为通过）**：PR\-1/PR\-2 拆分分支的代码验证（T0）；PR\-3 的 device numeric 验收（T1，Blocked）；`1PT` 单元素写回与 TopK 适配（T2）；CF/TopK/回归集成验证（T3）；Fragment/slot（T4）；gather/scatter 与打包类型（T5）；reducer v2（T6）；RMSNorm compile/numeric 与 ST 套件 21 kernel 验收（T7）；最终全量回归（T8）。

## 13\. 文档迁移和历史文档说明

### 13\.1 本文与配套文档的关系

- 本文是实现路线图，内容以 [Issue \#3](https://github.com/PTO-ISA/tilelang/issues/3) 发布版为基础，并按合并计划补充映射表与状态核对项；此前定稿时已一次性刷新 Issue \#3；本轮 5B/PR\-5 调整仅更新本地文档，尚未同步该 issue。

- 配套的《TileLang Parallel 到 PTO：目标设计方案》（草稿：`docs/drafts/pto-parallel-target-design-draft.md`）定义最终语义和职责，是长期有效的规范来源；本文引用其章节（§2），不重复定义 slot、mask 或 reducer 语义。

- 两份历史源文档保持原样、不覆盖：

    - 文档1：[TileLang Parallel 到 PTO 的向量化方案](https://hw-native-sys.feishu.cn/wiki/QLC6wq1mEi5XpBkaWaNcWNnknmf)（Wiki，核对时 revision 76）；

    - 文档2：[PTO Parallel 向量化：pass pipeline 与 Fragment 存储设计](https://hw-native-sys.feishu.cn/docx/I3dBdHwKuoKoVqxFtO9cKdrDnTh)（Docx，核对时 revision 37）。

### 13\.2 文档内容覆盖表

|源文档|原章节|合并位置|处理方式|
|---|---|---|---|
|文档1|§4\.3～§4\.6|目标设计 §4|保留循环布局、二维求逆和地址规则，补术语的词义区分|
|文档1|§5～§6|目标设计 §5|保留 Verify、地址分析、表达式和五模块 Vectorize|
|文档1|§7|目标设计 §9|保留 E/P/Q、remaining、外部访存 mask，补 reducer mask|
|文档1|§8～§9|目标设计 §12\.3；路线图 §11\.2|输出约束入设计 §12\.3；分层测试与验收要求恢复为路线图 §11\.2（术语统一为 VL、分期合并、负例按最终设计修正）；后续优化和当前状态按 feature 去重|
|文档2|§1|目标设计 §1～§2|保留 RMSNorm 动机、IR 形态、压力和 UB fallback|
|文档2|§2|目标设计 §3；路线图 §3/§10|目标 pipeline 与实现任务分离|
|文档2|§3|目标设计 §6|完整保留 Fragment/slot 语义和 UB 预算|
|文档2|§4\.1|目标设计 §8|保留 reducer v2 前端、canonicalize、layout/materialize|
|文档2|§4\.3～§4\.4|目标设计 §8～§9|保留 finalize、seed、inactive\-tail、验证条件|
|文档2|§4\.5～§5|目标设计 §8\.7；路线图 §11\.3、§12\.5|reducer 用例表入设计 §8\.7；slot 存储与 pipeline 验收、RMSNorm 基线断言恢复为路线图 §11\.3；历史探针与模型检查入 §12\.5 并标明证据层级|

## 附录：自查结果（对照合并计划 §8\.3）

* [x] **PR\-1～PR\-6 与 Issue \#3 的口径**：里程碑表（§1\.4）、gantt（§1\.3）、各 PR 章节（§3～§8）的依赖、预计发起日期、ST 解锁用例和状态列均取自 Issue \#3 发布版；direct reduce 内容按 2026\-10\-10 设计定稿先行更新，cf4 归属按「验收形态保留 Fragment Buffer」从 M2 调整为 M4，两者已随定稿一次性刷新至 Issue \#3。

* [x] **PTOAS P1/P2/P3 依赖写清楚**：§9 表格与 §9\.1 问题说明完整保留；P1→M3、P2→M4、P3→M5 的阻塞关系在 §2\.1 Feature 表、各 PR 章节和 §12\.2 中一致标注。

* [x] **PR\-3 numeric blocked 与 PR\-4/PR\-6 Todo 状态真实**：PR\-3 写明「实现中，numeric 验证被 PTOAS P1 阻塞」（§5、§12\.1、§12\.2）；PR\-4、PR\-6 均为 Todo；PR\-5 为方案待设计；均未写成已完成。

* [x] **fork 实现参照与上游路线图编号分开**：文首编号约定、§10\.2、各 PR 章节均区分「上游 PR\-1～PR\-6（未发起路标）」与「fork PR \#262/\#269（OPEN，原始特性实现参照）」。

* [x] **inactive\-lane 保留语义有落点**：合并计划原要求跟踪 fork 迁移分支的修复同步状态；因 reducer v2 在当前基线重新实现、不依赖 fork 迁移分支，该项按作者决定改为语义要求与验收覆盖（§8、§12\.2），不再跟踪 fork 分支状态。

* [x] **ST 21 kernel、TopK、RMSNorm 的验收归属以 Issue \#3 为基础**：§11 全部表格（特性归属、逐 kernel 矩阵、路标解锁汇总）以 Issue \#3 §6 为基础，唯一偏差为 cf4 归属调整（M2→M4，验收形态含 Fragment Buffer、需 F\-FRAG），已随定稿一次性刷新至 Issue \#3；F\-R2 由 RMSNorm 产品用例验收、ST 套件无直接消费者。

* [x] **原文档验收材料已承接**：文档1 的分层测试与验收要求恢复为 §11\.2，文档2 的 slot 存储与 pipeline 验收表、RMSNorm 基线断言恢复为 §11\.3，历史探针与整数模型检查入 §12\.5 并标明证据层级（不等于编译器或设备验收通过）。

* [x] **未执行的编译、设备和数值测试没有写成通过**：§12\.5 明确区分「已执行」（pto\-dev 基线验证、P1 手工注入实验、失败复现记录）与「未执行/未落地」（T0–T8 各项）；手工注入 mem\_bar 的 9/9 通过标注为实验性验证，不代表 PTOAS 自动插入已落地。

