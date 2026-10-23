# TileLang Parallel 到 PTO：PR4 设计评审历程归档

> 本文件归档 PR-4 详细设计与 R4 子设计在各轮评审中的意见、核对结论与采纳记录。设计正文（两份）只保留最终事实与登记项；"哪一轮谁说了什么、改了哪里"的流程记录在此，不进设计文档。评审轮次与对应基线：

| 轮次 | 对象 | 基线 | 形式 |
|---|---|---|---|
| 一轮 2026-10-19 | PR4 旧稿（rebase 前 488 行） | `977425df`（PR-1 基线） | `pr4-fragment-slot-design-review-20261019.md`（R1–R8 + 文档/交付修正） |
| 二轮 | R4 v1 | PR-2 head `3e6a289f` 的前身提交 | R4-1 ～ R4-12（问题陈述与代码不符、奇偶图建模、区域求解路线等） |
| 三轮 | PR4（G1–G3 后）+ 代码 | `3e6a289f` | 甲 A1–A8 / 乙 B1–B5 / 丙 C1–C5（claude review） |
| 三轮半 | R4 v2 | — | M1–M10 及原 M2 遗漏（求解时机、注解收缩、多下标、合并点撤销等） |
| 四轮 | R4 v3 | — | N1–N7 + SimdVF 注解决策（opt-in、双侧区间、方向依赖节点等） |
| 五轮 | R4 v4 + PR4 并入 | — | P2-a/P2-b + 口径统一六项 + 非法测试用例等 |
| 六轮 | 两文档定稿清扫 | — | 2×P2 + P3 残留（A2 归属、PR-6 指引、G2 四处限定等） |
| 七轮 | UB 预留专项 | `3e6a289f` | U1–U6（预算生效范围、对齐口径、资源域、诊断形态等） |
| 九轮 | 全量代码评审（实现复查） | `bc715beb` 之上 7 提交，评审前 HEAD `7bc8d16f` | R4-1～R4-5 + 主体 P1×5/P2×4/P3 若干 + 测试/提交卫生（C5）——只读评审，4 路并行审查 |
| 十轮 | 九轮修复的复核 | 重构后 9 提交序列 | N1（P1）/N2（P2）/N3/N4（P3）+ 文档同步——只读复查 |
| 十一轮 | 缺口全量实现 | 14 提交序列（`9eba46e7` 一线） | R4 求解器/A1/A4/G2/Q4/J3/诊断批/容量验收——实现轮 |
| 十二轮 | 决策落地 | 16 提交（`fb42b4b0`） | bar 移除/U3 断言/确定性用例/设备补测/范围裁定——执行轮 |
| 十三轮 | 设计对照终审 | 17 提交（`fa146932`） | PTOAS rebase master+274、J4/J2 master 复测、注解/退化维补测、形态差异登记 |
| 十四轮 | J4 定案 | wrapper 修复 | J4/J2/退出码定案为环境侧 bug；全量 628/628 |
| 十五轮 | 全量评审修复 | 18 提交（`21a4d976`） | P1-2/P1-3 + G2/J3/口径/死代码批修；P1-1 待决策 |

## 一轮（R1–R8）采纳结论

- R1（backing 识别）：随 G1 落地为声明基识别（`CollectAllSlotBackingVars`），Plan 期识别走 plan 注解。
- R2（容量预算覆盖面）：root 扫描已修（`metadata.kernel_root` 起扫）；零/负余额构造性不可达；固定分配/无 eligible buffer 的检查仍开放（并入 U1 影响面）。
- R3（最终容量检查路径）：Merge 末尾带门槛断言（声明基）；无 contract 跳过、常量另抄、区间核验未做（开放）。
- R4（布局联合推导）：立为子设计，历 v1→v5。
- R5（tail∧predicate 写回）：chunk mask + G3 谓词合成解决。
- R6（fill/copy 按 region）：G2 前置展开解决（后限定：仅对齐子区间）。
- R7（布局等价/view/生命周期）：晚期识别已修；IsEqual 显式判据与 view 对应随 R4。
- R8（声明→memref 集成）：G1 解决并设备复验；优化关闭配置未验（开放）。

## 二轮（R4-1～R4-12）采纳结论

问题陈述按真实失败模式重写（选维证明与 Verify 值分类均按逻辑行主序——"lane 只能落末维"且取向一致也被拒）；奇偶图建模 + 区域级求解路线（规避引擎分级/合并点/free-mode 三处冲突）；多下标收集（后收紧为 opt-in）；IsEqual 引证修正（LayoutNode 重载只比 shape+forward_index）；外部布局来源（annotate/floating/alias/TileOp）；无依赖访问判不可行；PR-6 边界（后修订为方向依赖节点）；提交切分（证明退出与求解同提交）；测试缺口。

## 三轮（A/B/C）核对结论

18 条：15 完全属实、3 大部分属实。要点：A1 mem_bar 覆盖矩阵（Unknown 只记录、mem-to-value 仅常量 offset）；A2 区域外无拒绝路径（→R4 前扫描）；A3 G2 与 PR-6 时序冲突 + ascend_copy 漏门；A4 entry 区间（→R4 双侧）；A5 G3 作用范围（决策：限定 fragment）；A6 Plan 无规范校验（→R4）；A7 容量/独占覆盖；A8 测试缺陷（S:341 恒真断言等五处）；B1 末维限制（→R4）；B2 [VL] 累加器 carry 分派不一致；B3 roundtrip 消除作用于 fragment（决策：关闭）；B4 多下标 ICHECK（→R4）；B5 floating FATAL（→R4）；C1 文档与代码不符（版本链 ICHECK 不存在等，已就地修正）；C2 工具链契约（J3 归因更正为主线 `2729bb556`；MR 未合入声明）；C3 验收口径（M4 新解锁 6 家族）；C5 提交重排。

## 三轮半（M1–M10）、四轮（N1–N7）、五轮（P2-a/b + 并入）

M 系列：求解时机移至 Run() 收集完成后、注解收缩为仅规范形态、多下标机制、撤销合并点改动（可达性论证）、同 data 确定性拒绝、entry 区间、并列规则程序序化、VF 外访问前置诊断、方向反推、杂项。N 系列：opt-in（默认严格）、双侧区间+全证明域、方向依赖节点（非 pinned）、写侧措辞、前扫描自遍历、判定口径、诊断恢复。P2-a：开关为构造参数+节点成员+ExpandBindValues 同源+逐循环 `InsidePtoSimdVF()`；P2-b：两处证明域绑定。五轮另并入 PR4 总设计（口径统一六项、B4/B5/A6 登记、G2 对齐限制、A5/B3 决策、J3 合入前置）。

## 六轮（定稿清扫）

2×P2：A2 归属统一到前扫描/R4 步骤②；§8.1 A3 与 §2.1 口径一致。P3：A4 双侧、G2 四处限定、第 62 行两句、重叠自拷贝定 Expand 检查、负例报文登记、提交计划两补充（上游序列要求 + 非 R4 修复落点 + G2↔A5 先后）、措辞四处；R4 侧删旧句/SimdVF 拼写/§4-15 补 SimtVF/外层循环含 kernel 级/Run() 残留/§3.7 澄清/PR-6 护栏/升 v5。

## 七轮（UB 预留专项，U1–U6）

U1（原 P1 后降 P2）：memory_limits_ 仅在循环调度内使用（约束版本选择）；直线列表无版本决策、预算无约束对象，物理容量由 Merge 终检保证——非有效性缺陷，并入 U5/R3；U2：预算不含对齐 padding（差额上界 31N）；U3：多 kernel root 时预算与终检口径不一致；U4：B1→entry 32× 膨胀→预留失败（报错面与根因错配）；U5：ICHECK 形态 + unlimit 指引误导；U6：HasSimtVF 探测器双份。核对确认：算式/扣减时机/去重/不重复计数/溢出风险低/32B 对齐当前恒无操作。容量四用例 + 逐 backing 区间断言升为第一优先。

## 八轮（rebase 影响，2026-10-21）

Rebase 到 `pto-tparallel-control-flow`@`bc715beb`（59 提交，含重写的 M1/M2 + M3 谓词化 Legalize）。影响：Q1（P0）HEAD 编译失败——base `predicated` 与 G3 `predicated` 同名冲突（rebase 文本合并无冲突标记，静默引入）；Q2（P1）fragment 分支丢条件写 predicate——base Legalize 已谓词化、不再产出 Select，G3 模式匹配成死代码；Q3 G3/A5 表述全部失效；Q4（P2）SubcoreProjector 不认识 slot plan；Q5 基线/行号锚点过期。修复：删 G3 Select 恢复、统一谓词消费（fragment/UB 共用 mask_and 合成）；144 结构 46/46 复验通过。

## 九轮（全量代码评审，2026-10-22，采纳全部处理）

只读评审覆盖 R4 求解器、Vectorize 存储模块、UB 预留链路、Expand/Verify/codegen、测试与提交卫生（C5）。

**R4 部分（P0×2）**：R4-1 求解器恒不运行（默认构造 `Target()` 判空短路）；R4-2 边编码矛盾（每方向各 Union 一次，`f[i,j]` 即 parity conflict）+ Pin 零点用错 + 前扫描/物化核对/`PtoRequiredVectorDim`/`AnalyzeLaneUse` 均未实现（R4-3）；R4-4 区间证明未绑外层范围（`f[k,j]` 回归拒绝）；R4-5 多下标容器只写不读、放宽波及 UB。**决定：求解器撤出本 PR，按 R4 子设计重写后独立合入；保留收集（限 fragment）与双侧区间（含外层范围绑定——Verify 绑区域 wrapper 与单元 shell，Vectorize 用 `EnclosingSerialRanges` 复现）**。

**主体 P1×5**：①`CheckEntryOffset` 只认 `x*L` 精确形态，`(i*Q+q)*L` 化简后崩溃（N>VL 2D fragment 不可用）——改整除证明；②Merge 把 base 有意的 `disable_reuse=false` 改回参数回传（AIV 独立规划被绕过）——恢复 `false`；③重叠自拷贝无检查（静默算错）——展开 pass 补分类拒绝；④B3 按整个区域关往返（同区域 UB 临时失去保护）——改按 buffer 门控 + load 侧 `is_fragment`；⑤B2 只修区域级（单元 shell 级误导报文 + 未写 vreg 写回）——`AnalyzeLoopShell` 排除 fragment + `EmitCarryWriteback` 防御断言。

**其余**：P2（A2 未关闭、Q4 未处理、store-predicate 声明收集盲区、跨 VF 报文误导）与 P3（A6 注释/形态、codegen 地址断言、后置检查弱化、对齐旁路、Merge 终检 dtype.lanes、展开丢注解、清理不完整、U5 文案）除 A6 注释勘误外维持登记。**测试**：断言缺陷 5 处修复 + 新增评审回归用例（多 chunk、重叠/不相交自拷贝、entry 平移正负例、UB 往返共存、shell 累加器、外层串行正负例、设备多 chunk）。**C5**：重构为 9 提交（builtin 前置、Verify 放开与转换同提交、pipeline 注册与实现同提交、路由断言随注册翻转；求解器拆出），终树与修复后工作树逐字节一致。

## 十轮（二轮只读复查，2026-10-22 晚）

九轮修复落地后的复核：确认上轮 P1×5、R4 撤出、R4-4/R4-5、提交卫生（逐提交符号核对）、测试缺陷修复全部到位。**新发现四项**：N1（P1）B2 单元级修复使 fragment shell 累加器能通过转换，但跨迭代同 entry RAW 无保序（bar 只在整个单元后）——修法取"写 backing 的 shell 每迭代末尾 VST_VLD"；N2（P2）vecscope split 提升循环后 part 内 `EnclosingSerialRanges` 看不到提升变量，`f[t,i]` 在 Verify 可证、Vectorize 报错——修法为 `split_loop_ranges_` 随 part 转换传入；N3（P3）重叠检查的 box 测试不含视图 elem_offset，同 data 不同视图改保守拒绝；N4（P3）常量 offset 分支缺整除粒度检查。文档：§8.1 状态表同步（B1/A2/B5 → 待求解器重写；A4/B4/A6 → 已关闭并注明窄范围/形态；B2 附 N1）、头部补开放项、U6 块名勘误（SIMT_VF）、门槛 B2 行改写、归档九轮基线更正、R4 文档 M10 引用与 pipeline 行号清理、路由/设备测试 docstring 如实化。

## 十一轮（缺口全量实现，2026-10-22 深夜）

十轮提出的"实现与设计差距清单"全部落地为提交 10–14：R4 求解器按 v3 重写合入（修正首轮三处致命错误：入口 target、单边编码、虚拟零点；B1 旗舰/B5 floating/A2 前扫描关闭，结构+设备双验证）；A4 扩展到 kernel 级循环；A1 补跨单元 WAR（VLD_VST），覆盖矩阵闭合；G2 未对齐 fill/copy 谓词化展开（同余 fragment↔fragment 可处理、UB↔fragment 未对齐拒绝、注解化调用拒绝）；Q4 投影 kernel 分类拒绝；J3 按 PTOAS !293 迁移契约改组合发射（brc/dintlv/intlv 三形态）；诊断批（A6、U5、预算、动态 shape、legality 整 entry 证明、codegen 断言、Merge dtype.lanes、kernel 作用域清理、predicate 声明收集、跨 VF 报文如实化）；容量验收面补齐（四用例 + 最终合并模块上的逐 backing 字节区间不相交直接断言，两调度路径）。验证：结构 36/36（含 8 项求解器用例）+ 容量 5/5 + 设备 slot 18/18（含 B1 转置形态与未对齐 fill 数值）；全目录与 base 回归随本轮收尾跑。

## 决策记录（2026-10-22，工具链口径）

PR owner 决策：**PTOAS 以当前 `pr274-pr307-merge` 分支版本为准**；MR274 与 PR307 均确定最终合入上游，PR-4 不以二者上游时点为阻塞。最低版本声明为含 MR274 + PR307 + dist_mode 移除（PTOAS !293）的第一个发布版本；旧 wheel（vmi-0.1.9）不再作为组合发射的兼容目标；上游发布后在该版本上复验一次设备面。据此，此前决策清单中的第 3 点（版本声明策略）与第 4 点（旧 wheel A/B）关闭。

## 十二轮（决策落地，2026-10-23）

十四项决策点全部裁定后执行：保守 mem_bar 移除（PR307 委派，工具链钉合并 lineage，shell 累加器无 bar 数值为证）；U3 单 root 断言；求解器确定性三用例（字节最小 4+4 逐值、并列 forward_thread 断言、1D pin dim0）；设备补测（[4,136] 三 chunk 尾部、shell 累加器）。范围裁定：shell vreg 化/单元素提取/2D 未对齐/投影改写 plan 不属 PR-4。J4/J2/fmaximum/动态 offset 转发/PR307 Unknown 形态已列 PTOAS 问题清单转发相关开发。144 复验与推送按用户指示搁置（PR 为临时）。

## 十三轮（设计对照终审，2026-10-23）

PTOAS rebase 至 master `d30269595`（PR307 已上游）+ MR274，全量 624/625 复验；J4/J2/退出码三问题在 master 上复测确认仍存（证据链更新转发）。设计对照终审：功能缺口清零；注解 pinned/形态不符/退化维补测；UB 多下标旧崩溃描述证伪（收集器只记录 fragment）；SimdVF 级注解经冲突路径 fail-closed。形态差异登记：非规范注解/SimdVF 注解/A6 报文形态、M9 共享 helper 与物化 ② 的等价偏差、同 data 视图防御路径不可测。

## 十五轮（全量代码评审修复，2026-10-23）

Claude 全量只读评审（17 提交 + PTOAS PR307 源码）确认 3 个 P1：P1-1 bar 移除后三类形态（>2 层嵌套、vecscope 外变量 offset——恰为本 PR 新增的 k×multi-chunk 2D / kernel 级 f[b,j] / split f[t,i]）在 PR307 只记 Unknown 不插 bar，现有设备全绿是幸存者偏差（修复方案待用户决策：区域门控/全量恢复/拒绝）；P1-2 求解器 DSU 重挂根丢约束 → 构建不可复现（已修：值迁移 + 冲突来源）；P1-3 组合发射丢 group → 静默算错（已修：stride-group 加载 + vbrc(group=)）。同轮修复：G2 未对齐门控收窄到 fragment 侧、J3 三处、T.unroll 绑定口径、死缓冲容忍、显式字段比较、no-contract 终检、死代码清理。另定案 J4=环境侧 wrapper bug、J2=CANN 版本要求。提交信息错位已改写。

## 附：ST 套件 annotate_layout 排查（R4 §3.0③ 的执行记录）

`peanutchan/tilelang-ai`@`593a8f0c`，全部 21 个 kernel + vf_mode.py + 两个 harness（PR diff 取完整文件、逐文件行数与 GitHub additions 一致）：`annotate_layout`/`T.Fragment(`/`make_*_layout`/`layout_map` 零命中——迁移删除清单为空。head 变化时按同方法复查。
