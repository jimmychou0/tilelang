# PR-4 Fragment/Slot 存储实现走读

> 按编译流水线顺序走读整个 PR（jimmychou0/tilelang#2；slot storage 部分为单提交 `afe123bc`：27 文件、+7941/-46，随 Ascend-owned 并行栈一并提交），每一步配真实 IR dump、示意图、流程图和**代码位置**（`文件:行号`，行号以当前 head 为准，跳转请认函数名）。所有 dump 均在 238 上实测取得，可复现（方法见附录）。

---

## 0. 这个 PR 做什么

`target="pto"` 下，`T.SimdVF` 内的向量化以前只认 shared buffer；用户 `T.alloc_fragment` 分配的缓冲被直接拒绝。本 PR 给 fragment 一套完整的物理实现：

**在 shared.dyn 里给它一块独占的专属空间（slot backing），编译期决定数据摆放布局，最终以 memref 形式交给 PTOAS。**

最典型的受益者是"累加器要跨循环存活"的写法（gym sv3，GEMV 部分和）：

```python
acc = T.alloc_fragment((24, 128), "float32")          # 累加器
for k in T.serial(K):                                  # 外层串行
    for i, j in T.Parallel(24, 128):                   # 向量化的并行单元
        acc[i, j] = acc[i, j] + a_ub[k, i, j] * x[k]   # 读写同一块 fragment
```

贯穿全文就用这个例子（下文 dump 时缩小为 `[8,128]`、K=4 便于阅读）。

### 设计难点地图（每个都有专节）

| 难点 | 一句话 | 专节 |
|---|---|---|
| R4 布局联合推导 | 布局是全局联立的（转置锁死两块 buffer 的选择），奇偶并查集 + 三件事证明选方向 | §2 |
| UB 预留与独占分配（D3/D4） | fragment 挤进已被管理的固定资源：独占不复用、预算先扣后调度、终检全路径、地址 Merge 统一分 | §3 配套难点节 |
| G2 fill/copy 展开 | 算子调用归一成循环；未对齐 1D 谓词化；多维不判对齐（判定资格问题） | §1 |
| 准入证明（A4） | 三件事证明 + 绑全部外层范围（含 kernel 级、穿 if）；一个证明两个消费者 | §4 |
| 保序责任划分（G4） | TileLang 侧 0 条 bar，整体委托 PTOAS（缺口 #1679→PR374 已修复验证） | §5"刻意不做" |

### 数据形态一路怎么变（总览）

```
alloc_fragment + Parallel 循环          ← 用户写的
  ↓ ② fill/copy 拆成循环
  ↓ ④ 布局求解: layout_map 说 lane 来自哪一维
  ↓ ⑥ 规划: shared.dyn 划专属 backing，写 tl.pto_slot_plan
  ↓ ⑧ 证明: 每个访存过三件事检查
  ↓ ⑩ 转换: Bind(handle, declare_fragment_buffer(...)) + vload/vstore
  ↓ ⑪ 代码生成: pto.memref_declare(...) + pto.vmi.vload/vstore
PTODSL → PTOAS（自动插 vecscope membar）→ 设备
```

### 文件清单（28 个，按流程）

| 阶段 | 文件 | 规模 | 职责 |
|---|---|---|---|
| 语句注册 | `src/ascend/op/builtin.h` | 小改 | 注册 declare_fragment_buffer（[builtin.h:169](src/ascend/op/builtin.h#L169)，随自有化重构移至 ascend 侧） |
| ② 前置展开 | `src/ascend/transform/expand_pto_fragment_fill_copy.cc` | +436 新文件 | fill/copy 拆成循环 |
| 访存收集 | 本 PR 各 pass 自有 | — | 多下标收集由 solver `AccessScanner`（[solver:85](src/ascend/transform/pto_fragment_layout_solver.cc#L85)）与 vectorize `AccessCollector`（[vectorize:525](src/ascend/transform/vectorize_parallel_to_pto.cc#L525)）完成；上游 ParallelOp 不再改动（保持严格默认） |
| ④ 布局求解 | `src/ascend/transform/pto_fragment_layout_solver.{h,cc}` | +1399 新文件 | 决定哪维进 lane |
| 求解挂载 | `src/ascend/transform/layout_inference.cc` | +50 | 注入最终布局 |
| ⑥ 规划 | `src/ascend/transform/plan_pto_slot_storage.cc`、`pto_slot_plan.h` | +592 新文件 | 划空间记账 |
| 预算联动 | `auto_schedule.cc`、`subcore.h`、`infer_buffer_aliases.cc`、`merge_ub_allocations.cc` | +225 | 四处账本（见 §3） |
| ⑧ 证明/校验 | `src/ascend/transform/parallel_to_pto_utils.{h,cc}`、`verify_parallel_to_pto.cc` | +556 | 三件事证明 + 安检 |
| ⑨ 收尾安检 | `src/ascend/transform/verify_pto_codegen_legality.cc` | +97 | 代码生成前最后一道 |
| ⑩ 转换 | `src/ascend/transform/vectorize_parallel_to_pto.cc` | +825 | 改写程序（Storage 模块） |
| ⑪ 代码生成 | `src/ascend/codegen/codegen_pto.{h,cc}` | +84 | 打印 PTODSL |
| 流水线 | `tilelang/ascend/pipeline.py`、`transform/__init__.py` | +41 | 注册两个新 pass |
| 测试 | 4 个测试文件 | +3035 | 结构/容量/设备/ABI |

---

## 1. ② ExpandPtoFragmentFillCopy：把算子调用归一成循环

**代码位置**：pass 主体 [expand_pto_fragment_fill_copy.cc:216（RegionExpander）](src/ascend/transform/expand_pto_fragment_fill_copy.cc#L216)；判定 [IsFillOrCopyOp:59](src/ascend/transform/expand_pto_fragment_fill_copy.cc#L59)、[TouchesFragmentBuffer:47](src/ascend/transform/expand_pto_fragment_fill_copy.cc#L47)；谓词化 [BuildPredicatedFill:172](src/ascend/transform/expand_pto_fragment_fill_copy.cc#L172)、[BuildPredicatedCopy:191](src/ascend/transform/expand_pto_fragment_fill_copy.cc#L191)；分流 [TryUnalignedFill:300](src/ascend/transform/expand_pto_fragment_fill_copy.cc#L300)、[TryUnalignedCopy:322](src/ascend/transform/expand_pto_fragment_fill_copy.cc#L322)；重叠自拷贝检查 [CheckSelfCopyRegionsDisjoint:119](src/ascend/transform/expand_pto_fragment_fill_copy.cc#L119)。

**动机**：`T.fill(acc, 0)` / `T.copy(a, acc)` 在 TIR 里是算子调用，而后面所有机制（布局求解、证明、转换）只认"T.Parallel 循环"这一种形态。与其在每个环节开特例，不如在最前面归一。顺带让 fill/copy 调用不进调度器。

**只管碰 fragment 的**：纯 UB 的 `T.copy(A, a_ub)` 原样放行（走 base 的普通搬运 lowering），对非 PTO 程序零影响。

### 变之前 → 变之后（真实 dump）

```python
# ① 整块 fill（sv3）
T.fill(acc, T.float32(0.0))            for i in T.parallel(8):
                                            for j in T.parallel(128):
                                                acc[i, j] = T.float32(0.0)

# ② 对齐区间 copy
T.copy(a_ub[0:128], keep[0:128])       for i in T.parallel(128):
                                            keep[i] = a_ub[i]

# ③ 未对齐区间 fill（起点 64 不是 128 的倍数）
T.fill(keep[64:192], T.float32(7.0))   for i in T.parallel(256):
                                            if i >= 64 and i < 192:
                                                keep[i] = T.float32(7.0)
```

未对齐的关键技巧（`BuildPredicatedFill`）：**循环基址向下对齐**（base = 64-64%128 = 0），**原区间变成 if 谓词**——之后 Legalize 把谓词降成 lane 掩码，恰好 entry 0 的 128 条 lane 里 0–63 关、64–127 开。地址永远整 entry，"起点不在边界上"这个信息从地址挪进了掩码。

### 流程图

```
语句是 fill/copy/ascend_copy 吗？──否──→ 原样放行
   │是                                    （前端会把 SimdVF 内的 T.copy
   ↓                                      降成 typed ascend_copy，都要认）
碰到 local.fragment 了吗？──否──→ 原样放行（base tileop lowering）
   │是
   ↓
调用带注解吗？──是──→ 分类拒绝（路由/cache hint 循环形态承载不了）
   │否
   ↓
┌─ Fill ─────────────────────────────┐  ┌─ Copy ────────────────────────┐
│ 1D 且 fragment 侧起点未对齐？         │  │ 同一 buffer 既源又目的？        │
│  是→谓词化展开（对齐+if）             │  │  重叠/证不出不相交→拒绝          │
│  否→MakeSIMTLoop 普通展开            │  │ 两侧 1D？                      │
└────────────────────────────────────┘  │  多维→MakeSIMTLoop（不判对齐，  │
                                         │        交给 lane 证明裁决）     │
                                         │  目的 fragment 侧未对齐？        │
                                         │   源同 fragment 且 min 同余       │
                                         │    →谓词化 copy（共享平移+Cast）  │
                                         │   否则→拒绝（UB 侧会越界读）      │
                                         │  都对齐→MakeSIMTLoop            │
                                         └────────────────────────────────┘
```

### "1D / 多维"的判定资格问题

fragment 的访存必须落在整 entry 上（一个 entry = 128 lane 一起动）：

- **1D** region `keep[64:192]`：min=64 是**元素在 lane 空间的位置**，`64 % 128 ≠ 0` 说明起点在 entry 中间——本 pass 有资格判断；
- **多维** region `acc[1, :]`：dim0 的 min=1 是**行号 = entry 编号**，"第 1 行"天生是完整 entry；拿 1 做 `%128` 是把行号当 lane 偏移，会错杀合法程序（`test_pto_slot_plan_structure.py::test_fragment_2d_row_fill_compiles` 钉的就是这个）。

多维一律不做对齐判断，交给 ⑧ 的 lane 证明 fail-closed。另外哪一维是 lane 维要等 ④ 才知道，此刻也判不了。

---

## 2. ④ 布局求解器：哪一维放进 128 条 lane

**代码位置**：入口 [pto_fragment_layout_solver.cc:514（FragmentLayoutSolver::Run）](src/ascend/transform/pto_fragment_layout_solver.cc#L514)；UB 连续性投票 [ProbeDirection:153](src/ascend/transform/pto_fragment_layout_solver.cc#L153)；三件事证明 [FragmentDirectionFeasible:251](src/ascend/transform/pto_fragment_layout_solver.cc#L251)；约束构建 [ConstrainUnit:877](src/ascend/transform/pto_fragment_layout_solver.cc#L877)；奇偶并查集 [ParityDSU::Unite:406](src/ascend/transform/pto_fragment_layout_solver.cc#L406)；规范布局构造 [BuildCanonicalLayout:450](src/ascend/transform/pto_fragment_layout_solver.cc#L450)。头文件 [pto_fragment_layout_solver.h](src/ascend/transform/pto_fragment_layout_solver.h)。

### 问题本质

`[8,128]` 的 fragment，两种摆法都合法，代价不同：

```
选法 A：j 维当 lane —— 8 个 entry（紧凑）      选法 B：i 维当 lane —— 需补齐到 128
       j→ 0………128                                  j→ 0………128
   i=0 ┌────────────┐  entry 0                i=0 ┌──┐
   i=1 │  一整行     │  entry 1                i=1 │  │  entry 0 由 i=0..7
   ⋮   └────────────┘                          ⋮   │  │  的 128 个采样点
   i=7                                            i=7└──┘  组成（含补齐）
```

但不能各 buffer 贪心——选择是**全局联立**的（转置例子见下）。求解器四步：收集区域 → 前扫描拒绝/钉死 → 建约束 → 求解输出。

### 概念：方向与投票

2D 单元 `for i, j in T.parallel(8,128)` 有两个**候选方向**：j 当 lane（方向 0）或 i 当 lane（方向 1）。每个方向要过两道投票，**任何一个访存投反对票该方向就作废**：

**投票 A：UB 连续性**（`ProbeDirection`）。把被选变量写成 `q×128 + lane` 代入 UB 下标，证"地址 = 起点 + lane"（一轮 128 条 lane 读一段连续地址）：

```
方向 0（j=lane）: 地址 = k*1024 + i*128 + lane     lane: 0,1,2,…,127 → 连续斜带 ✅
方向 1（i=lane）: 地址 = k*1024 + lane*128 + j     lane 各取相隔 128 的"梳子" ❌
```

sv3 的求解判词 `forward_thread = _j` 就是这么来的——不是偏好，是 UB 访问只有一种切法喂得饱向量访存。

**投票 B：fragment 三件事证明**（`FragmentDirectionFeasible`）。假设按候选方向摆，逐访存检查：

1. **lane 匹配**：存储位置 == 执行位置。`f[i, j*2]` 在 j 方向下存储位置 = 2j mod 128 ≠ j ❌；但在 i 方向下 lane=i ✅、entry=2j 也在界内 ✅——**换个切法程序就活了**，所以可行性必须在选方向时证明，不能选完让 Verify 毙掉（误拒）；
2. **entry 与 lane 无关**：一轮之内 128 条 lane 必须同属一个 entry（否则一次 vload 跨 entry）；
3. **entry 双侧区间**：绑上全部外层范围后证 `0 ≤ entry < count`。

**关键实现细节**：方向作废用 `continue` 不是 `break`——另一方向还要继续评估并记录约束（漏记会让 fragment 失去约束、字节最小选出与单元矛盾的方向；`f[i,j*2]` 无 UB 版是回归用例）。

### 产出三种东西

投票后每个单元产出：

1. **一元约束**（直接钉死）：只剩一个方向活 → 钉单元方向；两方向殊途同归（1D fragment）→ 只钉 fragment；用户注解 → pinned；
2. **奇偶边**（变量间的锁链）：两方向都活、且分别把同一 fragment 钉到**不同** lane 维时。经典例子 `g[j,i] = f[i,j]`（无 UB）：

   ```
   方向0 ⟹ f=dim1, g=dim0        方向1 ⟹ f=dim0, g=dim1
   不管单元选哪边，f 与 g 永远相反 ⟹ 记 XOR 边：f⊕u=1, g⊕u=0
   （单元保持自由，但两块 buffer 必须一横一竖）
   ```

   `Solve` 用带权并查集沿链做 XOR 传播；奇环（绕一圈推出"f 与自己反向"）= **已证明冲突**，分类拒绝；
3. **自由分量**（没被锚定的变量群）：**字节最小**选根——`count = 其余维积 × ceil(lane维/VL)`，`[128,4]`@VL=128 取 dim0（4 entry）弃 dim1（128 entry）；**仍并列** → 源码序第一个自由单元取内维（保证确定性/可复现）。

### 流程图

```
Run(body)
 ├─ ① CollectRegions   拆单元、收集访存、记录 fragment、收集外层范围(含kernel级、穿if)
 ├─ ② PreScan          注解校验为pinned / 区域外访问拒 / 同data拒 / 跨区域拒 / 无契约TileOp拒
 ├─ ③ BuildConstraints 逐单元：UB连续性投票 → 三件事投票 → mask 幸存方向
 │                      两方向全死→"无可行方向"拒；幸存→一元约束 或 奇偶边
 ├─ ④ Solve            XOR传播；奇环→冲突拒；自由分量→字节最小(+源码序破并列)
 └─ ⑤ EmitLayouts      造规范布局注入 layout_map（注入即最终答案，引擎 step0 跳过）
```

---

## 3. ⑥ PlanPtoSlotStorage：量地、发证

**代码位置**：pass 主体 [plan_pto_slot_storage.cc](src/ascend/transform/plan_pto_slot_storage.cc)；count 读取 [EntryCountFromLayout:377](src/ascend/transform/plan_pto_slot_storage.cc#L377)；记录结构 [pto_slot_plan.h:49（PtoSlotPlanNode）](src/ascend/transform/pto_slot_plan.h#L49)、对齐常量 [kPtoSlotAlignmentBytes:37](src/ascend/transform/pto_slot_plan.h#L37)。

Stage 4 定了布局，但 fragment 还是逻辑对象。本 pass 是**土地测量员兼会计**：只添账本，不动程序一行。

造两样东西（sv3 dump）：

```python
acc_slot_backing = T.sblock_alloc_buffer((1024,), scope="shared.dyn")  # 土地（物理身体）
T.sblock_attr({..., "tl.pto_slot_plan": [...]})                        # 证照
```

**backing 是 `shared.dyn`**——只有 shared.dyn 进 MergeUB 统一合并与容量核算，这是可被地址 pass 改写、进容量体系的前提。

**PtoSlotPlan 字段与消费者**：

| 字段 | sv3 的值 | 谁读 |
|---|---|---|
| fragment / backing | acc / acc_slot_backing | 所有人（对象级对应） |
| count | 8 | Vectorize 声明、容量检查 |
| vector_lanes | 128 | Vectorize（vload size） |
| fragment_mem_var | handle 名 | Vectorize 每区域造全新同名 Var |
| vector_type_name | "float32x128" | 校验访问类型一致 |
| bytes（32 对齐） | 4096 | AutoSchedule 预算、Merge 容量 |

**count 公式**（`其余维积 × ceil(lane维/VL)`，永不用 元素数/VL）——ragged 形状必须按 entry 分配：

```
f = [4,130]，VL=64，lane 维=dim1：
  count = 4 × ceil(130/64) = 12     ← 第 3 个 entry 只用 8 lane，也占整格
  错误算法 4×130/64 = 8.125→8       ← 少算 4 个 entry，越界写进邻居
```

**位置**：在 AutoSchedule **之前**——预算扣减必须先于"普通 buffer 能否双缓冲"的决策。**本 pass 只登记不分地址**：具体字节地址（sv3 PTODSL 里 `memref_declare(20480,...)` 的 20480）由 MergeUB 统一分配，保证所有 shared.dyn 两两不相交（容量测试直接从最终模块的声明地址断言）。

防护：A6 规范形态断言（恰一 entry 维 / replicate==1 / 常量正 thread extent——怪布局进来 count 会算错）；count×lanes 溢出检查；一块 fragment 只发一张证。

### ⑥配套难点：UB 预留与独占分配（D3 预算 + D4 独占）

**代码位置**：预算扣减 [auto_schedule.cc:1413-1422（slot_bytes 汇总）](src/ascend/transform/auto_schedule.cc#L1413)；容量终检 [merge_ub_allocations.cc:579（CheckPtoSlotUnifiedBufferCapacity）](src/ascend/transform/merge_ub_allocations.cc#L579)，no-contract 路径调用 [:666](src/ascend/transform/merge_ub_allocations.cc#L666)、合并后调用 [:746](src/ascend/transform/merge_ub_allocations.cc#L746)；验收用例 `testing/ascend/language/pto/test_pto_slot_capacity.py`。

这是本 PR 另一个单独设计过的难点（和布局推导并列），核心矛盾：**fragment 要挤进一块"已经被管理的固定资源"**。

UB（约 216–248KB）不是空地，现有体系已经在管理它：AutoSchedule 按剩余容量决定普通 buffer 能否双缓冲，MergeUB 把 shared.dyn 分配合并进统一地址空间。fragment backing 破了两个规矩：**值要跨循环存活 → 不能参与生命周期复用**（必须独占）；**布局一确定（④）就知道要多大，但地址要到 Merge 才落**——中间隔着调度，调度器若不知道地已被占，会一地两许。

**四个核心决定：**

1. **独占、不参与 lifetime reuse**：TileLang 证不了"复用安全"（需要生命周期不相交的访问分析）；独占最多浪费容量、不会地址冲突——正确性优先，浪费由预算控制；
2. **预算扣减放在调度之前**（Plan pass 的位置就是为此）：`剩余容量 = 物理上限 − Σbacking.bytes`，双缓冲决策基于真实余量。否则调度器按满额批双缓冲，加 backing 超容量——到 Merge 才炸，报错点离决策点十万八千里；
3. **预算不是有效性保障，物理容量由 Merge 终检兜底**：预算的作用面是"循环调度的多缓冲版本选择"，直线 kernel 没有版本决策、预算无对象（不是缺陷）；真正的容量保证是 Merge 末尾终检——且**全路径**（无别名合同的早退路径曾跳过，已修）；超限报分类错误带可操作建议（减小 fragment/tile）；
4. **地址由 Merge 统一分配，验收从最终模块断言**：Plan 只登记大小；"所有 shared.dyn（普通+backing）两两不相交"一次性全局解决。容量测试**不查中间状态**，直接从最终合并模块的声明地址（access_ptr 的 offset/extent）断言逐 backing 字节区间两两不相交——手动/自动两条调度路径都验，另断言 backing 基址非零、合并总量不超上限。

**三个联动陷阱（难点的真正藏身处）：**

- **布局取向 × 预算错配（B1→U4）**：`[256,4]`@VL=128 lane 落错维 → 32× entry 膨胀 → 预留失败，但报错是"容量不足"、根因是"布局取向"，用户无从下手。解法在布局侧：求解器**字节最小取小**从源头不再产生膨胀布局——这正是该规则的动机之一；
- **ragged count**（前述 12 ≠ 8）：预算按 entry 算才对；
- **预算与终检口径（U3）**：预算按 kernel root 分组、终检函数级——多 root 口径分裂，解法是单 root 前提断言（AutoSchedule 拒绝多 tilelang_root 的 slot kernel）。

**专测场景 ping/pong**：软件流水多轮 live range 不同，普通 buffer 按轮复用、backing 贯穿全程——最终模块必须**恰好一次声明**（容量测试钉住）。

```
④ 求解器：字节最小选解 ──→ 膨胀布局从源头不产生
⑥ Plan：count×lanes×elem → bytes(32B对齐) → 登记（只记大小，不分地址）
AutoSchedule：剩余容量 = 物理上限 − Σbytes → 双缓冲决策基于真实余量
MergeUB：统一地址空间排地址（backing 独占，不参与复用）
容量终检（全路径含 no-contract）→ 超限分类报错
最终模块断言：逐 backing 字节区间两两不相交（两条调度路径）
```

---

## 4. ⑧ 证明与校验：一个证明，两个消费者

**代码位置**：证明本体 [parallel_to_pto_utils.cc:639（AnalyzeFragmentAccess）](src/ascend/transform/parallel_to_pto_utils.cc#L639)；UB 侧同框架 [:489（AnalyzeBufferAccess）](src/ascend/transform/parallel_to_pto_utils.cc#L489)；范围收集 [:93（CollectEnclosingSerialRanges）](src/ascend/transform/parallel_to_pto_utils.cc#L93)；消费方之一 [verify_parallel_to_pto.cc:1073 附近（fragment 分支）](src/ascend/transform/verify_parallel_to_pto.cc#L1073)、另一消费方即求解器的 `FragmentDirectionFeasible`。

两个文件一个职责：`AnalyzeFragmentAccess`（证明本体）+ `VerifyParallelToPTO`（跑证明的安检门，以前"见 fragment 就拒"，现在真正逐访存跑证明）。

两条铁律：**证不出 = 拒绝**（fail-closed）；**双侧**（上下界都要证）。

**道具 1 代换**：被选变量 = `q×VL + lane`，代换后用布局正向映射换算出"存储 lane / entry"两个表达式，对它们做证明。

**道具 2 证明域**：lane / q / 保留维 / **全部外层串行循环范围**（`CollectEnclosingSerialRanges` 收集，穿 if/while）。不绑范围，`f[k,j]` 的 entry<k 就证不出——历史上两类真 bug：不绑 kernel 级循环（误拒 f[k,j]）、范围收集不穿 if（for k: if…: f[k,j] 误拒）。

**三件事证明**（真实报文示例）：

| 证明 | 问什么 | 反例与真实报文 |
|---|---|---|
| (a) lane 匹配 | 存储位置 == 执行位置 | `keep[i, j+1]`：j=127 时存储 lane=0≠127。方向 j 死于此、方向 i 死于 entry 区间 → 报"no feasible direction"（真实报文） |
| (b) entry 无关 lane | entry(lane=0) == entry(lane) | 一轮内 lane 被劈到两个 entry，一次 vload 表达不了（保险丝） |
| (c) entry 区间 | 0 ≤ entry < count | `keep[k,j]` k∈[0,8) 而 f 只有 4 行：真实报文 "the slot entry index (k) is not provably within [0, 4)"；`f[i+128]` 多 chunk：entry=q+1 在 q=1 时 = count ❌ |

```
AnalyzeFragmentAccess:  解bind别名 → 下标合法性 → 代换 → 套布局映射 → 绑证明域
                        → 证(a) → 证(b) → 证(c) → 全过返回（带 entry 表达式供转换用）
```

**设计灵魂**：求解器投票（候选布局下试探）与 Verify 放行（最终布局下复核）**用同一个函数、绑同一个范围集**——判定不可能分叉；且求解器先报，报错点确定、文案稳定可测试。若各写一套（历史上出现过），"求解器放行 Verify 拒"或反之都可能出现。

---

## 5. ⑩ Vectorize Storage 模块：真正改写程序

**代码位置**：声明发射 [vectorize_parallel_to_pto.cc:645 附近（每区域 Bind declare_fragment_buffer；消费核对 :697 起）](src/ascend/transform/vectorize_parallel_to_pto.cc#L645)；读转换 [ConvertFragmentLoad:2155](src/ascend/transform/vectorize_parallel_to_pto.cc#L2155)；写转换 [ConvertFragmentStore:2172](src/ascend/transform/vectorize_parallel_to_pto.cc#L2172)；carry 排除 [AnalyzeLoopShell:1089](src/ascend/transform/vectorize_parallel_to_pto.cc#L1089)；范围复现 `EnclosingSerialRanges`（[:1644](src/ascend/transform/vectorize_parallel_to_pto.cc#L1644)，转调共享实现）；收尾核对 [:1136 起（declare 消费检查）](src/ascend/transform/vectorize_parallel_to_pto.cc#L1136)。

五件事：**① 区域开头发声明 ② 循环重组 ③ fragment 读写 → handle 寻址 ④ 条件写 → 掩码合成 ⑤ 收尾清理**。

### 循环重组：j 维去哪了（sv3 真实 dump）

```python
# 转换前                                  # 转换后
for k in range(4):                        for k, i, q in T.grid(4, 8, 1):
    for i in T.parallel(8):                   vload_0 = vload(acc_fragment_mem, i*128)
        for j in T.parallel(128):             vload_1 = vload(a_ub[k*1024+i*128+q*128], 0)
            acc[i,j] = acc[i,j]               sload_2 = x_ub[k]; brc_3 = vbrc(sload_2)
              + a_ub[...]*x[k]                vadd_5  = vadd(vload_0, vmul(vload_1, brc_3))
                                              vstore(vadd_5, acc_fragment_mem, i*128)
```

j 塌缩进 128 条 lane；i 保留为循环；**q 是 chunk 维**（ragged 时 `ceil(lane维/VL)` 个轮次，尾部 chunk 的 store mask 只有前若干位为 1）。

### 声明：每区域一次、全新 Var

```python
keep_fragment_mem: T.handle = T.tl.vmi.declare_fragment_buffer(
    T.access_ptr(keep_slot_backing[0], 128, 3),  # backing 地址（⑥证照）
    T.int64(1), T.int64(128))                    # count、lanes（⑥证照）
```

vecscope 分裂会让一个 plan 对应多个区域 → 每区域各声明一次、用 plan 里的名字**现造全新 Var**（同一 Var 绑两次违反 SSA）。

### 访存转换的三个细节

- **`entry × lanes` 是布局的直译**：entry=i（⑧证明的产出）→ 偏移 `i*128`；
- **两次读就是两条 vload，绝不 CSE**（Select 形态的真实 dump 里 `keep_fragment_mem` 同地址读两遍、两条 vload 原样保留）——合并访存是 PTOAS 的职权（mem-to-value），前端不越权；
- **带偏移读无特例**：`out[i] = f[i+128]` 的 entry 表达式 = q+1，直接填进偏移 `vload(handle, (q+1)*128)`，与 `f[i]` 同一条代码路径。

### 条件写：两种形态两种打法（真实 dump）

双边 `T.Select(cond, a, b)`（值依赖条件）→ `vcmp` 谓词 + `vsel` 逐 lane 挑。

单边 `if cond: keep[i] += 1`（只有"写不写"依赖条件）→ **全量算值 + 掩码合成**：

```python
vadd_5  = vadd(vload_3, brc_4, mask)          # ①无条件算新值（全量算）
vcmp_2  = vcmp(0, a_ub_broadcast, "lt")       # ②条件 → 谓词向量
mand_6  = mask_and(mask, vcmp_2)              # ③chunk掩码 ∧ 条件谓词  ← 关键
vstore(vadd_5, keep_fragment_mem, 0, mand_6)  # ④合成掩码写入
```

对照错误做法（`select(cond, old+1, old)` 后全量写）：实际做法**少一次旧值读、少一次 vsel**，且与 VMI masked-store 语义（inactive lane 不写、保留旧值）严丝合缝；ragged 尾巴与条件写并存时两个掩码天然取交集。

### 刻意不做的两件事

1. **一条 mem_bar 都不插**：vecscope 内保序整体委托 PTOAS（PR307/PR374；覆盖缺口经 [#1679](https://github.com/hw-native-sys/PTOAS/issues/1679) 推动、PR374 修复并验证：三类复现形态 bar 1→3 / 0→6 / 跨 vecscope 0→1 每迭代，数值全绿）；
2. **fragment 不进 carry 提升**（`AnalyzeLoopShell` 排除）：carry 改写访问结构，fragment 访问必须留在 backing 走 vload/vstore，保序结构才完整（B2 防线）。

### 收尾

摘除 fragment 分配与已消费注解；**残留 fragment 访问直接拒绝**（转换有漏网就是 bug，带病前行不如当场崩）；handle 只能作 vload/vstore 基址（⑨ `verify_pto_codegen_legality.cc` 复核）。

```
进入区域 → 逐plan声明(全新Var)+chunk mask → 逐单元：
             循环重组(lane塌缩/q显形) → 标量读+vbrc → UB访存(连续形态)
             → fragment访存 vload/vstore(handle, entry×lanes)
             → 条件写(双边vcmp+vsel / 单边mask_and+masked vstore)
          → vecscope分裂的各part各自重新声明（split范围随part带上）
          → 收尾：摘注解、残留拒绝、handle核对
```

---

## 6. ⑪ 代码生成 + 流水线

**代码位置**：声明打印 [codegen_pto.cc:5538（memref_declare 发射）](src/ascend/codegen/codegen_pto.cc#L5538)；常量断言 [:5527（backing offset ICHECK）](src/ascend/codegen/codegen_pto.cc#L5527)。

声明 → `pto.memref_declare(字节地址, (count,), dtype)`，**地址强制常量**（ICHECK：动态地址下 PR307 判 Unknown 不插 bar；Merge 常量检查保证不可达，断言防未来 pass 重排放回）。读写走通用 `pto.vmi.vload/vstore`，`(source, offset)` 两参 ABI 按 PTODSL 要求打印。流水线（`tilelang/ascend/pipeline.py`）只加两个 pass：Expand 在布局推导前、Plan 在调度前。

---

## 7. sv3 逐 pass 对照表（实测 dump）

| 阶段 | 程序形态 | fragment 长相 |
|---|---|---|
| 用户写的 | T.fill + acc[i,j] | 语法对象 |
| ② 展开 | 全是 Parallel 循环 | 二维下标 |
| ④ 求解 | 循环不变 | + layout_map（**lane=j**，UB 连续性投出来的） |
| ⑥ 规划 | 循环不变 | + 独占 backing（1024 f32）+ tl.pto_slot_plan |
| ⑧ 校验 | 循环不变 | 每个访问过三件事证明 |
| ⑩ 转换 | j 维消失、`grid(4,8,1)` | vload/vstore(handle, i*128, mask) |
| ⑪ 生成 | PTODSL python | memref_declare(20480, (1024,), f32)（20480 是 Merge 分的最终地址） |

---

## 8. 关键设计决策一览（决策与理由）

| 决策 | 理由 | 代码位置 |
|---|---|---|
| fill/copy 最早处归一成循环 | 后续所有环节零特例；不进调度器 | §1 各函数 |
| 布局全局联立求解（奇偶并查集） | 转置等访问把多块 fragment 的选择锁在一起，贪心会错 | `ParityDSU::Unite` :406 |
| 方向可行性=三件事证明（而非语法） | `f[i,j*2]` 语法上两个方向都"沾边"，只有证明能筛掉不可行的，避免误拒 | `FragmentDirectionFeasible` :251 |
| 求解器与 Verify 共用一个证明函数 | 判定永不分叉；求解器先报 → 报错确定可测试 | utils :639 两个消费方 |
| count = 其余维积 × ceil(lane维/VL) | 存储按 entry 分配，ragged 尾巴也占整格 | `EntryCountFromLayout` :376 |
| backing 独占、不参与 lifetime reuse | 证不了复用安全；独占最多浪费容量、不会地址冲突（浪费由预算控制） | merge_ub_allocations.cc |
| 预算扣减在调度之前 | 双缓冲决策必须看到扣掉 backing 后的真实余量，否则一地两许到 Merge 才炸 | auto_schedule.cc :1413 |
| 预算管决策、终检管有效性（全路径） | 直线 kernel 无版本决策、预算无对象；容量保证是 Merge 终检（含 no-contract） | :666 / :746 |
| 登记与分地址分离（Merge 统一分配） | 不相交性一次性全局解决；验收从最终模块声明地址直接断言 | test_pto_slot_capacity.py |
| 布局字节最小取小 | 从源头消除 32× 膨胀布局——否则预留失败的报错与根因错配 | Solve（§2） |
| 每区域全新同名 handle Var | vecscope 分裂 + SSA 单定义 | vectorize :645 |
| 条件写 = mask_and 合成、不读旧值 | 少一次读一次 select，与 masked-store 语义吻合 | `ConvertFragmentStore` :2172 |
| 两次读不合并（不 CSE） | 合并访存是 PTOAS 职权（mem-to-value），前端不越权 | `ConvertFragmentLoad` :2155 |
| TileLang 侧 0 条 mem_bar | 保序整体委托 PR307/PR374（缺口 #1679 已修复验证） | —（刻意不做） |
| fragment 不进 carry 提升 | 保序结构完整；backing 上走 vload/vstore | `AnalyzeLoopShell` :1089 |
| 声明地址强制常量（ICHECK） | 动态地址 PR307 判 Unknown 不插 bar | codegen :5527 |
| 多下标收集 pass 自有 | 上游 ParallelOp 零改动（保持严格默认）；solver/vectorize 各自收集 | solver :85 / vectorize :525 |

## 9. 验证状态

- 以下数字为分支结构重组（Ascend-owned 栈）**之前**的内容等价树实测，重组后复验待跑：全 pto 目录（设备含）**681/681**；slot 设备 22/22 + 容量设备 5 项；fragment 条件写回归 16/16；
- gym 数值 **16/16**（GA 面 11：sv1/sv2v×2/sv3v×2/cf1v–cf6v；weekly CANN 面 5：sv5v×2/sv6v/sv7v×2，含 fragment 工作集 keep_in_rf，maxdiff 全 0）；
- 结构套件 50 项含三轮评审回归（非规范注解拒绝、2D 行 fill 编译、`f[i,j*2]` outer 定向、if 包裹串行 entry 1D/2D、1D 多 chunk 求解器拒绝等）；
- 外部依赖：PTOAS PR374（#1679 修复）已验证，等合入 master；MR274 待上游。

## 附录：如何自己 dump 这些中间 IR

结构测试的 `_lower_capture(func, pass_name)`（`testing/ascend/language/pto/test_pto_slot_plan_structure.py`）在指定 pass 后停下抓 IR。模板：

```python
from test_pto_slot_plan_structure import _lower_capture, _region_layouts
mod, err = _lower_capture(my_kernel(), "VectorizeParallelToPTO")   # 或
    # "ExpandPtoFragmentFillCopy" / "AscendLayoutInference" / "PlanPtoSlotStorage" / ...
print(mod.script())
print(_region_layouts(mod))        # 求解器判词：每块 fragment 的 forward_thread
```

238 环境需 merge 工具链（PTOAS-merge + ptoas-merge-bin，`TMPDIR` 指 /data）。完整六阶段 dump 存于 `doc/tilelang_parallel/sv3_pass_tour.txt`。
