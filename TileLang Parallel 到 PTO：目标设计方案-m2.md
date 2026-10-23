# TileLang Parallel 到 PTO：目标设计方案

本文定义 `target="pto"` 下、`T.SimdVF(lanes=...)` 内的 `T.Parallel` 到 PTO VMI 的目标转换语义：输入 TIR 如何变成 SIMD TIR，TileLang 与 PTOAS 各自负责什么，哪些形式属于目标支持边界。本文只描述长期有效的语义、接口和职责，统一使用「目标/应当/必须」的表述，不包含实现进度、任务拆分和验证状态。

## 1\. 背景、目标与术语

### 1\.1 背景与目标

PTO SIMD 以逻辑 lane 执行向量计算。`T.SimdVF(lanes=...)` 指定逻辑 lane 数，`T.Parallel` 定义逻辑迭代空间；Parallel 本身不提供 SIMT thread 信息，PTO SIMD 的 lane 范围来自 `SimdVF(lanes=...)`。本方案的目标是把 `SimdVF` 内的标量 Parallel、标量表达式、Fragment Buffer 和 reducer 语义统一转换为 SIMD TIR，再由 PTO codegen 和 PTOAS 完成物理化。

以 RMSNorm 为例，其计算体同时包含三类需求，能够代表本方案必须同时覆盖的问题：

- 跨循环存储：一个 Parallel 把整行输入保存到 Fragment Buffer，后续 Parallel 再次读取；

- 归约：一个 Parallel 反复更新部分和，finalize 后得到完整结果并广播；

- 逐元素计算：归约结果参与后续 Parallel 的向量算术。

跨循环保存的容量压力是设计 Fragment 存储的直接动机。设逻辑 lane 数 `VL = 128`，FP32 每个元素占 4 字节，对 4096 元素的整行保存：

```text
x_frag 的元素数            = 4096
逻辑向量数                = 4096 / 128 = 32
每个逻辑向量的大小         = 128 * 4 = 512 B
x_frag 的总数据量          = 16 KiB
```

按 256 B 物理向量寄存器计算，一个 128\-lane FP32 逻辑向量需要两个物理寄存器；若 32 个 x 向量同时保留就需要 64 个物理寄存器，超出 32 个物理向量寄存器的上限，且还未计入部分和、权重和临时结果。改变 VL 不会减少整行数据的总量：VL 为 64、128、256 时，逻辑向量数分别为 64、32、16，全部保留仍占用相当于 64 个 256 B 寄存器的空间。这里的数量计算针对「整行 x 同时驻留」的方案，不是对某个已编译 kernel 的实测压力。

将 `x_frag` 改成逻辑 vreg 数组还会遇到运行时 q 索引的问题：静态 vreg 数组不支持一般的动态寻址。固定 UB backing 和 memref 提供可索引的存储表达，动态 entry 放在 vload/vstore 的元素 offset 中；寄存器驻留选择留给后端。

Fragment Buffer 使用固定预留的 UB backing：编译器必须在选择 double buffer 等调度方案前知道 Fragment 存储占用多少空间，并在生成代码前给它分配不与其他数据重叠的地址。AutoSchedule 在 Vectorize 前做容量预算，`MergeUBAllocations` 在 Vectorize 后分配地址；因此 Fragment 存储的 UB 需求必须在调度前声明，否则容量预算会漏算。栈上保存不作为目标方案的一部分。

### 1\.2 术语约定

本文中「Fragment」一词有两种历史含义，先区分清楚：

- **Fragment Buffer** 指 `T.alloc_fragment` 或 `scope="local.fragment"` 产生的 TIR Buffer，是可被映射到 Fragment 存储的存储对象；

- `tilelang.layout.Fragment` 描述坐标到执行位置的映射对象，本文称其为 **lane 布局** 或 **Fragment layout**。

全文使用以下术语，含义唯一：

|术语|含义|
|---|---|
|Fragment Buffer|`T.alloc_fragment` 或 `scope="local.fragment"` 产生的 TIR Buffer|
|lane layout / Fragment layout|逻辑坐标到 lane、entry 的映射对象|
|PartialFragment|reducer 前置阶段的部分和参与关系布局|
|VL|逻辑 lane 数|
|E|原始逻辑 extent|
|P|选中向量化维补齐后的执行 extent|
|Q|chunk 数|
|q|当前 chunk；是否等于 entry 由布局决定|
|entry|Fragment backing 中一个完整逻辑向量的位置|
|版本槽位|普通 UB 的 ping/pong 或 multi\-buffer 版本|
|Fragment backing|Plan 创建的普通 shared\.dyn 预留 Buffer|
|Fragment 存储句柄|TIR 声明返回的 Handle Var，关联 backing、count 和 lanes|
|Fragment memref|PTOAS 中对已有 UB 地址的类型化视图|

普通 UB 的多缓冲索引称为“版本槽位”。Fragment Buffer 是源存储对象，backing 是预留的 UB Buffer，TIR handle 是声明返回值，PTOAS memref 是已有 UB 地址的有类型视图。entry 表示一个完整逻辑向量的位置；`element_offset = entry_index * VL`。`PlanPtoSlotStorage` 及内部规划结构名继续保留。

### 1\.3 目标支持范围总览

目标支持边界按能力分别定义，各有独立的准入条件，不在此处用一个统一排除列表覆盖：

- 基础 element\-wise 向量化：一维/二维 Parallel、连续访存、dtype/cast，见第 4、5 章；

- 可规范化控制流：已证明安全的 If→Select、Min/Max→Select、布尔表达式和条件更新，见第 5、9 章；

- direct reduce：对已有 UB region 的直接规约，见第 7 章；

- Fragment Buffer 与 memref 存储：跨循环的向量保存，见第 6 章；

- reducer v2：前端 legacy reducer 经规范化、计划和 materialization 后的 finalize 路径，见第 8 章。

各能力的共同边界条件和正确性前提汇总于第 11 章。符号有效长度、uniform scatter 以外的通用 gather/scatter、打包类型等扩展方向见 11\.3 节。

## 2\. 输入、输出与贯穿示例

### 2\.1 vecadd：基础 element\-wise

以 `example_simdvf_vecadd.py` 的计算体为例，编译入口指定 `target="pto"`，并为 SimdVF 指定 lanes：

```python
with T.SimdVF(lanes=128):
    for i in T.Parallel(TILE_ELEMS):
        temp3[i] = temp1[i] * (temp1[i] + temp2[i])
```

示例文件名为 vecadd，计算体实际是 `A * (A + B)`。本例中 `TILE_ELEMS=8192`；`NUM_THREADS=2048` 只用于计算 tile 大小，不表示 PTO SIMD 的线程数。`T.SimdVF(lanes=128)` 指定逻辑 lane 数 `VL=128`。

`T.Parallel(M, N)` 定义 `M*N` 个逻辑迭代点，每个点由坐标 `(i, j)` 标识。进入 Verify 和 Vectorize 时，调度、同步和 `NormalizeBufferVersion` 已完成，`SIMD_VF` 内仍是标量 Parallel 和标量表达式。目标输入应同时带有前端写入的 `tl.simdvf_lanes` 和 LayoutInference 写入的 `pto_parallel_loop_layout`：

```python
with T.sblock("SIMD_VF", no_realize=True):
    T.sblock_attr({
        "tl.vf_source_index": T.int64(0),
        "tl.simdvf_lanes": T.int64(128),
    })
    T.attr("simdvf", "tl.simdvf_scope", 1)
    for i in T.parallel(
        8192,
        annotations={
            "pto_parallel_loop_layout": metadata["tl.Fragment"][0],
        },
    ):
        temp3[iter % 2, i] = (
            temp1[iter % 2, i] *
            (temp1[iter % 2, i] + temp2[iter % 2, i])
        )
```

`metadata["tl.Fragment"][0]` 表示第 4 章定义的 lane 布局对象。对本例，`lane=i%128`，`q=i//128`，共 64 个 chunk，`replicate_size=1`。双缓冲版本下标 `iter % 2` 对所有 lane 相同，不参与迭代点分配，但必须保留在访存地址中。

转换保留原始 For 的逻辑 extent；lane 布局改用补齐后的执行空间，而 Buffer 地址仍按输入 Buffer 的实际 shape、strides 和 `elem_offset` 计算。补齐点的结果不写回，尾部 load 的可读范围由调用方保证，详见第 9 章。

vecadd 覆盖了本方案的基础要素：`T.Parallel`、lane、连续 load/store、E/P/Q、mask 和 VMI 输出；完整的转换输出见第 12 章。

### 2\.2 RMSNorm：Fragment、reducer 和跨循环保存

以 `examples/ascend/example_rmsnorm.py` 为例，省略 GM/UB 搬运和外层流水循环后，计算体为：

```python
x_frag = T.alloc_fragment((TILE,), "float32")
for i in T.Parallel(TILE):
    x_frag[i] = x_ub[i]

sum_sq = T.alloc_reducer((1,), "float32", op="sum", replication="all")
T.clear(sum_sq)
for i in T.Parallel(TILE):
    if i < d:
        sum_sq[0] += x_frag[i] * x_frag[i]
T.finalize_reducer(sum_sq)

var = sum_sq[0] / d + eps
rstd_val = T.rsqrt(var)
z_rstd_ub[0] = rstd_val
for i in T.Parallel(TILE):
    if i < d:
        y_ub[i] = x_frag[i] * rstd_val * w_ub[i]
```

其中 `TILE = next_power_of_2(d)`。当 `d = TILE = 4096` 时，两个 `if i < d` 都会被简化掉；当 `d < TILE` 时，条件限制归约贡献和输出写入。后一种情况还需要核对完整向量读取的范围，不能因为计算有 mask 就假定 load 安全。

三个 Parallel 之间有明确的数据关系：第一个循环保存整行 x，第二个循环反复更新部分和，第三个循环在归约完成后再次读取 x。归约结果广播后，`var` 和 `rstd_val` 按向量运算生成；后续输出计算直接使用 `rstd_val` 的向量表示。归约完成之前的 `sum_sq` 则保存各 lane 不同的部分和。

VF 转换需要同时处理 Parallel 内的 binary 表达式、VF 内的分配和初始化、finalize、copy，以及循环之间的值使用。

**历史捕获**：以下是进入 Vectorize 前的 RMSNorm TIR（区域为原始 `SIMT_VF`），用于确认源程序在该阶段保留的信息，不表示 SIMD 路径已接受该程序，也不是目标输出。此处省略 `data=`、layout 注解、线程声明和外围同步，只保留计算体：

```python
x_frag = T.sblock_alloc_buffer((4096,), scope="local.fragment")
sum_sq_1 = T.sblock_alloc_buffer((1,), scope="local.fragment")
sum_sq_result = T.sblock_alloc_buffer((1,), scope="local.fragment")

for i in T.parallel(4096):
    x_frag[i] = x_ub[t % 2, i]

T.fill(T.region(sum_sq_1[0], 2, 1), T.float32(0.0))
for i in T.parallel(4096):
    T.attr(sum_sq_1.data, "tl.parallel_multiplicity", 1)
    sum_sq_1[0] = sum_sq_1[0] + x_frag[i] * x_frag[i]

T.finalize_reducer(T.region(sum_sq_1[0], 3, 1), 0)
T.copy(T.region(sum_sq_1[0], 1, 1), T.region(sum_sq_result[0], 2, 1))
var: T.float32 = sum_sq_result[0] / T.float32(4096.0) + eps
rstd_val: T.float32 = T.rsqrt(var)
z_rstd_ub[t % 2, 0] = rstd_val

for i in T.parallel(4096):
    y_ub[t % 2, i] = x_frag[i] * rstd_val * w_ub[i]
```

**历史捕获（失败位置）**：将 VF 声明改成 `SimdVF(lanes=128)` 后，当时的流水线更早停在 `AscendLayoutInference`，报错为：

```text
The partial layout for reducer sum_sq can not be inferred correctly.
```

该失败位置是输入证据：reducer 布局推导与 materialization 必须先适配 PTO SIMD，仅调整后面的 VF pass 顺序不能解决。

这些语句对应的节点如下：

|TIR 内容|节点或含义|
|---|---|
|`T.sblock_alloc_buffer(...)`|Buffer 记录在 `SBlock.alloc_buffers` 中；此时不必是独立 `Allocate` 语句|
|`x_frag[i] = ...`|`BufferStore`，明确保存对象和逻辑索引|
|表达式中的 `x_frag[i]`、`sum_sq_1[0]`|`BufferLoad`|
|`var`、`rstd_val`|`Bind` 定义的值|
|UB 访问中的 `t % 2`|双缓冲版本槽位，与 Fragment Buffer 的元素索引分开处理|

`sum_sq_result` 由 `CanonicalizeLegacyReducer` 引入。该 pass 将源码中原地 finalize 的写法拆成累加状态和最终结果，finalize 后对原 `sum_sq` 的读取改为读 `sum_sq_result`。随后 `ReducerPlanAndMaterialize` 将更新展开成普通算术，并生成原地 finalize 和结果 copy。

这里 `x_frag` 的逻辑形状仍为 `[4096]`。SIMT 路径后续应用线程布局，才得到生成代码中的每线程 `float x_frag[32]`；对应样例的 128 个线程各保存 32 个 float，加载循环实际使用 16 次 `float2` 读取。PTO 转换应从逻辑 Buffer 出发建立 SIMD 表示，不直接套用这组 SIMT 打包布局。

## 3\. 目标 pass pipeline 与阶段契约

### 3\.1 编译路径与目标分流

```text
target="pto"
    LayoutInference
        → …（见 3.2 节完整 pipeline）
        → VerifyParallelToPTO
        → LegalizeParallelToPTO
        → VectorizeParallelToPTO
        → LowerTileOp / LowerAccessPtr / 其他现有 pass
        → PTO codegen
        → PTOAS

target="ascend" 或默认 Ascend
    现有 Ascend pipeline
        → AscendSimdVFLowerParallel
```

`VerifyParallelToPTO` 检查失败时直接终止编译，不回退到 AscendC。

PTO 与 AscendC 共用 Ascend target kind。`target="pto"` 规范化后为 `kind="ascend"`、keys 包含 `pto`；Python pipeline 使用 `target_is_pto` 判断，C\+\+ 布局分析使用相同的目标识别规则。PTO lane 上下文、布局候选和独立注解写回只对 PTO 的 `SIMD_VF` 启用，AscendC 和 PTO `SIMT_VF` 保持各自的处理路径。

### 3\.2 完整 pipeline

目标 pipeline 在现有 Ascend pipeline 中增加 `PlanPtoSlotStorage` 和 `LegalizeParallelToPTO`，并扩展 Verify/Vectorize。新增规划仅处理 PTO 目标下待转换的 SIMD\_VF；SIMT\_VF 和其他后端保留各自路径。

```text
输入 TIR：SimdVF、Parallel、标量表达式、Fragment Buffer、reducer 语义
  ↓
CanonicalizeLegacyReducer → VerifyReducerEpoch / VerifyBufferInit
  （仅 reducer v2 路径；识别 init/update/finalize_v2，检查 reducer epoch 与初始化）
  ↓
LayoutInference
  ├─ 普通 Parallel：构造 PTO lane layout
  └─ reducer v2：建立 PartialFragment / Buffer layout
  ↓
ReducerPlanAndMaterialize
  （仅 reducer v2 路径；展开为 fill、普通算术更新、显式 finalize 和结果 copy）
  ↓
PlanPtoSlotStorage
  （读取 Fragment/Buffer layout，提前声明 UB backing 和容量）
  ↓
调度、同步、普通 Buffer 版本和 AutoSchedule
  ↓
NormalizeBufferVersion
  （规范化普通 Buffer 版本，同时维护 Fragment 与 Fragment backing 的对应关系）
  ↓
VerifyParallelToPTO
  ↓
LegalizeParallelToPTO
  ↓
VectorizeParallelToPTO
  （Parallel / Expression / Storage / ControlFlow / Reduce）
  ↓
LowerAccessPtr / LowerOpaqueBlock / FlattenBuffer / MergeUBAllocations
  （改写 backing 地址，Fragment 声明和 VMI 访存 继续保留）
  ↓
PTO codegen：declare_fragment_buffer → memref_declare，普通 VMI 读写直接输出
  ↓
PTOAS
  （物理寄存器、mask 物理化、Fragment 值驻留/UB 访问、依赖和 mem_bar）
```

普通 Parallel 路径不经过 reducer 的三个前置 pass；两条路径在 PlanPtoSlotStorage 之前汇合，之后共用同一段 pipeline。

reducer 的语法规范化、epoch 与初始化检查及 materialization 由前置 pass 负责（见第 8 章）。`PlanPtoSlotStorage` 在布局和 reducer materialization 之后、AutoSchedule 及其准备阶段之前运行，接在 `AscendInsertOOBPadding` 等前置 Buffer 处理之后。规划与最终 UB 分配在自动和手动调度路径上都执行。Vectorize 后，`AscendLowerTileOp` 处理剩余 tile 操作，随后 `LowerAccessPtr`、`LowerOpaqueBlock`、版本布局重写、`FlattenBuffer` 等完成地址处理；关闭 AutoSchedule 时，`InferBufferAliases` 在 `SplitHostDevice` 之后、Merge 之前运行。`declare_fragment_buffer` 保留到 codegen，输出为 `pto.memref_declare`；Fragment 的普通 VMI 访存在 Vectorize 中生成。

提前声明 backing，使调度器按实际可用容量选择 Buffer 版本；Vectorize 在同一 VF 内统一生成算术、控制流、归约、Fragment 声明和读写。最终 UB 地址由分配器确定，PTOAS 决定哪些访存可消除、哪些仍访问 backing。声明直接映射到 memref，无需新增接口 lowering pass。

direct reduce 和 reducer v2 都由 Vectorize 的 Reduce 模块消费，但前置输入不同（见第 7、8 章）。direct reduce 的规约指令生成逻辑是 Vectorize Reduce 模块的内部能力；目标架构不要求存在独立的前置 Reduce pass，现有实现中的规约 lowering 模块可以由 Vectorize 调用复用。

### 3\.3 各阶段契约

- `CanonicalizeLegacyReducer` / `VerifyReducerEpoch` / `VerifyBufferInit`：仅 reducer v2 路径。将前端 `+=` 识别为 init/update/finalize\_v2，检查 reducer epoch 与初始化合法性。

- `LayoutInference`：读取 `tl.simdvf_lanes`，建立 `[0,VL)` 的逻辑 lane 范围，根据 Buffer 访存选择向量化维，构造 lane 布局并写入 `pto_parallel_loop_layout`；reducer v2 路径还需推导普通 Fragment Buffer、reducer 部分和与结果 Buffer 的布局并写回 `layout_map`（见第 8 章）。

- `ReducerPlanAndMaterialize`：仅 reducer v2 路径。按布局展开为 fill、普通算术更新、显式 finalize 和结果 copy（见第 8 章）。

- `PlanPtoSlotStorage`：读取待转换 SIMD\_VF 中的 `local.fragment` 分配及 `layout_map`，确定 entry 数量、向量类型、UB 大小和对齐，在支配所有使用点的 SBlock 中声明隐藏 `shared.dyn` backing，并保存 Fragment→backing 的规划记录（见 6\.1 节）。

- 调度、同步、多缓冲与 `NormalizeBufferVersion`：沿用现有 pass；Fragment backing 固定预留一份，不参与普通 Buffer 的版本选择，也不与其他对象复用地址（见 6\.5 节）。

- `VerifyParallelToPTO`：只读检查 PTO 路径的输入契约，不修改 TIR；接受后续 Legalize 可以规范化的形式；遇到第一个错误就终止（见 5\.2 节）。

- `LegalizeParallelToPTO`：只做已证明安全的 If→Select、Min/Max→Select 和布尔表达式规范化，输出仍是标量并行 TIR；不重新执行 reducer canonicalization、layout inference 或 materialization（见 5\.3 节）。

- `VectorizeParallelToPTO`：消费已经验证的布局和规划，在同一个 VF 上统一处理 Parallel、Expression、Storage、ControlFlow、Reduce 五个模块，生成逻辑 VMI 和 Fragment 声明和 VMI 访存（见 5\.4 节）。

- PTO codegen：将声明输出为 `pto.memref_declare`，将普通 VMI 输出为对应的 `pto.vmi.*` 调用。

- PTOAS：完成 VMI layout、mask 粒度、物理寄存器拆分、Fragment 值驻留或 UB 访问选择、依赖分析和 `mem_bar` 插入，以及后续优化（职责边界见第 10 章）。

`VerifyParallelToPTO`、`LegalizeParallelToPTO` 与 `VectorizeParallelToPTO` 依次相邻；Verify 与 Vectorize 共享只读的布局求逆和单次访存分析 helper，Legalize 只做等价改写，不消费这些分析。Verify 用分析结果判断输入是否合法并报告错误；Vectorize 在变换时重新调用相同 helper，取得生成代码所需的映射和地址信息，不再执行一套独立的用户输入检查。Legalize 会替换节点，因此 Vectorize 需要重新分析其输出，不能直接复用源节点上的缓存。若相邻 pass 之间未来插入会重建循环的变换，必须重新执行 Verify 或调整 pass 顺序。

### 3\.4 多 Parallel 与多区域

一个 `SIMD_VF` 内可以有多个顺序排列的一维或二维 Parallel。本文将一个一维 Parallel For，或组成二维空间的两层紧邻 Parallel For，称为一个 Parallel 整体。不同整体之间可以经 Buffer store/load 传递结果，变换保持它们的原始顺序。

多个 `SIMD_VF` 区域独立处理，各自使用 lanes、布局、mask 和临时变量，不跨区域复用值。Vectorize 负责逻辑点到 lane 的映射和 VMI 生成；DMA、流水线和同步继续由现有前后置 pass 负责。

padding 的初始化、可读范围、生命周期和访问所需的同步属于调用方前提；Verify 不证明这些条件，Vectorize 不增加 padding、fill、边界检查或同步，也不修改调度逻辑。

## 4\. 循环布局、Buffer 布局和 E/P/Q/remaining

### 4\.1 长度定义（E/P/Q/remaining）

本节定义 E/P/Q/remaining，其他章节引用此处，不重复定义。

`E` 表示一维循环或二维布局中选定的向量化维的逻辑 extent。每个 chunk 最多处理该维上的 VL 个逻辑点。目标设计支持编译期常量 extent，整除与非整除统一处理：

```text
Q = ceil(E / VL)
P = Q * VL
tail_lanes = E % VL
```

`Q` 是 chunk 数，`P` 是选中维补齐后的执行长度，满足 `E <= P < E + VL` 且 `P % VL == 0`。`E`、`P` 和 `Q` 分别用于逻辑有效范围、lane 布局输入范围和 chunk 循环。构造常量时检查运算及索引类型的表示范围，避免补齐长度或二维展平乘积溢出。选中维要求 `E > 1`；`E=1` 的单点形态见 4\.6 节。`E % VL == 0` 时 `P=E`，全部规则退化为整除形式，mask 恒为全量。

非整除路径使用一个带 `remaining` 的 chunk 循环：进入第 q 轮循环体时 `remaining = E - q*VL`，有效 lane 满足 `lane < min(VL, remaining)`，等价于 `q*VL+lane < E`。完整 chunk 和尾部 chunk 共用一个循环体，不拆出 tail 循环，也不增加运行时 full/tail 分支。`remaining` 的表示和更新规则见 9\.2 节；mask 规则见第 9 章。

### 4\.2 `T.SimdVF(lanes=...)` 与布局注解

```python
T.SimdVF(latency=0, lanes=128)
```

`lanes` 默认值为 128，只接受 64、128、256，表示 PTO SIMD 的逻辑 lane 数。前端将它传入 FFI，退出 frame 时写入 `SIMD_VF` block 的 `tl.simdvf_lanes` 注解；整个过程不生成线程 launch。原有 latency、source index 和 scope 行为保留。

PTO SIMD 使用独立 key：

```python
for i in T.parallel(
    8192,
    annotations={
        "pto_parallel_loop_layout": metadata["tl.Fragment"][0],
    },
):
    ...
```

`pto_parallel_loop_layout` 是 annotation 的字符串 key，value 使用现有 `Fragment` 布局对象（即 lane 布局）。普通 `parallel_loop_layout` 继续表示 SIMT 线程布局，PTO SIMD 使用独立 key，避免被现有 SIMT lowering 误读。

布局在调度前生成，Vectorize 完成后删除 `pto_parallel_loop_layout`，保留外围 `SIMD_VF` block 和 `tl.simdvf_lanes`。如果中间 pass 改变循环范围或访问表达式，必须更新布局或使其失效。

### 4\.3 循环布局：lane 布局（Fragment）语义与求逆

PTO lane 布局表示执行输入坐标到 lane 及 lane 内部位置的映射：

|字段|语义|
|---|---|
|`input_size`|执行输入空间；选中向量化维补齐到 `P`，未选中维等于原始 extent|
|`forward_thread`|执行体编号；SIMT 中是 thread，PTO SIMD 中是 lane|
|`forward_index`|执行体内部位置的表达式数组；具体含义由布局决定|
|`replicate_size`|每个执行点的副本数；要求为 1|
|`thread_range`|`forward_thread` 的有效范围；PTO SIMD 中是 `[0,VL)`|

二维 Parallel 的逻辑点是坐标 `(i, j)`。lane 布局将它映射到 `(执行体内部位置, lane)`；辅助展平下标只用于表达这份映射，不改变输入坐标的维数。`forward_index` 只含一个位置表达式，记其值为 p；p 与每行或列内的 chunk 下标 q 的关系见 4\.4 节。

`Fragment::InverseWithLevel` 将复制维加入输入空间，将 `forward_thread` 追加到 `forward_index` 之后，构造普通 `Layout` 再求逆。`Layout::InverseWithLevel` 先用 `arith::DetectIterMap` 识别前向迭代映射，再用 `arith::InverseAffineIterMap` 得到原输入坐标表达式，返回逆 `Layout` 和使用的 `IterMapLevel`。

逆结果也包含复制维。`replicate_size=1` 时该维为 0，转换循环时取前面的输入坐标。求逆调用 `InverseWithLevel(false)` 并要求返回 `Bijective`，求逆的定义域为补齐后的 lane 布局输入范围。`NoCheck` 的返回结果不能代替可逆性检查，也不借用 SIMT padding guard 放宽求逆。

一维时，lane 布局的 `OutputShape()` 只有 lane 内部位置这一维，即 `[Q]`；追加 lane 后，逆 Layout 的输入 shape 为 `[Q, VL]`。它双射回 `[0, P)`，其中 `[0, E)` 是有效逻辑点，`[E, P)` 是补齐点。二维同理，只补齐选中维。可逆性检查不证明访存边界、依赖或别名。

### 4\.4 一维与二维循环映射

对 `T.Parallel(E)`，lane 布局输入 shape 为 `[P]`，`vectorize_size=1`。其 `forward_thread=i%VL`、`forward_index=[i//VL]`，执行体内部位置 p 就是 chunk 下标 q：

```text
lane = i % VL
q = i // VL
i = q * VL + lane
```

求逆时 `q ∈ [0,Q)`、`lane ∈ [0,VL)`，得到 `i ∈ [0,P)`；原始逻辑点仍要求 `i<E`。`E=8192, VL=128` 时生成 64 个完整 chunk。`E=150, VL=128` 时 lane 布局为 `[256]`，生成两个 chunk，分别有 128 和 22 个有效 lane。`E=22` 时 lane 布局为 `[128]`，仍保留 128 个逻辑 lane，不能让 `ThreadExtent()` 按未补齐的 shape 把 VL 缩成 22。

`T.Parallel(M, N)` 在 TIR 中表现为两个紧邻嵌套的 Parallel For，目标设计将它们识别为一个二维迭代空间：

```python
for i in T.parallel(M):
    for j in T.parallel(N):
        C[i, j] = A[i, j] + B[i, j]
```

识别依据是 TIR 结构；前端分别写两层 `T.Parallel`，只要最终结构相同，同样接受。`pto_parallel_loop_layout` 只挂在最外层 For，描述整个二维执行空间，Verify 和 Vectorize 都按整个循环嵌套处理。三维及以上嵌套，以及外层循环体同时包含计算语句和内层 Parallel、无法组成纯二维循环嵌套的情况仍拒绝。最内层计算体允许多条顺序语句。

转换保留非向量化维度，将物理连续维替换为 chunk 循环。`M`、`N` 都必须是编译期常量，只补齐选中的维度。lane 布局的输入坐标顺序保持 `(i,j)`，辅助展平表达式使用补齐后的选中维 extent：

令 `P_N = ceil(N/VL)*VL`、`P_M = ceil(M/VL)*VL`；`ceil` 按 4\.1 节的整数公式计算。

|向量化维|布局输入 shape|辅助展平|`forward_thread`|`forward_index`|
|---|---|---|---|---|
|j|`(M, P_N)`|`t = i * P_N + j`|`t % VL`|`[t // VL]`|
|i|`(P_M, N)`|`t = j * P_M + i`|`t % VL`|`[t // VL]`|

令 `p` 为布局中的全局 lane 内部位置，逆映射先恢复 `t = p * VL + lane`。用 `k` 表示生成循环中的保留维变量，`i`、`j` 表示待恢复的原始坐标：

|向量化维|逆映射|生成循环时的 p|代入后的坐标|
|---|---|---|---|
|j|`i=t//P_N`、`j=t%P_N`|`p=k*(P_N//VL)+q`|`i=k`、`j=q*VL+lane`|
|i|`j=t//P_M`、`i=t%P_M`|`p=k*(P_M//VL)+q`|`j=k`、`i=q*VL+lane`|

给定 p 时，先按选中维的 chunk 数 Q 分解：`k=p//Q`、`q=p%Q`，因此 `p=k*Q+q`。选 j 时 `Q=P_N//VL`，求逆后得到 `i=k`；选 i 时 `Q=P_M//VL`，求逆后得到 `j=k`。这里的分解只用已知的 Q，不预先使用待恢复的 i 或 j。

`p` 遍历整个布局，`q` 只遍历当前行或列。因为补齐长度是 VL 的整数倍，lane 等于选中坐标对 VL 取模，p 则按每行或列的 Q 个 chunk 编号。当选中维本来整除 VL 时，全部公式退化为整除形式。

Buffer 地址由输入 Buffer 的实际 shape、strides 和 `elem_offset` 决定，不能从 lane 布局的 `P_M` 或 `P_N` 推导物理跨度。例如逻辑循环为 `(4,150)`，输入 UB Buffer 显式分配为 `(4,256)` 时，其行跨度是 256，因为 Buffer 本身如此定义；另一个输出 Buffer 可以有不同的行跨度，各自用 `ElemOffset` 计算。

选中 j 维时，坐标恢复和 mask 的结构如下。lane 是分析占位符，实际输出通过 VMI 表示：

```text
for k in [0, M):
    remaining = N
    for q in [0, P_N/VL):
        i = k
        j = q * VL + lane
        mask = create_mask(remaining, size=VL)
        ...
        remaining = remaining - VL
```

选中 i 维时：

```text
for k in [0, N):
    remaining = M
    for q in [0, P_M/VL):
        i = q * VL + lane
        j = k
        mask = create_mask(remaining, size=VL)
        ...
        remaining = remaining - VL
```

每次进入保留维的迭代都重置 `remaining`。补齐 lane 不对应下一行或列的有效逻辑点；其实际 load 地址仍需满足调用方的可读范围约定。

`M=2, N=64, VL=128` 在访存沿 j 连续时使用 `(2,128)` 的 lane 布局，每行一个 chunk、64 个有效 lane。它不会合并为一个跨两行的 128\-lane chunk，也不仅因原 extent 非整除而拒绝。

### 4\.5 Buffer 布局与循环布局的差异

循环布局和 Buffer 布局使用不同的范围规则，这是有意设计，不能混用：

|对象|使用的范围|
|---|---|
|Parallel 循环布局|选中维使用 `P=ceil(E/VL)*VL`，用于 lane 求逆和 chunk 生成|
|Buffer 布局|保留逻辑 shape `E`；普通连续分布的 Fragment Buffer，entry 数按 `ceil(E/VL)` 计算（见下方映射表）；reducer 部分和与结果 Buffer 的 entry 映射见 8\.3 节|
|Fragment backing|按 entry 数、完整 vector type、alignment 计算容量|

Verify 对循环布局的「选中维 `input_size` 必须恰好为 `P=ceil(E/VL)*VL`」约束只适用于循环布局，不能套到 Buffer layout。Buffer 的 `[E]` 和执行布局的 `[P]` 是有意不同的：Buffer 映射只需在有效逻辑元素上无冲突，不要求 E 个元素与 `ceil(E/VL)*VL` 个存储 lane 构成满域双射，也不复用 Parallel 布局的满域双射检查。例如 `B[129]`、VL=128 对应两个 entry，但逻辑元素仍只有 129 个。

普通 Fragment Buffer 的 entry/lane 映射复用 lane 布局的字段：`forward_thread` 表示逻辑 lane，`forward_index` 表示 entry，均不指定物理寄存器。对普通 Buffer `B[M,N]`，令 `Q_N=ceil(N/VL)`、`Q_M=ceil(M/VL)`，布局按下表确定：

|向量化方向|Buffer 元素|entry|lane|entry 数量|
|---|---|---|---|---|
|j|`B[i,j]`|`i*Q_N + j//VL`|`j%VL`|`M*Q_N`|
|i|`B[i,j]`|`j*Q_M + i//VL`|`i%VL`|`N*Q_M`|
|一维|`B[j]`|`j//VL`|`j%VL`|`ceil(N/VL)`|

Buffer 的逻辑 shape 保持原值，预留容量则覆盖完整向量。例如 `B[129]`、VL=128 使用两个 entry，FP32 backing 为 256 个元素。额外 lane 只属于编译器保存空间，不扩大用户 UB Buffer 的可读写范围。同一个 Fragment Buffer 跨 Parallel 使用时必须满足同一映射；生产、消费之间需要重排且尚无对应 lowering 时，由 Verify 报错。

reducer 分析也要从真实 For extent 检查输出维与 reducer shape 的关系，不直接复用「将 loop layout 的 `InputShape` 与 reducer shape 比较」的规则：非整除时前者可能是 P，后者仍是 E。部分和和结果 Buffer 的逻辑 shape 保留 E；entry 数量不按 `ceil(E/VL)` 概括，按 8\.3 节的专门映射确定（如 `acc[i]` 行归约是每个输出一个完整 entry）；tail 只影响有效输出和更新，不能扩大用户 Buffer 或结果 copy 的逻辑范围。

### 4\.6 LayoutInference 的 PTO 分支

VL 来自前端 lanes 参数，向量化维由 LayoutInference 选择并编码进 lane 布局。Verify 检查布局与当前 Buffer 访问是否兼容，Vectorize 按映射生成循环，不重新选择维度或 VL。

布局推导按以下规则处理：

1. `BufferUseDefCollector` 进入 PTO `SIMD_VF` 时读取 `tl.simdvf_lanes`，为区域内的 Parallel 建立 `[0,VL)` 的分析上下文；退出区域后恢复上下文。不借用外围 SIMT 线程范围，也不生成线程 launch。

2. 根据 Buffer 下标及 strides 选择候选维。所有 store 都须沿该维逐元素连续，load 可以连续或 lane\-uniform。使用 `q ∈ [0,Q)`、`lane ∈ [0,VL)` 检查地址规律，不要求候选维整除 VL；多个候选都满足时，单元素维按 4\.6\.1 节选择，其余情况优先最内层维。不以无法证明 padding 安全为由排除候选。

3. PTO SIMD 分流发生在 `ParallelOp::InferLayout` 入口：`in_pto_simd_vf` 为真时走 `ParallelOp::InferPtoSimdLayout`，按 4\.3、4\.4 节构造补齐 shape 的 lane 布局。原始 For 的 extent 保留，只改变布局的定义域；`vectorize_size` 仍为 1，不进入 `ComputePlanCandidate` / `PlanLoopPartition` / `GetVectorizeSize` 的 SIMT 选宽逻辑。

4. `LayoutInferencer` 将 `pto_parallel_loop_layout` 写在每个 Parallel 整体的最外层 For，value 描述完整执行输入空间。PTO SIMD 不写入 SIMT 使用的 `parallel_loop_layout` 或 padding guard。写回按 `SIMD_VF` 块判断：只有块内含 PTO lane root 时才下降写回，其余块保持既有早退行为。

5. 调度和多缓冲变换保留有效注解。增加 Buffer 版本槽位下标不改变逻辑点分配；改变迭代空间的变换须更新布局，或使旧布局失效。

只有 Fragment Buffer 访问的 Parallel 需要使用已知 Buffer 布局或规范选维规则建立执行布局。`InferPtoSimdLayout` 保留 UB 连续地址选维逻辑，为 Fragment Buffer 和 reducer 增加单独的访问分派；普通 Fragment Buffer 的生产和消费共同约束同一份 Buffer 布局，已有布局时检查兼容性，不按当前循环重新覆盖它。reducer 的部分和布局推导见 8\.3 节。

#### 4\.6\.1 单元素维和退化二维的边界

普通 `Simplify` 会利用循环范围化简 `Parallel(1)` 的循环变量，例如将 `b[i] = a[i]` 化为 `b[0] = a[0]`。目标设计不在通用 Simplify 中加入 PTO 专用保护，也不增加单点计算 lowering。各类输入的处理如下：

|输入形态|处理|说明|
|---|---|---|
|一维 `Parallel(1)`|分类拒绝|只剩单点计算，目标边界内没有单点 lowering|
|二维 `(1,1)`|分类拒绝|两个维度都退化，无法形成支持的连续向量访问|
|紧凑 `(127,1)`|选择 i 并向量化|生成一个含 127 个有效 lane 的 chunk|
|紧凑 `(1,127)`|选择 j 并向量化|生成一个含 127 个有效 lane 的 chunk|
|`(3,1)`、行距 128|分类拒绝|i 方向是离散地址，留待 gather/scatter 或等价 stride lowering|

候选分析始终先按实际 Buffer 下标、strides 和所有访问证明合法性。当两个候选都合法且只有一个维度非退化时，选择非退化维；其他情况沿用内层优先。该规则只处理已经满足 continuous 证明的退化二维，不引入通用性能代价模型。

`(3,1)`、行距 128 的访问需要逐 lane 的离散 gather/scatter 或等价的 stride load/store，还要定义无效 lane 的读写屏蔽和 tail 可读范围约定。目标设计不因 shape 看起来可以降维而放宽 continuous 访存契约。

`T.Parallel(1)` 的单点边界只针对普通 Parallel。累加器 shape `[1]` 的 fill/finalize 直接依据 Buffer 布局转换（见第 8 章），不要求先构造一个单次 Parallel；也不能因为这类 `[1]` 累加器处理已定义，就宣称普通 `T.Parallel(1)` 已被支持。

后续若需要支持单元素 Parallel，有两条路线：

1. 在 Simplify 前标记 PTO `SIMD_VF` 内的 Parallel，让 Simplify 在外层上下文中判断 extent 是否化简为 1，并保留循环体中的下标依赖；

2. 在 LayoutInference 前消除 extent=1 的循环维，让剩余 Parallel 重新推导布局，全部维度消除时进入单点计算路径。

两条路线都需要补齐 BF16、FP8 的单有效 lane 向量计算，以及必要的 PTOAS 外提能力。目标设计不选择其中任何一条。

### 4\.7 地址归一化与 load/store 选择

布局求逆得到 `indices_after_inverse` 后，使用 `buffer->ElemOffset(indices_after_inverse)` 计算相对于 `buffer.data` 的元素偏移。该接口统一处理多维下标、显式 strides、`elem_offset` 和版本槽位维；strides 为空时按紧凑行主序计算。

连续访问在当前 chunk 内满足：

```text
data_offset(lane, q, outer) = start_offset(q, outer) + lane
```

`start_offset` 包含 `elem_offset`、版本槽位下标、外层坐标和 chunk 偏移，不含 lane。地址分析确认各 lane 按元素步长 1 连续访问；步长为 2、lane 分组取模等不满足该关系的访问不属于 continuous 路径。

构造访存指针时，将逆映射下标中的 lane 置为 0，得到本次向量访问的起始下标：

```text
start_indices = Simplify(Substitute(indices_after_inverse, lane = 0))
start_offset = ElemOffset(start_indices)[0]
ptr = tl.access_ptr(BufferLoad(buffer, start_indices), VL, rw_mask)

read:  vload(ptr, 0, size=VL)       // rw_mask = 1
write: vstore(value, ptr, 0, mask) // rw_mask = 2
```

这里的 `start_offset` 用于地址分析，生成 `tl.access_ptr` 时保留原 Buffer 和 `start_indices`。`LowerAccessPtr` 根据这些下标、strides 和 `elem_offset` 生成 `tvm_access_ptr`，因此 `elem_offset` 只计入一次。

vecadd 的 Buffer 形状为 `(2, 8192)`，stride 为 `(8192, 1)`，结果为：

```text
indices_after_inverse = (iter % 2, q * VL + lane)
start_indices = (iter % 2, q * VL)
start_offset = elem_offset + (iter % 2) * 8192 + q * VL
```

LowerAccessPtr 输出的指针描述本次访问的 `[start_offset, start_offset + VL)` 区间及读写属性：

```text
tvm_access_ptr(TypeAnnotation(dtype), buffer.data, start_offset, VL, rw_mask)
```

非整除路径沿用 extent=VL 的指针描述。连续 load 读取这 VL 个元素；store 的指针范围保守覆盖 VL 个元素，实际写入由 mask 限定为前 `min(VL, remaining)` 个元素。load 的 extent 不缩成 tail 大小；store 的描述范围也不表示 VL 个元素都被写入。

向量 load 和 store 共用这套地址构造，VMI 额外 offset 均为 0。offset 和 extent 以所访问 Buffer 的元素为单位。后续版本布局改写和访存分析沿用 `tvm_access_ptr` 的区间描述；自动调度和 InsertSync 已在 Vectorize 之前完成。Vectorize 不自行重算普通 Buffer 的版本槽位，也不生成流水线同步。

lane\-uniform 的标量 load 在当前 chunk 内保留为 `BufferLoad(buffer, indices_after_inverse)`，绑定到标量 Var；其下标经化简后不含 lane。后续 Buffer flatten 和 codegen 沿用现有标量访存路径。该值用于向量运算或向量 store 时再生成 `vbrc`。

UB store 按实际地址与 RHS 类型选择指令：

|形态|输出|
|---|---|
|Parallel 外普通标量地址、标量 RHS|普通标量 store|
|continuous 地址、向量 RHS|NORM `vstore`；lane k 写入 `base+k`，稀疏 mask 不压紧地址|
|uniform 地址、向量 RHS|`vscatter`；以该地址为 base，各 lane 的 offset 为零，实际写入者由 store mask 决定|
|direct Reduce 的单元素结果|一元素 VMI store，由 PTOAS lowering 为 `1PT`，见 7\.2 节|

uniform scatter 的指针只描述目标单元素，offset 向量使用现有 scatter ABI 的类型和寻址单位。不能用 NORM store 表达多 lane 同址写入；否则会访问相邻元素。Verify 只检查地址模式、类型、predicate 等转换条件，不证明唯一写者或判断数据竞争。多个 active lane 对同址冲突写入时，不承诺确定的最终值。单侧条件 store 的 predicate 必须保留，具体规则见 5\.3、9\.1 节。

目标设计只接收 `ElemOffset` 返回一个物理轴、Buffer 元素 dtype 自身不带 lanes 的形式。地址分析无法确认 continuous 或 lane\-uniform 关系时报告不支持。

单次访存的分析结果记为 `MemoryAccess`，供 Verify 判断模式、Vectorize 构造指针。它需要保留以下信息，具体 C\+\+ 字段名由实现确定：

|信息|用途|
|---|---|
|原 Buffer、元素 dtype、读写属性|确定存储对象、类型和 `rw_mask`|
|`indices_after_inverse`|保留版本槽位维、外层坐标、chunk 和 lane 的下标关系|
|`data_offset`|分析相对于 `buffer.data` 的完整元素偏移|
|`access_pattern`|记录 continuous 或 lane\-uniform 等单次访存模式|
|`start_indices`、`start_offset`|前者用于构造 `tl.access_ptr`，后者用于分析；生成指针时不重复叠加两者|

`MemoryAccess` 不保存已经生成的向量 Var，也不表示 load 缓存或跨访问依赖关系。

单次访存分类使用有界等价证明。`data_offset` 是对逆映射下标调用 `ElemOffset` 所得的第一个物理轴偏移，`start_offset` 是 lane 置 0 后的偏移。把 `lane ∈ [0,VL)`、`chunk ∈ [0,Q)` 和保留坐标的原始 Range 绑定到 `arith::Analyzer`，再证明偏移等于 `start_offset + lane`（连续）或 `start_offset`（lane\-uniform）。

候选选维、Verify 和 Vectorize 使用同一证明范围，包括最后一个 chunk 的全部 VL 个 lane。这个检查判断地址规律，不证明 tail load 的地址可读。`DetectIterMap` / `InverseAffineIterMap` 用于布局求逆；在补齐后的完整执行空间上仍要求 `arith::IterMapLevel::Bijective`。

这套地址分析由 Verify 和 Vectorize 共享。Verify 只读消费模式及合法性结果；Vectorize 再按同一结果生成标量 BufferLoad 或 continuous VMI。地址分类不依赖 Vectorize 已生成的 Var 或表达式转换状态，避免两个 pass 对同一访问采用不同的判定规则。

## 5\. Verify、Legalize、Vectorize 的职责

### 5\.1 共同上下文与状态模型

Vectorize 在同一个 VF 上统一处理五个模块，一次转换共用以下信息：

|信息|用途|
|---|---|
|Parallel 迭代映射|确定向量化维、保留维、q/lane 和 tail|
|值的标量/向量表示|确定计算宽度，按需广播标量|
|访问分类|区分普通 UB 连续/标量访问和 Fragment handle 基址的 VMI 访问|
|Fragment 存储映射|保存源 Buffer→backing→handle、count、dtype、VL 和 entry offset|
|Fragment UB 预留|消费 PlanPtoSlotStorage 记录，生成声明并核对容量，不重新分配|
|执行 mask|组合当前循环有效范围与源条件|
|Var/Bind 词法关系|按类型映射标量或向量 SSA 值，与 Buffer 存储映射分开管理|

这些信息只在本次转换的编译器上下文中存在，不增加 Fragment 存储的 `logical_scope` 或生命周期参数。控制流与循环关系由输出 IR 的结构表达。

状态模型必须按以下方式划分，不能整体按 Parallel 隔离：

以下信息属于整个 SIMD\_VF 的共同上下文：

- Buffer 身份和 `layout_map`；

- Fragment Buffer→backing 映射；

- reducer partial/result/seed 的存储布局；

- Var/Bind 的词法定义使用关系。

以下信息可以按 Parallel 独立管理：

- q 循环；

- E/P/Q；

- 当前 Parallel 的 loop mask 和 remaining；

- 当前访问的具体 entry。

二维的两层 For 共用一个 lane 布局和一套变换状态，不将内层 For 当作独立 Parallel 再次转换。循环、分支和不同 Parallel 内的节点共用 Buffer 映射，同时按词法作用域管理变量和当前 mask。转换按源程序的语句顺序进行。

### 5\.2 `VerifyParallelToPTO` 输入契约

Verify 只读检查输入，不修改 TIR；它按支持的转换规则检查源程序，接受后续 Legalize 能规范化的形式，遇到第一个错误时终止编译。

**处理区域。** 只处理含待转换 Parallel 的 `SIMD_VF`，跳过 `SIMT_VF` 和没有 Parallel 的纯手写 VMI 区域。纯手写 VMI 不要求提供 `pto_parallel_loop_layout`，也不受 Parallel 路径的类型限制，已有通用检查照常执行。

待转换区域不能出现线程绑定：`ThreadBinding` 的 For 形式与 `thread_extent` 的 AttrStmt 形式都拒绝，且不限 `threadIdx`（`blockIdx` 同样拒绝），并列与包裹两种位置同等处理。区域内也不能混合手写 `tl.vmi.*` 和 Parallel。一个区域内允许多个顺序排列的 Parallel 整体。

接受包住 Parallel 整体的 `Serial` 和 `Unrolled` 循环，后者只携带展开提示。`Vectorized` 包装仍拒绝，不将该限制扩到并列位置；`ThreadBinding` 按区域禁令处理。诊断给出实际 `ForKind`，线程绑定还给出绑定名。

**检查项：**

|类别|要求|
|---|---|
|lanes|`tl.simdvf_lanes` 是编译期整数，取值为 64、128、256|
|循环范围|各维 min 为 0、extent 为编译期正整数，原始 Parallel For 的 step 未指定或化简为 1；选中维允许非整除，chunk 数为 `Q=ceil(E/VL)`；选中维 `E=1` 的单点形态按 4\.6\.1 节分类拒绝|
|循环结构|一维 Parallel 或两层紧邻 Parallel For，布局只挂在整体最外层。拒绝三维及以上嵌套，Parallel 体内的 Serial、Pipelined、While 循环，以及不可规范化的条件分支（准入与拒绝的分界见本节末）。区域内串行 For 可包住一个或多个整体，串行变量可作 uniform 标量，整体之间可夹按序执行的标量语句（如 `sa[k] = t1[k]`）。非整除整体保留嵌套循环，remaining 在每次外层迭代、进入 chunk 循环前初始化|
|布局|`pto_parallel_loop_layout` 是可逆的 lane 布局；选中维的 `input_size` 必须恰好为 `P=ceil(E/VL)*VL`，未选中维等于原始 extent；lane 范围为 `[0,VL)`，`replicate_size=1`，映射符合 4\.3、4\.4 节的规范形式。求逆失败在调用处转换为分类布局诊断。Buffer 布局按 4\.5 节的规则核对|
|访存|不允许 SIMD\_VF 直接访问 GM；每次 UB 访问须能确认为 continuous 向量 load/store，或 lane\-uniform 标量 load、普通标量 store 及 uniform 地址的向量 scatter store；不支持 volatile 访问（带 `tirx.volatile` 注解的分配按函数级收集、转换区域内的 `volatile_scope` marker，Verify 拒绝）|
|表达式和语句|BufferLoad、常量、标量变量、循环下标、Add/Mul/Div 等算术、比较、Select、布尔运算、有限 Cast、受支持的数学 Call（如 rsqrt；各操作按 5\.6 节的 dtype/VL 支持范围核对）、原 TIR 中的 Bind 和 BufferStore；`Evaluate` 作为死表达式的承载形态也接受，其表达值走与 Bind 右侧、store 值相同的检查；Bind 的右侧也须满足相同范围|
|类型|操作数、结果 dtype 和 VL 符合对应 VMI 操作及 PTO codegen 的约束；FP4 等特殊打包类型暂不支持，见 5\.6 节|

**条件分支的准入与拒绝分界。** 可由 Legalize 规范化的形式属于准入：两臂写同一位置的简单 `IfThenElse`（→Select）、`Min/Max`（→比较\+Select）、可安全求值的布尔表达式（见 5\.3 节）；每层都满足两臂写同一位置、且条件与表达式可安全求值的嵌套 `IfThenElse`，可逐层规范化，同样准入。reducer update 上 `i < d` 这类能生成安全逐 lane mask 的条件（见 8\.6 节），以及单侧条件 store（保留 predicate，见 5\.3、9\.1 节）也属准入；其目标可以是 continuous 或 uniform 地址。其余形态拒绝：分支内对多个位置的写入相互影响、两臂包含依赖条件才能合法执行的 load、提前退出，以及其他不能归入上述形式的结构。

向量 Cast 只接受同类型转换，或能直接映射为语义一致的 VMI 转换的组合，不自动经中间类型拆分。类型、舍入和饱和等规则见 5\.6 节。shuffle、原子操作、未定义 lowering 的 Call 和其他副作用均报错。Fragment Buffer 的变址按 entry/lane 兼容性检查（见第 6 章和 8\.6 节）：常量或循环变量 entry 索引合法；需要逐 lane gather、跨 entry 拼接或 lane 重排的访问报错。普通 UB 的动态变址由 4\.7 节的地址分析判定，不能确认为 continuous 或 lane\-uniform 的报错。跨 lane 的 Reduce 只接受第 7 章 direct reduce 和第 8 章 materialized finalize 两种形式；其他跨 lane 归约形式报错。多个 BufferStore 按源程序顺序转换。

访存检查调用共享的只读地址分析，只判断每次访问的地址规律，不比较不同访问之间的依赖或别名。

`E % VL != 0` 时，未补齐的 `[E]` 循环布局不满足契约；也不接受任意更大的整除 shape。`E % VL == 0` 时 `P=E`，原有布局保持合法。二维由 shape 和规范映射共同确定选中维，不能把两维一起补齐或在消费布局时另选一维。退化维使多种解释等价时沿用内层优先。

**Fragment Buffer 与 reducer 相关检查。** Verify 按 Buffer scope 区分 Fragment Buffer 与普通 UB，再检查访问映射：Fragment Buffer 的跨语句访问必须符合 `layout_map` 中的 entry/lane 映射；`acc[0]` 这类固定逻辑索引可以对应各 lane 独立保存的部分和，不能按共享 UB 的同址写入处理，也不能按 shape 或固定索引推断各 lane 值相等（见 8\.3 节）。Verify 还检查：前置展开生成的 finalize 参数和结果 copy 是否受支持；init/update/finalize\_v2 和 `local.reducer` 已被 materializer 消费；每个待转换 Fragment Buffer 是否有匹配的 Fragment 存储规划，count、类型和容量是否与当前布局一致，backing 是否仍在合法作用域内；不把 backing 当作用户计算中的普通 UB Buffer。UB 写回按 4\.7 节的地址和 RHS 类型分类；uniform 地址本身不是拒绝理由，不要求证明条件只选中一条 lane。Verify 不做 liveness、寄存器压力或物理存储决策。

**检查边界和诊断。**`VerifyParallelToPTO` 和 `VectorizeParallelToPTO` 均不检查不同逻辑点之间的 RAW、WAR、WAW 依赖，不证明 Buffer 之间是否存在别名，也不因未知别名而拒绝输入。程序满足 `T.Parallel` 并行迭代语义是输入契约，由程序员负责；Verify 通过只表示满足本 pass 检查的结构和指令约束，不表示已经证明逻辑点独立性。多个 active lane 的同址冲突写入没有确定结果保证。现有 pipeline 中其他检查仍按原有配置执行。

tail load 可读范围和 padding 初始化也属于调用方前提。Verify 不增加 Buffer padding 证明，不要求额外的「安全 padding」注解，也不因无法证明可读范围而拒绝；地址模式、数据类型及其他既有检查仍执行。

区域级语句（Parallel 整体之外的串行语句、区域级 `AssertStmt` 的条件、分支与循环条件等）同样按表达式位置检查变量定义：`AssertStmt` 只有条件字段（没有 body），条件里引用未定义变量会在 Verify 阶段被拒绝，而不会带着未定义引用进入后续 lowering。`T.Parallel` 体内出现 `AssertStmt` 仍按单元语句白名单拒绝。

前端负责 `lanes` 参数和 `T.SimdVF` 注解，`LayoutInference` 负责选维及布局构造，Verify 检查本节的输入契约。Vectorize 消费已验证的映射生成 VMI；布局和地址规则由共享分析实现。

错误信息包含 checker 名称、`SIMD_VF`/Parallel 位置、检查类别、实际值和失败原因。已确认本区域 lanes 后才打印该值；提前失败时省略，不能沿用前一区域的 lanes。例如：

```text
[VerifyParallelToPTO] T.SimdVF lanes must be one of {64, 128, 256}, got 32.
[VerifyParallelToPTO] pto_parallel_loop_layout is missing on the outermost T.Parallel.
```

### 5\.3 `LegalizeParallelToPTO`：规范化语句，保留标量并行表示

Legalize 接收前置 pass 已处理过的标量并行 TIR，承接 If→Select、Min/Max→Select 和布尔表达式改写。reducer 的语法规范化、初始化和更新展开不属于它的职责，也不在这里再次调用前置 pass。输入已经满足约定时，Legalize 原样保留。输出仍是标量并行 TIR，`tl.vmi.*` 和 Fragment 声明和 VMI 访存由 Vectorize 生成。

|输入 TIR|Legalize 输出|作用与前提|
|---|---|---|
|`Min(a, b)`、`Max(a, b)`|比较加 `Select`|表达式改写，让 Vectorize 使用已有比较/选择路径；只用于已证明等价的 dtype 和语义|
|两个分支向相同位置写入的简单 `IfThenElse`|一个 `BufferStore(Select(...))`|仅在条件和两臂表达式可安全求值、读写顺序保持时使用；store 使用 parent mask|
|所有单侧条件 store|带 predicate 的 `BufferStore`|保留原始写入条件；continuous、uniform 和 Fragment Buffer 使用同一条件表示，由 Storage 按存储种类消费|
|`Select(And/Or/Not(...), a, b)`|嵌套 Select|布尔表达式展开，限定为可安全求值的布尔表达式|

**Min/Max。** 对已经绑定的 `int32` 值 a、b，改写可以表示为：

```text
# 输入
y = Max(a, b)

# 输出：仍然是标量表达式
y = Select(a > b, a, b)
```

这一步不生成 `vcmp` 或 `vsel`。后续 Vectorize 根据 a、b 的 lane 关系决定使用标量计算还是向量比较和选择。浮点 NaN、有符号零等语义需要单独核对，不能把上述整数示例直接套到所有浮点类型。

**简单条件写入。** 若 c、a、b 已经求值，两个分支写入同一个位置：

```text
# 输入
if c:
    out[i] = a
else:
    out[i] = b

# 输出
out[i] = Select(c, a, b)
```

原条件只求值一次。若两臂含依赖条件才合法的 load，或存在不能安全合并的写入，就不能套用 If→Select。单侧条件 store 保留 predicate，不补读目标旧值构造普通 Select；Vectorize 将 predicate 与 parent mask 合成，普通 UB 和 Fragment backing 的写入都使用该 mask。Fragment masked vstore 保留 inactive lane 的旧值（见 9\.4 节）。表达式中的 Select 仍生成 vsel；Legalize 不扩大条件保护的 load 范围。

嵌套条件按作用域组合 predicate。只有双侧同址 store 可以合并为 Select：数据由 c 选择，写入范围由 parent mask 限定。`vsel` 的条件只选择值，不能代替 store 的执行 mask。

**Select 中的布尔组合。** 对可安全求值的布尔值 c1、c2：

```text
Select(And(c1, c2), a, b)
    → Select(c1, Select(c2, a, b), b)

Select(Or(c1, c2), a, b)
    → Select(c1, a, Select(c2, a, b))

Select(Not(c1), a, b)
    → Select(c1, b, a)
```

这里处理的是 Select 的条件表达式。依赖短路保护的读取仍需保持执行条件，不能因为改成了嵌套 Select 就无条件执行。

普通 `x_frag[i] = x_ub[i]`、已经展开的 binary 更新、`T.fill`、`T.copy`、materialized finalize 和 `T.reduce_*` 都直接交给 Vectorize。

### 5\.4 `VectorizeParallelToPTO`：统一转换整个 VF

Vectorize 先分析 Legalize 的输出，再改写该 VF。转换共用 5\.1 节的共同上下文，按源程序的语句顺序进行。算术、存储、控制流和归约分别实现为模块：

```text
VectorizeParallelToPTO
  ├─ Parallel：q 循环、二维选维、E/P/Q、tail mask
  ├─ Expression：binary、cast、compare、broadcast、Select
  ├─ Storage：UB 访问、Fragment Buffer→backing/handle、fill、copy
  ├─ ControlFlow：条件、mask、vsel、条件更新
  └─ Reduce：direct reduce / reducer v2 finalize / 结果发布
```

pass 调用共享的布局求逆 helper，按 4\.3～4\.4 节的映射生成循环：保留非向量化维坐标，将连续维写成 `q * VL + lane`。它使用已确定的 VL，不调用 `GetVectorizeSize`，也不复用通用 `LowerParallelLoop` 的线程循环和 guard 生成逻辑。

共享映射结果区分 `logical_extent=E`、`padded_extent=P`、`chunk_count=Q`、`tail_lanes=E%VL`。二维保留原始两维 extent 和选维信息。布局求逆及执行范围使用 P，mask 初值使用 E；实现中使用 `extent / lanes` 的位置应按用途逐一改用 E、P 或 Q，不能把 E 全部替换成 P。

非整除整体生成一个 chunk 循环；输出中 `remaining` 的初值、更新和作用域见 9\.2 节。二维每次进入保留维迭代时重置初值；多个顺序 Parallel 各自初始化，不能沿用前一个整体的剩余数。

lane 是内部地址分析的占位符，不能作为未绑定的标量 Var 留在输出中。连续访存的 lane 偏移由 VMI 隐含表示；下标作为计算值时，按 5\.5\.3 节生成显式向量。

Vectorize 消费原 Fragment Buffer 分配，保留 backing，并生成 `tl.vmi.declare_fragment_buffer` 和 handle 基址的 `tl.vmi.vload/vstore`。内部后置检查从声明恢复 backing/count/dtype/VL，核对来源、作用域、定义使用、元素 offset、访问范围以及向量与 mask 宽度，并拒绝残留待处理的 Parallel/reducer。Verify 检查源输入，后置检查定位转换错误，不新增独立验证 pass。

### 5\.5 表达式和语句转换

#### 5\.5\.1 值的分类

`VectorizeParallelToPTO` 按同一个值在各 lane 之间的关系决定保留标量、广播或生成向量计算。目标设计不引入显式的三态枚举对象：计算值的角色由转换结果的 dtype（标量或带 lanes 的向量）表示，地址与 lane 的关系由共享的只读地址分析（4\.7 节）判定；先按值转换、后发现仅由地址环境消费的发射结果由 5\.5\.5 节的局部 DCE 清理。这套关系是 pass 内部的临时状态，不写入 TIR，也不要求 Verify 维护。

值的分类如下（描述转换规则，不是独立的转换状态对象）：

|分类|语义|处理|
|---|---|---|
|`uniform`|所有 lane 使用同一个值，如常量、标量参数、当前程序点的 `scale_buf[0]`|保持标量；用于向量运算或向量 store 时广播|
|`lane_indexed`|可表示为公共 base 加已知的 lane 偏移，如整数下标 `q * VL + lane`|作为计算值转换时生成显式向量（`vci`）；仅被地址消费时直接输出不保留向量结果（5\.5\.5 节 DCE）|
|`varying`|没有可利用的 lane 数值规律，如连续 load 读出的数据、两个向量的乘积|使用普通向量表示；仍需满足具体操作的 dtype 约束|

`temp1[i]` 地址连续，可以使用 continuous vload，但读出的数据可能是任意值，不能据此推断数据也满足 `base + lane`。

访存模式保存在 `MemoryAccess.access_pattern` 中，由地址分析得到。`continuous`、`interleaved` 和 gather/scatter 等模式描述某次 load/store 如何访问内存，不作为值的分类。目标设计支持 continuous 向量 load/store、lane\-uniform 标量 load、普通标量 store 和 uniform 地址的向量 scatter store；不支持的访存模式或操作单独报告错误。

对 vecadd，分类关系为：

```text
i = q * VL + lane          → 值分类: lane_indexed
temp1[i] 的访问地址        → MemoryAccess.access_pattern: continuous
temp1[i] 的加载结果        → 值分类: varying
temp1[i] + temp2[i]       → 值分类: varying
```

`temp1[2*i]` 的地址仍有已知的 lane 规律，但步长为 2，不属于 continuous 访问；拒绝它的原因是访存模式不受支持。`scale_buf[0]` 在当前程序点加载为 uniform 标量，不代表 Buffer 的内容跨语句或跨 chunk 恒定，也不因此启用 load 复用。

表达式按 lane 依赖关系转换：

|原始表达式|PTO 表达|
|---|---|
|随 lane 变化的 BufferLoad|向量 load|
|地址与 lane 无关的普通 UB BufferLoad|标量 load|
|只由 lane\-uniform 标量组成的子表达式|保持标量|
|lane 相关的标量表达式|提升为向量表达式|
|标量和向量共同参与运算|标量值 \+ `vbrc`，后续可匹配 `vadds`、`vmuls`|
|标量和标量共同参与运算|保持标量|
|标量结果作为向量 store 的值|先完成标量计算，再广播结果|

上表的标量保留以目标标量指令能力为限：整数、地址和索引类标量运算保持标量；SIMD\_VF 内不使用标量浮点 ALU，运行时浮点标量运算的操作数广播后按向量计算（见 8\.5 节），不支持的组合由 Verify 诊断。Fragment Buffer 的读取不适用上表的标量 load 行，先按布局映射转换（见 8\.3 节）。

按需广播同时适用于向量算术和向量 store。对 `c[i] = alpha + beta`，`alpha`、`beta` 为整数标量时，加法保持标量，向量 store 消费该结果时再生成一次广播：

```text
sum_scalar = alpha + beta
sum_vec = vbrc(sum_scalar, size=VL)
vstore(sum_vec, access_ptr(c[q * VL], "w", VL), 0, mask)
```

`sum_scalar` 是否单独绑定由原 TIR 和 Bind 规则决定。alpha、beta 先完成标量加法，再广播结果，元素 dtype 和原有 Cast 保持不变。

标量 UB load 放在 chunk 循环内。Cast 保留原 lane 映射：向量 Cast 生成对应的 VMI 转换，lane\-uniform 标量的 Cast 保持标量。

#### 5\.5\.2 Visitor 和语句生成

变换沿用 TVM `StmtExprMutator` 做外层语句遍历；单元内部按有限白名单显式分派 Var、BufferLoad、算术与比较二元节点、布尔运算、Select、Div、受支持的数学 Call、Cast、Bind 和 BufferStore，而不是依赖 mutator 的默认 `VisitExpr_` 递归。死表达式以 `Evaluate` 承载，其表达值用同一套转换与准入检查改写（结果没有消费者）；这不表示可以承载任意副作用语句，副作用调用仍被白名单拒绝。表达式访问返回 `PrimExpr`，表示标量表达式或向量 Var；值的角色由该结果的 dtype（标量或带 lanes 的向量）表示。Add/Mul 显式分派到二元转换，后者递归转换 `lhs`/`rhs`，再根据转换上下文决定保留标量、广播或发射向量操作。

原 TIR 已有的 Bind 按原位置转换右侧表达式，生成标量或向量绑定，并记录原 Var 到转换结果的映射，保留原作用域。例如 `x = a[i]; c[i] = x * x` 只转换 x 定义处的 load，两个 x 引用同一结果。这是原程序已有的值复用；后续写入源 Buffer 不改变已经绑定的 x。

表达式转换产生的 VMI Call 通过 `EmitBind` 绑定，绑定语句追加到当前语句序列，返回的 Var 供父表达式使用。BufferStore 的处理先访问右侧表达式，将其产生的绑定语句放在对应 vstore 之前，再输出这组语句。同一 chunk 内的下一条原始语句随后转换，不把后续语句的 load 提前到前一条 store 之前。

例如，访问 Mul 时先转换 `op->a` 和 `op->b`，根据转换上下文判断是否需要广播，再生成标量 Mul 或绑定后的 VMI vmul。

#### 5\.5\.3 下标作为计算值

下标仅用于地址时，按 4\.7 节的地址分析直接构造本次访存的 `access_ptr`；若下标别名先按值转换、随后发现仅被地址环境消费，其向量结果由 5\.5\.5 节的局部 DCE 清除，直接输出不保留它。下标作为算术操作数或输出值时，才生成显式向量。对一维下标 `i = q * VL + lane`，使用 `vci(q * VL, size=VL)` 得到 `[q * VL, q * VL + 1, ..., q * VL + VL - 1]`，通过 `EmitBind` 绑定。

例如，`a`、`c` 和循环下标都是 int32 时：

```python
for i in T.Parallel(N):
    c[i] = a[i] + i
```

当前 chunk 的变换结果如下，VMI 调用按语义示意：

```text
a_vec = vload(access_ptr(a[q * VL], "r", VL), 0, size=VL)
i_vec = vci(q * VL, size=VL)
result = vadd(a_vec, i_vec, mask)
vstore(result, access_ptr(c[q * VL], "w", VL), 0, mask)
```

下标向量保留原整数 dtype，原 TIR 中的 Cast 按既定转换规则处理；`vci` 的 dtype 和 VL 组合也必须受支持。对 `2 * i` 等计算值，先生成 `i_vec`，再按已有 Mul 规则转换，不能只改变 `vci` 的 base 而漏掉 lane 步长变化。

二维路径中，连续维下标按同样方式生成向量，保留的外层坐标仍是 uniform 标量。pass 的直接输出只保留实际参与计算的下标向量。对于先按值转换、后发现仅由地址别名环境消费的 Bind，通过 5\.5\.5 节的局部 DCE 清理无用向量链。

#### 5\.5\.4 语句顺序和作用域

Parallel 体内的语句按原始顺序逐条转换。前一条语句绑定的向量 SSA 值可以作为后一条语句的输入；如果后一条语句通过 BufferLoad 读取前一条语句写入的 Buffer，则保留实际 store，再生成 load。除 5\.5\.5 节允许删除的死绑定及其生产链外，不跨语句或跨 Parallel 重排、删除、延迟或合并访存。

语义保持以合法的 `T.Parallel` 输入为前提：保留每个逻辑点内的表达式、类型转换和语句顺序，以及顺序 Parallel 之间的执行顺序。不同逻辑点遵循 Parallel 的并行语义，不额外规定串行次序。依赖和别名的责任边界见 5\.2 节。

向量中间值的 Bind 作用域按 chunk 划分。整除整体的全量 mask 可以在 VF 内、chunk 循环外复用；非整除整体在每个 chunk 内根据 `remaining` 生成 mask。其他依赖 q、lane 或当前地址的向量值仍在当前 chunk 内绑定。

#### 5\.5\.5 局部 DCE

`VectorizeParallelToPTO` 内保留局部死代码消除（DCE），让直接生成的 TIR/PTODSL 更干净，便于检查和调试。它在 chunk 体转换完成后，清理结果无人使用、且已确认可安全丢弃的 Bind 及其生产链。地址专用下标经值转换后无人使用的死向量链属于必须清理的项；源程序本就无用的普通 UB load 删除是明确选择的额外优化收益，后续可留给 PTOAS。有用的 load/store、元数据引用和语句顺序必须保留，不做重复 load 合并或跨写操作复用。Fragment vstore 的写效果必须被正确保留，Fragment 存储方案本身不要求关闭或重做普通 SSA 清理。

后续可以将这部分代码清理交给 PTOAS 的 canonicalization/DCE，以减少 TileLang 代码和维护成本。迁移时需确认正式 PTOAS pipeline 能清除相应用例，并将「输出无死链」的验收移到 PTOAS 优化后的 IR；TileLang 的原始 dump 可以保留待清理操作。无论在哪一层做 DCE，Vectorize 生成的 IR 在清理前就必须合法、语义正确，不能依靠删除死代码来掩盖非法 IR。

非整除路径的 `remaining` 使用 9\.2 节的可变标量表示。局部 DCE 只清理计算体中的死 Bind，不能把循环尾部的标量更新当成无人使用的临时值删除；初始化、mask 使用和更新必须保留正确的先后关系。

### 5\.6 VMI 构造、dtype 和 Cast 支持规则

`VectorizeParallelToPTO` 使用 C\+\+ helper 构造 TIR `Call` 和 `Bind`，与 Python 前端 `tilelang.ascend.language.vmi` 共用相同的 op 名、位置参数和 annotation 约定，不调用 Python API。`size`、`to_dtype` 等参数按 `vmi.py` 的约定保存在 Call annotations 中。

helper 根据原始表达式和具体 VMI 操作确定元素 dtype。逻辑向量使用 `element_dtype.with_lanes(VL)`，逻辑 mask 使用 `DataType::Bool(VL)`；操作、dtype、VL 组合还须通过后端支持范围检查。

`vload`、`vbrc`、`vci` 和 `vcvt` 不接收 mask。`vadd`、`vmul` 使用当前执行 mask，外部 UB 的 `vstore/vscatter` 还需合入源 store predicate。无内层条件时使用当前 chunk 的 mask：整除路径为全量 mask，非整除路径为 `create_mask(remaining, size=VL)`。Cast helper 按 `vcvt(source, to_dtype, ...)` 构造，保持 VL 个逻辑 lane 及已有转换属性；不增加 masked vcvt 接口。

Expression 与 ControlFlow 模块的操作映射如下（逻辑 TIR/VMI lowering 名称按当前实现约定）：

|源 TIR 节点|逻辑 VMI|mask 与 inactive lane|
|---|---|---|
|Add/Sub/Mul（受支持 dtype）|`vadd`/`vsub`/`vmul`；标量参与时可匹配 `vadds`/`vmuls` 等 vector\-scalar 形式|使用当前 chunk mask；inactive lane 的结果不由 masked store 写出；需要保留旧值的场景用 `vsel` 合并（见 9\.4 节）|
|Div|`vdiv`；FP32 已有设计依据（见 8\.5 节），VMI 没有独立精度参数，结果按端到端数值核对；其余 dtype 组合按 5\.6\.2 节核对，未核对的由 Verify 拒绝|同上|
|`T.rsqrt(x)` 等受支持数学 Call|按既定分解生成，如 `vsqrt(x)` 后以广播的 1 做 `vdiv`（见 8\.5 节）；不新增带精度参数的接口|同上|
|Min/Max|Legalize 已规范化为比较加 Select（见 5\.3 节），Vectorize 不直接接收|—|
|比较（LT/LE/GT/GE/EQ/NE）|`vcmp(lhs, rhs, mask, cmp)` / `vcmps(src, scalar, mask, cmp)`：以当前执行 mask 为 seed，生成已含当前 mask 约束的比较 lane mask|结果是 mask，不产生数据 lane|
|Select / 条件更新|`vsel(cond, a, b)`|cond 外 lane 取 b，是 merge 语义而非清零；条件更新中 b 为旧值（见 9\.4 节）|
|条件的布尔组合|`mask_and`/`mask_or`/`mask_not`，作用于 `boolxVL` 逻辑 mask|只作用于 mask|
|整数的 And/Or/Not|`vand`/`vor`/`vnot`|使用当前 chunk mask|

dtype/VL 组合按本节及 5\.6\.2 节的支持范围核对：未核对的组合、FP4 等打包类型（5\.6\.2 节）和 FP8 算术（5\.6\.2\.1 节）由 Verify 拒绝。

整数标量源（VCI 的常量与动态输入、VBRC 的动态整数输入）在 PTODSL 生成时统一做 signless 桥接：内层以同位宽 signless 标量构造（常量打印为 `pto.i<bits>(值)`，动态表达式包 `scalar.cast(expr, pto.i<bits>)`），外层用 `vinterpret_cast(..., to_dtype=原 TIR 元素类型)` 按位重解释恢复 signedness——同位宽、同 lane 数、无数值转换。这样内层产出的 signless 向量满足 PTOAS 多物理向量展开时 `arith.constant`/`arith.addi` 偏移合成的 signless 约束（VL\>64 时第 2\+ 个物理向量的 ramp 需要 `base+64k` 偏移常量），出口类型又与 `vload` 的元素类型一致。VBRC 常量与浮点输入不经此桥接：广播展开无偏移算术，typed\-literal 的带符号类型与 vload 匹配，保持原打印路径。

#### 5\.6\.1 中间值和 `EmitBind`

先构造结果表达式，再用其 dtype 创建 Var。`EmitBind` 统一追加绑定语句并返回该 Var，接口示意如下：

```cpp
Var EmitBind(const std::string& name,
             PrimExpr value,
             Array<Stmt>* statements) {
  Var result(name, value.dtype());
  statements->push_back(tirx::Bind(result, value));
  return result;
}
```

名称由调用方生成。每个 Var 只定义一次，定义位于使用之前；这些临时值不额外分配 Buffer。TVM `Bind` 要求 Var 与右侧表达式的 dtype 一致。

|操作|结果 dtype 来源|
|---|---|
|连续 `vload`|源 Buffer 元素 dtype，增加 VL 个 lanes|
|标量 `BufferLoad`|源 Buffer 的标量 dtype|
|`vbrc`|输入标量 dtype，增加 VL 个 lanes|
|用于下标的 `vci`|原循环下标的整数 dtype，增加 VL 个 lanes|
|`vadd/vmul`|原 TIR 算术表达式的元素 dtype，增加 VL 个 lanes|
|`vcvt`|原 Cast 的目标 dtype，保持 VL 个 lanes|
|mask|`DataType::Bool(VL)`|

类型传播遵循原 TIR 的显式 Cast。例如 FP16 输入转 FP32 后相加，应生成 `float16xVL` 的 load、`float32xVL` 的转换结果和加法结果，不能根据输出 Buffer 的 dtype 改变源 load 的元素类型。支持哪些转换组合仍由 dtype 支持范围约束。

有返回值的 VMI 操作使用 `EmitBind`；`vstore` 生成 `Evaluate(Call(...))`。常量和纯标量子表达式保留为 `PrimExpr`，标量 load 在当前 chunk 中绑定。整除路径的全量 mask 在循环外绑定，非整除路径的 mask 在循环内绑定；向量计算结果均在当前 chunk 内绑定。`remaining` 的更新使用可变标量表示，不重复定义同一个 Bind Var。

#### 5\.6\.2 dtype 和 Cast 支持规则

目标类型范围采用通用元素寻址和逻辑 mask 约定：Buffer 元素类型自身不带向量 lanes，逻辑向量包含 VL 个元素，mask 为 `boolxVL`。FP4 等需要单独处理打包单位或改变 mask lane 数的类型暂不支持；待转换 Parallel 中的 load、store 或 Cast 涉及这些类型时，Verify 直接报错。未来增加支持时，需同时定义地址、向量类型和 mask 的对应关系，不能只放开 dtype 检查。

例如，手写 VMI 的 FP4 路径将两个逻辑元素打包为一个字节，并要求 store mask 的 lane 数为逻辑 FP4 元素数的一半。该路径不符合本设计的统一约定。已有手写 VMI 的类型处理继续沿用现有实现；此约定只约束待转换的 Parallel，不改变手写 VMI 对这些类型的支持。

dtype 支持范围按具体操作和逻辑 lane 数 VL 核对。向量 Cast 除了源、目标 dtype 和 VL 合法，还必须确认 VMI 的舍入、溢出和饱和处理与原 TIR 的转换语义一致；仅有同名类型或后端能够编译，不足以认定转换等价。

无法将某个向量 Cast 直接映射为语义一致的 VMI 转换时，由 `VerifyParallelToPTO` 报错。Vectorize 不自动引入 `源类型 → 中间类型 → 目标类型` 的转换链，避免额外舍入或数值范围变化。原 TIR 已显式包含的多次 Cast 按各节点分别处理。这里的直接映射指逻辑 VMI 层，物理寄存器拆分和指令选择仍由 PTOAS 完成。

具体的操作、dtype、VL 组合和转换属性矩阵需对照 VMI、PTO codegen 和 PTOAS 核对，并用转换用例验证；核对完成前不列为已支持。诊断应说明源类型、目标类型、VL，以及不支持或无法保持语义的原因。

#### 5\.6\.2\.1 FP8 转换矩阵

支持的 FP8 元素类型只有两个：`float8_e4m3fn`、`float8_e5m2`。`float8_e4m3` 是另一个 TIR 类型（不能因为 codegen 映射到同一个 PTO 名字就纳入），E8M0 / HiF8 / FNUZ / FP4 同样不支持。

|方向|形态|转换属性|
|---|---|---|
|FP8 → FP32|直接 `vcvt`，精确加宽|不生成 rounding/saturate|
|FP32 → FP8|直接 `vcvt`|默认 `rounding="R"` \+ `saturate="SAT"`|
|同类型 FP8 → 自身|identity，不生成转换|无；显式 `sat=True` 与显式空 `round` 都按缺省处理|

可传递的舍入模式（FP32→FP8）为 `{R, A, H, Z}`。前端别名先规范化再按方向集合检查：`rn→R`、`rz→Z`、`rp→C`、`rm→F`；C/F 不在上述集合内，因此 rp/rm 被拒（理由写明是 FP8 conversion policy 不接受，而非未知模式）。`rbits` 与 `round="rs"` 一律拒绝。`sat=False` → `NOSAT`，缺省或 `sat=True` → `SAT`；方向本身不需要属性而用户显式携带时**拒绝**，不静默丢弃。identity 仅接受缺省属性、显式 `sat=True` 或显式空 `round`（`{"round": ""}` 表示取后端默认，见下）。**空 ****`round`**** 的方向差异**：该字段显式存在但无模式，identity 接受；`FP32→FP8` 按默认 R（`sat` 单独决定）；`FP8→FP32` 因字段显式存在而拒绝（方向本身不需要属性）。Python `str` 与 TIR `StringImm` 两种写法行为一致。

TileLang 侧分类拒绝（都有 `[VerifyParallelToPTO]` 诊断）：FP8 算术（向量与 lane\-uniform 两种形态）、lane\-uniform 的非 identity FP8 Cast（标量 PTO 路径在 SIMD\_VF 内没有 FP8 支持）、**FP16/BF16→FP8**、FP8→FP16/BF16、FP8↔整数、E4M3FN↔E5M2、其他 FP8 变体，以及上述属性违规。以上拒绝都不会插入隐式的 `FP8 → FP32` 计算或任何中间 dtype；物理 part、layout 与寄存器拆分仍由 PTOAS 负责。

实现约定：转换属性从 Cast annotations 读取（同时接受 TIR Imm 与 Python 原生值两种写法；`sat` 只接受布尔或 0/1，其他值报错）；表达式 visitor 不进入 Cast 的 annotations，属性合法性只由转换计划判定。**属性安全对每一个 Cast 生效，不限于 FP8 方向**：任何方向上出现该转换无法兑现的属性（`rbits`/随机舍入、不可映射的舍入模式、未知 key、方向不需要的属性、与既定语义冲突的 `sat`）都分类拒绝，不静默丢弃；可映射的属性显式写进生成的 VMI 调用，不依赖 PTODSL 默认值。

**方向边界（按指令集）**：FP8 转换指令只有 FP32↔FP8；half（FP16/BF16）没有直转指令。因此 `FP16/BF16→FP8` 在 TileLang 侧直接拒绝，并提示用户显式经 float32（写成 `T.cast(T.cast(x, "float32"), "float8_e4m3fn")`），TileLang 不自动插入该中间转换。旁证：手写 VMI 上 half→FP8 直转会让向量核异常，经 float32 两步则正确，与指令集口径一致；PTOAS 的 vpto 用例里 fp8 目标一律来自 f32，也没有 half→FP8 直转的设备用例。

**证据口径**：SIMD wrapper 生成的 `f162f8e4m3`/`bf162f8e5m2` 等 LLVM intrinsic 只说明 BiSheng 前端能生成该 intrinsic，**不作为**本节的 ISA 依据、PTO 支持矩阵或设备验收依据。

## 6\. Fragment Buffer 的 UB backing 与 memref 访问

### 6\.1 `PlanPtoSlotStorage`：在调度前规划 Fragment backing 的 UB 预留

`PlanPtoSlotStorage` 在 3\.2 节的位置读取待转换 PTO SIMD\_VF 的 `local.fragment` Buffer 和 `layout_map`。普通 Fragment Buffer、reducer 部分和、结果及可选 seed 的布局此时都应已确定，pass 据此计算每个 Buffer 的向量 entry 数量。

对每个 Fragment Buffer，规划记录包含：

|字段|含义|
|---|---|
|`fragment`|输入 TIR 中的 `local.fragment` Buffer|
|`vector_type`|一个 entry 的完整向量类型，例如 `vector<128xf32>`|
|`count`|按已选布局覆盖全部 entry 索引的数量，不能只用源元素总数除以 VL|
|`alignment`、`bytes`|backing 的字节对齐，以及 `align_up(count * sizeof(vector_type), alignment)`|
|`backing`|隐藏的 `shared.dyn` UB Buffer，作为该 Fragment 存储的物理预留对象|
|`fragment_mem_var`|预建的 Fragment 存储句柄 Var，实际定义由 Vectorize 插入|

例如 RMSNorm（`VL=128`、FP32）对应：

```python
# PlanPtoSlotStorage 新增的分配；原 Fragment Buffer 和标量读写此时仍保留。
x_fragment_backing = T.sblock_alloc_buffer((32 * 128,), "float32", scope="shared.dyn")
acc_fragment_backing = T.sblock_alloc_buffer((128,), "float32", scope="shared.dyn")
result_fragment_backing = T.sblock_alloc_buffer((128,), "float32", scope="shared.dyn")
```

它们依次对应 `x_frag`、`sum_sq_1` 和 `sum_sq_result`。无 seed 时固定预留 `16 KiB + 512 B + 512 B = 17 KiB`。UB 合并对齐为 32 B 时，这三个对象的大小都满足该对齐；若后端要求更强的对齐，应同时用于容量预算和最终分配。entry 按 `sizeof(vector_type)` 连续排列；目标设计只接受该大小满足所需对齐的类型，不隐式增加 entry 间距。末尾 entry 已包含完整 VL 的容量，不再额外预留一条尾向量。

规划记录随拥有分配的 SBlock 保存，引用实际 Buffer 和 Var，不按名称查找对象；具体注解名由实现确定。backing 分配放在支配 Fragment 存储使用点、且能够进入 kernel UB 合并的作用域中。它固定预留一份，不参与普通 Buffer 的 double/multi\-buffer 复制，也不与其他 UB 对象复用地址。规划信息还需使这些暂时没有访存的分配穿过调度和清理 pass，避免在 Vectorize 前被当成无用对象删除。

RMSNorm 的输入、输出 UB 即使采用双缓冲，上述 Fragment backing 预留仍为 17 KiB。`x_ub[t % 2, ...]`、`y_ub[t % 2, ...]` 保留版本槽位索引，`x_fragment_backing`、`acc_fragment_backing`、`result_fragment_backing` 不增加版本维。这里复用的是同一 Fragment 存储在顺序 VF 调用之间的 backing；不同 Fragment 存储对象仍各占独立地址。

Plan 保留原 Fragment Buffer 和标量计算，只增加分配与规划记录。Vectorize 根据记录定义 `fragment_mem_var`、生成声明和普通 VMI 读写并消费原 Fragment Buffer 分配。后续 pass 替换 Buffer 或 Var 时同步改写记录；只复制任务的不同执行阶段时，继续引用同一 backing。若某项变换确实引入需要同时保存的独立状态，必须在调度前纳入容量；目标设计不允许在 Vectorize 时静默增加 backing。

Plan 预建 fragment\_mem\_var，但不插入 Bind。声明在所属 VF 内定义并支配全部访问。count、lanes 为正整数常量，规划中的 vector\_type 继续用于核对；声明不依赖字符串类型参数。

### 6\.2 声明 Fragment 存储并使用 VMI 读写

每个 Fragment backing 保存 count 个完整逻辑向量 entry。TileLang 为进入 Vectorize 前已分配为 local\.fragment 的 Buffer 生成声明，把读写转换为普通 vload/vstore，fill、copy 和 finalize 共用同一映射。PTOAS 可以消除访存并传递寄存器值，也可以保留对既定 UB backing 的实际访问。

这包括用户 x\_frag 和前置 reducer pass 生成的部分和、结果及 seed Fragment。普通算术、广播、cast、VMI load 和规约结果通过 SSA 传递；只有源语句写 Fragment Buffer 时才生成对应 vstore，不为普通中间结果新增 backing。

TIR 接口使用 backing 地址、count 和 lanes；元素 dtype 从有类型地址取得：

```text
fragment_mem = tl.vmi.declare_fragment_buffer(backing_ptr, count, lanes)
value = tl.vmi.vload(fragment_mem, entry_index * lanes, size=lanes)
tl.vmi.vstore(value, fragment_mem, entry_index * lanes, active_mask)
```

每个 Fragment backing 对应一个扁平标量 memref，shape 为 count\*VL。声明返回 handle，动态 entry 放在 vload/vstore 的 offset 中，不为每个 q 创建新 memref。单个部分和位置使用 count=1；backing 地址始终由 Plan 和分配器提供。

|参数或行为|约定|
|---|---|
|backing\_ptr|已预留且满足对齐的 UB 地址，引用真实 Buffer/Var|
|count、lanes|正的编译期常量；count 是 entry 数，lanes 与 VF/layout 一致|
|element dtype|由有类型的 backing 地址取得，必须是受支持标量类型|
|element\_offset|entry\_index \* lanes；entry 可为常量或循环变量，在 \[0,count\) 内|
|读取|一次 vload 读取完整 entry，size=lanes|
|写回|vstore 使用 full 或 active mask；inactive lane 保留目标旧值|
|声明|不分配、不初始化、不承诺寄存器驻留或 noalias|
|作用域|在所属 VF 中定义并支配全部使用，不作为跨 VF 通信|

坐标到 entry/lane 的变换由 TileLang 完成；声明保留 count、lanes 和有类型的 backing 地址。它只建立关联，初始化由源 fill/store 转换。handle 的名称和定义随 Bind 保存，不能在后续访存中改用独立的 backing 指针而丢失 memref 关系。

fill/copy 根据 region 的 min/extent 和各自 Buffer layout 计算访问 entry。部分 fill 使用目标 mask；copy 分别计算源、目的 offset，不能用整个 allocation count 替代 region，也不能假定源目的起点相同。VF 区域级按源顺序处理 Parallel 外的 fill/copy。

### 6\.3 TIR、PTODSL 与 PTOAS 的接口对应

TileLang 只新增一个 Fragment 声明 intrinsic；读写复用通用 VMI。TIR handle 不要求新增公共类型，PTODSL 中对应普通 memref：

|TIR|PTODSL|语义|
|---|---|---|
|`tl.vmi.declare_fragment_buffer(backing_ptr, count, lanes)`|`pto.memref_declare(address, (count * lanes,), scalar_dtype)`|声明已预留 UB；address 为字节地址|
|`tl.vmi.vload(fragment_mem, offset, size=lanes)`|`pto.vmi.vload(fragment_mem, offset, size=lanes)`|固定 memref 基址，offset 为标量元素偏移|
|`tl.vmi.vstore(value, fragment_mem, offset, mask)`|`pto.vmi.vstore(value, fragment_mem, offset, mask)`|更新 active lane，保留 inactive lane 旧值|

节点编码为 `Bind(fragment_mem_var, Call(Handle, op, [backing_ptr, count, lanes]))`。count、lanes 是正整数常量；元素 dtype 从 backing 的有类型地址取得，不能从 Handle 推断。backing 使用覆盖全部 entry 的标准地址操作数，最终输出 signless i64 字节地址；memref shape 为 `count * lanes` 个标量元素。两类基址共用 VMI 访存：普通 UB 用现有 access\_ptr，Fragment 用声明返回的 handle。后置检查核对 handle 来源及 count/dtype/VL/offset，不把任意 Handle Var 都视为 Fragment 存储。

声明只绑定地址和存储属性，不产生数据访存；vload 读取可变内存，不能注册为纯函数；vstore 即使没有返回值也必须保留写效果。它们共用 tl\.vmi 前缀不表示效果相同。PTOAS 只在证明等价后消除、合并或移动访问。

`declare_fragment_buffer` 在所属 VF 内定义，支配其使用；不同 VF 不共享句柄。声明不承诺 noalias，PTOAS 按普通 memref 的别名关系优化。循环嵌套、条件和定义使用由 IR 表达，不增加生命周期参数。源 Fragment 分配和 plan 消费后，下游仍可扫描声明的地址操作数识别 backing，按实际 storage Var 去重，不要求修改 TVM Buffer。

### 6\.4 从实际 Buffer 推导 entry 数量与访问 offset

对连续的 RMSNorm 一维划分：

```text
i = q * 128 + lane
q ∈ [0, 32)
lane ∈ [0, 128)
```

Plan 先为 x 预留 `32 * 128` 个 FP32 元素，Vectorize 再转换分配和访问。下文示例用 backing Buffer 名称简写其基址操作数，完整地址形式见 6\.5 节：

```text
# layout: entry = i // 128, lane = i % 128
# Plan 已预留 4096 个 FP32 元素。
x_frag_mem = tl.vmi.declare_fragment_buffer(x_backing_ptr, 32, 128)

# BufferStore(x_frag, value, [i])
tl.vmi.vstore(vectorized(value), x_frag_mem, q * 128, active_mask)

# BufferLoad(x_frag, [i])
value = tl.vmi.vload(x_frag_mem, q * 128, size=128)
```

这 32 个位置来自已有 Buffer 的大小和访问映射。TileLang 保留全部逻辑位置，不通过检查某次使用是否紧邻定义来缩小分配。跨三个 Parallel 的 x 访问共用同一个 `x_frag_mem`。entry 数量按 4\.5 节的 Buffer 布局表确定。

例如 B\[4,130\] 沿第二维向量化、VL=128，每行需要两个 entry，共八个；backing 为 1024 个标量元素，不能用 ceil\(520/128\)=5。

累加器的 `[1]` 表示一个逻辑归约输出。Vectorize 读取 materializer 写入 `layout_map` 的累加器布局，确定输出 0 对应各 lane 独立保存的部分和，并建立一个 `vector<128xf32>` entry。这个映射由 fill、普通算术更新和 finalize 共用，不要求完整 `PartialFragment` 一直保留到这里。固定索引 `[0]` 转成固定 entry 索引 0，后续读取已经是向量，无需再次把一个标量广播成部分和。

若循环中的 `v` 是 Bind 定义的值：

```text
for q:
    v = compute(q)
    use(v)
```

转换后 `compute(q)` 的向量结果直接传给 `use(v)`，不生成 Fragment 存储。一个或多个 entry 的区别应从实际 Fragment Buffer 的分配和布局判断。例如，下面两段伪代码中的 Buffer 都明确分配为 `local.fragment`，采用连续一维映射，且 N=Q\*VL；`compute/use` 分别代表源计算和消费表达式：

```text
# 一个 VL 元素的 Fragment Buffer：一个 entry，声明 count=1、lanes=VL。
tmp_frag = T.sblock_alloc_buffer((VL,), "float32", scope="local.fragment")
for q in range(Q):
    for lane in T.parallel(VL):
        tmp_frag[lane] = compute(q, lane)
        use(tmp_frag[lane])

# N=Q*VL 个元素：Q 个 entry，声明 count=Q、lanes=VL。
saved_frag = T.sblock_alloc_buffer((N,), "float32", scope="local.fragment")
for i in T.parallel(N):
    saved_frag[i] = compute(i)
for i in T.parallel(N):
    use(saved_frag[i])
```

第一段每轮读写同一个 entry，第二段按 q 读写 Q 个 entry。两者都保留源 Buffer 的读写，数量由 Buffer shape 和布局决定。reducer 的部分和及结果 Buffer 使用 8\.3 节的布局，其中一个逻辑输出可能对应完整 VL 的保存空间；同样不能只套用 `元素总数 / VL`。

一个 Fragment 跨两个顺序 Parallel 使用时，声明一次，生产与消费共用 handle；以下是目标 TIR 示意，省略同步和注解，赋值表示 Bind：

```text
x_fragment_backing = T.sblock_alloc_buffer(
    (32 * 128,), "float32", scope="shared.dyn"
)

with T.sblock("SIMD_VF"):
    x_frag_mem = T.tl.vmi.declare_fragment_buffer(
        T.tl.access_ptr(x_fragment_backing[0], 32 * 128, 3),
        count=32,
        lanes=128,
    )
    full = T.tl.vmi.create_mask(128, size=128)

    for q in T.serial(32):
        x = T.tl.vmi.vload(
            T.tl.access_ptr(x_ub[q * 128], 128, 1), 0, size=128
        )
        T.tl.vmi.vstore(x, x_frag_mem, q * 128, full)

    for q in T.serial(32):
        x_lhs = T.tl.vmi.vload(x_frag_mem, q * 128, size=128)
        x_rhs = T.tl.vmi.vload(x_frag_mem, q * 128, size=128)
        sq = T.tl.vmi.vmul(x_lhs, x_rhs, full)
        T.tl.vmi.vstore(
            sq, T.tl.access_ptr(y_ub[q * 128], 128, 2), 0, full
        )

```

### 6\.5 AutoSchedule 容量预算、地址分配与 codegen

同一执行核心上的 VF 顺序发射和执行，不会同时执行两个 VF。Fragment 存储的数据完全在一次完整 VF 内产生和消费，相邻调用对 backing 的访问有序完成。因此每个 Fragment 存储只预留一份 backing，后续调用继续使用同一地址；TileLang 无需为此分析各个中间值的生命周期。普通输入、输出 UB 跨搬运和计算任务使用，仍可由 AutoSchedule 选择 double/multi\-buffer。

AutoSchedule 从任务的读写区域收集片上 Buffer，再按大小及版本数约束容量。Fragment backing 在规划时尚无计算访问，因此仅向 `SBlock.alloc_buffers` 添加分配还不够。需要从规划记录显式取得预留总量，按所属 kernel 的 UB 资源域扣除一次；普通 Buffer 枚举跳过这些预留对象，避免重复计数。

容量收集包含拥有分配的 root SBlock 和嵌套 VF，按 backing 去重扣除一次。只剩固定分配、零或负余额也要诊断；最终检查覆盖无实际合并、单分配、禁用复用及保留独立 UB 分配，包含对齐与 padding。

```text
普通 UB 可用容量 = 目标可用 UB 容量 - Fragment 存储固定预留总量
```

AutoSchedule 在剩余容量中选择普通 Buffer 的 double/multi\-buffer 版本。Fragment backing 自身固定一份，不参与版本选择或复制，也不与其他对象复用地址。若双缓冲不满足容量约束，调度器只能选择可行的版本数；连最低需求也超过容量时报告错误，不假定 PTOAS 以后会将 Fragment 值提升到寄存器。

`InsertSync` 仍处理原有 VF 与搬运任务之间的依赖。预留 Buffer 不生成 DMA、copy 或 fill 任务，也不增加虚假的读写来影响调度；VF 内最终保留的 Fragment UB 访存由 PTOAS 生成同步。关闭 AutoSchedule 时仍执行 `PlanPtoSlotStorage`，并在最终 UB 分配后检查容量。

Vectorize 之后，`AscendLowerTileOp` 继续处理剩余 tile 操作。`LowerAccessPtr` 将 backing 的地址表达转换为可重写的 `tvm_access_ptr`；`LowerOpaqueBlock` 将 SBlock 中的分配转成后续 pass 可处理的分配节点；`FlattenBuffer` 展平 Buffer。`RewriteAscendBufferVersionLayout` 继续处理普通多缓冲对象，不给 Fragment backing 增加版本槽位。

`MergeUBAllocations` 在这些 pass 和 `SplitHostDevice` 之后运行，按大小、对齐和可复用关系生成合并 UB Buffer 及字节偏移。Fragment backing 参加地址分配，但禁止与普通 UB 或其他 Fragment backing 重叠。自动调度路径使用已有 alias contract；手动调度路径先运行 `InferBufferAliases`，需显式排除 Fragment 存储的复用关系。最终容量检查计入合并后的对齐与 padding，覆盖自动和手动调度两种路径。

这些地址 pass 改写 `declare_fragment_buffer` 的 backing 操作数，VMI 访存继续引用同一个 handle。codegen 特判声明，输出 `pto.memref_declare`；读写输出 `pto.vmi.vload/vstore`。PTOAS 在既定 UB 上消除或保留访问，并为实际保留的访问生成必要 mem\_bar；外层 AscendThreadSync 不承担 VF 内同步。

以 x 为例，下面展示从预留到最终地址的变化，省略其他 Buffer 和计算；`tl.access_ptr` 的 extent 单位为 FP32 元素，`3` 表示读写权限：

```text
# Plan：保留原 x_frag 和计算，新增 backing 与规划。
x_fragment_backing = T.sblock_alloc_buffer((4096,), "float32", scope="shared.dyn")
# x_frag -> (x_fragment_backing, x_frag_mem, count=32, vector<128xf32>)

# Vectorize：在所属 VF 内定义声明，entry 放在访存 offset 中。
x_frag_mem = tl.vmi.declare_fragment_buffer(
    tl.access_ptr(x_fragment_backing[0], 4096, 3), 32, 128)
tl.vmi.vstore(x_def, x_frag_mem, q * 128, active)
x_use = tl.vmi.vload(x_frag_mem, q * 128, size=128)

# LowerAccessPtr
x_frag_mem = tl.vmi.declare_fragment_buffer(
    tvm_access_ptr(float32, x_fragment_backing.data, 0, 4096, 3), 32, 128)

# Merge：O_x 为分配器确定的字节偏移。
x_frag_mem = tl.vmi.declare_fragment_buffer(
    tvm_access_ptr(float32, buf_dyn_shmem.data, O_x / 4, 4096, 3), 32, 128)
tl.vmi.vstore(x_def, x_frag_mem, q * 128, active)
x_use = tl.vmi.vload(x_frag_mem, q * 128, size=128)
```

这是地址等价关系的示意，实际 TIR 可以带指针 cast；具体拼写按现有地址重写和 codegen 接口生成。`declare_fragment_buffer` 的 backing 参数必须使用这些 pass 能遍历和重写的地址表达，不能只在字符串或旁表里保存 `x_fragment_backing` 的名字。`tl.access_ptr` 内的 BufferLoad 用于描述地址，不会生成一次标量数据读取。

Fragment backing 的 `O_x` 是固定偏移，不随外层流水迭代的 `t % 2` 切换；entry 的动态寻址仍由 q 表达。普通 UB 的 ping/pong 版本槽位索引由前面的多缓冲 pass 生成，MergeUB 将其已有动态偏移与分配基址相加。Fragment 存储不需要这项版本偏移。

`MergeUBAllocations` 按 alias contract 分配地址。自动调度和关闭 AutoSchedule 的 `InferBufferAliases` 路径都要把 Fragment backing 登记为不与任何其他对象复用的分配，包括其他 Fragment 存储；不能让「尚无普通 BufferLoad/Store」成为复用依据。分配器保留其容量和对齐要求，检查最终 UB 总量。后端消除 Fragment 访存后，这段地址仍保留，不重新运行调度或压缩 TileLang 的 UB 分配。

codegen 从标准地址操作数提取最终 UB 字节地址，保留原 backing 的标量 dtype；不能把 tvm\_access\_ptr 当作整数地址传入，也不能把 dtype 误改成合并 arena 的字节类型。memref shape 是 count\*VL，vload/vstore offset 是元素单位。声明只绑定地址，不申请空间或初始化内容。

```text
with pto.vecscope():
    # 例如 UB arena 基址为 0，x backing 位于字节偏移 16384。
    x_frag_mem = pto.memref_declare(16384, (4096,), pto.f32)
    x = pto.vmi.vload(x_frag_mem, q * 128, size=128)
    pto.vmi.vstore(x, x_frag_mem, q * 128, active)
```

上述 offset 是元素单位；entry q 的字节地址为 16384 \+ q\*128\*4。声明已计入 backing 偏移，访存不得重复相加。基址解析保留普通 UB access\_ptr 路径，并接受由声明定义的 Fragment handle；后者直接打印为 memref 名。

静态分区可用 PTOAS 的 memref\_view：对 rank\-1 连续 memref，用静态 offset/size 命名不越界的子区间，类型和地址空间保持一致。view 不分配空间，也不提供 noalias。动态 entry 仍用固定 memref 加元素 offset，默认转换无需新增 TIR view 接口。

```text
second_half = pto.memref_view(x_frag_mem, 2048, 2048)
for q in range(16):
    value = pto.vmi.vload(second_half, q * 128, size=128)
# 等价于原 x_frag_mem 的元素 offset 2048 + q * 128。
```

## 7\. direct reduce 语义

direct reduce 指直接对一个已存在的向量/UB region 做规约的操作（`tl.tileop.reduce` 及 `T.reduce_max/min/sum` 等直接规约调用），由 Vectorize 的 Reduce 模块消费。它不经过 legacy reducer 的 `CanonicalizeLegacyReducer` 和 `ReducerPlanAndMaterialize`，不涉及 PartialFragment、Fragment 部分和或 seed；这些区别必须与第 8 章的 reducer v2 分开理解。

### 7\.1 输入契约与 lowering 语义

输入 TIR 形态示例：

```python
for i in T.parallel(256):
    tmp_ub[i] = x_ub[i] * x_ub[i]
T.reduce_max(tmp_ub, max_ub, dim=0)
```

首批 contract：调用位于含待转换 Parallel 的 `SIMD_VF` 内、Parallel 体外的区域级语句位置；source 为静态一维 UB Buffer，destination 为一元素 UB Buffer，`dim=0`、`batch=1`、`clear=True`、`nan_propagate=False`，支持 sum/max/min。`clear=True` 覆盖 destination；`clear=False` 的累积语义、`batch>1` 和 `nan_propagate=True` 留待扩展。前端调用为 `Evaluate(Call)`；绑定 handle 的 Reduce 表达式不属于准入形式。

Reduce 的 VL 读取当前 `SIMD_VF` 的 `tl.simdvf_lanes`，extent 读取自身输入 region，不从前序 Parallel 推断。source 每个 chunk 的满宽读取由调用方保证容量、对齐和可访问性；被归约 mask 排除的尾部数据不得影响结果。Reduce 模块依据输入 region 的 extent 和操作类型生成规约指令。对 extent 为 E、逻辑 lane 数 VL 的一维输入，目标 lowering 形态为：

```text
# 目标设计示意：ceil 分 chunk 的满宽读取 + 折叠 + 跨 lane 规约
nchunks = ceil(E / VL)
acc = vload(src_ptr(0), 0, size=VL)              # 第 0 个 chunk 直接作为初值
for chunk in 1 .. nchunks-1:                     # 按源顺序折叠
    loaded = vload(src_ptr(chunk * VL), 0, size=VL)
    chunk_mask = create_mask(chunk_active(chunk), size=VL)
    folded = vadd/vmax/vmin(acc, loaded, chunk_mask)
    acc = vsel(chunk_mask, folded, acc)          # masked 折叠会把无效 lane 清零，vsel 恢复先前累加值
reduced = vcadd/vcmax/vcmin(acc, mask0)          # 跨 lane 规约；浮点 sum 带 reassoc=True
```

其中 `chunk_active(c)` 是第 c 个 chunk 的有效 lane 数（`E - c*VL` 截断到 `[0, VL]`）。`mask0` 是第 0 个 chunk 的域 mask：`E >= VL` 时为全量 mask；`E < VL` 时为 `E` 个有效 lane 的部分域 mask，使补齐 lane 不参与跨 lane 规约。 lowering 直接把首个输入向量作为累加初值、没有初始化其无效 lane，因此 `E < VL` 时必须使用输入域 mask，不能套用已初始化 reducer 的全量 mask。

浮点 sum 的跨 lane 规约使用 `vcadd(reassoc=True)`，表示允许改变浮点加法结合顺序；严格要求原顺序的程序不走这条 lowering。max/min 使用 `vcmax/vcmin`，NaN 和有符号零行为列入准入核对。逻辑宽度超过单个物理寄存器时，物理拆分与最终合并由 PTOAS 完成。

### 7\.2 direct Reduce 的单元素写回

跨 lane 规约返回单元素 VMI 结果，直接写入 `dst[0]`。TileLang 发射一元素 `vstore`，PTOAS 将其 lowering 为 `vsts` 的 `1PT` 模式：

```text
reduced = vcadd/vcmax/vcmin(acc, reduce_mask)  # 单元素 VMI 结果
vstore(reduced, dst_ptr, 0)                  # 一元素写入
```

若 VMI ABI 要求显式 store mask，使用一元素全有效 mask；不为写回广播成 VL 元素，也不创建 VL 宽度的 lane0 mask。输入的 `reduce_mask` 继续限定参与归约的 lane。写回必须只影响 destination 的一个元素，相邻 UB 哨兵值不变；数据类型、对齐和实际 `1PT` lowering 按所用 PTOAS 版本验证。

这是 direct Reduce 自身的结果写回规则。多输出 region 的支持需按其输出域单独定义，不套用单元素写回。

### 7\.3 用户语句的写回边界

用户语句按照 4\.7 节的地址与 RHS 类型规则转换。例如 Parallel 外的 `idx_ub[k] = best[0]` 是普通标量读写。实际的向量 RHS 按 continuous 或 uniform 地址选择 NORM store 或 scatter，并使用源语句对应的执行 mask。

Fragment Buffer 读取先满足布局和访问规则；未支持的单元素提取在读取处报告，不因目标写回形式改变读取含义。reducer v2 的 finalize/copy 另按第 8 章定义，普通语句的写回不能直接套用 direct Reduce 的 `1PT` 规则。

### 7\.4 契约一览与验收用例

direct reduce 与第 8 章 reducer v2 的六维度对照：

|维度|direct reduce（本章）|reducer v2（第 8 章）|
|---|---|---|
|输入 TIR 形态|直接规约调用（`tl.tileop.reduce` / `T.reduce_max/min/sum`），作用于已存在的 UB region；无 init/update/finalize 结构|前端 `T.alloc_reducer` \+ `T.clear` \+ Parallel 内 `+=` \+ `T.finalize_reducer`，经规范化与展开后的 fill、普通算术更新、materialized finalize 和 copy|
|layout 来源|输入 region 的连续地址与 extent；不需要 Buffer 布局|LayoutInference 推导的 PartialFragment 及 materialize 后 `layout_map` 中的普通 Buffer 布局|
|是否经 `ReducerPlanAndMaterialize`|否|是（PTO SIMD 分支）|
|update mask / finalize mask|逐 chunk 输入域 mask 折叠，`vsel` 保留先前累加值；跨 lane 规约使用首 chunk 域 mask（`E<VL` 时部分域）|update 使用 `active = loop_mask AND source_condition`；finalize 使用完整参与组 mask，不复用最后一轮 tail mask|
|结果发布与 seed|单元素结果使用 `1PT` 写回；无 seed|跨 lane 规约后 AllReduce 广播，再与初始化处捕获的共同标量 seed 向量 combine（每个逻辑输出一次）；seed 不在 finalize 重读源值|
|验收用例|sum/max/min；`E<VL`、`E=VL`、`E>VL` 非整除；普通 store/load 顺序和 PTOAS `mem_bar` 分类|见 8\.7 节|

direct reduce 的验收用例必须覆盖：sum/max/min 三种操作；`E<VL`（部分域 mask）、`E=VL`（全量）、`E>VL` 且非整除（尾部 chunk 的 keep\-prior 折叠）；同一区域内规约结果经普通 BufferLoad 被后续计算消费；同地址 store/load 未保序时记录为 PTOAS 工具链问题。

## 8\. reducer v2、finalize、seed 与 inactive\-tail

reducer 转换确定每次更新的逻辑输出、部分和的 lane 分布和 finalize 的合并方式；第 6 章负责这些 Buffer 的声明与读写。归约规则不依赖普通 memref 访存最终是否被消除；两部分通过向量值和存储映射衔接。

本章讨论 `T.alloc_reducer` 及其在 Parallel 内的更新（reducer v2 路径）。`T.reduce_max/min` 等直接对一个 Buffer region 做规约的操作由第 7 章的 direct reduce 处理，不经过本章的 legacy reducer 规范化流程。

### 8\.1 前端、规范化与布局推导

RMSNorm 前端用普通计算语句描述归约：

```python
sum_sq = T.alloc_reducer((1,), "float32", op="sum", replication="all")
T.clear(sum_sq)
for i in T.Parallel(TILE):
    if i < d:
        sum_sq[0] += x_frag[i] * x_frag[i]
T.finalize_reducer(sum_sq)
var = sum_sq[0] / d + eps
```

前端 TIR 已将 `+=` 表达为读、加、写：`BufferStore(sum_sq, BufferLoad(sum_sq) + contribution)`；clear 对应 fill。此时没有显式的 `reducer_init` 或 `reducer_update`。

`CanonicalizeLegacyReducer` 根据 reducer 分配及运算标记识别这些语句，生成下面的形式。该例省略 region 参数和变量重命名（目标设计示意）：

```text
T.reducer_init(acc)
for i in T.parallel(TILE):
    if i < d:
        T.reducer_update(acc[0], x_frag[i] * x_frag[i])
T.finalize_reducer_v2(acc, result)
var = result[0] / d + eps
```

acc 使用 `local.reducer`；result 是新建的 `local.fragment` Buffer，即 2\.2 节 dump 中的 `sum_sq_result`。finalize 之后对原 reducer 的读取被重定向到 result。这个规范化 pass 提取的是归约语义，尚未决定线程或 lane 上的存储。

随后执行 `VerifyReducerEpoch` 和 `VerifyBufferInit`。SIMT 的 `AscendLayoutInference` 根据 Parallel 线程布局推导 reducer 的 `PartialFragment`，说明哪些参与者保存不同部分和、哪些只是相同副本。RMSNorm 的布局结果（历史捕获，SIMT 路径）中，sum\_sq 为：

```text
shape = [1]
每线程局部位置 = 0
参与线程 = 128
replicate = 128 = combine 128 × copy 1
```

`ReducerPlanAndMaterialize` 消费这份计划，再生成初始化、普通更新、materialized finalize 和 copy。它复用 `ReducerV2Identity` 生成单位元，复用 `ReducerV2Combine` 生成 combine 表达式；同时处理部分和存储、执行次数、可选 seed 和结果分布。这个 pass 含有布局与通信计划，不能当成无条件的语法替换。

SIMT materializer 的显式计划路径已将 `PartialFragment` 分成两部分使用：`AsPostCollective()` 生成普通 Buffer 存储布局，`CombineSteps()` 提取合并步骤并写入 finalize。PTO 分支沿用这项分工，再将坐标和合并步骤适配为逻辑 lane。

|阶段|TIR 形态（目标设计示意/历史捕获）|
|---|---|
|前端|fill、普通累加赋值、原地 finalize|
|`CanonicalizeLegacyReducer` 之后|init、update、finalize\_v2|
|`ReducerPlanAndMaterialize` 之后|fill、普通乘加、finalize、copy|
|PTO Vectorize 入口|展开后的计算体 \+ lane 布局与 Buffer 布局|

### 8\.2 Vectorize 的接入位置与输入契约

目标 pipeline 中，reducer 前置顺序固定为：

```text
前端 TIR
→ CanonicalizeLegacyReducer
→ VerifyReducerEpoch / VerifyBufferInit
→ AscendLayoutInference（含 PTO SIMD 分支）
→ ReducerPlanAndMaterialize（含 PTO SIMD 分支）
→ 其他布局处理、调度、同步、多缓冲
→ NormalizeBufferVersion
→ PTO 转换入口：VerifyParallelToPTO → LegalizeParallelToPTO → VectorizeParallelToPTO
```

materialization 后的对照计算体（历史捕获，SIMT 路径输出，仅省略 Parallel 的布局注解；不是 SIMD 输入）：

```python
T.fill(T.region(sum_sq_1[0], 2, 1), T.float32(0.0))
for i in T.parallel(4096):
    T.attr(sum_sq_1.data, "tl.parallel_multiplicity", 1)
    sum_sq_1[0] = sum_sq_1[0] + x_frag[i] * x_frag[i]
T.finalize_reducer(T.region(sum_sq_1[0], 3, 1), 0)
T.copy(T.region(sum_sq_1[0], 1, 1), T.region(sum_sq_result[0], 2, 1))
```

目标设计要求适配后的前置路径向 PTO 转换提供相同层次的输入：普通算术更新、明确的归约结束操作和结果 copy，外加仍可读取的 lane/部分和语义。Vectorize 处理普通更新和 finalize；Legalize 不重复调用 reducer 前置 pass。

### 8\.3 PTO SIMD 的 LayoutInference：确定部分和与 lane 的对应关系

PTO SIMD 已经有 `q * VL + lane` 的执行布局，还需推导普通 Fragment Buffer、reducer 部分和与结果 Buffer 的布局。前置阶段将 Buffer 的 entry/lane 映射写入 `layout_map`，materializer 更新后供 Plan、Verify 和 Vectorize 消费；finalize 参数确定哪些部分和需要合并。SIMT 转成 SIMD 后，线程私有部分和改用向量的各 lane 表示，计算含义保持不变。

目标设计只支持每个 reducer 在初始化到 finalize 之间有一个静态 update 语句。循环多次执行这条语句仍算一个 update，贡献表达式中的乘法、加法等也不分别计数。两个 update 即使布局相同也不支持；检查在前置阶段、`reducer_update` 尚未展开时完成，具体规则见 8\.6 节。

输入是 canonicalization 后的 init/update/finalize\_v2、Parallel 的逻辑范围和 VF 的 `tl.simdvf_lanes`。在 PTO SIMD 区域收集阶段，按 reducer 统计静态 update 数量，确认只有一处后，再分析该语句的目标索引。`ParallelOpNode::InferLayout` / `InferPtoSimdLayout` 建立 lane 布局时还需为 reducer 推导部分和布局，供 materializer 确定归约范围和结果布局。该路径使用 `[0, VL)` 的逻辑 lane 范围，不要求真实的 `threadIdx.x`。

**普通 Fragment Buffer 的映射** 使用 4\.5 节的 entry/lane 公式。用户 UB 地址仍按原 Buffer 的 shape、stride 和 offset 计算，Fragment backing 的容量由 Plan 按 entry 数另行预留。`reducer_update` 的目标按部分和关系分析，不能当成普通 UB 的同址写入。

以 `j = q * VL + lane` 为例，将 update 的目标索引代入即可看出每条 lane 为哪个输出贡献。这里的「逻辑输出」是源程序的一个结果元素，例如 `acc[0]` 是 RMSNorm 的平方和，`acc[j]` 是第 j 列的结果。VL=4、固定当前 i 和 q 时：

|更新语句|lane 0、1、2、3 各自保存的部分和对应哪个逻辑输出|是否跨 lane 合并|
|---|---|---|
|`acc[0] += x[j] * x[j]`|四条 lane 都对应 `acc[0]`，但各自保存不同输入的部分和|是，四份部分和合为一个结果|
|`acc[i] += x[i,j]`|四条 lane 都对应 `acc[i]`，但各自保存不同输入的部分和|是，对输出 i 合并四份部分和|
|`acc[j] += x[i,j]`|分别对应 `acc[4q]`、`acc[4q+1]`、`acc[4q+2]`、`acc[4q+3]`|否，四条 lane 已经对应四个不同输出|

以前两行为例，finalize 前可以记作 `[a0, a1, a2, a3]`：四个值对应同一个 `acc[0]` 或 `acc[i]`，却不是四条 lane 争写一个 UB 标量。sum 的 finalize 计算 `s = a0 + a1 + a2 + a3`，AllReduce 再让参与 lane 各自取得 s，即 `[s, s, s, s]`。SIMT 下的线程私有部分和，在 SIMD 下对应向量的不同 lane。

`PartialFragment` 可以描述上述关系。其定义以一个结果元素为单位，将保存它的部分和或等价副本的位置分为需要合并的贡献和重复计算相同结果的组。在 SIMT 布局中，位置对应线程；在 PTO SIMD 布局中，位置对应逻辑 lane。这里计数的是固定一个输出后的保存位置，不是整个 VF 的线程或 lane 总数，也不是 update 的动态执行次数。

|字段或推导量|语义|
|---|---|
|`replicate = R`|为一个结果元素保存部分和或等价副本的位置总数|
|`combine = C`|每组中需要合并的不同贡献份数；即使数值相同，也都要计入|
|`copy = R / C`|等价计算组的数量；每组独立计算同一个逻辑结果，组与组不再相加|

`copy` 由 `replicate/combine` 推导，没有独立的 `copy_size` 字段。参与坐标 `_rep % C` 枚举组内贡献，`_rep // C` 枚举等价组。例如 `replicate=4、combine=2、copy=2`：

```text
第 0 组：a、b → a+b
第 1 组：a、b → a+b
```

四个位置一起求和会得到 `2*(a+b)`，重复计入同一组贡献。组内两个位置则都必须参加归约。实际位置还由 `forward_thread`、`forward_index` 确定；在 PTO 路径中，它们需要对应逻辑 lane 及该 lane 保存的位置。

在本文的规范布局下，复用 `PartialFragment` 时预期得到：

|场景，VL=128|每个逻辑输出的 replicate|combine|copy|
|---|---|---|---|
|RMSNorm 的输出 0|128|128|1|
|`acc[i] += x[i,j]`，j 向量化|128|128|1|
|`acc[j] += x[i,j]`，j 向量化|1|1|1|

对应的部分和存储坐标如下。`rep` 枚举同一个输出的参与 lane；结果 Buffer 沿用这些坐标，跨 lane finalize 后其中的值才变成完整结果。

|场景|Buffer 逻辑 shape|`forward_index`（entry）|`forward_thread`（lane）|entry 数量|
|---|---|---|---|---|
|RMSNorm 的 `acc[0]`|`[1]`|`0`|`rep`|`1`|
|`acc[i] += x[i,j]`，j 向量化|`[M]`|`i`|`rep`|`M`|
|`acc[j] += x[i,j]`，j 向量化|`[N]`|`j//VL`|`j%VL`|`ceil(N/VL)`|

第三行使用了 128 条 lane，但它们对应不同输出；每个输出只由一条 lane 累积。Parallel 布局的 `replicate_size=1` 表示每个输入迭代点只执行一次，`vectorize_size=1` 表示每条 lane 每轮处理一个元素。RMSNorm reducer 的 `replicate=128` 则表示输出 0 有 128 份待合并的部分和。不能据此启用循环副本去重、丢掉其中 127 份贡献。

以 sum 展开两类计算。设输入为 M×N，j 为向量化维，lane 编号为 ℓ。先考虑整除情形，Q=N/VL：

```text
j = q*VL + ℓ，0 <= q < Q，0 <= ℓ < VL
```

沿 j 规约，即 `acc[i] += x[i,j]`。每条 lane 沿 q 累积，得到输出 i 的一个长度为 VL 的部分和向量：

```text
p[i][ℓ] = sum_{q=0}^{Q-1} x[i][q*VL + ℓ]
```

finalize 再合并这个向量中的所有 lane：

```text
y[i] = sum_{ℓ=0}^{VL-1} p[i][ℓ]
```

对应的向量计算如下。`x_vector(i, q)` 表示第 i 行第 q 段的 VL 个元素，`partial[i]` 和 `result[i]` 在这段伪代码中都是完整向量，后者各 lane 均为逻辑结果 `y[i]`。示例省略 mask 参数和第 6 章规定的 Fragment 读写；初始化、更新和 finalize 保持源程序的顺序：

```text
for i in range(M):
    partial[i] = zero_vector(VL)
for i in range(M):
    for q in range(Q):
        partial[i] = vadd(partial[i], x_vector(i, q))
for i in range(M):
    reduced = vcadd(partial[i])
    result[i] = broadcast(reduced, VL)
```

广播对应 AllReduce 的结果分布，让参与 lane 都取得输出 i 的完整值。RMSNorm 只有一个逻辑输出，将上式中的 x 换成 x²，`Q=32、VL=128`。若源程序分配了 M 个输出，就保留 M 个逻辑状态，不因逐行执行而自动复用成一个位置。

沿 i 规约，即 `acc[j] += x[i,j]`。对每个 q，各 lane 沿 i 累积各自的输出：

```text
p[q][ℓ] = sum_{i=0}^{M-1} x[i][q*VL + ℓ]
```

此时每个 lane 已经得到一个完整的列结果：

```text
y[q*VL + ℓ] = p[q][ℓ]
```

```text
for q in range(Q):
    partial[q] = zero_vector(VL)
for i in range(M):
    for q in range(Q):
        partial[q] = vadd(partial[q], x_vector(i, q))
for q in range(Q):
    result[q] = partial[q]
```

这里的 `partial[q]` 和 `result[q]` 各保存一个完整向量，共有 Q 个不同的输出向量，不能用 `vcadd` 把它们内部的各列结果相加。N=4096、VL=128 时，需要保留 32 个输出向量。最后的赋值对应结果 Buffer 的 copy；有额外 seed 时按 8\.4 节处理。

非整除时取 `Q=ceil(N/VL)`，只有 `q*VL+ℓ < N` 的元素产生贡献。无效 lane 保留旧部分和，条件与 tail 处理见 8\.5 节，不扩大用户 Buffer。max/min 使用相同的参与关系，换成各自的 combine 和单位元。选择另一条向量化维时，两类处理随之交换。

**布局如何跨 pass 保存。** 复用现有注解键，但区分循环布局和 Buffer 布局：

|注解键|所在节点与内容|消费者|
|---|---|---|
|`kPtoParallelLoopLayout` / `pto_parallel_loop_layout`|Parallel 根 For；一个 lane 布局对象，描述迭代点到 lane/q 的映射|materializer 的 update 收集器、Verify、Vectorize|
|`kLayoutMap` / `layout_map`|SBlock；推导后为 `Map<Buffer, Layout>`，普通 Fragment Buffer 和结果对应普通布局对象，`local.reducer` 对应 `PartialFragment`|materializer；替换后的布局供 PlanPtoSlotStorage、Verify、Vectorize 使用|
|`kReducerInfoV2` / `reducer_info_v2`|现有 reducer 操作类型等语义信息|materializer 消费并清理|

`kPtoParallelLoopLayout` 的值仍然是一个循环布局，不能把多个 Buffer 的布局塞进同一个值。Buffer 布局使用已有 `kLayoutMap` 即可，无需新增 PTO 专用 key。用户输入的 `T.annotate_layout` 可以以 data Var 为键；经过 `AscendLayoutInference` 后按现有约定解析成 Buffer 为键，不能混用两种阶段的格式。

读写路径要求：

1. 推导 `x_frag` 等普通 Fragment Buffer、reducer 的 `PartialFragment`，以及 finalize 结果的布局。结果优先采用部分和归约后的坐标分布；下游已有约束时检查兼容性，不静默改成另一种布局。

2. 在持有这些分配的 SBlock 写回 `layout_map`，并让待转换的 PTO SIMD\_VF 能读取相同条目。SIMD\_VF 分支不能只写循环注解而跳过 Buffer 布局写回；其他区域的现有路径保持不变。

3. materializer 替换 `local.reducer` 时，沿用现有的存储布局与归约计划分工：在所有携带旧条目的 block 上删除旧 Buffer→`PartialFragment`，加入新累加器 Buffer→普通布局，保留 entry/lane 坐标，并将合并步骤写入 finalize。结果 Buffer 条目继续维护；有 seed 时还要给新建的 seed Fragment Buffer 加入布局。相同 Buffer 的重复条目必须一致，不能让内层旧注解遮蔽新映射。

4. Verify 和 Vectorize 按区域读取 `layout_map`，结合当前 Parallel 的逆映射检查每次访问。Buffer 被版本化、复制或替换时，分配、读写、布局 key 和 finalize 引用一起改写；不同 Buffer view 不能只凭名称或同一个 data Var 就视为等价。

5. `PlanPtoSlotStorage` 按替换后的 Buffer 布局预留 UB，Vectorize 消费这份记录建立 Buffer→Fragment 存储。转换完成后清理已消耗的 Fragment Buffer 分配、布局条目及循环注解，保留 backing 的分配和地址引用，避免后续 `AscendLowerTileOp` 再做 SIMT 存储改写。

materializer 之后的普通布局提供存储坐标，finalize 表达合并操作。`acc[0]` 在 finalize 前仍可保存各 lane 不同的部分和，单凭 shape 或 `replicate` 不能断言值相等。PTO 的公共 lane\-use 分析需按 Buffer scope 分派：UB 的 uniform 地址可以产生标量 load；Fragment Buffer 的固定逻辑索引则按其布局读取，可能对应一个完整的部分和向量。本设计不新增整个 VF 的 uniform 值传播分析；finalize 的广播语义、表达式的向量类型和写回规则分别在各自的转换点处理。

### 8\.4 `ReducerPlanAndMaterialize` 的 PTO 分支与 seed 捕获

SIMT 路径保持现有实现。`target=pto` 且区域属于 SIMD\_VF 时进入独立分支。`ReducerEpochCollector` 在该区域读取 `pto_parallel_loop_layout`，从 `tl.simdvf_lanes` 取得 lane 范围，并消费 `layout_map` 中的部分和布局。PTO 分支复用 `ReducerV2Identity` 和 `ReducerV2Combine`；初始化到 finalize 的语义合法性仍由前置 `VerifyReducerEpoch` 检查。

PTO 分支在判断 SIMT wide/narrow plan 之前分流，不要求 `threadIdx.x`，不生成 packed partial、`tl.parallel_multiplicity` 或 `AscendAllReduce`。即使 `replicate=combine=VL`，也仍按一个向量内的 lane partial 处理。无法生成受支持的 PTO 计划时报告错误，不回退到 SIMT wide plan。

单位元用于初始化各份部分和：sum 为 0，浮点 max 为负无穷，浮点 min 为正无穷；整数 max/min 使用相应 dtype 的边界值。RMSNorm 只需要把部分和初始化为 0。

seed 是可选的用户逻辑初值。例如 v2 接口 `T.reducer_init(acc, 7.0)` 在 sum 中要求最终结果为 `7 + sum(x)`。各 lane 仍从单位元 0 开始，归约后再加入一次 7。若每条 lane 都从 7 开始，全 lane 归约会得到 `VL*7 + sum(x)`，重复计入初值。seed 在源初始化位置求值并保存，防止 finalize 时重新读取已经变化的输入；最终通过已有 finalize 的 `seed` 注解使用这个已保存的值。无跨 lane 合并时，每个逻辑输出也只合并一次 seed。

目标设计只接受所有逻辑输出共用的一个标量 seed。例如四条 lane 对应同一个输出时，seed=7 在跨 lane 规约后加入一次；四条 lane 对应四个不同输出时，各输出分别加入一次 7。编译器可以将这个共同初值广播成向量，但不接受用户提供逐输出的 seed Buffer、数组或向量。seed 可以随外层循环变化，每次初始化都重新捕获。

这与前端接口相符：`T.reducer_init` 接收一个 `PrimExpr` 或数值，reducer 校验要求其 dtype 与 reducer 元素类型一致。对 shape 为 `[4]`、dtype 为 `float32` 的 reducer，前端与 `VerifyReducerEpoch` 的输入契约如下（历史捕获，仅确认前端输入契约，不表示 PTO 向量化已验证）：

|seed 写法|前端与校验结果|
|---|---|
|`T.float32(7.0)`|通过，表示各输出共用的标量初值|
|`seed_ub[0]`|通过，表达式类型为 `float32`；PTO 路径还需检查它是否是受支持的共同初值来源|
|`seed_ub` 或 Python 数值列表|前端拒绝，不是所需的标量表达式|
|`seed_ub[0:4]` 或 `tirx.Broadcast(T.float32(7.0), 4)`|校验拒绝，`float32x4` 与 reducer 元素类型 `float32` 不匹配|

PTO 分支依据已有的定义、地址和布局检查接受常量或其他可确认的共同标量初值；无法确认时直接报不支持，不为此增加通用值传播分析。源表达式的 scalar dtype 本身不够：线程私有 Fragment Buffer 的一个元素也可能在各 lane 中不同，不能直接作为共同 seed。

SIMT materializer 在初始化处用 `local` 标量 Buffer `[1]` 捕获 seed。PTO 分支改用逻辑 shape 为 `[1]` 的 Fragment Buffer，在 `layout_map` 中将同一个 seed 映射到当前 VF 的所有 lane，而不沿用部分和的 `replicate/combine` 关系。初始化处的两个写入不能混淆：先对源 seed 求值并保存，再用归约单位元填充部分和。finalize 的 `seed` 注解必须引用新建的 seed Buffer，而不是原 `local` Buffer 或原始 `seed_ub[0]`。例如：

```text
seed_frag[0] = seed_ub[0]               # 普通 BufferStore：在 init 处捕获用户初值
T.fill(T.region(acc[0], 2, 1), 0.0)      # 各 lane 的部分和从单位元开始
... update acc ...
T.finalize_reducer(T.region(acc[0], 3, 1), ..., seed=seed_frag[0])
```

最后一行的 `seed=...` 表示 Call 注解，不是用户源码的关键字参数。seed 捕获保留为普通 `BufferStore`，使 `seed_ub[0]` 的读取和 `seed_frag[0]` 的写入继续参与依赖分析；`T.fill` 只用于部分和的单位元初始化。Vectorize 将标量 seed 写入转换为广播及完整 Fragment 存储写入，finalize 读回该 Fragment 存储；即使 `seed_ub[0]` 在两者之间被覆盖，也不能重读它。现有 `Fill` 只显式登记目标写区域，因此不应依赖 `T.fill(seed_frag, seed_ub[0])` 捕获带 BufferLoad 的初值。具体 Fragment 读写见 12\.2 节。

PTO 分支将 `reducer_init(acc)` 展开为单位元 fill；有 seed 时先在初始化位置保存其值。唯一的 `reducer_update(acc[index], value)` 展开为 `BufferStore(acc, combine(BufferLoad(acc), value))`，保留源条件和语句顺序。对于 `finalize_reducer_v2(acc, result)`，materializer 根据部分和布局生成以下三种输出：

|部分和是否需要跨 lane 合并|seed|materializer 的输出|
|---|---|---|
|需要|有或无|带 `plan=1` 和非空 `plan_steps` 的 `T.finalize_reducer`，随后 `T.copy`|
|不需要|有|带 `plan=1`、空 `plan_steps` 和 seed 的 `T.finalize_reducer`，随后 `T.copy`|
|不需要|无|不生成 `T.finalize_reducer`，只生成 `T.copy`|

每条实际生成的 PTO SIMD finalize 都有显式计划。非空 `plan_steps` 只接受单个 `(VL,1)`，表示合并完整逻辑向量的 lane；空步骤仅在有 seed 时生成 finalize。其他子组宽度、scale 或多步骤计划均报不支持。`plan=1` 只表示使用 `plan_steps`；没有显式计划的 finalize 依赖 SIMT lowering 从线程布局推导范围，不能作为 PTO SIMD 的输出。

materialization 保留逻辑 Buffer 的 shape，并将累加状态改写为 `local.fragment` Buffer 的普通读写，更新相应布局引用。输出仍是标量并行 TIR，不生成 q 循环、VMI 或 Fragment 存储。后续 Vectorize 按普通表达式转换展开后的算术，其 Reduce 模块根据 finalize 的操作与归约范围生成指令；两个模块共用同一份 Buffer 到向量存储的映射。

RMSNorm 的目标输出如下（目标设计示意）。示意中 finalize 复用现有参数编码：0 是 sum 的枚举值，`(128, 1)` 表示合并 128 条连续逻辑 lane，Call 的注解 `plan=1` 表示使用显式归约步骤：

```text
T.fill(T.region(acc[0], 2, 1), T.float32(0.0))
for i in T.parallel(4096):
    acc[0] = acc[0] + x_frag[i] * x_frag[i]
T.finalize_reducer(T.region(acc[0], 3, 1), 0, 128, 1)  # Call.annotations: {"plan": 1}
T.copy(T.region(acc[0], 1, 1), T.region(sum_sq_result[0], 2, 1))
```

归约操作和范围已经体现在 finalize 中，初始化和更新也已经展开；后续无需再附加一个完整的 reducer 计划对象。materializer 消费 `reducer_info_v2` 和旧 `PartialFragment`，在 `layout_map` 中发布替换后的累加器和结果布局。Vectorize 使用这些坐标建立 Fragment 存储映射，信息来源见 8\.6 节。

### 8\.5 finalize、广播、结果 copy 与 inactive\-tail

Vectorize 重新分析 Legalize 的输出，使用同一 VF 上的 q/lane 映射、Buffer/Fragment 存储映射、值类型和当前 mask。展开后的 `x * x` 和加法分别生成 `vmul`、`vadd`；cast 走已有类型转换路径。算术节点不需要额外的 reducer update 标记。累加器的读写按第 6 章的 Fragment 存储规则完成，PTOAS 决定寄存器驻留或保留对已绑定 UB backing 的访问。

Storage 模块根据 `layout_map` 将 `T.fill` 的单位元广播为完整向量并写入累加器 Fragment 存储。RMSNorm 的逻辑 shape `[1]` 对应一个 `vector<VLxf32>` entry，fill 必须初始化全部 lane，无需先变成 `T.Parallel(1)`。Reduce 模块新增 materialized finalize 的处理入口，从共同存储映射取得部分和向量；它只处理 materializer 实际留下的两种 finalize：

|Vectorize 收到的 `T.finalize_reducer`|Reduce 模块的处理|
|---|---|
|`plan=1`，`plan_steps=[(VL,1)]`|用完整参与组 mask 生成 `vcadd/vcmax/vcmin`；按 AllReduce 语义广播结果，再与已捕获并广播的可选 seed 做向量 combine，写回累加器|
|`plan=1`，`plan_steps` 为空且有 seed|不生成跨 lane 规约；将共同 seed 广播后与各 lane 的输出做向量 combine，每个逻辑输出只合并一次，写回累加器|

第三种 materializer 输出不含 finalize，Vectorize 的 Reduce 模块无需处理；Storage 模块正常转换留下的 `T.copy`。缺少显式计划的 finalize 不属于 PTO SIMD 的合法输入，Verify 应在 Vectorize 前诊断，Reduce 模块不从 SIMT 线程布局猜测归约范围。

VMI reduce 的 `group` 表示结果分组数量，全 lane 合并使用 `group=1` 或等价默认形式。`plan_steps` 中的 VL 是参与 lane 数，不能直接填入 `group`。逻辑向量跨多少个物理寄存器、各 part 如何合并由 PTOAS 决定；TileLang 验证完整逻辑向量的结果，不生成物理拆分计划。

**finalize 后的向量计算。** RMSNorm 的累加器 shape `[1]` 表示一个逻辑输出，其向量表示包含 128 份部分和。全 lane 规约之后先广播结果；有 seed 时，再与广播后的 seed 做向量 combine。这样每条 lane 都取得完整结果，也不需要在广播前做标量浮点加法。column reduction 没有跨 lane 规约，各 lane 的输出直接与同一个广播 seed 做 combine；有效结果域以 Buffer 的逻辑范围为准，尾部无效位置保留原值。

`var`、`rstd_val` 虽在源 TIR 中写成标量 Bind，读取广播结果后已经是向量表达式。Expression 模块沿用向量与标量运算的类型提升规则，将除数、epsilon 等标量广播，生成向量除法、加法和平方根倒数。`T.rsqrt(var)` 用 `vsqrt` 再以广播的 1 做 `vdiv`，精度按源语义验证。`pto.vmi.vdiv` 支持 FP32；VMI 指令没有独立的精度参数，仍需用端到端数值测试核对结果。SIMD\_VF 内没有标量浮点 ALU，不能因源 dtype 为 `float32` 就退回标量浮点运算。这两个 Bind 对应向量 SSA 值，不分配 Fragment 存储；这里不新增跨语句的 uniform 传播。

RMSNorm 的 `z_rstd_ub[t % 2,0] = rstd_val` 按 4\.7、7\.3 节依据实际 RHS 类型、地址和执行 mask 写回；向量 RHS 写 uniform 地址依赖 scatter。`rstd_val` 对应的向量 SSA 值继续供后续输出循环使用；这次 UB 单点写回不改变它的向量类型，也不要求把它写入 Fragment 存储。

**finalize 与结果 copy 的循环结构。**`T.finalize_reducer` 在源 TIR 中位于整个更新 Parallel 之后，作用于完整 reducer region。Vectorize 按这个语句位置生成独立的 entry 遍历，不为了省掉一次循环而把 finalize 提前移进更新循环。下面只展示更新与发布的循环层级（目标设计示意）；`init_all_entries` 和 `combine_saved_seed_if_any` 是伪代码名称，不是新增 intrinsic。初始化在更新之前填充全部部分和，向量计算中的 Fragment 读写、条件和 tail 按第 6 章和第 9 章生成：

```text
# acc[i] += x[i,j]：每个 i 一条部分和向量。
init_all_entries(acc_frag_mem, identity)
for i in range(M):
    for q in range(Q):
        active = update_mask(i, q)
        old = vload(acc_frag_mem, i * VL, size=VL)
        candidate = vadd(old, x_vector(i, q), active)
        vstore(candidate, acc_frag_mem, i * VL, active)

# finalize 在整个更新 Parallel 之后。
for i in range(M):
    partial = vload(acc_frag_mem, i * VL, size=VL)
    reduced = vcadd(partial, full, reassoc=True)
    result = vbrc(reduced, size=VL, group=1)
    result = combine_saved_seed_if_any(result, i)
    vstore(result, acc_frag_mem, i * VL, full)
# 源结果 copy，保留语句位置。
for i in range(M):
    copied = vload(acc_frag_mem, i * VL, size=VL)
    vstore(copied, result_frag_mem, i * VL, full)

# acc[j] += x[i,j]：每条 lane 是不同输出，沿 i 累积。
init_all_entries(acc_frag_mem, identity)
for i in range(M):
    for q in range(Q):
        active = update_mask(i, q)
        old = vload(acc_frag_mem, q * VL, size=VL)
        candidate = vadd(old, x_vector(i, q), active)
        vstore(candidate, acc_frag_mem, q * VL, active)
# 仅有 seed 时生成以下 combine/writeback，空通信、不跨 lane 规约。
if has_seed:  # 编译期条件，表示是否存在 materialized finalize。
    for q in range(Q):
        result = vload(acc_frag_mem, q * VL, size=VL)
        result = combine_saved_seed_if_any(result, q, output_mask(q))
        vstore(result, acc_frag_mem, q * VL, output_mask(q))
for q in range(Q):
    copied = vload(acc_frag_mem, q * VL, size=VL)
    vstore(copied, result_frag_mem, q * VL, output_mask(q))
```

`combine_saved_seed_if_any` 仅在有 seed 时生成向量 combine。`output_mask(q)` 来自结果 Buffer 的有效域，决定 column reduction 的写回位置；inactive lane 保留目标旧值。结果 backing 有完整 entry 容量，padding 不作为新增输出。finalize、copy 保持源语句顺序；循环融合另行证明。

若源程序是外层 Serial i 包住更新 Parallel 与 finalize，则在该 i 内按源顺序生成，跨 lane 规约仍在全部 q 更新完成后执行。

`plan=1` 表示使用显式通信计划，不表示「一步规约」或「空通信」。没有通信也没有 seed 时，materializer 会省去 finalize，留下结果 copy。普通 Buffer 的存储映射必须独立可用，不能依赖已省去的调用来识别累加器。缺少显式计划的 finalize 在进入 Vectorize 前诊断。

**column reduction 的结果 copy 保持列索引不变。** 例如 VL=4 时：

```text
acc：    [第 0 列的和，第 1 列的和，第 2 列的和，第 3 列的和]
result： [第 0 列的和，第 1 列的和，第 2 列的和，第 3 列的和]
```

对这个向量执行 `vcadd` 会把不同列的结果相加，不符合源语义，也不需要把某一列的结果广播到其他列。有共同标量 seed 时，将它广播后分别与各列结果 combine；目标设计不接受每列不同的 seed。

**inactive\-tail。** update 的条件和 tail 由现有控制流生成逐 lane mask；finalize 合并参与组的完整部分和。运行时条件全假时，部分和保持单位元，最终结果为单位元或可选 seed。column reduction 按结果 Buffer 的有效域合并 seed，不将补齐的 lane 当成新增输出，也不借用上一轮 update 的 tail mask。

条件和尾部 mask 在 ControlFlow 模块处理。对每轮向量操作：

```text
i_vec = q * VL + lane_ids
active = loop_mask AND source_condition(i_vec)
old = vload(acc_frag_mem, entry_index * VL, size=VL)
candidate = combine(old, contribution, active)
vstore(candidate, acc_frag_mem, entry_index * VL, active)
```

更新的 candidate 使用 `active` 计算，并以同一个 mask 写入累加器 backing。masked vstore 不写 inactive lane，保留此前部分和；无需额外 vsel 合成完整写回值。表达式本身的 Select 仍按原规则转换。条件化的源更新也必须包含 Parallel tail 限制。

Reduce 模块处理 reducer finalize 时，重新取得完整部分和参与组的 mask，不读取上一轮 update 留下的 tail mask。RMSNorm 对应 full VL：所有 lane 已在 fill 时初始化，没有贡献的 lane 保持单位元，有过贡献的 lane 保留历史部分和。以 `N=129、VL=128` 且输入全为 1 为例，最终向量应为 `[2, 1, ..., 1]`；若只规约最后一轮的有效 lane，结果是 2，正确结果为 129。

**数值语义。** 浮点 sum 仅对 materialized reducer finalize 使用 `vcadd(reassoc=True)`；源操作是 `T.alloc_reducer(op="sum")` 下的并行归约，而非要求保持原顺序的普通加法链。浮点 `vcadd` 接口必须携带显式 `reassoc=True` 注解；不能把它当作另一条可用的精确路径。RMSNorm 应按并行归约的数值契约验证误差。max/min 的 NaN、有符号零和 dtype 规则也列入准入检查。`finalize` 位于尚不支持的分支、提前退出或 divergent loop 内时，不能仅凭调用存在就生成全 VL 指令，按不支持的控制流诊断。

### 8\.6 跨 pass 接口与结构准入

**Vectorize 所需信息及其来源。**`ReducerPlanAndMaterialize` 已将初始化和更新展开，Vectorize 可以直接转换普通算术。finalize 本身也已经标明待处理的 Buffer 和归约操作。目标设计优先复用这些表达，仅为已有 IR 无法表达的信息补充注解：

|Vectorize 需要的信息|已有来源|消费方式|
|---|---|---|
|初始化值|`T.fill` 的值表达式|广播单位元并写入累加器向量|
|单次更新的计算|普通 `BufferLoad/BufferStore` 和算术表达式|按普通表达式向量化，不再添加 reducer update 身份标记|
|归约操作和待处理 Buffer|`finalize_reducer` 的 region 与操作类型|取得部分和向量并选择 sum/max/min 指令|
|归约参与范围|实际存在的 finalize 中的 `explicit_plan` 和 `plan_steps`，结合 VF lane 范围|步骤非空时确定跨 lane 归约；步骤为空时只处理 seed|
|可选逻辑初值|初始化位置的 seed 捕获语句及 finalize 的 `seed` 注解|读取已保存的共同初值，以向量形式对每个输出合并一次；不在 finalize 重读源 seed|
|Fragment Buffer 元素对应的 entry/lane|materializer 更新后的 `layout_map`，以及当前 Parallel 的 `pto_parallel_loop_layout`|检查访问与已选布局兼容，为 fill、update、finalize 和 copy 建立一致的 Fragment 存储映射|
|Fragment Buffer 的 UB 预留|`PlanPtoSlotStorage` 的 backing、count、向量类型、容量与对齐记录|生成 declare\_fragment\_buffer 的地址关联，验证与当前 Buffer/布局一致，不重新扩大分配|
|结果分布和读取关系|AllReduce 的约定、结果 Buffer 映射及 `T.copy`|需要合并时规约并广播，再按源顺序复制结果|
|条件、逻辑范围、dtype 和 VL|分支或 Select、循环 extent、Buffer 类型及 `tl.simdvf_lanes`|生成值类型、tail 和当前执行 mask|

这里的归约范围与每轮 update 的 mask 分开：前者说明哪些 lane 的部分和要合并，后者由循环和条件直接生成。identity 已在 fill 中，贡献已在算术表达式中，也不需要再次存入 annotation。

现有 FinalizeReducerOp 已经包含 `op`、`explicit_plan`、`plan_steps`、`seed` 等字段。到达 Vectorize 的 PTO SIMD finalize 必须有 `explicit_plan`；步骤只允许一个 `(VL,1)`，或带 seed 的空步骤，均使用逻辑 lane 坐标。RMSNorm 对应 `(128,1)`，复杂分组和多步骤计划留作后续扩展。materializer 未生成 finalize 时，Vectorize 按普通存储规则处理结果 copy。

PTO 沿用现有 materializer 的信息分工：归约范围写入 finalize，存储坐标保留在替换后的普通 Buffer 布局中，无需把完整 `PartialFragment` 保留到 Vectorize。Verify/Vectorize 必须能从 `layout_map` 读到这些坐标，缺失时报告前置阶段的输出不完整，不从 `[1]` shape 或一条普通加法反推 reducer 计划。identity、seed 和条件继续使用各自已有的语句或参数。UB 写回依据实际 RHS 类型、地址和执行 mask，规则见 4\.7、7\.3 节。

**访问兼容性检查。** 将当前 Parallel 的逆映射代入 Buffer 索引，再代入 Buffer 的 entry/lane 映射。对于直接转为一次完整 Fragment 读写的访问，entry 索引必须在本轮 lane 间一致，元素对应的存储 lane 必须等于执行 lane。对累加器或广播结果的复制坐标，使用当前执行 lane 对应的位置。条件和 tail 只限制实际更新的元素，不改变 Buffer 的保存方式。

检查以实际映射为准，不按 `i/j` 名称、循环顺序或 ND/DN 名称判断。`B[i+VL]` 若满足范围与上述条件，可以只是访问下一个 entry；交换循环书写顺序也不一定需要重排。以下情况由 Verify 分类报错：

|场景|例子或判定条件|处理|
|---|---|---|
|缺失或陈旧的 Buffer 布局|读写使用新 Buffer，`layout_map` 仍只有替换前的 Buffer；同一 Buffer 的内外层条目不同|报布局缺失或冲突，指出 Buffer 和使用位置|
|同一 Fragment Buffer 的生产、消费方向不兼容|按行保存的 `B[i,j]` 在另一 Parallel 中要求按列取得一条完整向量|报需要 entry/lane 重排，不能直接生成一个 `vload`|
|一个访问需要跨 lane 置换|`B[VL-1-lane]`，或从普通分布式 Fragment Buffer 的某个 lane 取值再广播|已有专用 lowering 之外的情况报不支持；不能把固定索引自动当作 uniform 值|
|从普通分布式 Fragment Buffer 任意提取单元素|Parallel 外读取 `x_frag[7]`|在读取处报不支持，不能先读整个 entry 再借用目标写回形式提取元素；按布局读取累加器完整向量或 finalize 广播结果不属于此类|
|同一轮的 lane 需要不同 entry|`B[q*VL+lane+1]` 横跨两个 entry；任意逐 lane gather/scatter|报访问不能表示为一次完整 entry 读写；不把向量索引传给 Fragment 存储|
|映射范围、shape 或宽度不匹配|用 P 替换逻辑输出 E；某个 entry 越界；Buffer 映射的 VL 与所属 VF 不同|报具体字段不一致；tail 的补齐 lane 不算新增逻辑输出|
|fill/copy 需要尚未定义的重排|源、目的将同一结果元素放在不同 lane；部分 region 不能由现有完整 entry 和条件更新规则表示|报该 fill/copy 无合法 lowering，不默认为逐 entry 原样复制|
|Buffer view 或跨区域使用不受支持|同一 data 的两个 view 无法证明坐标对应；Fragment Buffer 穿越不同 VF 或不同 VL|报别名映射或区域不兼容，不创建互不相干的 Fragment 存储代替同一对象|
|UB 写入不符合当前准入范围|地址随 lane 改变的通用 scatter（4\.7 节定义的 uniform scatter 以外）|报写入不受支持；uniform 地址的同址写入与逐 lane 条件不是拒绝理由（见 5\.2、9\.1 节），也不检查 RHS 各 lane 是否相等|

这些是 Verify 的最终输入检查。LayoutInference 在选择布局时若已经发现无兼容候选，应当就地诊断；不能为了留到 Verify 才报错而写入一份已知矛盾的布局。

**materialization 完成检查。** 同一 Verify 还检查待转换的 PTO SIMD 区域不再包含 `reducer_init`、`reducer_update`、`finalize_reducer_v2`、`local.reducer` 分配、旧 `PartialFragment` 条目或 `reducer_info_v2`。展开后的 `finalize_reducer` 可以存在，但必须带显式计划：单个 `(VL,1)`，或者带 seed 的空步骤。操作只接受 sum/max/min；其他步骤、缺少计划或遗留 `tl.parallel_multiplicity` 均报错。该检查并入 `VerifyParallelToPTO`，不再增加一个 materializer 后置 pass。

单 update 限制在前置收集阶段已经检查。到这里更新已展开成普通算术，不再靠识别 `+=` 或增加 reducer 标记重新计数。后续编译器对循环的复制和展开也不作为新的源 update 计数。

**单 update、目标索引和向量化维。** 一个静态 update 指源程序中对某个 reducer 的一条更新语句。下面只有一个 update，贡献表达式中的两次乘法和加法不分别计数：

```python
for i in T.Parallel(N):
    acc[0] += x[i] * x[i] + y[i] * y[i]
```

下面有两个 update，目标设计拒绝，即使两处索引与布局相同：

```python
for i in T.Parallel(N):
    acc[0] += x[i] * x[i]
    acc[0] += y[i] * y[i]
```

分布在两个 Parallel 中、共同位于同一次初始化和 finalize 之间的两个更新语句也不支持。检查按 reducer 分别计数，不限制整个 VF 只能分配一个 reducer。

在 `AscendLayoutInference` 的 PTO SIMD 区域收集阶段、更新尚为 `reducer_update` 时执行这一准入检查。超过一处直接报告 reducer 名称和更新点数量，不合并兼容布局，也不通过重写两条更新绕过限制。SIMT 的既有规则保持不变，后续编译器复制或展开循环不重新按复制后的节点计数。

一条 update 被循环执行多次，或者其运行时条件全假、实际贡献为零，都仍是单 update。只有初始化而没有静态 update 的程序不列入支持范围。

结构准入还包括：update 位于一个受支持的 1D Parallel 或两层 2D Parallel 中，两层之间不夹串行循环；update 与所选 Parallel 之间不再嵌套额外串行累积循环。`i < d` 这类能由现有控制流规则生成安全逐 lane mask 的条件允许。init 与 finalize 可以一起位于外层串行循环的同一次迭代中，finalize 必须在更新 Parallel 之后；额外的逐 lane 分支、提前退出和不受支持的 finalize 控制流按不支持诊断。materialized finalize 的 `batch` 只接受省略或 `1`，计划只接受单个 `(VL,1)` 或带 seed 的空步骤。单 update 是必要条件，不替代这些结构检查。

目标索引先支持单位维上的常量零和本次 Parallel 的直接循环变量：既包括 `acc[i]` 的保留维输出，也包括 `acc[j]` 的向量化维输出。`acc[j // G]` 等分组索引、需要重排的置换和其他非连续索引留作后续扩展，不因表达式可做 affine 分析就自动准入。二维选维变化时，row/column reduction 的处理随之交换。

逻辑范围 E 由源循环保留，Parallel 执行布局使用的补齐范围为 `P=ceil(E/VL)*VL`（见 4\.5 节的循环布局/Buffer 布局分工）。这项适配属于 LayoutInference 和 PTO materializer，Verify 再核对其输出。

### 8\.7 与 direct reduce 的契约对照和验收用例

reducer v2 的六维度契约见 7\.4 节对照表的右列；此处补充其验收用例。以下用例定义 reducer v2 的正确性边界（验收矩阵与执行进度见配套路线图文档）：

|用例|预期检查|
|---|---|
|RMSNorm：一条平方和更新|单 update 准入；VL=128 时保留 128 份部分和，生成 `vcadd`、广播和结果 copy|
|一条更新，贡献包含多个算术操作|仍只计为一个 update，各算术节点生成向量 SSA 值；仅 Fragment Buffer 的读写转换为 Fragment 声明及读写|
|同一 reducer 的两条更新|无论在同一 Parallel 还是两个 Parallel 中、布局是否相同，前置阶段都报告不支持|
|不同 reducer 各有一条更新|分别计数，不误报为同一 reducer 的多个 update|
|沿向量化维/保留维的直接索引更新|`acc[i]` 在所有 `j/q` 更新后另按 i 遍历执行 `vcadd→广播→copy`；`acc[j]` 在所有 i 更新后按 q 遍历，只做逐 lane combine/copy|
|结构边界|受支持的 1D/2D Parallel、直接目标索引、外层串行循环内同轮 init/finalize 通过；update 前额外串行累积循环、`batch>1` 或多步计划报不支持|
|materializer：跨 lane 步骤非空|输出带 `plan=1` 和非空 `plan_steps` 的 finalize，再输出结果 copy|
|materializer：跨 lane 步骤为空、有 seed|输出带 `plan=1`、空 `plan_steps` 和 seed 的 finalize，再输出结果 copy|
|materializer：跨 lane 步骤为空、无 seed|只输出结果 copy；不生成无操作的 finalize|
|Vectorize：显式步骤非空的 finalize|生成归约指令，先广播结果，再以向量 combine 合并可选 seed、写回并转换结果 copy|
|Vectorize：显式步骤为空、带 seed 的 finalize|不生成跨 lane 规约，以广播的共同 seed 合并各输出，写回并转换结果 copy；结果尾部不越界|
|Vectorize：没有 finalize 或 finalize 没有显式计划|前者按普通规则转换结果 copy；后者在 Verify 报错，不进入 Reduce lowering|
|单 update 的运行时条件全假|部分和保持单位元，结果为单位元或 seed，不误判成没有静态 update|
|Buffer 布局写回与替换|PTO SIMD\_VF 可读到普通 Fragment Buffer、部分和和结果条目；materializer 后无旧 Buffer key 或 `PartialFragment` 残留|
|跨 Parallel 的兼容与不兼容访问|同方向完整 entry 和合法整 entry 偏移通过；按行写按列读、反向 lane、跨 entry 读取按 8\.6 节诊断|
|E=VL\-1、VL\+1、2\*VL\+17 的 row/column reducer|输出 shape 保留 E；entry 向上取整，tail 不改变逻辑结果数量；两个向量化方向均覆盖|
|max 全负值、min 全正值的多轮非整除输入|inactive lane 保留此前累加值，不能被尾轮 zeroing 改成 0|
|RMSNorm 的 `var/rstd_val`|源标量 Bind 随向量输入提升，保持 SSA 定义使用，不额外创建 Fragment 存储；常量广播，`rsqrt` 生成 `vsqrt` 与 `vdiv`，不产生标量浮点 ALU 计算；数值结果按 VMI FP32 指令实际精度核对|
|浮点 sum 的 `vcadd`|materialized reducer finalize 输出显式 `reassoc=True`；普通有序加法链不误识别成 reducer，也不生成 `reassoc=False`|
|reducer v2 / RMSNorm 的 UB 写回|按实际 RHS 类型、地址和执行 mask 检查普通语句的输出；向量 RHS 写 uniform 地址依赖 scatter，不能将 direct Reduce 的 `1PT` 自动套用到该路径|
|Fragment Buffer 任意单元素读取|未支持的 `x_frag[7]` 在读取处报错；不能借用目标写回规则改变该元素的含义|
|非零共同 seed 与多个输出|跨 lane 规约后先广播再 combine；column reduction 的每个有效输出也恰好加入一次共同 seed|
|seed 捕获与外层循环|初始化后覆盖源输入，finalize 仍使用捕获值；下一次初始化重新捕获；前置 materializer 生成的 `seed_frag` 映射到匹配类型的 Fragment 存储|
|seed 输入边界|保留共同标量 seed 的前端用例；Buffer、列表和向量输入被拒绝；PTO 对无法确认的共同初值来源报不支持|
|前置或向量转换遗漏|Verify 拒绝遗留 init/update/finalize\_v2；Vectorize 内部后置验证拒绝遗留 materialized finalize 或 SIMT AllReduce|

数值覆盖 sum/max/min、完整 VL、非整除范围、非零 seed、多输出，以及合法外层循环中重复执行初始化的情况。检查尾部更新保留旧部分和，finalize 不使用最后一轮的 tail mask，seed 按每次初始化求值且每个输出只计入一次。写回测试按 4\.7 节区分标量、连续向量和 uniform scatter，核对实际写入范围；冲突写入用例只检查转换结构，不断言某条 lane 胜出。Fragment 存储与 PTOAS 驻留另行验证。

### 8\.8 pipeline 接入与后置消费者

按第 3 章的位置接入 Fragment 存储规划和 VF 转换：

```text
PTO SIMD LayoutInference / ReducerPlanAndMaterialize
→ PlanPtoSlotStorage
→ 调度、同步、多缓冲 / NormalizeBufferVersion
→ VerifyParallelToPTO
→ LegalizeParallelToPTO
→ VectorizeParallelToPTO
```

前置阶段完成单 update 检查、Buffer 布局推导和 init/update 展开，并生成必要的 finalize/copy。Plan 因而能在调度前计算部分和、结果和 seed Fragment Buffer 的 UB 需求。`VerifyParallelToPTO` 检查前置展开已经完成；`VectorizeParallelToPTO` 的内部后置验证检查展开后的 finalize 也已消耗。两项检查都在 `AscendLowerTileOp` 之前，避免遗漏的 PTO SIMD finalize 进入 `AscendAllReduce`。现有 `VerifyReducerConsumed` 继续作为外层 pipeline 的通用检查，无需为本方案新增公开 pass。

调度、同步、Buffer 版本和多缓冲 pass 维护普通读写、布局、Fragment 规划及 finalize 的引用，包括 seed。此时 backing 已预留，声明和向量访存尚未生成。Vectorize 用记录生成 `declare_fragment_buffer` 和普通 VMI 读写；普通输入 UB 保留 t % 2 版本索引，backing 不复制版本。普通 Fragment、结果 Buffer 和 fill/copy 的后置消费者继续按区域检查。

部分和布局确定 lane 与逻辑输出，finalize 参数确定合并方式。TileLang 按第 6 章预留 UB，通过声明和 VMI 访存表达读写；PTOAS 选择消除访存或使用既定 backing。归约信息不进入 memref 类型，TileLang 不据此做寄存器驻留选择。

Vectorize 内部后置验证只检查本次转换的 PTO SIMD 区域：无待处理的 Parallel、finalize\_reducer\_v2、materialized finalize 或 SIMT AllReduce；原 Fragment 分配和布局条目已经消费，保留 backing。声明与 VMI 访问的来源、类型、entry offset、作用域及 mask 宽度一致。测试另外核对旧值保护、源 predicate、finalize/copy 顺序和 UB 写回范围，不拒绝其他 SIMT\_VF 的合法节点。

## 9\. mask、条件更新、访存和同步边界

### 9\.1 五类 mask 的区分

目标设计中有五类互不相同的 mask 语义，不能混用：

1. update/计算 mask：`active = loop_mask AND source_condition(i_vec)`，组合当前 Parallel 的 tail 与源条件。

2. 条件更新：Fragment 使用 `vstore(candidate, fragment_mem, offset, active)` 保留 inactive lane 旧值；表达式 Select 和双侧选择仍生成 vsel。

3. finalize participant mask：覆盖所有历史部分和的完整参与组，不复用上一轮 tail mask。

4. Fragment 声明/访存：声明无 mask，vload 读取完整 entry，vstore 使用 full 或 active mask；新声明不初始化内容。

5. 普通 UB load/store：单侧条件写入用 parent\_mask \& condition，无条件及双侧 Select 后写入用 parent mask；load 仍依赖调用方可读范围。

若条件对所有 lane 相同，则保留标量条件执行。对逐 lane 条件，store 的 predicate 决定哪些 lane 实际写入；无 mask 的 load 仍要求调用方保证可读范围。

```text
# 单侧 continuous store
vstore(value, base, 0, parent_mask & condition)
# 单侧 uniform store：offset 向量各元素为零
vscatter(value, base, zero_offsets, parent_mask & condition)
# 双侧同址 store 已安全合并为 Select
value = vsel(condition, then_value, else_value)
store_by_address_pattern(value, address, parent_mask)
```

`if tid == 6: ub[0] = x[tid]` 只能让对应 active lane 执行 scatter。不能给 false 分支补写旧值并扩大为 parent mask，否则会增加同址写入者。原程序无条件的多 lane 同址冲突写入使用 parent mask，其结果不作确定性承诺。

### 9\.2 remaining、create\_mask 与物理 predicate

非整除路径采用 4\.1 节的 `P`、`Q`。完整 chunk 和尾部 chunk 共用一个循环体，mask 根据剩余元素数生成。下面是目标 PTODSL 的结构示意，省略计算和访存：

```python
remaining = pto.const(E, dtype=pto.i32)
for q in range(0, Q):
    chunk_base = q * VL
    mask = pto.vmi.create_mask(remaining, size=VL)
    ...
    remaining = remaining - VL
```

以 `E=150, VL=128, P=256, Q=2` 为例：

|q|chunk\_base|进入本轮的 remaining|有效 lane 数|更新后的 remaining|
|---|---|---|---|---|
|0|0|150|128|22|
|1|128|22|22|\-106|

remaining 使用有符号标量，最后一次更新后的负值不再用于 mask 或访存。无需为这个未使用的结果额外生成饱和减法。`E<VL` 时循环执行一次；`E=VL` 和其他整除输入的每轮 mask 都全有效，可以沿用整除路径循环外的全量 mask，省去剩余数状态。

二维按 4\.4 节在每行或列开始前重置 remaining。顺序 Parallel 即使共用同一个 VL，也各自使用自己的 E 和 remaining。选择这种递推写法是为了直接表达剩余数；是否比 `E-q*VL` 少一条机器指令，需要看优化后的 IR 或汇编，不能仅按源码算术次数判断。

**TileLang 和 PTODSL 的分工。** Vectorize 输出的 TIR 必须保留 remaining 的初始化、读取和更新。复用现有单元素 `local.var` 标量表示：初始化和更新用 `BufferStore`，读取用 `BufferLoad`，PTO codegen 将它们输出为同名 Python 标量赋值。这是生成代码内部的状态，不放宽用户 Parallel 体的访存准入。

普通临时结果继续用一次定义的 Bind，remaining 复用可变标量表示，无需给 TIR For 新增循环参数接口。PTO codegen 输出普通 `range` 循环和标量递推，PTODSL AST rewrite 根据赋值关系生成 `scf.for iter_args` 和 `scf.yield`。初始化须产生与更新类型一致的运行时整数标量；本节以 i32 示意，实际沿用现有 codegen 的整数类型约定。

实现时必须检查这段标量状态经过 DCE、后续 TIR pass 和 codegen 后仍能被识别为循环递推，且不产生额外的 UB 或栈上状态访存。

**create\_mask 与物理 predicate。** VMI `create_mask(n, size=VL)` 生成前缀 mask：`n>=VL` 时全有效，`0<n<VL` 时前 n 个 lane 有效，`n<=0` 时全无效。它返回 mask，不修改参数；循环间的递减来自上面的显式 `remaining = remaining - VL`。

PTO C\+\+ 的 `CreatePredicate<T>(sreg)` 使用 `plt_b8/b16/b32(sreg, POST_UPDATE)`，会更新引用参数，扣减量由物理 predicate 的粒度决定。动态 VMI `create_mask` 可 lowering 为同一类 `plt_b*` 操作，但其中产生的剩余数只用于构造本次逻辑 mask 的各个物理 part，不会隐式更新下一轮循环的 remaining。VL 是逻辑 lane 数，不能直接当成一条物理 `plt` 指令的宽度。

PTOAS 负责 mask 的 b8/b16/b32 粒度、layout 和物理拆分。混合位宽的 load、算术、转换及 store 都保持 VL 个逻辑元素；当前 chunk 的有效逻辑 lane 集合保持一致，不在 TileLang 按字节重新缩放 mask。

### 9\.3 计算和访存范围

|操作|处理|
|---|---|
|`vadd`、`vmul`|使用当前 chunk mask|
|`vdiv`、`vsqrt` 等其他受支持算术|使用当前 chunk mask；inactive lane 语义同 `vadd`/`vmul`|
|`vcmp`、`vcmps`|以当前执行 mask 为 seed 生成比较 lane mask，不产生数据写回|
|`vsel`|cond 外 lane 取另一操作数（merge 语义），写回范围由对应 store 的 mask 限定|
|`mask_and`、`mask_or`、`mask_not`|只作用于逻辑 mask，不产生数据 lane|
|NORM `vstore` / uniform `vscatter`|单侧写入使用 `parent_mask & predicate`，其余使用 parent mask；NORM 按 lane 地址写入，scatter 按 offset 写入，均不把 inactive lane 加入写入者|
|`vload`|连续读取完整 VL 个元素，mask 不保护读取|
|`vbrc`、`vci`|生成 VL 个逻辑元素|
|`vcvt`|按原转换属性处理 VL 个逻辑元素，不传 mask；最终由 masked store 限定写回范围|
|lane\-uniform 标量 load|仍只读取当前程序点对应的一个元素，按需广播|

无效 lane 的中间值不作为程序输出，也不经跨 lane 操作影响有效结果；跨 lane 归约只接受第 7、8 章定义的两种形式。若目标 Buffer 只分配 E 个元素，masked store 的有效写入应落在其合法范围内，仍须满足现有后端的对齐和寻址约束。原地更新的目标如果也被 vload 读取，同样需要满足完整向量可读条件。

无效 lane 不写回是必要语义：sentinel 检查和只分配 E 个元素的紧凑输出都依赖它。部分写回的实现方式以后端 predicated store 的实际行为为准，须按实际使用的后端版本和受支持 dtype 逐一验证，NORM store 不压紧稀疏 active lane 的地址；direct Reduce 的单元素 `1PT` 写回另见 7\.2 节。

调用方保证每个 tail 连续 load 起点后的完整 VL 个元素可读，并在进入 SIMD 区域前初始化所需 padding。可以显式分配补齐 Buffer，也可以提供其他满足条件的可读范围；本设计不要求固定的 Buffer shape 或证明注解。对偏移访问、多 Buffer 以及二维的每行或每列，该前提分别成立，不能只按 For 的 E 推断分配大小。

Verify 不证明 padding、边界或初始化，也不因无法证明而拒绝。Vectorize 不自动扩展 Buffer，不插入 fill、边界检查或额外同步，不调整 pass 位置和调度逻辑。padding 的生命周期及其访问所需同步由调用方保证。

### 9\.4 Fragment 条件写回与 inactive lane 语义

Fragment vload 读取完整 entry，vstore 按逻辑 mask 更新其中的 lane。inactive lane 保留目标 backing 的旧值；声明和分配本身不初始化内容。

以尾部累加为例，某个 lane 在最后一轮无效，不代表它前面各轮的部分和可以丢弃。语义上需要：

```text
old = vload(acc_frag_mem, 0, size=VL)
candidate = vadd(old, contribution, active_mask)
vstore(candidate, acc_frag_mem, 0, active_mask)
```

candidate 是 SSA 计算结果，masked vstore 负责保留目标旧值，不额外生成仅为写回保留旧值的 vsel。累加仍需要读取 old；表达式 Select 和双侧分支选择仍生成 vsel。目标组合不支持该存储语义时应诊断或使用经验证的等价 merge，不能静默覆盖全向量。

普通 Fragment 先把 256 个元素填为 7，再把前 129 个改为 9，最后读出全部元素，应得到前 129 个 9、后 127 个 7。稀疏条件与 tail 相交时，同样只更新 active lane。

累加器初始化为归约单位元，无贡献的 lane 保持单位元或此前部分和。完成所有轮次后，finalize 应覆盖参与部分和的 lane，不能误用最后一轮的 tail mask。对于 4096/128 的例子，每轮都是 full mask。

普通 element\-wise 的 tail 规则（masked store 保护写回、满宽 load 依赖调用方 padding）不能替代 reducer 的 inactive\-tail 规则：前者只涉及单轮读写的有效范围，后者要求跨轮次保留历史部分和并在 finalize 时使用完整参与组 mask。两者是不同的验收规则。

内部 backing 的完整 entry 容量由 Plan 保证，内容由源 fill/store 定义；reducer 初始化必须覆盖全部参与 lane。普通 UB 的满宽 vload 仍依赖调用方 padding、初始化、对齐和同步，条件不能保护无 mask 的读取。声明不使无效 UB 地址变得可读。

### 9\.5 同步边界

- 普通 UB 的版本槽位、调度与任务间同步继续由 Vectorize 之前的 AutoSchedule、InsertSync、多缓冲等外层 pass 负责；Vectorize 不新增或调整这些同步。

- Fragment backing 不生成 DMA、copy 或 fill 任务，也不增加虚假的读写来影响调度。

- 同一 SIMD\_VF 内部 UB 地址之间的读写依赖由 PTOAS 分析并插入必要的 `mem_bar`，覆盖 Vectorize 生成的普通 UB load/store 和 Fragment Buffer 对应的 Fragment UB 读写；TileLang 的外层 `AscendThreadSync` 不承担这项 VF 内同步。跨 VF 访问使用源程序的显式 barrier 或既有跨区域同步。

- 普通输入 UB 的版本槽位与同步按源程序保留；把 vload 改成从原始输入 UB 地址重读不属于本设计范围（见 10\.2 节）。

优化关闭或未能消除访存时也必须生成正确同步。mem2value 删除已无用途的 barrier 不能替代保留访问的依赖分析和 mem\_bar 生成。

## 10\. TileLang / PTOAS 职责边界

### 10\.1 职责划分

|范围|TileLang|PTOAS / 工具链|
|---|---|---|
|总体职责|lane 布局、E/P/Q/remaining、tail mask、pass pipeline、VMI 生成、Fragment Buffer→backing/handle 逻辑映射、UB 容量预算|物理寄存器拆分、mask 物理化、Fragment 存储的 vreg 驻留 / 访存消除或保留（对已绑定 backing）决策、依赖分析、`mem_bar`、CSE/LICM|

按设计内容展开：

|内容|TileLang|PTOAS|
|---|---|---|
|逻辑数据对应关系|确定 q/lane、向量类型、entry 数量和访问索引|消费这些已确定的关系|
|计算与控制流|生成运算、mask 和归约；普通计算结果通过 SSA 定义使用关系传递|保持语义地优化数据流，并为向量 SSA 值分配寄存器|
|Fragment Buffer 存储|仅将输入 TIR 中 `local.fragment` Buffer 的分配和读写转换为 Fragment 声明及读写|根据使用关系决定寄存器驻留或保留对 Fragment UB backing 的读写|
|UB 容量与地址|在 AutoSchedule 前固定预留；MergeUBAllocations 分配不重叠地址；backing 不参与多缓冲复制|使用已预留地址，不能因消除访存而减少 TileLang 的预算|
|生命周期与压力|不分析 liveness 或决定寄存器驻留；backing 地址不复用|分析 liveness、物理拆分后的 vreg/predicate 压力和成本|
|实际存储与同步|输出带 backing 的 Fragment 声明及读写，保留已有外层任务同步|将未消除的 Fragment 读写 lower 为已绑定地址上的 UB 访存，并插入必要的 `mem_bar`|
|mask|生成逻辑 mask（`boolxVL`）和 `create_mask` 调用|mask 的 b8/b16/b32 粒度、layout 和物理拆分|

PTOAS 可以将某个 Fragment 存储的 store/load 消除，让一个值跨多个使用点保留在 vreg；也可以保留对同一 Fragment entry 的多次 load，从已绑定的 UB backing 读取，缩短寄存器占用时间。RMSNorm 的部分和适合评估循环携带的寄存器表示，x 则需要结合整行容量和两次消费的距离决定是否保留 Fragment 访存。本设计不预先指定优化结果。

普通向量 SSA 值未经过 Fragment 存储，也不承诺永久占用物理寄存器，仍按 PTOAS/后端既有的寄存器分配流程处理。本设计不扩展普通 SSA 值的溢出机制，也不为它们额外预留 Fragment backing。TileLang 保留 SSA 表达是基线行为，不属于 Fragment 读写消除，也不需要先分析生命周期。

重复向量 load 是否合并由 PTOAS 的 CSE 决定；TileLang 不维护 `MemoryAccessKey → vector Var` 缓存，每次 continuous 访问的原始 BufferLoad 分别生成 VMI load。标量 UB load 保留在 chunk 循环内，是否提升由 PTOAS 的 LICM 决定；lane\-uniform 只表示同一程序点各 lane 取值相同，不能据此认定内存跨 chunk 不变。标量参与向量运算时使用 `vbrc`；若 PTO 和 PTOAS 支持 `vadds`、`vmuls` 等价操作，可匹配 vector\-scalar 形式直接生成专用指令，否则回退到 `vbrc + vector-vector op`。不做浮点重结合。

### 10\.2 职责不倒置

- TileLang 负责逻辑 lane、entry、E/P/Q、mask，Fragment 声明与访问契约，AutoSchedule 前 UB 容量预算，backing 地址进入 UB allocation，Fragment 声明和 VMI 访存生成。这些职责不能交给 PTOAS。

- PTOAS 负责物理寄存器拆分和 predicate 物理化，Fragment 存储的 vreg 驻留、访存消除或保留对已绑定 backing 的读写等选择，liveness/pressure 和成本分析，依赖分析、必要 `mem_bar`，CSE/LICM 等后端优化。TileLang 不生成物理拆分计划。

- 目标设计可以说 PTOAS 可能保留或消除 Fragment 读写，但不能把 UB 容量预留、UB 地址分配或 TileLang Fragment 存储契约 交给 PTOAS。

- 重算（rematerialization）和将 vload 改成从 x 的原始输入 UB 地址重读，不属于本设计的范围，也不转交给 TileLang；值的重算列为未来优化方向。本设计所说的保留多次 load，均指读取该 Fragment 已绑定的 backing，PTOAS 不另行分配存储；原输入 UB 的版本槽位与同步仍按源程序保留。动态索引 q 本身合法；若选择寄存器表示需要进一步展开或改写，由 PTOAS 决定，不能把 q 直接用作物理 vreg 编号。

第一版使用普通 memref 的别名语义。声明不承诺 noalias；无法证明安全时保留访存，正确性不依赖优化。若实测确认别名分析过于保守，造成必须解决的性能问题，可考虑新增 slot\_load/slot\_store 等专用接口，明确该区域仅由专用接口访问。该方向不是第一版前置条件。

## 11\. 支持边界与正确性条件

### 11\.1 输入支持边界

目标支持边界按能力分别定义：

**基础 element\-wise（第 4、5 章）：**

- `T.SimdVF(lanes=...)` 内的一维 `T.Parallel(N)`，或满足连续维约束的二维 `T.Parallel(M, N)`；

- 各维 extent 都是编译期常量；选中向量化维允许常量非整除；

- UB 上的连续向量 load/store，元素 dtype 符合通用元素寻址和 mask 约定，并受对应 VMI 操作的支持范围约束；

- 地址不依赖 lane 的标量 UB load，进入向量运算时再广播；普通标量 store，以及按执行 mask 写入 uniform 地址的向量 scatter；

- `BufferLoad`、常量、标量变量、循环下标、Add、Mul、有限 Cast、原 TIR 中的 `Bind` 和 `BufferStore`；

- 一个区域内多个顺序排列的 Parallel，以及多个按原顺序排列的 `BufferStore`。

不属于目标支持边界的形式：跨 lane 的 shuffle、原子操作、复杂 Call、三维及以上 Parallel、混合循环结构、Parallel 内的 Serial/Pipelined/While 或不受支持的条件分支、包住 Parallel 整体的 `Vectorized` 循环、转换区域内的 `ThreadBinding` 循环（并列或包裹）、GM 直接访问，以及 group、block\-stride、dintlv、通用 gather/scatter 等其他访存模式（4\.7 节定义的 uniform scatter 除外）。FP4 等需要额外打包、解包、寻址单位转换或 mask lane 数调整的特殊类型不纳入待转换 Parallel 的支持范围；此范围只约束待转换的 Parallel，不改变已有手写 VMI 对这些类型的支持。一维 `Parallel(1)` 和二维 `(1,1)` 的单点形态按 4\.6\.1 节分类拒绝。

**可规范化控制流（第 5\.3、9 章）：** 已证明安全的 If→Select、Min/Max→Select 和布尔表达式规范化由 Legalize 完成；单侧 store 保留 predicate，双侧同址 store 才合并 Select；逐 lane 条件由 ControlFlow 模块转成 mask 与 `vsel`；不能安全求值或会扩大读取范围的形式拒绝。

**direct reduce（第 7 章）：** 直接规约调用作用于已存在的 UB region；sum/max/min；单元素结果使用 `1PT` 写回；多输出 region 留待扩展。

Fragment Buffer（第 6 章）：仅处理待转换 VF 中已有的 local\.fragment Buffer；生产和消费布局兼容，entry 索引在本轮 lane 间一致。声明使用静态 count/VL、标量 dtype、UB 地址空间；读写使用固定 handle 加元素 offset，可含循环变量。访问必须落在完整 backing entry 内；partial write 使用 mask，不扩大逻辑输出域。

内部 backing 的完整 entry 容量由 Plan 保证；逻辑 padding 不扩大用户可见范围，未初始化 lane 不参与有效输出或规约。普通 UB 的 tail 可读与初始化契约继续由调用方承担。

**reducer v2（第 8 章）：** 每个 reducer 一个静态 update；目标索引为常量零或当前 Parallel 直接循环变量；sum/max/min；finalize 带显式计划（单个 `(VL,1)` 或带 seed 的空步骤）；共同标量 seed；`batch` 省略或 `1`。

### 11\.2 正确性条件与程序员契约

1. **并行语义与依赖。** 程序满足 `T.Parallel` 并行迭代语义是输入契约，由程序员负责。Verify 和 Vectorize 不检查不同逻辑点之间的 RAW、WAR、WAW 依赖，不证明 Buffer 别名，也不因未知别名拒绝输入。Verify 通过只表示满足结构和指令约束；多 lane 同址冲突写入不承诺确定结果，不证明唯一写者。

2. **tail load 可读范围。** 调用方保证每个 tail 连续 load 起点后的完整 VL 个元素可读，并在进入 SIMD 区域前初始化所需 padding；padding 的生命周期和同步由调用方保证。Verify 不证明这些条件，Vectorize 不自动扩展 Buffer。

3. **无效 lane 不写回。** masked store 只写入有效 lane；sentinel 检查和紧凑输出都依赖该语义。部分写回的实现以后端 predicated store 的实际行为为准，需逐一验证。

4. **不得以 shape 或固定索引推断 uniform。**`acc[0]` 在 reducer finalize 前可以是每条 lane 不同的部分和；Fragment Buffer 的固定逻辑索引不表示各 lane 的值相等。只有由地址分析确认的 lane\-uniform UB load，或 materialized reducer finalize 后明确广播的结果，才可以按 uniform 语义处理。

5. **浮点 sum 的重结合。** 浮点跨 lane `vcadd` 必须携带显式 `reassoc=True`；严格要求原顺序的程序不属于该 lowering 的支持范围。max/min 需核对 NaN 和有符号零行为；验证容差不能代替指令语义约定。

6. **无效 lane 不污染有效结果。** 无效 lane 的中间值不作为程序输出，也不经跨 lane 操作影响有效结果；reducer 的 inactive lane 保留历史部分和，finalize 使用完整参与组 mask。

7. **依赖相关用例的前提。** 覆盖各点原地更新、同一点内多语句依赖、顺序 Parallel 之间经 Buffer 传递结果的用例，必须使用合法 Parallel 输入；不以串行执行结果作为非法 Parallel 的验收标准，不执行已知越界输入来验证拒绝行为。

8. **数值比较方法。** 应与参考计算和被测 dtype 的语义一致，记录所用容差或精确比较条件；以独立参考计算为基准，其他后端结果相同并不足以确认正确性。

9. **兼容性。** AscendC 仍走现有 `AscendSimdVFLowerParallel`；PTO SIMT 仍使用原线程布局；没有待转换 Parallel 的手写 VMI 区域由本路径跳过，继续正常编译。AscendC 的 `SIMD_VF` 区域内 Parallel 不新增任何 PTO 布局注解（无 PTO key，也无 `parallel_loop_layout` / padding guard）；PTO 的无 Parallel 手写 VMI 区域同样不新增注解。

### 11\.3 后续扩展方向

以下方向不属于当前目标支持边界，记录为扩展设计入口：

**符号有效长度。** 要求前端显式写出规范化的 guard：

```python
for i in T.Parallel(E_padded):
    if i < E:
        out[i] = inp[i] + 1.0
```

其中 `E_padded % VL == 0`，并由程序员保证 `E <= E_padded` 和尾部 load 安全。pass 将 `i < E` 代入 `i=q*VL+lane`，生成当前 chunk 的 lane mask，用于约束支持 mask 的计算和 store。无 mask 的 load、vbrc、vci 和 vcvt 不受该 mask 保护；尤其不能用它保证尾部 load 不越界。先以 guard 直接位于 Parallel 体最外层、条件为当前循环变量 `i < E`、条件体内没有其他分支的形式为目标，复用每个 chunk 的 remaining 和 prefix mask；若循环还覆盖有效长度之后的 chunk，remaining 可以为零或负数，mask 全无效，但完整 load 仍依赖 padding 前提。原始 For extent 和布局 shape 也为符号值的情况，还需设计布局求逆、范围证明及循环上界，不能只把整数 E 换成表达式便视为支持。

**其他访存模式。** 后续扩展仍以逆映射后的元素地址为依据，不仅凭原循环维数选择指令：

|模式|需要从地址分析中确认的信息|
|---|---|
|group / block\-stride|组内连续范围、组间步长和重复步长，以及这些范围与逻辑 lane 的对应关系|
|dintlv / interleaved|交错周期、各路地址及 lane 顺序；还需定义多路向量结果如何被后续表达式使用|
|通用 gather/scatter（uniform scatter 以外）|每条 lane 的偏移向量、寻址单位和可用操作；无法归入规则连续模式时不能直接按 continuous lowering|

`vmi.py` 已有 group、block\_stride、repeat\_stride 和 dist\_mode 等接口，可作为扩展时的核对入口。每种模式仍需确认 PTO codegen、PTOAS 的实际支持，以及 dtype、对齐、mask 和读写范围约束，并增加对应测试。它们不属于当前目标范围，也不改变程序员对合法 Parallel 输入和访存安全的责任。

**其他扩展入口。** 单元素 Parallel 的两条路线见 4\.6\.1 节；FP4 等打包类型见 5\.6\.2 节；reducer 的复杂分组、多步骤计划、分组目标索引（`acc[j // G]`）和逐输出 seed 见第 8 章；局部 DCE 迁移到 PTOAS 的条件见 5\.5\.5 节。Parallel 体内的串行循环当前拒绝（见 5\.2 节）；候选支持路线有三条，待讨论确定：Legalize 展开编译期常量界的串行循环，并把分支臂内同一位置的顺序写链合并为表达式；Vectorize 直接支持 uniform 串行循环（chunk 内的向量循环携带值）；或在验收层改写用例形态。

## 12\. 目标输出示例

### 12\.1 vecadd：Vectorize 输出

以下展示 vecadd 经 Vectorize 的目标直接输出，省略外围 `SIMD_VF` block。`access_ptr` 按 `(本次访问起点, access_type, extent)` 表示完整访问区间，VMI 的额外 offset 为 0。

两个 `temp1` BufferLoad 分别生成 VMI load，尚未经过后续 load 消除：

```python
mask: T.boolx128 = T.tl.vmi.create_mask(128, size=128)
for q in T.serial(0, 64):
    a_mul: T.float32x128 = T.tl.vmi.vload(
        T.access_ptr(temp1[iter % 2, q * 128], "r", 128),
        0,
        size=128,
    )
    a_add: T.float32x128 = T.tl.vmi.vload(
        T.access_ptr(temp1[iter % 2, q * 128], "r", 128),
        0,
        size=128,
    )
    b: T.float32x128 = T.tl.vmi.vload(
        T.access_ptr(temp2[iter % 2, q * 128], "r", 128),
        0,
        size=128,
    )
    s: T.float32x128 = T.tl.vmi.vadd(a_add, b, mask)
    c: T.float32x128 = T.tl.vmi.vmul(a_mul, s, mask)
    T.tl.vmi.vstore(
        c,
        T.access_ptr(temp3[iter % 2, q * 128], "w", 128),
        0,
        mask,
    )
```

`access_ptr` 的起始下标包含版本槽位、外层坐标和 chunk 位置，extent 为 VL 个元素；连续 VMI 从该指针开始访问，lane 偏移由向量语义隐含表示。逻辑向量类型和 mask 保留到 PTOAS，由 PTOAS 负责物理寄存器拆分和 predicate 粒度处理。

若后端 CSE 确认两个 temp1 load 等价，可以将加法和乘法中的对应输入统一引用同一个向量值。预期优化形态如下，仅用于说明值的复用关系：

```text
a = vload(temp1_ptr, 0, size=VL)
b = vload(temp2_ptr, 0, size=VL)
s = vadd(a, b, mask)
c = vmul(a, s, mask)
vstore(c, temp3_ptr, 0, mask)
```

这里的指针均指向当前 chunk。是否发生合并，以 PTOAS 优化前后的 IR 对比为准。

### 12\.2 RMSNorm：Fragment memref、finalize 与广播

以下是 RMSNorm 拟生成的语义示意（目标设计示意，不是编译器当前输出）。`V = vector<128xf32>`，backing 使用 6\.1 节的分配，`vload/vmul/vadd` 表示对应 VMI 操作。示例显式保留原表达式中两次 `x_frag[i]` 读取，不在 TileLang 中做 load CSE。

```text
x_frag_mem = declare_fragment_buffer(x_backing_ptr, 32, 128)
acc_frag_mem = declare_fragment_buffer(acc_backing_ptr, 1, 128)
full = create_mask(128, size=128)

# 第一个 Parallel：保存整行 x。
for q in range(32):
    x_def = vload(x_ub_ptr, q * 128, size=128)
    vstore(x_def, x_frag_mem, q * 128, full)

zero_def = vbrc(0.0, size=128)
vstore(zero_def, acc_frag_mem, 0, full)

# 第二个 Parallel：更新部分和。
for q in range(32):
    acc_use = vload(acc_frag_mem, 0, size=128)
    x_lhs = vload(x_frag_mem, q * 128, size=128)
    x_rhs = vload(x_frag_mem, q * 128, size=128)
    sq_def = vmul(x_lhs, x_rhs, full)
    acc_def = vadd(acc_use, sq_def, full)
    vstore(acc_def, acc_frag_mem, 0, full)
```

`x_frag_mem` 对应 `x_frag`，保留 32 个 entry，最后的输出循环还要再次读取；`acc_frag_mem` 对应 materializer 生成的 `sum_sq_1`，包含一个 entry。按固定 UB backing 计算，x 预留 16 KiB，累加器预留 512 B。这些是 TileLang 为 Fragment Buffer 预留的保存容量；PTOAS 可以消除部分 Fragment 读写，让值驻留在寄存器中，也可以保留对该 UB backing 的实际读写，但不能把已向 AutoSchedule 报告的预留容量返还给调度器。

`sq_def` 是原表达式 `x_frag[i] * x_frag[i]` 的向量结果，直接传给 vadd，不创建 backing。x 的保存、单位元初始化和 acc 更新分别对应源 BufferStore、T\.fill 和累加器 BufferStore，才生成 Fragment vstore。地址、循环计数和 mask 按各自类型表达。

RMSNorm 的 finalize 需要合并各 lane 的部分和，并广播完整结果。`result_frag_mem` 对应前置 pass 已生成的 `sum_sq_result` Fragment Buffer。`vcadd` 默认返回单 lane 向量，直接广播成 128\-lane 向量后写回累加器 Fragment 存储，再执行结果 copy（目标设计示意）：

```text
result_frag_mem = declare_fragment_buffer(result_backing_ptr, 1, 128)
acc_use = vload(acc_frag_mem, 0, size=128)
red_def = vcadd(acc_use, full, reassoc=True)
replicated_def = vbrc(red_def, size=128, group=1)
vstore(replicated_def, acc_frag_mem, 0, full)

# 对应源 T.copy。
result_use = vload(acc_frag_mem, 0, size=128)
vstore(result_use, result_frag_mem, 0, full)
```

广播在本设计中是 AllReduce lowering 的组成部分。`acc_frag_mem` 始终保存 `vector<128xf32>`：finalize 前各 lane 是不同的部分和，finalize 后各 lane 是相同的完整结果。规约和广播之间直接传递 `red_def`，不分配临时 Fragment 存储。后续读取原累加器或 `sum_sq_result` 都能直接取得正确的向量结果，无需切换到另一份单 lane 读取映射。

这保证的是每条 lane 观察到的值，不要求后端永久保留 128 份物理副本。PTOAS 可以在保持该语义的前提下消除冗余复制和 Fragment 存储往返；TileLang 的基线转换完整保留规约、广播及写回。

有共同标量 seed 时，materializer 在初始化位置生成 `seed_frag` 的捕获写入。该 Buffer 的 scope 为 `local.fragment`，因此 Vectorize 将其映射为 `seed_frag_mem`；SIMT 路径使用的 `local` 标量 Buffer 不直接映射为 Fragment 存储。以 sum 为例，`acc_frag_mem` 的单位元初始化和循环更新仍按上例生成，`result_frag_mem` 沿用上例（目标设计示意）：

```text
# Plan 为 seed_frag 预留一个 FP32x128 entry。
seed_frag_mem = declare_fragment_buffer(seed_backing_ptr, 1, 128)

# 源初始化位置捕获 seed。
seed_scalar = evaluate_source_seed()
seed_def = vbrc(seed_scalar, size=128)
vstore(seed_def, seed_frag_mem, 0, full)

# 单位元初始化和全部更新后，只读取捕获的 seed。
acc_use = vload(acc_frag_mem, 0, size=128)
red_def = vcadd(acc_use, full, reassoc=True)
replicated_def = vbrc(red_def, size=128, group=1)
seed_use = vload(seed_frag_mem, 0, size=128)
combined_def = vadd(replicated_def, seed_use, full)
vstore(combined_def, acc_frag_mem, 0, full)
result_use = vload(acc_frag_mem, 0, size=128)
vstore(result_use, result_frag_mem, 0, full)
```

`evaluate_source_seed()` 表示在源初始化位置对受支持的标量初值求值，不是新增 intrinsic。若该初值在转换中已经被广播成向量，就直接写入 `seed_frag_mem`，不重复广播。`red_def`、`replicated_def` 和 `combined_def` 按定义使用关系传递；Fragment 读写只对应 seed 捕获 Buffer、累加器和结果 Buffer。

max/min 使用对应的归约和 combine 指令，并遵守 8\.5 节的数值语义。column reduction 省去 `vcadd` 和结果广播，直接读取部分和及 `seed_frag_mem` 做向量 combine，按有效输出 mask 保留尾部旧值后写回。

后续 `var/rstd_val` 随向量输入按类型规则提升，常量广播后参与向量除法、加法及平方根倒数计算，不能生成标量浮点 ALU 指令。`z_rstd_ub[t % 2,0]` 的写回按 4\.7、7\.3 节依据实际 RHS 类型和地址处理；若 RHS 为向量，则依赖 uniform scatter。该路径需在 reducer v2/RMSNorm 阶段核对有效 lane、数值和相邻地址，不由 direct Reduce 的单元素写回验收代替。

最终输出循环用 `vload(x_frag_mem, q * 128, size=128)` 读取 x，使用已定义的 rstd 向量，加载权重并完成乘法。权重和乘法结果直接传递，每轮按输出 mask 写 UB。跨语句或跨循环使用 SSA 值不触发 backing 分配；物理寄存器处理由 PTOAS 完成。

### 12\.3 输出约束

Vectorize 完成后，`T.Parallel` 替换为 chunk 循环，二维情况下保留非向量化维的外层循环。每行或列有 Q 个 chunk，mask 只允许原始 E 个逻辑点写回。`pto_parallel_loop_layout` 已消费并删除，外围 `SIMD_VF` block、scope、`tl.vf_source_index` 和 `tl.simdvf_lanes` 保留。

双缓冲版本槽位下标、同步、调度以及有效的 Buffer、来源元数据随区域保留。元数据若引用被替换的循环变量，须按新循环和访问范围更新，不能留下失效引用或未绑定的 lane 占位符。

VMI 使用逻辑向量类型，物理寄存器和 predicate 分配由 PTOAS 完成。计算、访存和顺序 Parallel 的执行顺序遵循 5\.5\.4 节。

声明在 VF 内支配全部使用，Fragment 访存使用固定 handle 加元素 offset，输出 memref 的 shape=count\*VL、element type 为标量、地址为 signless i64 字节地址。对非零 backing 偏移、动态 entry、单份与双缓冲共存、partial fill/copy、inactive lane 保留和完整参与组 finalize 分别验收。优化开/关均对独立数值参考验证，并检查保留访存时的同步；不能只比较两者相同。无法消除访存单独记录优化结果，不直接判为转换失败。

实现状态与验收进度见配套路线图文档。

