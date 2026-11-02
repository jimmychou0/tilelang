/*!
 * \file pto_parallel_layout.h
 * \brief PTO lane-layout construction for T.Parallel units.
 *
 * The PTO SIMD_VF pipeline distributes the logical points of a T.Parallel unit
 * over the block's [0, L) lanes. This helper reads the original loop nest,
 * applies the PTO dimension-selection, padding and access proofs, and returns
 * the lane Fragment. It does not rewrite the input TIR and has no dependency
 * on the generic parallel layout-inference worklist.
 */

#ifndef TVM_TL_ASCEND_LAYOUT_PTO_PARALLEL_LAYOUT_H_
#define TVM_TL_ASCEND_LAYOUT_PTO_PARALLEL_LAYOUT_H_

#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>

#include "layout/layout.h"

namespace tvm {
namespace tl {

namespace attr {
// ForAttr, PTO SIMD lane layout for a parallel for loop inside a PTO
// SIMD_VF block. Distinct from kParallelLoopLayout so SIMT lowering never
// misreads a PTO lane fragment as a thread layout.
constexpr const char *kPtoParallelLoopLayout = "pto_parallel_loop_layout";
} // namespace attr

namespace pto {

/*!
 * \brief Build the PTO lane Fragment for one T.Parallel unit.
 *
 * \param root       Outermost For of the unit (1D T.Parallel or a two-layer
 *                   Parallel nest).
 * \param lane_bounds Lane range [0, L) of the enclosing SIMD_VF block.
 * \return Fragment mapping the unit's input coordinates to lanes and per-lane
 *         chunk positions.
 *
 * The original loop extents, buffer shapes and access indices are not
 * modified; only the Fragment input extent of the vectorized dimension is
 * padded when the logical extent is not a multiple of L.
 */
Fragment InferPtoParallelLayout(
    const tirx::For &root, const Range &lane_bounds,
    /*! Solved fragment layouts (layout-solver output injected into the
     * annotated map). When a collected access's buffer has an entry here,
     * its candidate proof uses the layout's forward maps (storage lane
     * equals the execution lane) instead of the logical row-major offset.
     * Null when no fragment layouts are in scope. */
    const ffi::Map<tirx::Buffer, Layout> *fragment_layouts = nullptr);

} // namespace pto
} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_LAYOUT_PTO_PARALLEL_LAYOUT_H_
