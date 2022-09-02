/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_SINGLE_DIMENSION_COEFFICIENT_KERNEL_TEMPLATE
#define MGARD_X_SINGLE_DIMENSION_COEFFICIENT_KERNEL_TEMPLATE

#include "../../../RuntimeX/RuntimeX.h"

#include "../../MultiDimension/Coefficient/GPKFunctor.h"

#define DECOMPOSE 0
#define RECOMPOSE 1

namespace mgard_x {

template <SIZE BlockSize, DIM D, typename T, typename DeviceType>
MGARDX_EXEC bool SingleDimensionCoefficientComputeCoarseIndex(
    const Functor<DeviceType> &functor,
    const SubArray<D, T, DeviceType> &coefficients, SIZE coarse_index[D]) {
  // For up to 3 dimensions, the array indices will follow the
  // block/thread indices. However, the scheduling can only happen
  // up to those 3 dimensions, so all further array directions are
  // appended to the first block scheduling dimension.
  SIZE first_dimension_size = div_roundup(coefficients.shape(D - 1), BlockSize);
  SIZE block_idx_x = functor.GetBlockIdX();

  coarse_index[D - 1] =
      (block_idx_x % first_dimension_size) * BlockSize + functor.GetThreadIdX();
  block_idx_x /= first_dimension_size;

  if (D >= 2) {
    // The block size is always 1 in the y dimension.
    coarse_index[D - 2] = functor.GetBlockIdY();
  }
  if (D >= 3) {
    // The block size is always 1 in the z dimension.
    coarse_index[D - 3] = functor.GetBlockIdZ();
  }

  for (int d = D - 4; d >= 0; --d) {
    coarse_index[d] = block_idx_x % coefficients.shape(d);
    block_idx_x /= coefficients.shape(d);
  }

  // Check if value to compute is out of range (where the thread block
  // spills over the end of the array.
  bool in_range = true;
  for (DIM d = 0; d < D; ++d) {
    in_range &= (coarse_index[d] < coefficients.shape(d));
  }
  return in_range;
}

// The functor for finding the coefficients in the "X" direction.
// That is, in the dimension where adjacent values are adjacent in
// memory.
template <DIM D, typename T, SIZE BlockSize, OPTION OP, typename DeviceType>
class SingleDimensionCoefficientFunctorX;

template <DIM D, typename T, SIZE BlockSize, typename DeviceType>
class SingleDimensionCoefficientFunctorX<D, T, BlockSize, DECOMPOSE, DeviceType>
    : public Functor<DeviceType> {
  // functor parameters
  static constexpr DIM current_dim = D - 1;
  SubArray<1, T, DeviceType> ratios;
  SubArray<D, T, DeviceType> fine_values;
  SubArray<D, T, DeviceType> coarse_values;
  SubArray<D, T, DeviceType> coefficients;

  // thread local variables
  T *fine_values_sm;
  bool in_range;
  SIZE coarse_index[D];

public:
  SingleDimensionCoefficientFunctorX() = default;
  MGARDX_CONT
  SingleDimensionCoefficientFunctorX(DIM current_dim_,
                                     SubArray<1, T, DeviceType> ratios_,
                                     SubArray<D, T, DeviceType> fine_values_,
                                     SubArray<D, T, DeviceType> coarse_values_,
                                     SubArray<D, T, DeviceType> coefficients_)
      : ratios(ratios_), fine_values(fine_values_),
        coarse_values(coarse_values_), coefficients(coefficients_) {
    assert(current_dim_ == (D - 1));
  }

  MGARDX_CONT size_t shared_memory_size() {
    return ((BlockSize * 2) + 2) * sizeof(T);
  }

  MGARDX_EXEC void Operation1() {
    in_range = SingleDimensionCoefficientComputeCoarseIndex<BlockSize>(
        *this, coefficients, coarse_index);

    const THREAD_IDX thread_idx = this->GetThreadIdX();

    // Even if this thread is not in range, it might need to participate
    // in loading data.
    fine_values_sm = reinterpret_cast<T *>(this->GetSharedMemory());

    // We need all threads to participate in loading data along the
    // current_dim (i.e., D - 1). Figure out the start and end of the
    // block that the thread block has to collectively load.
    SIZE load_idx_start = (coarse_index[current_dim] - thread_idx) * 2;
    SIZE load_idx_end = load_idx_start + BlockSize * 2 + 1;
    if (load_idx_end > fine_values.shape(current_dim)) {
      // Do not read past end of array
      load_idx_end = fine_values.shape(current_dim);
    } else if (load_idx_end == fine_values.shape(current_dim) - 1) {
      // If we are one value array from the end of the array, then the
      // array must be even and we need to load that last value because
      // it will not be shared with another thread block.
      ++load_idx_end;
    } else {
      // No further modifications to load_idx_end needed.
    }

    SIZE fine_index[D];
    for (DIM d = 0; d < current_dim; ++d) {
      fine_index[d] = coarse_index[d];
    }
    // For loading data, we want fine_index[current_dim] (i.e., D - 1) to
    // point to i-th value in the block, not the left value we are
    // operating on.
    fine_index[current_dim] = load_idx_start + thread_idx;

    SIZE sm_index = thread_idx;
    while (fine_index[current_dim] < load_idx_end) {
      fine_values_sm[sm_index] = fine_values[fine_index];
      sm_index += BlockSize;
      fine_index[current_dim] += BlockSize;
    }
  }

  MGARDX_EXEC void Operation2() {
    if (in_range) {
      const THREAD_IDX thread_idx = this->GetThreadIdX();

      T ratio = *ratios(2 * coarse_index[current_dim]);
      T left_value = fine_values_sm[2 * thread_idx + 0];
      T middle_value = fine_values_sm[2 * thread_idx + 1];
      T right_value = fine_values_sm[2 * thread_idx + 2];

      coefficients[coarse_index] =
          middle_value - lerp(left_value, right_value, ratio);
      coarse_values[coarse_index] = left_value;
      // If this is the last coefficient in the current_dim, need to also
      // write out the right coarse. Normally, the right value is written by
      // an overlapping thread, but there is no overlap at the right edge.
      if (coarse_index[current_dim] == coefficients.shape(current_dim) - 1) {
        ++coarse_index[current_dim];
        coarse_values[coarse_index] = right_value;
        // If the size of the array is even, there is an extra value that
        // cannot be used for interpolation. We need to copy that, too.
        if ((fine_values.shape(current_dim) % 2) == 0) {
          ++coarse_index[current_dim];
          coarse_values[coarse_index] = fine_values_sm[2 * thread_idx + 3];
        }
      }
    }
  }
};

template <DIM D, typename T, SIZE BlockSize, typename DeviceType>
class SingleDimensionCoefficientFunctorX<D, T, BlockSize, RECOMPOSE, DeviceType>
    : public Functor<DeviceType> {
  // functor parameters
  static constexpr DIM current_dim = D - 1;
  SubArray<1, T, DeviceType> ratios;
  SubArray<D, T, DeviceType> fine_values;
  SubArray<D, T, DeviceType> coarse_values;
  SubArray<D, T, DeviceType> coefficients;

  // thread local variables
  T *fine_values_sm;
  bool in_range;
  SIZE coarse_index[D];

public:
  SingleDimensionCoefficientFunctorX() = default;
  MGARDX_CONT
  SingleDimensionCoefficientFunctorX(DIM current_dim_,
                                     SubArray<1, T, DeviceType> ratios_,
                                     SubArray<D, T, DeviceType> fine_values_,
                                     SubArray<D, T, DeviceType> coarse_values_,
                                     SubArray<D, T, DeviceType> coefficients_)
      : ratios(ratios_), fine_values(fine_values_),
        coarse_values(coarse_values_), coefficients(coefficients_) {
    assert(current_dim_ == (D - 1));
  }

  MGARDX_CONT size_t shared_memory_size() {
    return ((BlockSize * 2) + 2) * sizeof(T);
  }

  MGARDX_EXEC void Operation1() {
    in_range = SingleDimensionCoefficientComputeCoarseIndex<BlockSize>(
        *this, coefficients, coarse_index);

    // Even if this thread is not in range, it might need to participate
    // in storing data.
    fine_values_sm = reinterpret_cast<T *>(this->GetSharedMemory());

    if (in_range) {
      const THREAD_IDX thread_idx = this->GetThreadIdX();
      const SIZE save_coarse_index = coarse_index[current_dim];

      T ratio = *ratios(coarse_index[current_dim] * 2);
      T coeff = coefficients[coarse_index];
      T left_value = coarse_values[coarse_index];
      ++coarse_index[current_dim];
      T right_value = coarse_values[coarse_index];

      // Write left value
      fine_values_sm[thread_idx * 2 + 0] = left_value;

      // Write middle value
      fine_values_sm[thread_idx * 2 + 1] =
          coeff + lerp(left_value, right_value, ratio);

      // If this is the last coefficient in the current_dim, need to also
      // write out the right fine value. Normally, the right value is written
      // by an overlapping thread, but there is no overlap at the right edge.
      // (Remember that we added one to coarse_index[current_dim].)
      if (coarse_index[current_dim] == coefficients.shape(current_dim)) {
        fine_values_sm[thread_idx * 2 + 2] = right_value;
        // If the size of the array is even, there is an extra value that
        // cannot be used for interpolation. We need to copy that, too.
        if ((fine_values.shape(current_dim) % 2) == 0) {
          ++coarse_index[current_dim];
          fine_values_sm[thread_idx * 2 + 3] = coarse_values[coarse_index];
        }
      }

      coarse_index[current_dim] = save_coarse_index;
    }
  }

  MGARDX_EXEC void Operation2() {
    const THREAD_IDX thread_idx = this->GetThreadIdX();

    // We need all threads to participate in storing data along the
    // current_dim (i.e., D - 1). Figure out the start and end of the
    // block that the thread block has to collectively store.
    SIZE store_idx_start = (coarse_index[current_dim] - thread_idx) * 2;
    SIZE store_idx_end = store_idx_start + BlockSize * 2 + 1;
    if (store_idx_end > fine_values.shape(current_dim)) {
      // Do not write past end of array
      store_idx_end = fine_values.shape(current_dim);
    } else if (store_idx_end == fine_values.shape(current_dim) - 1) {
      // If we are one value array from the end of the array, then the
      // array must be even and we need to store that last value because
      // it will not be shared with another thread block.
      ++store_idx_end;
    } else {
      // No further modifications to store_idx_end needed.
    }

    SIZE fine_index[D];
    for (DIM d = 0; d < current_dim; ++d) {
      fine_index[d] = coarse_index[d];
    }
    // For storing data, we want fine_index[current_dim] (i.e., D - 1) to
    // point to i-th value in the block, not the left value we are
    // operating on.
    fine_index[current_dim] = store_idx_start + thread_idx;

    SIZE sm_index = thread_idx;
    while (fine_index[current_dim] < store_idx_end) {
      fine_values[fine_index] = fine_values_sm[sm_index];
      sm_index += BlockSize;
      fine_index[current_dim] += BlockSize;
    }
  }
};

// The functor for finding the coefficients any direction other
// than the "X" direction. That is, adjacent values in the dimension
// we are decomposing are not adjacent to each other.
template <DIM D, typename T, SIZE BlockSize, OPTION OP, typename DeviceType>
class SingleDimensionCoefficientFunctorD : public Functor<DeviceType> {
  // functor parameters
  DIM current_dim;
  SubArray<1, T, DeviceType> ratios;
  SubArray<D, T, DeviceType> fine_values;
  SubArray<D, T, DeviceType> coarse_values;
  SubArray<D, T, DeviceType> coefficients;

public:
  SingleDimensionCoefficientFunctorD() = default;
  MGARDX_CONT
  SingleDimensionCoefficientFunctorD(DIM current_dim_,
                                     SubArray<1, T, DeviceType> ratios_,
                                     SubArray<D, T, DeviceType> fine_values_,
                                     SubArray<D, T, DeviceType> coarse_values_,
                                     SubArray<D, T, DeviceType> coefficients_)
      : current_dim(current_dim_), ratios(ratios_), fine_values(fine_values_),
        coarse_values(coarse_values_), coefficients(coefficients_) {}

  MGARDX_CONT size_t shared_memory_size() { return 0; }

  MGARDX_EXEC void Operation1() {
    SIZE coarse_index[D];
    bool in_range = SingleDimensionCoefficientComputeCoarseIndex<BlockSize>(
        *this, coefficients, coarse_index);

    if (in_range) {
      SIZE fine_index[D];
      for (DIM d = 0; d < D; ++d) {
        fine_index[d] = coarse_index[d];
      }
      fine_index[current_dim] *= 2;

      this->decompose_or_recompose(std::integral_constant<OPTION, OP>{},
                                   fine_index, coarse_index);
    }
  }

private:
  MGARDX_EXEC void
  decompose_or_recompose(std::integral_constant<OPTION, DECOMPOSE>,
                         SIZE fine_index[D], SIZE coarse_index[D]) {
    this->decompose(fine_index, coarse_index);
  }
  MGARDX_EXEC void
  decompose_or_recompose(std::integral_constant<OPTION, RECOMPOSE>,
                         SIZE fine_index[D], SIZE coarse_index[D]) {
    this->recompose(fine_index, coarse_index);
  }

  MGARDX_EXEC void decompose(SIZE fine_index[D], SIZE coarse_index[D]) {
    T ratio = *ratios(fine_index[current_dim]);
    T left_value = fine_values[fine_index];
    ++fine_index[current_dim];
    T middle_value = fine_values[fine_index];
    ++fine_index[current_dim];
    T right_value = fine_values[fine_index];

    coefficients[coarse_index] =
        middle_value - lerp(left_value, right_value, ratio);
    coarse_values[coarse_index] = left_value;
    // If this is the last coefficient in the current_dim, need to also
    // write out the right coarse. Normally, the right value is written by
    // an overlapping thread, but there is no overlap at the right edge.
    if (coarse_index[current_dim] == coefficients.shape(current_dim) - 1) {
      ++coarse_index[current_dim];
      coarse_values[coarse_index] = right_value;
      // If the size of the array is even, there is an extra value that
      // cannot be used for interpolation. We need to copy that, too.
      if ((fine_values.shape(current_dim) % 2) == 0) {
        ++fine_index[current_dim];
        ++coarse_index[current_dim];
        coarse_values[coarse_index] = fine_values[fine_index];
      }
    }
  }

  MGARDX_EXEC void recompose(SIZE fine_index[D], SIZE coarse_index[D]) {
    T ratio = *ratios(fine_index[current_dim]);
    T coeff = coefficients[coarse_index];
    T left_value = coarse_values[coarse_index];
    ++coarse_index[current_dim];
    T right_value = coarse_values[coarse_index];

    // Write left value
    fine_values[fine_index] = left_value;
    ++fine_index[current_dim];

    // Write middle value
    fine_values[fine_index] = coeff + lerp(left_value, right_value, ratio);

    // If this is the last coefficient in the current_dim, need to also
    // write out the right fine value. Normally, the right value is written
    // by an overlapping thread, but there is no overlap at the right edge.
    // (Remember that we added one to coarse_index[current_dim].)
    if (coarse_index[current_dim] == coefficients.shape(current_dim)) {
      ++fine_index[current_dim];
      fine_values[fine_index] = right_value;
      // If the size of the array is even, there is an extra value that
      // cannot be used for interpolation. We need to copy that, too.
      if ((fine_values.shape(current_dim) % 2) == 0) {
        ++fine_index[current_dim];
        ++coarse_index[current_dim];
        fine_values[fine_index] = coarse_values[coarse_index];
      }
    }
  }
};

// TODO: DELETE THIS CLASS
template <DIM D, typename T, SIZE R, SIZE C, SIZE F, OPTION OP,
          typename DeviceType>
class SingleDimensionCoefficientFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT SingleDimensionCoefficientFunctor() {}
  MGARDX_CONT SingleDimensionCoefficientFunctor(
      DIM current_dim, SubArray<1, T, DeviceType> ratio,
      SubArray<D, T, DeviceType> v, SubArray<D, T, DeviceType> coarse,
      SubArray<D, T, DeviceType> coeff)
      : current_dim(current_dim), ratio(ratio), v(v), coarse(coarse),
        coeff(coeff) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    SIZE v_left_idx[D];
    SIZE v_middle_idx[D];
    SIZE v_right_idx[D];
    SIZE coeff_idx[D];
    SIZE corase_idx[D];

    SIZE firstD = div_roundup(coeff.shape(D - 1), F);

    SIZE bidx = FunctorBase<DeviceType>::GetBlockIdX();
    coeff_idx[D - 1] =
        (bidx % firstD) * F + FunctorBase<DeviceType>::GetThreadIdX();

    bidx /= firstD;
    if (D >= 2)
      coeff_idx[D - 2] = FunctorBase<DeviceType>::GetBlockIdY() *
                             FunctorBase<DeviceType>::GetBlockDimY() +
                         FunctorBase<DeviceType>::GetThreadIdY();
    if (D >= 3)
      coeff_idx[D - 3] = FunctorBase<DeviceType>::GetBlockIdZ() *
                             FunctorBase<DeviceType>::GetBlockDimZ() +
                         FunctorBase<DeviceType>::GetThreadIdZ();

    for (int d = D - 4; d >= 0; d--) {
      coeff_idx[d] = bidx % coeff.shape(d);
      bidx /= coeff.shape(d);
    }

    bool in_range = true;
    for (int d = D - 1; d >= 0; d--) {
      if (coeff_idx[d] >= coeff.shape(d))
        in_range = false;
    }

    if (in_range) {
      for (int d = D - 1; d >= 0; d--) {
        if (d != current_dim) {
          v_left_idx[d] = coeff_idx[d];
          v_middle_idx[d] = coeff_idx[d];
          v_right_idx[d] = coeff_idx[d];
          corase_idx[d] = coeff_idx[d];
        } else {
          v_left_idx[d] = coeff_idx[d] * 2;
          v_middle_idx[d] = coeff_idx[d] * 2 + 1;
          v_right_idx[d] = coeff_idx[d] * 2 + 2;
          corase_idx[d] = coeff_idx[d];
        }
      }

      if (OP == DECOMPOSE) {
        coeff[coeff_idx] =
            v[v_middle_idx] - lerp(v[v_left_idx], v[v_right_idx],
                                   *ratio(v_left_idx[current_dim]));
        // if (coeff_idx[current_dim] == 1) {
        //   printf("left: %f, right: %f, middle: %f, ratio: %f, coeff: %f\n",
        //         *v(v_left_idx), *v(v_right_idx), *v(v_middle_idx),
        //         *ratio(v_left_idx[current_dim]), *coeff(coeff_idx));
        // }
        coarse[corase_idx] = v[v_left_idx];
        if (coeff_idx[current_dim] == coeff.shape(current_dim) - 1) {
          corase_idx[current_dim]++;
          coarse[corase_idx] = v[v_right_idx];
          if (v.shape(current_dim) % 2 == 0) {
            v_right_idx[current_dim]++;
            corase_idx[current_dim]++;
            coarse[corase_idx] = v[v_right_idx];
          }
        }
      } else if (OP == RECOMPOSE) {
        T left = coarse[corase_idx];
        corase_idx[current_dim]++;
        T right = coarse[corase_idx];
        corase_idx[current_dim]--;

        v[v_left_idx] = left;
        if (coeff_idx[current_dim] == coeff.shape(current_dim) - 1) {
          corase_idx[current_dim]++;
          v[v_right_idx] = right;
          if (v.shape(current_dim) % 2 == 0) {
            v_right_idx[current_dim]++;
            corase_idx[current_dim]++;
            v[v_right_idx] = coarse[corase_idx];
            v_right_idx[current_dim]--;
            corase_idx[current_dim]--;
          }
          corase_idx[current_dim]--;
        }

        v[v_middle_idx] = coeff[coeff_idx] +
                          lerp(left, right, *ratio(v_left_idx[current_dim]));
        // if (coeff_idx[current_dim] == 1) {
        // printf("left: %f, right: %f, middle: %f (%f), ratio: %f, coeff:
        // %f\n",
        //       *v(v_left_idx), *v(v_right_idx), *v(v_middle_idx),
        //       *coeff(coeff_idx) + lerp(*v(v_left_idx), *v(v_right_idx),
        //       *ratio(v_left_idx[current_dim])),
        //       *ratio(v_left_idx[current_dim]), *coeff(coeff_idx));
        // }
      }
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  // functor parameters
  DIM current_dim;
  SubArray<1, T, DeviceType> ratio;
  SubArray<D, T, DeviceType> v;
  SubArray<D, T, DeviceType> coarse;
  SubArray<D, T, DeviceType> coeff;
};

template <DIM D, typename T, OPTION OP, typename DeviceType>
class SingleDimensionCoefficient : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  SingleDimensionCoefficient() : AutoTuner<DeviceType>() {}

  template <SIZE BlockSize, typename FunctorType>
  MGARDX_CONT Task<FunctorType>
  GenTask(DIM current_dim, SubArray<1, T, DeviceType> ratio,
          SubArray<D, T, DeviceType> v, SubArray<D, T, DeviceType> coarse,
          SubArray<D, T, DeviceType> coeff, int queue_idx) {

    FunctorType functor(current_dim, ratio, v, coarse, coeff);

    SIZE nr = 1, nc = 1, nf = 1;
    if (D >= 3)
      nr = coeff.shape(D - 3);
    if (D >= 2)
      nc = coeff.shape(D - 2);
    nf = coeff.shape(D - 1);

    SIZE total_thread_z = (D >= 3) ? coeff.shape(D - 3) : 1;
    SIZE total_thread_y = (D >= 2) ? coeff.shape(D - 2) : 1;
    SIZE total_thread_x = coeff.shape(D - 1);

    size_t sm_size = functor.shared_memory_size();

    SIZE tbz = 1;
    SIZE tby = 1;
    SIZE tbx = BlockSize;
    SIZE gridz = ceil((float)total_thread_z / tbz);
    SIZE gridy = ceil((float)total_thread_y / tby);
    SIZE gridx = ceil((float)total_thread_x / tbx);

    for (DIM d = 3; d < D; d++) {
      gridx *= coeff.shape(D - (d + 1));
    }

    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                "SingleDimensionCoefficient");
  }

  MGARDX_CONT
  void Execute(DIM current_dim, SubArray<1, T, DeviceType> ratio,
               SubArray<D, T, DeviceType> v, SubArray<D, T, DeviceType> coarse,
               SubArray<D, T, DeviceType> coeff, int queue_idx) {
    int range_l = std::min(6, (int)std::log2(coeff.shape(D - 1)));
    int prec = TypeToIdx<T>();
    int config =
        AutoTuner<DeviceType>::autoTuningTable.gpk_reo_nd[prec][range_l];
    double min_time = std::numeric_limits<double>::max();
    int min_config = 0;
    ExecutionReturn ret;

#define GPK(CONFIG, FunctorTemplate)                                           \
  if (config == CONFIG || AutoTuner<DeviceType>::ProfileKernels) {             \
    const int BlockSize = GPK_CONFIG[0][CONFIG][2];                            \
    using FunctorType = FunctorTemplate<D, T, BlockSize, OP, DeviceType>;      \
    using TaskType = Task<FunctorType>;                                        \
    TaskType task = GenTask<BlockSize, FunctorType>(current_dim, ratio, v,     \
                                                    coarse, coeff, queue_idx); \
    DeviceAdapter<TaskType, DeviceType> adapter;                               \
    ret = adapter.Execute(task);                                               \
    if (AutoTuner<DeviceType>::ProfileKernels) {                               \
      if (ret.success && min_time > ret.execution_time) {                      \
        min_time = ret.execution_time;                                         \
        min_config = CONFIG;                                                   \
      }                                                                        \
    }                                                                          \
  }

    if (current_dim == (D - 1)) {
      GPK(6, SingleDimensionCoefficientFunctorX) if (!ret.success) config--;
      GPK(5, SingleDimensionCoefficientFunctorX) if (!ret.success) config--;
      GPK(4, SingleDimensionCoefficientFunctorX) if (!ret.success) config--;
      GPK(3, SingleDimensionCoefficientFunctorX) if (!ret.success) config--;
      GPK(2, SingleDimensionCoefficientFunctorX) if (!ret.success) config--;
      GPK(1, SingleDimensionCoefficientFunctorX) if (!ret.success) config--;
      GPK(0, SingleDimensionCoefficientFunctorX) if (!ret.success) config--;
    } else {
      GPK(6, SingleDimensionCoefficientFunctorD) if (!ret.success) config--;
      GPK(5, SingleDimensionCoefficientFunctorD) if (!ret.success) config--;
      GPK(4, SingleDimensionCoefficientFunctorD) if (!ret.success) config--;
      GPK(3, SingleDimensionCoefficientFunctorD) if (!ret.success) config--;
      GPK(2, SingleDimensionCoefficientFunctorD) if (!ret.success) config--;
      GPK(1, SingleDimensionCoefficientFunctorD) if (!ret.success) config--;
      GPK(0, SingleDimensionCoefficientFunctorD) if (!ret.success) config--;
    }
    if (config < 0 && !ret.success) {
      std::cout << log::log_err
                << "no suitable config for SingleDimensionCoefficient.\n";
      exit(-1);
    }
#undef GPK

    if (AutoTuner<DeviceType>::ProfileKernels) {
      FillAutoTunerTable<DeviceType>("SingleDimensionCoefficient", prec,
                                     range_l, min_config);
    }
  }
};

} // namespace mgard_x

#endif
