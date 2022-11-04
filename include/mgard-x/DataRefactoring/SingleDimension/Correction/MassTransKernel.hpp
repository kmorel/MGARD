/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_SINGLE_DIMENSION_MASSTRANS_KERNEL_TEMPLATE
#define MGARD_X_SINGLE_DIMENSION_MASSTRANS_KERNEL_TEMPLATE

#include "../../../RuntimeX/RuntimeX.h"

#include "../../MultiDimension/Correction/LPKFunctor.h"

namespace mgard_x {

template <DIM D, typename T, SIZE BlockSize, typename DeviceType>
class SingleDimensionMassTransFunctorX : public Functor<DeviceType> {
  // functor parameters
  static constexpr DIM current_dim = D - 1;
  SubArray<1, T, DeviceType> dist;
  SubArray<1, T, DeviceType> ratio;
  SubArray<D, T, DeviceType> coefficients;
  SubArray<D, T, DeviceType> values;

  // thread-local variables
  bool in_range;
  SIZE value_index[D];
  T *coeff_sm;
  T *dist_sm;
  T *ratios_sm;

public:
  MGARDX_CONT SingleDimensionMassTransFunctorX() {}
  MGARDX_CONT SingleDimensionMassTransFunctorX(SubArray<1, T, DeviceType> dist,
                                               SubArray<1, T, DeviceType> ratio,
                                               SubArray<D, T, DeviceType> coeff,
                                               SubArray<D, T, DeviceType> v)
      : dist(dist), ratio(ratio), coefficients(coeff), values(v) {
    Functor<DeviceType>();
  }

  MGARDX_CONT size_t shared_memory_size() {
    return (BlockSize + 1       // coefficients
            + 2 * BlockSize + 2 // dist
            + 2 * BlockSize + 1 // ratios
            ) *
           sizeof(T);
  }

  // Finds the beginning of the block of data to be operated on.
  MGARDX_EXEC void FindIndex() {
    // For up to 3 dimensions, the array indices will follow the
    // block/thread indices. However, the scheduling can only happen
    // up to those 3 dimensions, so all further array directions are
    // appended to the first block scheduling dimension.
    SIZE first_dimension_size =
        div_roundup(coefficients.shape(D - 1), BlockSize);
    SIZE block_idx_x = this->GetBlockIdX();

    value_index[D - 1] = (block_idx_x % first_dimension_size) * BlockSize;
    block_idx_x /= first_dimension_size;

    if (D >= 2) {
      // The block size is always 1 in the y dimension.
      value_index[D - 2] = this->GetBlockIdY();
    }
    if (D >= 3) {
      // The block size is always 1 in the z dimension.
      value_index[D - 3] = this->GetBlockIdZ();
    }

    for (int d = D - 4; d >= 0; --d) {
      value_index[d] = block_idx_x % coefficients.shape(d);
      block_idx_x /= coefficients.shape(d);
    }
  }

  MGARDX_EXEC void Operation1() {
    FindIndex();

    coeff_sm = reinterpret_cast<T *>(this->GetSharedMemory());

    // We need all threads to participate in loading data along the
    // current_dim (i.e., D - 1). Figure out the start and end of the
    // block that the thread block has to collectively load.
    const THREAD_IDX thread_idx = this->GetThreadIdX();
    SIZE save_index = value_index[current_dim];
    if (thread_idx == 0) {
      if (save_index > 0) {
        value_index[current_dim] = save_index - 1;
        coeff_sm[0] = coefficients[value_index];
      } else {
        coeff_sm[0] = 0;
      }
    }
    value_index[current_dim] = save_index + thread_idx;
    if (value_index[current_dim] < coefficients.shape(current_dim)) {
      coeff_sm[thread_idx + 1] = coefficients[value_index];
    } else {
      coeff_sm[thread_idx + 1] = 0;
    }

    dist_sm = coeff_sm + BlockSize + 1;
    ratios_sm = dist_sm + BlockSize * 2 + 2;

    SIZE load_start_idx = save_index * 2 - 2;
    if (save_index < 1) {
      // At the boundary, cannot load overwrite to the left. (Special condition
      // for stupid unsigned integers.)
      if (thread_idx == 0) {
        dist_sm[0] = dist_sm[1] = 0;
        ratios_sm[0] = ratios_sm[1] = 0;
      }
      load_start_idx = 0;
    }
    SIZE load_end_idx = (save_index + BlockSize) * 2 + 1;

    SIZE sm_index = thread_idx;
    if (save_index < 1) {
      // Skip first 2 entries (see above)
      sm_index += 2;
    }
    SIZE load_idx = load_start_idx + thread_idx;
    while (load_idx < load_end_idx) {
      if (load_idx < dist.shape(0)) {
        dist_sm[sm_index] = *dist(load_idx);
      } else {
        dist_sm[sm_index] = 0;
      }
      sm_index += BlockSize;
      load_idx += BlockSize;
    }

    --load_end_idx;
    sm_index = thread_idx;
    if (save_index < 1) {
      // Skip first 2 entries (see above)
      sm_index += 2;
    }
    load_idx = load_start_idx + thread_idx;
    while (load_idx < load_end_idx) {
      if (load_idx < ratio.shape(0)) {
        ratios_sm[sm_index] = *ratio(load_idx);
      } else {
        ratios_sm[sm_index] = 0;
      }
      sm_index += BlockSize;
      load_idx += BlockSize;
    }

    // Done loading with value_index. Set it to the value this thread
    // will save.
    value_index[current_dim] = save_index + thread_idx;

    // Check if value to compute is out of range (where the thread block
    // spills over the end of the array.
    in_range = true;
    for (DIM d = 0; d < D; ++d) {
      in_range &= (value_index[d] < values.shape(d));
    }
  }

  MGARDX_EXEC void Operation2() {
    if (in_range) {
      SIZE coeff_index = this->GetThreadIdX() + 1;
      SIZE interp_index = this->GetThreadIdX() * 2 + 2;

      const T a = 0.0;
      const T b = coeff_sm[coeff_index - 1];
      const T c = 0.0;
      const T d = coeff_sm[coeff_index];
      const T e = 0.0;
      const T h1 = dist_sm[interp_index - 2];
      const T h2 = dist_sm[interp_index - 1];
      const T h3 = dist_sm[interp_index];
      const T h4 = dist_sm[interp_index + 1];
      const T r1 = ratios_sm[interp_index - 2];
      const T r2 = ratios_sm[interp_index - 1];
      const T r3 = ratios_sm[interp_index];
      const T r4 = 1 - r3;
      values[value_index] =
          mass_trans(a, b, c, d, e, h1, h2, h3, h4, r1, r2, r3, r4);
    }
  }
};

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, typename DeviceType>
class SingleDimensionMassTransFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT SingleDimensionMassTransFunctor() {}
  MGARDX_CONT SingleDimensionMassTransFunctor(DIM current_dim,
                                              SubArray<1, T, DeviceType> dist,
                                              SubArray<1, T, DeviceType> ratio,
                                              SubArray<D, T, DeviceType> coeff,
                                              SubArray<D, T, DeviceType> v)
      : current_dim(current_dim), dist(dist), ratio(ratio), coeff(coeff), v(v) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    SIZE v_idx[D];

    SIZE firstD = div_roundup(v.shape(D - 1), F);

    SIZE bidx = FunctorBase<DeviceType>::GetBlockIdX();
    v_idx[D - 1] =
        (bidx % firstD) * F + FunctorBase<DeviceType>::GetThreadIdX();

    bidx /= firstD;
    if (D >= 2)
      v_idx[D - 2] = FunctorBase<DeviceType>::GetBlockIdY() *
                         FunctorBase<DeviceType>::GetBlockDimY() +
                     FunctorBase<DeviceType>::GetThreadIdY();
    if (D >= 3)
      v_idx[D - 3] = FunctorBase<DeviceType>::GetBlockIdZ() *
                         FunctorBase<DeviceType>::GetBlockDimZ() +
                     FunctorBase<DeviceType>::GetThreadIdZ();

    for (int d = D - 4; d >= 0; d--) {
      v_idx[d] = bidx % v.shape(d);
      bidx /= v.shape(d);
    }

    bool in_range = true;
    for (int d = D - 1; d >= 0; d--) {
      if (v_idx[d] >= v.shape(d))
        in_range = false;
    }

    if (in_range) {

      const T a = 0.0;
      const T c = 0.0;
      const T e = 0.0;
      T b = 0.0;
      T d = 0.0;

      if (v_idx[current_dim] > 0 &&
          v_idx[current_dim] < coeff.shape(current_dim)) {
        v_idx[current_dim]--;
        b = coeff[v_idx];
        v_idx[current_dim]++;
      }

      if (v_idx[current_dim] < coeff.shape(current_dim)) {
        // v_idx[current_dim] ++;
        d = coeff[v_idx];
        // v_idx[current_dim] --;
      }

      T h1 = 0, h2 = 0, h3 = 0, h4 = 0;
      T r1 = 0, r2 = 0, r3 = 0, r4 = 0;

      if (v_idx[current_dim] > 0 &&
          v_idx[current_dim] * 2 <
              coeff.shape(current_dim) + v.shape(current_dim) - 1) {
        h1 = *dist(v_idx[current_dim] * 2 - 2);
        h2 = *dist(v_idx[current_dim] * 2 - 1);
        r1 = *ratio(v_idx[current_dim] * 2 - 2);
        r2 = *ratio(v_idx[current_dim] * 2 - 1);
      }
      if (v_idx[current_dim] * 2 <
          coeff.shape(current_dim) + v.shape(current_dim) - 1) {
        h3 = *dist(v_idx[current_dim] * 2);
        h4 = *dist(v_idx[current_dim] * 2 + 1);
        r3 = *ratio(v_idx[current_dim] * 2);
        r4 = 1 - r3;
      }

      // printf("v_idx = [%d %d] %f %f %f %f %f f_sm_h %f %f %f %f f_sm_r %f %f
      // %f %f, out: %f\n",
      //   v_idx[1], v_idx[0], a,b,c,d,e, h1,h2,h3,h4, r1,r2,r3,r4,
      //   mass_trans(a, b, c, d, e, h1, h2, h3, h4, r1, r2, r3, r4));

      v[v_idx] = mass_trans(a, b, c, d, e, h1, h2, h3, h4, r1, r2, r3, r4);
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  // functor parameters
  DIM current_dim;
  SubArray<1, T, DeviceType> dist;
  SubArray<1, T, DeviceType> ratio;
  SubArray<D, T, DeviceType> coeff;
  SubArray<D, T, DeviceType> v;
};

template <DIM D, typename T, typename DeviceType>
class SingleDimensionMassTrans : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  SingleDimensionMassTrans() : AutoTuner<DeviceType>() {}

  template <SIZE BlockSize>
  MGARDX_CONT
      Task<SingleDimensionMassTransFunctorX<D, T, BlockSize, DeviceType>>
      GenTaskX(SubArray<1, T, DeviceType> dist,
               SubArray<1, T, DeviceType> ratio,
               SubArray<D, T, DeviceType> coeff, SubArray<D, T, DeviceType> v,
               int queue_idx) {

    using FunctorType =
        SingleDimensionMassTransFunctorX<D, T, BlockSize, DeviceType>;
    FunctorType functor(dist, ratio, coeff, v);

    SIZE nr = 1, nc = 1, nf = 1;
    if (D >= 3)
      nr = v.shape(D - 3);
    if (D >= 2)
      nc = v.shape(D - 2);
    nf = v.shape(D - 1);

    SIZE total_thread_z = (D >= 3) ? v.shape(D - 3) : 0;
    SIZE total_thread_y = (D >= 2) ? v.shape(D - 2) : 0;
    SIZE total_thread_x = v.shape(D - 1);

    size_t sm_size = functor.shared_memory_size();

    SIZE gridz = total_thread_z;
    SIZE gridy = total_thread_y;
    SIZE gridx = ceil((float)total_thread_x / BlockSize);

    for (int d = D - 4; d >= 0; d--) {
      gridx *= coeff.shape(d);
    }

    return Task(functor, gridz, gridy, gridx, 1, 1, BlockSize, sm_size,
                queue_idx, "SingleDimensionMassTrans");
  }

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<SingleDimensionMassTransFunctor<D, T, R, C, F, DeviceType>>
  GenTask(DIM current_dim, SubArray<1, T, DeviceType> dist,
          SubArray<1, T, DeviceType> ratio, SubArray<D, T, DeviceType> coeff,
          SubArray<D, T, DeviceType> v, int queue_idx) {

    using FunctorType =
        SingleDimensionMassTransFunctor<D, T, R, C, F, DeviceType>;
    FunctorType functor(current_dim, dist, ratio, coeff, v);

    SIZE nr = 1, nc = 1, nf = 1;
    if (D >= 3)
      nr = v.shape(D - 3);
    if (D >= 2)
      nc = v.shape(D - 2);
    nf = v.shape(D - 1);

    SIZE total_thread_z = nr;
    SIZE total_thread_y = nc;
    SIZE total_thread_x = nf;

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();

    tbz = R;
    tby = C;
    tbx = F;
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);

    for (int d = D - 4; d >= 0; d--) {
      gridx *= coeff.shape(d);
    }

    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                "SingleDimensionMassTrans");
  }

  MGARDX_CONT
  void Execute(DIM current_dim, SubArray<1, T, DeviceType> dist,
               SubArray<1, T, DeviceType> ratio,
               SubArray<D, T, DeviceType> coeff, SubArray<D, T, DeviceType> v,
               int queue_idx) {
    if (current_dim == (D - 1)) {
      int range_l = std::min(6, (int)std::log2(v.shape(D - 1)) - 1);
      int prec = TypeToIdx<T>();
      int config =
          AutoTuner<DeviceType>::autoTuningTable.gpk_reo_nd[prec][range_l];
      double min_time = std::numeric_limits<double>::max();
      int min_config = 0;
      ExecutionReturn ret;

#define GPK(CONFIG)                                                            \
  if (config == CONFIG || AutoTuner<DeviceType>::ProfileKernels) {             \
    const int BlockSize = GPK_CONFIG[0][CONFIG][2];                            \
    using FunctorType =                                                        \
        SingleDimensionMassTransFunctorX<D, T, BlockSize, DeviceType>;         \
    using TaskType = Task<FunctorType>;                                        \
    TaskType task = GenTaskX<BlockSize>(dist, ratio, coeff, v, queue_idx);     \
    DeviceAdapter<TaskType, DeviceType> adapter;                               \
    ret = adapter.Execute(task);                                               \
    if (AutoTuner<DeviceType>::ProfileKernels) {                               \
      if (ret.success && min_time > ret.execution_time) {                      \
        min_time = ret.execution_time;                                         \
        min_config = CONFIG;                                                   \
      }                                                                        \
    }                                                                          \
  }

      GPK(6) if (!ret.success) config--;
      GPK(5) if (!ret.success) config--;
      GPK(4) if (!ret.success) config--;
      GPK(3) if (!ret.success) config--;
      GPK(2) if (!ret.success) config--;
      GPK(1) if (!ret.success) config--;
      GPK(0) if (!ret.success) config--;
      if (config < 0 && !ret.success) {
        std::cout << log::log_err
                  << "no suitable config for SingleDimensionMassTrans.\n";
        exit(-1);
      }
#undef GPK

      if (AutoTuner<DeviceType>::ProfileKernels) {
        FillAutoTunerTable<DeviceType>("SingleDimensionMassTrans", prec,
                                       range_l, min_config);
      }
    } else {
      // Should this be v.shape(current_dim)?
      int range_l = std::min(6, (int)std::log2(v.shape(D - 1)) - 1);
      int prec = TypeToIdx<T>();
      int config =
          AutoTuner<DeviceType>::autoTuningTable.gpk_reo_nd[prec][range_l];
      double min_time = std::numeric_limits<double>::max();
      int min_config = 0;
      ExecutionReturn ret;

#define GPK(CONFIG)                                                            \
  if (config == CONFIG || AutoTuner<DeviceType>::ProfileKernels) {             \
    const int R = GPK_CONFIG[D - 1][CONFIG][0];                                \
    const int C = GPK_CONFIG[D - 1][CONFIG][1];                                \
    const int F = GPK_CONFIG[D - 1][CONFIG][2];                                \
    using FunctorType =                                                        \
        SingleDimensionMassTransFunctor<D, T, R, C, F, DeviceType>;            \
    using TaskType = Task<FunctorType>;                                        \
    TaskType task =                                                            \
        GenTask<R, C, F>(current_dim, dist, ratio, coeff, v, queue_idx);       \
    DeviceAdapter<TaskType, DeviceType> adapter;                               \
    ret = adapter.Execute(task);                                               \
    if (AutoTuner<DeviceType>::ProfileKernels) {                               \
      if (ret.success && min_time > ret.execution_time) {                      \
        min_time = ret.execution_time;                                         \
        min_config = CONFIG;                                                   \
      }                                                                        \
    }                                                                          \
  }

      GPK(6) if (!ret.success) config--;
      GPK(5) if (!ret.success) config--;
      GPK(4) if (!ret.success) config--;
      GPK(3) if (!ret.success) config--;
      GPK(2) if (!ret.success) config--;
      GPK(1) if (!ret.success) config--;
      GPK(0) if (!ret.success) config--;
      if (config < 0 && !ret.success) {
        std::cout << log::log_err
                  << "no suitable config for SingleDimensionMassTrans.\n";
        exit(-1);
      }
#undef GPK

      if (AutoTuner<DeviceType>::ProfileKernels) {
        FillAutoTunerTable<DeviceType>("SingleDimensionMassTrans", prec,
                                       range_l, min_config);
      }
    }
  }
};

} // namespace mgard_x

#endif
