#include <cassert>

namespace mgard {

template <std::size_t N, typename Real>
ConstituentRestriction<N, Real>::ConstituentRestriction(
    const TensorMeshHierarchy<N, Real> &hierarchy, const std::size_t l,
    const std::size_t dimension)
    : ConstituentLinearOperator<N, Real>(hierarchy, l, dimension),
      coarse_indices(hierarchy.indices(l - 1, dimension)) {
  // This is almost certainly superfluous, since `hierarchy.indices` checks that
  // `l - 1` is a valid mesh index.
  if (!l) {
    throw std::invalid_argument("cannot restrict from the coarsest level");
  }
  // We'll dereference `coarse_indices.begin()` and `indices.end()`.
  if (coarse_indices.empty() || CLO::indices.empty()) {
    throw std::invalid_argument("dimension must be nonzero");
  }
}

template <std::size_t N, typename Real>
void ConstituentRestriction<N, Real>::do_operator_parentheses(
    const std::array<std::size_t, N> multiindex, Real *const v) const {
  const std::vector<Real> &xs = CLO::hierarchy->coordinates.at(CLO::dimension_);

  // `x_left` and `out_left` is declared and defined inside the loop.
  Real x_right;
  Real *out_right;

  std::array<std::size_t, N> alpha = multiindex;
  std::size_t &variable_index = alpha.at(CLO::dimension_);

  std::vector<std::size_t>::const_iterator p = coarse_indices.begin();
  std::size_t i;

  variable_index = i = *p++;
  x_right = xs.at(i);
  out_right = &CLO::hierarchy->at(v, alpha);

  std::array<std::size_t, N> ALPHA = multiindex;
  std::size_t &VARIABLE_INDEX = ALPHA.at(CLO::dimension_);

  std::vector<std::size_t>::const_iterator P = CLO::indices.begin();
  std::size_t I;

  VARIABLE_INDEX = I = *P++;

  const std::vector<std::size_t>::const_iterator p_end = coarse_indices.end();
  while (p != p_end) {
    assert(I == i);

    const Real x_left = x_right;
    Real *const out_left = out_right;

    variable_index = i = *p++;
    x_right = xs.at(i);
    out_right = &CLO::hierarchy->at(v, alpha);

    const Real width_reciprocal = 1 / (x_right - x_left);

    while ((VARIABLE_INDEX = I = *P++) != i) {
      const Real x_middle = xs.at(I);
      assert(x_left < x_middle && x_middle < x_right);
      const Real v_middle = CLO::hierarchy->at(v, ALPHA);
      *out_left += v_middle * (x_right - x_middle) * width_reciprocal;
      *out_right += v_middle * (x_middle - x_left) * width_reciprocal;
    }
  }
}

} // namespace mgard