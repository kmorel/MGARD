#ifndef MGARD_MESH_HPP
#define MGARD_MESH_HPP

#include <cstddef>

#include <array>
#include <iterator>

#include "utilities.hpp"

namespace mgard {

//! Forward declaration.
template <std::size_t N, typename Real> class LevelValuesIterator;

//! Dimensions for a tensor product mesh on a Cartesian product of intervals.
template <std::size_t N> struct Dimensions2kPlus1 {
  //! Constructor.
  //!
  //!\param input_ Numbers of nodes in each dimension.
  Dimensions2kPlus1(const std::array<std::size_t, N> input_);

  //! Mesh dimensions (number of nodes in each dimension) originally input.
  std::array<std::size_t, N> input;

  //! Mesh dimensions (number of nodes in each dimension) rounded to a power of
  //! two plus one.
  std::array<std::size_t, N> rnded;

  //! Overall largest index in the mesh hierarchy. The largest index is one less
  //! than the number of levels. Any dimension of size `1` is ignored.
  std::size_t nlevel;

  //! Determine whether all dimensions are either equal to `1` or of the form
  //! `2^k + 1`.
  bool is_2kplus1() const;

  //! Access the subset of a dataset associated to the nodes of a level.
  //!
  //!\param coefficients Values associated to nodes.
  //!\param index_difference Difference between the index of the finest mesh
  //! level and the index of the mesh level to be iterated over.
  template <typename Real>
  RangeSlice<LevelValuesIterator<N, Real>>
  on_nodes(Real *const coefficients, const std::size_t index_difference) const;
};

//! Equality comparison.
template <std::size_t N>
bool operator==(const Dimensions2kPlus1<N> &a, const Dimensions2kPlus1<N> &b);

//! Inequality comparison.
template <std::size_t N>
bool operator!=(const Dimensions2kPlus1<N> &a, const Dimensions2kPlus1<N> &b);

// As of this writing, these are only needed in the implementation of the
// `Dimensions2kPlus1` constructor.
std::size_t nlevel_from_size(const std::size_t n);

std::size_t size_from_nlevel(const std::size_t n);

// These were originally `inline`.
int get_index(const int ncol, const int i, const int j);

int get_lindex(const int n, const int no, const int i);

int get_index3(const int ncol, const int nfib, const int i, const int j,
               const int k);

//! Compute the stride for a mesh level.
//!
//!\param index_difference Difference between the index of the finest mesh level
//! and the index of the mesh level in question.
std::size_t stride_from_index_difference(const std::size_t index_difference);

//! Iterator over the values associated to a mesh level in a structured mesh
//! hierarchy.
template <std::size_t N, typename Real> class LevelValuesIterator {
public:
  using iterator_category = std::input_iterator_tag;
  using value_type = Real &;
  using difference_type = std::ptrdiff_t;
  // I'm not sure that these are right, since `value_type` isn't `Real`.
  using pointer = Real *;
  using reference = Real &;

  //! Constructor.
  //!
  //!\param dimensions Dimensions of the finest mesh level.
  //!\param coefficients Dataset being iterated over.
  //!\param index_difference Difference between the index of the finest mesh
  //! and the index of the mesh level to be iterated over.
  //!\param indices Indices of the current node in the finest mesh level.
  LevelValuesIterator(const Dimensions2kPlus1<N> &dimensions,
                      Real *const coefficients,
                      const std::size_t index_difference,
                      const std::array<std::size_t, N> &indices);

  //! Equality comparison.
  bool operator==(const LevelValuesIterator &other) const;

  //! Inequality comparison.
  bool operator!=(const LevelValuesIterator &other) const;

  //! Preincrement.
  LevelValuesIterator &operator++();

  //! Postincrement.
  LevelValuesIterator operator++(int);

  //! Dereference.
  value_type operator*() const;

private:
  //! Dimensions of the finest mesh level.
  const Dimensions2kPlus1<N> &dimensions;

  //! Dataset being iterated over.
  Real *const coefficients;

  //! Stride for the mesh level being iterated over.
  const std::size_t stride;

  //! Indices of the current node in the finest mesh level.
  std::array<std::size_t, N> indices;
};

} // namespace mgard

#include "mgard_mesh.tpp"
#endif
