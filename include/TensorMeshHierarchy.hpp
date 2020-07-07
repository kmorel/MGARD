#ifndef TENSORMESHHIERARCHY_HPP
#define TENSORMESHHIERARCHY_HPP
//!\file
//!\brief Increasing hierarchy of tensor meshes.

#include <cstddef>

#include <array>
#include <type_traits>
#include <vector>

#include "TensorMeshLevel.hpp"
#include "utilities.hpp"

namespace mgard {

//! Forward declaration.
template <std::size_t N, typename T> class TensorLevelValues;

//! Hierarchy of meshes produced by subsampling an initial mesh.
template <std::size_t N, typename Real> class TensorMeshHierarchy {
public:
  //! Constructor.
  //!
  //!\param mesh Initial, finest mesh to sit atop the hierarchy.
  TensorMeshHierarchy(const TensorMeshLevel<N, Real> &mesh);

  //! Constructor.
  //!
  //!\param mesh Initial, finest mesh to sit atop the hierarchy.
  //!\param coordinates Coordinates of the nodes in the finest mesh.
  TensorMeshHierarchy(const TensorMeshLevel<N, Real> &mesh,
                      const std::array<std::vector<Real>, N> &coordinates);

  // TODO: We may want to remove these. Using it refactoring.
  // TODO: Instead, we may want to remove the previous constructors. Check
  // whether `TensorMeshLevel` is needed anywhere.

  //! Constructor.
  //!
  //!\param shape Shape of the initial, finest mesh to sit atop the hiearachy.
  TensorMeshHierarchy(const std::array<std::size_t, N> &shape);

  //! Constructor.
  //!
  //!\param shape Shape of the initial, finest mesh to sit atop the hiearachy.
  //!\param coordinates Coordinates of the nodes in the finest mesh.
  TensorMeshHierarchy(const std::array<std::size_t, N> &shape,
                      const std::array<std::vector<Real>, N> &coordinates);

  //! Report the number of degrees of freedom in the finest TensorMeshLevel.
  std::size_t ndof() const;

  //! Report the number of degrees of freedom in a TensorMeshLevel.
  //!
  //!\param l Index of the TensorMeshLevel.
  std::size_t ndof(const std::size_t l) const;

  //! Calculate the stride between entries in a 1D slice on some level.
  //!
  //!\param l Index of the TensorMeshLevel.
  //!\param dimension Index of the dimension.
  std::size_t stride(const std::size_t l, const std::size_t dimension) const;

  // TODO: Temporary member function to be removed once indices rather than
  // index differences are used everywhere.
  std::size_t l(const std::size_t index_difference) const;

  //! Generate the indices (in a particular dimension) of a mesh level.
  //!
  //!\param l Mesh index.
  //!\param dimension Dimension index.
  std::vector<std::size_t> indices(const std::size_t l,
                                   const std::size_t dimension) const;

  //! Compute the offset of the value associated to a node.
  //!
  //! The offset is the distance in a contiguous dataset defined on the finest
  //! mesh in the hierarchy from the value associated to the zeroth node to
  //! the value associated to the given node.
  //!
  //!\param multiindex Multiindex of the node.
  std::size_t offset(const std::array<std::size_t, N> multiindex) const;

  //! Access the value associated to a particular node.
  //!
  //!\param v Dataset defined on the hierarchy.
  //!\param multiindex Multiindex of the node.
  Real &at(Real *const v, const std::array<std::size_t, N> multiindex) const;

  //! Access the subset of a dataset associated to the nodes of a level.
  //!
  //!\param coefficients Values associated to the nodes of the finest mesh
  //! level.
  //!\param l Index of the mesh level to be iterated over.
  template <typename T>
  TensorLevelValues<N, T> on_nodes(T *const coefficients,
                                   const std::size_t l) const;

  //! Meshes composing the hierarchy, in 'increasing' order.
  std::vector<TensorMeshLevel<N, Real>> meshes;

  //! Coordinates of the nodes in the finest mesh.
  std::array<std::vector<Real>, N> coordinates;

  //! Index of finest TensorMeshLevel.
  std::size_t L;

protected:
  //! Check that a mesh index is in bounds.
  //!
  //!\param l Mesh index.
  void check_mesh_index_bounds(const std::size_t l) const;

  //! Check that a pair of mesh indices are nondecreasing.
  //!
  //!\param l Smaller (nonlarger) mesh index.
  //!\param m Larger (nonsmaller) mesh index.
  void check_mesh_indices_nondecreasing(const std::size_t l,
                                        const std::size_t m) const;

  //! Check that a mesh index is nonzero.
  //!
  //!\param l Mesh index.
  void check_mesh_index_nonzero(const std::size_t l) const;

  //! Check that a dimension index is in bounds.
  //!
  //!\param dimension Dimension index.
  void check_dimension_index_bounds(const std::size_t dimension) const;
};

//! Equality comparison.
template <std::size_t N, typename Real>
bool operator==(const TensorMeshHierarchy<N, Real> &a,
                const TensorMeshHierarchy<N, Real> &b);

//! Inequality comparison.
template <std::size_t N, typename Real>
bool operator==(const TensorMeshHierarchy<N, Real> &a,
                const TensorMeshHierarchy<N, Real> &b);

//! View of values (with multiindices and coordinates) associated to the nodes
//! of a particular level in a mesh hierarchy.
//!
//! We template on `T` to allow `coefficients` to be a pointer to either `Real`
//! or `const Real`.
template <std::size_t N, typename T> class TensorLevelValues {

  //! Type of the node coordinates.
  using Real = typename std::remove_const<T>::type;

public:
  //! Constructor.
  //!
  //!\param hierarchy Associated mesh hierarchy.
  //!\param coefficients Values associated to the nodes of the finest mesh
  //! level.
  //!\param l Index of the mesh level to be iterated over.
  TensorLevelValues(const TensorMeshHierarchy<N, Real> &hierarchy,
                    T *const coefficients, const std::size_t l);

  //! Forward declaration.
  class iterator;

  //! Return an interator to the beginning of the values.
  iterator begin() const;

  //! Return an interator to the end of the values.
  iterator end() const;

  //! Equality comparison.
  bool operator==(const TensorLevelValues &other) const;

  //! Inequality comparison.
  bool operator!=(const TensorLevelValues &other) const;

  //! Associated mesh hierarchy.
  const TensorMeshHierarchy<N, Real> &hierarchy;

  //! Values associated to the nodes of the finest mesh level.
  T *const coefficients;

private:
  //! Index of the level being iterated over.
  //!
  //! This is only stored so we can avoid comparing `factors` and `product` in
  //! the (in)equality comparison operators.
  const std::size_t l;

  // The indices whose Cartesian product will give the multiindices of the nodes
  // on the level being iterated over.
  const std::array<std::vector<std::size_t>, N> factors;

  //! Multiindices of the nodes on the level being iterated over.
  const CartesianProduct<std::size_t, N> multiindices;
};

//! A value associated to a node, along with its multiindex and coordinates.
template <std::size_t N, typename T> struct SituatedCoefficient {
  //! Type of the node coordinates.
  using Real = typename std::remove_const<T>::type;

  //! Multiindex of the node.
  std::array<std::size_t, N> multiindex;

  //! Coordinates of the node.
  std::array<Real, N> coordinates;

  //! Pointer to the value at the node.
  T *value;
};

//! Iterator over the values associated to a mesh level in a structured mesh
//! hierarchy.
template <std::size_t N, typename T> class TensorLevelValues<N, T>::iterator {
public:
  using iterator_category = std::input_iterator_tag;
  using value_type = SituatedCoefficient<N, T>;
  using difference_type = std::ptrdiff_t;
  using pointer = value_type *;
  using reference = value_type &;

  //! Constructor.
  //!
  //!\param iterable View of nodal values to be iterated over.
  //!\param inner Underlying multiindex iterator.
  iterator(const TensorLevelValues &iterable,
           const typename CartesianProduct<std::size_t, N>::iterator &inner);

  //! Equality comparison.
  bool operator==(const iterator &other) const;

  //! Inequality comparison.
  bool operator!=(const iterator &other) const;

  //! Preincrement.
  iterator &operator++();

  //! Postincrement.
  iterator operator++(int);

  //! Dereference.
  value_type operator*() const;

  //! View of nodal values being iterated over
  const TensorLevelValues &iterable;

private:
  //! Underlying multiindex iterator.
  typename CartesianProduct<std::size_t, N>::iterator inner;
};

} // namespace mgard

#include "TensorMeshHierarchy.tpp"
#endif