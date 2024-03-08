
fixStuff = """
#define EIGEN_DEFAULT_TO_ROW_MAJOR 1
#include <dune/functions/functionspacebases/subspacebasis.hh>
#include <dune/functions/functionspacebases/subentitydofs.hh>
#include <dune/iga/utils/igahelpers.hh>
#include <dune/python/pybind11/eigen.h>
void fixStuff(auto& basis_, auto dirichletFlags_) {
 auto dirichletFlags = dirichletFlags_.template cast<Eigen::Ref<Eigen::VectorX<bool>>>();
 std::cout << Dune::className(dirichletFlags) << std::endl;
 Dune::Functions::forEachUntrimmedBoundaryDOF(Dune::Functions::subspaceBasis(basis_, 2),
                                                 [&](auto&& localIndex, auto&& localView, auto&& intersection) {
                                                   dirichletFlags[localView.index(localIndex)] = true;
                                                 });
    auto fixEverything = [&](auto&& subBasis_) {
      auto localView       = subBasis_.localView();
      auto seDOFs          = subEntityDOFs(subBasis_);
      const auto& gridView = subBasis_.gridView();
      for (auto&& element : elements(gridView)) {
        localView.bind(element);
        for (const auto& intersection : intersections(gridView, element))
          for (auto localIndex : seDOFs.bind(localView, intersection))
           dirichletFlags[localView.index(localIndex)] = true;
      }
    };
    fixEverything(Dune::Functions::subspaceBasis(basis_, 0));
    fixEverything(Dune::Functions::subspaceBasis(basis_, 1));
}
"""