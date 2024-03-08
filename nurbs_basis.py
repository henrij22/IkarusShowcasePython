from dune.iga.basis import defaultGlobalBasis, Power, Lagrange, Nurbs
import ikarus as iks
import ikarus.finite_elements
import ikarus.utils
import ikarus.assembler
import ikarus.dirichlet_values
from ikarus import basis
from dune.iga import (
    IGAGrid,
)

from dune.generator.generator import SimpleGenerator
from dune.iga.basis import preBasisTypeName
from dune.common.hashit import hashIt

# this is a stupied workaround since dune-function has a not very general impl of preBasis.
# to support Nurbs() we have to do it ourselves, it creates a ikarus basis
def globalBasis(gv, tree):
    generator = SimpleGenerator("BasisHandler", "Ikarus::Python")

    pbfName = preBasisTypeName(tree, gv.cppTypeName)
    element_type = f"Ikarus::BasisHandler<{pbfName}>"
    includes = []
    includes += list(gv.cppIncludes)
    includes += ["dune/iga/nurbsbasis.hh"]
    includes += ["ikarus/python/basis/basis.hh"]

    moduleName = "Basis_" + hashIt(element_type)
    module = generator.load(
        includes=includes, typeName=element_type, moduleName=moduleName
    )
    basis = defaultGlobalBasis(gv, tree)
    return module.BasisHandler(basis)