from nurbs_basis import globalBasis
from fix_dirichlet import fixStuff

import sys
import os
from io import StringIO

import pyvista as pv
import numpy as np
import scipy as sp

import ikarus as iks
import ikarus.finite_elements

import ikarus.assembler
import ikarus.dirichlet_values


from dune.vtk import vtkWriter
from dune.iga import IGAGrid
from dune.generator.algorithm import run

from dune.grid import gridFunction
from dune.iga import reader as readeriga
from dune.iga.basis import defaultGlobalBasis, Power, Lagrange, Nurbs

from dune.vtk import vtkWriter
from dune.functions import subspaceBasis 

LAMBDA_LOAD = 1.0
THICKNESS = 0.1  # 10 cm


if __name__ == "__main__":
    # reader = (readeriga.json, "input/plate_holes.ibra")

    reader = {
        "reader": readeriga.json,
        "file_path": "input/plate_holes.ibra",
        "trim": True,
        "degree_elevate": (1, 1)
    }

    gridView = IGAGrid(reader, dimgrid=2, dimworld=3)
    gridView.hierarchicalGrid.globalRefine(5)

    # gridView.hierarchicalGrid.globalRefineInDirection(0, 1, True)

    basis = globalBasis(gridView, Power(Nurbs(), 3))
    flatBasis = basis.flat()

    ## Define Load
    def volumeLoad(x, lambdaVal):
        return np.array([0, 0, THICKNESS**3 * LAMBDA_LOAD])
    

    ## Define Dirichlet Boundary Conditions
    dirichletValues = iks.dirichletValues(flatBasis)

    def fixDofs_lambda(basis_, dirichletFlags_):
        run("fixStuff", StringIO(fixStuff), basis_, dirichletFlags_)
        
                     
    dirichletValues.fixDOFs(fixDofs_lambda)
       


    ## Create Elements
  
    fes = []
    for e in gridView.elements:
        fes.append( iks.finite_elements.KirchhoffLoveShell(
                basis, e, 1000, 0.0, THICKNESS, volumeLoad))

    assembler = iks.assembler.sparseFlatAssembler(fes, dirichletValues)

    print(f"Size of full System: {flatBasis.dimension}, size of red. System: {assembler.reducedSize()}")

  
    ## Solve non-linear Kirchhol-Love-Shell problem
    def gradient(dRed_):
        req = ikarus.FERequirements()
        req.addAffordance(iks.VectorAffordances.forces)
        lambdaLoad = iks.ValueWrapper(LAMBDA_LOAD)
        req.insertParameter(iks.FEParameter.loadfactor, lambdaLoad)
    
        dFull = assembler.createFullVector(dRed_)
        req.insertGlobalSolution(iks.FESolutions.displacement, dFull)
    
        return assembler.getReducedVector(req)
    
    def jacobian(dRed_): 
        req = ikarus.FERequirements()
        req.addAffordance(iks.MatrixAffordances.stiffness)
        lambdaLoad = iks.ValueWrapper(LAMBDA_LOAD)
        req.insertParameter(iks.FEParameter.loadfactor, lambdaLoad)
    
        dFull = assembler.createFullVector(dRed_)
        req.insertGlobalSolution(iks.FESolutions.displacement, dFull)

        return assembler.getReducedMatrix(req)


    maxiter = 100
    abs_tolerance = 1e-8
    d = np.zeros(assembler.reducedSize())
    for k in range(maxiter):
        R = gradient(d)
        K = jacobian(d)
        r_norm = sp.linalg.norm(R)

        deltad = sp.sparse.linalg.spsolve(K, R)
        d -= deltad
        print(k, sp.linalg.norm(deltad))
        if r_norm < abs_tolerance:
            break

    dFull = assembler.createFullVector(d)
    dispFunc = flatBasis.asFunction(dFull)


    vtkWriter = gridView.trimmedVtkWriter()
    vtkWriter.addPointData(dispFunc, name="displacement")

    vtkWriter.write(name="out/result")


    # grid_repr = pv.UnstructuredGrid("out/result.vtu")
    # grid_repr.plot(cpos='xy', scalars="displacement", component=2, show_edges=True)
