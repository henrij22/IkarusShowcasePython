from nurbs_basis import globalBasis
from fix_dirichlet import *

from io import StringIO

import pyvista as pv
from tabulate import tabulate
import pandas as pd
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
from dune.iga.basis import defaultGlobalBasis, Power, Nurbs

from dune.vtk import vtkWriter

LAMBDA_LOAD = 1.0
THICKNESS = 0.1  # 10 cm


def run_simulation(deg: int, refine: int):
    reader = {
        "reader": readeriga.json,
        "file_path": "input/plate_holes.ibra",
        "trim": True,
        "elevate_degree": (deg - 1, deg - 1),
    }

    gridView = IGAGrid(reader, dimgrid=2, dimworld=3)
    gridView.hierarchicalGrid.globalRefine(refine)

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
        fes.append(
            iks.finite_elements.KirchhoffLoveShell(
                basis, e, 1000, 0.0, THICKNESS, volumeLoad
            )
        )

    assembler = iks.assembler.sparseFlatAssembler(fes, dirichletValues)

    print(
        f"Size of full System: {flatBasis.dimension}, size of red. System: {assembler.reducedSize()}"
    )

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

    maxiter = 300
    abs_tolerance = 1e-9
    d = np.zeros(assembler.reducedSize())
    for k in range(maxiter):
        R = gradient(d)
        K = jacobian(d)
        r_norm = sp.linalg.norm(R)

        deltad = sp.sparse.linalg.spsolve(K, R)
        d -= deltad
        # print(k, sp.linalg.norm(deltad))
        if r_norm < abs_tolerance:
            print(
                f"Solution found after {k} iterations, norm: {sp.linalg.norm(deltad)}"
            )
            break
        if k == maxiter:
            print(
                f"Solution not found after {k} iterations, norm: {sp.linalg.norm(deltad)}"
            )

    dFull = assembler.createFullVector(d)
    dispFunc = flatBasis.asFunction(dFull)

    vtkWriter = gridView.trimmedVtkWriter()
    vtkWriter.addPointData(dispFunc, name="displacement")

    vtkWriter.write(name=f"out/result_d{deg}_r{refine}")

    # Do some postprocessing with pyVista
    mesh = pv.UnstructuredGrid(f"out/result_d{deg}_r{refine}.vtu")

    # nodal displacements in z-Direction
    disp_z = mesh["displacement"][:, 2]

    max_d = np.max(disp_z)
    print(f"Max d: {max_d}")

    return max_d, k


def postprocess():
    pass
    # Postprocessing with pyVista (doesnt seem to work within devcontainer)

    # mesh = pv.UnstructuredGrid("out/result.vtu")
    # plotter = pv.Plotter(off_screen=True, notebook=True)
    # plotter.view_xy()
    # plotter.add_mesh(mesh, scalars="displacement", component=2, show_edges=True)
    # plotter.screenshot("out/displacement.png", transparent_background=True)
    # grid_repr.plot(cpos='xy', scalars="displacement", component=2, show_edges=True)


if __name__ == "__main__":
    # degree: 1 to 4 (1 = linear)
    # refine: 3 to 6

    data = []
    for i in range(1, 4):
        for j in range(3, 6):
            max_d, iterations = run_simulation(deg=i, refine=j)
            data.append((i, j, max_d, iterations))

    df = pd.DataFrame(data, columns=["Degree", "Refinement", "max d", "iterations"])
    print(tabulate(df, headers="keys", tablefmt="psql"))
