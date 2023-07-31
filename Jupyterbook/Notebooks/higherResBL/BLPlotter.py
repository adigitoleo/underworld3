"""
This function plots out the flow plots for for BL.py 



What do I need to do in this function.

1. I need for BL.py to save the mesh file
2. I need to read the mesh file in
3. I need to read the v variable in

Then, I should just be able to plot it all out
"""

## imports
import os
import petsc4py
import underworld3 as uw
import numpy as np
import sympy
import mpi4py
import matplotlib.pyplot as plt
import copy
import math 
from mpi4py import MPI
import pickle

import os
import re


mesh = uw.discretisation.Mesh(f"BLResults/mesh/meshFile.h5",
                                  qdegree=3)

VDegree = 2
PDegree = 1

v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=VDegree)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=PDegree)

## plotting of the velocity field
def plot(mesh, v, step, path):
    """ 
    Plots the velocity field on the mesh at a timestep
    Inputs:
    mesh: uw mesh variable
    v: 2 dimensional underworld mesh variable repressenting flow velocity
    ns: underworld 3 navier stokes solver
    step: int repressenting the step in the simulation (used for plotting)
    path (optional): Path where plot is saved

    Output:
    File located at nsPlots that shows the velocity of the fluid throughout the 
    simulaton domain 
    """
    if mpi4py.MPI.COMM_WORLD.size == 1:
        import numpy as np
        import pyvista as pv
        import vtk
        
        pv.start_xvfb()
        pv.global_theme.background = "white"
        pv.global_theme.window_size = [750, 1200]
        pv.global_theme.antialiasing = True
        pv.global_theme.jupyter_backend = "panel"
        pv.global_theme.smooth_shading = True
        mesh.vtk("tmp_mesh.vtk")
        pvmesh = pv.read("tmp_mesh.vtk")
        ##pvmesh.point_data["P"] = uw.function.evaluate(p.sym[0], mesh.data)
        pvmesh.point_data["V"] = uw.function.evaluate( (v.sym.dot(v.sym))**0.5, mesh.data)
        #pvmesh.point_data["V_Star"] = uw.function.evaluate(v_star.sym.dot(v_star.sym), mesh.data)
        arrow_loc = np.zeros((v.coords.shape[0], 3))
        arrow_loc[:, 0:2] = v.coords[...]
        arrow_length = np.zeros((v.coords.shape[0], 3))
        arrow_length[:, 0] = uw.function.evaluate(v.sym[0], v.coords)*0.01
        arrow_length[:, 1] = uw.function.evaluate(v.sym[1], v.coords)*0.01
        pl = pv.Plotter(window_size=[1000, 1000], off_screen=True)
        pl.add_axes()
        pl.add_mesh(
            pvmesh,
            cmap="coolwarm",
            edge_color="Black",
            show_edges=True,
            scalars="V",
            use_transparency=False,
            opacity=1.0,
        )
        pl.show(cpos="xy", screenshot = path+str(step)+".png")

                        
def find_k(path='./BLResults/states'):
    files = os.listdir(path)
    pattern = re.compile(r'(\d+)swarm\.h5$')

    k = -1
    for filename in files:
        match = pattern.match(filename)
        if match:
            number = int(match.group(1))
            if number > k:
                k = number

    return k


filePath = "BLResults/states/"

largestIndex = find_k(path = filePath)
for index in range(0, largestIndex):
    v.read_from_vertex_checkpoint(path+str(index)+".U.0.h5", data_name="U")
    p.read_from_vertex_checkpoint(path+str(index) + ".P.0.h5", data_name="P")
    plot(mesh, v, step, "BLResults/FlowPlots/")
    
    
    

    


    

    










