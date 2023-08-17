## This benchmark is for Poiseuille flow in a pipe in underworld 3


import os
import petsc4py
import underworld3 as uw
import numpy as np
import sympy
import mpi4py
##import pygmsh
import matplotlib.pyplot as plt
import copy
import math 

from mpi4py import MPI
import pickle

# In[2]:

boxLength = 1
boxHeight = 1
##boxLength = 4/Re*5 ## 3
##boxHeight = 4/Re*3 ## 3
##normedRes = 30
resolution = 0.1

vel = 1
viscosity = 0.001 ## bl height condition
if (uw.mpi.rank == 0):
    print(vel * boxLength/viscosity)
    print((2 * viscosity*boxLength/vel)**0.5)

mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0,0), maxCoords=(boxLength, boxHeight), cellSize=resolution, qdegree=3)
mesh.dm.view()

v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)      

swarm = uw.swarm.Swarm(mesh=mesh, recycle_rate=20) ## length of streak

v_star = uw.swarm.SwarmVariable("Vs", swarm, mesh.dim, 
                            proxy_degree=2, proxy_continuous=True) 
                            
swarm.populate(fill_param=2)

ns = uw.systems.NavierStokesSwarm(
    mesh,
    velocityField = v,
    pressureField = p,
    velocityStar_fn = v_star.sym,
)


ns.add_dirichlet_bc( (0.0, 0.0), "Top", (0,1))
ns.add_dirichlet_bc( (0.0, 0.0), "Bottom", (0,1))
ns.add_dirichlet_bc( 1, "Left", (0,))
ns.add_dirichlet_bc( 0 , "Right", (0,))


##ns.add_dirichlet_bc( (0, 0), "right", (0, 1))

ns.bodyforce = sympy.Matrix([0.0, 0.0])
ns.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(mesh.dim)
ns.constitutive_model.Parameters.viscosity = viscosity

ns.saddle_preconditioner = 1.0 / ns.constitutive_model.Parameters.viscosity


# In[4]:

def plot(mesh, v, ns,step):
    print("in plot")
    
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
        pvmesh.point_data["P"] = uw.function.evaluate(p.sym[0], mesh.data)
        pvmesh.point_data["V"] = uw.function.evaluate( (v.sym.dot(v.sym))**0.5, mesh.data)
        #pvmesh.point_data["V_Star"] = uw.function.evaluate(v_star.sym.dot(v_star.sym), mesh.data)
        arrow_loc = np.zeros((ns.u.coords.shape[0], 3))
        arrow_loc[:, 0:2] = ns.u.coords[...]
        arrow_length = np.zeros((ns.u.coords.shape[0], 3))
        arrow_length[:, 0] = uw.function.evaluate(ns.u.sym[0], ns.u.coords)*0.01
        arrow_length[:, 1] = uw.function.evaluate(ns.u.sym[1], ns.u.coords)*0.01
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
        ##pl.add_arrows(arrow_loc, arrow_length, mag=3)
        pl.show(cpos="xy", screenshot = str(step)+".png")




# In[5]:
def getDifference(oldVars, newVars):
    error = 0
    counter = 0
    for vIndex in range(len(oldVars)):
        oldVar = oldVars[vIndex]
        newVar = newVars[vIndex]

        dimension = len(oldVar[0])


        for elIndex in range(len(oldVar)):
                
            for dIndex in range(dimension):
                error += abs(oldVar[elIndex, dIndex] - newVar[elIndex, dIndex])
                counter += 1

    return error/counter

def saveState(mesh, filename):
    ## save the mesh, save the mesh variables
    mesh.write_timestep_xdmf(filename = filename, meshVars=[v], index=0)

# In[6]:
ts = 0
dt_ns = 1.0e-10
maxsteps = 10
differences= []
pdifferences=[]



for step in range(0, maxsteps):
    if (uw.mpi.rank == 0):
        print("step", str(step))

    if (uw.mpi.rank == 0):
        with mesh.access():
            old_v_data = copy.deepcopy(v.data)
    
    if (uw.mpi.rank == 0):
        plot(mesh, v, ns, step)
    ns.solve(timestep= dt_ns, zero_init_guess=False)
    delta_t_swarm = 1.0 * ns.estimate_dt()
    delta_t = min(delta_t_swarm, dt_ns)
    phi = min(1.0, delta_t/dt_ns)
    
    with swarm.access(v_star):
        v_star.data[...] = (
            phi * v.rbf_interpolate(swarm.data) + (1.0 -  phi) * v_star.data
        )

    swarm.advection(v.fn, delta_t, corrector=False)

    if (uw.mpi.rank == 0):
        if (step != 0):
            with mesh.access():
                v_data = v.data
                differences.append(getDifference([old_v_data], [v_data]) )

        plt.plot(differences)
        plt.savefig("differences.png")
        plt.clf()
        try:
            logDifferences = [math.log(el) for el in differences]
            plt.plot(logDifferences)
            plt.savefig("logDifferences.png")
            plt.clf()
        except:
            print("Converged!")




