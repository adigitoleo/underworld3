#!/usr/bin/env python
# coding: utf-8

# # Playing around with navier stokes in underworld 3

# In[1]:
print("starting")
print("******************")

import time
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

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Blasius boundary layer simulation")
    parser.add_argument('-restart', action='store_true', help='Start from previous timestep?')
    return parser.parse_args()

args = parse_args()
restart = args.restart
print(restart)

def loadState(v,p,v_star,swarm):
    v.read_from_vertex_checkpoint("meshvars" + ".U.0.h5", data_name="U")
    p.read_from_vertex_checkpoint("meshvars" + ".P.0.h5", data_name="P")

    swarm.populate(fill_param=2)
    v_star.load(filename="Vs-0000.h5", swarmFilename="swarm-0000.h5")




vel = 1000
viscosity = 1 ## bl height condition

boxLength = 2
boxHeight = 1.72*(viscosity*boxLength/vel)**0.5*6
resolution = 1/20 * 1/10**0.25


if (uw.mpi.rank == 0):
    print(vel * boxLength/viscosity)
    print(1.72*(viscosity*boxLength/vel)**0.5)

mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0,0), maxCoords=(boxLength, boxHeight), cellSize=resolution, qdegree=3)

mesh.dm.view()


## define the mesh variables
v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)      

swarm = uw.swarm.Swarm(mesh=mesh, recycle_rate=5) ## length of streak

v_star = uw.swarm.SwarmVariable("Vs", swarm, mesh.dim, 
                            proxy_degree=2, proxy_continuous=True) ## 


ns = uw.systems.NavierStokesSwarm(
    mesh,
    velocityField = v,
    pressureField = p,
    velocityStar_fn = v_star.sym,
)

## set the boundary conditions
ns.add_dirichlet_bc( (vel, 0), "Bottom", (0,1))
ns.add_dirichlet_bc( (vel, 0.0), "Left", (0, 1) )
ns.add_dirichlet_bc( (vel, 0.0), "Top", (0,))

## set up the solver
ns.bodyforce = sympy.Matrix([0.0, 0.0])
ns.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(mesh.dim)
ns.constitutive_model.Parameters.viscosity = viscosity

ns.saddle_preconditioner = 1.0 / ns.constitutive_model.Parameters.viscosity

ns.rho = 1


## lets use the projectors here

term1 = uw.discretisation.MeshVariable("Term1", mesh, 2, degree=2)
term2 = uw.discretisation.MeshVariable("Term2", mesh, 2, degree=2)
term3 = uw.discretisation.MeshVariable("Term3", mesh, 2, degree=2)
term3u= uw.discretisation.MeshVariable("term3u", mesh, 2, degree=3)
term3l= uw.discretisation.MeshVariable("term3l", mesh, 2, degree=3)

term1_projector=uw.systems.Vector_Projection(mesh, term1)
term1_projector.uw_function = (mesh.vector.jacobian(v.sym) *v.sym.T).T



term2_projector=uw.systems.Vector_Projection(mesh, term2)
term2_projector.uw_function = -1/ns.rho * mesh.vector.gradient(p.sym)


term3u_projector=uw.systems.Vector_Projection(mesh, term3u)
term3u_projector.uw_function = mesh.vector.gradient(v.sym[0])


term3l_projector=uw.systems.Vector_Projection(mesh, term3l)
term3l_projector.uw_function = mesh.vector.gradient(v.sym[1])

term3_projector=uw.systems.Vector_Projection(mesh, term3)
term3_projector.uw_function = viscosity * sympy.Matrix([mesh.vector.divergence(term3u.sym), mesh.vector.divergence(term3l.sym)]).T 


if (restart == True):                     
    swarm.populate(fill_param=2) 
else:
    print("loading state")
    loadState(v, p, v_star, swarm)
    print("done")

## plotting things
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
        pl.show(cpos="xy", screenshot = "nsPlots/FreeStream"+str(step)+".png")




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

def saveState():
    ## save the mesh, save the mesh variables
    swarm.save_checkpoint(
        swarmName="swarm",
        swarmVars=[v_star],
        index=0

    )
    mesh.write_timestep_xdmf(filename = "meshvars", meshVars=[v,p], index=0)

def saveTerms(index):
    mesh.write_timestep.xdmf(filename="savedTerms"+str(index), meshVar=[v, p, term1, term2, term3], index=0)

def getBL(mesh, v):
    x,y = mesh.X

    stepSize = 1*resolution
    slides = [i*stepSize for i in range(int( boxLength / stepSize))]

    functions = [
        1/stepSize * (1 - v.sym[0]/vel) * sympy.Piecewise(
            (1,  sympy.And( (s < x), (x<=s + stepSize))  ),
            (0, True)
        ) for s in slides
    ]

    integrals = [uw.maths.Integral(mesh=mesh, fn=f) for f in functions]
    results = [i.evaluate() for i in integrals]
    
    return [slides, results]

ts = 0
dt_ns = 0.01*1/vel
maxsteps = 100
differences= []
pdifferences=[]
blStep = 10

startStep = 0 
if (restart==False):
    ## something like that
    with open("step.pkl", "rb") as f:
        startStep = pickle.load(f) + 1

    
        

time_passed = 0

for step in range(startStep, startStep+maxsteps):

    if (uw.mpi.rank == 0):
        print("step", str(step))

    if (uw.mpi.rank == 0):
        with mesh.access():
            old_v_data = copy.deepcopy(v.data)
    
    if (uw.mpi.rank == 0):
        if (step % 10 == 0):
            plot(mesh, v, ns, step)

    ## Then lets plot and save the boundary layer stuff

    BLData = getBL(mesh, v)
    
    if (uw.mpi.rank == 0):
        ## then lets save it using pickle and plot
        blpath = "bl/dataFreeStream"+str(step) + ".pkl"
        with open(blpath, 'wb') as f:
            pickle.dump(BLData, f)

        blPlotsPath = "blPlots/plotFreeStream"+str(step) + ".png"

        plt.plot(BLData[0], BLData[1])
        plt.savefig(blPlotsPath)
        plt.clf()

    if (step >= blStep):
        ns.add_dirichlet_bc( (0, 0), "Bottom", (0,1))
        ns.add_dirichlet_bc( (vel, 0.0), "Left", (0, 1) )
        ns.solve(timestep= dt_ns, zero_init_guess=False)
        delta_t_swarm = 10.0 * ns.estimate_dt()
        delta_t = min(delta_t_swarm, dt_ns)
        phi = min(1.0, delta_t/dt_ns)
        time_passed += delta_t
        print("delta_t_swarm:", delta_t_swarm, "dt_ns:", dt_ns, "delta_t:", delta_t, "phi:", phi, "time:", time_passed)
    else:
        with mesh.access(v,p):
            v.data[:,0] = vel
            v.data[:,1] = 0
            p.data[...] = 0

        with swarm.access(v_star):
            v_star.data[:,1] = vel
            v_star.data[:,0] = 0
        delta_t_swarm = dt_ns
        delta_t = dt_ns
        phi = 1

        

    if (uw.mpi.rank == 0):
        print("here is delta_t", str(delta_t) )
    
    with swarm.access(v_star):
        v_star.data[...] = (
            phi * v.rbf_interpolate(swarm.data) + (1.0 -  phi) * v_star.data
        )

    if (uw.mpi.rank == 0):
        print("starting to advect around")

    swarm.advection(v.fn, delta_t, corrector=False)

    ## now lets solve all the projectors
    term1_projector.solve()
    term2_projector.solve()
    term3u_projector.solve()
    term3l_projector.solve()
    term3_projector.solve()

    ## then lets plot them out
    print("pickle time")
    start = time.time()
    with mesh.access():

       with open('size_data.pkl', 'wb') as f:
           pickle.dump([term1.data, term2.data, term3.data], f)
    end = time.time()
    print("done pickle, time:", str(end - start), "seconds")


    ## Now lets save everything
    print("starting save")
    saveState()
    saveTerms(step)
    
    print("start saving")

    ## lets save our previous step
    with open("step.pkl", "wb") as f:
        pickle.dump(step, f)

    if (uw.mpi.rank == 0):
        print("starting to plot")

    if (uw.mpi.rank == 0):
        if (step != 0):
            with mesh.access():
                v_data = v.data
                differences.append(getDifference([old_v_data], [v_data]) )

        with open("bl/differenceDataFreeStream.pkl", 'wb') as f:
            pickle.dump(differences, f)

        plt.plot(differences)
        plt.savefig("differencesFreeStream.png")
        plt.clf()


    


"""
for step in range(0, maxsteps):
    plot(mesh, v, ns, step)
    delta_t_swarm = 2.0 * ns.estimate_dt()
    delta_t = min(delta_t_swarm, dt_ns)
    phi = min(1.0, delta_t / dt_ns)

    ns.solve(timestep=dt_ns, 
                        zero_init_guess=False)

    ## no need for this
    with swarm.access(v_star):
        v_star.data[...] = (
            phi * v.rbf_interpolate(swarm.data) 
            # phi * uw.function.evaluate(v_soln.fn, swarm.data)
            + (1.0 - phi) * v_star.data
        )
    # update integration swarm
    swarm.advection(v.fn, delta_t, corrector=False)
"""

# In[ ]:


"""
dt = 1
ns_dt = t

ts = 0
dt_ns = 1.0e-2

ns.solve(timestep=dt_ns)

with swarm.access(v_star):
    v_star.data[...] = uw.function.evaluate(v.fn, swarm.data)

print("starting the loop")

for index in range(10):
    print("plotting")
    plot(mesh, v_star, ns, index)
    dt = ns.estimate_dt()
    print(dt)



    ns.solve(timestep=dt, zero_init_guess=False)
    print("advecting the swarm")

    ## update the swarm values

    ## no need for this
    with swarm.access(v_star):
        v_star.data[...] = (
            phi * v_soln.rbf_interpolate(swarm.data) 
            # phi * uw.function.evaluate(v_soln.fn, swarm.data)
            + (1.0 - phi) * v_star.data
        )



    ## update the swarme
    swarm.advection(v.fn, dt, corrector=False) ##
    print("starting the loop")
"""



# In[ ]:




