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

# In[2]:

vel = 1
viscosity = 1 ## bl height condition

boxLength = 2
boxHeight = 1.72*(viscosity*boxLength/vel)**0.5*1
##boxLength = 4/Re*5 ## 3
##boxHeight = 4/Re*3 ## 3
##normedRes = 30
resolution = 1/10


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
                            
swarm.populate(fill_param=2) ## 

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

term1 = uw.discretisation.MeshVariable("Term1", mesh, 2, degree=0)
term2 = uw.discretisation.MeshVariable("Term2", mesh, 2, degree=0)
term3 = uw.discretisation.MeshVariable("Term3", mesh, 2, degree=0)
term3u= uw.discretisation.MeshVariable("term3u", mesh, 2, degree=1)
term3l= uw.discretisation.MeshVariable("term3l", mesh, 2, degree=1)

term1_projector=uw.systems.Vector_Projection(mesh, term1)
term1_projector.uw_function = (mesh.vector.jacobian(v.sym) *v.sym.T).T



term2_projector=uw.systems.Vector_Projection(mesh, term2)
term2_projector.uw_function = -1/ns.rho * mesh.vector.gradient(p.sym)

term3u_projector=uw.systems.Vector_Projection(mesh, term3u)
term3u_projector.uw_function = mesh.vector.gradient(v.sym[0])

term3l_projector=uw.systems.Vector_Projection(mesh, term3l)
term3l_projector.uw_function = mesh.vector.gradient(v.sym[1])

print(term3l_projector.uw_function)

term3_projector=uw.systems.Vector_Projection(mesh, term3)
term3_projector.uw_function = viscosity * sympy.Matrix([mesh.vector.divergence(term3u.sym), mesh.vector.divergence(term3l.sym)]).T 
"""
dvdx = uw.discretisation.MeshVariable("dvdx", mesh, 1, degree=1)
dvdy = uw.discretisation.MeshVariable("dvdy", mesh, 1, degree=1)


term1_projector =uw.systems.Projection(mesh, term1)
term2_projector =uw.systems.Projection(mesh, term2)
dvdx_projector = uw.systems.Projection(mesh, dvdx)
dvdy_projector = uw.systems.Projection(mesh, dvdy)
term3_projector = uw.systems.Projection(mesh, term3)

term1_projector.uw_function = v.sym[1] * (mesh.vector.jacobian(v.sym)[2] + mesh.vector.jacobian(v.sym)[3])
term2_projector.uw_function = -1/ns.rho * mesh.vector.jacobian(p.sym)[1]

dvdx_projector.uw_function = mesh.vector.jacobian(v.sym)[2]
dvdy_projector.uw_function = mesh.vector.jacobian(v.sym)[3]

term3_projector.uw_function = viscosity  * (mesh.vector.jacobian(dvdx.sym)[0] + mesh.vector.jacobian(dvdy.sym)[1])
"""

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

def saveState(mesh, filename):
    ## save the mesh, save the mesh variables
    mesh.write_timestep_xdmf(filename = filename, meshVars=[v], index=0)

def loadState(mesh, filename):
    mesh = uw.discretisation.Mesh(f"meshes/ns_bl_test_{resolution}.msh", 
                                    markVertices=True, 
                                    useMultipleTags=True, 
                                    useRegions=True,
                                    qdegree=3)

    v_prev = uw.discretisation.MeshVariable("V2", mesh, mesh.dim, degree=3)
    p_prev = uw.discretisation.MeshVariable("P2", mesh, mesh.dim, degree=2)

    v_prev.read_from_vertex_checkpoint(infile + ".U.0.h5", data_name="U")
    p_prev.read_from_vertex_checkpoint(infile + ".P.0.h5", data_name="P")

    ## dont actually know how to load back in a swarm. fuck

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
dt_ns = 0.1*1/vel
maxsteps = 1000
differences= []
pdifferences=[]
blStep = 5



## lets set the initial fields as we want
"""
for step in range(5):
    with mesh.access(v,p):
        v.data[:,0] = vel
        v.data[:,1] = 0
        p.data[...] = 0

    with swarm.access(v_star):
        v_star.data[:,1] = vel
        v_star.data[:,0] = 0
    
    swarm.advection(v.fn, 0.1, corrector=False)
"""
        

    
        

time_passed = 0
for step in range(0, maxsteps):

    
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




