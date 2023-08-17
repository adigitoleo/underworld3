#!/usr/bin/env python
# coding: utf-8

# # Playing around with navier stokes in underworld 3

# In[1]:


import os
import petsc4py
import underworld3 as uw
import numpy as np
import sympy
import mpi4py
import pygmsh
import matplotlib.pyplot as plt

from mpi4py import MPI

# In[2]:


resolution = 0.1

if uw.mpi.rank == 0:

    # Generate local mesh on boss process

    with pygmsh.geo.Geometry() as geom:

        geom.characteristic_length_max = resolution

        inclusion = geom.add_circle(
            (0.5, 1, 0.0), 0.3, make_surface=False, mesh_size=resolution
        )
        domain = geom.add_rectangle(
            xmin=0.0,
            ymin=0.0,
            xmax=2,
            ymax=2,
            z=0,
            holes=[inclusion],
            mesh_size=resolution,
        )

        geom.add_physical(domain.surface.curve_loop.curves[0], label="bottom")
        geom.add_physical(domain.surface.curve_loop.curves[1], label="right")
        geom.add_physical(domain.surface.curve_loop.curves[2], label="top")
        geom.add_physical(domain.surface.curve_loop.curves[3], label="left")
        geom.add_physical(inclusion.curve_loop.curves, label="inclusion")
        geom.add_physical(domain.surface, label="Elements")

        geom.generate_mesh(dim=2, verbose=False)
        geom.save_geometry(f"meshes/ns_pipe_flow_{resolution}.msh")

mesh = uw.discretisation.Mesh(f"meshes/ns_pipe_flow_{resolution}.msh", 
                                  markVertices=True, 
                                  useMultipleTags=True, 
                                  useRegions=True,
                                  qdegree=3)
mesh.dm.view()


# In[3]:


##mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0,0), maxCoords=(5,1), cellSize=1/10, qdegree=3)

v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=2)
vs = uw.discretisation.MeshVariable("Us", mesh, mesh.dim, degree=2) ## not useful 


swarm = uw.swarm.Swarm(mesh=mesh, recycle_rate=20) ## length of streak


v_star = uw.swarm.SwarmVariable("Vs", swarm, mesh.dim, 
                            proxy_degree=2, proxy_continuous=True) ## 
                            
swarm.populate(fill_param=2)



ns = uw.systems.NavierStokesSwarm(
    mesh,
    velocityField = v,
    pressureField = p,
    velocityStar_fn = v_star.sym,
)



ns.add_dirichlet_bc( (1, 0), "top", (0, 1))
ns.add_dirichlet_bc( (0.0, 0.0), "bottom", (0, 1) )
ns.add_dirichlet_bc( (0.0, 0.0), "left", (0, 1) )
ns.add_dirichlet_bc( (0.0, 0.0), "inclusion", (0, 1)) 

ns.bodyforce = sympy.Matrix([0.0, 0.0])
ns.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(mesh.dim)
ns.constitutive_model.Parameters.viscosity = 1

ns.saddle_preconditioner = 1.0 / ns.constitutive_model.Parameters.viscosity


# In[4]:

def plot(mesh, v, ns,step):
    
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
        pvmesh.point_data["V"] = uw.function.evaluate(v.sym.dot(v.sym), mesh.data)
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
        pl.add_arrows(arrow_loc, arrow_length, mag=3)
        pl.show(cpos="xy", screenshot = "nsPlots/"+str(step)+".png")



def plot2(mesh, v, ns,step):
    
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
        pvmesh.point_data["V"] = uw.function.evaluate(v.sym.dot(v.sym), mesh.data)
        #pvmesh.point_data["V_Star"] = uw.function.evaluate(v_star.sym.dot(v_star.sym), mesh.data)
        arrow_loc = np.zeros((ns.u.coords.shape[0], 3))
        arrow_loc[:, 0:2] = ns.u.coords[...]
        arrow_length = np.zeros((ns.u.coords.shape[0], 3))
        arrow_length[:, 0] = uw.function.evaluate(ns.u.sym[0], ns.u.coords)*0.01
        arrow_length[:, 1] = uw.function.evaluate(ns.u.sym[1], ns.u.coords)*0.01
        pl = pv.Plotter(window_size=[1000, 1000])
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
        pl.add_arrows(arrow_loc, arrow_length, mag=3)
        pl.show(cpos="xy", screenshot = "nsPlots/"+str(step)+".png")
        #pl.screenshot("nsPlots/"+str(step)+".png")
        #plt.imshow(pl.image)
        #plt.savefig("nsPlots/"+str(step)+".png")

# In[5]:




# In[6]:


ts = 0
dt_ns = 1.0e-1
maxsteps = 100


for step in range(0, maxsteps):
    if (uw.mpi.rank == 0):
        plot(mesh, v, ns, step)
    ns.solve(timestep= dt_ns, zero_init_guess=False)
    delta_t_swarm = 0.5 * ns.estimate_dt()
    delta_t = min(delta_t_swarm, dt_ns)
    phi = min(1.0, delta_t/dt_ns)
    
    with swarm.access(v_star):
        v_star.data[...] = (
            phi * v.rbf_interpolate(swarm.data) + (1.0 -  phi) * v_star.data
        )

    swarm.advection(v.fn, delta_t, corrector=False)


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




