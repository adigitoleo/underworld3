#!/usr/bin/env python
# coding: utf-8

# # Playing around with convection

# In[1]:


import underworld3 as uw
import sympy
from mpi4py import MPI  # library for displaying


# In[2]:


mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0,0), maxCoords=(1,1), cellSize=0.05, qdegree=3)


# In[3]:


v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=3)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=3)
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=3)


# In[4]:


## make the bodyforce
g=9.81 ## gravities acceleration
alpha = 0.1 ## coeffecient of thermal expansion
rho0 = 1000 ## background density
kappa = 0.001 ## thermal conductivity
heatingTemp = 100
viscosity = 1

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(mesh.dim)
stokes.constitutive_model.Parameters.viscosity = 1

stokes.add_dirichlet_bc( (0, 0), ["Top", "Bottom", "Left", "Right"], (0, 1) )




stokes.bodyforce = sympy.Matrix( [0, g * rho0*(1 - alpha*T.sym[0]) ] )

adSolver = uw.systems.AdvDiffusion(mesh,u_Field=T, V_Field=v)
adSolver.constitutive_model = uw.systems.constitutive_models.DiffusionModel(mesh.dim)
adSolver.constitutive_model.Parameters.diffusivity = kappa
adSolver.theta = 0.5

x,y = mesh.X

adSolver.add_dirichlet_bc(heatingTemp*x, "Bottom")
adSolver.add_dirichlet_bc(-heatingTemp*(1-x), "Top")
adSolver.add_neumann_bc(0, "Left")
adSolver.add_neumann_bc(0, "Right")


# Then we want to know what the Raylieigh number and Gr is here. Lets make both large to get some nice turbulent flows.
# 

# In[5]:


Ra = alpha*(2*heatingTemp) * g * 1**3 /(kappa*viscosity)
typicalVel = g * alpha*(2 * heatingTemp) * 1**2/viscosity
print(Ra, typicalVel)


# In[6]:


def plotTemp(s, mesh, step, time):
    ## s is a solver?

    import os
    import numpy as np
    import pyvista as pv
    from mpi4py import MPI

    # Ensure the output directory exists
    os.makedirs('images', exist_ok=True)

    with mesh.access():  # Access the mesh
        # Evaluate the mesh function
        mesh_numerical_soln = uw.function.evaluate(s.u.fn, mesh.data)

    if MPI.COMM_WORLD.size == 1:
        pv.global_theme.background = "white"
        pv.global_theme.window_size = [500, 500]
        pv.global_theme.antialiasing = True
        pv.global_theme.jupyter_backend = "panel"
        pv.global_theme.smooth_shading = True
        
        mesh_file = "ignore_mesh.vtk"
        
        # Ensure the mesh file is written
        try:
            mesh.vtk(mesh_file)
        except Exception as e:
            print(f"Error writing the mesh file: {e}")
            raise
        
        # Ensure the mesh file can be read
        try:
            pvmesh = pv.read(mesh_file)
        except Exception as e:
            print(f"Error reading the mesh file: {e}")
            raise
        
        pvmesh.point_data["T"] = mesh_numerical_soln
        sargs = dict(interactive=True)
        
        # Create the plot
        pl = pv.Plotter()
        

        
        # Add the mesh
        pl.add_mesh(
            pvmesh,
            cmap="coolwarm",
            edge_color="Black",
            show_edges=True,
            scalars="T",
            use_transparency=False,
            opacity=0.5,
            scalar_bar_args=sargs,
        )
        
        # Add the information text
        info = " steps: " + str(step) + " time: " + str(time)
        pl.add_text(info, position='upper_left', font_size=20, color='red')
        
        pl.camera_position = "xy"
        
        # Define output file name
        stringName = "images/" + str(step) + ".png"
        
        # Take a screenshot
        try:
            pl.screenshot(stringName)
        except Exception as e:
            print(f"Error taking a screenshot: {e}")
            raise

        






# In[7]:


time = 0
for step in range(10000):
    if (step % 10 == 0):
        plotTemp(adSolver, mesh, step, time)
    stokes.solve()
    dt = stokes.estimate_dt()
    time += dt
    adSolver.solve(timestep=dt)
    print("step", step)

    


# In[ ]:





# In[ ]:




