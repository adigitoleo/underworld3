#!/usr/bin/env python
# coding: utf-8

# # Projections in underworld 3
# 
# In underworld 3 we solve our equations for the mesh variables. However, sometimes, we also want to find functions of the mesh variables and their derivatives. To do this, we use projections. 
# 
# We demonstrate this idea by solving the steady state heat equation $\nabla \cdot (k \nabla T)=0$ with conductivity $k = 5 + \frac{(\nabla T) \cdot (\nabla T)}{1000}$ on the domain $[0,1] \times [0,1]$ with boundary conditions $T(x,0) = 0, T(x,1) = x, T(0,y) = 0, T(1,y) = y$. 
# 
# We then use projections to find how the conductivity varies in space.

# In[1]:


import underworld3 as uw ## import underworld
from mpi4py import MPI  # library for displaying
import sympy ## library for symbolc expressions


# We define a mesh on the domain $[0,1]\times[0,1]$ where we solve our problem.

# In[2]:


## define a 1x1 square and mesh it
mesh = uw.meshing.UnstructuredSimplexBox(minCoords = (0,0), maxCoords=(1,1), cellSize=1/32, qdegree=3)


# We define the temperature on the mesh aswell as a variable kappa which will store our thermal conducitivity $k$. 

# In[3]:


temperature = uw.discretisation.MeshVariable("T", mesh, 1, degree=3)
kappa = uw.discretisation.MeshVariable("k", mesh, 1, degree=3)


# Our governing equation is a poisson equation, so we use a Poisson solver and give it a diffusion consitutive model.

# In[4]:


poisson_solver = uw.systems.Poisson(mesh, temperature)
poisson_solver.constitutive_model = uw.systems.constitutive_models.DiffusionModel(mesh.dim)


# We set the Dirichlet Boundary conditions

# In[12]:


x,y = mesh.X
top = x
right = y

poisson_solver.add_dirichlet_bc(top, "Top")
poisson_solver.add_dirichlet_bc(0, "Bottom")
poisson_solver.add_dirichlet_bc(0, "Left")
poisson_solver.add_dirichlet_bc(right, "Right")


# To write out diffusivity/conductivity $k$, we need to access the symbolic repressentations for the gradient of the temperature. Then, we build the diffusivity/conductivity out of these derivatives.

# In[5]:


## Get the symbolic repressentation of the temperature gradient
delT = mesh.vector.gradient(temperature.sym)
## write our diffusivity symbolically
k = 5 + (delT.dot(delT))/1000



# We use the same method describe in Poisson3.py to slowly introduce non-linearity to avoid divergence of the answer.

# In[ ]:


aList = [1, 0.8, 0.6, 0.4, 0.2, 0]
for a in aList:
    poisson_solver.constitutive_model.Parameters.diffusivity = a + (1 - a)*k
    poisson_solver.solve()


# Now, lets plot our numerical solution for temperature

# In[6]:


with mesh.access():  # Access the mesh
    # Get the numerical solution
    mesh_numerical_soln = uw.function.evaluate(poisson_solver.u.fn, mesh.data)


if MPI.COMM_WORLD.size == 1:
    import numpy as np
    import pyvista as pv
    pv.global_theme.background = "white"
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True
    mesh.vtk("ignore_mesh.vtk")
    pvmesh = pv.read("ignore_mesh.vtk")
    pvmesh.point_data["phi"] = mesh_numerical_soln
    sargs = dict(interactive=True)  # Doesn't appear to work :(
    pl = pv.Plotter()
    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="phi",
        use_transparency=False,
        opacity=0.5,
        scalar_bar_args=sargs,
    )
    pl.camera_position = "xy"
    pl.show(cpos="xy")


# Now, we find the conducitivity using a projection. 

# In[14]:


## Create a solver on the mesh that stores its data in the mesh variable kappa
diffusivity_solver = uw.systems.Projection(mesh, kappa)

## give the solver a symbolc function so that we can compute it
diffusivity_solver.uw_function = sympy.Matrix(
    [poisson_solver.constitutive_model.Parameters.diffusivity]
)
## give the solver boundary conditions
diffusivity_solver.add_dirichlet_bc(k, ["Top", "Bottom", "Left", "Right"])
## smooth our the answer
diffusivity.smoothing = 1.0e-3


# In[15]:


## Go and solve the solver. Here, we are solving $5 +  \frac{(\nabla T) \cdot (\nabla T)}{1000}$
## and storing the answer in the mesh variable kappa.
diffusivity_solver.solve()


# Now, lets plot our conductivity/diffusivity

# In[9]:


with mesh.access():  # Access the mesh
    # Get the numerical solution
    diffusivity_numerical = uw.function.evaluate(kappa.sym[0], mesh.data)


if MPI.COMM_WORLD.size == 1:
    import numpy as np
    import pyvista as pv
    pv.global_theme.background = "white"
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True
    mesh.vtk("ignore_mesh.vtk")
    pvmesh = pv.read("ignore_mesh.vtk")
    pvmesh.point_data["kappa"] = diffusivity_numerical
    sargs = dict(interactive=True)  # Doesn't appear to work :(
    pl = pv.Plotter()
    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="kappa",
        use_transparency=False,
        opacity=0.5,
        scalar_bar_args=sargs,
    )
    pl.camera_position = "xy"
    pl.show(cpos="xy")



# In[10]:





# In[11]:






# In[ ]:




