#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8


# # Solving the Poisson Equation in underworld 3 (Part 2)<br>
# <br>
# This notebook shows how to solve the Poisson equation ∇ · (k ∇ φ) = S for spatially non-constant k and non-zero source term S with Dirichlet boundary conditions in a 2-dimensional Cartesian domain.

# In[1]:

# In[2]:


import underworld3 as uw  # import underworld
from mpi4py import MPI  # library for displaying
import sympy  # for generating symbolic expressions


# Here, we solve the equation ∇ · ((x+y) ∇ φ) = 2π((1-2π(x+y)) cos(2πx) - (2π(x+y)+1) sin(2πx)) on [0,1] × [0,1] with the boundary conditions <br>
# <br>
# φ(x,0) = φ(x,1) = sin(2πx) + cos(2πx), φ(0,y) = φ(1,y) = 1 <br>
# This can be done by finding the flux F = k ∇ φ and then solving ∇ · F = S with k = x+y and S = 2π((1-2π(x+y)) cos(2πx) - (2π(x+y)+1) sin(2πx))<br>
# <br>
# We setup our mesh, mesh variables, and solver as in Part 1.

# In[2]:

# In[3]:


mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 24, qdegree=5)
phi = uw.discretisation.MeshVariable(r"\phi", mesh, 1, degree=3)
poisson_solver = uw.systems.Poisson(mesh, phi)


# We set our source function.

# In[3]:

# Access the mesh coordinates

# In[4]:


x, y = mesh.X 


# Define our source function

# In[5]:


source_fn = 2 * sympy.pi * ((1 - 2 * sympy.pi * (x + y)) * sympy.cos(2 * sympy.pi * x) - (2 * sympy.pi * (x + y) + 1) * sympy.sin(2 * sympy.pi * x))


# Set our source term in the Poisson solver

# In[6]:


poisson_solver.f = -source_fn


# In[7]:


source_fn


# We use a diffusion constitutive model to relate the flux F to the derivatives of the field ∇φ by F = k ∇φ. However, this time k is a function of the mesh coordinates.

# In[4]:

# Give our Poisson solver a diffusion model

# In[8]:


poisson_solver.constitutive_model = uw.systems.constitutive_models.DiffusionModel(mesh.dim) 


# Access the mesh coordinates

# In[9]:


x, y = mesh.X


# Write our diffusivity as a symbolic function of the mesh coordinates

# In[10]:


k_fn = x + y


# Set our diffusivity

# In[11]:


poisson_solver.constitutive_model.Parameters.diffusivity = k_fn


# We set the boundary conditions for the field φ.

# In[10]:

# In[12]:


top_condition = sympy.cos(2 * sympy.pi * x) + sympy.sin(2 * sympy.pi * x)
bottom_condition = sympy.cos(2 * sympy.pi * x) + sympy.sin(2 * sympy.pi * x)
left_condition = 1
right_condition = 1


# In[13]:


poisson_solver.add_dirichlet_bc(top_condition, "Top")
poisson_solver.add_dirichlet_bc(bottom_condition, "Bottom")


# In[14]:


poisson_solver.add_dirichlet_bc(left_condition, "Left")
poisson_solver.add_dirichlet_bc(right_condition, "Right")


# And solve.

# In[11]:

# In[15]:


poisson_solver.solve()


# 

# In[12]:

# In[16]:


with mesh.access():  # Access the mesh
    # Get the numerical solution
    mesh_numerical_soln = uw.function.evaluate(poisson_solver.u.fn, mesh.data)


# In[17]:


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


# The analytical solution for this is φ = sin(2πx) + cos(2πx). Let's plot it.

# In[13]:

# In[18]:


analytic_fn = sympy.sin(2 * sympy.pi * x) + sympy.cos(2 * sympy.pi * x)
analytic_fn


# In[14]:

# In[19]:


with mesh.access():
    mesh_analytic_soln = uw.function.evaluate(analytic_fn, mesh.data)


# In[20]:


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
    pvmesh.point_data["phiAnalytic"] = mesh_analytic_soln
    sargs = dict(interactive=True)  # Doesn't appear to work :(
    pl = pv.Plotter()
    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="phiAnalytic",
        use_transparency=False,
        opacity=0.5,
        scalar_bar_args=sargs,
    )
    pl.camera_position = "xy"
    pl.show(cpos="xy")

