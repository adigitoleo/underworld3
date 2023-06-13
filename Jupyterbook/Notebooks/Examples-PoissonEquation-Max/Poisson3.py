#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8


# # Solving the Poisson Equation in underworld 3 (Part 3)<br>
# <br>
# This notebook shows how to solve the non-linear Poisson equation $\nabla \cdot (k \nabla \phi) = S$ for $k$ as a function of the field $\phi$ and non-zero source term $S$ with Dirichlet boundary conditions in a 2-dimensional Cartesian domain. 

# In[2]:


import underworld3 as uw  # import underworld
from mpi4py import MPI  # library for parallel computing
import sympy  # for generating symbolic expressions


# Here, we solve a non-linear Poisson equation $\nabla \cdot ((1 + \phi^2) \nabla \phi) = 4(1 + 3x^4 + 6x^2y^2 + 3y^4)$ using the Poisson solver on the domain $[0,1] \times [0,1]$ with boundary conditions $\phi(x,0) = x^2$, $\phi(x,1) = 1 + x^2$, $\phi(0,y) = y^2$, $\phi(1,y) = 1 + y^2$. To do this, we set $k = 1 + \phi^2$ and set our source term to $S(\mathbf{x}) = 4(1 + 3x^4 + 6x^2y^2 + 3y^4)$. We will follow the basic setup of Poisson1.ipynb and Poisson2.ipynb.

# Define our mesh

# In[3]:


mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 24, qdegree=5)


# Define our scalar variable phi

# In[4]:


phi = uw.discretisation.MeshVariable(r"\phi", mesh, 1, degree=3)


# Define the Poisson solver

# In[5]:


poisson_solver = uw.systems.Poisson(mesh, phi)


# Set the source term

# In[6]:


x, y = mesh.X
source_fn = 4 * (1 + 3 * x ** 4 + 6 * x ** 2 * y ** 2 + 3 * y ** 4)
poisson_solver.f = -source_fn


# Set the diffusive model

# In[7]:


poisson_solver.constitutive_model = uw.systems.constitutive_models.DiffusionModel(mesh.dim)


# Set the boundary conditions

# In[8]:


top_condition = 1 + x ** 2
bottom_condition = x ** 2
left_condition = y ** 2
right_condition = 1 + y ** 2


# In[9]:


poisson_solver.add_dirichlet_bc(top_condition, "Top")
poisson_solver.add_dirichlet_bc(bottom_condition, "Bottom")
poisson_solver.add_dirichlet_bc(left_condition, "Left")
poisson_solver.add_dirichlet_bc(right_condition, "Right")


# Set the diffusivity k as a function of the scalar field phi

# In[10]:


k_fn = 1 + phi.sym[0] ** 2
poisson_solver.constitutive_model.Parameters.diffusivity = k_fn


# Introduce non-linearity by solving for multiple a values

# In[11]:


aList = [1 - i / 5 for i in range(6)]


# In[12]:


for a in aList:
    poisson_solver.constitutive_model.Parameters.diffusivity = a + (1 - a) * k_fn
    poisson_solver.solve()


# Plotting the numerical solution

# In[14]:


with mesh.access():
    mesh_numerical_soln = uw.function.evaluate(poisson_solver.u.fn, mesh.data)


# In[15]:


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
    pvmesh.point_data["phiNumerical"] = mesh_numerical_soln
    sargs = dict(interactive=True)  # doesn't appear to work :(
    pl = pv.Plotter()
    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="phiNumerical",
        use_transparency=False,
        opacity=0.5,
        scalar_bar_args=sargs,
    )
    pl.camera_position = "xy"
    pl.show(cpos="xy")


# The analytical solution here is $\phi = x^2 + y^2$. Lets plot that.

# In[18]:


x, y = mesh.X ## access the symbolic repressentation of the coordinates
analytic_fn = x ** 2 + y ** 2 ## symbolic repressentation 


# In[19]:


with mesh.access():
    mesh_analytic_soln = uw.function.evaluate(analytic_fn, mesh.data)


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

    pvmesh.point_data["phiNumerical"] = mesh_numerical_soln

    sargs = dict(interactive=True)  # doesn't appear to work :(
    pl = pv.Plotter()

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="phiNumerical",
        use_transparency=False,
        opacity=0.5,
        scalar_bar_args=sargs,
    )

    pl.camera_position = "xy"

    pl.show(cpos="xy")


# In[ ]:




