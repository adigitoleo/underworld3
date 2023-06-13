#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8


# # Solving the Poisson Equation in underworld 3 (Part 4)<br>
# <br>
# This notebook shows how to solve the non-linear Poisson equation $\nabla \cdot (k \nabla \phi) = S$ for a $k$ as a function of the field $\phi$ and non-zero source term $S$ with Dirichlet boundary conditions in a 2-Dimensional annulus.

# In[2]:


import underworld3 as uw  ## import underworld
from mpi4py import MPI  ## library for displaying
import sympy  ## for generating symbolic expressions


# We solve $\nabla \cdot ((1 + \phi) \nabla \phi) = 4 + 8 x^2 + 8 y^2$ on an annulus with inner radius $0.1$ and outer radius $1$ with boundary conditions of $\phi=0.1$ on the inner boundary and $\phi=1$ on the outer boundary. Unlike the Cartesian examples Poisson[1-4].ipyn, we use an annulus as our domain. Underworld 3 features other geometries - see [here](https://underworldcode.github.io/underworld3/development_api/meshing.html) for more.

# In[3]:


mesh = uw.meshing.Annulus(radiusInner=0.1, radiusOuter=1, cellSize=0.025, qdegree=5)


# Like the Cartesian meshes, we can access the coordinates of the mesh using the X method.

# In[4]:


mesh.X


# The coordinates here are still Cartesian x, y. As in Poisson[1-3].ipynb, we set our solver, source, constitutive model, and boundary conditions.

#  define our scalar variable phi 

# In[5]:


phi = uw.discretisation.MeshVariable(r"\phi", mesh, 1, degree=3)


#  define the Poisson solver

# In[6]:


poisson_solver = uw.systems.Poisson(mesh, phi)


#  set the source term

# In[7]:


x, y = mesh.X 
source_fn = 4 + 8 * x**2 + 8 * y**2
poisson_solver.f = -source_fn


#  set the diffusive model 

# In[8]:


poisson_solver.constitutive_model = uw.systems.constitutive_models.DiffusionModel(mesh.dim)


#  boundary conditions

# In[9]:


bottomCondition = 0.01
topCondition = 1
poisson_solver.add_dirichlet_bc((bottomCondition,), "Lower", (0,))
poisson_solver.add_dirichlet_bc((topCondition,), "Upper", (0,))


# We slowly introduce non-linearity as in Poisson3.ipynb to avoid divergence in the solver.

#  get the symbolic representation of the diffusivity

# In[10]:


k_fn = 1 + phi.sym[0]


# In[11]:


aList = [1 - da/10 for da in range(11)]
for a in aList:
    poisson_solver.constitutive_model.Parameters.diffusivity = a + (1 - a) * k_fn
    poisson_solver.solve()


# Evaluate our numerical solution and plot it

# In[12]:


with mesh.access():    
    mesh_numerical_soln = uw.function.evaluate(poisson_solver.u.fn, mesh.data, mesh.N)


# In[17]:


if MPI.COMM_WORLD.size == 1:
    import numpy as np
    import pyvista as pv
    import vtk
    pv.global_theme.background = "white"
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True
    mesh.vtk("mesh_tmp.vtk")
    pvmesh = pv.read("mesh_tmp.vtk")
    with mesh.access():
        pvmesh.point_data["2DannulusAnalytic"] = mesh_numerical_soln
    pl = pv.Plotter()
    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="2DannulusAnalytic",
        use_transparency=False,
        opacity=0.5,
    )
    pl.camera_position = "xy"
    pl.show(cpos="xy")
    # pl.screenshot(filename="test.png")


# The analytic solution is $\phi = x^2 + y^2$. Evaluate it and plot it.

# In[14]:


x, y = mesh.X  ## access the symbolic representations of the coordinates
analytic_fn = x**2 + y**2  ## the analytic solution


# In[15]:


with mesh.access():
    mesh_analytic_soln = uw.function.evaluate(analytic_fn, mesh.data, mesh.N)


# In[16]:


if MPI.COMM_WORLD.size == 1:
    import numpy as np
    import pyvista as pv
    import vtk
    pv.global_theme.background = "white"
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True
    mesh.vtk("mesh_tmp.vtk")
    pvmesh = pv.read("mesh_tmp.vtk")
    with mesh.access():
        pvmesh.point_data["2DannulusAnalytic"] = mesh_analytic_soln
    pl = pv.Plotter()
    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="2DannulusAnalytic",
        use_transparency=False,
        opacity=0.5,
    )
    pl.camera_position = "xy"
    pl.show(cpos="xy")
    # pl.screenshot(filename="test.png")

