#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8


# # Solving the Poisson Equation in Underworld 3 (Part 5)<br>
# <br>
# This notebook shows how to solve the non-linear Poisson equation $\nabla \cdot (k \nabla \phi) = S$ for a $k$ as a function of the field $\phi$ and non-zero source term $S$ with Dirichlet boundary conditions in a spherical shell domain.

# In[2]:


import underworld3 as uw
from mpi4py import MPI
import sympy


# We solve $\nabla \cdot ((1 + \phi) \nabla \phi) = 6 + 10 x^2 + 10 y^2 + 10 z^2$ in a spherical shell with inner radius $0.1$, outer radius $1$ and boundary conditions of $\phi=0.1$ on the inner boundary and $\phi=1$ on the outer boundary. This is a 3 Dimensional domain, others include 'CubedSphere' and 'SegmentedSphere' - see [here](https://underworldcode.github.io/underworld3/development_api/meshing.html) for more.

# In[3]:


mesh_3D = uw.meshing.SphericalShell(radiusInner=0.1, radiusOuter=1, cellSize=0.1, qdegree=5)


# Similar to the annulus in Poisson4.ipynb we define scalar variable, Poisson solver, constitutive model, source function, and boundary conditions.

#  define our scalar variable on the mesh

# In[4]:


phi = uw.discretisation.MeshVariable("ϕ", mesh_3D, 1, degree=3)


#  define our Poisson solver on our 3-dimensional mesh with our scalar field

# In[5]:


poisson_solver_3D = uw.systems.Poisson(mesh_3D, phi)
## set the constitutive model to be the diffusion model
poisson_solver_3D.constitutive_model = uw.systems.constitutive_models.DiffusionModel(mesh_3D.dim)


#  Set the source function

# In[6]:


x, y, z = mesh_3D.X  # access the symbolic representations of the coordinates
source_fn = 6 + 10 * (x**2 + y**2 + z**2)  # source function
poisson_solver_3D.f = -source_fn  # set the source function


#  Set the diffusivity function

# In[7]:


k_fn = 1 + phi.sym[0]


#  set the Dirichlet boundary conditions

# In[8]:


poisson_solver_3D.add_dirichlet_bc((0.01,), "Lower", (0,))
poisson_solver_3D.add_dirichlet_bc((1,), "Upper", (0,))


# Similar to Poisson4.ipynb, this is a non-linear Poisson equation. We introduce the non-linearity slowly into the system.

# In[9]:


aList = [1 - i / 3 for i in range(4)]
for a in aList:
    poisson_solver_3D.constitutive_model.Parameters.diffusivity = a + (1 - a) * k_fn
    poisson_solver_3D.solve()


# The analytic solution here is $\phi = x^2 + y^2 + z^2$. Let's print out our analytic and numerical solutions.

#  define our analytic functions

# In[10]:


x, y, z = mesh_3D.X  # get the symbolic expressions for the mesh coordinates
analytic_fn = x**2 + y**2 + z**2  # define the analytic function


#  compute our analytic and numerical solutions on the mesh

# In[11]:


with mesh_3D.access():
    mesh_analytic_soln = uw.function.evaluate(analytic_fn, mesh_3D.data, mesh_3D.N)
    mesh_numerical_soln = uw.function.evaluate(poisson_solver_3D.u.fn, mesh_3D.data, mesh_3D.N)


# Print out the analytic solution

# In[12]:


if MPI.COMM_WORLD.size == 1:
    import numpy as np
    import pyvista as pv
    import vtk
    pv.global_theme.background = "white"
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True
    mesh_3D.vtk("mesh_tmp.vtk")
    pvmesh = pv.read("mesh_tmp.vtk")
    with mesh_3D.access():
        pvmesh.point_data["T"] = mesh_analytic_soln
        pvmesh.point_data["T2"] = mesh_numerical_soln
    clipped = pvmesh.clip(origin=(0.001, 0.0, 0.0), normal=(1, 0, 0), invert=True)
    pl = pv.Plotter()
    pl.add_mesh(
        clipped,
        cmap="coolwarm",
        edge_color="Grey",
        show_edges=True,
        scalars="T",
        use_transparency=False,
        opacity=1.0,
    )
    pl.camera_position = "xy"
    pl.show(cpos="xy")
    # pl.screenshot(filename="test.png")


# Print out the numerical solution

# In[13]:


if MPI.COMM_WORLD.size == 1:
    import numpy as np
    import pyvista as pv
    import vtk
    pv.global_theme.background = "white"
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True
    mesh_3D.vtk("mesh_tmp.vtk")
    pvmesh = pv.read("mesh_tmp.vtk")
    with mesh_3D.access():
        pvmesh.point_data["T"] = mesh_analytic_soln
        pvmesh.point_data["T2"] = mesh_numerical_soln
    clipped = pvmesh.clip(origin=(0.001, 0.0, 0.0), normal=(1, 0, 0), invert=True)
    pl = pv.Plotter()
    pl.add_mesh(
        clipped,
        cmap="coolwarm",
        edge_color="Grey",
        show_edges=True,
        scalars="T2",
        use_transparency=False,
        opacity=1.0,
    )
    pl.camera_position = "xy"
    pl.show(cpos="xy")
    # pl.screenshot(filename="test.png")

