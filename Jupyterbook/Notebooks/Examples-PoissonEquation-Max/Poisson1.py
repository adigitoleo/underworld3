#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8


# # Solving the Poisson Equation in underworld 3 (Part 1)<br>
# <br>
# This notebook shows how to solve the Poisson equation $\nabla \cdot (k \nabla \phi) = S$ for a constant $k$ and non-zero source term $S$ with Dirichlet boundary conditions in a 2-dimensional Cartesian domain.

# In[13]:

# In[2]:


import underworld3 as uw  # import underworld
from mpi4py import MPI  # library for displaying
import sympy  # for generating symbolic expressions


# Here, we solve the equation $\nabla^2 \phi = \sin(2 \pi x) \cos(2 \pi y)$ in the domain $[0,1]\times[0,1]$ with boundary conditions $\phi(0,y)=\phi(1,y)=0$ and $\phi(x,0) = \phi(x,1)= - \frac{\sin(2 \pi x)}{8 \pi^2}$.

# We discretize our domain using a mesh. There are several types of mesh in underworld 3 - see 'underworld3.meshing' or documentation [here](https://underworldcode.github.io/underworld3/development_api/meshing.html). Here UnstructuredSimplexBox generates a 2-dimensional mesh with vertices at coordinates (0,0) and (1,1). It discretizes the domain into cells of size $\frac{1}{24}$ each with a quadrature degree of 5.

# In[5]:

# In[3]:


mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / (24), qdegree=5)


# On our mesh, we define mesh variables. Here, we create a mesh variable phi, on our, with dimension 1 of degree 2 (The degree is the number of continuous derivatives this variable admits within each element of the mesh).

# In[7]:

# In[4]:


phi = uw.discretisation.MeshVariable(r"\phi", mesh, 1, degree=2)


# Underworld 3 uses a set of templates to solve standard problems in geophysics. Here, we use the Poisson solver to solve equations of the form $\nabla \cdot (k \nabla \phi) = S$. Other solvers available in underworld 3 can be found [here](https://underworldcode.github.io/underworld3/development_api/systems/solvers.html). We define our Poisson solver 'poisson_solver' on our mesh with our scalar mesh variable phi.

# In[8]:

# In[5]:


poisson_solver = uw.systems.Poisson(mesh, phi)


# Each solver in underworld 3 contains a set of parameters - in the Poisson solver, this is the diffusivity $k$. To specify these parameters, underworld 3 uses constitutive models. These relate derivatives of quantities to fluxes or other quantities in the system. In the Poisson solver, a diffusion model is used, which relates the flux $F$ and the derivatives of the scalar field $\nabla \phi$ through the diffusivity $k$ according to $\vec{F} = k \nabla \phi$. The Poisson solver then solves $\nabla \cdot \vec{F} = S$. Other constitutive models are available and can be found [here](https://underworldcode.github.io/underworld3/development_api/systems/constitutive_models.html#underworld3.systems.constitutive_models.Constitutive_Model). To solve our problem, we need to set our diffusivity to $k=1$ and set the source term to $S = \sin(2 \pi x) \cos(2 \pi y)$.

# In[10]:

#  giving our poisson_solver a diffusion model

# In[6]:


poisson_solver.constitutive_model = uw.systems.constitutive_models.DiffusionModel(mesh.dim)
## setting the diffusivity to 1.
poisson_solver.constitutive_model.Parameters.diffusivity = 1


# To set our source term $S$, we must make it a function of the coordinates of the mesh, x and y.

# In[16]:

# In[7]:


x, y = mesh.X  # access symbolic representations of the mesh coordinates x and y
source_fn = sympy.sin(2 * sympy.pi * x) * sympy.cos(2 * sympy.pi * y)  # write our symbolic source function
poisson_solver.f = -source_fn  # set our source function


# Now, we specify our boundary conditions on our scalar variable.

# In[17]:

# In[8]:


top_boundary = -sympy.sin(2 * sympy.pi * x) / (8 * sympy.pi ** 2)
bottom_boundary = -sympy.sin(2 * sympy.pi * x) / (8 * sympy.pi ** 2)


# In[9]:


poisson_solver.add_dirichlet_bc(0, "Left")
poisson_solver.add_dirichlet_bc(0, "Right")
poisson_solver.add_dirichlet_bc(top_boundary, "Top")
poisson_solver.add_dirichlet_bc(bottom_boundary, "Bottom")


# Solve our system using our poisson_solver.

# In[18]:

# In[10]:


poisson_solver.solve()


# By accessing the mesh, we can evaluate our poisson_solver on the mesh and store the values for the scalar field $\phi$ as a numpy array in a numpy array mesh_numerical_soln.

# In[21]:

# In[11]:


with mesh.access():  # access the mesh
    ## Get our
    mesh_numerical_soln = uw.function.evaluate(poisson_solver.u.fn, mesh.data)


# Then, let's plot our numerical solution.

# In[23]:

# In[12]:


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
    sargs = dict(interactive=True)  # doesn't appear to work :(
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


# The analytic solution is $\phi = - \frac{\sin(2 \pi x) \cos(2 \pi y)} {8 \pi^2}$. We can write that symbolically, evaluate it on the mesh, and plot it.

# In[24]:

#  Write out the analytic solution symbolically using sympy.

# In[13]:


analytic_fn = -sympy.cos(2 * sympy.pi * y) * sympy.sin(2 * sympy.pi * x) / (8 * sympy.pi ** 2)


# Evaluate the analytic function on the mesh.

# In[26]:

# In[14]:


with mesh.access():
    mesh_analytic_soln = uw.function.evaluate(analytic_fn, mesh.data)


# Print out the analytic solution.

# In[27]:

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
    pvmesh.point_data["phiAnalytic"] = mesh_analytic_soln
    sargs = dict(interactive=True)  # doesn't appear to work :(
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

