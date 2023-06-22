#!/usr/bin/env python
# coding: utf-8

# # Emulating steady state boundary layer growth

# In[1]:


import underworld3 as uw
import sympy
import mpi4py


# In[2]:


U = 1 ## velocitiy of the main flow


# In[3]:


mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0,0), maxCoords=(3,1), cellSize=1/10, qdegree=3)

v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=2)

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(mesh.dim)
stokes.constitutive_model.Parameters.viscosity = 0.001


##body_force_function = mesh.vector.divergence(v.sym) * v.sym
body_force_function = (mesh.vector.jacobian(v.sym.T) * v.sym.T).T

stokes.add_dirichlet_bc( (0, 0), "Bottom", (0, 1) )
stokes.add_dirichlet_bc( (U, 0), "Left", (0, 1) )

stokes.bodyforce = body_force_function










# In[4]:


"""
import concurrent.futures
import time

def my_function(a, body_force_function):
    stokes.bodyforce = a * body_force_function
    return stokes.solve(zero_init_guess=False)

def nonLinearSolve(multiStep, start):
    a = start
    while a != 1:
        print("trying a:", a)
        status = ""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(my_function, a, body_force_function)
            try:
                status = future.result(timeout=20)
            except concurrent.futures.TimeoutError:
                print("timed out")
                status = "timeout"
        print("here is status:", status)
        if status == "timeout":
            a = a / (1 + multiStep)
            multiStep = multiStep / 2
        elif status < 0:
            a = start
            multiStep = multiStep / 2
        elif status >= 0:
            a = a * (1 +  multiStep)
        if a > 1:
            a = 1






import signal
import time

def timeout_handler(num, stack):
    print("Received SIGALRM")
    raise Exception("TIMEOUT")



def nonLinearSolve(multiStep, start):
    a = start
    while a != 1:
        print("trying a:", a)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)
        stokes.bodyforce = a * body_force_function
        status = ""
        try:
            status = stokes.solve()
        except Exception as ex:
            print("timed out")
            status = "timeout"
        finally:
            signal.alarm(0)
        print("here is status:", status)
        if (status == "timeout"):
            ## do nothing
            a = a/(1 + multiStep)
            multiStep = multiStep/2
        elif (status < 0):
            a = start
            multiStep = multiStep/2
        elif (status >= 0):
            a = a * (1 +  multiStep)

        if (a > 1):
            a = 1
    """
    
        

        
        


# In[5]:


for a in [0, 0.0001, 0.001, 0.0005, 0.01, 0.1, 1]:
    stokes.bodyforce = -a * body_force_function
    stokes.solve(zero_init_guess=False)
    


# In[ ]:


##nonLinearSolve(9, 0.00001)


# In[ ]:


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
    arrow_loc = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_loc[:, 0:2] = stokes.u.coords[...]
    arrow_length = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_length[:, 0] = uw.function.evaluate(stokes.u.sym[0], stokes.u.coords)*0.05
    arrow_length[:, 1] = uw.function.evaluate(stokes.u.sym[1], stokes.u.coords)*0.05
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
    
    pl.screenshot("flowInPipe.png")
    ##pl.show(cpos="xy")


# In[ ]:




