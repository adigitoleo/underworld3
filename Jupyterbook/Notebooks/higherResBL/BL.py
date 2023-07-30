"""
Underworld 3 benchmark for the Blasius boundary layer. 
By Maximilian Williams

Setup:
Consider a flow of velocity U in the x-direction encountering an infinite non-slip plate
lying along the x-axis. 

Measurement:
Here, we measure the boundary layer thickness $\delta$ defined by:

$$
\delta = \int_{0}^{\inf} (1 - \frac{u}{U}) dy
$$
where $u$ is the x-component of the flow velocity and $y$ is the vertical coordinate

Analytic Result:
    Assumptions:
        1. at x = 0, the flow has velocity U in the x-direction and 0 in the y-direction
        2. at y = 0 (on the non-slip plate), the x and y components of flow velocity are zero
        3. at $y \rightarrow \inf$ the horizontal component of velocity approaches $U$

From https://en.wikipedia.org/wiki/Blasius_boundary_layer, the boundary layer should be:
$$
\delta(x) \approx 1.72 \sqrt{\frac{\nu x}{U} },
$$
where $\nu$ is the kinematic viscousity.

Simulation:

    Domain:
    We define a domain for simulation with length L and height H. We set the length L and give 
    the height a value $H = 10 \delta(L)$ - this way the top boundary of the domain is atleast 10
    times larger than the boundary layer thickness. 

    Boundary Conditions:
    1. At x = 0, we set the flow velocity to be U in the x-direction and 0 in the y-direction
    2. At y = 0, we set a non-slip condition - the x and y components of flow velocity are zero
    3. At the top boundary (y = H), we set the x-direction flow velocity to U


    Setup:
    1. Set the flow to be U in the x-direction through the whole domain without any boundary conditions
    2. Apply the boundary conditions above and run the simulationsaf
"""

## imports
import os
import petsc4py
import underworld3 as uw
import numpy as np
import sympy
import mpi4py
import matplotlib.pyplot as plt
import copy
import math 
from mpi4py import MPI
import pickle


resultsFolder = "BLResults"

## files
outdir = resultsFolder + "/"

import os

def create_folder_if_not_exists(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
        print(f"Folder '{folder_name}' created successfully.")
    else:
        print(f"Folder '{folder_name}' already exists.")

if (uw.mpi.rank == 0):
    # Main folder name
    main_folder_name = 'BLResults'
    create_folder_if_not_exists(main_folder_name)

    # Sub-folder names
    sub_folders = ['plots', 'data', 'FlowPlots']

    # Creating sub-folders within the main folder
    for folder in sub_folders:
        folder_path = os.path.join(main_folder_name, folder)
        create_folder_if_not_exists(folder_path)


## setup variables for our mesh
boxLength = 1 ## length of our box
##boxHeight = 1.72*(boxLength)**0.5 * 10
boxHeight = 1.72*(boxLength)**0.5 * 1
resolution = 0.1


vel = 1 ## the velocity of the flow on the inlet in the x-direction
viscosity = 1 ## the viscosity of the fluid
if (uw.mpi.rank == 0):
    print(vel * boxLength/viscosity)
    print((2 * viscosity*boxLength/vel)**0.5)


## mesh degree
qdegree = 3

## set the mesh
mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0,0), maxCoords=(boxLength, boxHeight), cellSize=resolution, qdegree=qdegree)

##mesh.dm.view()

## Degrees of the velocity and pressure within the system
VDegree = 2 
PDegree = 1

## define the mesh variables
v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=VDegree)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=PDegree)      

## define the swarm
swarm = uw.swarm.Swarm(mesh=mesh, recycle_rate=20)

## define the swarm variables
v_star = uw.swarm.SwarmVariable("Vs", swarm, mesh.dim, 
                            proxy_degree=VDegree, proxy_continuous=True)  
                            
## populate elements in the mesh with swarm particles
swarm.populate(fill_param=2)  





## set the navier stokes solver
ns = uw.systems.NavierStokesSwarm(
    mesh,
    velocityField = v,
    pressureField = p,
    velocityStar_fn = v_star.sym,
)

## Here we set boundary conditions for a free stream through the domain
ns.add_dirichlet_bc( (vel, 0), "Bottom", (0,1))
ns.add_dirichlet_bc( (vel, 0.0), "Left", (0, 1))
ns.add_dirichlet_bc( (vel, 0.0), "Top", (0,))

## set the body forces
ns.bodyforce = sympy.Matrix([0.0, 0.0])

## add a viscous model
ns.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(mesh.dim)
ns.constitutive_model.Parameters.viscosity = viscosity

ns.saddle_preconditioner = 1.0 / ns.constitutive_model.Parameters.viscosity

## set the density of the fluid to 1
ns.rho = 1

## plotting of the velocity field
def plot(mesh, v, ns, step, path = outdir+"/FlowPlots/"):
    """ 
    Plots the velocity field on the mesh at a timestep
    Inputs:
    mesh: uw mesh variable
    v: 2 dimensional underworld mesh variable repressenting flow velocity
    ns: underworld 3 navier stokes solver
    step: int repressenting the step in the simulation (used for plotting)
    path (optional): Path where plot is saved

    Output:
    File located at nsPlots that shows the velocity of the fluid throughout the 
    simulaton domain 
    """
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
        pl.show(cpos="xy", screenshot = path+str(step)+".png")


def getDifference(oldVars, newVars):
    """
    Gets the difference between new and old variables - used to check for convergence
    Inputs:
    oldVars: list of variables from the previous timestep
    newVars: list of variables from the current timestep
    Outputs:
    float: average of difference between old and new variables
    """
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

def saveState(mesh, swarm, v, p, v_star, differences, index, path=outdir):
    """
    Save the state of the simulation
    """
    swarm.save(path+'swarm.h5') ## save the swarm

    v_star.save(path+'v_star.h5') ## save history of temperature

    swarm.petsc_save_checkpoint('swarm', index=0, outputPath=path)

    mesh.write_timestep_xdmf(filename = path, meshVars=[v, p], index=0)

    with open(path+"differences.pkl", 'wb') as f:
        pickle.dump(differences, f)

    with open(path+"index.pkl", 'wb') as f:
        pickle.dump(index, f)

def loadState(path=outdir):
    """
    Load in the saved system
    """
    try:
        print("checkpoint1")
        v.read_from_vertex_checkpoint(path + ".U.0.h5", data_name="U")
        print("checkpoint2")
        p.read_from_vertex_checkpoint(path + ".P.0.h5", data_name="P")
        print("checkpoint3")
        v_star.load(filename=path+'v_star.h5', swarmFilename=path+"swarm.h5")
        print("checkpoint4")
        ##swarm.load(path+'swarm.h5')
        print("checkpoint5")
        with open(path+"index.pkl", 'rb') as f:
            index = pickle.load(f)
        print("checkpoint6")
        with open(path + "differences.pkl", 'rb') as f:
            differences = pickle.load(f)
        print("checkpoint7")
    except:
        if (uw.mpi.rank == 0):
            print("Was not able to load state")
        index = 0
        differences = []
    if (uw.mpi.rank == 0):
        print("done loading state")

    return index, differences

    
    mesh_prev = uw.meshing.UnstructuredSimplexBox(minCoords=(0,0),
    maxCoords=(boxLength, boxHeight), cellSize=resolution, qdegree=qdegree)

    swarm_prev = uw.swarm.Swarm(mesh=mesh_prev)

    v_prev = uw.discretisation.MeshVariable("U", mesh_prev, mesh_prev.dim, degree=VDegree)
    p_prev = uw.discretisation.MeshVariable("P", mesh_prev, 1, degree=PDegree)   
    v_star_prev = uw.swarm.SwarmVariable("Vs", swarm_prev, mesh_prev.dim, 
                            proxy_degree=VDegree, proxy_continuous=True) 

     

    swarm_prev.load(path+"swarm.h5")

    ##v_star_prev.load(filename=path+"v_star.h5", swarmFilename=path+"swarm.h5")
    v_star.load(filename=path+"v_star.h5", swarmFilename=path+"swarm.h5")
    

    v_prev.read_from_vertex_checkpoint(path + ".U.0.h5", data_name="U")
    p_prev.read_from_vertex_checkpoint(path + ".P.0.h5", data_name="P")

    
    with mesh.access(v, p):

        v.data[:] = uw.function.evaluate(v_prev.fn, v.coords)
        
        p.data[:, 0] =  uw.function.evaluate(p_prev.sym[0], p.coords)

    del mesh_prev
    del swarm_prev
    del v_prev
    del p_prev
    del v_star_prev
    with open(path+"index.pkl", 'rb') as f:
        index = pickle.load(f)
    print("checkpoint6")
    with open(path + "differences.pkl", 'rb') as f:
        differences = pickle.load(f)
    """
    except:
        if (uw.mpi.rank == 0):
            print("Was not able to load state")
        index = 0
        differences = []
    if (uw.mpi.rank == 0):
        print("done loading state")
    """

    return index, differences

def getBL(mesh, v):
    """ 
    Function that gets that boundary layer thickness throughout the domain
    """
    x,y = mesh.X

    stepSize = 1*resolution
    slides = [i*stepSize for i in range(int( boxLength / stepSize))]

    functions = [
        1/stepSize * (vel - v.sym[0]/vel) * sympy.Piecewise(
            (1,  sympy.And( (s < x), (x<=s + stepSize))  ),
            (0, True)
        ) for s in slides
    ]

    integrals = [uw.maths.Integral(mesh=mesh, fn=f) for f in functions]
    results = [i.evaluate() for i in integrals]
    
    return [slides, results]

startIndex, differences = loadState()

ts = 0
dt_ns = 0.01
maxsteps = 100
runSteps = 5
##differences = []
pdifferences = []
blStep = 10
savePeriod = 1


## the simulation time-step
for step in range(startIndex, min(maxsteps, startIndex + runSteps) ) :
    ## make a copy of the velocity field
    ## to use for later in getting the difference
    if (uw.mpi.rank == 0):
        with mesh.access():
            old_v_data = copy.deepcopy(v.data)

    ## solve step
    if (step >= blStep):
        ns.add_dirichlet_bc( (0, 0), "Bottom", (0,1))
        ns.add_dirichlet_bc( (vel, 0.0), "Left", (0, 1) )
        ns.solve(timestep= dt_ns, zero_init_guess=False)
        delta_t_swarm = 1.0 * ns.estimate_dt()
        delta_t = min(delta_t_swarm, dt_ns)
        phi = min(1.0, delta_t/dt_ns)

        with swarm.access(v_star):
            v_star.data[...] = (
                phi * v.rbf_interpolate(swarm.data) + (1.0 -  phi) * v_star.data
            )
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
    


    swarm.advection(v.fn, delta_t, corrector=False)

    print("save step")


    print("Plot and update step")
    ## plot step
    if (uw.mpi.rank == 0):
        plot(mesh, v, ns, step)

    BLData = getBL(mesh, v)
    
    if (uw.mpi.rank == 0):
        ## then lets save it using pickle and plot
        blpath = outdir + "data/"
        with open(blpath+str(step), 'wb') as f:
            pickle.dump(BLData, f)

        blPlotsPath = outdir + "plots/"

        plt.plot(BLData[0], BLData[1])
        plt.savefig(blPlotsPath+str(step)+".png")
        plt.clf()

    if (uw.mpi.rank == 0):
        if (True):
            with mesh.access():
                v_data = v.data
                differences.append(getDifference([old_v_data], [v_data]) )

        with open(outdir + "differences.pkl", 'wb') as f:
            pickle.dump(differences, f)

        plt.plot(differences)
        plt.savefig(outdir+"differences.png")
        plt.clf()
        try:
            logDifferences = [math.log(el) for el in differences]
            plt.plot(logDifferences)
            plt.savefig(outdir+"logDifferences.png")
            plt.clf()
        except:
            print("Converged!")

    ## save step
    saveState(mesh, swarm, v, p, v_star, differences, step, path=outdir)


