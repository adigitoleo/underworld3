## program that runs stokes convection in a box heated from the bottom and cooled from
## the top 
import os 
os.environ["UW_TIMING_ENABLE"] = "1"
# %%
import petsc4py
from petsc4py import PETSc
from mpi4py import MPI
import math

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function
import time as timer


import numpy as np
import sympy
from copy import deepcopy 
import pickle
import matplotlib.pyplot as plt

import math, sympy

from underworld3.utilities import generateXdmf
#os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # solve locking issue when reading file

import argparse





def parse_args():
    parser = argparse.ArgumentParser(description="Blasius boundary layer simulation")
    parser.add_argument('-restart', action='store_true', help='Start from previous timestep?')
    return parser.parse_args()

args = parse_args()
restart = args.restart
print(restart)

#os.environ["HDF5"]
comm = MPI.COMM_WORLD
time = 0
if (uw.mpi.rank == 0):
    start = timer.time()

# %% [markdown]
# ### Set parameters to use 
# %%
Ra = 1e4 #### Rayleigh number
k = 1.0 #### diffusivity     

boxLength = 1.0
boxHeight = 1.0
tempMin   = 0.
tempMax   = 1.

viscosity = 1

tol = 1e-5
res = 48
maxRes = 96                        ### x and y res of box
nsteps = 100                ### maximum number of time steps to run the first model 
epsilon_lr = 1e-3              ### criteria for early stopping; relative change of the Vrms in between iterations  

## parameters for case 2 (a):
b = math.log(1000)
c = 0

## choice of degrees for variables
TDegree = 1
PDegree = 1
VDegree = 2

##########
# parameters needed for saving checkpoints
# can set outdir to None if you don't want to save anything
outdir = "./results" 
outfile = outdir + "/output"
save_every = 5



##infile = outdir + "/conv4_run12_" + str(prev_res)    # set infile to a value if there's a checkpoint from a previous run that you want to start from
if (restart):
    infile = None
else:
    infile = outfile

if (infile == None):
    prev_res = res
else:
    with open('res.pkl', 'rb') as f:
        prev_res = pickle.load(f)


def getDifference(oldVars, newVars):
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

if uw.mpi.rank == 0:
    os.makedirs(outdir, exist_ok = True)


# %% [markdown]
# ### Create mesh and variables

# %%
meshbox = uw.meshing.UnstructuredSimplexBox(
                                                minCoords=(0.0, 0.0), 
                                                maxCoords=(boxLength, boxHeight), 
                                                cellSize=1.0 /res,
                                                qdegree = 3
                                        )

# %%
# visualise the mesh if in a notebook / serial

# %%
v_soln = uw.discretisation.MeshVariable("U", meshbox, meshbox.dim, degree=VDegree) # degree = 2
p_soln = uw.discretisation.MeshVariable("P", meshbox, 1, degree=PDegree) # degree = 1
t_soln = uw.discretisation.MeshVariable("T", meshbox, 1, degree=TDegree) # degree = 3
t_0 = uw.discretisation.MeshVariable("T0", meshbox, 1, degree=TDegree) # degree = 3
x, z = meshbox.X


## lets create a swarm variable for velocity
swarm = uw.swarm.Swarm(mesh = meshbox, recycle_rate=0)

t_soln_star = uw.swarm.SwarmVariable("Ts", swarm, 1, proxy_degree=TDegree, proxy_continuous=True)

stokes = Stokes(
    meshbox,
    velocityField=v_soln,
    pressureField=p_soln,
    solver_name="stokes",
)

# try these
if (uw.mpi.size==1):
    print("running the linear solver")
    stokes.petsc_options['pc_type'] = 'lu' # lu if linear
stokes.tolerance = tol


stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(meshbox.dim)

viscosityfn = viscosity*sympy.exp(-b*t_soln.sym[0]/(tempMax - tempMin) + c * (1 - z)/boxHeight)

stokes.constitutive_model.Parameters.viscosity=viscosity
stokes.saddle_preconditioner = 1.0 / viscosity

# Free-slip boundary conditions
stokes.add_dirichlet_bc((0.0,), "Left", (0,))
stokes.add_dirichlet_bc((0.0,), "Right", (0,))
stokes.add_dirichlet_bc((0.0,), "Top", (1,))
stokes.add_dirichlet_bc((0.0,), "Bottom", (1,))


buoyancy_force = Ra * t_soln.sym[0]
stokes.bodyforce = sympy.Matrix([0, buoyancy_force])

# %%
# Create adv_diff object

adv_diff = uw.systems.AdvDiffusionSwarm(
    meshbox,
    u_Field=t_soln,
    V_Field=v_soln,
    u_Star_fn=t_soln_star.sym
)

adv_diff.constitutive_model = uw.systems.constitutive_models.DiffusionModel(meshbox.dim)
adv_diff.constitutive_model.Parameters.diffusivity = k

adv_diff.theta = 0.5

# Dirichlet boundary conditions for temperature
adv_diff.add_dirichlet_bc(1.0, "Bottom")
adv_diff.add_dirichlet_bc(0.0, "Top")

adv_diff.petsc_options["pc_gamg_agg_nsmooths"] = 5


def saveState():
    ## save the mesh, save the mesh variables
    swarm.save_checkpoint(
        swarmName="swarm",
        swarmVars=[t_soln_star],
        index=0
    )
    meshbox.write_timestep_xdmf(filename = "meshvars", meshVars=[v_soln, p_soln, t_soln], index=0)

def loadState(v_soln,p_soln,t_soln,t_soln_star, swarm):
    v_soln.read_from_vertex_checkpoint("meshvars" + ".U.0.h5", data_name="U")
    p_soln.read_from_vertex_checkpoint("meshvars" + ".P.0.h5", data_name="P")
    t_soln.read_from_vertex_checkpoint("meshvars" + ".T.0.h5", data_name="T")
    swarm.populate(fill_param=2)
    t_soln_star.load(filename="Ts-0000.h5", swarmFilename="swarm-0000.h5")


if restart:
    swarm.populate(fill_param=2)
    pertStrength = 0.1
    deltaTemp = tempMax - tempMin

    with meshbox.access(t_soln, t_0):
        t_soln.data[:] = 0.
        t_0.data[:] = 0.

    with meshbox.access(t_soln):
        for index, coord in enumerate(t_soln.coords):
            # print(index, coord)
            pertCoeff = math.cos( math.pi * coord[0]/boxLength ) * math.sin( math.pi * coord[1]/boxLength )
        
            t_soln.data[index] = tempMin + deltaTemp*(boxHeight - coord[1]) + pertStrength * pertCoeff
            t_soln.data[index] = max(tempMin, min(tempMax, t_soln.data[index]))
            
        
    with meshbox.access(t_soln, t_0):
        t_0.data[:,0] = t_soln.data[:,0]
else:
    loadState(v_soln,p_soln,t_soln,t_soln_star, swarm)

def v_rms(mesh = meshbox, v_solution = v_soln): 
    # v_soln must be a variable of mesh
    v_rms = math.sqrt(uw.maths.Integral(mesh, v_solution.fn.dot(v_solution.fn)).evaluate())
    return v_rms

# %%
# function for calculating the surface integral 
def surface_integral(mesh, uw_function, mask_fn):

    calculator = uw.maths.Integral(mesh, uw_function * mask_fn)
    value = calculator.evaluate()

    calculator.fn = mask_fn
    norm = calculator.evaluate()

    integral = value / norm

    return integral

''' set-up surface expressions for calculating Nu number '''
# the full width at half maximum is set to 1/res
sdev = 0.5*(1/math.sqrt(2*math.log(2)))*(1/res) 

up_surface_defn_fn = sympy.exp(-((z - 1)**2)/(2*sdev**2)) # at z = 1
lw_surface_defn_fn = sympy.exp(-(z**2)/(2*sdev**2)) # at z = 0

# %% [markdown]
# ### Main simulation loop

# %%
t_step = 0
time = 0.

if infile == None:
    timeVal =  []    # time values
    vrmsVal =  []  # v_rms values 
    NuVal =  []      # Nusselt number values
    vrmsVal.append(0)
    timeVal.append(0)
else:
    if (uw.mpi.rank == 0):
        with open(infile + "markers.pkl", 'rb') as f:
            loaded_data = pickle.load(f)
            timeVal = loaded_data[0]
            vrmsVal = loaded_data[1]
            NuVal = loaded_data[2]
time = timeVal[-1]

    

print("started the time loop")
delta_t_natural = 1.0e-2


while t_step < nsteps:
    uw.timing.start()
    if (uw.mpi.rank == 0):
        print("starting stokes", str(t_step))
    
    # solve the systems
    
    stokes.solve(zero_init_guess=True)
    delta_t = stokes.estimate_dt()
    delta_t = min(delta_t, delta_t_natural)

    """
    delta_t_adv_diff = 0.5 * adv_diff.estimate_dt()
    
    delta_t = min([delta_t_natural, delta_t_stokes, delta_t_adv_diff])
    print([delta_t_natural, delta_t_stokes, delta_t_adv_diff, delta_t])

    phi = min(1.0, delta_t/delta_t_natural) 
    """

    adv_diff.solve(timestep=delta_t, zero_init_guess=False)
    if (uw.mpi.rank == 0):
        print("done advection diffusion")
    with swarm.access(t_soln_star):
        ##a = uw.function.evaluate(t_soln.fn, swarm.data)
        t_soln_star.data[:,0] = uw.function.evaluate(t_soln.fn, swarm.data)
    if (uw.mpi.rank == 0):
        print("done setting values")
    
    
    """
    with swarm.access(t_soln_star):
        t_soln_star.data[...] = (
            t_soln.rbf_interpolate(swarm.data) 
            # phi * uw.function.evaluate(v_soln.fn, swarm.data)
            ##+ (1.0 - phi) * t_soln_star.data
        )
    """
    
    
    swarm.advection(v_soln.sym, delta_t = delta_t)
    if (uw.mpi.rank == 0):
        print("done swarm advection")

    vrmsVal.append(v_rms())
    time += delta_t
    timeVal.append(time)
    t_step += 1

    # save the state after updating vrms and time
    if (t_step % save_every == 0 and t_step > 0) or (t_step+1==nsteps) :
        if uw.mpi.rank == 0:
            print("Saving checkpoint for time step: ", t_step, "total steps: ", nsteps , flush = True)
            print(timeVal)
            plt.plot(timeVal, vrmsVal)
            plt.title(str(len(vrmsVal))+" " + str(res))
            plt.savefig(outdir + "vrms.png")
            plt.clf()
            print("saving state")
            saveState()

    # Save state and measurements after each complete timestep
    if uw.mpi.rank == 0:
        with open(outfile+"markers.pkl", 'wb') as f:
            pickle.dump([timeVal, vrmsVal, NuVal], f)
    uw.timing.stop()
    uw.timing.print_table()



saveState()

if (uw.mpi.rank == 0):
    plt.plot(timeVal, vrmsVal)
    plt.title(str(len(vrmsVal))+" " + str(res))
    plt.savefig(outdir + "vrms.png")
    plt.clf()
    saveState()

if (uw.mpi.rank == 0):
    end = timer.time()
    print("Time taken: ", str(end - start))

    with open('res.pkl', 'wb') as f:
        pickle.dump(res, f)

