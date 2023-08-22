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

import os 
import numpy as np
import sympy
from copy import deepcopy 
import pickle
import matplotlib.pyplot as plt
import argparse

from underworld3.utilities import generateXdmf
#os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # solve locking issue when reading file
#os.environ["HDF5"]
comm = MPI.COMM_WORLD
time = 0
if (uw.mpi.rank == 0):
    start = timer.time()

rank = comm.Get_rank()
print('My rank is ',rank)

# %% [markdown]
# ### Set parameters to use 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set parameters for the program.")

    # Add arguments for resolution (res), save_every, Re (Ra), restart, and nsteps with default values
    parser.add_argument("--res", type=int, default=96, help="Resolution (int)")
    parser.add_argument("--save_every", type=int, default=10, help="Save frequency (int)")
    parser.add_argument("--Ra", type=float, default=1e4, help="Representing Re (float) but saved to 'Ra'")
    parser.add_argument("--restart", type=lambda x: (str(x).lower() == 'true'), default=False, help="Restart (bool)")
    parser.add_argument("--nsteps", type=int, default=100, help="Number of steps (int)")

    args = parser.parse_args()
    # Assign values from args to variables
    Ra = args.Ra
    res = args.res
    save_every = args.save_every
    restart = args.restart
    nsteps = args.nsteps
    # Print the received values (or process them further if needed)
    print(f"Received values:\nRes: {res}\nSave_every: {save_every}\nRa: {Ra}\nRestart: {restart}\nNsteps: {nsteps}")



k = 1.0 #### diffusivity     

boxLength = 1.0
boxHeight = 1.0
tempMin   = 0.
tempMax   = 1.

viscosity = 1

tol = 1e-5
epsilon_lr = 1e-3              ### criteria for early stopping; relative change of the Vrms in between iterations  

## parameters for case 2 (a):
b = math.log(1000)
c = 0

## choice of degrees for variables
TDegree = 3
PDegree = 1
VDegree = 2

##########
# parameters needed for saving checkpoints
# can set outdir to None if you don't want to save anything
outdir = "./results" 
outfile = outdir + "/output"



##infile = outdir + "/conv4_run12_" + str(prev_res)    # set infile to a value if there's a checkpoint from a previous run that you want to start from

if (restart == True):
    infile = None
    prev_res = res
else:
    infile = outfile
    with open('res.pkl', 'rb') as f:
        prev_res = pickle.load(f)
        print("loaded previous res:", str(res))


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


# %% [markdown]
# ### System set-up 
# Create solvers and set boundary conditions

# %%
# Create Stokes object

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

stokes.constitutive_model.Parameters.viscosity=viscosityfn
stokes.saddle_preconditioner = 1.0 / viscosityfn

# Reflective symmetry
# We constrain the normal component of the velocity
stokes.add_dirichlet_bc((0.0,), "Left", (0,))
stokes.add_dirichlet_bc((0.0,), "Right", (0,))

stokes.add_dirichlet_bc((0.0,), "Top", (1,))
stokes.add_dirichlet_bc((0.0,), "Bottom", (1,))

# We constrain the dv/dx = 0 on the boundrie (uw3 does this automatically)

buoyancy_force = Ra * t_soln.sym[0]
stokes.bodyforce = sympy.Matrix([0, buoyancy_force])

# %%
# Create adv_diff object

adv_diff = uw.systems.AdvDiffusionSLCN(
    meshbox,
    u_Field=t_soln,
    V_Field=v_soln,
    solver_name="adv_diff",
)

adv_diff.constitutive_model = uw.systems.constitutive_models.DiffusionModel(meshbox.dim)
adv_diff.constitutive_model.Parameters.diffusivity = k

adv_diff.theta = 0.5

# Dirichlet boundary conditions for temperature
adv_diff.add_dirichlet_bc(1.0, "Bottom")
adv_diff.add_dirichlet_bc(0.0, "Top")

adv_diff.petsc_options["pc_gamg_agg_nsmooths"] = 5

# %% [markdown]
# ### Set initial temperature field 
# 
# The initial temperature field is set to a sinusoidal perturbation. 

# %%
import math, sympy

if restart == True:
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

    #meshbox.write_timestep_xdmf(filename = outfile, meshVars=[v_soln, p_soln, t_soln, dTdZ, sigma_zz], index=0)

    #saveData(0, outdir) # from AdvDiff_Cartesian_benchmark-scaled
else:
    meshbox_prev = uw.meshing.UnstructuredSimplexBox(
                                                            minCoords=(0.0, 0.0), 
                                                            maxCoords=(boxLength, boxHeight), 
                                                            cellSize=1.0/prev_res,
                                                            qdegree = 3,
                                                            regular = False
                                                        )
    
    # T should have high degree for it to converge
    # this should have a different name to have no errors
    v_soln_prev = uw.discretisation.MeshVariable("U2", meshbox_prev, meshbox_prev.dim, degree=VDegree) # degree = 2
    p_soln_prev = uw.discretisation.MeshVariable("P2", meshbox_prev, 1, degree=PDegree) # degree = 1
    t_soln_prev = uw.discretisation.MeshVariable("T2", meshbox_prev, 1, degree=TDegree) # degree = 3

    # force to run in serial?
    v_soln_prev.read_from_vertex_checkpoint(infile + ".U.0.h5", data_name="U")
    p_soln_prev.read_from_vertex_checkpoint(infile + ".P.0.h5", data_name="P")
    t_soln_prev.read_from_vertex_checkpoint(infile + ".T.0.h5", data_name="T")

    with meshbox.access(v_soln, t_soln, p_soln):    
        t_soln.data[:, 0] = uw.function.evaluate(t_soln_prev.sym[0], t_soln.coords)
        p_soln.data[:, 0] = uw.function.evaluate(p_soln_prev.sym[0], p_soln.coords)

        #for velocity, encounters errors when trying to interpolate in the non-zero boundaries of the mesh variables 
        v_coords = deepcopy(v_soln.coords)

        v_soln.data[:] = uw.function.evaluate(v_soln_prev.fn, v_coords)

    meshbox.write_timestep_xdmf(filename = outfile, meshVars=[v_soln, p_soln, t_soln], index=0)


    del meshbox_prev
    del v_soln_prev
    del p_soln_prev
    del t_soln_prev



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


if infile == None:
    timeVal =  []    # time values
    vrmsVal =  []  # v_rms values 
    NuVal =  []      # Nusselt number values
    start_step = 0
    time = 0
else:
    with open(infile + "markers.pkl", 'rb') as f:
        loaded_data = pickle.load(f)
        timeVal = loaded_data[0]
        vrmsVal = loaded_data[1]
        NuVal = loaded_data[2]

    with open(infile+"step.pkl", 'rb') as f:
        start_step = pickle.load(f)
    
    with open(infile+"time.pkl", "rb") as f:
        time = pickle.load(f)

t_step = start_step
 
#### Convection model / update in time

print("started the time loop")
uw.timing.start()
while t_step < nsteps + start_step:

    ## solve step
    stokes.solve(zero_init_guess=True) # originally True

    delta_t = 0.5 * stokes.estimate_dt() # originally 0.5
    adv_diff.solve(timestep=delta_t, zero_init_guess=False) # originally False

    ## update values
    vrmsVal.append(v_rms())
    timeVal.append(time)

    ## save values and print them
    if (t_step % save_every == 0 and t_step > 0) or (t_step+1==nsteps) :
        if uw.mpi.rank == 0:
            print("Timestep {}, dt {}, v_rms {}".format(t_step, timeVal[t_step], vrmsVal[t_step]), flush = True)
            print("Saving checkpoint for time step: ", t_step, "total steps: ", nsteps+start_step , flush = True)
            print(timeVal)
            plt.plot(timeVal, vrmsVal)
            plt.savefig(outdir + "vrms.png")
            plt.clf()
        meshbox.write_timestep_xdmf(filename = outfile, meshVars=[v_soln, p_soln, t_soln], index=0)
        ##meshbox.write_timestep_xdmf(filename = outfile, meshVars=[v_soln, p_soln, t_soln], index=t_step)

    if uw.mpi.rank == 0:
        with open(outfile+"markers.pkl", 'wb') as f:
            pickle.dump([timeVal, vrmsVal, NuVal], f)
    if (uw.mpi.rank == 0):
        with open("res.pkl", "wb") as f:
            pickle.dump(res, f)

    ## iterate

    time += delta_t
    t_step += 1

    ## here is where we will start next ime
    if (uw.mpi.rank == 0):
        with open(outfile+"step.pkl", "wb") as f:
            pickle.dump(t_step, f)

    if (uw.mpi.rank == 0):
        with open(outfile + "time.pkl", "wb") as f:
            pickle.dump(time, f)
    uw.timing.stop()
    uw.timing.print_table()


# save final mesh variables in the run 
meshbox.write_timestep_xdmf(filename = outfile, meshVars=[v_soln, p_soln, t_soln], index=0)
if (uw.mpi.rank == 0):
    plt.plot(timeVal, vrmsVal)
    plt.savefig(outdir + "vrms.png")
    plt.clf()
