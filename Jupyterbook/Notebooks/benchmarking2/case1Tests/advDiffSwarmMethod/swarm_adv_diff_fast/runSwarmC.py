## program that runs stokes convection in a box heated from the bottom and cooled from
## the top 

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

from underworld3.utilities import generateXdmf
#os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # solve locking issue when reading file
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
res = 12
maxRes = 96                        ### x and y res of box
nsteps = 100                 ### maximum number of time steps to run the first model 
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
infile = outfile
# example infile settings: 
# infile = outfile # will read outfile, but this file will be overwritten at the end of this run 
# infile = outdir + "/convection_16" # file is that of 16 x 16 mesh 

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


def saveData(step, outputPath): # from AdvDiff_Cartesian_benchmark-scaled
    
    ### save mesh vars
    fname = f"{outputPath}/mesh_{'step_'}{step:02d}.h5"
    xfname = f"{outputPath}/mesh_{'step_'}{step:02d}.xmf"
    viewer = PETSc.ViewerHDF5().createHDF5(fname, mode=PETSc.Viewer.Mode.WRITE,  comm=PETSc.COMM_WORLD)

    viewer(meshbox.dm)

    ### add mesh vars to viewer to save as one h5/xdmf file. Has to be a PETSc object (?)
    viewer(v_soln._gvec)         # add velocity
    viewer(p_soln._gvec)         # add pressure
    viewer(t_soln._gvec)           # add temperature
    #viewer(density_proj._gvec)   # add density
    # viewer(materialField._gvec)    # add material projection
    #viewer(timeField._gvec)        # add time
    viewer.destroy()              
    generateXdmf(fname, xfname)

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
swarm = uw.swarm.Swarm(mesh = meshbox)

t_soln_star = uw.swarm.SwarmVariable("Ts", swarm, 1, proxy_degree=TDegree, proxy_continuous=True)



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

# stokes.petsc_options["snes_max_it"] = 1000
#stokes.petsc_options["snes_type"] = "ksponly"
stokes.tolerance = tol
#stokes.petsc_options["snes_max_it"] = 1000

# stokes.petsc_options["snes_atol"] = 1e-6
# stokes.petsc_options["snes_rtol"] = 1e-6


#stokes.petsc_options["ksp_rtol"]  = 1e-5 # reduce tolerance to increase speed

# Set solve options here (or remove default values
# stokes.petsc_options.getAll()

#stokes.petsc_options["snes_rtol"] = 1.0e-6
# stokes.petsc_options["fieldsplit_pressure_ksp_monitor"] = None
# stokes.petsc_options["fieldsplit_velocity_ksp_monitor"] = None
# stokes.petsc_options["fieldsplit_pressure_ksp_rtol"] = 1.0e-6
# stokes.petsc_options["fieldsplit_velocity_ksp_rtol"] = 1.0e-2
# stokes.petsc_options.delValue("pc_use_amat")

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

# %% [markdown]
# ### Set initial temperature field 
# 
# The initial temperature field is set to a sinusoidal perturbation. 

# %%
import math, sympy

if infile is None:
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
    ##swarm_prev = uw.swarm.Swarm(mesh=meshbox_prev)


    
    # T should have high degree for it to converge
    # this should have a different name to have no errors
    v_soln_prev = uw.discretisation.MeshVariable("U2", meshbox_prev, meshbox_prev.dim, degree=VDegree) # degree = 2
    p_soln_prev = uw.discretisation.MeshVariable("P2", meshbox_prev, 1, degree=PDegree) # degree = 1
    t_soln_prev = uw.discretisation.MeshVariable("T2", meshbox_prev, 1, degree=TDegree) # degree = 3
    ##t_soln_star_prev = uw.swarm.SwarmVariable("Tsp", swarm_prev, 1, proxy_degree=TDegree, proxy_continuous=True)

    swarm.load(outfile+'swarm.h5')
    
    t_soln_star.load(filename=outfile+"t_soln_star.h5", swarmFilename=outfile+"swarm.h5")





    

    # force to run in serial?
    v_soln_prev.read_from_vertex_checkpoint(infile + ".U.0.h5", data_name="U")
    p_soln_prev.read_from_vertex_checkpoint(infile + ".P.0.h5", data_name="P")
    t_soln_prev.read_from_vertex_checkpoint(infile + ".T.0.h5", data_name="T")

    ## Okay, now I need to read in the swarm here
    ## we will do this later  - right now, lets just try and run the swarm.


    


    #comm.Barrier()
    # this will not work in parallel?
    #v_soln_prev.load_from_h5_plex_vector(infile + '.U.0.h5')
    #p_soln_prev.load_from_h5_plex_vector(infile + '.P.0.h5')
    #t_soln_prev.load_from_h5_plex_vector(infile + '.T.0.h5')

    with meshbox.access(v_soln, t_soln, p_soln):    
        t_soln.data[:, 0] = uw.function.evaluate(t_soln_prev.sym[0], t_soln.coords)
        p_soln.data[:, 0] = uw.function.evaluate(p_soln_prev.sym[0], p_soln.coords)

        #for velocity, encounters errors when trying to interpolate in the non-zero boundaries of the mesh variables 
        v_coords = deepcopy(v_soln.coords)

        v_soln.data[:] = uw.function.evaluate(v_soln_prev.fn, v_coords)

    ## wait, is that wrong?? Should I just take the data from the swarm??
    ## I dont think that is correct, I want to load the swarm using .load
    ##with swarm.access(t_soln_star):
        ##t_soln_star.data[:,0] = uw.function.evaluate(t_soln_prev.sym[0], swarm.data)


    


    meshbox.write_timestep_xdmf(filename = outfile, meshVars=[v_soln, p_soln, t_soln], index=0)

    del meshbox_prev
    del v_soln_prev
    del p_soln_prev
    del t_soln_prev
    


# %% [markdown]
# ### Some plotting and analysis tools 

# %%
# check the mesh if in a notebook / serial
# allows you to visualise the mesh and the mesh variable
'''FIXME: change this so it's better'''

def v_rms(mesh = meshbox, v_solution = v_soln): 
    # v_soln must be a variable of mesh
    v_rms = math.sqrt(uw.maths.Integral(mesh, v_solution.fn.dot(v_solution.fn)).evaluate())
    return v_rms

#print(f'initial v_rms = {v_rms()}')

# %% [markdown]
# #### Surface integrals
# Since there is no uw3 function yet to calculate the surface integral, we define one.  \
# The surface integral of a function, $f_i(\mathbf{x})$, is approximated as:  
# 
# \begin{aligned}
# F_i = \int_V f_i(\mathbf{x}) S(\mathbf{x})  dV  
# \end{aligned}
# 
# With $S(\mathbf{x})$ defined as an un-normalized Gaussian function with the maximum at $z = a$  - the surface we want to evaluate the integral in (e.g. z = 1 for surface integral at the top surface):
# 
# \begin{aligned}
# S(\mathbf{x}) = exp \left( \frac{-(z-a)^2}{2\sigma ^2} \right)
# \end{aligned}
# 
# In addition, the full-width at half maximum is set to 1/res so the standard deviation, $\sigma$ is calculated as: 
# 
# \begin{aligned}
# \sigma = \frac{1}{2}\frac{1}{\sqrt{ 2 log 2}}\frac{1}{res} 
# \end{aligned}
# 

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
    with open(infile + "markers.pkl", 'rb') as f:
        loaded_data = pickle.load(f)
        timeVal = loaded_data[0]
        vrmsVal = loaded_data[1]
        NuVal = loaded_data[2]
time = timeVal[-1]

    

print("started the time loop")
delta_t_natural = 1.0e-4


while t_step < nsteps:
    
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

    
    with swarm.access(t_soln_star):
        t_soln_star.data[:, 0] = uw.function.evaluate(t_soln.sym[0], swarm.data)
    
    
    """
    with swarm.access(t_soln_star):
        t_soln_star.data[...] = (
            t_soln.rbf_interpolate(swarm.data) 
            # phi * uw.function.evaluate(v_soln.fn, swarm.data)
            ##+ (1.0 - phi) * t_soln_star.data
        )
    """
    
    
    swarm.advection(v_soln.sym, delta_t = delta_t)

    # calculate Nusselt number and other stats
    # ...

    # update time and vrms
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
        meshbox.write_timestep_xdmf(filename = outfile, meshVars=[v_soln, p_soln, t_soln], index=0)
        
        ## save the swarm and its variables
        swarm.save(outfile+'swarm.h5') ## save the swarm

        t_soln_star.save(outfile+'t_soln_star.h5') ## save history of temperature

        swarm.petsc_save_checkpoint('swarm', index=0, outputPath=outfile)


    # Save state and measurements after each complete timestep
    if uw.mpi.rank == 0:
        with open(outfile+"markers.pkl", 'wb') as f:
            pickle.dump([timeVal, vrmsVal, NuVal], f)

            



# save final mesh variables in the run 
meshbox.write_timestep_xdmf(filename = outfile, meshVars=[v_soln, p_soln, t_soln], index=0)
## save the swarm and its variables
swarm.save(outfile+'swarm.h5') ## save the swarm

t_soln_star.save(outfile+'t_soln_star.h5') ## save history of temperature

swarm.petsc_save_checkpoint('swarm', index=0, outputPath=outfile)

if (uw.mpi.rank == 0):
    plt.plot(timeVal, vrmsVal)
    plt.title(str(len(vrmsVal))+" " + str(res))
    plt.savefig(outdir + "vrms.png")
    plt.clf()

if (uw.mpi.rank == 0):
    end = timer.time()
    print("Time taken: ", str(end - start))

    with open('res.pkl', 'wb') as f:
        pickle.dump(res, f)
    print("final VRMS: ", str(vrmsVal[-1]))

