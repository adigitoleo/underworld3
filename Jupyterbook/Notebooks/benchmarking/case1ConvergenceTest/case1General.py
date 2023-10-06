"""  

Case 1 code from the Blenkenbach paper A benchmark comparison of mantle 
convection codes




"""

## IMPORTS 

import argparse
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set parameters for the program")
    # Add arguments
    parser.add_argument("--res", type=int, default=12, help="Resolution (int)")
    parser.add_argument("--save_every", type=int, default=1, help="Save frequency (int) (default is 1 and will take up a lot of space)")
    parser.add_argument("--Ra", type=float, default=216000, help="Rayleigh number")
    parser.add_argument("--restart", type=lambda x: (str(x).lower() == 'true'), default=False, help="Restart from the initial state")
    parser.add_argument("--nsteps", type=int, default=100, help="Number of steps to iterate the program, recomended to be kept under 100 due to RAM constraints")
    parser.add_argument("--temperatureIC", type=float, default=None, help="temperature of starting condition")
    parser.add_argument("--indir", type=str, default=None, help='If restarting, the program will continue on from the initial state located in the directoru indir')
    parser.add_argument("--TDegree", type=int, default=3, help='Degree of temperature')
    parser.add_argument("--speedUp", type=int, default=1, help="speed up factor (do not use for accurate computation)")
    parser.add_argument("--width", type=int, default=1, help="width parameter of mask functions for integral evaluation")
    parser.add_argument("--qdegree", type=int, default=4, help="quadrature degree of mesh")
    parser.add_argument("--stoppingTime", type = float, default = None, help = 'Time to stop simulation')
    parser.add_argument("--useSwarm", type=lambda x: (str(x).lower() == 'true'), default=False, help="Use the explicit swarm method?")

    args = parser.parse_args()

    ## set variables input by user
    Ra = args.Ra
    res = args.res
    save_every = args.save_every
    restart = args.restart
    nsteps = args.nsteps
    tempMax = args.temperatureIC
    indir = args.indir
    TDegree = args.TDegree
    speedUp = args.speedUp
    width = args.width 
    qdegree = args.qdegree
    stoppingTime = args.stoppingTime
    useSwarm = args.useSwarm


    if tempMax is None:
        if restart:
            print("temperatureIC required!, when restart is true. Killing program.")
            quit()
        else:
            tempMax = 1.0

    print(f"Received values:\n"
        f"Res: {res}\n"
        f"Save_every: {save_every}\n"
        f"Ra: {Ra}\n"
        f"Restart: {restart}\n"
        f"Nsteps: {nsteps}\n"
        f"TemperatureIC: {tempMax}\n"
        f"Indir: {indir}\n"
        f"TDegree: {TDegree}\n"
        f"SpeedUp: {speedUp}\n"
        f"Width: {width}\n"
        f"Qdegree: {qdegree}\n"
        f"StoppingTime: {stoppingTime}\n"
        f"UseSwarm: {useSwarm}")


    ## outdir is the path where the outputs of the program are stored
    if (useSwarm == False):
        outdir = f"./general_res{res}_Ra{Ra}_saveEvery{save_every}_temp{tempMax}_TDegree{TDegree}_speedUp{speedUp}_width{width}_qdegree{qdegree}"
    else:
        outdir = f"./general_swarm_res{res}_Ra{Ra}_saveEvery{save_every}_temp{tempMax}_TDegree{TDegree}_speedUp{speedUp}_width{width}_qdegree{qdegree}"        
    outfile = f"{outdir}/output"

    tmpdir =   outdir + "/tmp"      
    resultsdir = outdir + "/results"
    savesdir = outdir + "/saves"     
    print(f"Output directory: {outdir}")
    print(f"Output file: {outfile}")    

fill_param = 5
# ### Set parameters to use 
k = 1.0 #### diffusivity     

boxLength = 1.0 
boxHeight = 1.0

tempMin   = 0.

viscosity = 1

tol = 1e-5

VDegree = 2
PDegree = 1




## lets design this a bit better
if (restart == True):
    indir = None 
    ## set all the parameters to start the solve step fresh
    prev_res = res
    timeVal =  []    # time values
    vrmsVal =  []    # v_rms values 
    nuVal =  []      # Nusselt number values
    start_step = 0   # index of starting step
    t_step = start_step 
    time = 0         # time of starting step

if (restart == False):
    if (indir == None):
        print("WARNING: Assuming restart state located in " + outdir)
        indir = outdir
    ## setting indirectories
    intmpdir =   indir + "/tmp"
    inresultsdir = indir + "/results"
    insavesdir = indir + "/saves"
    
    ## Now, lets load in everything from the previous run here
    with open(intmpdir+"/res.pkl", "rb") as f:
        prev_res = pickle.load(f)

    with open(intmpdir +"/markers.pkl", 'rb') as f:
        loaded_data = pickle.load(f)
        timeVal = loaded_data[0]
        vrmsVal = loaded_data[1]
        nuVal = loaded_data[2]

    with open(intmpdir + "/step.pkl", 'rb') as f:
        start_step = pickle.load(f)
        t_step = start_step
    
    with open(intmpdir + "/time.pkl", "rb") as f:
        time = pickle.load(f)

if (time > stoppingTime):
    print("TIME GREATER THAN STOPPING TIME, CLOSING PROGRAM")
    quit()



from mpi4py import MPI
import math
import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function
import math, sympy
import os 
import numpy as np
import sympy
from copy import deepcopy 

import matplotlib.pyplot as plt
from underworld3.utilities import generateXdmf
comm = MPI.COMM_WORLD


if uw.mpi.rank == 0:
    os.makedirs(outdir, exist_ok = True)
    os.makedirs(tmpdir, exist_ok = True)
    os.makedirs(resultsdir, exist_ok = True)
    os.makedirs(savesdir, exist_ok = True)


## setup the mesh
meshbox = uw.meshing.UnstructuredSimplexBox(
                                                minCoords=(0.0, 0.0), 
                                                maxCoords=(boxLength, boxHeight), 
                                                cellSize = 1.0 / res,
                                                qdegree = qdegree
                                        )
## the mesh variables
v_soln = uw.discretisation.MeshVariable("U", meshbox, meshbox.dim, degree=VDegree) # degree = 2
p_soln = uw.discretisation.MeshVariable("P", meshbox, 1, degree=PDegree) # degree = 1
t_soln = uw.discretisation.MeshVariable("T", meshbox, 1, degree=TDegree) 
## t_0 is the initial state variable
t_0 = uw.discretisation.MeshVariable("T0", meshbox, 1, degree=TDegree) # degree = 3

if (useSwarm):
    swarm = uw.swarm.Swarm(mesh = meshbox, recycle_rate=0)
    t_soln_star = uw.swarm.SwarmVariable("Ts", swarm, 1, proxy_degree=TDegree, proxy_continuous=True)

## set up projection for dTdz, this is used only to compute the Nusselt number
dTdz = uw.discretisation.MeshVariable("dTdz", meshbox, 1, degree = TDegree - 1)
dTdz_projector = uw.systems.Projection(meshbox, dTdz)
dTdz_projector.uw_function = meshbox.vector.gradient(t_soln.sym)[1]



## setup the stokes solver
stokes = Stokes(
    meshbox,
    velocityField=v_soln,
    pressureField=p_soln,
    solver_name="stokes",
)

stokes.tolerance = tol

stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(meshbox.dim)

stokes.constitutive_model.Parameters.viscosity=viscosity
stokes.saddle_preconditioner = 1.0 / viscosity



## free slip boudary conditions
stokes.add_dirichlet_bc((0.0,), "Left", (0,))
stokes.add_dirichlet_bc((0.0,), "Right", (0,))
stokes.add_dirichlet_bc((0.0,), "Top", (1,))
stokes.add_dirichlet_bc((0.0,), "Bottom", (1,))



# add the body force term tto the stokes solver
buoyancy_force = Ra * t_soln.sym[0]
stokes.bodyforce = sympy.Matrix([0, buoyancy_force])


# Setup the advection diffusion solver to solve for the temperature
adv_diff = uw.systems.AdvDiffusionSLCN(
    meshbox,
    u_Field=t_soln,
    V_Field=v_soln,
    solver_name="adv_diff",
)

if (useSwarm):
    adv_diff = uw.systems.AdvDiffusionSwarm(
    meshbox,
    u_Field=t_soln,
    V_Field=v_soln,
    u_Star_fn=t_soln_star.sym
)

adv_diff.constitutive_model = uw.systems.constitutive_models.DiffusionModel(meshbox.dim)
adv_diff.constitutive_model.Parameters.diffusivity = k

adv_diff.theta = 0.5

## heated bottom and cooled top
adv_diff.add_dirichlet_bc(1.0, "Bottom")
adv_diff.add_dirichlet_bc(0.0, "Top")

adv_diff.petsc_options["pc_gamg_agg_nsmooths"] = 5


def saveState():
    with open(tmpdir+"/step.pkl", "wb") as f:
        pickle.dump(t_step+1, f)
    with open(tmpdir + "/time.pkl", "wb") as f:
        pickle.dump(time+delta_t, f)
    with open(tmpdir + "/res.pkl", "wb") as f:
        pickle.dump(res, f)
    with open(tmpdir+"/markers.pkl", 'wb') as f:
        pickle.dump([timeVal, vrmsVal, nuVal], f)
    meshbox.write_timestep_xdmf(filename = savesdir+"/savedState", meshVars=[v_soln, p_soln, t_soln], index=-1)
    meshbox.write_timestep_xdmf(filename = savesdir+"/savedState", meshVars=[v_soln, p_soln, t_soln], index=t_step)


def swarmSaveState():
    ## save the mesh, save the mesh variables
    with open(tmpdir+"/step.pkl", "wb") as f:
        pickle.dump(t_step+1, f)
    with open(tmpdir + "/time.pkl", "wb") as f:
        pickle.dump(time+delta_t, f)
    with open(tmpdir + "/res.pkl", "wb") as f:
        pickle.dump(res, f)
    with open(tmpdir+"/markers.pkl", 'wb') as f:
        pickle.dump([timeVal, vrmsVal, nuVal], f)

    swarm.save_checkpoint(
        swarmName="swarm",
        swarmVars=[t_soln_star],
        index=-1,
        force_sequential = True,
        outputPath = savesdir + "/savedState"
    )
    swarm.save_checkpoint(
        swarmName="swarm",
        swarmVars=[t_soln_star],
        index=t_step,
        force_sequential = True,
        outputPath = savesdir + "/savedState"
    )

    meshbox.write_timestep_xdmf(filename = savesdir+"/savedState", meshVars=[v_soln, p_soln, t_soln], index=-1)
    meshbox.write_timestep_xdmf(filename = savesdir+"/savedState", meshVars=[v_soln, p_soln, t_soln], index=t_step)

def swarmLoadState():
    v_soln.read_from_vertex_checkpoint(insavesdir + "/savedState.U.-1.h5", data_name="U")
    p_soln.read_from_vertex_checkpoint(insavesdir + "/savedState.P.-1.h5", data_name="P")
    t_soln.read_from_vertex_checkpoint(insavesdir + "/savedState.T.-1.h5", data_name="T")
    swarm.populate(fill_param=fill_param)
    t_soln_star.load(filename=insavesdir+"/savedStateTs--001.h5", swarmFilename=insavesdir + "/savedStateswarm--001.h5")


def loadState():

    meshbox_prev = uw.meshing.UnstructuredSimplexBox(
                                                            minCoords=(0.0, 0.0), 
                                                            maxCoords=(boxLength, boxHeight), 
                                                            cellSize=1.0/prev_res,
                                                            qdegree = qdegree,
                                                            regular = False
                                                        )

    v_soln_prev = uw.discretisation.MeshVariable("U2", meshbox_prev, meshbox_prev.dim, degree=VDegree) # degree = 2
    p_soln_prev = uw.discretisation.MeshVariable("P2", meshbox_prev, 1, degree=PDegree) # degree = 1
    t_soln_prev = uw.discretisation.MeshVariable("T2", meshbox_prev, 1, degree=TDegree) # degree = 3
    
    v_soln_prev.read_from_vertex_checkpoint(insavesdir + "/savedState.U.-1.h5", data_name="U")
    p_soln_prev.read_from_vertex_checkpoint(insavesdir + "/savedState.P.-1.h5", data_name="P")
    t_soln_prev.read_from_vertex_checkpoint(insavesdir + "/savedState.T.-1.h5", data_name="T")


    with meshbox.access(v_soln, t_soln, p_soln): 
        t_soln.data[:, 0] = uw.function.evaluate(t_soln_prev.sym[0], t_soln.coords)
        p_soln.data[:, 0] = uw.function.evaluate(p_soln_prev.sym[0], p_soln.coords)
        v_coords = deepcopy(v_soln.coords)
        v_soln.data[:] = uw.function.evaluate(v_soln_prev.fn, v_coords)

    meshbox.write_timestep_xdmf(filename = savesdir+"/savedState", meshVars=[v_soln, p_soln, t_soln], index=-1)

    del meshbox_prev
    del v_soln_prev
    del p_soln_prev
    del t_soln_prev


if restart == True:
    if (useSwarm):
        swarm.populate(fill_param=fill_param)
    pertStrength = 0.1
    deltaTemp = tempMax - tempMin

    with meshbox.access(t_soln, t_0):
        t_soln.data[:] = 0.
        t_0.data[:] = 0.

    with meshbox.access(t_soln):
        for index, coord in enumerate(t_soln.coords):
            # print(index, coord)
            pertCoeff = math.cos( math.pi * coord[0]/boxLength ) * math.sin( math.pi * coord[1]/boxLength )
        
            t_soln.data[index] = tempMin + deltaTemp*(boxHeight - coord[1] + pertStrength * pertCoeff)
            t_soln.data[index] = max(tempMin, min(tempMax, t_soln.data[index]))
            
        
    with meshbox.access(t_soln, t_0):
        t_0.data[:,0] = t_soln.data[:,0]
    meshbox.write_timestep_xdmf(filename = savesdir+"/savedState", meshVars=[t_0, t_soln], index=0)
    if (useSwarm):
            with swarm.access(t_soln_star):
                t_soln_star.data[:,0 ] = uw.function.evaluate(t_soln.fn, swarm.data)
    
    ## we need to set the intial state here if it is a swarm
else:
    if (useSwarm == False):
        loadState()
    else:
        swarmLoadState()


## We just set the value of T, now lets go and solve for dTdz
dTdz_projector.solve()


def v_rms(mesh = meshbox, v_solution = v_soln): 
    # v_soln must be a variable of mesh
    v_rms = boxHeight * math.sqrt( 1/(boxHeight * boxLength) * uw.maths.Integral(mesh, v_solution.fn.dot(v_solution.fn)).evaluate())
    return v_rms

def surface_integral(mesh, uw_function, mask_fn):

    calculator = uw.maths.Integral(mesh, uw_function * mask_fn)
    value = calculator.evaluate()

    calculator.fn = mask_fn
    norm = calculator.evaluate()

    integral = value / norm

    return integral

def getNu(mesh = meshbox, dTdz_solution = dTdz, t_solution = t_soln):
    x,z = meshbox.X

    sdev = 0.5*(1/math.sqrt(2*math.log(2)))*(1/res) * 1/width

    up_surface_defn_fn = sympy.exp(-((z - boxHeight)**2)/(2*sdev**2)) # at z = top
    lw_surface_defn_fn = sympy.exp(-(z**2)/(2*sdev**2))               # at z = 0

    top = surface_integral(mesh, dTdz_solution.fn, up_surface_defn_fn)
    bottom = surface_integral(mesh, t_solution.fn, lw_surface_defn_fn)
    rtn = -boxHeight * top/bottom

    return rtn


print("started the time loop")
while t_step < nsteps + start_step:
    if (stoppingTime != None):
        if (time > stoppingTime):
            break;

    ## solve step
    stokes.solve(zero_init_guess=True) # originally True

    delta_t = stokes.estimate_dt() * speedUp # originally 0.5 
    adv_diff.solve(timestep=delta_t, zero_init_guess=False) # originally False

    if (useSwarm):
        with swarm.access(t_soln_star):
                ##a = uw.function.evaluate(t_soln.fn, swarm.data)
                t_soln_star.data[:,0] = uw.function.evaluate(t_soln.fn, swarm.data)
        swarm.advection(v_soln.sym, delta_t = delta_t)


    ## keep dTdz inline with T
    dTdz_projector.solve()
    ## update values
    vrmsVal.append(v_rms())
    nuVal.append(getNu())
    timeVal.append(time)

    ## save values and print them
    if (t_step % save_every == 0 and t_step > 0) or (t_step+1==nsteps) :
        if uw.mpi.rank == 0:
            print("Timestep {}, dt {}, v_rms {}".format(t_step, timeVal[t_step], vrmsVal[t_step]), flush = True)
            print("Saving checkpoint for time step: ", t_step, "total steps: ", nsteps+start_step , flush = True)
            plt.plot(timeVal, vrmsVal)
            plt.savefig(resultsdir + "/vrms.png")
            plt.xlabel("time")
            plt.ylabel("vrms")
            plt.clf()

            plt.plot(timeVal, nuVal)
            plt.savefig(resultsdir + "/nu.png")
            plt.xlabel("time")
            plt.ylabel("Nusselt Number")
            plt.clf()


            if (useSwarm == False):
                saveState()
            else:
                swarmSaveState()


    ## iterate
    time += delta_t
    t_step += 1


## lets plot out the final
if (uw.mpi.rank == 0):
    plt.plot(timeVal, vrmsVal)
    plt.savefig(resultsdir + "/vrms.png")
    plt.clf()

    plt.plot(timeVal, nuVal)
    plt.savefig(resultsdir + "/nu.png")
    plt.clf()




