# %%
from petsc4py import PETSc
import underworld3 as uw
from underworld3.systems import Stokes
import numpy as np
import sympy
from mpi4py import MPI

import math

import matplotlib.pyplot as plt


# %%
sys = PETSc.Sys()
sys.pushErrorHandler("traceback")

# Set the resolution.
res = 32

xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0

pipe_thickness = 0.4
velocity = 0

mesh = uw.meshing.StructuredQuadBox(
    elementRes=(int(res), int(res)), minCoords=(xmin, ymin), maxCoords=(xmax, ymax)
)

# Set some values of the system

k = 0.00001 # diffusive constant

tmin = 0.5 # temp min
tmax = 1.0 # temp max

# Create an adv
V = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)

swarm = uw.swarm.Swarm(mesh=mesh, recycle_rate=20)

T_star = uw.swarm.SwarmVariable('Ts', swarm, 1, proxy_degree=T.degree, proxy_continuous=True)

swarm.populate(fill_param=2)

adv_diff = uw.systems.AdvDiffusionSwarm(mesh,
    u_Field=T,
    V_Field=V,
    u_Star_fn=T_star.sym
)

adv_diff.constitutive_model = uw.systems.constitutive_models.DiffusionModel(mesh.dim)
adv_diff.constitutive_model.Parameters.diffusivity = k

adv_diff.add_dirichlet_bc(0.5, "Bottom")
adv_diff.add_dirichlet_bc(0.5, "Top")

maxY = T.coords[:, 1].max()
minY = T.coords[:, 1].min()

with mesh.access(T):
    T.data[...] = tmin

    pipePosition = ((maxY - minY) - pipe_thickness) / 2.0

    T.data[
        (T.coords[:, 1] >= (T.coords[:, 1].min() + pipePosition))
        & (T.coords[:, 1] <= (T.coords[:, 1].max() - pipePosition))
    ] = tmax

with swarm.access(T_star):
        T_star.data[...] = (
            T.rbf_interpolate(swarm.data, nnn=1)
    )

with mesh.access(V):
    V.data[:, 0] = 0
    V.data[:, 1] = velocity


### get the initial temp profile
##tempData = uw.function.evaluate(adv_diff.u.fn, sample_points)



step = 0
time = 0.0

nsteps = 5
every = 1

### y coords to sample
sample_y = np.arange(
    mesh.data[:, 1].min(), mesh.data[:, 1].max(), mesh.get_min_radius()
)  ### Vertical profile

### x coords to sample
sample_x = np.zeros_like(sample_y)  ### LHS of box

sample_points = np.empty((sample_x.shape[0], 2))
sample_points[:, 0] = sample_x
sample_points[:, 1] = sample_y

tempData = uw.function.evaluate(adv_diff.u.fn, sample_points)


round(time, 5)

def diffusion_1D(sample_points, tempProfile, k, model_dt):
    x = sample_points
    T = tempProfile

    dx = sample_points[1] - sample_points[0]

    dt = 0.5 * (dx**2 / k)

    """ max time of model """
    total_time = model_dt

    """ get min of 1D and 2D model """
    time_1DModel = min(model_dt, dt)

    """ determine number of its """
    nts = math.ceil(total_time / time_1DModel)

    """ get dt of 1D model """
    final_dt = total_time / nts

    for i in range(nts):
        qT = -k * np.diff(T) / dx
        dTdt = -np.diff(qT) / dx
        T[1:-1] += dTdt * final_dt

    return T

fig, ax = plt.subplots(1, int(nsteps/every), figsize=(15,3), sharex=True, sharey=True)

dt_adv = 0.00001

while step < nsteps:    
    delta_t_swarm = 1.0 * adv_diff.estimate_dt()
    delta_t = min(delta_t_swarm, dt_adv)
    phi = min(1.0, delta_t / dt_adv) ## not used

    ### 1D profile from underworld
    t1 = uw.function.evaluate(adv_diff.u.fn, sample_points)

    if uw.mpi.size == 1 and step % every == 0:
        """compare 1D and 2D models"""
        ### profile from UW
        ax[int(step/every)].plot(t1, sample_points[:, 1], ls="-", c="red", label="UW numerical solution")
        ### numerical solution
        ax[int(step/every)].plot(tempData, sample_points[:, 1], ls=":", c="k", label="1D numerical solution")
        ax[int(step/every)].set_title(f'time: {round(time, 5)}', fontsize=8)
        ax[int(step/every)].legend(fontsize=8)

    ### 1D diffusion
    tempData = diffusion_1D(
        sample_points=sample_points[:, 1], tempProfile=tempData, k=k, model_dt=delta_t
    )
    
    ### diffuse through underworld
    ## something wrong here
    adv_diff.solve(timestep=delta_t) ## do the advection-diffusion test

    with swarm.access(T_star):
        T_star.data[...] = (
            T.rbf_interpolate(swarm.data, nnn=1)
    )

    swarm.advection(V.fn, delta_t, corrector=False)
    
    step += 1
    time += delta_t
    
plt.savefig('Transport_evolution_swarm.pdf', bbox_inches='tight', dpi=500)
# -






































