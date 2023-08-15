"""

File for testing saving and loading swarms.
"""
import os

os.environ['UW_TIMING_ENABLE'] = "1"

import underworld3 as uw
import numpy
import sympy
import random
from mpi4py import MPI
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
print(size)
print(rank)


def setup():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0,0), maxCoords = (1,1), cellSize=0.1, qdegree=3)

    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)


    swarm = uw.swarm.Swarm(mesh = mesh, recycle_rate=10)
    
    v_star = uw.swarm.SwarmVariable("Vs", swarm, mesh.dim, proxy_degree=2, proxy_continuous=True)
    
    ns = uw.systems.NavierStokesSwarm(
        mesh,
        velocityField=v,
        pressureField=p,
        velocityStar_fn = v_star.sym
            )
    ns.add_dirichlet_bc( (0,0), "Bottom", (0,1))
    ns.add_dirichlet_bc( (0,0), "Left", (0,1))
    ns.add_dirichlet_bc( (0,0), "Top", (0,1) )
    ns.add_dirichlet_bc( (0,0), "Right",(0,1) )

    ns.bodyforce = sympy.Matrix([0,0])
    ns.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(mesh.dim)
    ns.constitutive_model.Parameters.viscosity = 1
    ns.saddle_preconditioner = 1.0/1
    ns.rho = 1.0

    d = dict()
    d['mesh'] = mesh
    d['ns'] = ns
    d['v'] = v
    d["p"] = p
    d['v_star'] = v_star
    d['swarm'] = swarm

    return d

def populateVStar(d):
    swarm = d['swarm']
    v_star = d['v_star']
    with swarm.access(v_star):
        for index in range(len(v_star.data)):
            v_star.data[index,0] += 1

        
def saveAll(d):
    d['swarm'].save_checkpoint(
        swarmName="swarm",
        swarmVars=[d['v_star']],
        index = 0
            )

def solveStep(d):
    dt = 0.001


    for index in range(10):
        d['ns'].solve(timestep=dt)
        with d['swarm'].access(d['v_star']):
            d['v_star'].data[...] = d['v'].rbf_interpolate(d['swarm'].data)
        
        d['swarm'].advection(d['v'].fn, dt, corrector=True)
        print()
    
def reload():
    d = setup()
    ##d['swarm'].load(filename="swarm-0000.h5")
    d['swarm'].populate(fill_param=2)
    d['v_star'].load(filename="Vs-0000.h5", swarmFilename = "swarm-0000.h5")
    #d['swarm'].load(filename="swarm-0000.h5")
    return d


if (True):
    uw.timing.start()
    d = setup()
    d['swarm'].populate(fill_param=2)
    with d['swarm'].access(d['v_star']):
        print(d['v_star'].data[...])
    with d['swarm'].access():
        print(d['swarm'].data.shape)
    solveStep(d)
    uw.timing.stop()
    uw.timing.print_table()
    saveAll(d)

    print("saved and starting to load")
    with d['swarm'].access(d['v_star']):
        print(d['v_star'].data[...])
else:
    d = reload()
    solveStep(d)
    
