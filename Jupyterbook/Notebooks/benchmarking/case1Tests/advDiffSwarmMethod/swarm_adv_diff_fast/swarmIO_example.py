# %% [markdown]
# # swarm IO example

# %%
from petsc4py import PETSc
import underworld3 as uw
from underworld3.systems import Stokes
import numpy as np
import sympy
from mpi4py import MPI

import os

# from underworld3.utilities import generateXdmf, swarm_h5, swarm_xdmf

# %%
outputPath = './output/swarmTest/'


if uw.mpi.rank == 0:
    # checking if the directory demo_folder 
    # exist or not.
    if not os.path.exists(outputPath):

        # if the demo_folder directory is not present 
        # then create it.
        os.makedirs(outputPath)

# %% [markdown]
# ### Create mesh

# %%
# mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0,0.0), 
#                                               maxCoords=(1.0,1.0), 
#                                               cellSize=1.0/res, 
#                                               regular=True)

# mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(xmin, ymin), maxCoords=(xmax, ymax), cellSize=1.0 / res, regular=False)


mesh = uw.meshing.StructuredQuadBox(elementRes =(int(32),int(32)),
                                    minCoords=(0,0), 
                                    maxCoords=(1,1))

if uw.mpi.rank == 0:
    print('finished mesh')


# %% [markdown]
# # Setup initial swarm

# %%
swarm     = uw.swarm.Swarm(mesh=mesh)

# %%
# material  = uw.swarm.IndexSwarmVariable("M", swarm, indices=2, proxy_continuous=False, proxy_degree=0)
# material  = uw.swarm.IndexSwarmVariable("material", swarm, indices=2)

material      = swarm.add_variable(name="materialVariable", size=1, dtype=PETSc.IntType)

# test0      = swarm.add_variable(name="test0", num_components=1, dtype=PETSc.RealType)
# test1      = swarm.add_variable(name="test1", num_components=2, dtype=PETSc.RealType)

rank      = swarm.add_variable(name="rank", size=1, dtype=PETSc.RealType)

if uw.mpi.rank == 0:
    print('add swarm vars')

# test2      = swarm.add_variable(name="test2", num_components=1, dtype=)
# test3      = swarm.add_variable(name="test3", num_components=2, dtype=np.float64)

swarm.populate(2)

if uw.mpi.rank == 0:
    print('populate swarm')

# %% [markdown]
# #### Modify variables

# %%
with swarm.access(rank):
    rank.data[:] = uw.mpi.rank

# %% [markdown]
# #### create a block at the base of the model

# %%
for i in [material]:
        with swarm.access(i):
            i.data[:] = 0
            i.data[(swarm.data[:,1] <= 0.1) & 
                  (swarm.data[:,0] >= (((1 - 0) / 2.) - (0.1 / 2.)) ) & 
                  (swarm.data[:,0] <= (((1 - 0) / 2.) + (0.1 / 2.)) )] = 1

if uw.mpi.rank == 0:
    print('finished updating material')


# %% [markdown]
# ### Save the swarm fields

# %%
swarm.save(outputPath + 'swarm.h5')

# %%
material.save(outputPath + 'material.h5')

# %%
rank.save(outputPath + 'rank.h5')

# %%
swarm.petsc_save_checkpoint('swarm', index=0, outputPath=outputPath)

# %% [markdown]
# #### Create second mesh and swarm to reload data

# %%
mesh1 = uw.meshing.StructuredQuadBox(elementRes =(int(32),int(32)),
                                    minCoords=(0,0), 
                                    maxCoords=(1,1))

if uw.mpi.rank == 0:
    print('finished mesh')

# %%
swarm1     = uw.swarm.Swarm(mesh=mesh1)

material1      = swarm1.add_variable(name="materialVariable", size=1, dtype=PETSc.IntType)

# test0      = swarm.add_variable(name="test0", num_components=1, dtype=PETSc.RealType)
# test1      = swarm.add_variable(name="test1", num_components=2, dtype=PETSc.RealType)

rank1      = swarm1.add_variable(name="rank", size=1, dtype=PETSc.RealType)


rank2      = swarm1.add_variable(name="rank_2ndSwarm", size=1, dtype=PETSc.RealType)

# %%
swarm1.load(outputPath + 'swarm.h5')

# %%
material1.load(filename = outputPath + 'material.h5', swarmFilename = outputPath + 'swarm.h5')

# %%
rank1.load(filename = outputPath + 'rank.h5', swarmFilename = outputPath + 'swarm.h5')

# %%
with swarm1.access(rank2):
    rank2.data[:] = uw.mpi.rank

# %%
swarm1.petsc_save_checkpoint('swarm1', index=0, outputPath=outputPath)

# %% [markdown]
# #### Check values are the same on both meshes
#
# only works in serial

# %%
if uw.mpi.size == 1:
    with swarm.access() and swarm1.access():
        swarmCoordsClose = np.allclose(swarm.data, swarm1.data)
    
    print(f'is rank swarm reloaded correctly?: {swarmCoordsClose}')

# %%
if uw.mpi.size == 1:

    with swarm.access(rank) and swarm1.access(rank1):
        rankClose = np.allclose(rank.data, rank1.data)
    
    print(f'is rank swarm variable reloaded correctly?: {rankClose}')

# %%
if uw.mpi.size == 1:
    with swarm.access(material) and swarm1.access(material1):
        materialClose = np.allclose(rank.data, rank1.data)
    
        print(f'is rank swarm variable reloaded correctly?: {materialClose}')

# %%

# %%
