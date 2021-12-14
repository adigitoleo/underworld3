# underworld3

## Documentation

The underworld documentation is in two parts: the user manual / theory manual is a jupyterbook that is built from this repository automatically from the sources in the `Jupyterbook` directory

- https://underworldcode.github.io/underworld3/FrontPage.html

The API documentation is built ... 


## Building

Refer to the Dockerfile for uw3 build instructions.  

For development, building inplace will prob be preferable.  Remove
any existing installations, then run.

```shell
pip install -e .
```

For in place usage, you will need to set an appropriate PYTHONPATH.


## Development milestones

- [ ] Spherical stokes
- [x] Buoyancy drive stokes
- [ ] Advection diffusion
- [ ] High Ra, constant viscosity
- [ ] Highly temp-dep viscosity


### Checklist

Ingredients in achieving the above

[[T](https://github.com/underworldcode/underworld3/blob/master/src/ex1.c#L174)] Topology & Meshing
- [x] spherical annulus - https://github.com/julesghub/cubie
- [x] Cartesian
- [x] Different element types (at least Linear / Quadratic & Hex, Tet)

[[D](https://github.com/underworldcode/underworld3/blob/master/src/ex1.c#L268)] Disc 
- [x] Cont Galerkin 
- [ ] ~Disc Galerkin~
- [ ] Semi-lagrangian
- [ ] Free-slip BC on surface

[[P](https://github.com/underworldcode/underworld3/blob/master/src/ex1.c#L73)] Physics
- [x] Stokes-Boussinesq
- [x] Temp-dep rheology
- [ ] Buoyancy driven convection
- [ ] Non-linear viscosity (Jacobian ?) and yielding in particular
- [ ] Viscoelasticity
- [ ] Energy equation, resolve bdry layers
- [ ] kermit the 🐸 

[[S](https://github.com/underworldcode/underworld3/blob/master/src/ex1.c#L354)] Solvers
- [ ] Block Stokes solvers
- [ ] Semi-lagrangian
- [ ] ~TS~  (address this later)

PIC for composition
- [x] Viscosity, buoyancy, ... 
- [ ] Nearest neighbour (k-d tree ? 🌳 )
- [x] 2D - L2 projection into FEM space (Petsc shall provide)
- [ ] 3D - L2 projection into FEM space (Petsc shall provide but not in 3D)

[[O1](https://github.com/underworldcode/underworld3/blob/master/src/ex1.c#L218) [O2](https://github.com/underworldcode/underworld3/blob/master/src/ex1.c#L382)] Output
- [ ] HDF5 -> XDMF -> Paraview
- [ ] LavaVu  

[[V](https://github.com/underworldcode/underworld3/blob/master/src/ex1.c#L35)] Exact solutions
- [ ] MMS
- [ ] Analytical 
  - https://www.solid-earth-discuss.net/se-2017-71/se-2017-71.pdf
  -https://www.researchgate.net/publication/304784132_Benchmark_solutions_for_Stokes_flows_in_cylindrical_and_spherical_geometry



### Tasks

  - [ ] Solver options - robust for viscosity contrasts, customisable and quick.
  - [ ] Investigate generalising context managers. LM - example of mesh vars dependent on functions.
  - [ ] Proper quadratice mesh interpolations for deformed meshes.
  - [ ] DMLabels for higher order meshes, ie. using a label to set values in a Vec. How do you label mid-points?
  - [ ] Further integrals/reduction operators on fields variables.
  - [ ] nKK nanoflann exposure.
  - [ ] create developer docs for software stack and general development strategy.
