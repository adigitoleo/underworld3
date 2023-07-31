import underworld3 as uw

mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0,0), maxCoords=(1, 1), cellSize=0.1, qdegree=3)

mesh.save("meshFile.h5")

mesh2 = uw.discretisation.Mesh(f"meshFile.h5",
                                  qdegree=3)

print(mesh2.coords)