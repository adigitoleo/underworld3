import h5py
import numpy as np
import matplotlib.pyplot as plt
import underworld3 as uw
import pyvista as pv
import os



import argparse

parser = argparse.ArgumentParser(description='Settings for plotting')
    
    # Define a command line argument named 'input_string'
parser.add_argument('fileName', type=str, help='fileName to plot')
parser.add_argument("--res", type=int, help='resolution of file to plot', default=24)
parser.add_argument('--boxLength', type=int, help="height of box", default = 1)
parser.add_argument('--boxHeight', type=int, help='length of box', default = 1)




args = parser.parse_args()  # Parse the provided command line arguments
    



boxLength = args.boxLength
boxHeight = args.boxHeight
res = args.res
fileName = args.fileName

mesh = uw.meshing.UnstructuredSimplexBox(
                                                        minCoords=(0.0, 0.0), 
                                                        maxCoords=(boxLength, boxHeight), 
                                                        cellSize=1.0/res,
                                                        qdegree = 3,
                                                        regular = False
                                                    )

# T should have high degree for it to converge
# this should have a different name to have no errors

t_soln_prev = uw.discretisation.MeshVariable("T", mesh, 1, degree=1) # degree = 3

t_soln_prev.read_from_vertex_checkpoint(fileName, data_name="T")

with mesh.access():
    data = t_soln_prev.data[...]
print(np.min(data) )

pv.global_theme.background = "white"
pv.global_theme.window_size = [500, 500]
pv.global_theme.antialiasing = True
pv.global_theme.jupyter_backend = "panel"
pv.global_theme.smooth_shading = True
mesh.vtk("ignore_mesh.vtk")
pvmesh = pv.read("ignore_mesh.vtk")
pvmesh.point_data["T"] = data
sargs = dict(interactive=True)  # Doesn't appear to work :(

pl = pv.Plotter(off_screen=True)  # Use off-screen rendering to save image without displaying
pl.add_mesh(
    pvmesh,
    cmap="coolwarm",
    edge_color="Black",
    show_edges=True,
    scalars="T",
    use_transparency=False,
    opacity=0.5,
    scalar_bar_args=sargs,
)
pl.camera_position = "xy"

# Capture the visualization
img_array = pl.screenshot()


# Define the directory name
dir_name = "images"


def onlyName(string):
    index = len(string) - 1
    revrtn = ""
    while index >= 0:
        if (string[index] != "/"):
            revrtn += (string[index])
        else:
            break;
        index -= 1
    return revrtn[::-1]

        
# Check if the directory exists
if not os.path.exists(dir_name):
    # If not, create it
    os.makedirs(dir_name)
    print(f"'{dir_name}' directory created.")
else:
    print(f"'{dir_name}' directory already exists.")
print(onlyName(fileName))
# Save the captured visualization using matplotlib
plt.imsave("images/"+onlyName(fileName)+".png", img_array)









