import matplotlib.pyplot as plt
from src import Simulator, Cell3D, Macrophage, T_Cell
import numpy as np
from matplotlib import pyplot as plt
dx = dy = 0.3
dz = 1


# Run cell 
M = Macrophage(dx, dy, dz)
M.run()
print('Image shape: ',M.image.shape)
import napari
viewer = napari.Viewer()
viewer.add_image(M.image, colormap='gray', rendering='average')
napari.run()

# Run cell simulator

c1 = Macrophage(dx, dy, dz)
c2 = T_Cell(dx, dy, dz)

sim = Simulator(canvas_shape = (300,300),
                    list_of_cells = [c],
                    cell_number = 10,
                    dx = dx, dy = dy, dz = dz,
                    sigma = 2,
                    noise_signal_level = 0.9,
                    noise_background_level = 0.2)

sim.run()
sim.show()           
