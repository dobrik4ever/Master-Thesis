import matplotlib.pyplot as plt
from src import Simulator3D, Cell3D, Macrophage, T_Cell, Neutrophil
import numpy as np
from matplotlib import pyplot as plt

dx = dy = 0.3
dz = 0.3

c1 = Macrophage(dx, dy, dz)
c2 = T_Cell(dx, dy, dz)
# c3 = Neutrophil(dx, dy, dz)

sim = Simulator3D(  canvas_shape = (300,300,300),
                    list_of_cell_types = [c1, c2],
                    cell_number = 15,
                    sigma = 2,
                    noise_signal_level = 0.2,
                    noise_background_level = 0.1)
sim.run()
# sim.get_point_mask()
sim.save('3 Methods, material/data', 'Macrophage_T_Cell_3D_simulation', tif=True)
sim.show()           
