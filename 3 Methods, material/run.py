import matplotlib.pyplot as plt
from src import *
import numpy as np
from matplotlib import pyplot as plt

dx = dy = 0.3
dz = 0.3

c1 = Macrophage(dx, dy, dz)
c2 = T_Cell(dx, dy, dz)
# c3 = Neutrophil(dx, dy, dz)

sim = Simulator3D_with_crypts(canvas_shape = (1277,1277,100),
                    list_of_cell_types = [c1, c2],
                    cell_number = 1000,
                    sigma = 2,
                    noise_signal_level = 0.05,
                    noise_background_level = 0.01, dx=dx, dy=dy, dz=dz)
sim.run()
# sim.get_point_mask()
sim.save('data', 'Macrophage_T_Cell_3D_simulation', tif=True)
sim.show()           