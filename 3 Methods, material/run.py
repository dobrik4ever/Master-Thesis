from src import Simulator
import matplotlib.pyplot as plt
from src import Cell, Macrophage
import numpy as np
from matplotlib import pyplot as plt
dx = dy = 0.3
dz = 1

# np.random.seed(0)
c = Macrophage(dx, dy, dz)

# c.run()
sim = Simulator(canvas_shape = (300,300),
                    list_of_cells = [c],
                    cell_number = 10,
                    dx = dx, dy = dy, dz = dz,
                    sigma = 2,
                    noise_signal_level = 0.9,
                    noise_background_level = 0.2)

sim.run()
sim.show()           
