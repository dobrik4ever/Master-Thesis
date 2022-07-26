from src import Simulator
import matplotlib.pyplot as plt
from src import Cell

dx = dy = dz = 0.5
C1 = Cell(  
    size_cytoplasm          = 20,
    size_nucleus            = 10,
    dx = dx, dy = dy, dz = dz,
    gamma_nucleus           = 1.5,
    circ_nucleus            = 2,
    gamma_cytoplasm         = 0.5,
    circ_cytoplasm          = 2.0,
    gamma_cytoplasm_texture = .5,
    lowest_intensity        = 0.5)

C2 = Cell(  
    size_cytoplasm          = 30,
    size_nucleus            = 15,
    dx = dx, dy = dy, dz = dz,
    gamma_nucleus           = 1.5,
    circ_nucleus            = 2.5,
    gamma_cytoplasm         = 1.0,
    circ_cytoplasm          = 2.0,
    gamma_cytoplasm_texture = .5,
    lowest_intensity        = 0.5)

sim = Simulator(canvas_shape = (300,300),
                list_of_cells = [C1, C2],
                cell_number = 100,
                dx = dx, dy = dy, dz = dz,
                sigma = 2,
                noise_signal_level = 0.5,
                noise_background_level = 0.2)
sim.run()
plt.figure(figsize =(12,12))

plt.imshow(sim.image, cmap='gray')
plt.colorbar()
plt.show()
