# Cell Simulation and MBT-net training pipeline

1. Setup global_settings.py
    - Most improtant file. Here you define settings that are used across the whole project
2. run generate_raw_files.py
    - Generates raw files. You can change parameters in this file
3. run generate_train_data.py
    - Generates training data, based on raw files. You can change parameters in this file
4. run train.py
    - It will execute trianing of selected network. If different training parameters required, it can be changed in train.py file
5. run.py to run model and count cells. It will save a table with cells and coordinates in selected folder.

# Usage of MBT-net
Setup settings in run.py and then execute. It will save a .csv table in selected folder and optionally visualize it in napari.

# Usage of the CellSimulator framework
Threre are two main classes for stack generation:

- Cell3D
- Simulator3d

And there is a collection of establised cell types ```CellTypes```

## Cell3D
If only generation of the cell is needed, than a CellSimulator3d can be used. Here  is an example:

Example:
```Python
from src import Cell3D
import napari

dx = dy = dz = 1 # Sampling resolution
cell = Cell3D( 
    size_cytoplasm          = 30,
    size_nucleus            = 10,
    dx = dx, dy = dy, dz = dz,
    gamma_nucleus           = 1.5,
    circ_nucleus            = 2,
    gamma_cytoplasm         = 1,
    circ_cytoplasm          = 2.0,
    gamma_cytoplasm_texture = .1,
    lowest_intensity        = 0.1)

cell.run() # Generates a cell

# For visualization purposes napari can be used:
viewer = napari.Viewer() 
viewer.add_image(   cell.image,
                    name='cell',
                    blending='additive',
                    colormap='gray',
                    rendering='average')
napari.run()
```

## Simulator3D

This class is used to create stack, with given cell types and settings

Example:
```Python
from src import CellTypes, Simulator3D
dx = dy = dz = 0.5 # Sampling resolution
C1 = CellTypes.Macrophage(dx, dy, dz)
C2 = CellTypes.T_Cell(dx, dy, dz)

sim = Simulator3D(  canvas_shape = (100, 100, 100),
                    list_of_cell_types = [C1, C2],
                    cell_number = 10,
                    sigma = 2,
                    noise_signal_level = 0.5,
                    noise_background_level = 0.2)
sim.run() # Generates stack
sim.show() # Shows stack in napari
```

Also there is a child class class ```Simulator3D_with_crypts``` that could be accessed the same way as ```Simulator3D```
```Python
from src import Simulator3D_with_crypts
```
