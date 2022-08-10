from src import Cell3D
import numpy as np

class Neutrophil(Cell3D):
    size_cytoplasm          = 15
    size_nucleus            = 6
    gamma_nucleus           = 1.5
    circ_nucleus            = 2.0
    gamma_cytoplasm         = 1.0
    circ_cytoplasm          = 1.5
    gamma_cytoplasm_texture = 0.1
    lowest_intensity        = 0.1

    def __init__(self, dx, dy, dz):
        self.dx = dx
        self.dy = dy
        self.dz = dz

class Macrophage(Cell3D):
    size_cytoplasm          = 15
    size_nucleus            = 6
    gamma_nucleus           = 1.5
    circ_nucleus            = 2.0
    gamma_cytoplasm         = 0.5
    circ_cytoplasm          = 1.5
    gamma_cytoplasm_texture = 0.1
    lowest_intensity        = 0.05

    def __init__(self, dx, dy, dz):
        self.dx = dx
        self.dy = dy
        self.dz = dz

class T_Cell(Cell3D):
    
    size_cytoplasm          = 9
    size_nucleus            = 4
    gamma_nucleus           = 1.5
    circ_nucleus            = 2.0
    gamma_cytoplasm         = 1.5
    circ_cytoplasm          = 2.0
    gamma_cytoplasm_texture = .1
    lowest_intensity        = 0.05

    def __init__(self, dx, dy, dz):
        self.dx = dx
        self.dy = dy
        self.dz = dz

if __name__ == '__main__':
    M = Macrophage(0.5,0.5,1)
    M.run()
    import napari
    viewer = napari.Viewer()
    viewer.add_image(M.image, colormap='gray', rendering='average')
    napari.run()
    # M.show()