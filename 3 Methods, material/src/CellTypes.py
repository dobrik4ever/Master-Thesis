from src.CellSimulator import Cell, NoiseBlob, PolygonBlob
import numpy as np

class Macrophage(Cell):
    size_cytoplasm          = 14
    size_nucleus            = 6
    gamma_nucleus           = 1.5
    circ_nucleus            = 2.0
    gamma_cytoplasm         = 1.0
    circ_cytoplasm          = 2.0
    gamma_cytoplasm_texture = 0.1
    lowest_intensity        = 0.0

    def __init__(self, dx, dy, dz):
        # super().__init__(   dx = dx, dy = dy, dz = dz)
        self.dx = dx
        self.dy = dy
        self.dz = dz

if __name__ == '__main__':
    M = Macrophage(0.5,0.5,1)
    M.run()
    M.show()