from src import CellTypes, Simulator3D_with_crypts
import os
from global_settings import stack_size, dx, dy, dz

c1 = CellTypes.Macrophage(dx, dy, dz)
c2 = CellTypes.T_Cell(dx, dy, dz)
cell_types = [c1, c2]
# output_folder = 'data/raw'
output_folder = 'data/test'
N = 10 # Number of files to be created

# Check if folder exists, if not, creates it
if not os.path.exists('data'):
    os.mkdir('data')
    os.mkdir('data/raw')
    os.mkdir('data/train')
    os.mkdir('data/test')
    os.mkdir('data/output')

for i in range(0, N):
    print(f'Files {i} / {N} are done')
    sim = Simulator3D_with_crypts(
        canvas_shape=stack_size,
        list_of_cell_types=cell_types,
        cell_number=700,
        sigma = 1.5,
        noise_signal_level = 0.5,
        noise_background_level = 0.1, dx=dx, dy=dy, dz=dz)

    sim.run()
    sim.save(output_folder, f'stack_{i}', tif=False)
