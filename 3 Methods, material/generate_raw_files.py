from src import CellTypes, Simulator3D_with_crypts
import os

# Check if folder exists, if not, creates it
if not os.path.exists('data'):
    os.mkdir('data')
    os.mkdir('data/raw')
    os.mkdir('data/train')

# Sampling parameters for "acquisiton"
dx = dy = 0.3
dz = 0.3

c1 = CellTypes.Macrophage(dx, dy, dz)
c2 = CellTypes.T_Cell(dx, dy, dz)

cell_types = [c1, c2]
N = 10 # Number of files to be created
for i in range(3, N):
    print(f'Files {i} / {N} are done')
    sim = Simulator3D_with_crypts(
        canvas_shape=(1000,1000,100),
        list_of_cell_types=cell_types,
        cell_number=500,
        sigma = 1.5,
        noise_signal_level = 0.5,
        noise_background_level = 0.1, dx=dx, dy=dy, dz=dz)

    sim.run()
    sim.save('data/raw', f'stack_{i}', tif=False)
