"""This script is used to run model and count cells"""
from Models import U_net3D
from global_settings import stack_size, correspondances_table, probability_threshold
from utils import CellCounter, Evaluator
import pandas as pd
import os
import numpy as np
import napari

path_to_model = 'MBT-net_1.0.ckpt'
path_to_stack = 'data/raw/stack_1.npy'
path_to_output_folder = 'results'
show_napari = True

colors = ['green', 'red', 'blue']

for file in [path_to_model, path_to_stack, path_to_output_folder]:
    if not os.path.exists(file):
        raise OSError('File or folder is not found: "'+file+'"')
        
if __name__ == '__main__':

    # Initialization of the model and computation of output
    print(f'Loading model from {path_to_model}')
    model = U_net3D(stack_size).load_from_checkpoint(path_to_model).cuda()
    output = model.forward_from_file(path_to_stack)

    CC = CellCounter(output, correspondances_table, probability_threshold)
    results = CC.find_cells()
    results.to_csv(f'{path_to_output_folder}/cells.csv')
    print('Predicted cells:')
    print(results['class'].value_counts())

    if show_napari:
        # Visualization
        print('Visualization...')
        stack = np.load(path_to_stack)
        viewer = napari.Viewer()
        viewer.add_image(stack, blending='additive')
        viewer.add_image(output, blending='additive', colormap='plasma')
        for i, cell_class in enumerate(correspondances_table.keys()):
            df = results.drop(['probability'], axis=1)
            df = df[df['class']==cell_class]
            df = df.drop(['class'], axis=1)
            viewer.add_points(df, size=2, opacity=0.5, edge_color=colors[i], name=cell_class)
        viewer.show()
        napari.run()
    
