"""This script is used to run model and count cells.
Optionally it can evaluate performance, given ground truth
    """
from Models import U_net3D
from global_settings import stack_size, correspondances_table, probability_threshold
from utils import CellCounter, Evaluator
import pandas as pd
import numpy as np
import napari

path_to_model = 'MBT-net_1.0.ckpt'
path_to_stack = 'data/raw/stack_1.npy'
path_to_output_folder = 'results'
show_napari = False
evaluate_performance = True
path_to_actual_cell_locations = 'data/raw/stack_1.csv'

colors = ['red', 'green', 'blue']
df_error = pd.DataFrame(columns=['Stack','Macrophages','T-Cell %'])
df_class = pd.DataFrame(columns=['Stack','Macrophage N true', 'T-Cell N true', 'Macrophage N pred', 'T-Cell N pred'])

if __name__ == '__main__':

    # Initialization of the model and computation of output
    print(f'Loading model from {path_to_model}')
    model = U_net3D(stack_size).load_from_checkpoint(path_to_model).cuda()
    for i in range(10):
        print(f'{i} / 10')
        output = model.forward_from_file(f'data/test/stack_{i}.npy')
        np.save(f'{path_to_output_folder}/output_{i}.npy', output)
        # Counting of cells
        print('Counting cells...')
        CC = CellCounter(output, correspondances_table, probability_threshold)
        results = CC.find_cells()
        results.to_csv(f'{path_to_output_folder}/cells_{i}.csv')
        print('Predicted cells:')
        print(results['class'].value_counts())

        if evaluate_performance:
            # Evaluation of performance
            print('Evaluating performance...')
            results_GT = pd.read_csv(f'data/test/stack_{i}.csv', index_col=0)
            QC = Evaluator(results_GT, results)
            print('Real quantity:')
            print(results_GT['class'].value_counts())
            print('Misclassification %:', end=' ')
            clssfd = QC.compute_cell_numbers()
            msclsfd = QC.compute_misclassified_percent()
            print(msclsfd)
            with open('results/msclsfd.txt', 'a') as file:
                file.write(str(msclsfd))

            df_error.loc[i] = [str(i), msclsfd['Macrophage'], msclsfd['T_Cell']]
            pred = results['class'].value_counts()
            true = results_GT['class'].value_counts()
            df_class.loc[i] = [str(i), true['Macrophage'], true['T_Cell'], pred['Macrophage'], pred['T_Cell']]

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
    
        print()
    df_error.to_csv('results/error.csv')
    df_class.to_csv('results/classified.csv')
