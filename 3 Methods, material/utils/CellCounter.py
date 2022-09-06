import numpy as np
from skimage import feature
import pandas as pd
import torch

class CellCounter:

    def __init__(self, pdf:torch.tensor, correspondances_table:dict, threshold = 0.95) -> None:
        if type(pdf) == torch.tensor:
            pdf = pdf.cpu().detach().numpy()
        self.threshold = threshold
        self.channels = {name : pdf[correspondances_table[name]] for name in correspondances_table}
        self.classes = [name for name in correspondances_table if 'Background' not in name]

    def find_cells(self):
        cell_pdf = 1 - self.channels['Background']
        cell_coordinates = feature.peak_local_max(cell_pdf, threshold_abs=self.threshold)

        # Per class
        df = pd.DataFrame(columns = ['class', 'probability', 'axis-0', 'axis-1', 'axis-2'])
        for i, (y,x,z) in enumerate(cell_coordinates):
            probabilities = [self.channels[name][y,x,z] for name in self.classes]
            indx = np.argmax(probabilities)
            winner = self.classes[indx]
            row = [winner, np.max(probabilities), y, x, z]
            df.loc[len(df.index)]=row
        print(df['class'].value_counts())
        return df