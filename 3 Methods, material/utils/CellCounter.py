import numpy as np
from skimage import feature, measure
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
        mask = measure.label(cell_pdf > self.threshold)
        props = measure.regionprops(mask, cell_pdf)
        cell_coordinates = []
        for prop in props:
            segment = prop.image_intensity
            bbox = prop.bbox
            origin = bbox[:3]
            local_argmax = np.unravel_index(np.argmax(segment), segment.shape)
            global_argmax = np.array(origin) + np.array(local_argmax)
            cell_coordinates.append(global_argmax)
            

        # Per class
        df = pd.DataFrame(columns = ['class', 'probability', 'axis-0', 'axis-1', 'axis-2'])
        for i, (y,x,z) in enumerate(cell_coordinates):
            probabilities = [self.channels[name][y,x,z] for name in self.classes]
            indx = np.argmax(probabilities)
            winner = self.classes[indx]
            row = [winner, np.max(probabilities), y, x, z]
            df.loc[len(df.index)]=row
        return df

class Evaluator:

    def __init__(self, df_true, df_pred):
        self.df_true = df_true
        self.df_pred = df_pred
        self.classes = df_true['class'].unique()

    def compute_cell_numbers(self):
        counts_true, counts_pred = self.get_counts()
        counts_dict = {name:counts_pred[name] for name in self.classes}
        return counts_dict

    def compute_misclassified_percent(self):
        """Calculates percent of misclassified cells

        Returns:
            dict: dictionary with misclassified percent 0-100%
        """
        counts_true, counts_pred = self.get_counts()
        try:
            diff = np.abs(counts_true - counts_pred)
            diff = diff / counts_true * 100
            return {name:diff[name] for name in self.classes}
        except:
            for c in self.classes:
                counts_true[c] = 100
            return counts_true

    def get_counts(self):
        counts_true = self.df_true['class'].value_counts()
        counts_pred = self.df_pred['class'].value_counts()
        return counts_true, counts_pred