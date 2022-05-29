from PyQt5 import QtWidgets, QtGui, QtCore
from PIL import Image, ImageQt
import numpy as np
from skimage import io, transform, measure, exposure
from GUI.Ui_flatten_stack import Ui_Form
from ImageProcessing.scripts import Denoiser
import sys

class Flatten_stack_window(QtWidgets.QWidget, Ui_Form):

    stack = None
    initial_image = None
    processed_image = None
    image_sideview = None

    def __init__(self, parent = None) -> None:
        super().__init__()
        self.setupUi(self)

        self.stack = io.imread('/Users/dobrik4ever/Library/CloudStorage/OneDrive-Personal/Documents/Study/winter_21-22/Master Thesis/4 Raw data/stack.tif')
        
        # Define QThreads

        # self.image_crop = Image_crop(self)
        # self.image_crop.finished.connect(self.update_initial_image)
        self.worker = Worker(self)
        self.worker.finished.connect(self.update_initial_image)

        
        # Define connections

        self.slider_depth_selector.sliderReleased.connect(self.worker.start)
        # self.slider_slices_average.valueChanged.connect(self.image_processor.start)
    

        # # Define borders
        self.slider_depth_selector.setRange(0, self.stack.shape[1] - 1)
        self.spinBox_channel_selector.setRange(0, self.stack.shape[0] - 1)

    def update_initial_image(self):
        self.array_to_image(self.initial_image, self.label_image_initial)

    # def update_image_sideview(self):
    #     self.label_image_sideview.setPixmap(self.array_to_image(self.image_sideview))

    def array_to_image(self, array:np.array, label:QtWidgets.QLabel):
        
        array = exposure.rescale_intensity(array, out_range=(0, 255)).astype(np.uint8).transpose([1,2,0])
        image = Image.fromarray(array)
        image = ImageQt.ImageQt(image)
        qpixmap = QtGui.QPixmap.fromImage(image)
        label.setPixmap(qpixmap)

    # @property
    # def channel(self):
    #     return self.spinBox_channel_selector.value()

    # @property
    # def num_of_slices(self):
    #     return self.slider_slices_average.value()

    @property
    def depth(self):
        return self.slider_depth_selector.value()

    # @property
    # def slice_sideview(self):
    #     return self.slider_slice_selector.value()

    # @property
    # def image_sideview(self):
    #     slice = self.stack[:,:,self.slice_sideview]
    #     return slice

class Worker(QtCore.QThread):

    def __init__(self, parent:Flatten_stack_window = None):
        super().__init__()
        self.p = parent

    def run(self):
        self.p.initial_image = np.max(self.p.stack[:,self.p.depth:,:,:], axis=1)


# class Flatten_stack_window(QtWidgets.QWidget, Ui_Form):

#     stack = None
#     initial_image = None
#     processed_image = None
#     image_sideview = None

#     def __init__(self, parent = None) -> None:
#         super().__init__()
#         self.setupUi(self)

#         self.stack = io.imread('/Users/dobrik4ever/Library/CloudStorage/OneDrive-Personal/Documents/Study/winter_21-22/Master Thesis/4 Raw data/stack.tif')
        
#         # Define QThreads

#         self.image_processor = Image_processor(self)
#         self.image_processor.finished.connect(self.update_processed_image)

#         self.image_crop = Image_crop(self)
#         self.image_crop.finished.connect(self.update_initial_image)

        
#         # Define connections

#         self.slider_depth_selector.valueChanged.connect(self.image_crop.start)
#         self.slider_slices_average.valueChanged.connect(self.image_processor.start)
    

#         # Define borders
#         self.slider_depth_selector.setRange(0, self.stack.shape[1] - 1)
#         self.spinBox_channel_selector.setRange(0, self.stack.shape [0] - 1)

#         self.slider_slice_selector.setValue(1)
#         self.slider_depth_selector.setValue(1)


#     def update_processed_image(self):
#         self.label_image_processed.setPixmap(self.array_to_image(self.image_processor.updated_image))
    
#     def update_initial_image(self):
#         self.label_image_initial.setPixmap(self.array_to_image(self.initial_image))

#     def update_image_sideview(self):
#         self.label_image_sideview.setPixmap(self.array_to_image(self.image_sideview))

#     def array_to_image(self, array:np.array):
#         print(array.shape)
#         array = exposure.rescale_intensity(array, out_range=(0, 255)).astype(np.uint8).transpose([1,2,0])
#         image = Image.fromarray(array)
#         image = ImageQt.ImageQt(image)
#         qpixmap = QtGui.QPixmap.fromImage(image)

#         return qpixmap

#     @property
#     def channel(self):
#         return self.spinBox_channel_selector.value()

#     @property
#     def num_of_slices(self):
#         return self.slider_slices_average.value()

#     @property
#     def depth(self):
#         return self.slider_depth_selector.value()

#     @property
#     def slice_sideview(self):
#         return self.slider_slice_selector.value()

#     @property
#     def image_sideview(self):
#         slice = self.stack[:,:,self.slice_sideview]
#         return slice

class Image_crop(QtCore.QThread):
    def __init__(self, parent: Flatten_stack_window = None):
        super().__init__()
        self.p = parent

    def run(self):
        self.p.initial_image = np.max(self.p.stack[:,self.p.depth:,:,:], axis=1)

class Image_processor(QtCore.QThread):
    def __init__(self, parent:Flatten_stack_window = None):
        super().__init__()
        self.p = parent
        self.updated_image = None

    def run(self):
        image = self.p.stack.copy()
        if self.p.radioButton_median_filtering.isChecked():
            image = Denoiser.depth_median(image[self.p.channel], self.p.num_of_slices)
        elif self.p.radioButton_median_filtering.isChecked():
            image = Denoiser.depth_average(image[self.p.channel], self.p.num_of_slices)
        elif self.p.radioButton_nofiltering.isChecked():
            image = image[self.p.channel]
        image = Denoiser.max_projection(image)
        self.updated_image = image
