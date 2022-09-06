from Models import U_net3D
import numpy as np

input_filename = 'C:/Users/Sergei/Documents/Test_output/stack_0.npy'
output_filename = 'C:/Users/Sergei/Documents/Test_output/output_0.npy'

if __name__ == '__main__':
    model = U_net3D([500,500,100]).cuda()
    output = model.forward_from_file(input_filename)
    np.save(input_filename, output)
    