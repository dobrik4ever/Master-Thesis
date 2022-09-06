from Models import U_net3D

if __name__ == '__main__':
    model = U_net3D([500,500,100]).cuda()
    output = model.forward_from_file('C:/Users/Sergei/Documents/Test_output/stack_0.npy')
    print('done')