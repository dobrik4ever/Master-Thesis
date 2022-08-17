import napari
import numpy as np
import os

if __name__ == '__main__':
    max_epoch = 0
    for file in os.listdir('data/output'):
        epoch = int(file.split('epoch_')[1].split('-')[0])
        max_epoch = max(max_epoch, epoch)

    classes = {
        0: {
            "name":"Macrophage",
            'color':'green'
            },
        1: {
            'name':"T-cell",
            'color':'red'
        },
        2: {
            'name':"Background",
            'color':None
        }
    }
    stack = np.load('data/raw/stack_0.npy')
    print(stack.shape)
    stack = np.transpose(stack, [2,0,1])

    viewer = napari.Viewer()
    viewer.add_image(
       stack,
        name = 'stack',
        colormap='gray',
        rendering='average',
        blending='additive')
    # epoch = 198
    fname_template = 'data/output/U_net_2_epoch_{epoch}-class_{Class}.npy'
    for channel_idx in classes:
        if classes[channel_idx]['name'] != 'Background':
            stack = np.load(fname_template.format(epoch=max_epoch, Class=channel_idx))
            print(stack.shape)
            stack = np.transpose(stack, [2,0,1])
            viewer.add_image(
                stack,
                name = classes[channel_idx]['name'],
                colormap=classes[channel_idx]['color'],
                rendering='iso',
                iso_threshold=0.8,
                blending='additive')

napari.run()