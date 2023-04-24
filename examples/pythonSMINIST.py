
import os
import tonic


#torch.manual_seed(1234)

#dataset = tonic.datasets.SMNIST(
#    save_to="../datasets", train=False, transform=transform
#)
try: 
    import torch
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True)
except:
    dataset = tonic.datasets.SMNIST(os.getcwd()+"../datasets",train=False,num_neurons=999,dt=1.0)

    #pass
frames, target = next(iter(dataloader))

plot_frames(frames.squeeze())