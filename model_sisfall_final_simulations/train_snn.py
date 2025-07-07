import torch
import torch.nn as nn
import snntorch as snn
from snntorch import functional as SF
from dataloader import SisFallDataset
from torch.utils.data import DataLoader
from utils import *
import torch.optim.lr_scheduler as lr_scheduler
from criterion import *
from model import SNNModel0HLayers
import tqdm

torch.manual_seed(41) #0-9
#0, 41, 1234, 1984, 111, 2718, 666, 2468


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

num_steps = 25
batch_size = 4
train_dataset = SisFallDataset('/Users/hemanthsabbella/Documents/SNN_Responsiveness/dataset/sis_fall/time_window_500ms_sliding_50ms/train')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataset = SisFallDataset('/Users/hemanthsabbella/Documents/SNN_Responsiveness/dataset/sis_fall/time_window_500ms_sliding_50ms/test')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


snn_model = SNNModel0HLayers(time_steps=num_steps, input_features=6)
snn_model = snn_model.to(device)
num_epochs = 30
optimizer = torch.optim.Adam(params=snn_model.parameters(), lr=1e-3)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
loss_function = SF.mse_count_loss(correct_rate=0.7, incorrect_rate=0.3)

checkpoint_path = './saves/snn_model_500ms_30ep_mse_count_loss_quick_encoding_seed41.pt'  

#snn_model_500ms_30ep_linear_weighted_quick_encoding_seed0
#snn_model_500ms_50ep_mse_count_loss_lc_sampling_seed0


with tqdm.trange(num_epochs) as pbar:
    for epoch in pbar:
        snn_model.train()
        train_acc = 0
        train_loss = 0
        val_acc = 0
        val_loss = 0

        for _, (inputs, label) in enumerate(train_loader):

            #Quick Encoding
            inputs = quick_spikes_encoding(inputs, 25)
            inputs = torch.cat((inputs, torch.flip(inputs, dims=[2])), dim=1)

            # LC Sampling
            # spikes_up_input, spikes_down_input = lc_sampling(inputs)
            # spikes_up_input = time_slot_accumulation(spikes_up_input, sampling_freq=100, subsampling_freq=50)
            # spikes_down_input = time_slot_accumulation(spikes_down_input, sampling_freq=100, subsampling_freq=50)
            # inputs = torch.concat((spikes_up_input, spikes_down_input), dim=1)

            inputs = inputs.to(device)
            label = label.to(device)
            target = torch.argmax(label, dim=1)
            # print(inputs.shape)
            spk = snn_model(inputs)  # forward-pass

            loss_val = loss_function(spk, target)
            train_loss += loss_val.item()

            optimizer.zero_grad()  # zero out gradients
            loss_val.backward()  # calculate gradients
            optimizer.step()  # update weights

            acc_val = SF.accuracy_rate(spk, target)
            train_acc += acc_val


        avg_loss = train_loss / len(train_loader)
        avg_acc = train_acc / len(train_loader)


        snn_model.eval()
        with torch.no_grad():
            for _, (inputs, label) in enumerate(test_loader):

               #Quick Encoding
                inputs = quick_spikes_encoding(inputs, 25)
                inputs = torch.cat((inputs, torch.flip(inputs, dims=[2])), dim=1)

                # LC Sampling
                # spikes_up_input, spikes_down_input = lc_sampling(inputs)
                # spikes_up_input = time_slot_accumulation(spikes_up_input, sampling_freq=100, subsampling_freq=50)
                # spikes_down_input = time_slot_accumulation(spikes_down_input, sampling_freq=100, subsampling_freq=50)
                # inputs = torch.concat((spikes_up_input, spikes_down_input), dim=1)


                inputs = inputs.to(device)
                label = label.to(device)
                target = torch.argmax(label, dim=1)

                spk = snn_model(inputs)

                loss_val = loss_function(spk, target)
                val_loss += loss_val.item()

                acc_val = SF.accuracy_rate(spk, target)
                val_acc += acc_val
                

        avg_val_loss = val_loss / len(test_loader)
        avg_val_acc = val_acc / len(test_loader)

        scheduler.step(avg_val_loss)

        pbar.set_postfix({
            'epoch': epoch + 1,
            'train_loss': '{0:1.5f}'.format(avg_loss),
            'train_acc': '{:.5f}'.format(avg_acc),
            'val_loss': '{0:1.5f}'.format(avg_val_loss),
            'val_acc': '{:.5f}'.format(avg_val_acc)
        })

        if (epoch + 1) % 5 == 0:
            torch.save(snn_model.state_dict(), checkpoint_path)

