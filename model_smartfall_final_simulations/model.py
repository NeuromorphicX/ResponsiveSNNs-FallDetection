import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SNNModel0HLayers(torch.nn.Module):

    def __init__(self, time_steps, input_features, beta=0.9):
        super(SNNModel0HLayers, self).__init__()

        self.time_steps = time_steps
        self.input_features = input_features
        self.beta = beta

        # layer 1
        self.fc1 = torch.nn.Linear(in_features=self.input_features, out_features=2)
        self.lif1 = snn.Leaky(beta=self.beta, learn_beta=True)


    def forward(self, x):
        """Forward pass for several time steps."""

        # Initalize membrane potential
        mem1 = self.lif1.init_leaky()


        # Empty lists to record outputs
        spk_recording = []

        for step in range(self.time_steps):
            # print(x.shape)
            input = x[:,:,step]
            # print(input.shape)
            cur1 = self.fc1(input)
            spk1, mem1 = self.lif1(cur1, mem1)

            spk_recording.append(spk1)

    
        out = torch.stack(spk_recording, dim=0)
        # print(out.shape)
        # # out = torch.stack(mem_recording, dim=0)
        # out = torch.squeeze(out)
        # out = torch.sum(out, dim=0)
        return out
