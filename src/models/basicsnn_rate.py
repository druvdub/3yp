import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate

class BasicSNNRate(nn.Module):
    def __init__(self, beta=1, num_steps=4, num_classes=2):
        super(BasicSNNRate, self).__init__()

        self.num_steps = num_steps
        self.num_classes = num_classes
        self.spike_grad = surrogate.fast_sigmoid(slope=25)

        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=4)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=self.spike_grad, threshold=0.3)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=self.spike_grad, threshold=0.3)


        # flattened_size = 32 * (32 // 4) * (32 // 4)

        self.fc1 = nn.Linear(32, self.num_classes)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=self.spike_grad, threshold=0.3)

        self.spk1 = None
        self.spk2 = None
        self.mem1 = None
        self.mem2 = None


    def forward(self, x):
        # x shape: [batch_size, num_steps, C, H, W]
        x = x.permute(1, 0, 2, 3, 4)  # New shape: [num_steps, batch_size, C, H, W]
        
        mem1 = self.lif1.reset_mem()
        mem2 = self.lif2.reset_mem()
        mem3 = self.lif3.reset_mem()

        spk_rec = []
        mem_rec = []

        spk1_rec = []
        spk2_rec = []
        mem1_rec = []
        mem2_rec = []

        for step in range(self.num_steps):
            x_step = x[step]  # [batch_size, C, H, W] for current timestep

            # Layer 1
            cur1 = self.conv1(x_step)
            cur1 = F.max_pool2d(cur1, 2)  # Optional pooling
            spike1, mem1 = self.lif1(cur1, mem1)

            # Layer 2
            cur2 = self.conv2(spike1)
            cur2 = F.max_pool2d(cur2, 2)  # Optional pooling
            spike2, mem2 = self.lif2(cur2, mem2)

            # print(spike2.shape)
            # print(spike2.flatten(1).shape)

            # Classifier
            cur3 = self.fc1(spike2.flatten(1))
            spike3, mem3 = self.lif3(cur3, mem3)


            spk1_rec.append(spike1)
            spk2_rec.append(spike2)
            mem1_rec.append(mem1)
            mem2_rec.append(mem2)

            spk_rec.append(spike3)
            mem_rec.append(mem3)

        self.spk1 = torch.stack(spk1_rec, dim=0)
        self.spk2 = torch.stack(spk2_rec, dim=0)
        self.mem1 = torch.stack(mem1_rec, dim=0)
        self.mem2 = torch.stack(mem2_rec, dim=0)

        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)