import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate


class BinaryActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

binary_activation = BinaryActivation.apply

class BinarizedSpikingNetwork(nn.Module):
    def __init__(self, input_size=1024, hidden_size=64, output_size=2, beta=0.9, threshold=0.5, num_steps=8):
        super().__init__()
        
        # Neuron parameters
        self.beta = beta  # Decay rate for membrane potential
        self.threshold = threshold  # Firing threshold
        self.num_steps = num_steps  # Number of time steps
        
        # Surrogate gradient function for backpropagation
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        # Network architecture
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.LayerNorm(hidden_size)
        
        # Spiking neurons
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.LayerNorm(hidden_size // 2)
        
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)
        
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.lif_out = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)


        self.spk1 = self.spk2 = None
        self.mem1 = self.mem2 = None
    
    def forward(self, x):
        # Initialize membrane potentials and output spikes storage
        batch_size = x.size(0)
        mem1 = self.lif1.reset_mem()
        mem2 = self.lif2.reset_mem()
        mem_out = self.lif_out.reset_mem()
        
        # Storage for recording final layer spikes
        spk_rec = []
        mem_rec = []

        spk1_rec = []
        mem1_rec = []
        spk2_rec = []
        mem2_rec = []
        
        # Flatten input if needed (e.g., for MNIST images)
        x = x.view(batch_size, -1)
        
        # Simulate network for multiple time steps
        for step in range(self.num_steps):
            # First layer
            cur1 = self.fc1(x)
            cur1 = self.bn1(cur1)
            
            # Binary activation with STE
            cur1 = binary_activation(cur1)
            
            # Update membrane potential and generate spikes
            spk1, mem1 = self.lif1(cur1, mem1)
            
            # Second layer 
            cur2 = self.fc2(spk1)
            cur2 = self.bn2(cur2)
            
            # Binary activation with STE
            cur2 = binary_activation(cur2)
            
            # Update membrane potential and generate spikes
            spk2, mem2 = self.lif2(cur2, mem2)
            
            # Output layer
            cur_out = self.fc3(spk2)
            
            # Binary activation with STE
            cur_out = binary_activation(cur_out)
            
            # Output neuron
            spk_out, mem_out = self.lif_out(cur_out, mem_out)


            spk1_rec.append(spk1)
            mem1_rec.append(mem1)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
            
            # Record spikes
            spk_rec.append(spk_out)
            mem_rec.append(mem_out)
        

        self.spk1 = torch.stack(spk1_rec, dim=0)
        self.mem1 = torch.stack(mem1_rec, dim=0)
        self.spk2 = torch.stack(spk2_rec, dim=0)
        self.mem2 = torch.stack(mem2_rec, dim=0)

        # Aggregate spikes over time (sum or mean)
        spk_rec = torch.stack(spk_rec, dim=0)  # Shape: (num_steps, batch_size, output_size)
        mem_rec = torch.stack(mem_rec, dim=0)

        return spk_rec, mem_rec