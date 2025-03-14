import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SNNClassifier(nn.Module):
    def __init__(self, num_classes=2, time_steps=4, threshold=0.3, alpha=0.5):
        super(SNNClassifier, self).__init__()
        self.time_steps = time_steps
        
        # Spiking layers
        self.conv1 = ConvSpikingLayer(1, 16, 4, 4, threshold, alpha)
        self.conv2 = ConvSpikingLayer(16, 32, 2, 2, threshold, alpha)
        
        # Time-value encoder
        self.encoder = TimeValEncoder(time_steps)
        
        # Classifier
        self.fc = nn.Linear(32*4*4, num_classes)
        
        # State trackers
        self.spk1 = self.spk2 = None
        self.mem1 = self.mem2 = None

    def forward(self, x):
        # Add time dimension: (B,C,H,W) → (T,B,C,H,W)
        x = x.unsqueeze(0).repeat(self.time_steps, 1, 1, 1, 1)
        
        # Process through layers
        spk1, mem1 = self.conv1(x)
        spk2, mem2 = self.conv2(spk1)
        
        # Temporal encoding
        encoded = self.encoder(spk2)


        self.spk1 = spk1
        self.spk2 = spk2
        self.mem1 = mem1
        self.mem2 = mem2
        
        out = self.fc(encoded.flatten(1))
        return out

class ConvSpikingLayer(nn.Module):
    """Single convolutional spiking layer with document-specific reset"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, threshold=0.3, alpha=0.3):
        super(ConvSpikingLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding='valid')
        
        self.lif = snn.Leaky(
            beta=1.0,  # No leakage: V(t) = V(t-1) + I(t)
            threshold=threshold,
            reset_mechanism="none",  # Disable built-in reset
            spike_grad=surrogate.fast_sigmoid(slope=25),
            output=True
        )
        self.alpha = alpha

    def forward(self, x):
        """Input shape: (T, B, C, H, W)"""
        time_steps, batch_size = x.shape[:2]
        spk_rec = []
        mem_rec = []

        mem = self.lif.reset_mem()
        
        for t in range(time_steps):
            conv_out = self.conv(x[t])
            spk, mem = self.lif(conv_out, mem)
            
            # Document-specific reset: (V - V_thr) * α
            mem = torch.where(spk > 0,(mem - self.lif.threshold) * self.alpha, mem)
            
            # clamp negative values to zero
            # mem = F.relu(mem)
            
            spk_rec.append(spk)
            mem_rec.append(mem)
            
        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)

class TimeValEncoder(nn.Module):
    def __init__(self, time_steps):
        super(TimeValEncoder, self).__init__()
        weights = [2**(time_steps-i-1) for i in range(time_steps)]
        weights = torch.tensor(weights, dtype=torch.float32)
        self.weights = nn.Parameter(weights/weights.sum(), requires_grad=False)

    def forward(self, x):
        # x: (time_steps, batch_size, channels, height, width)
        return torch.einsum('tb...,t->b...', x, self.weights.to(x.device))
    
class CustomLoss(nn.Module):
    def __init__(self, num_classes, class_weights=None):
        super(CustomLoss, self).__init__()
        self.n_classes = num_classes
        self.class_weights = class_weights

    def forward(self, predict, target):
        predict = torch.log_softmax(predict, dim=1)

        # Convert targets to one-hot encoding (if needed)
        if target.dim() == 1 or target.size(1) != self.n_classes:
            target_onehot = torch.zeros_like(predict).scatter(1, target.unsqueeze(1), 1)
        else:
            target_onehot = target  # Assume already one-hot
        
        # 1. Compute α term (difference between max prediction and correct class score)
        cor = (predict * target_onehot).sum(dim=1)  # Correct class scores
        pre = predict.max(dim=1)[0]                 # Max prediction scores
        alpha = pre - cor

        # 2. Compute β term (ranking penalty)
        val = predict.gather(1, target.unsqueeze(1)).squeeze()  # Correct class values
        ids = (predict > val.unsqueeze(1)).sum(dim=1).float()   # Number of classes ranked higher
        beta = 1 - cor

        # 3. loss (Eq. in Algorithm 2) 
        loss = (self.n_classes * alpha + (ids + 1) * beta)

        # if class weights are provided, apply them
        if self.class_weights is not None:
            loss *= self.class_weights[target]

        return loss.mean()
