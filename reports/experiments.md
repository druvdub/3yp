# Model Info

- BasicSNN
- BasicSNN (Multi)
- SNN (Paper)
- SNN (Paper) (Multi)
- BinaryCNN (1,2,3)
- MultiCNN (1,2,3)

## BasicSNN

- save_path: "../models/checkpoints/basic_snn/"

### Experiment 1. modelv1.pt

- Time_Steps: 4
- Epochs: 50
- 2 classes
- learning_rate: 1e-3

> 💾 **Note**:
>
> The metrics below are averaged

- Accuracy: 0.96
- Precision: 0.96
- Recall: 0.96
- F1: 0.96
- Used Transform: `transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))
])`

### Experiment 2. modelv2.pt

## BasicSNN (Multi)

- save_path: "../models/checkpoints/basic_multisnn/"

### Experiment 1. modelv1.pt

- Time_Steps: 4
- Epochs: 50
- 6 classes
- learning_rate: 1e-3

> 💾 **Note**:
>
> - The metrics below are averaged
> - But big issue with `Benign` and `Infiltration` class classification
> - `Benign` had a precision of 0.57 and `Infiltration` had a precision of 0.87
> - `Benign` had a recall of 0.95 and `Infiltration` had a recall of 0.29
> - Inflitration being misclassified as Benign

- Accuracy: 0.87
- Precision: 0.90
- Recall: 0.87
- F1: 0.85
- Used Transform: `transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))
])`

> ⚠️ **Issues**:
> Since we are aggressively reducing the input size using a 4x4 kernel, and a stride of 4, we are losing a lot of information. This is causing the model to misclassify the `Infiltration` class as `Benign`. We need to either include padding or reduce the kernel and stride size.

## SNN (Paper)

- save_path: "../models/checkpoints/paper_snn/"

## SNN (Paper) (Multi)

- save_path: "../models/checkpoints/paper_multisnn/"
