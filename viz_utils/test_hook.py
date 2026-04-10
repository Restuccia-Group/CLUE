import torch
import torch.nn as nn

# Define a dropout layer with p=0
dropout = nn.Dropout(p=0)

# Define a simple hook function
def forward_hook(module, input, output):
    print(f"Hook called for {module}")
    print(f"Input: {input}")
    print(f"Output: {output}")

# Register the hook to the dropout layer
hook = dropout.register_forward_hook(forward_hook)

# Create some input data
x = torch.tensor([1.0, 2.0, 3.0])

# Pass data through the dropout layer
output = dropout(x)

# Remove the hook after use
hook.remove()