import torch
import torch.nn as nn


# fix seed so that random initialization always performs the same 
torch.manual_seed(13)


# create the model N as described in the question
N = nn.Sequential(nn.Linear(10, 10, bias=False),
                  nn.ReLU(),
                  nn.Linear(10, 10, bias=False),
                  nn.ReLU(),
                  nn.Linear(10, 3, bias=False))

# random input
x = torch.rand((1,10)) # the first dimension is the batch size; the following dimensions the actual dimension of the data
x.requires_grad_() # this is required so we can compute the gradient w.r.t x

t = 1 # target class

# Try different epsilon values for FGSM
eps_values = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4]

## Standard FGSM Attack
print("\nStandard FGSM Attack:")
print("-" * 50)
for epsReal in eps_values:
    x.requires_grad_() # Reset gradients
    eps = epsReal - 1e-7 # small constant to offset floating-point errors
    
    # The network N classfies x as belonging to class 2
    original_class = N(x).argmax(dim=1).item()
    print(f"\nEpsilon = {epsReal}")
    print("Original Class: ", original_class)
    
    # compute gradient
    L = nn.CrossEntropyLoss()
    loss = L(N(x), torch.tensor([t], dtype=torch.long))
    loss.backward()
    
    # FGSM attack
    adv_x = x - eps * x.grad.sign()
    
    new_class = N(adv_x).argmax(dim=1).item()
    print("New Class: ", new_class)
    print(f"L-inf norm between x and adv_x: {torch.norm((x-adv_x), p=float('inf')).item():.6f}")
    if new_class == t:
        print("Successfully attacked!")
        break
    else:
        print("Failed to attack!")