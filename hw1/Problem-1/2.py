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
eps_values = [0.5, 1, 10, 100, 0.05]


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




## Iterative FGSM Attack
print("\nIterative FGSM Attack:")
print("-" * 50)
# Try different epsilon values for iterative FGSM
eps_values_iter = [0.125, 0.25, 0.5, 1.0]
num_iterations = 100

for epsReal in eps_values_iter:
    print(f"\nTesting with epsilon = {epsReal}")
    print("-" * 30)
    
    alpha = epsReal  # step size proportional to epsilon
    
    x_iter = x.clone().detach().requires_grad_(True)
    for i in range(num_iterations):
        outputs = N(x_iter)
        loss = L(outputs, torch.tensor([t], dtype=torch.long))
        loss.backward()
        
        # Take small step
        x_iter = x_iter - alpha * x_iter.grad.sign()
        
        # Project back onto epsilon ball
        x_iter = torch.min(torch.max(x_iter, x - epsReal), x + epsReal)
        
        x_iter = x_iter.clone().detach().requires_grad_(True)
        
        new_class = N(x_iter).argmax(dim=1).item()
        if new_class == t:
            print(f"Success at iteration {i+1}")
            print(f"L-inf norm between x and x_iter: {torch.norm((x-x_iter), p=float('inf')).item():.6f}")
            break
        elif i == num_iterations - 1:
            print("Failed to find adversarial example")
            print(f"Final L-inf norm: {torch.norm((x-x_iter), p=float('inf')).item():.6f}")

    final_class = N(x_iter).argmax(dim=1).item()
    print(f"Original Class: {original_class}")
    print(f"Final Class: {final_class}")