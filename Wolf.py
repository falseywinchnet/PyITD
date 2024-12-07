#MIT License Copyright 2024 joshuah rainstar

from torch.optim.optimizer import Optimizer
class Wolf(Optimizer):
  """Implements Wolf algorithm."""

  def __init__(self, params, lr=0.25, betas=(0.9, 0.999), eps=1e-8):
        # Define default parameters
        defaults = dict(lr=lr, betas=betas, eps=eps)
        self.lr = lr
        # Initialize the parent Optimizer class first
        super().__init__(params, defaults)
        # Constants specific to Wolf
        # Initialize state for each parameter
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['p'] = torch.zeros_like(p)  # Second moment estimate

  @torch.no_grad()
  def step(self, closure=None):
    """Performs a single optimization step.

    Args:
      closure (callable, optional): A closure that reevaluates the model
        and returns the loss.

    Returns:
      the loss.
    """
    etcerta = 0.367879441
    et = 1 - etcerta

    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue
        state = self.state[p]
            
            # Update step count

        
        # Perform stepweight decay
        grad = p.grad
        state = self.state[p]
        # State initialization

        exp_avg = state['p'] 
        # Weight update
        update = exp_avg * et + grad * etcerta
        state['p']  = exp_avg * et + update * etcerta
        sign_agreement = torch.sign(update) * torch.sign(grad)
        # The value to use for adaptive_alpha depends upon your model.
        #in general, test and set it as high as you can without it exploding
        #it may favor a backoff where it starts small and then gets larger as the model converges on the global minimum
        
        adaptive_alpha = self.lr
        # Where signs agree (positive), apply normal update
        mask = (sign_agreement > 0)
        p.data = torch.where(mask, 
                            p.data - adaptive_alpha * update,
                            p.data)
    return loss


import torch
from torch.optim.optimizer import Optimizer

class TigerOptimizer(Optimizer):
    def __init__(self, model, params, lr=0.01, betas=(0.1, 0.1)): #betas just for the optimizer generator 
        self.model = model
        defaults = dict(lr=lr, betas=betas)
        super().__init__(params, defaults)
        
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['p'] = torch.zeros_like(p)

    #somewhere in your training loop:
    #    def closure():
     #           optimizer.zero_grad()
     #           outputs = net(input)
     #           loss = loss_function(outputs, labels)
     #           loss.backward()
    #            return loss
        
    def step(self, closure):
        etcerta = 0.367879441
        et = 1 - etcerta

        # First compute initial loss and grads
        loss = closure()

        init_weights = {}
        init_grads = {}
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    init_weights[p] = p.data.clone()
                    init_grads[p] = p.grad.clone()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.data -=  (2/3) * p.grad

        # First step evaluation
        loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.data -=    2/3*  (p.grad +  init_grads[p])  # Take second step
        
        # Second step evaluation
        rko_grads = {}
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    rko_grads[p]=  init_weights [p]- (0.25*init_weights [p]+ 0.75* p.data)    #Ralston's 
        
        # Process all updates
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                exp_avg = state['p']

                update = exp_avg * et + rko_grads[p] * etcerta
                state['p'] = exp_avg * et + update * etcerta

                # Reset and apply final update
                #note: for some types of optimization problems, ADD, do not subtract, the update
                p.data = init_weights[p] - update * 0.5

                p.grad.zero_()

        return loss

import torch
from torch.optim.optimizer import Optimizer
class WolfLearnRate(Optimizer):
  """Implements Wolf algorithm."""

  def __init__(self, params, lr=0.25, betas=(0.9, 0.999), eps=1e-8):
        # Define default parameters
        defaults = dict(lr=lr, betas=betas, eps=eps)
        self.lr = lr
        self.t = 0
        self.alpha=256 #not yet figured out
        # Initialize the parent Optimizer class first
        super().__init__(params, defaults)
        # Constants specific to Wolf
        # Initialize state for each parameter
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['p'] = torch.zeros_like(p)  # Second moment estimate

  @torch.no_grad()
  def step(self, closure=None):
    """Performs a single optimization step.

    Args:
      closure (callable, optional): A closure that reevaluates the model
        and returns the loss.

    Returns:
      the loss.
    """
    etcerta = 0.367879441
    et = 1 - etcerta

    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue
        state = self.state[p]
            
            # Update step count

        
        # Perform stepweight decay
        grad = p.grad
        state = self.state[p]
        # State initialization

        exp_avg = state['p'] 
        # Weight update
        update = exp_avg * et + grad * etcerta
        state['p']  = exp_avg * et + update * etcerta
        sign_agreement = torch.sign(update) * torch.sign(grad)
        # The value to use for adaptive_alpha depends upon your model.
        #in general, test and set it as high as you can without it exploding
        #it may favor a backoff where it starts small and then gets larger as the model converges on the global minimum
        print(p.data)
        adaptive_alpha = self.lr
        # Where signs agree (positive), apply normal update
        mask = (sign_agreement > 0)
        lr = 2/(self.t/self.alpha + 2) - math.log(2/(self.t/self.alpha + 2) + 1)

        p.data = torch.where(mask, 
                            p.data - lr * update,
                            p.data)
        p.grad.zero_()
        self.t = self.t + 1
    return loss
