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
# 4. Tiger optimizer (Ralston-inspired)
class Tiger(Optimizer):
    def __init__(self, model, params, lr=0.01):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.model = model
        for group in self.param_groups:
            for p in group['params']:
                # persistent state for smoothing
                self.state[p]['p'] = torch.zeros_like(p.data)

    def step(self, closure):
        etcerta = 0.367879441
        et = 1.0 - etcerta
        lr = self.param_groups[0]['lr']

        # 1) initial forward+backward
        loss = closure()

        # 2) snapshot weights and initial gradients
        init_w = {}
        g0     = {}
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                init_w[p] = p.data.clone()
                g0[p]     = p.grad.clone()

        # 3) Ralston stage 1: x1 = x0 - (2/3)*lr * g0
        for p, grad0 in g0.items():
            p.data = p.data - (2/3) * lr * grad0

        # 4) second forward+backward to get g1
        loss = closure()
        g1 = {}
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                g1[p] = p.grad.clone()

        # 5) Ralston stage 2: x2 = x1 - (2/3)*lr * (g0 + g1)
        for p in g1:
            p.data = p.data - (2/3) * lr * (g0[p] + g1[p])

        # 6) compute Ralston residual: rko = x_R - x0
        #    where x_R = x0 + lr*(¼·k1 + ¾·k2)
        #    but since we mutated p.data in place, do:
        rko = {}
        for p, w0 in init_w.items():
            # current p.data == x2
            rko[p] = w0 - (0.25 * w0 + 0.75 * p.data)

        # 7) Wolf‐style smoothing + final update
        with torch.no_grad():
            for p in init_w:
                state_p = self.state[p]['p']
                update  = state_p * et + etcerta * rko[p]
                new_st  = state_p * et + etcerta * update
                # final step: subtract the entire smoothed residual
                p.data  = init_w[p] - update
                # save new persistent state
                self.state[p]['p'] = new_st
                # clear grads
                p.grad = None

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
