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

class Phoenix(Optimizer):
    """
    M-order leaky-integrator optimizer with elementwise directional + magnitude confidence weighting,
    and optional sign-gating based on the smallest integrator's immediate signal.
    """
    def __init__(self, params, lr=1e-2, etcerta=0.367879441, M=7, noise_scale=0.367879441, eps=1e-6):
        defaults = dict(lr=lr, etcerta=etcerta, M=M, noise_scale=noise_scale, eps=eps)
        super().__init__(params, defaults)
        for group in self.param_groups:
            group['et'] = 1.0 - group['etcerta']
            for p in group['params']:
                # initialize M integrator states per parameter tensor
                self.state[p]['I'] = [torch.zeros_like(p.data) for _ in range(group['M'])]

    def step(self, closure=None):
        """Performs a single optimization step with per-element confidence."""
        loss = None
        # forward/backward to populate gradients
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            etc = group['etcerta']
            et  = group['et']
            M   = group['M']
            noise_scale = group['noise_scale']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                with torch.no_grad():
                    # raw gradient step
                    u = lr * p.grad.data

                    # capture previous integrator states
                    I = self.state[p]['I']
                    old_I = [i.clone() for i in I]

                    # build list of stage contributions
                    contributions = []
                    # stage 0 contribution
                    contrib0 = etc * u
                    contributions.append(contrib0)
                    I[0].mul_(et).add_(contrib0)

                    # deeper stages
                    for i in range(1, M):
                        contrib = etc * old_I[i-1]
                        contributions.append(contrib)
                        I[i].mul_(et).add_(contrib)

                    # compute elementwise confidence using all I states
                    stack_I = torch.stack(I, dim=0)
                    mean_I  = stack_I.mean(dim=0)
                    signs   = (stack_I.sign() * mean_I.sign()).gt(0).float()
                    dir_conf = signs.mean(dim=0)
                    abs_I   = stack_I.abs()
                    mag_conf = 1.0 / (abs_I.amax(dim=0) - abs_I.amin(dim=0) + eps)
                    conf    = 0.5 * (dir_conf + mag_conf)

                    # new update = mean of contributions
                    update = sum(contributions) / M
                    if noise_scale > 0.0:
                        update = update + noise_scale * (2*torch.rand_like(update)-1) * update

                    # sign gating from fastest integrator
                    sign_small = I[0].sign()
                    gated = torch.where(sign_small * update.sign() > 0,
                                        conf * update,
                                        torch.zeros_like(update))

                    # apply update
                    p.data.add_(-gated)

        return loss
