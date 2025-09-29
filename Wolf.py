#MIT License Copyright 2024 joshuah rainstar
#Wolf, Tiger, Phoenix optimizers
#Think of these as three exploratory approaches
#Wolf is simpler, but often sufficient and sometimes better than adam.
#on FIR filtration coefficient finding, where there's a lot of anistropy, it's good.
#Tiger is expensive,  basically an efficient second order integrator,
#but can be useful for fine tuning where accuracy with fewer steps is important.
#Where you're not going over lots and lots of data, but trying to minimize the damage to an existing model.
#where the function is smooth. It's actually just an integration wrapped up in an optimizer.
#Pheonix is experimental, and something like Wolf on steroids.
#it currently does not converge well on classifier problems, but it does well on noisy quadratics.
#its virtue is, depending on the problem, it can converge very quickly to a plateau, and then you let adam/wolf take over.

#In conclusion a finely tuned Adam optimizer is often the correct tool for the job, but when there is uncertainty,
#or the problem resists learning, try one of these.

from torch.optim.optimizer import Optimizer
class Wolf(Optimizer):
  """Implements Wolf algorithm."""
  #Wolf, also called Rainstar Optimizer, is fast. it is resistant to valleys and other things where adam hangs.
  #on some problems, it is faster than adam. Try a high LR and lower it until it doesnt explode.
  #wolf is initially smoother than adam over difficult plateaus and at high LR.
  def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8):
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

        update = update + (torch.rand_like(update)*2 - 1) * etcerta * update
        # Where signs agree (positive), apply normal update
        mask = (sign_agreement > 0)
        p.data = torch.where(mask, 
                            p.data - self.lr * update,
                            p.data - p.data*self.lr)
    return loss



import torch
from torch.optim.optimizer import Optimizer

class Phoenix(Optimizer):
    """
    M-order leaky-integrator optimizer with per-order decaying integrator rates,
    elementwise directional + magnitude confidence, and optional noise.
    fast, clean initial convergence, but noisy deep down and unable to converge.
    a LR scheduler may help here.
    """
    def __init__(self, params, lr=1e-2, M=7, noise_scale=0.0, eps=1e-6):
        defaults = dict(lr=lr, M=M, noise_scale=noise_scale, eps=eps)
        super().__init__(params, defaults)
        # precompute per-order integrator rates: etcerta_i = 1/e^(i+1), et_i = 1 - etcerta_i
        for group in self.param_groups:
            M = group['M']
            etc_list = [math.exp(-(i+1)) for i in range(M)]
            group['etc_list'] = etc_list
            group['et_list']  = [1.0 - etc for etc in etc_list]
            for p in group['params']:
                # initialize M integrator states per parameter tensor
                self.state[p]['I'] = [torch.zeros_like(p.data) for _ in range(M)]

    def step(self, closure=None):
        """Performs a single optimization step with per-order decaying integrators."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr          = group['lr']
            M           = group['M']
            noise_scale = group['noise_scale']
            eps         = group['eps']
            etc_list    = group['etc_list']
            et_list     = group['et_list']

            for p in group['params']:
                if p.grad is None:
                    continue
                with torch.no_grad():
                    # raw gradient step
                    u = lr * p.grad.data

                    # update M-order integrators with per-order rates
                    I = self.state[p]['I']
                    # first integrator takes raw u
                    etc0 = etc_list[0]
                    et0  = et_list[0]
                    I[0].mul_(et0).add_(etc0 * u)

                    # deeper integrators cascade previous state
                    for i in range(1, M):
                        etc_i = etc_list[i]
                        et_i  = et_list[i]
                        I[i].mul_(et_i).add_(etc_i * I[i-1])

                    # build stack for confidence
                    stack_I = torch.stack(I, dim=0)
                    mean_I  = stack_I.mean(dim=0)
                    # directional: fraction of states with same sign as mean
                    signs   = (stack_I.sign() * mean_I.sign()).gt(0).float()
                    dir_conf = signs.mean(dim=0)
                    # magnitude: inverse spread
                    abs_I   = stack_I.abs()
                    spread  = abs_I.amax(dim=0) - abs_I.amin(dim=0)
                    mag_conf = 1.0 / (spread + eps)
                    # combined confidence
                    conf    = 0.5 * (dir_conf + mag_conf)

                    # new update: mean of each stage's contribution (= etc_i * input_i)
                    contributions = [etc_list[0] * u] + [etc_list[i] * I[i-1] for i in range(1, M)]
                    update = sum(contributions) / M
                    # optional multiplicative noise
                    if noise_scale > 0.0:
                        update = update + noise_scale * (2*torch.rand_like(update)-1) * update

                    # gate by fastest integrator sign
                    sign_small = I[0].sign()
                    gated = torch.where(sign_small * update.sign() > 0,
                                        conf * update,
                                        torch.zeros_like(update))

                    # apply update
                    p.data.add_(-gated)
        return loss

