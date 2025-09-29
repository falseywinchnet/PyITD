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

        update = update + (torch.rand_like(update)*2 - 1) * etcerta * update
        # Where signs agree (positive), apply normal update
        mask = (sign_agreement > 0)
        p.data = torch.where(mask, 
                            p.data - self.lr * update,
                            p.data)
    return loss


#Should be same math but faster
class WolfFast(Optimizer):
    def __init__(self, params, lr=0.25, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        self.lr = lr
        super().__init__(params, defaults)
        self._etcerta = 0.367879441
        self._et = 1.0 - self._etcerta
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["p"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["buf_update"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["buf_rand"] = torch.empty_like(p, memory_format=torch.preserve_format)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        etcerta = self._etcerta
        et = self._et
        lr = self.lr

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                exp_avg = state["p"]
                grad = p.grad

                # update = exp_avg*et + grad*etcerta  (into buf_update)
                update = state["buf_update"]
                update.copy_(exp_avg).mul_(et).add_(grad, alpha=etcerta)

                # exp_avg = exp_avg*et + update*etcerta  (in-place on exp_avg)
                # We recompute exp_avg*et first, then add update*etcerta
                exp_avg.mul_(et).add_(update, alpha=etcerta)

                # sign mask without sign(): (update*grad) > 0
                mask = update.mul(grad) > 0  # new bool tensor

                # Add multiplicative jitter in-place to update
                rnd = state["buf_rand"]
                rnd.uniform_(0.0, 1.0).mul_(2.0).add_(-1.0)   # [-1,1)
                update.mul_(1.0 + etcerta * rnd)

                # Elementwise mask (cast to update dtype), then single in-place add on params
                m = mask.to(update.dtype)
                update.mul_(m)
                p.add_(update, alpha=-lr)

        return loss



import torch
from torch.optim.optimizer import Optimizer
# 4. Tiger optimizer (Ralston-inspired)
#convergence tends to be slower than most. but, for simpler problems, this is the most efficient way to "jump" the landscape
#use this where you'd use SGD and try a high LR.
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

