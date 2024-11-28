MIT License Copyright 2024 joshuah rainstar

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
