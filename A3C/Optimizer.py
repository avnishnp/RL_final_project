import math
import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.optim as optim

def init_weights(layer):
    layer_type = layer.__class__.__name__
    if 'Conv' in layer_type:
        weight_shape = list(layer.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
    elif 'Linear' in layer_type:
        weight_shape = list(layer.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
    else:
        return
    bound = np.sqrt(6.0 / (fan_in + fan_out))
    layer.weight.data.uniform_(-bound, bound)
    layer.bias.data.fill_(0)


class Adam(optim.Adam):
    """Implements Adam algorithm with shared states.
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0):
        super(Adam, self).__init__(params, lr, betas, eps, weight_decay)

        for x in self.param_groups:
            for p in x['params']:
                state = self.state[p]
                
                state['momentum_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['variance_avg'] = p.data.new().resize_as_(p.data).zero_()
                
                state['step'] = torch.zeros(1)
                
    def share_memory(self):
        for x in self.param_groups:
            for p in x['params']:
                state = self.state[p]
                
                state['momentum_avg'].share_memory_()
                state['variance_avg'].share_memory_()
                
                state['step'].share_memory_()
            
    
    def step(self, compute_loss=None):
        """
        Performs a single optimization step.

        Args:
            compute_loss (callable, optional): Function to compute the loss.

        Returns:
            The computed loss, if applicable.
        """
        current_loss = None
        if compute_loss is not None:
            current_loss = compute_loss()

        for parameter_group in self.param_groups:
            for parameter in parameter_group['params']:
                if parameter.grad is None:
                    continue

                gradient_data = parameter.grad.data
                param_state = self.state[parameter]

                momentum_avg, variance_avg = param_state['momentum_avg'], param_state['variance_avg']
                momentum_decay, variance_decay = parameter_group['betas']

                # Increment update count
                param_state['step'] += 1

                # Apply weight decay if specified
                if parameter_group['weight_decay'] != 0:
                    gradient_data.add_(parameter_group['weight_decay'], parameter.data)

                # Update running averages for moments
                momentum_avg.mul_(momentum_decay).add_(1 - momentum_decay, gradient_data)
                variance_avg.mul_(variance_decay).addcmul_(1 - variance_decay, gradient_data, gradient_data)

                # Compute bias-corrected learning rate
                corrected_momentum = 1 - momentum_decay ** param_state['step'].item()
                corrected_variance = 1 - variance_decay ** param_state['step'].item()
                adjusted_lr = parameter_group['lr'] * math.sqrt(corrected_variance) / corrected_momentum

                # Update parameters using computed learning rate
                parameter.data.addcdiv_(-adjusted_lr, momentum_avg, variance_avg.sqrt().add_(parameter_group['eps']))

        return current_loss

    
    
    
    
    
    
