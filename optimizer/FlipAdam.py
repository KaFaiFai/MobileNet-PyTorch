import torch
from torch.optim import Optimizer

from model import MobileNet, LeNet


class FlipAdam(Optimizer):
    def __init__(self, params, lr=3e-4, betas=(0.999, 0.9), eps=1e-8, weight_decay=1):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        params = list(params)
        self.exp_avgs = [torch.zeros_like(param, memory_format=torch.preserve_format) for param in params]
        self.exp_avg_sqs = [torch.zeros_like(param, memory_format=torch.preserve_format) for param in params]

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, step=0, exp_avgs=self.exp_avgs,
                        exp_avg_sqs=self.exp_avg_sqs)
        super(FlipAdam, self).__init__(params, defaults)

    def step(self, closure=None, *, grad_scaler=None):
        for group in self.param_groups:
            group["step"] += 1

            params = group["params"]
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]  # to be implemented
            step = group["step"]
            exp_avgs = group["exp_avgs"]
            exp_avg_sqs = group["exp_avg_sqs"]

            bias_correction1 = 1 - beta1 ** step  # exp_avg bias correction
            bias_correction2 = 1 - beta2 ** step  # exp_avg_sq bias correction

            with torch.no_grad():
                for param, exp_avg, exp_avg_sq in zip(params, exp_avgs, exp_avg_sqs):
                    grad = param.grad + param * weight_decay

                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

                    step_size = lr
                    nominator = exp_avg / bias_correction1
                    denominator = (exp_avg_sq / bias_correction2).sqrt()
                    # param.addcdiv_(nominator, denominator.add_(eps), value=-step_size)  # normal Adam
                    param.addcdiv_(denominator, nominator.add_(eps), value=-step_size)


if __name__ == '__main__':
    net = LeNet(num_class=10)
    o = FlipAdam(net.parameters())
    o.step()
    print(o.state)
    print(o.param_groups)
