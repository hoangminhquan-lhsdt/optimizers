r"""Functional interface"""
import math
import torch
from torch import Tensor
from typing import List, Optional

# TODO: use foreach API in optim._functional to do all the computation

def _make_sparse(grad, grad_indices, values):
    size = grad.size()
    if grad_indices.numel() == 0 or values.numel() == 0:
        return torch.empty_like(grad)
    return torch.sparse_coo_tensor(grad_indices, values, size)


def adagrad(params: List[Tensor],
            grads: List[Tensor],
            state_sums: List[Tensor],
            state_steps: List[int],
            lr: float,
            weight_decay: float,
            lr_decay: float,
            eps: float):
    r"""Functional API that performs Adagrad algorithm computation.

    See :class:`~torch.optim.Adagrad` for details.
    """

    for (param, grad, state_sum, step) in zip(params, grads, state_sums, state_steps):
        if weight_decay != 0:
            if grad.is_sparse:
                raise RuntimeError("weight_decay option is not compatible with sparse gradients")
            grad = grad.add(param, alpha=weight_decay)

        clr = lr / (1 + (step - 1) * lr_decay)

        if grad.is_sparse:
            grad = grad.coalesce()  # the update is non-linear so indices must be unique
            grad_indices = grad._indices()
            grad_values = grad._values()
            size = grad.size()

            state_sum.add_(_make_sparse(grad, grad_indices, grad_values.pow(2)))
            std = state_sum.sparse_mask(grad)
            std_values = std._values().sqrt_().add_(eps)
            param.add_(_make_sparse(grad, grad_indices, grad_values / std_values), alpha=-clr)
        else:
            state_sum.addcmul_(grad, grad, value=1)
            std = state_sum.sqrt().add_(eps)
            param.addcdiv_(grad, std, value=-clr)


def adam(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         state_steps: List[int],
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         weight_decay: float,
         eps: float):
    r"""Functional API that performs Adam algorithm computation.

    See :class:`~torch.optim.Adam` for details.
    """

    for i, param in enumerate(params):

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        param.addcdiv_(exp_avg, denom, value=-step_size)


def adamw(params: List[Tensor],
          grads: List[Tensor],
          exp_avgs: List[Tensor],
          exp_avg_sqs: List[Tensor],
          max_exp_avg_sqs: List[Tensor],
          state_steps: List[int],
          amsgrad: bool,
          beta1: float,
          beta2: float,
          lr: float,
          weight_decay: float,
          eps: float):
    r"""Functional API that performs AdamW algorithm computation.

    See :class:`~torch.optim.AdamW` for details.
    """
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        param.addcdiv_(exp_avg, denom, value=-step_size)


def sgd(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    for i, param in enumerate(params):

        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        param.add_(d_p, alpha=-lr)


def adadelta(params: List[Tensor],
             grads: List[Tensor],
             square_avgs: List[Tensor],
             acc_deltas: List[Tensor],
             lr: float,
             rho: float,
             eps: float,
             weight_decay: float):
    r"""Functional API that performs Adadelta algorithm computation.

    See :class:`~torch.optim.Adadelta` for details.
    """

    for (param, grad, square_avg, acc_delta) in zip(params, grads, square_avgs, acc_deltas):
        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        square_avg.mul_(rho).addcmul_(grad, grad, value=1 - rho)
        std = square_avg.add(eps).sqrt_()
        delta = acc_delta.add(eps).sqrt_().div_(std).mul_(grad)
        param.add_(delta, alpha=-lr)
        acc_delta.mul_(rho).addcmul_(delta, delta, value=1 - rho)


def rmsprop(params: List[Tensor],
            grads: List[Tensor],
            square_avgs: List[Tensor],
            grad_avgs: List[Tensor],
            momentum_buffer_list: List[Tensor],
            lr: float,
            alpha: float,
            eps: float,
            weight_decay: float,
            momentum: float,
            centered: bool):
    r"""Functional API that performs rmsprop algorithm computation.

    See :class:`~torch.optim.RMSProp` for details.
    """

    for i, param in enumerate(params):
        grad = grads[i]
        square_avg = square_avgs[i]

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

        if centered:
            grad_avg = grad_avgs[i]
            grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
            avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_().add_(eps)
        else:
            avg = square_avg.sqrt().add_(eps)

        if momentum > 0:
            buf = momentum_buffer_list[i]
            buf.mul_(momentum).addcdiv_(grad, avg)
            param.add_(buf, alpha=-lr)
        else:
            param.addcdiv_(grad, avg, value=-lr)



# MY IMPLEMENTATIONS
def my_sgd(params: List[Tensor],
           grads: List[Tensor],
           lr: float) -> None:
    for i, param in enumerate(params):
        grad = grads[i]
        param.add_(grad, alpha=-lr)  # param - lr*grad


def my_momentum(params: List[Tensor],
                grads: List[Tensor],
                momentum_list: List[Tensor],
                lr: float,
                momentum: float) -> None:
    r"""Implemented based on the original formula
            v_t = m*v_t-1 + lr*grad
            theta = theta - v_t
        instead of Pytorch implementation
            v_t = m*v_t-1 + grad
            theta = theta - lr*v_t
    """
    for i, param in enumerate(params):
        grad = grads[i]
        m = momentum_list[i]

        if m is None:
            """ In the first step,
                    momentum term = lr*grad
                since previous momentum accumulation do not exist
            """
            m = torch.clone(grad).detach()
            momentum_list[i] = m
            m.mul_(lr)
        else:
            m.mul_(momentum).add_(grad, alpha=lr)  # muy*m_t-1 + grad

        param.add_(m, alpha=-1)


def my_nesterov(params: List[Tensor],
                grads: List[Tensor],
                momentum_list: List[Tensor],
                lr: float,
                momentum: float) -> None:
    r'''Implementation of NAG based on Algorithm 7 
        in "Incorporating Nesterov into Adam"
    '''
    for i, param in enumerate(params):
        grad = grads[i]
        m = momentum_list[i]

        if m is None:
            """ In the first step,
                    momentum term = lr*grad
                since previous momentum accumulation do not exist
            """
            m = torch.clone(grad).detach()
            momentum_list[i] = m
            m_bar = m.mul(lr)
        else:
            # momentum scheduler not used, thus `muy` is the same at every step
            m.mul_(momentum).add_(grad)                      # muy*m_t-1 + grad
            m_bar = grad.add(m, alpha=momentum).mul_(lr)     # grad + muy*m_t

        param.add_(m_bar, alpha=-1)


def my_adagrad(params: List[Tensor],
               grads: List[Tensor],
               state_sums: List[Tensor],
               lr: float,
               eps: float) -> None:
    for i, param in enumerate(params):
        grad = grads[i]
        sums = state_sums[i]

        sums.add_(torch.mul(grad, grad))                    # G += grad**2
        v = grad.mul(lr).div_(sums.add_(eps).sqrt_())       # lr * grad / sqrt(G + eps)
        param.add_(v, alpha=-1)


def my_adadelta(params: List[Tensor],
                grads: List[Tensor],
                state_Eg2: List[Tensor],
                state_Edelta2: List[Tensor],
                rho: float,
                eps: float) -> None:
    for i, param in enumerate(params):
        grad = grads[i]
        E_g2 = state_Eg2[i]
        E_delta2 = state_Edelta2[i]

        E_g2.mul_(rho).add_(grad.square(), alpha=1-rho)
        RMS_g = E_g2.add(eps).sqrt()
        RMS_delta = E_delta2.add(eps).sqrt()
        delta = -grad.mul(RMS_delta.div(RMS_g))
        param.add_(delta)
        E_delta2.mul_(rho).add_(delta.square(), alpha=1-rho)



def my_rmsprop(params: List[Tensor],
               grads: List[Tensor],
               state_Eg2: List[Tensor],
               lr: float,
               gamma: float,
               eps: float) -> None:
    for i, param in enumerate(params):
        grad = grads[i]
        Eg2 = state_Eg2[i]

        g2 = grad.square()                               # grad**2
        Eg2.mul_(gamma).add_(g2.mul(1-gamma))            # gamma*E_g2 + (1-gamma)*(grad**2)
        delta = grad.mul(lr).div(Eg2.sqrt().add(eps))    # lr * grad / sqrt(E_g2 + eps)
        param.add_(delta, alpha=-1)


def my_adam(params: List[Tensor],
            grads: List[Tensor],
            state_Ms: List[Tensor],
            state_Vs: List[Tensor],
            state_steps: List[int],
            lr: float,
            beta1: float,
            beta2: float,
            eps: float) -> None:
    for i, param in enumerate(params):
        grad = grads[i]
        M = state_Ms[i]
        V = state_Vs[i]
        step = state_steps[i]

        M.mul_(beta1).add_(grad, alpha=1-beta1)
        V.mul_(beta2).add_(grad.square().mul(1-beta2))

        M_hat = M.div(1 - beta1**step)
        V_hat = V.div(1 - beta2**step)

        update = M_hat.div(V_hat.sqrt().add(eps))
        param.add_(update, alpha=-lr)


def my_amsgrad(params: List[Tensor],
               grads: List[Tensor],
               state_Ms: List[Tensor],
               state_Vs: List[Tensor],
               state_max_Vs: List[Tensor],
               state_steps: List[int],
               lr: float,
               beta1: float,
               beta2: float,
               eps: float) -> None:
    for i, param in enumerate(params):
        grad = grads[i]
        M = state_Ms[i]
        V = state_Vs[i]
        step = state_steps[i]

        M.mul_(beta1).add_(grad, alpha=1-beta1)
        V.mul_(beta2).add_(grad.square().mul(1-beta2))

        M_hat = M.div(1 - beta1**step)
        V_hat = torch.maximum(state_max_Vs[i], V, out=state_max_Vs[i]).div(1 - beta2**step)

        update = M_hat.div(V_hat.sqrt().add(eps))
        param.add_(update, alpha=-lr)


def my_adamw(params: List[Tensor],
             grads: List[Tensor],
             state_Ms: List[Tensor],
             state_Vs: List[Tensor],
             state_steps: List[int],
             lr: float,
             beta1: float,
             beta2: float,
             eps: float,
             weight_decay: float) -> None:
    for i, param in enumerate(params):
        regularization = param.mul(weight_decay)
        grad = grads[i].add(regularization)
        M = state_Ms[i]
        V = state_Vs[i]
        step = state_steps[i]

        M.mul_(beta1).add_(grad, alpha=1-beta1)
        V.mul_(beta2).add_(grad.square().mul(1-beta2))

        M_hat = M.div(1 - beta1**step)
        V_hat = V.div(1 - beta2**step)

        update = M_hat.div(V_hat.sqrt().add(eps))
        param.add_(update.add_(regularization), alpha=-lr)


def my_nadam(params: List[Tensor],
             grads: List[Tensor],
             state_Ms: List[Tensor],
             state_Vs: List[Tensor],
             state_steps: List[int],
             lr: float,
             beta1: float,
             beta2: float,
             eps: float) -> None:
    for i, param in enumerate(params):
        grad = grads[i]
        M = state_Ms[i]
        V = state_Vs[i]
        step = state_steps[i]

        M.mul_(beta1).add_(grad, alpha=1-beta1)
        V.mul_(beta2).add_(grad.square().mul(1-beta2))

        M_hat = M.div(1 - beta1**step)
        V_hat = V.div(1 - beta2**step)

        update = M_hat.div(V_hat.sqrt().add(eps))
        param.add_(update, alpha=-lr)