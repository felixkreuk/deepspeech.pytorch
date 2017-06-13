import torch
from torch.autograd import Function
from loss.num_dist_loss import num_dist_loss
from loss.seg_dist_loss import seg_dist_loss
from loss.cosine_similarity import cosine_similarity
import numpy as np


class OrbitLoss(Function):
    def __init__(self, batch_size, n_classes, cost='seg_dist', is_avg='true', cuda=False):
        super(OrbitLoss, self).__init__()
        self.y_hat = 0
        self.y_direct = 0
        self.is_avg = is_avg
        self.mul = torch.range(0, batch_size * n_classes - 1, n_classes).long()
        self.coeff = 1 / np.sqrt(2 * np.pi)
        if cuda:
            self.mul = self.mul.cuda()
        if cost == 'num_dist':
            self.cost = num_dist_loss()
        elif cost == 'seg_dist':
            self.cost = seg_dist_loss(cuda=cuda)
        elif cost == 'cosine':
            self.cost = cosine_similarity(cuda=cuda)
        else:
            self.cost = seg_dist_loss()

    def forward(self, input, target):
        # get y_hat
        self.y_hat = torch.max(input, 1)[1]
        self.y_hat = self.y_hat.t()[0]

        # calc loss
        self.output = self.cost.cost(target, self.y_hat).float()
        o = self.output.sum()
        # average loss
        if self.is_avg.lower() == 'true':
            o /= float(target.size(0))

        self.save_for_backward(input, target)
        return input.new((o,))

    def backward(self, grad_output):
        x, y = self.saved_tensors

        # resize grad_input to a vector instead of matrix
        grad_input = x.new().resize_as_(x).fill_(0)
        grad_input = grad_input.view(x.size(0) * x.size(1))

        # calc the gradients
        idcs_y = torch.add(y, self.mul)
        idcs_y_hat = torch.add(self.y_hat, self.mul)

        coeffs_y = x.new().resize_as_(x).fill_(0)
        coeffs_y = coeffs_y.view(x.size(0) * x.size(1))
        coeffs_y_hat = x.new().resize_as_(x).fill_(0)
        coeffs_y_hat = coeffs_y_hat.view(x.size(0) * x.size(1))

        tmp = y.new().resize_as_(y).fill_(1).float()
        coeffs_y.index_copy_(0, idcs_y, tmp)
        coeffs_y_hat.index_copy_(0, idcs_y_hat, tmp)

        coeffs_y = coeffs_y.view(x.size(0), x.size(1))
        coeffs_y_hat = coeffs_y_hat.view(x.size(0), x.size(1))

        coeff_2_y = torch.mul(x, coeffs_y)
        coeff_2_y_hat = torch.mul(x, coeffs_y_hat)
        coeff_2_y = torch.sum(coeff_2_y, 1)
        coeff_2_y_hat = torch.sum(coeff_2_y_hat, 1)

        res = (-0.5) * torch.pow(coeff_2_y - coeff_2_y_hat, 2)
        coeff = self.coeff * np.exp(res).t()[0]

        # populate them
        grad_input.index_copy_(0, idcs_y, torch.mul(self.output, -1 * coeff))
        grad_input.index_copy_(0, idcs_y_hat, torch.mul(self.output, coeff))

        # resize grad_input to its original size - matrix
        grad_input = grad_input.view(x.size(0), x.size(1))
        return grad_input, None
