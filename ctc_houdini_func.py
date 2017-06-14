import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from torch.nn import Module
from warpctc_pytorch import CTCLoss, _CTC
import numpy as np


class _ctc_houdini_loss(Function):
    def __init__(self, decoder, cuda=False):
        self.decoder = decoder
        self.grads = None
        self.y_hat = None
        self.y_hat_labels = None
        self.y_hat_label_lens = None
        self.y_hat_ctc_grad = None
        self.y_ctc_grad = None
        self.delta = None
        self.coeff = 1 / np.sqrt(2 * np.pi)
        self.cuda = cuda

    def forward(self, acts, labels, act_lens, label_lens):
        """
        MUST get Tensors and return a Tensor.
        """
        self.grads = torch.zeros(acts.size()).type_as(acts)
        acts, labels, act_lens, label_lens = Variable(acts), \
                                             Variable(labels), \
                                             Variable(act_lens), \
                                             Variable(label_lens)
        ctc = _CTC()

        # predict y_hat [as in argmax y_hat = phi(x,y) + L]
        y_hat = self.decoder.decode(acts.data, act_lens)
        # translate string prediction to tensors of labels
        y_hat_labels, y_hat_label_lens = self.decoder.strings_to_labels(y_hat)
        y_hat_labels, y_hat_label_lens = Variable(y_hat_labels), Variable(y_hat_label_lens)

        # calc delta
        delta = ctc(acts, labels, act_lens, label_lens)
        y_ctc_grad = ctc.grads
        delta -= ctc(acts, y_hat_labels, act_lens, y_hat_label_lens)
        y_hat_ctc_grad = ctc.grads
        delta /= len(y_hat)

        # calc grad
        coeff = (-0.5) * torch.pow(delta, 2).data
        coeff = (self.coeff * torch.exp(coeff))[0]
        self.grads = (y_ctc_grad - y_hat_ctc_grad) * coeff

        return delta.data

    def backward(self, grad_output):
        return self.grads, None, None, None

class ctc_houdini_loss(Module):
    def __init__(self, decoder, cuda=False):
        super(ctc_houdini_loss, self).__init__()
        self.decoder = decoder
        self.cuda = cuda

    def forward(self, acts, labels, act_lens, label_lens):
        return _ctc_houdini_loss(self.decoder, cuda=self.cuda)(acts, labels, act_lens, label_lens)
