import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from torch.nn import Module
from warpctc_pytorch import CTCLoss, _CTC


class _ctc_hinge_loss(Function):
    def __init__(self, decoder, aug_loss=1):
        self.decoder = decoder
        self.aug_loss = aug_loss
        self.grads = None

    def forward(self, acts, labels, act_lens, label_lens):
        grads = torch.zeros(acts.size()).type_as(acts)
        acts, labels, act_lens, label_lens = Variable(acts), \
                                             Variable(labels), \
                                             Variable(act_lens), \
                                             Variable(label_lens)
        ctc = CTCLoss()

        y_hat = self.decoder.decode(acts.data, acts.size(0))

        costs = self.aug_loss
        costs -= ctc(acts, labels, act_lens, label_lens)
        grads += ctc.grads
        costs += ctc(acts, labels, act_lens, label_lens)
        grads += ctc.grads
        self.grads = grads

        return costs

    def backward(self, grad_output):
        return self.grads, None, None, None


class ctc_hinge_loss(Module):
    def __init__(self, decoder, aug_loss=1):
        super(ctc_hinge_loss, self).__init__()
        self.decoder = decoder
        self.aug_loss = aug_loss

    def forward(self, acts, labels, act_lens, label_lens):
        return _ctc_hinge_loss(self.decoder, self.aug_loss)\
            (acts, labels, act_lens, label_lens)