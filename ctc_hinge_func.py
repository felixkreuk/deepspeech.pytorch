import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from torch.nn import Module
from warpctc_pytorch import CTCLoss, _CTC
import torch.nn.functional as F


class _ctc_hinge_loss(Function):
    def __init__(self, decoder, aug_loss=1):
        self.decoder = decoder
        self.aug_loss = aug_loss
        self.grads = None

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
        # y_hat = self.decoder.decode(acts.data, act_lens)
        y_hat = self.decoder.decode(F.log_softmax(acts).data, act_lens)
        # translate string prediction to tensors of labels
        y_hat_labels, y_hat_label_lens = self.decoder.strings_to_labels(y_hat)
        y_hat_labels, y_hat_label_lens = Variable(y_hat_labels), Variable(y_hat_label_lens)

        # hinge-loss calculation
        costs = ctc(acts, labels, act_lens, label_lens)
        self.grads += ctc.grads
        costs -= ctc(acts, y_hat_labels, act_lens, y_hat_label_lens)
        self.grads -= ctc.grads
        costs += self.aug_loss

        return costs.data

    def backward(self, grad_output):
        return self.grads, None, None, None


class ctc_hinge_loss(Module):
    def __init__(self, decoder, aug_loss=1):
        super(ctc_hinge_loss, self).__init__()
        self.decoder = decoder
        self.aug_loss = aug_loss

    def forward(self, acts, labels, act_lens, label_lens):
        return _ctc_hinge_loss(self.decoder, self.aug_loss)(acts, labels, act_lens, label_lens)
