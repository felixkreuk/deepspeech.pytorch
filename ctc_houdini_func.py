import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from torch.nn import Module
from my_ctc import CTCLoss, _CTC
import numpy as np
import Levenshtein as Lev
import math
from utils import border_msg


class _ctc_houdini_loss(Function):
    def __init__(self, decoder, min_coeff, task_loss=Lev.distance, cuda=False):
        self.decoder = decoder
        self.grads = None
        self.coeff = 1 / np.sqrt(2 * np.pi)
        self.cuda = cuda
        self.task_loss = task_loss
        self.min_coeff = min_coeff
        self.P = 0.01

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
        seq_len, batch_size, n_fears = self.grads.size()

        ### predict y_hat ###
        y_hat1 = self.decoder.decode(acts.data, act_lens.data.int())
        y_hat = self.decoder.process_strings(y_hat1)
        # translate string prediction to tensors of labels
        y_hat_labels, y_hat_label_lens = self.decoder.strings_to_labels(y_hat)
        y_hat_labels, y_hat_label_lens = Variable(y_hat_labels), Variable(y_hat_label_lens)

        ### convert y to strings ###
        split_targets = []
        offset = 0
        for size in label_lens.data:
            split_targets.append(labels.data[offset:offset + size])
            offset += size
        y_strings = self.decoder.convert_to_strings(split_targets)

        ### calc task loss ###
        border_msg(" TASK LOSS ")
        batch_task_loss = [1.0 * self.task_loss(s1, s2) / len(s1) for s1, s2 in zip(y_strings, y_hat)]
        batch_task_loss = torch.FloatTensor(batch_task_loss)
        if self.cuda: batch_task_loss = batch_task_loss.cuda()

        # calc delta & grads
        delta = ctc(acts, labels, act_lens, label_lens)
        y_ctc_grad = ctc.grads
        delta -= ctc(acts, y_hat_labels, act_lens, y_hat_label_lens)
        y_hat_ctc_grad = ctc.grads

        # delta = (delta - delta.mean().data[0]) / delta.std().data[0]  # normalize delta
        good_deltas = torch.ge(delta.data, torch.zeros(batch_size))
        if self.cuda: good_deltas = good_deltas.cuda()
        print "good deltas: %d/%d" % (good_deltas.sum(), len(good_deltas))

        border_msg(" NUMBERS ")
        # calc & clip coeff
        coeff = (-0.5) * torch.pow(delta, 2).data
        coeff = (self.coeff * torch.exp(coeff))
        print "\n%s\n" % ("#" * 50)

        if self.cuda: coeff = coeff.cuda()
        # coeff = torch.mul(batch_task_loss, coeff)
        # coeff = torch.clamp(coeff, 0.3, 1)
        coeff = batch_task_loss
        coeff *= good_deltas.float()  # mind only "good" deltas
        coeff = coeff.unsqueeze(0).unsqueeze(2).expand(seq_len, batch_size, n_fears)

        # calc grad
        # self.grads = -(y_hat_ctc_grad - y_ctc_grad) * coeff
        self.grads = -(y_hat_ctc_grad * 0.05 - y_ctc_grad) * coeff

        return torch.FloatTensor([batch_task_loss.sum()])

    def backward(self, grad_output):
        return self.grads, None, None, None

class ctc_houdini_loss(Module):
    def __init__(self, decoder, min_coeff, cuda=False):
        super(ctc_houdini_loss, self).__init__()
        self.decoder = decoder
        self.cuda = cuda
        self.min_coeff = min_coeff

    def forward(self, acts, labels, act_lens, label_lens):
        return _ctc_houdini_loss(self.decoder, min_coeff=self.min_coeff, cuda=self.cuda)(acts, labels, act_lens, label_lens)
