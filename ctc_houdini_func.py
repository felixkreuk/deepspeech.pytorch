import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from torch.nn import Module
from warpctc_pytorch import CTCLoss, _CTC
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

        ### PREDICT y_hat ###
        y_hat = self.decoder.decode(acts.data, act_lens)
        y_hat = self.decoder.process_strings(y_hat, remove_repetitions=True)
        # translate string prediction to tensors of labels
        y_hat_labels, y_hat_label_lens = self.decoder.strings_to_labels(y_hat)
        y_hat_labels, y_hat_label_lens = Variable(y_hat_labels), Variable(y_hat_label_lens)

        ### CONVERT y TO STRINGS ###
        split_targets = []
        offset = 0
        for size in label_lens.data:
            split_targets.append(labels.data[offset:offset + size])
            offset += size
        y_strings = self.decoder.convert_to_strings(split_targets)

        border_msg(" PREDICTIONS ")
        for a,b in zip(y_strings,y_hat):
            print "\t\"%s\" ### \"%s\"" % (a,b)

        ### CALC ACTUAL LOSS ###
        # task loss [cer by default]
        batch_task_loss = [1.0 * self.task_loss(s1, s2) / len(s1) for s1, s2 in zip(y_strings, y_hat)]
        border_msg(" EDs ")
        print batch_task_loss
        batch_task_loss = sum(batch_task_loss)

        # calc delta & grads
        delta = ctc(acts, labels, act_lens, label_lens)
        y_ctc_grad = ctc.grads
        delta -= ctc(acts, y_hat_labels, act_lens, y_hat_label_lens)
        y_hat_ctc_grad = ctc.grads


        # for i in xrange(min(len(y_strings[0]), len(y_hat[0]))):
        #     print "diff %s,%s: %s" % (y_strings[0][i], y_hat[0][i], y_hat_ctc_grad[i] - y_ctc_grad[i])

        border_msg(" NUMBERS ")
        print "delta:",delta
        # calc & clip coeff
        coeff = (-0.5) * torch.pow(delta, 2).data
        coeff = (self.coeff * torch.exp(coeff))[0]
        print "coeff (before clip) =", coeff
        if coeff < 1e-10 or math.isnan(coeff) or math.isinf(coeff):
            coeff = 1
        print "coeff (after clip) =", coeff

        # calc grad
        self.grads = (-y_hat_ctc_grad + y_ctc_grad) * coeff * batch_task_loss

        return torch.FloatTensor([batch_task_loss])

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
