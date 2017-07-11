import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from torch.nn import Module
from my_ctc import CTCLoss, _CTC
import numpy as np
import Levenshtein as Lev
import math
from utils import border_msg
import torch.nn.functional as F
from ctc_aligner import viterbi


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
        y_hat, y_hat_paths, y_hat_scores = self.decoder.decode(acts.data, act_lens.data.int())
        y_hat = self.decoder.process_strings(y_hat)

        ### convert y to strings ###
        split_targets = []
        offset = 0
        for size in label_lens.data:
            split_targets.append(labels.data[offset:offset + size])
            offset += size
        y_strings = self.decoder.convert_to_strings(split_targets)

        ### get highest path of y ###
        y_path = viterbi(acts.view(seq_len, n_fears).data.cpu().numpy(), labels.data.cpu().numpy(), blanks=True).view(1,-1)
        if self.cuda:
            y_hat_paths = y_hat_paths.cuda()
            y_path = y_path.cuda()

        ### calc task loss ###
        batch_task_loss = [1.0 * self.task_loss(s1, s2) / len(s1) for s1, s2 in zip(y_strings, y_hat)]
        batch_task_loss = torch.FloatTensor(batch_task_loss)
        if self.cuda: batch_task_loss = batch_task_loss.cuda()

        ### calc grads ###
        self.grads.fill_(0.0)
        grads1 = torch.zeros(acts.size())
        grads2 = torch.zeros(acts.size())
        # ctc(acts, labels, act_lens, label_lens)
        # grads2 = ctc.grads
        if self.cuda: grads1, grads2 = grads1.cuda(), grads2.cuda()
        grads1.scatter_(2, y_hat_paths.view(seq_len, 1, 1), 1)  # put 1s according to y_hat path
        grads2.scatter_(2, y_path.view(seq_len, 1, 1), -1)  # put 1s according to y path
        self.grads += grads1 - grads2
        coeff = batch_task_loss
        coeff = coeff.unsqueeze(0).unsqueeze(2).expand(seq_len, batch_size, n_fears)
        self.grads = self.grads * coeff

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
