import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from torch.nn import Module
from warpctc_pytorch import CTCLoss, _CTC
import torch.nn.functional as F
from utils import border_msg


class _ctc_hinge_loss(Function):
    def __init__(self, decoder, aug_loss=1, cuda=False):
        self.decoder = decoder
        self.aug_loss = aug_loss
        self.grads = None
        self.cuda = cuda

    def forward(self, acts, labels, act_lens, label_lens):
        """
        MUST get Tensors and return a Tensor.
        """
        acts_size = acts.size()
        seq_size, batch_size, n_feats = acts_size[0], acts_size[1], acts_size[2]
        P = 0.15
        Pk = P / (n_feats - 1)  # (P * (n_feats - 1)) / n_feats
        flat_idx = torch.range(0, seq_size * batch_size * n_feats - n_feats, n_feats).long()
        if self.cuda: flat_idx = flat_idx.cuda()
        self.grads = torch.zeros(acts.size()).type_as(acts)
        acts, labels, act_lens, label_lens = Variable(acts), \
                                             Variable(labels), \
                                             Variable(act_lens), \
                                             Variable(label_lens)
        ctc = _CTC()

        ### PREDICT y_hat ###
        y_hat1 = self.decoder.decode(acts.data, act_lens)
        y_hat = self.decoder.process_strings(y_hat1)
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
        for a, b in zip(y_strings, y_hat):
            print "\t\"%s\" (%d) --- \"%s\" (%d)" % (a, len(a), b, len(b))

        ### CALC DELTA & GRADS ###
        delta = ctc(acts, labels, act_lens, label_lens)
        y_ctc_grad = ctc.grads
        delta -= ctc(acts, y_hat_labels, act_lens, y_hat_label_lens)
        y_hat_ctc_grad = ctc.grads
        # calc grad
        self.grads = (-y_hat_ctc_grad * 0.5 + y_ctc_grad)

        # return delta.data

        y_ctc_grad_max_idx = y_ctc_grad.max(2)[1].view(-1)  # get indexes of max prob for each time t
        y_ctc_grad_max_idx = torch.add(y_ctc_grad_max_idx, flat_idx)  # indexes of max prob in flat acts vector
        y_hat_ctc_grad_max_idx = y_hat_ctc_grad.max(2)[1].view(-1)  # get indexes of max prob for each time t
        y_hat_ctc_grad_max_idx = torch.add(y_hat_ctc_grad_max_idx, flat_idx)  # indexes of max prob in flat acts vector
        # add -P to correct labels and Pk to incorrect ones
        self.grads = self.grads.view(-1)
        P_vec = torch.FloatTensor(seq_size * batch_size * n_feats).fill_(0).float()
        Pk_vec = torch.FloatTensor(seq_size * batch_size * n_feats).fill_(Pk).float()
        if self.cuda:
            P_vec = P_vec.cuda()
            Pk_vec = Pk_vec.cuda()
        # init p vec with zeros and fill correct places with p
        P_vec = P_vec.index_fill_(0, y_ctc_grad_max_idx, P)
        # init pk vec with pk and fill zeros in correct places (don't need to add pk to them, already have p)
        Pk_vec = Pk_vec.index_fill_(0, y_ctc_grad_max_idx, 0)
        Pk_vec = Pk_vec.index_fill_(0, y_hat_ctc_grad_max_idx, 0)
        self.grads = torch.add(self.grads, P_vec)
        self.grads = torch.add(self.grads, Pk_vec)
        self.grads = self.grads.view(seq_size, batch_size, n_feats)

        print "\n%s\n" % ("#" * 50)

        return delta.data

    def backward(self, grad_output):
        return self.grads, None, None, None


class ctc_hinge_loss(Module):
    def __init__(self, decoder, aug_loss=1, cuda=False):
        super(ctc_hinge_loss, self).__init__()
        self.decoder = decoder
        self.aug_loss = aug_loss
        self.cuda = cuda

    def forward(self, acts, labels, act_lens, label_lens):
        return _ctc_hinge_loss(self.decoder, self.aug_loss, self.cuda)(acts, labels, act_lens, label_lens)
