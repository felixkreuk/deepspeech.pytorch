import torch
from numpy import random
import torch.utils.data as data_utils

ABC = list("_ABCDEFGHIJKLMNOPQRSTUVWXYZ ")
ABC_SIZE = len(ABC)
print ABC
SAMPLES = 100

def create_data(samples, abc, seq_len_max, label_len_max, batch_size):
    feats, targets, feats_lens, targets_lens = [], [], [], []
    for i in xrange(samples):
        label_len = random.randint(3, label_len_max)
        label = random.choice(abc, label_len)
        # print label
        seq_feats = None
        seq_labels = torch.LongTensor(label_len_max).fill_(0)
        seq_feats_len = torch.LongTensor(1).fill_(0)
        seq_labels_len = torch.LongTensor(1).fill_(0)

        i = 0
        # create the sequence of features
        for c_idx, c in enumerate(label):
            c_repeat = random.randint(1, 3)  # repeat letter c for 1-3 times
            for j in xrange(c_repeat):
                if seq_feats is None:  # if new create
                    seq_feats = torch.FloatTensor(1, 1, len(abc))
                    seq_feats[0][0][abc.index(c)] = 1
                else:  # else concat
                    new_seq_feats = torch.FloatTensor(1, 1, len(abc))
                    new_seq_feats[0][0][abc.index(c)] = 1
                    seq_feats = torch.cat((seq_feats, new_seq_feats), dim=1)
                i += 1
            seq_labels[c_idx] = abc.index(c)
        # pad with zeros
        remainder_feats = torch.FloatTensor(1, seq_len_max - seq_feats.size(1), len(abc)).fill_(0)
        seq_feats = torch.cat([seq_feats, remainder_feats], dim=1)
        seq_feats = torch.transpose(seq_feats, 1, 2)
        # seq len and label len
        seq_feats_len[0] = i
        seq_labels_len[0] = len(label)
        feats.append(seq_feats)
        targets.append(seq_labels)
        feats_lens.append(seq_feats_len)
        targets_lens.append(seq_labels_len)

    feats = torch.stack(feats, 0)
    targets = torch.stack(targets, 0)
    feats_lens = torch.cat(feats_lens, 0)
    targets_lens = torch.cat(targets_lens, 0)

    feats = list(torch.split(feats, batch_size, dim=0))
    targets = list(torch.split(targets, batch_size, dim=0))
    feats_lens = list(torch.split(feats_lens, batch_size, dim=0))
    targets_lens = list(torch.split(targets_lens, batch_size, dim=0))

    data = zip(feats, targets, feats_lens, targets_lens)

    for i,(f,t,fl,tl) in enumerate(data):
        data[i] = (f, t.view(-1).index_select(0, t.view(-1).nonzero().view(-1)), fl, tl)

    return data


if __name__ == '__main__':
    data = create_data(100, ABC, 50, 10, 16)
    print "done"