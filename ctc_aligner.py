from torch import LongTensor as lt
import numpy as np


acts = np.load("acts.npy")
targets = np.load("targets.npy")

def viterbi(acts, targets, blanks=False):
    (num_frames, num_chars) = acts.shape
    (num_targets,) = targets.shape
    # print num_frames, num_chars, num_targets

    # first add blanks between characters
    if blanks:  # extend targets
        num_targets_ext = 2 * num_targets + 1
        targets_ext = np.zeros(num_targets_ext)
        targets_ext[0] = 0
        index = 0
        for target in np.nditer(targets):
            targets_ext[2 * index + 1] = target
            targets_ext[2 * index + 2] = 0
            index += 1
    else:
        num_targets_ext = num_targets
        targets_ext = targets
    # print "targets_ext=", targets_ext

    mispar_katan_meod = float(-10000000.0)

    # generate cumulative sum of activity matrix
    cumsum_acts = np.cumsum(acts, 0)

    # first target (which is blank)
    D = mispar_katan_meod * np.ones((num_targets_ext, num_frames))
    F = np.zeros((num_targets_ext, num_frames))
    D[0, 0] = acts[0, int(targets_ext[0])]
    for t in range(1, num_frames):
        D[0, t] = cumsum_acts[t, int(targets_ext[0])]

    # recursion
    for i in range(1, num_targets_ext - 1):
        for t in range(2, num_frames - 1):
            max_tmp = mispar_katan_meod
            tmp = mispar_katan_meod
            t_prev_best = 0
            for t_prev in range(1, t - 1):
                tmp = D[i - 1, t_prev] + cumsum_acts[t, int(targets_ext[i])] - cumsum_acts[t_prev, int(targets_ext[i])]
                if tmp > max_tmp:
                    max_tmp = tmp
                    t_prev_best = t_prev
            D[i, t] = max_tmp
            F[i, t] = t_prev_best

    # last target (which is blank)
    max_tmp = mispar_katan_meod
    tmp = mispar_katan_meod
    t_prev_best = 0
    for t_prev in range(1, num_frames):
        tmp = D[num_targets_ext - 2, t_prev] + cumsum_acts[num_frames - 1, int(targets_ext[0])] - cumsum_acts[
            t_prev, int(targets_ext[0])]
        if tmp > max_tmp:
            max_tmp = tmp
            t_prev_best = t_prev
    D[num_targets_ext - 1, num_frames - 1] = max_tmp
    F[num_targets_ext - 1, num_frames - 1] = t_prev_best

    # back track
    best_a = np.zeros_like(targets_ext)
    best_a[num_targets_ext - 1] = F[num_targets_ext - 1, num_frames - 1]
    for i in range(num_targets_ext - 2, 0, -1):
        best_a[i] = F[i, int(best_a[i + 1])]

    # print "\t\t",best_a

    # create complete path
    path = [int(targets_ext[0])]  # add extra first element
    repeat_list = []
    for i in xrange(len(best_a) - 1):
        repeat_list.append(best_a[i+1] - best_a[i])
    repeat_list.append(num_frames - best_a[-1])
    for c,repeat in zip(targets_ext, repeat_list):
        path.extend([int(c) for j in xrange(int(repeat))])
    return lt(path[:-1])

if __name__ == '__main__':
    x = viterbi(acts, targets)
    print 1