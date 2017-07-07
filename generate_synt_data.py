

import numpy as np

num_chars = 5
num_targets = 3
min_duration = 3
max_duration = 6

# generate targets

targets = np.random.choice(np.arange(1, num_chars), num_targets, replace=False)

# first add blanks between characters
num_targets_ext = 2*num_targets+1
targets_ext = np.zeros(num_targets_ext)
targets_ext[0] = 0
index = 0
for target in np.nditer(targets):
    targets_ext[2*index+1] = target
    targets_ext[2*index+2] = 0
    index += 1
print "targets_ext=", targets_ext

# for each target generate the duration and the probability
frame_num = 0
probs = np.array([]).reshape(0, num_chars)
for i in range(num_targets_ext):
    # generate hot vector for Direchlet
    alpha = np.ones(num_chars)
    alpha[int(targets_ext[i])] = 10
    # generate data
    dur = np.random.random_integers(min_duration, max_duration, 1)
    print targets_ext[i], dur, frame_num, "-",
    probs = np.concatenate((probs, np.random.dirichlet(alpha, int(dur))))
    frame_num += dur
print frame_num

np.save("acts.npy", probs)
np.save("targets.npy", targets)
