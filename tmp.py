from decoder import BeamSearchDecoder
import torch

labels = "_'ABCDEFGHIJKLMNOPQRSTUVWXYZ#"
chars = len(labels)
time = 5
batch_size = 3

decoder = BeamSearchDecoder(labels)
probs = torch.FloatTensor(time, batch_size, chars)
print(decoder.decode(probs))