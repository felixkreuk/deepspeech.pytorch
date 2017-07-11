import torch


def border_msg(msg):
    row = len(msg)
    h = ''.join(['+'] + ['-' *row] + ['+'])
    result= h + '\n'"|"+msg+"|"'\n' + h
    print(result)

def tensor_pad1d(tensor, pad_amount, pad_value=0, pad_at_the_end=True):
    assert len(tensor.size()) == 1
    if pad_at_the_end:
        return torch.cat([tensor.float(), torch.zeros(pad_amount).fill_(pad_value)], 0)
    else:
        return torch.cat([torch.zeros(pad_amount).fill_(pad_value), tensor.float()], 0)