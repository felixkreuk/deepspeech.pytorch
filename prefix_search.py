import numpy as np
import operator as op
from collections import defaultdict


p1 = 1
p0 = 0


def prefix_search(dist, alphabet, blank_symbol, k):
    T = dist.shape[0]
    blank_idx = alphabet.index(blank_symbol)
    # pb = {('', 1, 0): 1}
    pb = {(c, 1, 0): 0 for c in alphabet}
    pb[('', 1, 0)] = 1
    # pnb = {('', 1, 0): 0}
    pnb = {(c, 1, 0): 0 for c in alphabet}
    pnb[('', 1, 0)] = 0
    A_prev = ['']

    for t in xrange(1, T):
        A_next = []
        for l in A_prev:
            for c in alphabet:
                c_idx = alphabet.index(c)
                if c == blank_symbol:
                    pb[(l, 1, t)] = dist[t][blank_idx] * (pb[(l, 1, t - 1)] + pnb[(l, 1, t - 1)])
                    A_next.append(l)
                    if not (l, 1, t) in pnb:
                        pnb[(l, 1, t)] = 0
                else:
                    l_ = l + c

                    # 'c = l_end' case
                    if len(l) > 0 and c == l[-1]:
                        pnb[(l_, 1, t)] = dist[t][c_idx] * pb[(l, 1, t - 1)]
                        pnb[(l, 1, t)] = dist[t][c_idx] * pb[(l, 1, t - 1)]

                    # 'c = space' case
                    # TODO: add LM support

                    # 'otherwise' case
                    else:
                        pnb[(l_, 1, t)] = dist[t][c_idx] * (pb[(l, 1, t - 1)] + pnb[(l, 1, t - 1)])
                        if not (l_, 1, t) in pnb:
                            pnb[(l_, 1, t)] = 0

                    if l_ not in A_prev:
                        pb[(l_, 1, t)] = dist[t][blank_idx] * (pb[(l_, 1, t - 1)] + pnb[(l_, 1, t - 1)])
                        pnb[(l_, 1, t)] = dist[t][c_idx] * pnb[l_, 1, t - 1]

                    A_next.append(l_)

        # A_prev = A_next  # TODO
        A_prev.extend(A_next)  # TODO
    print A_prev


def padd(*probs):
    return reduce(np.logaddexp, probs)

def pmul(*probs):
    return reduce(op.add, probs)

def pexp(log_base, exp):
    return exp * log_base

def pstr(prob):
    return '{:.4g}'.format(np.exp(prob))

class PrefixNode(object):
    def __init__(self, prefix='', pnb=p0, pb=p0):
        self.prefix = prefix
        self.pnb = pnb
        self.pb = pb

    def __repr__(self):
        return 'PrefixNode("{}", {}, {})'.format(self.prefix, self.pnb, self.pb)

    def ptotal(self):
        return padd(self.pnb, self.pb)


def check_last_word(l_plus, lexicon):
    words = l_plus.split()
    if len(words) > 0 and words[-1] in lexicon:
        return p1
    else:
        return np.log(0.5)


def decoder(x, alphabet, blank, beam=20, alpha=1.0, beta=.1):
    # probs = np.log(x)
    probs = x

    T, N = probs.shape

    # this is used to rank prefixes for beam search selection
    prefix_prob = lambda (l, prefix): np.exp(prefix.ptotal()) * ((len(l.split()) + 1) ** beta)

    # new entries are initialized with default PrefixNode args
    A_old = defaultdict(lambda: PrefixNode())

    # initialize A_next with empty string
    A_curr = [('', PrefixNode(prefix='', pnb=p0, pb=p1))]

    # loop over time
    for t in xrange(T):
        A_curr = dict(A_curr)
        A_next = defaultdict(lambda: PrefixNode())

        for l, curr_prefix in A_curr.iteritems():
            # add l to A_{next}
            same_prefix = A_next[l]
            same_prefix.prefix = l
            # handle blanks
            same_prefix.pb = padd(pmul(probs[t, alphabet.index(blank)], curr_prefix.ptotal()), same_prefix.pb)
            if len(l) > 0:
                # handle repeat character collapsing
                same_prefix.pnb = padd(pmul(probs[t, alphabet.index(l[-1])], curr_prefix.pnb), same_prefix.pnb)

            # i is the index of the character c in the alphabet / neural network output
            for i, c in enumerate(alphabet):
                if c == blank:
                    # we already took care of this outside the loop
                    continue
                # l^{plus} <- concatenate l and c
                l_plus = l + c
                # add l^{plus} to A_{next}
                new_prefix = A_next[l_plus]
                new_prefix.prefix = l_plus

                # apply language model
                # if c == ' ':
                #     lm_prob = pexp(check_last_word(l_plus, lexicon), alpha)
                # else:
                #     lm_prob = p1
                lm_prob = p1  # TODO: in case no LM

                # if l is empty string or c is not repeat character
                if len(l) == 0 or (len(l) > 0 and c != l[-1]):
                    # add a character
                    new_prefix.pnb = padd(pmul(lm_prob, probs[t, i], curr_prefix.ptotal()), new_prefix.pnb)
                else:
                    # repeats must have blank between them
                    new_prefix.pnb = padd(pmul(lm_prob, probs[t, i], curr_prefix.pb), new_prefix.pnb)

                if l_plus not in A_curr:
                    old_prefix = A_old[l_plus]
                    # handle blank
                    new_prefix.pb = padd(pmul(probs[t, alphabet.index(blank)], old_prefix.ptotal()), new_prefix.pb)
                    # handle collapsing
                    new_prefix.pnb = padd(pmul(probs[t, i], old_prefix.pnb), new_prefix.pnb)

                if len(l) > 0 and i == l[-1]:
                    # TODO: check if curr_prefix.pb vs curr_prefix.pnb should be used
                    same_prefix.pb = padd(pmul(probs[t, i], curr_prefix.pb), same_prefix.pb)

        A_old = A_next

        A_curr = sorted(A_next.iteritems(), key=prefix_prob, reverse=True)[:beam]

    return A_curr[0][0]

if __name__ == '__main__':
    dist = np.array([[0.2, 0.3, 0.5],
                            [0.4, 0.4, 0.2],
                            [0.1, 0.3, 0.6],
                            [0.5, 0.2, 0.3]])
    print(decoder(dist, '_ab', '_'))