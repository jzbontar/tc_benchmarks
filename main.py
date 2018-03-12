import argparse
import timeit
# Hack the timeit module by injecting cuda.synchronize() before the two timer calls.
timeit.template = timeit.template.replace('    _t', '    cuda.synchronize(); _t')

import torch
from torch import nn, cuda
from torch.autograd import Variable
import tensor_comprehensions as tc

parser = argparse.ArgumentParser()
parser.add_argument('--generations', type=int, default=25)
parser.add_argument('--pop-size', type=int, default=100)
parser.add_argument('--number-elites', type=int, default=10)
args = parser.parse_args()

def compare(stmt1, stmt2, repeat, number):
    o1 = eval(stmt1)
    o2 = eval(stmt2)
    print((o1.view(-1) - o2.view(-1)).abs().max().data[0], 'diff')
    import builtins
    builtins.__dict__.update(globals())
    print(min(timeit.repeat(stmt1, repeat=repeat, number=number)), stmt1)
    print(min(timeit.repeat(stmt2, repeat=repeat, number=number)), stmt2)
    print()

def tmm(M, K, N, **compare_kwargs):
    global A, B, tc_tmm
    print('tmm(M={}, K={}, N={})'.format(M, K, N))
    A = Variable(torch.Tensor(M, K).cuda().normal_())
    B = Variable(torch.Tensor(N, K).cuda().normal_())
    tc_tmm = tc.define('''
    def tmm(float(M, K) A, float(N, K) B) -> (C) {
        C(m, n) +=! A(m, kk) * B(n, kk)
    }''', name='tmm')
    tc_tmm.autotune(A, B, options=tc.Options('mlp'), **autotune_kwargs)
    compare('tc_tmm(A, B)', 'torch.mm(A, B.t())', **compare_kwargs)

def tbmm(B, M, K, N, **compare_kwargs):
    global X, Y, tc_tbmm
    print('tbmm(B={}, M={}, K={}, N={})'.format(B, M, K, N))
    X = Variable(torch.Tensor(B, N, M).cuda().normal_())
    Y = Variable(torch.Tensor(B, K, M).cuda().normal_())
    tc_tbmm = tc.define('''
    def tbmm(float(B, N, M) X, float(B, K, M) Y) -> (Z) {
        Z(b, n, k) +=! X(b, n, m) * Y(b, k, m)
    }''', name='tbmm')
    tc_tbmm.autotune(X, Y, options=tc.Options('mlp'), **autotune_kwargs)
    compare('tc_tbmm(X, Y)', 'torch.bmm(X, Y.transpose(1, 2))', **compare_kwargs)

def gconv(N, G, F, C, W, H, KH, KW, **compare_kwargs):
    global tc_I, tc_W1, tc_gconv, nn_I, nn_gconv
    print('gconv(N={}, G={}, F={}, C={}, W={}, H={}, KH={}, KW={})'.format(N, G, F, C, W, H, KH, KW))
    tc_I = Variable(torch.Tensor(N, G, C, H, W).cuda().normal_())
    nn_I = tc_I.view(N, G * C, H, W)
    nn_gconv = nn.Conv2d(G * C, G * F, (KH, KW), groups=G, bias=False).cuda()
    tc_W1 = nn_gconv.weight.view(G, F, C, KH, KW)
    tc_gconv = tc.define('''
    def gconv(float(N, G, C, H, W) I, float(G, F, C, KW, KW) W1) -> (O) {
        O(n, g, o, h, w) +=! I(n, g, i, h + kh, w + kw) * W1(g, o, i, kh, kw)
    }''', name='gconv')
    tc_gconv.autotune(tc_I, tc_W1, options=tc.Options('group_conv'), **autotune_kwargs)
    compare('tc_gconv(tc_I, tc_W1)', 'nn_gconv(nn_I)', **compare_kwargs)

torch.backends.cudnn.benchmark = True
autotune_kwargs = dict(generations=args.generations, pop_size=args.pop_size, number_elites=args.number_elites)

tmm(128, 32, 256, repeat=100, number=10000)
tmm(128, 1024, 1024, repeat=10, number=1000)
tmm(128, 4096, 16384, repeat=10, number=10)

tbmm(500, 72, 26, 26, repeat=100, number=10000)

gconv(32, 32, 16, 16, 14, 14, 3, 3, repeat=10, number=1000)
gconv(32, 32, 32, 32, 7, 7, 3, 3, repeat=10, number=1000)
gconv(32, 32, 4, 4, 56, 56, 3, 3, repeat=10, number=1000)
gconv(32, 32, 8, 8, 28, 28, 3, 3, repeat=10, number=1000)
