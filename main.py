import argparse
import timeit
# Hack the timeit module by injecting cuda.synchronize() before the two timer calls.
timeit.template = timeit.template.replace('    _t', '    cuda.synchronize(); _t')

import torch
from torch import nn, cuda
from torch.autograd import Variable
import tensor_comprehensions as tc

parser = argparse.ArgumentParser()
parser.add_argument('--generations', type=int, default=2)
parser.add_argument('--pop-size', type=int, default=10)
parser.add_argument('--number-elites', type=int, default=1)
args = parser.parse_args()

def compare(stmt1, stmt2, repeat, number):
    o1 = eval(stmt1)
    o2 = eval(stmt2)
    print((o1 - o2).abs().max().data[0], 'output diff')
    import builtins
    builtins.__dict__.update(globals())
    print(min(timeit.repeat(stmt1, repeat=repeat, number=number)), stmt1)
    print(min(timeit.repeat(stmt2, repeat=repeat, number=number)), stmt2)
    print()

tests = ['tbmm']
autotune_kwargs = dict(
    generations=args.generations, 
    pop_size=args.pop_size, 
    number_elites=args.number_elites)

# matrix-vector multiplication
if 'mv' in tests:
    m, k = 256, 128
    A = Variable(torch.Tensor(m, k).cuda().normal_())
    x = Variable(torch.Tensor(k).cuda().normal_())
    mv = tc.define('''
    def mv(float(n, m) A, float(m) x) -> (y) {
        y(i) +=! A(i, j) * x(j)
    }
    ''', name='mv')
    mv.autotune(A, x, options=tc.Options('mlp'), **autotune_kwargs)
    compare('mv(A, x)', 'torch.mv(A, x)', repeat=10, number=10000)

# matrix-matrix multiplication
if 'mm' in tests:
    n, m, k = 128, 256, 128
    A = Variable(torch.Tensor(n, m).cuda().normal_())
    B = Variable(torch.Tensor(m, k).cuda().normal_())
    mm = tc.define('''
    def mm(float(N, M) A, float(M, K) B) -> (C) {
        C(i, j) +=! A(i, k) * B(k, j)
    }
    ''', name='mm')
    mm.autotune(A, B, options=tc.Options('mlp'), **autotune_kwargs)
    compare('mm(A, B)', 'torch.mm(A, B)', repeat=10, number=10000)

# Liner + ReLU
if 'fcrelu' in tests:
    n_in, n_out, bs = 128, 256, 32
    X = Variable(torch.Tensor(bs, n_in).cuda().normal_())
    module = nn.Sequential(nn.Linear(n_in, n_out), nn.ReLU(True)).cuda()
    W = module[0].weight
    b = module[0].bias
    fcrelu = tc.define('''
    def fcrelu(float(bs, n_in) X, float(n_out, n_in) W, float(n_out) bias) -> (Y) {
        Y(i, j) +=! W(j, k) * X(i, k)
        Y(i, j) += bias(j)
        Y(i, j) = fmax(Y(i, j), 0)
    }''', name='fcrelu')
    fcrelu.autotune(X, W, b, options=tc.Options('mlp'), **autotune_kwargs)
    compare('fcrelu(X, W, b)', 'module(X)', repeat=10, number=10000)

if 'conv' in tests:
    torch.backends.cudnn.benchmark = True
    input_fm, output_fm, ks, w, h, bs = 3, 64, 3, 224, 224, 32
    input = Variable(torch.Tensor(bs, input_fm, h, w).cuda().normal_())
    nn_conv2d = nn.Conv2d(input_fm, output_fm, ks).cuda()
    weight = nn_conv2d.weight
    bias = nn_conv2d.bias
    tc_conv2d = tc.define('''
    def conv(float(bs, input_fm, h, w) input, float(output_fm, input_fm, ks, ks) weight, float(output_fm) bias) -> (out) {
        out(b, o_fm, i, j) +=! input(b, i_fm, i + ii, j + jj) * weight(o_fm, i_fm, ii, jj)
        out(b, o_fm, i, j) += bias(o_fm)
    }''', name='conv')
    tc_conv2d.autotune(input, weight, bias, options=tc.Options('conv'), **autotune_kwargs)
    compare('tc_conv2d(input, weight, bias)', 'nn_conv2d(input)', repeat=10, number=100)

