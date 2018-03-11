import timeit
# Hack the timeit module by injecting cuda.synchronize() before the two timer calls.
timeit.template = timeit.template.replace('    _t', '    cuda.synchronize(); _t')

import torch
from torch import nn, cuda
from torch.autograd import Variable
import tensor_comprehensions as tc

def compare(stmt1, stmt2):
    o1 = eval(stmt1)
    o2 = eval(stmt2)
    print((o1 - o2).abs().max().data[0], 'output diff')
    kwargs = dict(repeat=100, number=1000)
    import builtins
    builtins.__dict__.update(globals())
    print(min(timeit.repeat(stmt1, **kwargs)), stmt1)
    print(min(timeit.repeat(stmt2, **kwargs)), stmt2)

B, M, N = 100, 128, 100
m = nn.Sequential(nn.Linear(M, N), nn.ReLU(True)).cuda()
 
lang = """
def fcrelu(float(B,M) I, float(N,M) W1, float(N) B1) -> (O1) {
   O1(b, n) +=! I(b, m) * W1(n, m)
   O1(b, n) = O1(b, n) + B1(n)
   O1(b, n) = fmax(O1(b, n), 0)
}
"""
fcrelu = tc.define(lang, name="fcrelu")
I = Variable(torch.randn(B, M).cuda())
W1 = m[0].weight
B1 = m[0].bias
options = tc.Options('mlp')
cache_file='fcrelu_100_128_100.tc'
# fcrelu.autotune(I, W1, B1, cache=cache_file, options=options, generations=10, pop_size=20, number_elites=1)
fcrelu(I, W1, B1, cache=cache_file)

compare('fcrelu(I, W1, B1)', 'm(I)')
