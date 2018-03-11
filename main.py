import time
import torch
import gc

from torch import nn, cuda
from torch.autograd import Variable
import tensor_comprehensions as tc

def repeat(stmt, repeat, number):
    template = '''def inner(_it):
        cuda.synchronize()
        _t0 = time.perf_counter()
        for _i in _it:
            {stmt}
        cuda.synchronize()
        _t1 = time.perf_counter()
        return _t1 - _t0'''
    code = compile(template.format(stmt=stmt), 'dummy_filename', 'exec')
    local_ns = {}
    exec(code, globals(), local_ns)
    inner = local_ns['inner']
    gc.disable()
    r = []
    for i in range(repeat):
        r.append(inner(range(number)))
    gc.enable()
    return r

def compare(stmt1, stmt2):
    o1 = eval(stmt1)
    o2 = eval(stmt2)
    print('eps', (o1 - o2).abs().max().data[0])
    kwargs = dict(repeat=10, number=10000)
    print(min(repeat(stmt1, **kwargs)), stmt1)
    print(min(repeat(stmt2, **kwargs)), stmt2)

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
fcrelu.autotune(I, W1, B1, cache=cache_file, options=options, generations=10, pop_size=20, number_elites=1)
fcrelu(I, W1, B1, cache=cache_file)

compare('fcrelu(I, W1, B1)', 'm(I)')
