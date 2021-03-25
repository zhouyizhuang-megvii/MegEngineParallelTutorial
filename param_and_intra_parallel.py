import os
import megengine
import megengine.distributed as dist
import megengine.functional as F
import megengine.module as M

def transpose_mp2dp(x, k=None):
    k = k if k else dist.get_world_size()
    n, c, *shp = x.shape
    x = dist.functional.all_to_all(x)
    x = x.reshape(k, n // k, -1)\
         .transpose(1, 0, 2)\
         .reshape(n // k, c * k, *shp)
    return x

def transpose_dp2mp(x, k=None):
    k = k if k else dist.get_world_size()
    n, c, *shp = x.shape
    x = x.reshape(n, k, -1)\
         .transpose(1, 0, 2)\
         .reshape(n * k, c // k, *shp)
    x = dist.functional.all_to_all(x)
    return x

class Conv2dStyleA(M.Conv2d):
    """ParamParallel"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = dist.functional.scatter(self.weight)

    def calc_conv(self, x, weight, bias):
        weight = dist.functional.all_gather(weight)
        return super().calc_conv(x, weight, bias)

    ##Inherit the default implementation
    # def forward(self, inp):
    #     return self.calc_conv(inp, self.weight, self.bias)

class Conv2dStyleB(M.Conv2d):
    """Intra-StyleB"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = dist.functional.scatter(self.weight)

    def calc_conv(self, x, weight, bias):
        x = dist.functional.all_gather(x)
        x = super().calc_conv(x, weight, None)
        x = transpose_mp2dp(x)
        #x = dist.functional.all_to_all(x)

        #nk, c_k, h, w = x.shape
        #k = dist.get_world_size()
        #x = x.reshape(k, nk//k, c_k, h, w).transpose(1, 0, 2, 3, 4)
        #x = x.reshape(nk//k, c_k*k, h, w)

        if bias is not None:
            x += bias.reshape(1, self.out_channels, 1, 1)
        return x


class Conv2dStyleC(M.Conv2d):
    """Intra StyleC"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = dist.functional.scatter(self.weight.transpose(1,0,2,3)).transpose(1,0,2,3)

    def calc_conv(self, x, weight, bias):
        #n, c, h, w = x.shape
        #k = dist.get_world_size()
        #x = x.reshape(n, k, self.in_channels // k, -1).transpose(1, 0, 2, 3)
        #x = dist.functional.all_to_all(x)
        #x = x.reshape(-1, self.in_channels // k, h, w)
        x = transpose_dp2mp(x)

        x = super().calc_conv(x, weight, None)
        x = dist.functional.reduce_scatter_sum(x)

        if bias is not None:
            x += bias.reshape(1, self.out_channels, 1, 1)
        return x

@dist.launcher
def main():
    import numpy as np
    import megengine.autodiff as ad
    import itertools
    os.environ["NCCL_LAUNCH_MODE"] = "PARALLEL"
    I, O = 16, 32
    w = megengine.tensor(np.random.randn(O, I, 3, 3))
    w = dist.functional.broadcast(w)  # sync
    w_by_rows = dist.functional.scatter(w)
    w_by_cols = dist.functional.scatter(w.transpose(1,0,2,3)).transpose(1,0,2,3)

    b = megengine.tensor(np.random.randn(1, O, 1, 1))
    b = dist.functional.broadcast(b)  # sync

    conv  = M.Conv2d(I, O, 3, padding=1)#, bias=False)
    conv.weight[:] = w
    conv.bias[:] = b
    convA = Conv2dStyleA(I, O, 3, padding=1)#, bias=False)
    convA.weight[:] = w_by_rows
    convA.bias[:] = b
    convB = Conv2dStyleB(I, O, 3, padding=1)#, bias=False)
    convB.weight[:] = w_by_rows
    convB.bias[:] = b
    convC = Conv2dStyleC(I, O, 3, padding=1)#, bias=False)
    convC.weight[:] = w_by_cols
    convC.bias[:] = b
    for n, p in convA.named_parameters():
        print(n, p.shape)

    x = megengine.tensor(np.random.randn(2, I, 8, 8))
    #y = conv(x)
    #yA = convA(x)
    #yB = convB(x)
    #yC = convC(x)
    #print(((yA - y) ** 2).mean())
    #print(((yB - y) ** 2).mean())
    #print(((yC - y) ** 2).mean())

    def model_parallel_scale_grad_cb(p, g):
        return g / dist.get_world_size()

    gm = ad.GradManager()
    # attach conv
    gm.attach(conv.parameters())
    # attach conv of model parallelism
    for n, p in itertools.chain(
            convA.named_parameters(),
            convB.named_parameters(),
            convC.named_parameters(),
        ):
        if "weight" in n:
            # 一：不做allreduce；2：rescale梯度
            gm.attach(p, model_parallel_scale_grad_cb)
        else:
            # 其他参数正常allreduce
            gm.attach(p, dist.make_allreduce_cb("MEAN"))
    with gm:
        y = conv(x)
        gm.backward(y.mean())
    with gm:
        yA = convA(x)
        gm.backward(yA.mean())
    with gm:
        yB = convB(x)
        gm.backward(yB.mean())
    with gm:
        yC = convC(x)
        gm.backward(yC.mean())

    gradA = dist.functional.all_gather(convA.weight.grad)
    gradB = dist.functional.all_gather(convB.weight.grad)
    gradC = dist.functional.all_gather(convC.weight.grad.transpose(1,0,2,3)).transpose(1,0,2,3)
    assert np.allclose(y, yA)
    assert np.allclose(conv.weight.grad, gradA)
    assert np.allclose(y, yB, rtol=1e-3, atol=1e-4), F.abs((y - yB)).max()
    assert np.allclose(conv.weight.grad, gradB)
    assert np.allclose(y, yC, rtol=1e-3, atol=1e-4), F.abs((y - yC)).max()
    assert np.allclose(conv.weight.grad, gradC)


if __name__ == "__main__":
    main()
