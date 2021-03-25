import megengine
import megengine.module as M
import megengine.functional as F
import megengine.distributed as dist
import megengine.autodiff as ad
import megengine.optimizer as optim
import numpy as np

def send_to_next_gpu(tensor):
    shape, dtype = tensor.shape, np.dtype(tensor.dtype).name
    dist.get_client().user_set(f"shape_of_src{dist.get_rank()}", shape)
    dist.get_client().user_set(f"dtype_of_src{dist.get_rank()}", dtype)
    return F.distributed.remote_send(tensor, dest_rank=dist.get_rank() + 1)

def recv_fr_prev_gpu():
    shape = dist.get_client().user_get(f"shape_of_src{dist.get_rank() - 1}")
    dtype = dist.get_client().user_get(f"dtype_of_src{dist.get_rank() - 1}")
    return F.distributed.remote_recv(src_rank=dist.get_rank() - 1, shape=shape, dtype=dtype)

def grad_to_prev_gpu(tensor):
    shape, dtype = tensor.shape, np.dtype(tensor.dtype).name
    dist.get_client().user_set(f"grad_shape_of_src{dist.get_rank()}", shape)
    dist.get_client().user_set(f"grad_dtype_of_src{dist.get_rank()}", dtype)
    return F.distributed.remote_send(tensor, dest_rank=dist.get_rank() - 1)

def grad_fr_next_gpu():
    shape = dist.get_client().user_get(f"grad_shape_of_src{dist.get_rank() + 1}")
    dtype = dist.get_client().user_get(f"grad_dtype_of_src{dist.get_rank() + 1}")
    return F.distributed.remote_recv(src_rank=dist.get_rank() + 1, shape=shape, dtype=dtype)

class BasicBlock(M.Module):
    def __init__(self, inp, oup, stride):
        super().__init__()
        if inp == oup and stride == 1:
            self.proj = M.Identity()
        else:
            self.proj = M.ConvBn2d(inp, oup, 1, stride=stride, bias=False)
        self.conv1 = M.ConvBnRelu2d(inp, oup, 3, padding=1, stride=stride, bias=False)
        self.conv2 = M.ConvBn2d(oup, oup, 3, padding=1, stride=1, bias=False)

    def forward(self, x):
        proj = self.proj(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.relu(proj + x)
        return x

class ResNet18(M.Module):
    def __init__(self):
        super().__init__()
        self.stem = M.Sequential(
            M.ConvBn2d(3, 64, 7, stride=2, padding=3, bias=False),
            M.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.features = M.Sequential(
            BasicBlock(64,  64,  1),
            BasicBlock(64,  64,  1),
            BasicBlock(64,  128, 2),
            BasicBlock(128, 128, 1),
            BasicBlock(128, 256, 2),
            BasicBlock(256, 256, 1),
            BasicBlock(256, 512, 2),
            BasicBlock(512, 512, 1),
        )
        self.classifier = M.Linear(512, 1000)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = F.avg_pool2d(x, 7)
        x = F.flatten(x, 1)
        x = self.classifier(x)
        return x

class ResNet18MP(M.Module):
    def __init__(self):
        super().__init__()
        self.classifier = None
        if dist.get_rank() == 0:
            self.features = M.Sequential(
                M.ConvBn2d(3, 64, 7, stride=2, padding=3, bias=False),
                M.MaxPool2d(kernel_size=3, stride=2, padding=1),
                BasicBlock(64,  64,  1),
                BasicBlock(64,  64,  1),
            )
        elif dist.get_rank() == 1:
            self.features = M.Sequential(
                BasicBlock(64,  128, 2),
                BasicBlock(128, 128, 1),
            )
        elif dist.get_rank() == 2:
            self.features = M.Sequential(
                BasicBlock(128, 256, 2),
                BasicBlock(256, 256, 1),
            )
        elif dist.get_rank() == 3:
            self.features = M.Sequential(
                BasicBlock(256, 512, 2),
                BasicBlock(512, 512, 1),
            )
            self.classifier = M.Linear(512, 1000)

    def forward(self, x):
        if dist.get_rank() > 0:
            x = recv_fr_prev_gpu()
        x = self.features(x)
        if dist.get_rank() != 3:
            _ = send_to_next_gpu(x)
        else:
            x = F.avg_pool2d(x, 7)
            x = F.flatten(x, 1)
            x = self.classifier(x)
        return x


class ResNet18Pipeline(M.Module):
    def __init__(self):
        super().__init__()
        self.classifier = None
        if dist.get_rank() == 0:
            self.features = M.Sequential(
                M.ConvBn2d(3, 64, 7, stride=2, padding=3, bias=False),
                M.MaxPool2d(kernel_size=3, stride=2, padding=1),
                BasicBlock(64,  64,  1),
                BasicBlock(64,  64,  1),
            )
        elif dist.get_rank() == 1:
            self.features = M.Sequential(
                BasicBlock(64,  128, 2),
                BasicBlock(128, 128, 1),
            )
        elif dist.get_rank() == 2:
            self.features = M.Sequential(
                BasicBlock(128, 256, 2),
                BasicBlock(256, 256, 1),
            )
        elif dist.get_rank() == 3:
            self.features = M.Sequential(
                BasicBlock(256, 512, 2),
                BasicBlock(512, 512, 1),
            )
            self.classifier = M.Linear(512, 1000)

    def forward(self, x):
        self.num_chunks = 4
        self.inp_chunks = []
        self.oup_chunks = []
        if dist.get_rank() == 0:
            self.inp_chunks = F.split(x, 4)
        
        for i in range(self.num_chunks):
            if dist.get_rank() == 0:
                x = self.inp_chunks[i]
            else:
                x = recv_fr_prev_gpu()
                self.inp_chunks.append(x)

            x = self.features(x)
            if dist.get_rank() != 3:
                _ = send_to_next_gpu(x)
            else:
                x = F.avg_pool2d(x, 7)
                x = F.flatten(x, 1)
                x = self.classifier(x)
            self.oup_chunks.append(x)

        return F.concat(self.oup_chunks)

    def backward(self, label, gm):
        label_chunks = F.split(label, 4)
        losses = []

        for i, x in enumerate(self.inp_chunks):
            with gm:#ad.GradManager().attach(self.parameters()) as gm:
                gm.attach(x)  # query gradient of the input
                y = self.features(x)

                if dist.get_rank() == 3:
                    y = F.avg_pool2d(y, 7)
                    y = F.flatten(y, 1)
                    y = self.classifier(y)
                    loss = F.nn.cross_entropy(y, label_chunks[i])
                    losses.append(loss)
                    gm.backward(loss)
                else:
                    grad = grad_fr_next_gpu()
                    gm.backward(y, dy=grad)

                if dist.get_rank() != 0:
                    _ = grad_to_prev_gpu(x.grad)

        return sum(losses) / self.num_chunks if losses else None


@dist.launcher(n_gpus=4)
def inference():
    m = ResNet18MP()
    x = F.ones([32, 3, 224, 224])
    y = m(x)
    print(y.shape)


@dist.launcher(n_gpus=4)
def train():
    m = ResNet18MP()
    x = F.ones([32, 3, 224, 224])
    label = F.zeros([32,], dtype="int32")

    gm = ad.GradManager().attach(m.parameters())
    opt = optim.SGD(m.parameters(), 1e-3, 0.9, 1e-4)

    for _ in range(2):
        with gm:
            y = m(x)
            if dist.get_rank() == 3:
                loss = F.nn.cross_entropy(y, label)
            else:
                loss = None
            gm.backward(loss)
        opt.step().clear_grad()
        print(loss)
    
@dist.launcher(n_gpus=4)
def inference_pipeline():
    m = ResNet18Pipeline()
    x = F.ones([32, 3, 224, 224])
    y = m(x)
    print(y.shape)

@dist.launcher(n_gpus=4)
def train_pipeline():
    m = ResNet18Pipeline()
    x = F.ones([32, 3, 224, 224])
    label = F.zeros([32,], dtype="int32")

    gm = ad.GradManager().attach(m.parameters())
    opt = optim.SGD(m.parameters(), 1e-3, 0.9, 1e-4)

    for _ in range(2):
        m(x)
        loss = m.backward(label, gm)
        opt.step().clear_grad()
        print(loss)


if __name__ == "__main__":
    inference()
    train()
    inference_pipeline()
    train_pipeline()
