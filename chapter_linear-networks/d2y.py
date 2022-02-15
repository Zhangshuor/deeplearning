import torch
import math
import numpy as np

x = torch.tensor([math.pi / 2, math.pi], requires_grad=True)
y = x ** 3 + torch.cos(x)


true_dy = 3 * x ** 2 - torch.sin(x)
true_d2y = 6 * x - torch.cos(x)

# 求出一阶导数，保存计算图后，在去求二阶导数
dy = torch.autograd.grad(y, x,
                         grad_outputs=torch.ones(x.shape),
                         create_graph=True,
                         retain_graph=True)  # 为计算二阶导保持计算图
# 在张量后加上.detach().numpy()可以仅输出张量数值
print("一阶导真实值：{} \n一阶导计算值：{}".format(true_dy.detach().numpy(), dy[0].detach().numpy()))

# 求二阶导。上面的dy的第一个元素是一阶导数值
d2y = torch.autograd.grad(dy, x,
                          grad_outputs=torch.ones(x.shape),
                          create_graph=False  # 不再弄计算图，销毁前面的计算图
                          )
print("\n二阶导真实值：{} \n二阶导计算值：{}".format(true_d2y.detach().numpy(), d2y[0].detach().numpy()))
