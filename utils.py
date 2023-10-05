import random
import torch
import numpy as np


def tensor2image(tensor):
    image = 127.5 * (tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))
    return image.astype(np.uint8)


# class ReplayBuffer():
#     def __init__(self, max_size=50):
#         assert (max_size > 0), "max_size should be greater than 0"
#         self.max_size = max_size
#         self.data = []
#
#     def push_and_pop(self, data):
#         to_return = []
#         for element in data.data:
#             element = torch.unsqueeze(element, 0)
#             if len(self.data) < self.max_size:
#                 self.data.append(element)
#                 to_return.append(element)
#             else:
#                 if random.uniform(0, 1) > 0.5:
#                     i = random.randint(0, self.max_size - 1)
#                     to_return.append(self.data[i].clone())
#                     self.data[i] = element
#         return torch.cat(to_return)

class ReplayBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        if len(self.data) < self.max_size:
            self.data.append(data)
            return data
        else:
            # Pop the oldest data
            self.data.pop(0)
            self.data.append(data)
            return torch.cat(self.data)


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay mush start over 0"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):  # 权重衰退
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
