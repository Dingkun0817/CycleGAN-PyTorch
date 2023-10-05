import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from models import Discriminator, Generator
from utils import ReplayBuffer, LambdaLR, weights_init_normal
from dataset import ImageDataset
import tensorboardX
import os
from torchvision.utils import save_image
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    netG_A2B = Generator().to(device)
    netG_B2A = Generator().to(device)

    netG_A2B.load_state_dict(torch.load("models/netG_A2B.pth")) # 加载对应参数
    netG_B2A.load_state_dict(torch.load("models/netG_B2A.pth"))

    netG_A2B.eval()
    netG_B2A.eval()

    size = 256
    input_A = torch.ones([1, 3, size, size], dtype=torch.float).to(device)
    input_B = torch.ones([1, 3, size, size], dtype=torch.float).to(device)

    transforms_ = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    data_root = '../../data/apple2orange'
    dataloader = DataLoader(ImageDataset(data_root, transforms_, "test"),
                            batch_size=1, shuffle=False, num_workers=8)

    if not os.path.exists("outputs/A"):
        os.makedirs("outputs/A")

    if not os.path.exists("outputs/B"):
        os.makedirs("outputs/B")

    for i, batch in enumerate(dataloader):
        real_A = torch.tensor(input_A.copy_(batch['A']), dtype=torch.float).to(device)
        real_B = torch.tensor(input_B.copy_(batch['B']), dtype=torch.float).to(device)

        fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)
        fake_A = 0.5 * (netG_B2A(real_B).data + 1.0)

        save_image(fake_A, "outputs/A/{}.png".format(i))
        save_image(fake_B, "outputs/B/{}.png".format(i))
        print(i)