import torch
import torch.nn as nn
from efficientnet import make_EfficientNet
from Config import config
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

"""
    Make Model
"""
width_coefs = [1., 1., 1.1, 1.2, 1.4, 1.6, 1.8, 2.0]
height_coefs = [1., 1.1, 1.2, 1.4, 1.8, 2.2, 2.6, 3.1]
scales = [224, 240, 260, 300, 380, 456, 528, 600]
dropout = [0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5]
scales_ratios = [i/224 for i in scales]


model = make_EfficientNet(
    config.num_classes,
    width = width_coefs[config.efficientnet_num],
    height = height_coefs[config.efficientnet_num],
    scale= scales_ratios[config.efficientnet_num],
    dropout= dropout[config.efficientnet_num],
    se_scale=config.se_scale,
    stochastic_depth=config.stochastic_depth,
    p=config.p
)



USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print(DEVICE)
model = model.to(DEVICE)
# model.load_state_dict(torch.load(config.save_path))


"""
    Load CIFAR 10
"""

transform = transforms.Compose([transforms.ToTensor()])

dataset = datasets.CIFAR100(root= './.data', transform = transform,train = True, download = True)

train_set, val_set = torch.utils.data.random_split(dataset, [40000,10000])

train_loader = DataLoader(train_set, config.batch_size, shuffle =True)
val_loader = DataLoader(val_set, config.batch_size, shuffle =True)


"""
    Train
"""

from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm.notebook import tqdm
import pickle

criterion = nn.CrossEntropyLoss(reduction='mean')
opt = torch.optim.Adam(model.parameters(), lr=config.lr)


# get current lr
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


train_history = []
val_history = []
train_acc_history = []
val_acc_history = []

lr_scheduler = None

if config.use_ReduceLROnPlateau:
    lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10)

train_length = 40000
val_length = 10000

best_loss = float('inf')

for epoch in range(config.epochs):
    model.train()
    train_losses = []
    train_epoch_acc = 0
    now_lr = get_lr(opt)
    print("Epcohs {}/{} \t Now lr : {}".format(epoch + 1, config.epochs, now_lr))
    for x, y in tqdm(train_loader):
        batch_size = x.size(0)
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        output = model(x)

        train_loss = criterion(output, y)
        train_losses.append(train_loss.item())

        opt.zero_grad()
        train_loss.backward()
        opt.step()

        _, pred_idx = torch.max(output, 1)

        train_epoch_acc += (pred_idx == y).sum().item()

    this_epoch_train_loss = np.mean(train_losses)
    train_history.append(this_epoch_train_loss)
    train_acc_history.append(train_epoch_acc / train_length * 100)

    model.eval()
    val_losses = []
    val_epoch_acc = 0
    with torch.no_grad():
        for val_x, val_y in tqdm(val_loader):
            val_x = val_x.to(DEVICE)
            val_y = val_y.to(DEVICE)

            val_output = model(val_x)
            val_loss = criterion(val_output, val_y)

            val_losses.append(val_loss.item())

            _, pred_idx = torch.max(output, 1)

            val_epoch_acc += (pred_idx == y).sum().item()

    if config.use_ReduceLROnPlateau:
        lr_scheduler.step(val_loss)

    this_epoch_val_loss = np.mean(val_losses)
    val_history.append(this_epoch_val_loss)
    val_acc_history.append(val_epoch_acc / val_length * 100)

    if (epoch + 1) % 10 == 0:
        with open(config.val_loss_history_path, "wb") as fw:
            pickle.dump(val_history, fw)
        with open(config.train_loss_history_path, "wb") as fw:
            pickle.dump(train_history, fw)
        with open(config.val_acc_history_path, "wb") as fw:
            pickle.dump(val_acc_history, fw)
        with open(config.train_acc_history_path, "wb") as fw:
            pickle.dump(train_acc_history, fw)

    if best_loss > this_epoch_val_loss:
        torch.save(model.state_dict(), config.save_path)
        best_loss = this_epoch_val_loss
        print("{} Epoch. Save Model".format(epoch + 1))

    print("Train Loss : {:.3f} \t Val Loss : {:.3f} \t Accuracy : {:.3f}\n\n".format(this_epoch_train_loss,
                                                                                     this_epoch_val_loss,
                                                                                     train_epoch_acc / train_length * 100))







