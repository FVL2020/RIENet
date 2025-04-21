import shutil
import torch
import torch.nn as nn
import torchvision
import torch.optim
import os
import argparse
import dataset
import model

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)


def train(config):
    # tensorboard
    if os.path.exists(os.path.join(config.snapshots_folder, 'tb')):
        shutil.rmtree(os.path.join(config.snapshots_folder, 'tb'))
    if not os.path.exists(os.path.join(config.snapshots_folder, 'sample')):
        os.makedirs(os.path.join(config.snapshots_folder, 'sample'))
    os.makedirs(os.path.join(config.snapshots_folder, 'tb'))
    # load model
    os.environ['CUDA_VISIBLE_DEVICES']=config.device
    net = model.RIENet(config).cuda()
    net.apply(weights_init)
    if config.load_pretrain == True:
        net.load_state_dict(torch.load(config.pretrain_dir))
    if len(config.device)>1:
        net = nn.DataParallel(net)
    # build dataset
    train_dataset = dataset.LOL_dataset(config.LI_path, config.HI_path, config.histeq)	if config.dataset=='lol' else dataset.EE_dataset(config.LI_path, config.HI_path, config.histeq)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)  # adjust beta1 to momentum

    # train
    net.train()
    for epoch in range(config.num_epochs):
        for iteration, data in enumerate(train_loader):
            LI, HI = data
            LI = LI.cuda()
            HI = HI.cuda()
            loss, loss_h, loss_rec, loss_z = net.loss(LI, HI)
            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # log
            if ((iteration+1) % config.log_iter) == 0:
                print(f"epoch: {epoch+1}, iter: {iteration+1}, loss: {loss.item()}, loss_h: {loss_h.item()}, loss_rec: {loss_rec.item()}, loss_z: {loss_z.item()}")

        if epoch%100 == 0:
            x_enhance, h, z_gt = net(LI, HI)
            torchvision.utils.save_image(torchvision.utils.make_grid(x_enhance), os.path.join(config.snapshots_folder, 'sample', f'{epoch}_output.png'))
            torchvision.utils.save_image(torchvision.utils.make_grid(h/2+0.5), os.path.join(config.snapshots_folder, 'sample', f'{epoch}h.png'))
            torchvision.utils.save_image(torchvision.utils.make_grid(LI), os.path.join(config.snapshots_folder, 'sample', f'{epoch}_input.png'))
            torchvision.utils.save_image(torchvision.utils.make_grid(HI), os.path.join(config.snapshots_folder, 'sample', f'{epoch}_GT.png'))
        # save param
        if ((epoch+1) % config.snapshot_iter) == 0:
            torch.save(net.module.state_dict() if len(config.device)>1 else net.state_dict(), os.path.join(config.snapshots_folder, "Epoch" + str(epoch+config.start_epoch) + '.pth')) 		

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--LI_path', type=str, default="./LOL_v2/Real_captured/train/input")
    parser.add_argument('--HI_path', type=str, default="./LOL_v2/Real_captured/train/gt")
    parser.add_argument('--dataset', type=str, default="lol", help='lol|EE')
    parser.add_argument('--w_h', type=float, default=0.01)
    parser.add_argument('--w_z', type=float, default=0.1)
    parser.add_argument('--lambda_', type=float, default=0.8)
    parser.add_argument('--warm_up', type=float, default=100.0)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=3000)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=100)
    parser.add_argument('--snapshots_folder', type=str, default="./snapshots")
    parser.add_argument('--load_pretrain', action='store_true', help='load pretrained weights or not')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--pretrain_dir', type=str, default= "")
    parser.add_argument('--device', type=str, default= "0")
    config = parser.parse_args()
    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)
    print(config)
    train(config)
