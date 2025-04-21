import torch
import os
import argparse
import model
import torch.nn.functional as F
import numpy as np
from PIL import Image

def load_image(path):
    image = Image.open(path)
    image = image.convert('RGB') # (h,w,c)
    image = (np.asarray(image)/255.0) 
    image = torch.from_numpy(image).float()
    return image.permute(2,0,1) # (c,h,w) 

def eval(config):
    paired_image_names = [
        ['2037.jpg', '2037.jpg'],
        ['2037.jpg', '2038.jpg'],
    ]
    # load model
    net = model.RIENet(config).to(torch.device('cuda:2'))
    net.load_state_dict(torch.load(config.model_path))
    # build dataset
    # eval
    net.eval()
    for paired_image_name in paired_image_names:
        A_image_name, B_image_name = paired_image_name[0], paired_image_name[1]
        A_image_path = os.path.join(config.HI_path, A_image_name)
        B_image_path = os.path.join(config.LI_path, B_image_name)
        A_image = load_image(A_image_path).unsqueeze(dim=0).to(torch.device('cuda:2'))
        B_image = load_image(B_image_path).unsqueeze(dim=0).to(torch.device('cuda:2'))

        b,c,h_image,w_image = A_image.shape
        w_pad = 0 if w_image%8 == 0 else 8-w_image%8
        h_pad = 0 if h_image%8 == 0 else 8-h_image%8
        left = w_pad//2
        right = w_pad - left
        top = h_pad//2
        bottom = h_pad - top
        A_image = F.pad(A_image, (left, right, top, bottom))
        B_image = F.pad(B_image, (left, right, top, bottom))
        A_image = A_image.to(torch.device('cuda:2'))
        B_image = B_image.to(torch.device('cuda:2'))
        x_enhance, h, z_gt, _ = net(B_image, A_image)
        x_enhance = x_enhance[:,:,top:h_image+top, left:w_image+left]
        saveimage_numpy = x_enhance[0, ...].permute(1,2,0).cpu().detach().numpy()*255
        saveimage_numpy = np.uint8(np.clip(saveimage_numpy, 0,255))
        saveimage_pil = Image.fromarray(saveimage_numpy)
        saveimage_pil.save(os.path.join(config.output_path, A_image_name+'_'+B_image_name+'.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--output_path', type=str, default="./output")
    parser.add_argument('--w_h', type=float, default=0.01)
    parser.add_argument('--w_z', type=float, default=0.1)
    parser.add_argument('--w_color', type=float, default=0.0)
    parser.add_argument('--lambda_', type=float, default=0.8)
    parser.add_argument('--warm_up', type=float, default=100.0)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--model_path', type=str, default="./weights/LOL.pth")
    parser.add_argument('--device', type=str, default= "2")
    config = parser.parse_args()
    print(config)
    eval(config)
