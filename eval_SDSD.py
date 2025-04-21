# eval on SDSD
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

@torch.no_grad()
def eval(config):
    paired_image_names = [
        ['/data1/wjh/SDSD/indoor_unzip/indoor/input', '/data1/wjh/LOL_v2/Real_captured/eval/ref/normal00745.png'],
    ]
    # load model
    net = model.RIENet(config).to(torch.device('cuda:2'))
    net.load_state_dict(torch.load(config.model_path))

    # eval
    net.eval()
    for paired_image_name in paired_image_names:
        A_images_path, B_image_path = paired_image_name[0], paired_image_name[1]
        
        B_image = load_image(B_image_path).unsqueeze(dim=0).to(torch.device('cuda:2'))
        B_image = B_image.to(torch.device('cuda:2'))
        for A_folder in os.listdir(A_images_path):
            if not os.path.exists(os.path.join(config.output_path, A_folder)):
                os.makedirs(os.path.join(config.output_path, A_folder))
            for A_image_name in os.listdir(os.path.join(A_images_path, A_folder)):
                A_image_path = os.path.join(A_images_path, A_folder, A_image_name)
                A_image = load_image(A_image_path).unsqueeze(dim=0).to(torch.device('cuda:2'))
                b,c,h_image,w_image = A_image.shape
                w_pad = 0 if w_image%8 == 0 else 8-w_image%8
                h_pad = 0 if h_image%8 == 0 else 8-h_image%8
                left = w_pad//2
                right = w_pad - left
                top = h_pad//2
                bottom = h_pad - top
                A_image = F.pad(A_image, (left, right, top, bottom))
                A_image = A_image.to(torch.device('cuda:2'))
                
                x_enhance, h, z_ref= net(A_image, B_image)
                x_enhance = x_enhance[:,:,top:h_image+top, left:w_image+left]
                saveimage_numpy = x_enhance[0, ...].permute(1,2,0).cpu().detach().numpy()*255
                saveimage_numpy = np.uint8(np.clip(saveimage_numpy, 0,255))
                saveimage_pil = Image.fromarray(saveimage_numpy)
                saveimage_pil.save(os.path.join(config.output_path, A_folder, A_image_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--output_path', type=str, default="./SDSD")
    parser.add_argument('--w_h', type=float, default=0.01)
    parser.add_argument('--w_z', type=float, default=0.1)
    parser.add_argument('--w_color', type=float, default=0.0)
    parser.add_argument('--lambda_', type=float, default=0.8)
    parser.add_argument('--warm_up', type=float, default=100.0)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--log_iter', type=int, default=10)
    parser.add_argument('--model_path', type=str, default="./weights/LOL.pth")
    parser.add_argument('--device', type=str, default= "2")
    config = parser.parse_args()
    print(config)
    eval(config)
