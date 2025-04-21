
import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageOps

class LOL_dataset(data.Dataset):
    def __init__(self, LI_path, HI_path, hist_eq=False):
        self.LI_list = [os.path.join(LI_path, i) for i in os.listdir(LI_path)]
        self.HI_list = [os.path.join(HI_path, os.path.basename(i).replace('low', 'normal')) for i in self.LI_list]
        self.size = 256
        self.hist_eq = hist_eq

    def __getitem__(self, index):
        return self.load_image(self.LI_list[index]), self.load_image(self.HI_list[index], False)
    
    def load_image(self, path, hist_eq=True):
        image = Image.open(path)
        if self.hist_eq and hist_eq:
            image = ImageOps.equalize(image)
        image = image.convert('RGB') # (h,w,c)
        image = image.resize((self.size,self.size), Image.ANTIALIAS)
        image = (np.asarray(image)/255.0) 
        image = torch.from_numpy(image).float()
        return image.permute(2,0,1) # (c,h,w) 
    
    def __len__(self):
        return len(self.LI_list)

class EE_dataset(data.Dataset):
    def __init__(self, LI_path, HI_path, hist_eq=False):
        self.LI_list = [os.path.join(LI_path, i) for i in os.listdir(LI_path)]
        self.HI_list =[]
        for li_path in self.LI_list:
            filename = os.path.basename(li_path)
            suffix = filename.split('_')[-1]
            new_filename = filename[:-len(suffix)-1]+'.jpg'
            hi_path = os.path.join(HI_path, new_filename)
            self.HI_list.append(hi_path)
        self.size = 256
        self.hist_eq = hist_eq

    def __getitem__(self, index):
        return self.load_image(self.LI_list[index]), self.load_image(self.HI_list[index], False)
    
    def load_image(self, path, hist_eq=True):
        image = Image.open(path)
        if self.hist_eq and hist_eq:
            image = ImageOps.equalize(image)
        image = image.convert('RGB') # (h,w,c)
        image = image.resize((self.size,self.size), Image.ANTIALIAS)
        image = (np.asarray(image)/255.0) 
        image = torch.from_numpy(image).float()
        return image.permute(2,0,1) # (c,h,w) 
    
    def __len__(self):
        return len(self.LI_list)

class EE_dataset_multiexpo(data.Dataset):
    def __init__(self, LI_path, HI_path):
        self.LI_list = [os.path.join(LI_path, i) for i in os.listdir(LI_path)]
        self.HI_list =[]
        for li_path in self.LI_list:
            filename = os.path.basename(li_path)
            suffix = filename.split('_')[-1]
            new_filename = filename[:-len(suffix)-1]+'.jpg'
            hi_path = os.path.join(HI_path, new_filename)
            self.HI_list.append(hi_path)
        self.size = 256

    def __getitem__(self, index):
        return self.load_image(self.LI_list[index]), self.load_image(self.HI_list[index])
    
    def load_image(self, path):
        image = Image.open(path).convert('RGB') # (h,w,c)
        image = image.resize((self.size,self.size), Image.ANTIALIAS)
        image = (np.asarray(image)/255.0) 
        image = torch.from_numpy(image).float()
        return image.permute(2,0,1) # (c,h,w) 
    
    def __len__(self):
        return len(self.LI_list)

class EE_dataset_eval(data.Dataset):
    def __init__(self, LI_path, HI_path, hist_eq=False):
        self.LI_list = [os.path.join(LI_path, i) for i in os.listdir(LI_path)]
        self.HI_list =[]
        for li_path in self.LI_list:
            filename = os.path.basename(li_path)
            suffix = filename.split('_')[-1]
            new_filename = filename[:-len(suffix)-1]+'.jpg'
            hi_path = os.path.join(HI_path, new_filename)
            self.HI_list.append(hi_path)
        self.size = 256
        self.hist_eq = hist_eq

    def __getitem__(self, index):
        return self.load_image(self.LI_list[index]), self.load_image(self.HI_list[index], hist_eq=False), os.path.basename(self.LI_list[index])
    
    def load_image(self, path, noresize=False, hist_eq=True):
        image = Image.open(path)
        if self.hist_eq and hist_eq:
            image = ImageOps.equalize(image)
        image = image.convert('RGB') # (h,w,c)
        w,h = image.size
        scale = np.ceil(w/600)
        if not noresize:
            image = image.resize((int(w/scale), int(h/scale)), Image.ANTIALIAS)
        # image = image.resize((self.size,self.size), Image.ANTIALIAS)
        image = (np.asarray(image)/255.0) 
        image = torch.from_numpy(image).float()
        return image.permute(2,0,1) # (c,h,w) 
    
    def __len__(self):
        return len(self.LI_list)

class LOL_dataset_eval(data.Dataset):
    def __init__(self, LI_path, HI_path, hist_eq=False):
        self.LI_list = [os.path.join(LI_path, i) for i in os.listdir(LI_path)]
        self.HI_list = [os.path.join(HI_path, os.path.basename(i).replace('low', 'normal')) for i in self.LI_list]
        self.size = 256
        self.hist_eq = hist_eq

    def __getitem__(self, index):
        return self.load_image(self.LI_list[index]), self.load_image(self.HI_list[index], hist_eq=False), os.path.basename(self.LI_list[index])
    
    def load_image(self, path, hist_eq=True):
        image = Image.open(path)
        if self.hist_eq and hist_eq:
            image = ImageOps.equalize(image)
        image = image.convert('RGB') # (h,w,c)
        # image = image.resize((self.size,self.size), Image.ANTIALIAS)
        image = (np.asarray(image)/255.0) 
        image = torch.from_numpy(image).float()
        return image.permute(2,0,1) # (c,h,w) 
    
    def __len__(self):
        return len(self.LI_list)

class LSRW_dataset_eval(data.Dataset):
    def __init__(self, LI_path, HI_path):
        self.LI_list = sorted([os.path.join(LI_path, i) for i in os.listdir(LI_path)])
        
        self.HI_list = [os.path.join(HI_path, os.path.basename(i)) for i in self.LI_list]
        self.size = 256

    def __getitem__(self, index):
        return self.load_image(self.LI_list[index]), self.load_image(self.HI_list[index]), os.path.basename(self.LI_list[index])
    
    def load_image(self, path):
        image = Image.open(path)
        image = image.convert('RGB') # (h,w,c)
        # image = image.resize((self.size,self.size), Image.ANTIALIAS)
        image = (np.asarray(image)/255.0) 
        image = torch.from_numpy(image).float()
        return image.permute(2,0,1) # (c,h,w) 
    
    def __len__(self):
        return len(self.LI_list)

class LOL_dataset_efficient(data.Dataset):
    def __init__(self, LI_path, HI_path):
        self.LI_list = [os.path.join(LI_path, i) for i in os.listdir(LI_path)]
        self.HI_list = [os.path.join(HI_path, os.path.basename(i).replace('low', 'normal')) for i in self.LI_list]
        self.size = 256

    def __getitem__(self, index):
        return self.load_image(self.LI_list[index]), self.load_image(self.HI_list[index]), os.path.basename(self.LI_list[index])
    
    def load_image(self, path):
        image = Image.open(path).convert('RGB') # (h,w,c)
        image = image.resize((self.size,self.size), Image.ANTIALIAS)
        image = (np.asarray(image)/255.0) 
        image = torch.from_numpy(image).float()
        return image.permute(2,0,1) # (c,h,w) 
    
    def __len__(self):
        return len(self.LI_list)
