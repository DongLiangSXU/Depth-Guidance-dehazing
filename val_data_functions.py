import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import numpy as np
import os
import torchvision.transforms as tfs
# --- Validation/test dataset --- #
def read_val(val_data_dir):
    filenames=[]
    name_list=os.listdir(val_data_dir)
    for name in name_list:
        filenames.append(name)
    return filenames

class ValData(data.Dataset):
    def __init__(self, val_data_dir):
        super().__init__()

        self.input_names = read_val(val_data_dir+'lowq')
        self.val_data_dir = val_data_dir

    def get_images(self, index):
        input_name = self.input_names[index]
        gt_name = self.input_names[index]
        input_img = Image.open(self.val_data_dir + 'lowq/'+input_name).convert('RGB')
        gt_img = Image.open(self.val_data_dir + 'clear/'+gt_name).convert('RGB')

        # Resizing image in the multiple of 32"

        wd_new,ht_new = input_img.size
  
        if ht_new>wd_new and ht_new>1024:
            wd_new = int(np.ceil(wd_new*1024/ht_new))
            ht_new =1024
        elif ht_new<=wd_new and wd_new>1024:
            ht_new = int(np.ceil(ht_new*1024/wd_new))
            wd_new = 1024
        wd_new = int(32*(wd_new//32))
        ht_new = int(32*(ht_new//32))
        input_img = input_img.resize((wd_new,ht_new), Image.ANTIALIAS)
        gt_img = gt_img.resize((wd_new, ht_new), Image.ANTIALIAS)

        # --- Transform to tensor --- #
        
        input_im = tfs.ToTensor()(input_img)
        gt = tfs.ToTensor()(gt_img)
       
        

        return input_im,gt,input_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)


