from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

class BratsNpyDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_files  = sorted(os.listdir(img_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.img_dir    = img_dir
        self.mask_dir   = mask_dir
        self.transform  = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = np.load(os.path.join(self.img_dir,  self.img_files[idx]))  # (M,H,W), float32
        msk = np.load(os.path.join(self.mask_dir, self.mask_files[idx])) # (H,W), uint8

        if self.transform:
            # e.g. torch transforms, albumentations, etc.
            data = self.transform(image=img, mask=msk)
            img, msk = data['image'], data['mask']

        # convert to torch.Tensor
        img = torch.from_numpy(img)           # (M,H,W), float32
        msk = torch.from_numpy(msk).long()    # (H,W), int64

        return img, msk

# usage
dataset = BratsNpyDataset('segformer_data_npy/imagesTr',
                          'segformer_data_npy/labelsTr',
                          transform=None)  # or your augment pipeline
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
