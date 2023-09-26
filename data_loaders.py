import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import cv2
import pandas as pd
from transform import MyToTensor, MyNormalization, RandomFlip


class CustomDataset(Dataset):
    def __init__(self, csv_file, infer=False):
        super().__init__()

        self.data = pd.read_csv(csv_file)

        self.to_tensor = MyToTensor()
        self.normalize = MyNormalization()
        self.flip = RandomFlip()
        self.transform = A.Compose(
            [
                A.OneOf([
                    A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                    A.GridDistortion(p=0.5),
                    A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1),
                ], p=0.5),
                A.Resize(224, 224),
                A.Normalize(),
                ToTensorV2()                
            ]
        )
        
        self.infer = infer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]
        img_path = img_path.replace('./', './dataset/')
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.infer and self.transform:
            image = self.transform(image=image)['image']
            return image

        mask_path = self.data.iloc[idx, 2]
        mask_path = mask_path.replace('./', './dataset/')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask == 255] = 12  # 배경을 픽셀값 12로 간주

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask
