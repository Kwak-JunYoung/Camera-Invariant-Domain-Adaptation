from models.unet import UNet
import torch

from tqdm import tqdm

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

import numpy as np
from PIL import Image
from utils import rle_encode


def model_train(
    model,
    device,
    opt,
    train_dataloader,
    test_dataloader,
    config,
):
    num_epochs = config["train_config"]["num_epochs"]
    model_name = config["model_name"]
    data_name = config["data_name"]
    train_config = config["train_config"]
    # loss function과 optimizer 정의

    # training loop
    for epoch in range(num_epochs):  # 20 에폭 동안 학습합니다.
        model.train()
        epoch_loss = 0
        for images, masks in tqdm(train_dataloader):
            images = images.float().to(device)
            masks = masks.long().to(device)

            opt.zero_grad()
            outputs = model(images)
            loss = model.loss(outputs, masks)
            loss.backward()
            opt.step()

            epoch_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(train_dataloader)}')

    with torch.no_grad():
        model.eval()
        result = []
        for images in tqdm(test_dataloader):
            images = images.float().to(device)
            outputs = model(images)
            outputs = torch.softmax(outputs, dim=1).cpu()
            outputs = torch.argmax(outputs, dim=1).numpy()
            # batch에 존재하는 각 이미지에 대해서 반복
            for pred in outputs:
                pred = pred.astype(np.uint8)
                pred = Image.fromarray(pred)  # 이미지로 변환
                # 960 x 540 사이즈로 변환
                pred = pred.resize((960, 540), Image.NEAREST)
                pred = np.array(pred)  # 다시 수치로 변환
                # class 0 ~ 11에 해당하는 경우에 마스크 형성 / 12(배경)는 제외하고 진행
                for class_id in range(12):
                    class_mask = (pred == class_id).astype(np.uint8)
                    if np.sum(class_mask) > 0:  # 마스크가 존재하는 경우 encode
                        mask_rle = rle_encode(class_mask)
                        result.append(mask_rle)
                    else:  # 마스크가 존재하지 않는 경우 -1
                        result.append(-1)

    return result
