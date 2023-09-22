from models.unet import UNet
import torch

from tqdm import tqdm

from data_loaders import dataloader

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


def model_train(
    dir_name,
    fold,
    model,
    accelerator,
    opt,
    train_loader,
    valid_loader,
    test_loader,
    config,
    n_gpu,
    early_stop=True,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model 초기화
    model = UNet().to(device)  # To main

    # loss function과 optimizer 정의
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # To main

    # training loop
    for epoch in range(20):  # 20 에폭 동안 학습합니다.
        model.train()
        epoch_loss = 0
        for images, masks in tqdm(dataloader):
            images = images.float().to(device)
            masks = masks.long().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = model.loss(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader)}')
