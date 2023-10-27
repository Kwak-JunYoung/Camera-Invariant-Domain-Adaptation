from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
from utils.utils import rle_encode
import warnings
warnings.filterwarnings('ignore')

# Load pretrained model
processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_dinat_large")
model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_dinat_large").to('cuda')

df = pd.read_csv('test.csv')
df.head()

ade20k_to_12 = {
    0: [6],
    1: [11, 53],
    2: [0, 1, 32],
    3: [32],
    4: [87],
    5: [136],
    6: [43],
    7: [4, 17],
    8: [2],
    9: [12],
    10: [],
    11: [20, 80, 116]
}

submit = pd.read_csv('./sample_submission.csv')

result = []
for i in tqdm(range(len(df))):
    image = Image.open(df['img_path'][i])
    # image resize
    image = image.resize((960, 540))
    
    semantic_inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt")

    for key in semantic_inputs.keys():
        semantic_inputs[key] = semantic_inputs[key].to('cuda')

    semantic_outputs = model(**semantic_inputs)

    # pass through image_processor for postprocessing
    predicted_semantic_map = processor.post_process_semantic_segmentation(semantic_outputs, target_sizes=[image.size[::-1]])[0]

    del semantic_inputs, semantic_outputs
    predicted_semantic_map_np = np.array(predicted_semantic_map.cpu().numpy())

    # convert to rle
    for key, value in ade20k_to_12.items():
        key_mask = np.isin(predicted_semantic_map_np, value)
        if np.sum(key_mask) > 0:
            mask_rle = rle_encode(key_mask)
            result.append(mask_rle)
        else:
            result.append(-1)

submit = pd.read_csv('./sample_submission.csv')
submit['mask_rle'] = result

submit.to_csv('segformer_pretrain_submit.csv', index=False)