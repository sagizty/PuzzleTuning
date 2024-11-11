import os
import cv2
import torch
import random
import numpy as np
from scipy import spatial
from model.JigsawNet import JigsawNet
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = JigsawNet(1, 1000)
model.to(device)
model.load_state_dict(torch.load('', map_location=device))
model.eval()
img_dir = ''

imgs_name = os.listdir(img_dir)
random.shuffle(imgs_name)
imgs_name = imgs_name
features = []
with torch.no_grad():
    with tqdm(total=len(imgs_name)) as bar:
        for name in imgs_name:
            img = cv2.imread(os.path.join(img_dir, name), 0)
            img = cv2.resize(img, (225, 225), cv2.INTER_LINEAR) / 255.0
            img = img[np.newaxis, :]
            imgclips = []
            for i in range(3):
                for j in range(3):
                    clip = img[:, i * 75: (i + 1) * 75, j * 75: (j + 1) * 75]
                    randomx = random.randint(0, 10)
                    randomy = random.randint(0, 10)
                    clip = clip[:, randomx: randomx + 64, randomy:randomy + 64]
                    imgclips.append(clip)
            imgclips = np.array(imgclips)
            imgclips = torch.from_numpy(imgclips).unsqueeze(0).to(device, dtype=torch.float32)
            feature = model.encode(imgclips).squeeze()
            features.append(feature.cpu().numpy())
            bar.update(1)

np.save('features', np.array(features))

# distance = [[0 for i in range(len(features))] for j in range(len(features))]
#
# for i in range(len(features)):
#     for j in range(i+1, len(features)):
#         cosdis = spatial.distance.cosine(features[i].cpu().numpy(), features[j].cpu().numpy())
#         distance[i][j] = cosdis
#         distance[j][i] = cosdis
#
# distance = np.array(distance).flatten()
#
# plt.hist(distance, bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
# plt.show()

