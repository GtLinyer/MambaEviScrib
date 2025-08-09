import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from networks.our_model import MyModel
from utils.metrics import calculate_metric_percase


# 2 class segmentation
def compute_metrics(pred, lab):
    if np.sum(pred == 1) == 0 or np.sum(lab == 1) == 0:
        first_metric = 0, 0, 0, 0
    else:
        first_metric = calculate_metric_percase(pred == 1, lab == 1)
    return first_metric


def run_test(data_type="BUSI", fold=1):
    data_path = f"<your dataset path>/{data_type}/fold{fold}"
    state_path = f"<your model path>/model/{data_type}/MambaEviScrib_fold{fold}/scribble/MambaEviScrib_best_model.pth"

    model = MyModel(in_chns=1, class_num=2)
    state = torch.load(state_path)
    # state = torch.load(state_path, map_location=torch.device('cpu'))  # cpu version
    model.load_state_dict(state)
    model.eval().cuda()

    first_total = 0.0
    plt.figure()

    data_name_list = os.listdir(f"{data_path}/images")
    for data_name in data_name_list:
        # print(data_name)
        image = cv2.imread(f"{data_path}/images/{data_name}", cv2.IMREAD_GRAYSCALE)
        # image = cv2.resize(image, (256, 256))
        label = cv2.imread(f"{data_path}/labels/{data_name}", cv2.IMREAD_GRAYSCALE)
        # label = cv2.resize(label, (256, 256), interpolation=cv2.INTER_NEAREST)
        image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).cuda()
        image = image / 255
        label = (label > 0).astype(np.uint8)
        with torch.no_grad():
            out = model(image)[0]
        out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        first_metric = compute_metrics(out, label)
        first_total += np.asarray(first_metric)
    avg_metric = [first_total / len(data_name_list)]
    print(fold, avg_metric)  # dc, jc, hd, asd
    return avg_metric


if __name__ == '__main__':
    data_type = "BUSI"
    rst = {}
    for i in [1, 2, 3, 4, 5]:
        print(i)
        m = run_test(data_type=data_type, fold=i)
        rst[i] = m
    print(rst)
