import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from networks.our_model import MyModel
from utils.metrics import calculate_metric_percase


# 5 class segmentation (CardiacUDA, 0: background, 1-4: foreground)
def compute_metrics(pred, lab):
    if np.sum(pred == 1) == 0 or np.sum(lab == 1) == 0:
        first_metric = 0, 0, 0, 0
    else:
        first_metric = calculate_metric_percase(pred == 1, lab == 1)

    if np.sum(pred == 2) == 0 or np.sum(lab == 2) == 0:
        second_metric = 0, 0, 0, 0
    else:
        second_metric = calculate_metric_percase(pred == 2, lab == 2)

    if np.sum(pred == 3) == 0 or np.sum(lab == 3) == 0:
        third_metric = 0, 0, 0, 0
    else:
        third_metric = calculate_metric_percase(pred == 3, lab == 3)

    if np.sum(pred == 4) == 0 or np.sum(lab == 4) == 0:
        forth_metric = 0, 0, 0, 0
    else:
        forth_metric = calculate_metric_percase(pred == 4, lab == 4)
    return first_metric, second_metric, third_metric, forth_metric


def run_test(fold=1):
    data_path = f"<your dataset path>/CardiacUDA/fold{fold}"
    state_path = f"<your model path>/model/CardiacUDA/MambaEviScrib_fold{fold}/scribble/MambaEviScrib_best_model.pth"

    model = MyModel(in_chns=1, class_num=5)
    state = torch.load(state_path)
    model.load_state_dict(state)
    model.eval().cuda()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    forth_total = 0.0
    plt.figure()

    data_name_list = os.listdir(f"{data_path}/images")
    for data_name in data_name_list:
        # print(data_name)
        image = cv2.imread(f"{data_path}/images/{data_name}", cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(f"{data_path}/labels/{data_name}", cv2.IMREAD_GRAYSCALE)
        if image.shape[0] != 256:
            image = cv2.resize(image, (256, 256))
            label = cv2.resize(label, (256, 256), interpolation=cv2.INTER_NEAREST)
        image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).cuda()
        image = image / 255
        with torch.no_grad():
            out = model(image)[0]
        out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        first_metric, second_metric, third_metric, forth_metric = compute_metrics(out, label)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
        forth_total += np.asarray(forth_metric)
    avg_first_metric = [first_total / len(data_name_list)]
    avg_second_metric = [second_total / len(data_name_list)]
    avg_third_metric = [third_total / len(data_name_list)]
    avg_forth_metric = [forth_total / len(data_name_list)]
    print(fold, "class_1", avg_first_metric)  # dc, jc, hd, asd
    print(fold, "class_2", avg_second_metric)
    print(fold, "class_3", avg_third_metric)
    print(fold, "class_4", avg_forth_metric)
    print(fold, "all_avg", [(a + b + c + d) / 4 for a, b, c, d in
                            zip(avg_first_metric, avg_second_metric, avg_third_metric, avg_forth_metric)])


if __name__ == '__main__':
    for i in range(1, 6):
        print(i)
        m = run_test(fold=i)
