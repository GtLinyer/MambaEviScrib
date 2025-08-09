import argparse
import logging
import os
import random
import shutil
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders.my_dataset import MyDataset
from networks.our_model import MyModel
from utils.gate_crf_loss import ModelLossSemsegGatedCRF
from utils.losses import edl_digamma_loss
from val_2D import test_single_volume


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='<your dataset path>/BSUI', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='BSUI/MambaEviScrib', help='experiment_name')
parser.add_argument('--fold', type=str,
                    default='fold1', help='cross validation')
parser.add_argument('--sup_type', type=str,
                    default='scribble', help='supervision type')
parser.add_argument('--num_classes', type=int, default=2,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=100000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=6,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.03,
                    help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=2024, help='random seed')
parser.add_argument('--cuda', type=str, default="0", help='cuda number')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    model = MyModel(in_chns=1, class_num=num_classes).cuda()
    db_train = MyDataset(file_path=args.root_path, run_type="train", val_fold=args.fold)
    db_val = MyDataset(file_path=args.root_path, run_type="val", val_fold=args.fold)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    model.train()

    p_ce_loss_fn = CrossEntropyLoss(ignore_index=num_classes)
    ce_loss_fn = CrossEntropyLoss()
    mse_loss_fn = MSELoss()

    gatecrf_loss = ModelLossSemsegGatedCRF()
    loss_gatedcrf_kernels_desc = [{"weight": 1, "xy": 6, "rgb": 0.1}]
    loss_gatedcrf_radius = 5

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    cur_threshold = 1 / num_classes

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch[0], sampled_batch[1]
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            label0 = (label_batch == 0).unsqueeze(1)
            label1 = (label_batch == 1).unsqueeze(1)
            mask = 1 - (label_batch == 2).int().unsqueeze(1)
            label_onehot = torch.cat([label0, label1], dim=1).int()
            # ====== start ======
            outputs_cnn, outputs_mamba = model(volume_batch)
            outputs_cnn_soft = torch.softmax(outputs_cnn, dim=1)
            outputs_mamba_soft = torch.softmax(outputs_mamba, dim=1)

            # pCE + pEDL loss
            loss_p_ce_cnn = p_ce_loss_fn(outputs_cnn, label_batch[:].long())
            evidence_cnn = torch.exp(torch.tanh(outputs_cnn) / 0.25)
            loss_p_edl_cnn = edl_digamma_loss(evidence_cnn.float(), label_onehot.float(),
                                              epoch_num, num_classes, (max_epoch // 2)) * mask
            loss_gcrf_cnn = gatecrf_loss(
                outputs_cnn_soft,
                loss_gatedcrf_kernels_desc,
                loss_gatedcrf_radius,
                volume_batch,
                256,
                256,
            )["loss"]
            loss_cnn = loss_p_ce_cnn + loss_p_edl_cnn.sum() / mask.sum() + 0.1 * loss_gcrf_cnn

            loss_p_ce_mamba = p_ce_loss_fn(outputs_mamba, label_batch[:].long())
            evidence_mamba = torch.exp(torch.tanh(outputs_mamba) / 0.25)
            loss_p_edl_mamba = edl_digamma_loss(evidence_mamba.float(), label_onehot.float(),
                                                epoch_num, num_classes, (max_epoch // 2)) * mask
            loss_gcrf_mamba = gatecrf_loss(
                outputs_mamba_soft,
                loss_gatedcrf_kernels_desc,
                loss_gatedcrf_radius,
                volume_batch,
                256,
                256,
            )["loss"]
            loss_mamba = loss_p_ce_mamba + loss_p_edl_mamba.sum() / mask.sum() + 0.1 * loss_gcrf_mamba

            supervised_loss = loss_cnn + loss_mamba

            # Evl loss
            alpha_cnn = evidence_cnn + 1
            s_cnn = torch.sum(alpha_cnn, dim=1, keepdim=True)
            belief_cnn = evidence_cnn / s_cnn

            evidence_cnn_1 = torch.exp(torch.tanh(outputs_cnn) / 0.125)
            alpha_cnn_1 = evidence_cnn_1 + 1
            s_cnn_1 = torch.sum(alpha_cnn_1, dim=1, keepdim=True)
            belief_cnn_1 = evidence_cnn_1 / s_cnn_1

            alpha_mamba = evidence_mamba + 1
            s_mamba = torch.sum(alpha_mamba, dim=1, keepdim=True)
            belief_mamba = evidence_mamba / s_mamba

            evidence_mamba_1 = torch.exp(torch.tanh(outputs_mamba) / 0.125)
            alpha_mamba_1 = evidence_mamba_1 + 1
            s_mamba_1 = torch.sum(alpha_mamba_1, dim=1, keepdim=True)
            belief_mamba_1 = evidence_mamba_1 / s_mamba_1

            output1_soft = belief_cnn
            output2_soft = belief_mamba

            output1_soft0 = belief_cnn_1
            output2_soft0 = belief_mamba_1

            with torch.no_grad():
                max_values1, _ = torch.max(output1_soft, dim=1)
                max_values2, _ = torch.max(output2_soft, dim=1)
                percent = (iter_num + 1) / max_iterations
                cur_threshold1 = (1 - percent) * cur_threshold + percent * max_values1.mean()
                cur_threshold2 = (1 - percent) * cur_threshold + percent * max_values2.mean()
                mean_max_values = min(max_values1.mean(), max_values2.mean())

                cur_threshold = min(cur_threshold1, cur_threshold2)
                cur_threshold = torch.clip(cur_threshold, 0.25, 0.95)

            mask_high = (output1_soft > cur_threshold) & (output2_soft > cur_threshold)
            mask_non_similarity = (mask_high == False)

            new_output1_soft = torch.mul(mask_non_similarity, output1_soft)
            new_output2_soft = torch.mul(mask_non_similarity, output2_soft)
            high_output1 = torch.mul(mask_high, outputs_cnn)
            high_output2 = torch.mul(mask_high, outputs_mamba)
            high_output1_soft = torch.mul(mask_high, output1_soft)
            high_output2_soft = torch.mul(mask_high, output2_soft)

            pseudo_output1 = torch.argmax(output1_soft, dim=1)
            pseudo_output2 = torch.argmax(output2_soft, dim=1)
            pseudo_high_output1 = torch.argmax(high_output1_soft, dim=1)
            pseudo_high_output2 = torch.argmax(high_output2_soft, dim=1)

            max_output1_indices = new_output1_soft > new_output2_soft

            max_output1_value0 = torch.mul(max_output1_indices, output1_soft0)
            min_output2_value0 = torch.mul(max_output1_indices, output2_soft0)

            max_output2_indices = new_output2_soft > new_output1_soft

            max_output2_value0 = torch.mul(max_output2_indices, output2_soft0)
            min_output1_value0 = torch.mul(max_output2_indices, output1_soft0)

            loss_dc0 = 0
            loss_cer = 0

            loss_dc0 += mse_loss_fn(max_output1_value0.detach(), min_output2_value0)
            loss_dc0 += mse_loss_fn(max_output2_value0.detach(), min_output1_value0)

            if mean_max_values >= 0.95:
                loss_cer += ce_loss_fn(outputs_cnn, pseudo_output2.long().detach())
                loss_cer += ce_loss_fn(outputs_mamba, pseudo_output1.long().detach())
            else:
                loss_cer += ce_loss_fn(high_output1, pseudo_high_output2.long().detach())
                loss_cer += ce_loss_fn(high_output2, pseudo_high_output1.long().detach())

            # all loss
            loss = supervised_loss + loss_dc0 + loss_cer

            # ====== end ======
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)

            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))

            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)

                outputs = torch.argmax(torch.softmax(outputs_cnn, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 125, iter_num)

                labs = label_batch[1, ...].unsqueeze(0) * 125
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0

                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(sampled_batch[0], sampled_batch[1], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)

                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(
                        iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path, 'MambaEviScrib_best_model.pth')

                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info('iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.fold, args.sup_type)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
