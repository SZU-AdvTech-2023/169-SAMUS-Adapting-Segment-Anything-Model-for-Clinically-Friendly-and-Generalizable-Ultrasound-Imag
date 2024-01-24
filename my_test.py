import argparse
import os

from hausdorff import hausdorff_distance

from utils.config import get_config
import torch
import time
from torch.utils.tensorboard import SummaryWriter
from utils.data_us import JointTransform2D, ImageToImage2D
from utils.evaluation import get_eval
from models.model_dict import get_model
import numpy as nppython
import random
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
from utils.loss_functions.sam_loss import get_criterion
from utils.generate_prompts import get_click_prompt
from thop import profile
import numpy as np
import FNPC_module as fnpc
import cv2
from PIL import Image
from torch.autograd import Variable
import utils.metrics as metrics


def main():
    #  ============================================================================= parameters setting ====================================================================================

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='SAMUS', type=str,
                        help='type of model, e.g., SAM, SAMFull, MedSAM, MSA, SAMed, SAMUS...')
    parser.add_argument('-encoder_input_size', type=int, default=256,
                        help='the image size of the encoder input, 1024 in SAM and MSA, 512 in SAMed, 256 in SAMUS')
    parser.add_argument('-low_image_size', type=int, default=128,
                        help='the image embedding size, 256 in SAM and MSA, 128 in SAMed and SAMUS')
    parser.add_argument('--task', default='UDIAT', help='task or dataset name')
    parser.add_argument('--vit_name', type=str, default='vit_b',
                        help='select the vit model for the image encoder of sam')
    parser.add_argument('--sam_ckpt', type=str,
                        default="/data/zwt/project/SAMUS/ckpt/BUSI/msaghfc/non_normal/SAMUS_12281209_191_0.898088711819689.pth",
                        help='Pretrained checkpoint of SAM')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch_size per gpu')  # SAMed is 12 bs with 2n_gpu and lr is 0.005
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--base_lr', type=float, default=0.0005,
                        help='segmentation network learning rate, 0.005 for SAMed, 0.0001 for MSA')  # 0.0006
    parser.add_argument('--warmup', type=bool, default=False,
                        help='If activated, warp up the learning from a lower lr to the base_lr')
    parser.add_argument('--warmup_period', type=int, default=250,
                        help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('-keep_log', type=bool, default=False, help='keep the loss&lr&dice during training or not')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = get_config(args.task)
    opt.mode = "val"
    opt.visual = False
    opt.modelname = args.modelname
    opt.load_path = opt.save_path + "SAMUS_12281209_191_0.898088711819689.pth"
    print("task", args.task, "checkpoints:", opt.load_path)
    device = torch.device(opt.device)

    #  =============================================================== add the seed to make sure the results are reproducible ==============================================================

    seed_value = 300  # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

    #  =========================================================================== model and data preparation ============================================================================

    opt.batch_size = args.batch_size * args.n_gpu

    tf_val = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size,
                              crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)
    val_dataset = ImageToImage2D(opt.data_path, opt.test_split, tf_val, img_size=args.encoder_input_size,
                                 class_id=1)  # return image, mask, and filename
    valloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = get_model(args.modelname, args=args, opt=opt)
    model.to(device)

    checkpoint = torch.load(opt.load_path)
    # ------when the load model is saved under multiple GPU 加载模型
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k[:7] == 'module.':
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)

    optimizer = optim.Adam(model.parameters(), lr=args.base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                           amsgrad=False)

    criterion = get_criterion(modelname=args.modelname, opt=opt)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))

    # ========================================================================= begin to evaluate the model ============================================================================

    # input = torch.randn(1, 1, args.encoder_input_size, args.encoder_input_size).cuda()  # 随机张量
    # points = (torch.tensor([[[1, 2]]]).float().cuda(), torch.tensor([[1]]).float().cuda())
    # flops, params = profile(model, inputs=(input, points), )
    # print('Gflops:', flops / 1000000000, 'params:', params)

    model.eval()

    if opt.mode == "train":
        dices, mean_dice, _, val_losses = get_eval(valloader, model, criterion=criterion, opt=opt, args=args)
        print("mean dice:", mean_dice)
    else:
        mean_dice, mean_hdis, mean_iou, mean_acc, mean_se, mean_sp, std_dice, std_hdis, std_iou, std_acc, std_se, std_sp = get_eval(
            valloader, model, criterion=criterion, opt=opt, args=args)
        print("dataset:" + args.task + " -----------model name: " + args.modelname)
        print("task", args.task, "checkpoints:", opt.load_path)
        print(mean_dice[1:], mean_hdis[1:], mean_iou[1:], mean_acc[1:], mean_se[1:], mean_sp[1:])
        print(std_dice[1:], std_hdis[1:], std_iou[1:], std_acc[1:], std_se[1:], std_sp[1:])
        with open("experiments.txt", "a+") as file:
            file.write(args.task + " " + args.modelname + "-pt10 " + '%.2f' % (mean_dice[1]) + "±" + '%.2f' % std_dice[
                1] + " ")
            file.write('%.2f' % mean_hdis[1] + "±" + '%.2f' % std_hdis[1] + " ")
            file.write('%.2f' % (mean_iou[1]) + "±" + '%.2f' % std_iou[1] + " ")
            file.write('%.2f' % (mean_acc[1]) + "±" + '%.2f' % std_acc[1] + " ")
            file.write('%.2f' % (mean_se[1]) + "±" + '%.2f' % std_se[1] + " ")
            file.write('%.2f' % (mean_sp[1]) + "±" + '%.2f' % std_sp[1] + "\n")

    if opt.mode == "val":
        val_dataset = ImageToImage2D(opt.data_path, opt.test_split, tf_val, img_size=args.encoder_input_size,
                                     class_id=1)  # return image, mask, and filename
        valloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
        dices_ave = []
        dices = []
        hd_ave = []
        hd = []
        for batch_idx, (datapack) in enumerate(valloader):
            imgs = Variable(datapack['image'].to(dtype=torch.float32, device=opt.device))
            label = Variable(datapack['label'].to(dtype=torch.float32, device=opt.device))
            hfcs = datapack['hfc'].to(dtype=torch.float32, device=opt.device)
            image_filename = datapack['image_name']
            pt = np.array(datapack['pt'])

            # initial_mask_path = opt.result_path + "mask/" + image_filename[0]
            # ref_mask = cv2.imread(initial_mask_path)
            # ref_mask = ref_mask.transpose(2,0,1)
            ref_mask = label[0, :, :, :].detach().cpu().numpy()
            pred_mask = cv2.imread(os.path.join(opt.result_path, "mask/", image_filename[0]), -1)
            radius, new_point_list, new_point_label_list = fnpc.sample_points(pred_mask, pt[0], M=5, N=10)
            input_box = fnpc.generate_box(ref_mask)
            ave_mask = fnpc.predict_ave_mask(new_point_list, new_point_label_list, model, imgs, hfcs)
            # input_box = fnpc.generate_box(ave_mask)
            FN_final_mask, FP_final_mask = fnpc.predict_FNPC_mask(new_point_list, new_point_label_list, input_box, model, imgs, hfcs, pred_mask, ref_mask, image_filename, opt.result_path)

            gt = label.detach().cpu().numpy()
            gt = gt[:, 0, :, :]
            b, h, w = gt.shape
            pred_i_ave = np.zeros((1, h, w))
            pred_i_ave[ave_mask[:, :, :] == 1] = 255
            pred_i = np.zeros((1, h, w))
            pred_i[FP_final_mask[:, :, :] == 1] = 255
            gt_i = np.zeros((1, h, w))
            gt_i[gt[:, :, :] == 1] = 255
            dice_i_ave = metrics.dice_coefficient(pred_i_ave, gt_i)
            dice_i = metrics.dice_coefficient(pred_i, gt_i)
            hd_i_ave = hausdorff_distance(pred_i_ave[0, :, :], gt_i[0, :, :], distance="euclidean")
            hd_i = hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="euclidean")
            print(image_filename, "ave", dice_i_ave, hd_i_ave)
            print(image_filename, "FNPC", dice_i, hd_i)
            list.append(dices_ave, dice_i_ave)
            list.append(dices, dice_i)
            list.append(hd_ave, hd_i_ave)
            list.append(hd, hd_i)

            ave_mask = ave_mask.transpose(1, 2, 0)
            cv2.imwrite(os.path.join(opt.result_path, "ave_mask/", image_filename[0]), ave_mask * 255)
            # FN_final_mask = FN_final_mask.transpose(1, 2, 0)
            # cv2.imwrite(os.path.join(opt.result_path, "FN_mask/", image_filename[0]), FN_final_mask * 255)
            FP_final_mask = FP_final_mask.transpose(1, 2, 0)
            cv2.imwrite(os.path.join(opt.result_path, "FNPC_mask/", image_filename[0]), FP_final_mask * 255)

            # image = cv2.imread(os.path.join(opt.result_path, "FNPC_mask/",image_filename[0]), cv2.IMREAD_GRAYSCALE)
            # kernel = np.ones((5, 5), np.uint8)
            # erosion = cv2.erode(image, kernel, iterations=1)
            # dilation = cv2.dilate(erosion, kernel, iterations=1)
            # dilation = np.expand_dims(dilation, axis=0)
            # dice_i_d = metrics.dice_coefficient(dilation, gt_i)
            # print("dice_i_d", dice_i_d)
            # dices_d = dices_d + dice_i_d
            # dilation = dilation.transpose(1, 2, 0)
            # cv2.imwrite(os.path.join(opt.result_path, "FNPC_mask_dilation/", str(dice_i_d) + image_filename[0]), dilation * 255)

        # dice_mean_ave = dices_ave / len(val_dataset)
        # print("ave_mask", dice_mean_ave)
        # dice_mean = dices / len(val_dataset)
        # print("FPNC_mask", dice_mean)
        # dice_mean_d = dices_d / len(val_dataset)
        # print("FPNC_mask_dilation", dice_mean_d)
        print("ave_mask", np.mean(dices_ave), np.mean(hd_ave))
        print("FPNC_mask", np.mean(dices), np.mean(hd))


if __name__ == '__main__':
    main()
