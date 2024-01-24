import argparse
import os
from utils.config import get_config
import torch
import time
from torch.utils.tensorboard import SummaryWriter
from utils.data_us import JointTransform2D, ImageToImage2D
from utils.evaluation import get_eval
from models.model_dict import get_model
import numpy as np
import random
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
from utils.loss_functions.sam_loss import get_criterion
from utils.generate_prompts import get_click_prompt


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
    parser.add_argument('--sam_ckpt', type=str, default="./sam_vit_b_01ec64.pth",
                        help='Pretrained checkpoint of SAM')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch_size per gpu')  # SAMed is 12 bs with 2n_gpu and lr is 0.005
    parser.add_argument('--n_gpu', type=int, default=2, help='total gpu')
    parser.add_argument('--base_lr', type=float, default=0.0005,
                        help='segmentation network learning rate, 0.005 for SAMed, 0.0001 for MSA')  # 0.0006
    parser.add_argument('--warmup', type=bool, default=False,
                        help='If activated, warp up the learning from a lower lr to the base_lr')
    parser.add_argument('--warmup_period', type=int, default=250,
                        help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('-keep_log', type=bool, default=True, help='keep the loss&lr&dice during training or not')

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    args = parser.parse_args()
    opt = get_config(args.task)

    device = torch.device(opt.device)

    if args.keep_log:
        logtimestr = time.strftime('%m%d%H%M')  # initialize the tensorboard for record the training process
        boardpath = opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr
        if not os.path.isdir(boardpath):
            os.makedirs(boardpath)
        TensorWriter = SummaryWriter(boardpath)

    #  =============================================================== add the seed to make sure the results are reproducible ==============================================================

    seed_value = 1234  # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

    #  =========================================================================== model and data preparation ============================================================================
    # set model
    model = get_model(args.modelname, args=args, opt=opt)
    opt.batch_size = args.batch_size * args.n_gpu

    tf_train = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size,
                                ori_size=opt.img_size, crop=opt.crop, p_flip=0.0, p_rota=0.5, p_scale=0.5, p_gaussn=0.0,
                                p_contr=0.5, p_gama=0.5, p_distor=0.0, color_jitter_params=None,
                                long_mask=True)  # image reprocessing
    tf_val = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size,
                              crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)
    train_dataset = ImageToImage2D(opt.data_path, opt.train_split, tf_train, img_size=args.encoder_input_size)
    val_dataset = ImageToImage2D(opt.data_path, opt.val_split, tf_val,
                                 img_size=args.encoder_input_size)  # return image, mask, and filename
    trainloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model.to(device)

    # 加载预训练模型对模型的状态字典进行修改（PyTorch默认会将模型的所有层都保存为'module.'开头的名字，而在加载模型时通常只需要加载主模型，不需要加载其他子模块。）
    # 遍历状态字典将所有以'module.'开头的键名改为去掉'module.'后的名字，然后用于加载模型。
    if opt.pre_trained:
        checkpoint = torch.load(opt.load_path)
        new_state_dict = {}
        for k, v in checkpoint.items():
            if k[:7] == 'module.':
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    # 设置学习率和其他优化器参数
    # 如果设置了预热（warmup）阶段，那么学习率会被设置为args.base_lr / args.warmup_period，并且使用带有预热功能的AdamW优化器。
    # 如果没有设置预热阶段，那么学习率会被设置为args.base_lr，并且使用普通的Adam优化器。
    if args.warmup:
        b_lr = args.base_lr / args.warmup_period
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr,
                                      betas=(0.9, 0.999), weight_decay=0.1)
    else:
        b_lr = args.base_lr
        optimizer = optim.Adam(model.parameters(), lr=args.base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                               amsgrad=False)

    # 获取损失函数 0.2mce+0.8dice
    criterion = get_criterion(modelname=args.modelname, opt=opt)

    # 计算训练参数总量
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))

    #  ========================================================================= begin to train the model ============================================================================
    iter_num = 0
    max_iterations = opt.epochs * len(trainloader)
    best_dice, loss_log, dice_log = 0.0, np.zeros(opt.epochs+1), np.zeros(opt.epochs+1)
    for epoch in range(opt.epochs):
        model.train()
        train_losses = 0
        for batch_idx, (datapack) in enumerate(trainloader):
            imgs = datapack['image'].to(dtype=torch.float32, device=opt.device)
            masks = datapack['low_mask'].to(dtype=torch.float32, device=opt.device)
            hfcs = datapack['hfc'].to(dtype=torch.float32, device=opt.device)
            bbox = torch.as_tensor(datapack['bbox'], dtype=torch.float32, device=opt.device)
            pt = get_click_prompt(datapack, opt)

            # forward
            # pred = model(imgs, pt, bbox)
            pred = model(imgs, hfcs, pt, bbox)
            train_loss = criterion(pred, masks)

            # backward
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_losses += train_loss.item()
            print(train_loss)

            # adjust lr & optimizer
            if args.warmup and iter_num < args.warmup_period:
                lr_ = args.base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                    lr_ = args.base_lr * (
                                1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_

            iter_num += 1

        # log the train progress
        print('epoch [{}/{}], train loss:{:.4f}'.format(epoch, opt.epochs, train_losses / (batch_idx + 1)))
        if args.keep_log:
            TensorWriter.add_scalar('train_loss', train_losses / (batch_idx + 1), epoch)
            TensorWriter.add_scalar('learning rate', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
            loss_log[epoch] = train_losses / (batch_idx + 1)

        # eval
        if epoch % opt.eval_freq == 0:
            model.eval()
            dices, mean_dice, _,val_losses = get_eval(valloader=valloader, model=model,criterion=criterion,opt=opt,args=args)
            print('epoch [{}/{}], val loss:{:.4f}'.format(epoch, opt.epochs, val_losses))
            print('epoch [{}/{}], val dice:{:.4f}'.format(epoch, opt.epochs, mean_dice))
            if args.keep_log:
                TensorWriter.add_scalar('val_loss',val_losses,epoch)
                TensorWriter.add_scalar('dices', mean_dice, epoch)
                dice_log[epoch] = mean_dice

            # 记录最大的dice
            if mean_dice>best_dice:
                best_dice = mean_dice
                timestr = time.strftime('%m%d%H%M')
                if not os.path.isdir(opt.save_path):
                    os.makedirs(opt.save_path)
                save_path = opt.save_path + args.modelname + opt.save_path_code + '%s' % timestr + '_' + str(
                    epoch) + '_' + str(best_dice)
                torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)

        # save
        if epoch % opt.save_freq ==0 or epoch == opt.epochs-1:
            if not os.path.isdir(opt.save_path):
                os.makedirs(opt.save_path)
            timestr = time.strftime('%m%d%H%M')
            save_path = opt.save_path + args.modelname + opt.save_path_code + '_' + '%s' % timestr + '_' + str(epoch)
            torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)

            # 写记录文件
            if args.keep_log:
                with open(opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr + '/trainloss.txt', 'w') as f:
                    for i in range(len(loss_log)):
                        f.write(str(loss_log[i])+'\n')
                with open(opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr + '/dice.txt', 'w') as f:
                    for i in range(len(dice_log)):
                        f.write(str(dice_log[i])+'\n')


if __name__ == '__main__':
    main()

