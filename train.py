import time
from tqdm import tqdm
import os
import random
import argparse
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from utils.visualizer import Visualizer
from utils.loss import seg_loss, motion_loss
from utils import scheduler, stream_metrics

from dataset.seg_dataset import KITTIDataset
from dataset.utils.warper import inverse_warp2
from monodepth2 import layers
from models.RMS import RMS


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--train_file", type=str, default='/home/zhanl/data/code/motion_seg/data/train.txt')
    parser.add_argument("--val_file", type=str, default='/home/zhanl/data/code/motion_seg/data/val.txt')
    parser.add_argument("--dataset", type=str, default='kitti',
                        choices=['voc', 'cityscapes', 'kitti'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=9,  # FIXME 一共7个class，加上背景是8个
                        help="num classes (default: None)")
    parser.add_argument("--resize_factor", type=tuple, default=(128, 416), help='the size of input images')

    # Deeplabv3+ Options
    parser.add_argument("--model", type=str, default='RMS', help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")  # a simple tool ``network.convert_to_separable_conv`` to convert ``nn.Conv2d`` to ``AtrousSeparableConvolution``
    #parser.add_argument("--pretrained_model", type=str, default=, help='the path of pretrained mobilenetV2_ca')
    parser.add_argument("--downsample_factor", type=int, default=8, choices=[8, 16])  # output_stride

    # Motion Options
    #parser.add_argument()
    #parser.add_argument()

    # Train Options
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")  # 迭代次数设置多少好

    parser.add_argument("--test_only", action='store_true', default=False, help='是否只验证不训练')
    parser.add_argument("--val_interval", type=int, default=100,  # TODO
                        help="epoch interval for eval (default: 100)多少个epoch评估一次并保存最新权值，评估消耗时间多，频繁的评估会导致训练慢")
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")  # 是否要保存每次验证的结果
    parser.add_argument("--save_dir", type=str, default='/home/zhanl/data/code/motion_seg/logs', help="权值和日志文件保存的文件夹")

    parser.add_argument("--lr", type=float, default=0.01,
                        help="初始learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)  # 学习率多久下降一次
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=16,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=10.0, help="超参:语义分割任务损失函数占比")
    parser.add_argument("--beta", type=float, default=1.0, help="超参：运动估计任务损失函数占比")
    # TODO
    parser.add_argument("--cls_weight", type=str, help="是否给不同种类赋予不同的损失权值，默认是平衡的。设置的话，注意设置成numpy形式的，长度和num_classes一样。")

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")  # 保存了要继续训练的本地模型
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type")  # 由于类别少于10类，除了交叉熵损失还加上了dice loss，focal loss主要为了防止正负样本的不平衡

    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")

    parser.add_argument("--gpu_id", type=str, default='0, 1, 2, 3',
                        help="GPU ID")
    parser.add_argument("--num_workers", type=int, default=4, help="多线程读取数据")

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=True,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    train_transform = et.ExtCompose([
        et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
        et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    val_transform = et.ExtCompose([
        # et.ExtResize( 512 ),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])"""
    if opts.dataset == 'kitti':
        #可能需要做一些图片增亮操作
        train_dst = KITTIDataset(file=opts.train_file, new_size=opts.resize_factor)
        val_dst = KITTIDataset(file=opts.val_file, new_size=opts.resize_factor)
    return train_dst, val_dst


def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()  # 重置对象的状态
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        # 防止visdom显示图像可见度不高，需要进行反归一化，但是原kitti图像输入到网络中都没有进行normalize
        #denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():

        for i, data in tqdm(enumerate(loader)):

            labels, Ir, It, Dr, Dt, pose, intrinsics_ = (data[s].to(device) for s in ['label', 'Ir', 'It', 'Dr', 'Dt', 'pose', 'intrinsics_mat'])
            warped_image, warped_depth = inverse_warp2(Ir, Dt, Dr, pose, intrinsics=intrinsics_,
                                                       rotation_mode='euler',
                                                       padding_mode='zeros')  # TODO 投影的方式不确
            # residual warped image 和 residual warped depth
            residual_image = torch.abs(It - warped_image)
            residual_depth = torch.abs(Dt - warped_depth)

            Segr, Seg, res_trans = model(torch.cat((It, Ir), dim=1),  # visual cues
                                       torch.cat((Dt, Dr, residual_depth, residual_image), dim=1))  # geometric cues
            # 分割结果seg没有经过归一化，这里输出的可能全是负数，经过np.uint8之后全部截断为0
            preds = F.softmax(Seg, dim=1).detach().max(dim=1)[1].cpu().numpy()  # 返回的是每个通道最大值
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)

            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples,选择一些结果可视化
                ret_samples.append(
                    (It[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(It)):
                    image = It[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]
                    #image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    image = It.transpose(1, 2, 0).astype(np.uint8)
                    # decode_target是一个classmethod（相当于修改构造函数），目的是将seg mask转换成rgb图像，(N, H, W, 3), ranged 0~255, numpy array
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)
                    #本地存储图片
                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)
                    img_id += 1
        score = metrics.get_results()
    return score, ret_samples


def main():
    opts = get_argparser().parse_args()

    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id  # 0,1,2,3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    train_dst, val_dst = get_dataset(opts)
    train_loader = DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers,
        drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=opts.num_workers)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # num_classes*2输出两张图片的分割结果
    model = RMS(image_size=opts.resize_factor, num_classes=opts.num_classes*2, downsample_factor=opts.downsample_factor)
    # 对decoder应用空洞卷积
    """if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)"""
    scheduler.set_bn_momentum(model.backbone, momentum=0.01)

    # Set up metrics
    metrics = stream_metrics.StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    '''params=[
            {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
             {'params': model.classifier.parameters(), 'lr': opts.lr},
             {'params': model.seg_decoder.parameters(), 'lr': opts.lr},
             {'params': model.motion_decoder.parameters(), 'lr':opts.lr},]'''
    optimizer = torch.optim.SGD(model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy == 'poly':
        scheduler_ = scheduler.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler_ = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion1 = seg_loss.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion1 = seg_loss.CELoss()
    criterion2 = motion_loss.MotionLoss()

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler_.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        # 继续之前中断的训练
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler_.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # ==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    #denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        return

    interval_loss = 0
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for data in train_loader:
            cur_itrs += 1

            labels, Ir, It, Dr, Dt, pose, intrinsics_ = (data[s].to(device) for s in ['label', 'Ir', 'It', 'Dr', 'Dt', 'pose', 'intrinsics_mat'])
            warped_image, warped_depth = inverse_warp2(Ir, Dt, Dr, pose, intrinsics=intrinsics_,
                                                       rotation_mode='euler', padding_mode='zeros')  # TODO 投影的方式不确
            # residual warped image 和 residual warped depth
            residual_image = torch.abs(It-warped_image)
            residual_depth = torch.abs(Dt-warped_depth)
            optimizer.zero_grad()
            # 前半部分是参考帧即t-1帧的分割结果, 后半部分是目标帧即t帧的分割结果
            Segr, Segt, res_trans = model(torch.cat((Ir, It), dim=1),  # visual cues
                                       torch.cat((Dr, Dt, residual_depth, residual_image), dim=1))  # geometric cue

            # appearance→occlusion mask,这里使用的entropycross损失函数内部会进行一次softmax，所以不需要再归一化了
            loss1 = criterion1(Segt, labels)  # TODO 可以加上warped seg和原seg的差值（可以加上occlusion mask）:这里得到的mask可以再判断其准确度是否能用在motion_loss里面

            #这里的两个mask也是没有经过softmax归一化的 TODO
            loss2 = criterion2(Ir, It, Dr, Dt, pose, intrinsics_, res_trans, torch.argmax(Segt, dim=1, keepdim=True),
                               torch.argmax(Segr, dim=1, keepdim=True), loss1)

            loss = opts.alpha * loss1 + opts.beta * loss2  # TODO 损失占比还不确定
            loss.backward()
            optimizer.step()

            np_loss, np_loss1, np_loss2 = loss.detach().cpu().numpy(), loss1.detach().cpu().numpy(), loss2.detach().cpu().numpy()
            interval_loss += np_loss
            if vis is not None:
                vis.vis_scalar('SegLoss', cur_itrs, np_loss1)  # 分开展示
                vis.vis_scalar('MotionLoss', cur_itrs, np_loss2)

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0

            # validation  这里每训练160个数据就要测试2574个数据？？？
            if (cur_itrs) % opts.val_interval == 0:
                save_ckpt('logs/latest_%s_%s_os%d.pth' %
                          (opts.model, opts.dataset, opts.downsample_factor))
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(  # dict:5 和
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
                    ret_samples_ids=vis_sample_id)

                print(metrics.to_str(val_score))

                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt('logs/best_%s_%s_os%d.pth' %
                              (opts.model, opts.dataset, opts.downsample_factor))

                if vis is not None:  # visualize validation score and samples
                    vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                    vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                    vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                    for k, (img, target, lbl) in enumerate(ret_samples):
                        img = (img*255).astype(np.uint8)  # [3, 128, 416]
                        target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)  # [3, 128, 416]
                        lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)  # [3, 128, 416]:最后输出的全变成了0
                        #moving_prediction =  # TODO
                        concat_img = np.concatenate((img, target, lbl), axis=1)  # [3, 384, 416]
                        vis.vis_image('Sample %d' % k, concat_img)
                model.train()
            scheduler_.step()

            if cur_itrs >= opts.total_itrs:
                return


if __name__ == '__main__':
    torch.multiprocessing.set_start_method(method='forkserver', force=True)#'spawn')
    main()