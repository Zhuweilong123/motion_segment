import argparse
import os
import torch
import torch.optim as optim
import datetime
from tqdm import tqdm
# import hiddenlayer as hl
import sys

sys.path.append("/home/zhanl/data/code/encoder/")

import random
import numpy as np
from torch.utils.data import DataLoader
from visdom import Visdom

from models.network.net import Rigidnet
from dataset.original_method.kitti_dataset_223 import KITTIDataset
from utils.loss import motion_loss, seg_loss
from dataset.utils.warper import invert_intrinsics_matrix

torch.set_default_tensor_type(torch.FloatTensor)
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"


### 本部分是为了进行 实验复现
seed = 3
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_args():
    # 建立解析对象
    parser = argparse.ArgumentParser()

    # 给实例添加属性
    parser.add_argument("--model_name", type=str, default=None, help="是否要加载模型继续训练")
    parser.add_argument("--batch_size", type=int, default=128, help="训练时每次加载数据的数量")
    parser.add_argument("--epochs", type=int, default=100, help="迭代次数")
    parser.add_argument("--lr", type=float, default=0.001, help="初始学习率")
    parser.add_argument("--acc_best", type=float, default=0.90, help="目标准确度")
    parser.add_argument("--dataset", type=str, default="/home/zhanl/data/kitti/data_splits/label/train.txt",
                        help="记录了所有训练集路径的文件路径")
    parser.add_argument("--dynamic_trainset", type=str,
                        default="/home/zhanl/data/kitti/data_splits/label/dynamic/train.txt",
                        help="记录了动态训练集路径的文件路径")
    parser.add_argument("--dynamic_valset", type=str,
                        default="/home/zhanl/data/kitti/data_splits/label/dynamic/val.txt",
                        help="记录了动态验证集路径的文件路径")
    parser.add_argument("--static_trainset", type=str,
                        default="/home/zhanl/data/kitti/data_splits/label/static/train.txt",
                        help="记录了静态训练集路径的文件路径")
    parser.add_argument("--static_valset", type=str,
                        default="/home/zhanl/data/kitti/data_splits/label/static/val.txt",
                        help="记录了静态验证集路径的文件路径")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/encoder.pth", help="存放模型检查点文件的路径")
    parser.add_argument("--Margin", type=float, default=0.7, help="三元组损失中负样本对应的大阈值")  # 静态和相邻帧之间的距离只有0.1则动态和其相邻的帧之间的距离有0.6（假设）这个可以根据分布自适应
    parser.add_argument("--margin", type=float, default=0.15, help="三元组损失中正样本对应的小阈值")  # 静态和任意静态之间的距离只有0.5+α 但和任意动态帧的帧之间的距离0.8+阿尔法（假设有） 这个可以根据推荐系统自适应
    parser.add_argument("--threshold", type=float, default=0.2, help="判断该帧是否是动态帧的阈值")
    parser.add_argument("--motion_field_burning_steps", type=int, default=20000, help="")
    parser.add_argument("--accumulate-grad-batches", dest="accumulate_grad_batches", type=int, default=4)

    args = parser.parse_args()  # 实例化，把上面的属性给args，之后就可以直接调用
    return args


def main():
    args = parse_args()
    time1 = datetime.datetime.now()

    ### 查看可用gpu 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # torch.multiprocessing.set_start_method('spawn')

    train_dataset = KITTIDataset(static_file=args.static_trainset,
                                 dynamic_file=args.dynamic_trainset,
                                 with_dynamic=True, depth_type='groundtruth')  # 包含了拼接后的rgb、depth、pose和视觉相似度

    train_num = len(train_dataset)
    train_loader = DataLoader(
        train_dataset, args.batch_size,
        shuffle=True, num_workers=4,
        # collate_fn=lambda x:x,
        pin_memory=True
    )

    val_dataset = KITTIDataset(static_file=args.static_valset,
                               dynamic_file=args.dynamic_valset,
                               with_dynamic=True, depth_type='groundtruth')
    val_num = len(val_dataset)
    val_loader = DataLoader(
        val_dataset, args.batch_size,
        shuffle=True, num_workers=4,
        # collate_fn=lambda x:x,
        pin_memory=True
    )

    print(' prepared datasets...')
    ## 输出用于训练以及验证的数据集数据的个数
    #print("using {} images for training.".format(train_num))
    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    # 实例化网络模型
    net = Rigidnet()
    net = net.type(torch.FloatTensor)
    ### 如果有模型的话就进行加载模型
    if os.path.exists(args.checkpoint_dir):
        print("---Loading_Model---")
        net.load_state_dict(torch.load(args.checkpoint_dir, map_location=device))
    net = torch.nn.DataParallel(net)
    net.to(device)

    # 每个loss的权重
    default_loss_weights = {
        'rgb_consistency': 1.0,
        'ssim': 3.0,
        'depth_consistency': 0.05,
        #'depth_smoothing': 0.05,
        #'rotation_cycle_consistency': 1e-3,
        'translation_cycle_consistency': 5e-2,
        #'depth_variance': 0.0,
        'motion_smoothing': 1.0,
        'motion_drift': 0.2,
    }
    loss_function1 = seg_loss.CrossEntropyLoss2d()
    loss_function2 = motion_loss.DMPLoss(default_loss_weights)





    ############################################可视化#######################################################
    # 可视化visdom类
    vis = Visdom()
    vis.line([0.], [0.],  # Y坐标的起点和X坐标的起点
             win='train_loss',  # 窗口名称
             opts=dict(title="train loss", xlabel="epoch", ylabel="loss"))  # 标题和横坐标纵坐标名称
    vis.line([0.], [0.], win="val_acc", opts=dict(title="acc", xlabel="epoch", ylabel="acc"))
    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # 自适应学习率

    train_steps = len(train_loader)  ###  将数据分为多少批次    4300/batch_size
    base_step = (train_steps) // args.accumulate_grad_batches
    print("train_steps", train_steps)
    #rows = []  # 记录的动态池列表
    print(' Initialized models...\n')


    ##################################################初始动态池构建####################################################

    for epoch in range(args.epochs):
        # train
        net.train()
        running_loss = 0.0  ## 初始化在一个epoch中的损失值
        train_bar = tqdm(train_loader)  ## 进度条库
        # prefetcher = data_prefetcher(train_loader)  # 每个epoch都重新包一次，作为一个一次性的train_loader

        for step, data in enumerate(train_bar, 1):  # 一次迭代要运行25次的batchsize，这里的data1从第二帧开始读取（为了防止下面的try except重新迭代一次）
            # 防止出现内存泄露的的错误，这里不使用zip而是iter(dataloader)来创建一个_DataLoaderIter对象
            """data = prefetcher.next()
            iteration = 0
            
            while data is not None:
                iteration += 1
                print(iteration)"""
            batch = data['rgb'].shape[0]
            #flow = data['flow_'].to(device)
            warped = data['warped_'].to(device)
            depth = data['depth_'].to(device)  #每个depth和相邻concate
            target_scales = [data[s].to(device) for s in ['label1', 'label2', 'label3', 'label4', 'label5']]

            # mask的类别有1→car,2→Van,3→Truck,4→Pedestrian,5→Sitter,6→Cyclist,7→Tram,8→Misc
            # label+数字表示不同的scales
            #targ_scales = [data[s].to(device) for s in [3, 4, 5, 6, 7]]

            optimizer.zero_grad()  ###梯度归零:具体的位置应该放在哪里？
            # 预测的语义分割图对应的是哪张rgb图像？
            pred_scales, field = net(warped, depth)  # pred_scales是不同deconv层输出的不同size的语义图，field记录了除去相机自运动后每个像素点的3D translation（xyz轴的偏移值）
            # field由[B,3,H,W]→[B,1,H,W]，三个方向的偏移值求平方和得到每个点总的移动距离
            distance = torch.sqrt(torch.add(torch.add(torch.pow(field[:,0,:,:], 2), torch.pow(field[:,1,:,:], 2)),
                                            torch.pow(field[:,2,:,:], 2))).unsqueeze(1)
            """归一化→得到一个dynamic map:按照普通归一化得到这个dynamic map和真正的动态概率不同，只是把数值缩小到0~1之间，还存在一些问题，
            比如说如果图片中只有一个移动的物体，即使移动距离大但是分子少也会导致最后求出来的移动概率小（怎么对比不同帧）"""
            distance1 = distance.reshape(batch, -1)
            mean = distance1.mean(dim=1).reshape(batch, 1, 1, 1)
            std = distance1.std(dim=1, unbiased=False).reshape(batch, 1, 1, 1)
            dynamic_map = (distance - mean)/std  # 最后输出的dynamic_map同一物体区域的值要相同（可以求均值）

            # 新问题：语义图输出了多种不同的scales， 所以对应的motion field也要有多种对应的pyramid
            # 可以依照几类物体最大的移动速度排序作为权重，即对应的label要比行人的label大
            motion_seg = pred_scales[0] * dynamic_map  # 即instance motion mask，可作为eval的输出
            #把motion seg里面所有不为0数值改为1，1为动态像素，0为静态像素
            zero = torch.zeros(motion_seg.shape)
            valid_mask = torch.gt(motion_seg, zero).type(torch.int8)

            ################                           字典：深度、rgb、T(u,v)                            ################
            endpoints = {}
            # 接下来的代码是为了把相同图像对应的rgb、depth拼起来
            rgb_images = data['rgb']  # [B, 6, H, W]
            rgb_seq_images = torch.split(rgb_images, rgb_images.shape[1] // 2, dim=1)  # [[B, 3, H, W], [B, 3, H, W]]
            #rgb_images = torch.cat((rgb_seq_images[0], rgb_seq_images[1]), dim=0)  # 前后两张rgb = source and target image
            # depth_images = self.depth_net(rgb_images)
            depth_images = data['depth']  # [B, 2, H, W]
            depth_seq_images = torch.split(depth_images, depth_images.shape[1] // 2, dim=1)  # 在batch_size维度把一分为2（第二个参数是块的大小）输出的数据为tensor([]), tensor([])
            endpoints['groundtruth_depth'] = depth_seq_images  # 0是1-N,1是0-N-1
            endpoints['rgb'] = rgb_seq_images

            motion_features = [
                torch.cat((endpoints['rgb'][0],
                           endpoints['predicted_depth'][0]), dim=1),  # [B, 4, H, W] 一个source
                torch.cat((endpoints['rgb'][1],
                           endpoints['predicted_depth'][1]), dim=1)]  # [B, 4, H, W] 一个target
            motion_features_stack = torch.cat(motion_features, dim=0)  # source + target
            flipped_motion_features_stack = torch.cat(motion_features[::-1], dim=0)  # [::-1]表示倒序，切步长为-1，即可以理解为target+source

            # 得到[B, 8, H, W]作为原Motion network的输入得到静态背景的运动和运动物体的运动信息（3d平移）
            pairs = torch.cat([motion_features_stack, flipped_motion_features_stack], dim=1)
            # rot, trans, residual_translation, intrinsics_mat = self.object_motion_net(pairs)

            rot, trans, intrinsics_mat = data['rot'], data['trans'], data['intrinsics']  # 直接输出groundtruth得到的pose值

            endpoints['validity_mask'] = valid_mask
            residual_translation = field
            #
            if args.motion_field_burning_steps > 0.0:
                # steps是什么？
                steps = base_step * epoch
                steps = torch.tensor(steps).type(torch.FloatTensor)
                burnin_steps = torch.tensor(args.motion_field_burning_steps).type(
                    torch.FloatTensor)
                residual_translation *= torch.clamp(2 * steps / burnin_steps - 1, 0.0,
                                                    1.0)
            endpoints['residual_translation'] = torch.split(residual_translation,
                                                            residual_translation.shape[0] // 2, dim=0)
            endpoints['background_translation'] = torch.split(trans,
                                                              trans.shape[0] // 2, dim=0)
            endpoints['rotation'] = torch.split(rot, rot.shape[0] // 2, dim=0)
            intrinsics_mat = 0.5 * sum(
                torch.split(intrinsics_mat,
                            intrinsics_mat.shape[0] // 2, dim=0))
            # 这里乘以2是因为通过数据增（翻转）得到了2*N组数组，对应的内参也要加倍
            endpoints['intrinsics_mat'] = [intrinsics_mat] * 2
            endpoints['intrinsics_mat_inv'] = [invert_intrinsics_matrix(intrinsics_mat)] * 2
            ################                           字典：深度、rgb、T(u,v)                            ################



            # 两个三元组，一个由当前帧和任一相邻帧和正样本，一个由当前帧和任一相邻帧或者正样本和负样本
            loss =  loss_function1(endpoints).float() + \
                   loss_function2(pred_scales, target_scales).float()


            loss.backward()  ### 反向传播
            optimizer.step()  ## 损失优化

            ##########   可视化打印
            running_loss += loss.item()  ## 累加损失
            #train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, args.epochs, loss)
            print("train epoch[{}/{}] loss:{:.3f}".format(step + 1, epoch, loss))
            #vis.line([loss.item()], [step], win='train_loss', update='append')
        scheduler.step()  # 更新学习率
        running_loss /= train_steps  # train_steps是dataloader读完整个训练集需要的次数
        vis.line([running_loss], [epoch], win='train_loss', update='append')  # append表示添加到上一个点的后面

        ################################################## 验证集：评估模型 ##############################################
        """net.eval()
        
        # 验证的时候不需要梯度
        val_bar = tqdm(val_loader)
        val_acc = 0
        total_correct = 0
        with torch.no_grad():
            for step, data in enumerate(val_bar):
                batch = data[0].shape[0]  # 一个batch里面的数据数量
                #label = torch.zeros([batch, 1]).to(device)  # 全部静态帧

                '''s1 = net(data[0].to(device), data[1].to(device))
                s2 = net(data[2].to(device), data[3].to(device))
                # 相似度是128维的，要批量和阈值作对比
                s = s1.lt(args.threshold).float() + s2.lt(args.threshold).float()'''  # 128x2，只有值全为1加起来为2的帧才是动态帧
                #threshold = (args.margin + args.Margin)/2
                last = net(data[4].to(device), data[5].to(device))
                next = net(data[6].to(device), data[7].to(device))
                s = last.lt(args.threshold).float() + next.lt(args.threshold).float()  # 前后帧的距离都小于该值即相异度，则证明其是静态帧
                tmp = torch.full([batch, 1], 2).to(device)  # 创建一个值全部为2的128x1维张量
                pred = torch.eq(s, tmp).float().to(device)  # 只保留两个值都为1的下标，预测其为静态帧，记录的值为1
                correct = pred.sum().item()  # 这里记录为1的为静态帧即为预测成功的

                # 输出预测和真实标签相同的图片数量
                #correct = torch.eq(pred, label).float().sum().item()  # .float()将truefalse转成1和0，.sum()将所有值加起来，还是tensor型数据，.item()可以得到值
                total_correct += correct
                # 还可以加上motion_segmentation
            val_acc = total_correct / val_num
            print(epoch, 'val acc:', val_acc)
            vis.line([val_acc], [epoch], win='val_acc', update='append')

        if (epoch + 1) % 10 == 0:
            if not os.path.exists("checkpoint"):
                os.mkdir("checkpoint")
            # if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
            # os.mkdir(os.path.join("checkpoint", opt.dataset))
            net_path = "checkpoint/net_epoch_{}.pth".format(epoch)
            torch.save(net, net_path)
        '''# 保存验证集准确度最高的网络每层的参数
        if val_acc < acc_best:
            acc_best = val_acc
            torch.save({'state_dict': net.state_dict(), 'epoch': epoch}, 'SimilarityNet_' + str(epoch) + '_best.pkl')
            print('Save best statistics done!')'''
        # 保存准确度高于给定值的模型
        if val_acc >= args.acc_best:
            net_best_path = "checkpoint/net_epoch_{}_best.pth".format(epoch)
            torch.save(net, net_best_path)"""

    print('Finished Training')
    #print(rows)


if __name__ == '__main__':
    main()
