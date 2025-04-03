import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import numpy as np
import cv2
import math
import torchvision.transforms as transforms
from hasiloss import RMI_ir,RMI_vi


def dis_loss_func(vis_output, fusion_output):
    # a = torch.mean(torch.square(vis_output - torch.Tensor(vis_output.shape).uniform_(0.7, 1.2)))
    return torch.mean(torch.square(vis_output - torch.Tensor(vis_output.shape).uniform_(0.7, 1.2).cuda())) + \
           torch.mean(torch.square(fusion_output.cuda() - torch.Tensor(fusion_output.shape).uniform_(0, 0.3).cuda()))

def loss_I(real_pair, fake_pair):
    batch_size = real_pair.size()[0]
    real_pair = 1 - real_pair
    real_pair = real_pair ** 2
    fake_pair  = fake_pair ** 2
    real_pair = torch.sum(real_pair)
    fake_pair = torch.sum(fake_pair)
    return (real_pair + fake_pair) / batch_size

def w_loss(img_ir):
    w1d = F.max_pool2d(img_ir, 2, 2)
    w2d = F.max_pool2d(w1d, 2, 2)
    w2u = F.upsample_bilinear(w2d, scale_factor=2)
    w_ir = F.upsample_bilinear(w2u, scale_factor=2)
    w_ir = F.softmax(w_ir, 0)
    w_vi = 1-w_ir
    return w_ir, w_vi


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)                            # sigma = 1.5    shape: [11, 1]
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)    # unsqueeze()函数,增加维度  .t() 进行了转置 shape: [1, 1, 11, 11]
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()   # window shape: [1,1, 11, 11]
    return window


# 计算 ssim 损失函数
def mssim(img1, img2, window_size=11):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).

    max_val = 255
    min_val = 0
    L = max_val - min_val
    padd = window_size // 2


    (_, channel, height, width) = img1.size()

    # 滤波器窗口
    window = create_window(window_size, channel=channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
    ret = ssim_map
    return ret

def mse(img1, img2, window_size=9):
    max_val = 255
    min_val = 0
    L = max_val - min_val
    padd = window_size // 2

    (_, channel, height, width) = img1.size()

    img1_f = F.unfold(img1, (window_size, window_size), padding=padd)
    img2_f = F.unfold(img2, (window_size, window_size), padding=padd)

    res = (img1_f - img2_f) ** 2

    res = torch.sum(res, dim=1, keepdim=True) / (window_size ** 2)

    res = F.fold(res, output_size=(256, 256), kernel_size=(1, 1))
    return res


# 方差计算
def std(img,  window_size=9):

    padd = window_size // 2
    (_, channel, height, width) = img.size()
    window = create_window(window_size, channel=channel).to(img.device)
    mu = F.conv2d(img, window, padding=padd, groups=channel)
    mu_sq = mu.pow(2)
    sigma1 = F.conv2d(img * img, window, padding=padd, groups=channel) - mu_sq

    return sigma1

def sum(img,  window_size=9):

    padd = window_size // 2
    (_, channel, height, width) = img.size()
    window = create_window(window_size, channel=channel).to(img.device)
    win1 = torch.ones_like(window)
    res = F.conv2d(img, win1, padding=padd, groups=channel)
    return res



def final_ssim(img_ir, img_vis, img_fuse):

    ssim_ir = mssim(img_ir, img_fuse)
    ssim_vi = mssim(img_vis, img_fuse)

    # std_ir = std(img_ir)
    # std_vi = std(img_vis)
    std_ir = std(img_ir)
    std_vi = std(img_vis)

    zero = torch.zeros_like(std_ir)
    one = torch.ones_like(std_vi)

    # m = torch.mean(img_ir)
    # w_ir = torch.where(img_ir > m, one, zero)

    map1 = torch.where((std_ir - std_vi) > 0, one, zero)
    map2 = torch.where((std_ir - std_vi) >= 0, zero, one)

    ssim = map1 * ssim_ir + map2 * ssim_vi
    # ssim = ssim * w_ir
    return ssim.mean()

def tensor_entropy(img):
    # 计算概率分布
    probs = F.softmax(img, dim=1)
    
    # 计算熵
    img_entropy = -(probs * torch.log2(probs + 1e-10)).sum(dim=1)
    
    return img_entropy


def final_mi(img_ir, img_vis, img_fuse):

    ir_entropy = tensor_entropy(img_ir)
    vis_entropy = tensor_entropy(img_vis)
    total = ir_entropy + vis_entropy + 0.01
    mi_ir = RMI_ir(img_ir, img_fuse)
    mi_vi = RMI_vi(img_vis, img_fuse)


    # mi = vis_entropy * mi_ir + ir_entropy * mi_vi
    mi = ((vis_entropy)/total) * mi_ir + ((ir_entropy)/total) * mi_vi
    # ssim = ssim * w_ir
    return mi.mean()


def final_mse(img_ir, img_vis, img_fuse):
    mse_ir = mse(img_ir, img_fuse)
    mse_vi = mse(img_vis, img_fuse)

    std_ir = std(img_ir)
    std_vi = std(img_vis)
    # std_ir = sum(img_ir)
    # std_vi = sum(img_vis)

    zero = torch.zeros_like(std_ir)
    one = torch.ones_like(std_vi)

    m = torch.mean(img_ir)
    w_vi = torch.where(img_ir <= m, one, zero)

    map1 = torch.where((std_ir - std_vi) > 0, one, zero)
    map2 = torch.where((std_ir - std_vi) >= 0, zero, one)

    res = map1 * mse_ir + map2 * mse_vi
    res = res * w_vi
    return res.mean()


def windows_mse(img_ir, img_vis, img_fuse):
    # 计算红外图像与融合图像之间的 MSE
    mse_ir_fuse = mse(img_ir, img_fuse)
    # 计算可见光图像与融合图像之间的 MSE
    mse_vis_fuse = mse(img_vis, img_fuse)
    
    res = mse_ir_fuse + mse_vis_fuse
    
    # 返回两个 MSE 的平均值
    return res.mean()





def add_edges_to_image(image):
    # 定义 Sobel 算子
    sobel_filter_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    sobel_filter_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

    # gaussian_blur = transforms.GaussianBlur(kernel_size=5, sigma=1)
    # image_smoothed = gaussian_blur(image)

    # 对输入图像进行填充
    padded_image = F.pad(image, (1, 1, 1, 1), mode='reflect')  # 使用反射填充模式

    # 在图像上应用 Sobel 算子进行边缘检测
    sobelx = F.conv2d(padded_image, sobel_filter_x.unsqueeze(0).unsqueeze(0))
    sobely = F.conv2d(padded_image, sobel_filter_y.unsqueeze(0).unsqueeze(0))

    # 计算梯度幅值
    sobel_mag = torch.sqrt(sobelx ** 2 + sobely ** 2)

    # 将梯度幅值归一化到0-1范围
    sobel_mag /= torch.max(sobel_mag)

    # 调整边缘图像大小以匹配原始图像的尺寸
    edge_image_resized = F.interpolate(sobel_mag, size=image.shape[2:], mode='nearest')

    # 将边缘图像叠加到原始图像上
    overlay = image + edge_image_resized

    return overlay, edge_image_resized









if __name__ == '__main__':
    criterion = mssim
    input = torch.rand([1, 1, 64, 64])
    output = torch.rand([1, 1, 64, 64])
    img_fuse = torch.rand([1, 1, 64, 64])
    uw = torch.Tensor(np.ones((11, 11), dtype=float)) / 11
    uw = uw.float().unsqueeze(0).unsqueeze(0)
    print(uw)
    input = input.cuda()
    output = output.cuda()
    img_fuse = img_fuse.cuda()
    ssim = final_ssim(input,  output, img_fuse)
    print(ssim)
