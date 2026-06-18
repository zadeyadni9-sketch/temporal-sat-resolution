import math
import torch
import torch.nn.functional as F

# CHANGE THESE paths to match where your tensors or images are
# For now, we assume you can load three tensors:
#   low_res (input), sr (model output), hr (ground truth)

def psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return 999
    return 10 * math.log10(1.0 / mse.item())

def ssim(pred, target):
    # Very simple SSIM over a single image; good enough as a first metric
    # pred and target in [0, 1]
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = pred.mean()
    mu_y = target.mean()
    sigma_x = pred.var()
    sigma_y = target.var()
    sigma_xy = ((pred - mu_x) * (target - mu_y)).mean()

    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return (num / den).item()

def main():
    # TODO: replace the following with actual loading code.
    # Example shape: (1, 1, H, W) or (1, 3, H, W), values in [0, 1]
    #
    # low_res = ...
    # sr = ...
    # hr = ...

    raise NotImplementedError("Load low_res, sr, hr and then compute metrics")

    # Example once you have tensors:
    # print("Input vs Target:")
    # print("PSNR:", psnr(low_res, hr))
    # print("SSIM:", ssim(low_res, hr))
    #
    # print("SR vs Target:")
    # print("PSNR:", psnr(sr, hr))
    # print("SSIM:", ssim(sr, hr))

if __name__ == "__main__":
    main()