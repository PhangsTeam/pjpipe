import numpy as np
from astropy.convolution import convolve_fft, Gaussian2DKernel
from scipy.ndimage import gaussian_filter as gf

def constrained_diffusion(data, err_rel=3e-2, n_scales=None):
    ntot = min(int(np.log(min(data.shape)) / np.log(2) - 1), n_scales or float('inf'))
    scalecube = np.zeros((ntot, *data.shape))

    for i in range(ntot):
        channel_image = np.zeros_like(data)
        scale_end, scale_begin = 2**(i+1), 2**i
        t_end, t_begin = scale_end**2 / 2, scale_begin**2 / 2
        delta_t_max = t_begin * (0.1 if i == 0 else err_rel)
        niter = int((t_end - t_begin) / delta_t_max + 0.5)
        delta_t = (t_end - t_begin) / niter
        kernel_size = np.sqrt(2 * delta_t)

        for _ in range(niter):
            smooth_image = convolve_fft(data, Gaussian2DKernel(kernel_size)) if kernel_size > 5 else gf(data, kernel_size, mode='constant', cval=0.0)
            sm_image_min, sm_image_max = np.minimum(data, smooth_image), np.maximum(data, smooth_image)
            diff_image = np.zeros_like(data)
            pos_1, pos_2 = np.where((data - sm_image_min > 0) & (data > 0)), np.where((data - sm_image_max < 0) & (data < 0))
            diff_image[pos_1], diff_image[pos_2] = data[pos_1] - sm_image_min[pos_1], data[pos_2] - sm_image_max[pos_2]
            channel_image += diff_image
            data -= diff_image

        scalecube[i] = channel_image

    return scalecube, data