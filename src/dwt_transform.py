import numpy as np
import pywt
import torch
import torchvision



class DWT2Numpy:
    def __init__(self, wavelet):
        self.wavelet = wavelet

    def __call__(self, sample):

        sample_array = np.array(sample)
        if sample_array.ndim == 2:
            sample_array = sample_array[..., None]
        wavelet_channels = []
        for channel_idx in range(sample_array.shape[-1]):
            coeffs = pywt.dwt2(sample_array[..., channel_idx], self.wavelet)
            cA, (cH, cV, cD) = coeffs
            wavelet_channels.extend([cA[None], cH[None], cV[None], cD[None]])
        wave_params = np.concatenate(wavelet_channels, axis=0)
        return wave_params

        sample_r = sample_array[:, :, 0]
        sample_g = sample_array[:, :, 1]
        sample_b = sample_array[:, :, 2]

        coeffs2_r = pywt.dwt2(sample_r, self.wavelet)
        cA_r, (cH_r, cV_r, cD_r) = coeffs2_r
        wave_param_r = np.vstack((np.hstack((cA_r, cH_r)), np.hstack((cV_r, cD_r))))

        coeffs2_g = pywt.dwt2(sample_g, self.wavelet)
        cA_g, (cH_g, cV_g, cD_g) = coeffs2_g
        wave_param_g = np.vstack((np.hstack((cA_g, cH_g)), np.hstack((cV_g, cD_g))))

        coeffs2_b = pywt.dwt2(sample_b, self.wavelet)
        cA_b, (cH_b, cV_b, cD_b) = coeffs2_b
        wave_param_b = np.vstack((np.hstack((cA_b, cH_b)), np.hstack((cV_b, cD_b))))

        wave_param = np.array([wave_param_r, wave_param_g, wave_param_b])
        wave_param = np.transpose(wave_param, (1, 2, 0))

        return np.float32(wave_param)
        # return sample


# class DWT2Tensor:
#     def __init__(self, wavelet):
#         self.wavelet = wavelet

#     def __call__(self, sample):
#         sample = torch.Tensor(sample)
#         # Assume the shape of (N, C, H, W)
#         for channel_idx in range(sample.shape[1]):

