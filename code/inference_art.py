import numpy
import torch
from torch.autograd import Variable
import torchvision.utils as vutils


def strToBool(str):
	return str.lower() in ('true', 'yes', 'on', 't', '1')

import argparse
parser = argparse.ArgumentParser()
parser.register('type', 'bool', strToBool)
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--predict_size', type=int, default=256)
parser.add_argument('--n_colors', type=int, default=3)
parser.add_argument('--z_size', type=int, default=128)
parser.add_argument('--G_h_size', type=int, default=128, help='Number of hidden nodes in the Generator. Used only in arch=0. Too small leads to bad results, too big blows up the GPU RAM.') # DCGAN paper original value
parser.add_argument('--SELU', type='bool', default=False, help='Using scaled exponential linear units (SELU) which are self-normalizing instead of ReLU with BatchNorm. Used only in arch=0. This improves stability.')
parser.add_argument("--NN_conv", type='bool', default=False, help="This approach minimize checkerboard artifacts during training. Used only by arch=0. Uses nearest-neighbor resized convolutions instead of strided convolutions (https://distill.pub/2016/deconv-checkerboard/ and github.com/abhiskk/fast-neural-style).")
parser.add_argument('--cuda', type='bool', default=True, help='enables cuda')
parser.add_argument('--n_gpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--spectral_G', type='bool', default=False, help='If True, use spectral normalization to make the generator Lipschitz (Generally only D is spectral, not G). This Will also remove batch norm in the discriminator.')
parser.add_argument('--gen_extra_images', type=int, default=200, help='Generate additional images with random fake cats in calculating FID (Recommended to use the same amount as the size of the dataset; for CIFAR-10 we use 50k, but most people use 10k) It must be a multiple of 100.')
parser.add_argument('--extra_folder', default='/home/ubuntu/RelativisticGAN/extra', help='Folder for extra photos (different so that my dropbox does not get overwhelmed with 50k pictures)')
parser.add_argument('--path_model', default='../extra/models/state_22.pth', help='Path to the pretrained model')
parser.add_argument('--no_batch_norm_G', type='bool', default=False, help='If True, no batch norm in G.')
parser.add_argument('--Tanh_GD', type='bool', default=False, help='If True, tanh everywhere.')
param = parser.parse_args()


class DCGAN_G(torch.nn.Module):
	def __init__(self):
		super(DCGAN_G, self).__init__()
		main = torch.nn.Sequential()

		# We need to know how many layers we will use at the beginning
		mult = param.image_size // 8

		### Start block
		# Z_size random numbers
		if param.spectral_G:
			main.add_module('Start-SpectralConvTranspose2d', torch.nn.utils.spectral_norm(torch.nn.ConvTranspose2d(param.z_size, param.G_h_size * mult, kernel_size=4, stride=1, padding=0, bias=False)))
		else:
			main.add_module('Start-ConvTranspose2d', torch.nn.ConvTranspose2d(param.z_size, param.G_h_size * mult, kernel_size=4, stride=1, padding=0, bias=False))
		if param.SELU:
			main.add_module('Start-SELU', torch.nn.SELU(inplace=True))
		else:
			if not param.no_batch_norm_G and not param.spectral_G:
				main.add_module('Start-BatchNorm2d', torch.nn.BatchNorm2d(param.G_h_size * mult))
			if param.Tanh_GD:
				main.add_module('Start-Tanh', torch.nn.Tanh())
			else:
				main.add_module('Start-ReLU', torch.nn.ReLU())
		# Size = (G_h_size * mult) x 4 x 4

		### Middle block (Done until we reach ? x image_size/2 x image_size/2)
		i = 1
		while mult > 1:
			if param.NN_conv:
				main.add_module('Middle-UpSample [%d]' % i, torch.nn.Upsample(scale_factor=2))
				if param.spectral_G:
					main.add_module('Middle-SpectralConv2d [%d]' % i, torch.nn.utils.spectral_norm(torch.nn.Conv2d(param.G_h_size * mult, param.G_h_size * (mult//2), kernel_size=3, stride=1, padding=1)))
				else:
					main.add_module('Middle-Conv2d [%d]' % i, torch.nn.Conv2d(param.G_h_size * mult, param.G_h_size * (mult//2), kernel_size=3, stride=1, padding=1))
			else:
				if param.spectral_G:
					main.add_module('Middle-SpectralConvTranspose2d [%d]' % i, torch.nn.utils.spectral_norm(torch.nn.ConvTranspose2d(param.G_h_size * mult, param.G_h_size * (mult//2), kernel_size=4, stride=2, padding=1, bias=False)))
				else:
					main.add_module('Middle-ConvTranspose2d [%d]' % i, torch.nn.ConvTranspose2d(param.G_h_size * mult, param.G_h_size * (mult//2), kernel_size=4, stride=2, padding=1, bias=False))
			if param.SELU:
				main.add_module('Middle-SELU [%d]' % i, torch.nn.SELU(inplace=True))
			else:
				if not param.no_batch_norm_G and not param.spectral_G:
					main.add_module('Middle-BatchNorm2d [%d]' % i, torch.nn.BatchNorm2d(param.G_h_size * (mult//2)))
				if param.Tanh_GD:
					main.add_module('Middle-Tanh [%d]' % i, torch.nn.Tanh())
				else:
					main.add_module('Middle-ReLU [%d]' % i, torch.nn.ReLU())
			# Size = (G_h_size * (mult/(2*i))) x 8 x 8
			mult = mult // 2
			i += 1

		### End block
		# Size = G_h_size x image_size/2 x image_size/2
		if param.NN_conv:
			main.add_module('End-UpSample', torch.nn.Upsample(scale_factor=2))
			if param.spectral_G:
				main.add_module('End-SpectralConv2d', torch.nn.utils.spectral_norm(torch.nn.Conv2d(param.G_h_size, param.n_colors, kernel_size=3, stride=1, padding=1)))
			else:
				main.add_module('End-Conv2d', torch.nn.Conv2d(param.G_h_size, param.n_colors, kernel_size=3, stride=1, padding=1))
		else:
			if param.spectral_G:
				main.add_module('End-SpectralConvTranspose2d', torch.nn.utils.spectral_norm(torch.nn.ConvTranspose2d(param.G_h_size, param.n_colors, kernel_size=4, stride=2, padding=1, bias=False)))
			else:
				main.add_module('End-ConvTranspose2dNew', torch.nn.ConvTranspose2d(param.G_h_size, param.n_colors, kernel_size=2, stride=2, padding=1, bias=False))
		main.add_module('End-Tanh', torch.nn.Tanh())
		# Size = n_colors x image_size x image_size
		self.main = main

	def forward(self, input):
		if isinstance(input.data, torch.cuda.FloatTensor) and param.n_gpu > 1:
			output = torch.nn.parallel.data_parallel(self.main, input, range(param.n_gpu))
		else:
			output = self.main(input)
		return output


# G = DCGAN_G(*args, **kwargs)
G = DCGAN_G()
G = G.cuda()

checkpoint = torch.load(param.path_model)

G.load_state_dict(checkpoint['G_state'], strict=False)
iter_offset = checkpoint['i']
current_set_images = checkpoint['current_set_images']


# Generate 50k images for FID/Inception to be calculated later (not on this script, since running both tensorflow and pytorch at the same time cause issues)
ext_curr = 0
z_extra = torch.FloatTensor(100, param.z_size, 1, 1)
if param.cuda:
	z_extra = z_extra.cuda()
for ext in range(int(param.gen_extra_images/100)):
	fake_test = G(Variable(z_extra.normal_(0, 1)))
	for ext_i in range(100):
		vutils.save_image((fake_test[ext_i].data*.50)+.50, '%s/%01d/fake_samples_%05d.png' % (param.extra_folder, current_set_images, ext_curr), normalize=False, padding=0)
		ext_curr += 1
del z_extra
