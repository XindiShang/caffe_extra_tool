__author__ = 'xindi shang'

"""
General usage
=============
net.predict(input, oversample = False)
net.blobs['data'].data
net.params['conv1'][0].data
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def load_model(model, weight, gpu = False):
	"""
	Parameters
	==========
	model: prototxt that describes network structure, 
		   e.g. 'deploy.prototxt'.
	weight: caffemodel that's produced by previous train phase, 
	       e.g. 'cifar10_full_iter_60000.caffemodel'.
	gpu: if use the GPU device
	"""
	try:
		import caffe
	except:
		import sys
		sys.path.insert(0, '[caffe_root]/build/python')
		import caffe

	net = caffe.Classifier(model, weight, gpu = gpu)
	return net

def visualize(data, pad_size = 1, pad_val = 0):
	"""
	Parameters
	==========
	data: 4-D tensor: (num, channel, height, width)
	pad_size: interval pixels between two images
	pad_val: interval pixel value
	"""
	assert data.ndim == 4, "Data should be 4-D tensor."
	data -= data.min()
	data /= data.max()

	padding = ((0, 0), (0, 0), (pad_size, 0), (pad_size, 0))
	data = np.pad(data, padding, mode = 'constant', 
			constant_values = (pad_val,))
	data = data.transpose(0, 2, 1, 3)
	data = data.reshape((data.shape[0] * data.shape[1], 
			data.shape[2] * data.shape[3]))
	padding = ((0, pad_size), (0, pad_size))
	data = np.pad(data, padding, mode = 'constant',
			constant_values = (pad_val,))

	plt.imshow(data, cmap = matplotlib.cm.gray, interpolation = 'none')
	plt.axis('off')
	plt.show()

if __name__ == '__main__':
	data = np.random.random((5, 3, 5, 5))
	visualize(data, pad_val = 1)