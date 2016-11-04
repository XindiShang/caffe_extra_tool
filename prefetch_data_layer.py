__author__ = 'Xindi Shang'

import numpy as np
import yaml
from multiprocessing import Process, Queue
from blob_generator import get_blob_generator
from utils import SharedArray

import sys
sys.path.append('/home/xdshang/workspace/caffe/build/install/python')
from caffe import Layer
import atexit, signal

PREFETCH_COUNT = 2

class PrefetchDataLayer(Layer):

  def _cleanup(self):
    print 'Terminating BlobFetcher'
    self._prefetch_process.terminate()
    self._prefetch_process.join()

  def setup(self, bottom, top):
    """Setup the FlowDataLayer."""

    # Parse the layer parameter string, which must be valid YAML
    layer_params = yaml.load(open(self.param_str, 'r'))
    # Get specific blob generator
    self._blob_gen = get_blob_generator(layer_params)

    self._blob_pool = [list() for i in range(PREFETCH_COUNT)]
    self._free_queue = Queue(PREFETCH_COUNT)
    self._full_queue = Queue(PREFETCH_COUNT)

    # Setup top shapes
    shapes = self._blob_gen.get_blob_shapes()
    assert len(top) == len(shapes)
    for i, shape in enumerate(shapes):
      top[i].reshape(*shape)
      # Allocate shared memory to blob_pool
      for j in range(PREFETCH_COUNT):
        self._blob_pool[j].append(SharedArray(shape, np.float32))

    for i in range(PREFETCH_COUNT):
      self._free_queue.put(i)
    self._prefetch_process = BlobFetcher(self._blob_gen, self._blob_pool,
        self._free_queue, self._full_queue)
    self._prefetch_process.start()
    # Terminate the child process when the parent exists
    atexit.register(self._cleanup)

  def forward(self, bottom, top):
    """Get blobs and copy them into this layer's top blob vector."""
    pool_ind = self._full_queue.get()

    for i, s_blob in enumerate(self._blob_pool[pool_ind]):
      blob = s_blob.get_value(copy = True)
      # Reshape net's input blobs
      top[i].reshape(*(blob.shape))
      # Copy data into net's input blobs
      top[i].data[...] = blob

    self._free_queue.put(pool_ind)

  def backward(self, top, propagate_down, bottom):
    """This layer does not propagate gradients."""
    pass

  def reshape(self, bottom, top):
    """Reshaping happens during the call to forward."""
    pass

class BlobFetcher(Process):
  """Class for prefetching blobs in a separate process."""

  def __init__(self, generator, blob_pool, free_queue, full_queue):
    super(BlobFetcher, self).__init__()
    self._generator = generator
    self._blob_pool = blob_pool
    self._free_queue = free_queue
    self._full_queue = full_queue

  def run(self):
    # Pass SIGINT to the parent process
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    print 'BlobFetcher started'
    while True:
      blobs = self._generator.get_blobs()
      pool_ind = self._free_queue.get()
      for i, s_blob in enumerate(self._blob_pool[pool_ind]):
        s_blob.set_value(blobs[i])
      self._full_queue.put(pool_ind)      

if __name__ == '__main__':
  import tempfile

  with tempfile.NamedTemporaryFile(mode = 'w+') as f:
    f.write("""name: 'SimpleBlobGenNet'
        layer {
          type: 'Python'
          name: 'layer'
          top: 'data'
          top: 'label'
          python_param {
            module: 'prefetch_data_layer'
            layer: 'PrefetchDataLayer'
            param_str: 'flow_blob_generator.yaml'
          }
        }""")
    f.flush()
    net = caffe.Net(f.name, caffe.TRAIN)

  for i in range(2):  
    net.forward()
