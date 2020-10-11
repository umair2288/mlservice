import os
import sys

# import skimage.draw
# import skimage
import numpy as np
# import matplotlib.pyplot as plt
#import mrcnn.model as modellib
from .mrcnn import model as modellib
# from mrcnn.model import log
from .config import balloon


class InferenceConfig(balloon.BalloonConfig):
  # Set batch size to 1 since we'll be running inference on
  # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
  GPU_COUNT = 1
  IMAGES_PER_GPU = 1

class LeavesSegmentation():
    def __init__(self, config, path="model/mask_rcnn_leave_0012.h5", logdir= './logs'):
      self.config = InferenceConfig()
      #self.config.display()
      # load model 
      self.model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=logdir)
      self.model.load_weights(path, by_name=True)
      # define display color
      # from your definition : {1: healthy , 2: unhealthy}
      self.color = [
          (155, 155, 0),
          (255,0,0)
      ]
    def CalculatePercentage(self, img, data):
      h, w = img.shape[:2]
      # mask to display result
      tmask = np.zeros([h, w, 3])
      # masks
      masks = data['masks']
      # ids label
      ids = data['class_ids']
      # sum of each label
      sum= [0,0]
      for i, _id in enumerate(ids):
        sum[_id -1] += np.sum(masks[:,:, i])
        tmask[masks[:, :, i] == True] = self.color[_id -1]
      # percentage
      healthy = sum[0] / (sum[0] + sum[1])
 
      return healthy, tmask

    def predict(self, img):
      res = self.model.detect([img], verbose=1)[0]
      return self.CalculatePercentage(img, res)