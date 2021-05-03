from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
from collections import defaultdict
from ..generic_dataset import GenericDataset
import cv2
import random
import math

class MOT(GenericDataset):
  num_categories = 1
  default_resolution = [544, 960]
  class_name = ['']
  max_objs = 256
  cat_ids = {1: 1, -1: -1}
  def __init__(self, opt, split):
    self.dataset_version = opt.dataset_version
    self.not_run_eval_motchallenge = opt.not_run_eval_motchallenge
    self.year = int(self.dataset_version[:2])
    print('Using MOT {} {}'.format(self.year, self.dataset_version))
    data_dir = os.path.join(opt.data_dir, 'mot{}'.format(self.year))

    if opt.dataset_version in ['17trainval', '17test']:
      ann_file = '{}.json'.format('train' if split == 'train' else \
        'test')
    elif opt.dataset_version == '17halftrain':
      ann_file = '{}.json'.format('train_half')
    elif opt.dataset_version == '17halfval':
      ann_file = '{}.json'.format('val_half')
    img_dir = os.path.join(data_dir, '{}'.format(
      'test' if ('test' in self.dataset_version and self.dataset_version!='17test') else 'train'))
    self.img_dir=img_dir
    print('ann_file', ann_file)
    ann_path = os.path.join(data_dir, 'annotations', ann_file)

    self.images = None
    # load image list and coco
    super(MOT, self).__init__(opt, split, ann_path, img_dir)

    self.num_samples = len(self.images)
    print('Loaded MOT {} {} {} samples'.format(
      self.dataset_version, split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    results_dir = os.path.join(save_dir, 'results_mot{}'.format(self.dataset_version))
    if not os.path.exists(results_dir):
      os.mkdir(results_dir)

    video_size = []
    for video in self.coco.dataset['videos']:
      video_id = video['id']
      file_name = video['file_name']
      #print(file_name)
      out_path = os.path.join(results_dir, '{}.txt'.format(file_name))
      f = open(out_path, 'w')
      images = self.video_to_images[video_id]
      tracks = defaultdict(list)


      color_dict = dict()

      img_path = os.path.join(self.img_dir, file_name, 'img1', '%06d.jpg' % 1)
      img_now = cv2.imread(img_path)
      for image_info in images:
        if not (image_info['id'] in results):
          continue
        result = results[image_info['id']]

        frame_id = image_info['frame_id']
        path = os.path.join(save_dir, 'imgs_results_mot{}'.format(self.dataset_version))
        imgs_save_dir = os.path.join(save_dir, 'imgs_results_mot{}'.format(self.dataset_version), file_name)
        if not os.path.exists(path):
          os.mkdir(path)
        if not os.path.exists(imgs_save_dir):
          os.mkdir(imgs_save_dir)

        num_now = 0
        num_next = 0
        if not (image_info['next_image_id']==-1):
          result_next = results[image_info['id']+1]
          img_path_next = os.path.join(self.img_dir, file_name, 'img1', '%06d.jpg' % (frame_id + 1))
          img_next = cv2.imread(img_path_next)
          position_now_dict = dict()
          for item in result:
            if not ('tracking_id' in item):
              item['tracking_id'] = np.random.randint(100000)
            if item['active'] == 0:
              continue
            num_now += 1
            tracking_id = item['tracking_id']
            #print(color_dict.keys())
            if not (tracking_id in color_dict.keys()):
              r = random.randint(0, 255)
              g = random.randint(0, 255)
              b = random.randint(0, 255)
              color_dict[tracking_id] = (b,g,r)

            bbox = item['bbox']
            bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]

            if not (tracking_id in position_now_dict.keys()):
              position_now_dict[tracking_id] = [(bbox[0]+bbox[2])/2 , (bbox[1]+bbox[3])/2]
              #position_now_dict[tracking_id] = [position_now_dict[tracking_id][0], position_now_dict[tracking_id][1]]

            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            cv2.rectangle(img_now, (x1,y1), (x2, y2), color_dict[tracking_id] , 2)
            cv2.putText(img_now, 'Id: %s'%tracking_id, (x2+10,y1 ), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_dict[tracking_id], 2)
            cv2.putText(img_now, 'p: (%.1f,' % position_now_dict[tracking_id][0] + '%.1f)' % position_now_dict[tracking_id][1], (x2+10, y1 + 20),cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_dict[tracking_id], 2)

          for item in result_next:
            if not ('tracking_id' in item):
              item['tracking_id'] = np.random.randint(100000)
            if item['active'] == 0:
              continue
            num_next += 1
            tracking_id = item['tracking_id']
            #print(color_dict.keys())
            if not (tracking_id in color_dict.keys()):
              r = random.randint(0, 255)
              g = random.randint(0, 255)
              b = random.randint(0, 255)
              color_dict[tracking_id]=(b,g,r)
            bbox = item['bbox']
            bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])

            position_next = [(bbox[0]+bbox[2])/2 , (bbox[1]+bbox[3])/2]

            cv2.rectangle(img_next, (x1,y1), (x2, y2), color_dict[tracking_id] , 2)
            cv2.putText(img_next, 'Id: %s'%tracking_id, (x2+10,y1 ), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_dict[tracking_id], 2)
            cv2.putText(img_next, 'p: (%.1f,' % position_next[0]+'%.1f)'%position_next[1], (x2+10, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_dict[tracking_id], 2)
            if tracking_id in position_now_dict:
              v = math.sqrt((position_next[0] - position_now_dict[tracking_id][0]) ** 2 + (
                      position_next[1] - position_now_dict[tracking_id][1]) ** 2)
              cv2.putText(img_next, 'v: %.2f' % v, (x2+10, y1+40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_dict[tracking_id], 2)


          cv2.putText(img_now, 'number: %s' % num_now, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(255,255,255),2)
          cv2.putText(img_next, 'number: %s' % num_next, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255),2)
          imgs = np.hstack([img_now, img_next])

          video_size = (imgs.shape[0], imgs.shape[1])
          cv2.namedWindow(file_name, cv2.WINDOW_FREERATIO)
          #cv2.setWindowProperty(file_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
          cv2.imshow(file_name, imgs)
          keyvalue = cv2.waitKey(1)
          if keyvalue & 0xFF == ord(' '):  # 按空格键时，暂停在当前帧
            cv2.waitKey(0)
          if keyvalue == 27:  # 按esc键时，退出程序
            exit(0)
          img_filename = os.path.join(imgs_save_dir, '%06d.jpg' %frame_id)
          cv2.imwrite(img_filename, imgs)

        img_now=img_next

        for item in result:
          if not ('tracking_id' in item):
            item['tracking_id'] = np.random.randint(100000)
          if item['active'] == 0:
            continue
          tracking_id = item['tracking_id']
          bbox = item['bbox']
          bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
          tracks[tracking_id].append([frame_id] + bbox)

      #cv2.destroyAllWindows()


      filelist = os.listdir(imgs_save_dir)

      fps = 3  # 视频每秒多少帧

      fourcc = cv2.VideoWriter.fourcc('M', 'J', 'P', 'G')
      size = (video_size[0], video_size[1])  # 需要转为视频的图片的尺寸
      size = (640,360)
      # 可以使用cv2.resize()进行修改

      video = cv2.VideoWriter("%s.avi"%file_name, fourcc , fps, size)
      print("%s.mp4" % file_name)
      # 视频保存在当前目录下

      for item in sorted(filelist):
        if item.endswith('.jpg'):
          # 找到路径中所有后缀名为.png的文件，可以更换为.jpg或其它
          item = os.path.join(imgs_save_dir , item)
          img = cv2.imread(item)
          video.write(img)

      video.release()
      print('end')
      cv2.destroyAllWindows()


      rename_track_id = 0
      for track_id in sorted(tracks):
        rename_track_id += 1
        for t in tracks[track_id]:
          f.write('{},{},{:.2f},{:.2f},{:.2f},{:.2f},-1,-1,-1,-1\n'.format(
            t[0], rename_track_id, t[1], t[2], t[3]-t[1], t[4]-t[2]))
      f.close()
  
  def run_eval(self, results, save_dir):
    self.save_results(results, save_dir)
    gt_type_str = '{}'.format(
                '_train_half' if '17halftrain' in self.opt.dataset_version \
                else '_val_half' if '17halfval' in self.opt.dataset_version \
                else '')
    gt_type_str = '_val_half' if self.year in [16, 19] else gt_type_str
    gt_type_str = '--gt_type {}'.format(gt_type_str) if gt_type_str != '' else \
      ''

    if not self.not_run_eval_motchallenge:
      os.system('python tools/eval_motchallenge.py ' + \
               '../data/mot{}/{}/ '.format(self.year, 'train') + \
                '{}/results_mot{}/ '.format(save_dir, self.dataset_version) + \
                gt_type_str + ' --eval_official')
