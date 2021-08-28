# 火灾检测

## 通过以下代码进行训练或者测试:

 **TensorFlow 1.x / TFLearn 0.3.2 / OpenCV 4.x** :

```
$ git clone https://github.com/tobybreckon/fire-detection-cnn.git
$ cd fire-detection-cnn
$ sh ./download-models.sh
$ python firenet.py models/test.mp4
$ python inceptionVxOnFire.py -m 1 models/test.mp4
$ python superpixel-inceptionVxOnFire.py -m 1 models/test.mp4
```
 **TensorFlow 2.x** 

```
$ virtualenv -p python3 ~/venv/tf-1.1.5-gpu
$ source ~/venv/tf-1.1.5-gpu/bin/activate
$ pip install tensorflow-gpu==1.15
$ pip install tflearn
$ pip install opencv-contrib-python
....
$ python3 firenet.py models/test.mp4
