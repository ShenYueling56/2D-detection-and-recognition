# 2D-detection-and-recognition
&emsp;&emsp;该项目包含我在做有关二维检测和识别的任务时跑通的一些代码，三个任务分别是多目标人的追踪（MOT）、车牌的识别以及人脸识别，对应的项目名称为CenterTrack、HyperLPR-master以及Arcface，前两个为github上的开源项目，后一个为虹软的开源人脸识别SDK。

原项目地址如下：<br>
https://github.com/xingyizhou/CenterTrack  
https://github.com/LINGYUWEN/HyperLPR-master  
https://ai.arcsoft.com.cn/third/ldpage.html?utm_source=baidu6&utm_campaign=hr39&utm_medium=cpc&bd_vid=6859621431400586012

  
  
## CenterTrack
&emsp;&emsp;该项目实现了多目标人的追踪，但在人物在画面中被遮挡后重新出现会给定新的ID。

* 配置环境：注意DCNv2和pytorch的适配，最终我使用的pytorch版本为1.3.1可做参考，其余配置依据原项目中`CenterTrack/readme/INSTALL.md`中按步骤配置即可。
* 数据集：data为空文件夹，数据集的配置参照原项目中CenterTrack/readme/DATA.md  

* nohup训练代码参考：  
```
nohup  python main.py tracking --exp_id mot17_fulltrain_sc --dataset mot --dataset_version 17trainval --pre_hm --ltrb_amodal --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 0 > mot17_fulltrain_sc_train.log 
```

* nohup测试代码参考：  
```
nohup python test.py tracking --exp_id mot17_fulltrain_sc --dataset mot --dataset_version 17test --pre_hm --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5 --load_model ../exp/tracking/mot17_fulltrain_sc/model_last.pth > mot17_fulltrain_sc_test.log 2>&1  &
```

* 修改：我在代码运行过程中加入了可视化程序，可以实现实时对相邻两帧中的人进行标注，不同颜色显示ID、位置、速度,实现代码增加在`CenterTrack/src/libdetector.py`中:
```
python test.py tracking --exp_id mot17_fulltrain_sc --dataset mot --dataset_version 17test --pre_hm --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5 --resume --load_results ../exp/tracking/mot17_fulltrain_sc/save_results_mot17test.json --use_loaded_results --not_run_eval_motchallenge
```
```--not_run_eval_motchallenge``` 表示不进行结果的评估（节约跑代码的时间）  
  
  
## HyperLPR-master  
&emsp;&emsp;HyperLPR-master是我找到的对车牌识别比较准确的项目，可以检测出一张图片中的所有车牌并进行识别，可根据需要选择一定可信度的识别结果
  
* 配置环境：主要是tensorflow的配置，参考版本为2.2.0，其余按`HyperLPR-master/README.md`即可。
* 运行代码：  
```
python demo.py
```
  
  
## Arcface 
&emsp;&emsp;开源SDK有C++和Java版本，我下载的为C++版本，利用该SDK和Opencv中的函数实现图片中的人脸检测、识别特征和匹配,并在代码中增添了可视化函数。

* 配置环境：不建议使用OpenCV4,我在运行时报错，建议使用OpenCV3.4.5;其余SDK的使用配置参考`ArcSoft_ArcFace_Linux_x64_V3.0/samplecode/ReadMe.txt`
* 运行：
```
cd ArcSoft_ArcFace_Linux_x64_V3.0/samplecode  
mkdir build  
cd build  
cmake ..  
make  
./arcsoft_face_engine_test  
```
* 更多有关SDK函数的信息可查阅`ArcSoft_ArcFace_Linux_x64_V3.0/docARCSOFT_ARC_FACE_DEVELOPER'S_GUIDE.pdf`
