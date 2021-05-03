# 2D-detection-and-recognition
  该项目包含我在做有关二位检测和识别的任务时跑通的一些代码，三个任务分别是多目标人的追踪（MOT）、车牌的识别以及人脸识别，对应的项目名称为CenterTrack、HyperLPR-master以及Arcface，前两个为github上的开源项目，后一个为虹软的开源人脸识别SDK。

原项目地址如下：
https://github.com/xingyizhou/CenterTrack  
https://github.com/LINGYUWEN/HyperLPR-master  
https://ai.arcsoft.com.cn/third/ldpage.html?utm_source=baidu6&utm_campaign=hr39&utm_medium=cpc&bd_vid=6859621431400586012

##CenterTrack
该项目实现了多目标人的追踪，但在人物在画面中被遮挡后重新出现会给定新的ID。

配置环境：注意DCNv2和pytorch的适配，最终我使用的pytorch版本为1.3.1,可做参考，其余配置依据原项目中CenterTrack/readme/INSTALL.md 中按步骤配置即可。
data为空文件夹，数据集的配置参照原项目中CenterTrack/readme/DATA.md  

nohup 训练代码参考：  
nohup  python main.py tracking --exp_id mot17_fulltrain_sc --dataset mot --dataset_version 17trainval --pre_hm --ltrb_amodal --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 0 > mot17_fulltrain_sc_train.log 

nohup 测试代码参考：  
nohup python test.py tracking --exp_id mot17_fulltrain_sc --dataset mot --dataset_version 17test --pre_hm --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5 --load_model ../exp/tracking/mot17_fulltrain_sc/model_last.pth > mot17_fulltrain_sc_test.log 2>&1  &

我在跑代码时加入了可视化程序，可以实现实时对相邻两帧中的人进行标注，不同颜色显示ID、位置、速度:
```java
python test.py tracking --exp_id mot17_fulltrain_sc --dataset mot --dataset_version 17test --pre_hm --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5 --resume --load_results ../exp/tracking/mot17_fulltrain_sc/save_results_mot17test.json --use_loaded_results --not_run_eval_motchallenge
```
```--not_run_eval_motchallenge 表示不进行结果的评估（节约跑代码的时间）  
##HyperLPR-master  


##Arcface  
