加入了efficient作为yolov3的backbone.

原github项目地址为[https://github.com/TencentYoutuResearch/ObjectDetection-OneStageDet/tree/master/yolo#evaluation](https://github.com/TencentYoutuResearch/ObjectDetection-OneStageDet/tree/master/yolo#evaluation), 这里添加了efficient, 安装扔按github上来.

使用: 
- 训练: 用Efficientnet时, 修改3个地方, `vedanet/models/_yolov3.py`里的inchannel_list为[160, 64, 24], self.backbone改为backbone.EfficientNet('efficientnet-b0')并把原先的注释掉, `vedanet/engine/_voc_train.py`在传入hyper_params在前加上一句`hyper_params.weights = None`这是不用预训练权重, 因为加了Efficientnet的预训练权重没有(如果有Efficientnet的预训练权重也能不用设为None,但仍需改加载权重的代码).运行命令为`python example/train.py Yolov3`
- 测试: 和训练一样, 只是运行命令为`python example/test.py Yolov3`. 会在result文件夹下得到20个txt,分别是每类的预测结果,再运行'python voc_eval.py`计算map, 注意修改该文件内的路径.会计算每类的ap和总的map




做了3组实验, 权重和log在outputs里, 测试结果在results里, 分别是预训练的darknet作为backbone, 随机初始化的darknet作backbone和随机初始化的Efficientnet-b0作backbone. 

都是3组baseline

1. 使用了darknet预训练权重的yolov3的map最高,0.745:      `results/darknet_nopretrain`
2. 然后是没使用darknet预训练的yolo3的map是0.55          `results/yolo3darknet`
3. 最后是没使用Efficientnet-b0的yolov3的map是0.41      `results/effi`

可能原因应该就是Efficientnet-b0的通道数太少了, 相比于darknet检测用的特征通道为512,256,128. 而Efficientnet-b0的特征通道分别为160,64,24
