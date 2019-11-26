GIthub使用指北:

**1.想将项目拷贝到自己帐号下就fork一下.**

**2.持续关注项目更新就star一下**

**3.watch是设置接收邮件提醒的.**

# Darknet19-pytorch

最近看到很多人在用yolo框架,但对于yolo的darknet特征提取网络却知之甚少.(darknet其实是一种纯C深度学习框架,darknet19是一种卷积分类网络.)

在目标检测任务以及实例分割任务中,vgg是较为常用的特征提取网络.

在imagenet上,darknet可以以更小的内存占用以及更快的计算效率,达到与不逊于VGG16的精确度.

---

本项目使用pytorch实现darknet19模型,并提供原始的预训练模型供大家使用.

采用了与torchvison中vgg相似的实现方法.

( The implementation method is similar to that of VGG in torchvision. )

在使用时,与调用torchvision.models中模型相似.

(you can use it same as models of the torchvision)

welcome **star**, **fork**, **watch**.

---

The model structure:
![image](imgs/model.png)


| | top-1 | top-5 |
|:---:|:---:|:---:|
| | 72.9% | 91.2% |
| | ~76.5%~ | ~93.3%~ |
