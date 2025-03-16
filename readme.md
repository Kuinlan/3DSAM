# 关于如何引入 Depth Anything V2 方法的介绍

## 1. 使用 Depth Anything V2 产生相对深度图

Depth Anything V2 源码已位于 src/da/depth_anything_v2，预训练权重下载地址：[github](https://github.com/DepthAnything/Depth-Anything-V2)

在 [lightning_3dsam.py](./src/lightning/lightning_3dsam.py "跳转至文件") 文件中：

12 行使用 import 导入 Depth Anything V2

46 行进行初始化，使用 Small 版本的 Depth Anything V2，预训练参数大小为 95MB

48 行加载预训练权重，下载权重后，对路径进行替换

在 [scannet.py](./src/datasets/scannet.py) 文件中：

113,114 行获得图像的灰度图和彩色图（彩色图作为 Depth Anything V2 的输入）

117,118 行对彩色图进行预处理

144 行将彩色图数据储存至数据字典 data 中

在 [lightning_3dsam.py](./src/lightning/lightning_3dsam.py "跳转至文件") 文件中：

169 行函数 _update_relative_depth() 首先将彩色图数据从 data 中取出，大小为 $[480, 640]$

在 175,176 行，将数据输入至 Depth Anything V2 进行推理，获得深度图，可通过 downsample 参数控制输出深度图的大小

在 177,178 行使用归一化将深度图转为相对深度图

在 180 行将相对深度图存入数据字典 data

## 2. 对相对深度信息进行特征编码

在 [depthnet.py](./src/threedsam/threedsam_modules/depthnet.py "跳转至文件") 文件中：

在 363,364 行将相对深度图从数据字典中提取出来

在 371,372 行对相对深度进行编码，具体过程可以参考论文

## 3. 相对深度特征与图像特征融合

在 383 行使用 self.se 模块进行特征融合，方式比较简单，就是相乘操作，这部分可以试试别的融合方法