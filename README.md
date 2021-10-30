# 简介
[image_morphing](https://github.com/cuteyyt/image_morphing.git)

浙江大学2021-2022学年秋学期计算机动画课程project。

实现[Feature-based image metamorphosis](https://dl.acm.org/doi/10.1145/133994.134003)

参考[bchao1/image-morphing](https://github.com/bchao1/image-morphing)

# 文件组织
文件组织如下所示：
```bash
project
├── ckpt # dlib预训练好的人脸特征提取模型
│   └── shape_predictor_68_face_landmarks.dat
├── imgs
│   ├── debug # 三幅图像，程序调试时使用
│   └── test # 20幅图像，最终动画生成所用
├── report # 报告markdown使用素材（未上传至git）
├── results # 生成结果
│   ├── inter # 中间结果（原图像上绘制人脸检测框和特征线）
│   └── morphing.mp4 # 生成动画
├── morphing.py # 源代码
├── README.md
├── 尹幽潭_3180105171_报告.md # 报告源代码（未上传至git）
├── 尹幽潭_3180105171_报告.pdf # 报告pdf（未上传至git）
└── materials # 暂存动画帧，程序运行完毕后会自动删除
```

# 使用方法
所有命令行参数都有其默认值，具体可参考代码。
```bash
cd project
python morphing.py
  --input_path '包含多张图像的文件夹，该参数有效代表文件夹中所有图像morphing为一个视频'
  --src_img_path '源图像路径' （如果input_path无效，这两个参数指定两幅图像间的morphing）
  --dst_img_path '目标图像路径'（如果input_path无效，这两个参数指定两幅图像间的morphing）
  --output_path '输出文件路径，必须是MP4格式'
  --predictor_path '人脸特征提取模型路径'
  
  --img_size 256 '输出视频分辨率，256x256'
  --a 1（论文中参数a）
  --b 2（论文中参数b）
  --p 0.5（论文中参数p）
  --eps 1e-8（预防乘或除0导致数值膨胀）
  
  --frames 30（生成视频的帧率）
  --frames_per_pair 1（从源图像到目标图像的帧数，代表morphing的步长为frames*frames_per_pair）
```