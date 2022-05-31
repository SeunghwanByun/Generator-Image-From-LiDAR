# Generator-Image-From-LiDAR
Camera view image generation from sparse images projecting LiDAR point clouds onto the image plane.


# Explain codes

The ImageGenerator.py code is literally an image generation model. In the case of using the raw data of the lidar sensor used in autonomous vehicles for image processing, the 3D lidar data is projected onto the 2D image plane and used.

Since the lidar data projected onto the 2D image plane still have dense values, the effect is insignificant in cnn models that require dense information.

Therefore, in many ways, people make the LiDAR data projected on the 2D image plane high in density through image preprocessing. Several methods exist, but the common point is that deep learning is not used.

Then I discovered this [paper](https://www.researchgate.net/publication/337080321_Asymmetric_Encoder-Decoder_Structured_FCN_Based_LiDAR_to_Color_Image_Generation), and it is that the image can be sufficiently restored with LiDAR data.

Although this algorithm is a wonderful algorithm that generates sparse LIDAR data as a dense image, it is questionable whether the spatial information of LIDAR itself remains intact in the process of image creation.

Anyway, this paper devised the idea from doing Semantic Segmentation with FCN (Fully Convolutional Network), and most of these models including FCN are symmetric. However, in this paper, we found that the part that extracts the features of the lidar data (encoder) is more important, so we asymmetrically set the encoder part deeper and get good results.

Based on this paper, the ImageGenerator network was implemented with the Deeplab V3+ algorithm that was being implemented at the time. DeepLab V3+ also has an asymmetric structure, so I thought that it would have a good effect.

The parameters used for learning proceeded as written in the paper.

As a result, the generated model did not perform well. To understand the cause, in the paper, the area that LiDAR could not detect, that is, the upper part of the image was removed and generated, but when I implemented it, the entire part of the image was put as input. And although more than 2,000 images were used in the training data transfer thesis, the actual images I used were only about 600 even after data augmentation.

Next, I will show you that good performance can be achieved with the identified cause.

And I was at a loss for a long time to freeze the trained model, save it as frozen.pb, and reload the frozen.pb file, here [blog](https://blog.metaflow.fr/tensorflow-saving-restoring-and-mixing-multiple-models-c4c94d5d7125) and was able to solve it well.

![Generated Image](https://user-images.githubusercontent.com/49049277/104814874-6c282700-5854-11eb-8fe3-b4f7b24539e5.png)
