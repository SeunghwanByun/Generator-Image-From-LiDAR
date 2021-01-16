# Generator-Image-From-LiDAR
Camera view image generation from sparse images projecting LiDAR point clouds onto the image plane.


# Explain codes

ImageGenerator.py 코드는 말그대로 이미지 생성 모델이다. 보통 자율주행자동차에서 사용하는 라이다 센서의 raw data를 활용하여 이미지 처리에 사용하는 경우는 3D 라이다 데이터를 2D 이미지 평면으로 투영을 시키어서 사용하게 된다.

2D 이미지 평면으로 투영된 라이다 데이타는 여전히 조밀한 값을 가지고 있기 때문에 밀도높은 정보를 필요로 하는 cnn 모델에서는 그 효과가 미미하다.

따라서 많은 방면으로 사람들은 2D 이미지 평면으로 투영된 라이다 데이터를 이미지 전처리를 통하여서 밀도를 높게 만들어준다. 그 방법론은 여러 가지가 존재하는데, 공통점은 딥러닝을 사용하지 않는다는 점이다.

그러다 이 [논문](https://www.researchgate.net/publication/337080321_Asymmetric_Encoder-Decoder_Structured_FCN_Based_LiDAR_to_Color_Image_Generation)을 발견하게 되었는데 라이다 데이터를 가지고 충분히 이미지를 복원해낼 수 있다는 점이다.

이 알고리즘은 희소한 라이다 데이터를 밀도높은 이미지로 생성해주는 멋진 알고리즘이지만, 이미지를 생성하는 과정에서 과연 라이다 자체가 가지고 있는 공간적 정보들이 그대로 남아있을까가 의문이다.

어찌됐든, 이 논문은 FCN(Fully Convolutional Network)로 Semantic Segmentation을 하는 것에서부터 아이디어를 고안해냈는데, FCN을 포함한 대부분의 이런 모델들은 Symmetric 하다는 것이다. 그러나 이 논문에서는 라이다 데이터의 feature를 extract 하는 부분(encoder)이 더 중요하다고 보고 비대칭적으로 encoder 부분을 더 깊게 설정해서 진행을 해서 좋은 성과를 얻었다.

이 논문에 착안하여 당시 구현하고 있던 Deeplab V3+ 알고리즘으로 ImageGenerator 네트워크를 구현했다. DeepLab V3+도 비대칭적인 구조를 가지고 있기 때문에 좋은 효과를 낼 것이라는 생각을 했었다.

학습에 사용된 파라미터들은 논문에 써있는 그대로 진행하였다.

결과적으로 생성된 모델은 좋은 성능을 내지는 못했는데, 원인을 파악해보자면 논문에서는 라이다가 감지하지 못하는 영역, 즉 이미지의 상단부분은 제거하고 생성했지만 내가 구현할 때는 이미지의 전체 부분을 입력으로 넣었다. 그리고 학습 데이터 양도 논문에서는 2천장 넘게 사용하였지만 실제 내가 사용한 이미지는 data augmentation을 진행해도 6백장 남짓이었다.

다음에는 파악한 원인을 가지고 좋은 성능을 내볼 수 있음 내보겠다.

그리고 학습시킨 모델을 freezing 하여 frozen.pb로 저장하고, frozen.pb 파일을 다시 load 해오는 작업을 꽤 오래 헤맸는데, 여기 [블로그](https://blog.metaflow.fr/tensorflow-saving-restoring-and-mixing-multiple-models-c4c94d5d7125)를 참조해서 잘 해결할 수 있었다.

![Generated Image](https://user-images.githubusercontent.com/49049277/104814874-6c282700-5854-11eb-8fe3-b4f7b24539e5.png)
