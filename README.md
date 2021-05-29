# HYU-2021-Deep-Learning

한양대학교, 2021년도 봄학기 딥러닝 수업을 위한 레퍼지터리 입니다  
과제별 설명은 하위 디렉터리에 있습니다.

Binary Classification 문제에 대해 라이브러리를 사용하지 않고, 직접 어레이로 노드를 구현하여 Forward-Propagtion 과 Backward-Propagation 에 의해  
parameter 들이 update 될 수 있도록 구현하는 과제[1] 로 부터 시작하였습니다.

과제[1] 에서는 hidden-layer 없이 2-dimensional input vector `x` 로부터 1-dimensional output `y`를 잘 예측해내는 모델을

- cross-entrophy (loss function)
- sigmoid (activation)

을 사용하여 구현하였고, vectorize 가 된 버전과 되지 않은 버전에 대해서 실험하였습니다. (속도)

과제[2] 에서는 hidden-layer (1개) 를 추가해 shallow-network 를 구성하였고, 해당 hidden-layer 의 노드의 개수에 따른 실험을 하였습니다. (속도, 정확도)

과제[3] 에서는 직접 구현햇던 과제[2] 의 코드를 Tensorflow Tool 을 이용하여 작성해보는 것으로 시작하였고

- loss-function (Cross-Entrophy, MSE)
- optimizer (SGD, RMSProp, Adam)
- batch size (4, 32(default), 128)

에 따른 결과를 확인해보았습니다. (속도 정확도)

## 과제 목록

### [1] Binary Clasficification

### [2] shallow network

### [3] Tensorflow
