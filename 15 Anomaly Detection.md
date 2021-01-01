# 15: Anomaly **Detection**

Anomaly Detection(이상 탐지) 는 정상적인 데이터집단이 있을 때, 새롭게 들어온 데이터가 정상적인 범주에 있는지 없는지 검사하는 방법이다.

![image-20201231225210083](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\110.png)

항공사 엔진 제조업체에 대한  x1(heat), x2(vibration) features들이 그래프 상에 plot되어 있다. 이 때 새로운 x값이 들어왔을 때, 데이터들의 집합의 평균에 가까이 위치하면 정상적인 데이터 그렇지 않으면 anaomaly 즉 이상데이터라고 할 수 있다.  

![image-20201231225536369](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\111.png)

Density estimation이라는 것은 모델 p(x) 즉 training data set에 대한 확률적 모델을 계산하여 test data가 들어왔을 때 확률을 계산하여 아래와 같이 분류할 수 있다.

* if p(x_test) < epsilon  :  flag anomaly(이상데이터)
* else if p(x_test) > epsilon  : 정상적인 데이터

이상탐지는 대략적으로 이러한 과정을 통해서 이루어진다. 본격적으로 이상탐지 알고리즘에 대해서 자세하게 배워보기 전에 이상탐지는 어떤 분야에 적용할 수 있는지 실제로 어떻게 사용되는지에 대해 간단히 예시를 살펴보고 시작하겠다.

![image-20201231230139591](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\112.png)

* Fraud detection(사기 탐지)
* 생산제품 QA
* 모니터링

이와 같은 예시들은 feature들의 정상적인 행동을 데이터화 시키고 이를 이용해 이상한 행동을 취하는 data를 탐지하는 것이 최종적인 목표이다.

## **The Gaussian distribution (optional)**

이상탐지에 대해서 공부하기 이전에 우리는 Gaussian distribution(정규분포) 에 대해서 알고 있어야 한다. 

![image-20201231230903950](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\113.png)

* x~N($\mu$,$\sigma^2$) : x가 정규분포를 따른다고 할 때 왼쪽의 식은, $\mu$(평균)  $\sigma^2$(분산) 을 갖는 정규분포를 따른다고 표현하는 것이다.

* 그에대한 확률은 위와같은 공식으로 구할 수 있다.
* 그래프 안쪽의 면적은 거의 1로 수렴한다.
* 평균은 그래프의 중심에 위치해 있고, 분산이 커질수록 그래프가 더 퍼지는 모양으로 만들어 질 것이고 분산이 작을 수록 더 뾰족한 모양으로 만들어 질 것이다. 표준편차, 분산에 따른 정규분포의 형태 변화는 아래와 같다.

![image-20201231231445812](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\114.png)

## **Anomaly detection algorithm**

이제 이 Gaussian distribution 을 이용해 anomaly detection에 적용하는 것에 대해 알아보겠다. data set x1...xm 은 각각 정규분포를 가질 수 있다.

![image-20201231231715501](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\115.png)

위의 그림처럼 x에 대해 정규분포를 구할 수 있는데, 정규분포에 필요한 파라미터 $\mu$, $\sigma^2$ 의 값은 위의 식으로 구할 수 있다.

이렇게 해서 구해진 각각의 x들의 확률, p(x; $\mu$,$\sigma^2$) 를 구한 후 데이터 전체의 확률 모델 p(x) 를 다음과 같은 공식으로 구할 수 있다.

![image-20201231232039131](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\116.png)

위의 공식에서 얻어진 p(x) 모델은 즉 각각의 feature x에 대한 확률을 구해 전부 곱해주어 만들어진 결과이다.

![image-20201231232218740](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\117.png)

전체적인 알고리즘의 형태는 위와 같은데 단계별로 분류해본다면

* 이상탐지를 할 대상 feature x를 선택한다
* 모든 features 들의 평균, 분산의 값을 위와 같은 공식을 통해 구한다
* 위에서 구한 값을 통해 p(x) 모델을 구한 후, 이상탐지를 위한 x를 적용하였을 때 의 값이 threshold 이상인지 이하인지 판단하여 이상탐지를 실시한다.

아래에서 실제 이상탐지에 대한 예시를 통해 정리해보겠다.

![image-20201231232559495](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\118.png)

* 예시에서는 threshold 를 0.02 로 잡아 test data의 확률이 threshold보다 크면 정상범주, 아니면 이상탐지에 걸린 모습을 보여주고 있다.

## Building an Anomaly Detection System

이제까지는 이상탐지 알고리즘을 알아보았고 이번에는 이상감지 시스템을 만들고 평가하는 법에 대해서 알아보겠다.

![image-20201231233719813](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\119.png)

이제부터 이상탐지 결과에 따라 data에 label에 따라 분류할 것이다.

* if data is anomaly data, label is y=1
* if data is not anomaly data, label is y=0

data set은 이전에와 같이 training data, cv data, test data  3가지로 분류하여 p(x) 를 학습하고 검증하는 방식을 사용할 것이다

예를 들어 Aircraft engines motivating example이 있다고 했을 때 적용하는 법을 알아보겠다

![image-20201231234242327](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\120.png)

10000개의 보통 엔진이 있고 20개의 결함이 있는 엔진이 있고, 이 데이터들을 각각 3개의 데이터로 나누어 볼 수 있다.

* Training set : 6000개의 보통 엔진
* CV : 2000개의 보통 엔진(y=0), 10개의 결함있는 엔진(y=1)
* Test: 2000개의 보통 엔진(y=0), 10개의 결함있는 엔진(y=1)

이렇게 나누어서 Training set으로 학습을 진행한 뒤 CV와 Test로 모델을 검증할 것이고 

![image-20201231234633863](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\121.png)

y조건에 따라 1 또는 0 의 값을 얻어 평가를 진행한다.

## **Anomaly detection vs. supervised learning**

상황에 따라 Anomaly detection 을 사용할 수도 있고 supervised learning을 사용할 수도 있다. 어떤 상황에 적절히 골라 사용할 수 있는지 비교해보겠다

![image-20210101001347594](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\122.png)

간단하게 정리하자면 한 집단에 비해 다른집단의 크기가 월등하게 작은 class에서는 Anomaly detection 방식을 사용하는 것이 좋고 적은수의 데이터 집합에서도 효율적이다. 반대로 충분히 많은 수의 pos/neg 데이터들이 있을 때 에는 supervised learning방식을 사용하는 것이 더욱 효과적이다.

아래에는 이상탐지와 지도학습에 적용하는 일반적인 예시를 보여주고 있다.

![image-20210101002056734](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\123.png)

## **Choosing features to use**

지금까지는 모든 데이터들이 정규분포를 따른다고 생각하고 문제를 해결했다. 만약 정규분포를 따르지 않을 때는 어떻게 해야 하는지 보겠다.

![image-20210101003008181](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\124.png)

정규분포를 따르지 않는 x데이터에 log함수를 적용하거나 제곱근 함수를 적용하거나 등의 방식을 사용하면 정규분포형태로 간단하게 만들 수가 있다.

## Error analysis for anomaly detection

![image-20210101003345993](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\125.png)

위의 그림에서 보면 p(x) 값이 크게 작은 값이 아니어서 normal data 처럼 보일 때가 있다. 하지만 실제로 오른쪽 그림에서 보았을 때는 이상데이터 인 것을 확인할 수 가 있다. 이러한 문제가 발생했을 때는 새로운 feature를 추가하여 데이터를 분리하는 방법을 사용하면 된다 

## Multivariate Gaussian Distribution

![image-20210101004126122](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\126.png)

위와 같이 각각의 feature x1, x2 에서는 비정상 데이터들의 p의 값이 각각 정상 범주안에 들어와 있지만, 실제로 두 개의 feature로 plotting 했을 때는

비정상 데이터란 것을 알 수가 있다. 이렇게 되는 이유는 왼쪽과 같이 분홍색 원처럼 확률이 분포되기 때문에 그렇다. 이 문제를 해결하기 위해 일반적 anomaly detection 모델은 새로운 feature를 추가하여 데이터의 분포를 분리시켜 anomaly를 찾는 방법을 사용했었다. 이번에 배울 모델은 Multivariate Gaussian Distribution은 이 문제를 해결하는 방식으로 수동이 아니라 자동으로 anomaly detection을 할 수 있는 방법이다.

![image-20210101171628596](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\127.png)

우의 공식이 바로 multivariate gaussian 분포이다. 이 모델을 사용하면 아래와 같이 확률 분포가 대각선 모양으로 데이터에 올바르게 분포가 가능하고 그 p(x)값의 크기에 따라 정상 / 비정상 데이터를 확인하면 된다.

![image-20210101171854791](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\128.png)



## **Original model vs. Multivariate Gaussian**

마지막으로 일반적모델과 Multivariate Gaussian 모델의 장단점과 각자 어느상황에 사용하면 좋은지에 대해 알아보고 마무리하겠다.

![image-20210101171929890](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\image-20210101171929890.png)

* Original model : 문제가 발생하면 수동으로 feature를 추가하여 문제를 해결했다. 대신에 이 모델은 시간적으로 효율이 좋고 적은 training data set에 대해서도 잘 작동하는 장점이 있다.
* Multivariate Gaussian : 수동으로 feature를 추가하지 않아도 된다. 하지만 행렬의 역행렬을 구해야 하는 연산이 있기 때문에 연산의 비용도 그만큼 증가하는 단점도 있다. 또한 m(training set number) > n(number of features) 의 조건을 만족하지 않으면 행렬의 역행렬이 존재하지 않을 수도 있어, 조건을 만족하지 않을 때에는 original model을 사용하는  것이 좋다.

 