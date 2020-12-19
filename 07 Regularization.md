# 07: Regularization

 ### The Problem of  Overfitting

![image-20201219184449228](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\13.png)

* 집값 예측 문제에 대해서 3개의 가설함수가 주어져 있다. 순서대로 가설함수 A,B,C라고 하겠다
* A = 데이터가 적합하게 맞지 않기 때문에 underfit or high bias 상태라고 말한다.
* B = underfit 과 overfit 중간 정도의 상태
* C = 데이터에 지나치게 맞춰 가설함수를 만들어 overfit or High varience 라고 말한다.

즉, Overfitting(과적합) 은 많은 feature들이 존재할 때 가설이 학습용 데이터에만 잘 맞춰져 있고 이 데이터에 대해서는 Cost function이 

0에 가까운 값으로 나오지만 학습 데이터 외의 새로운 데이터에 대해서는 제대로 예측하지 못하게 되는 문제가 발생한다

### Addressing overfitting

현재 과적합인지 아닌지를 알아내는 방법은 이러하다

* Training Set이 너무 적지 않은지 검사
* 가설함수 그래프를 그려서 확인

 과적합 문제를 해결하기 위한 방법은 이러하다

* feature 개수 줄이기
* ==Regularization==

feature들 중에서 중복되거나 예측하는데 도움이 안되는 요소들이 존재 할수 있다. 중복이 되는 경우는 하나의 feature로 합치면 되고

도움이 안되는 요소는 판단하여 제거해 줄 수 있다. 

두 번째 해결방법으로는 Regularization(정규화) 방법이 있는데 모든 feature들을 남긴채 각각의 특성이 갖는 영향 규모를 줄이는 것이다

정규화 방법은 모델이 너무 복잡해지는 것을 방지하여 임의로 제약을 거는 것을 의미한다.

### Cost Function 

![image-20201219190748020](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\14.png)

위의 두 개의 그래프를 보아 왼쪽에 그래프에 비해 오른쪽의 그래프는 과적합이 되어 있는 것을 한눈에 알 수 있다

여기서 $\theta$<sub>3</sub> , $\theta$<sub>4</sub> 이 0에 근접하게 한다면 분홍색 과 같은 그래프 개형이 될 것이고 이것이 우리가 원하는 feature는  제거하지 않고 파라미터가

갖는 영향 규모를 줄이는 ==정규화 방법== 이 된다







