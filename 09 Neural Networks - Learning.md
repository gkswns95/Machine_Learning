# 09: Neural Networks - Learning

## **Neural network cost function**

![image-20201222133536365](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\28.png)

* 분류 문제에 신경망을 적용하는 것에 대해 공부하기 위해 위의 그림처럼 신경망이 주어졌을 때 필요한 용어들을 정의해 보겠다

* L = 신경망의 layer의 개수이다 

     위의 모델에서 L=4 이다

* s<sub>l</sub> = 신경망의 뉴런의 개수를 표시한다. (bias unit은 세지 않는다)

  위의 모델에서 s<sub>1</sub> =3 , s<sub>2</sub> = 5, s<sub>3</sub> =5, s<sub>4</sub> = s<sub>l</sub> = 4 각 층의 뉴런의 개수이다

  ![image-20201222134014272](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\29.png)

  이제 두 가지 유형의 분류 문제를 고려하자

  * 첫 번째로 Binary classification 문제로 label 은 y=0 or y=1 로 Output unit 1개로 구성되어 있다

  * 두 번째로 Multi-class classification 문제로 k개의 별개의 클래스가 있는 다중 클래스 분류 문제이다. 

      y는 k개의 label로 이루어져 있고 Output unit 도 k개로 이루어져 있다. (s<sub>l</sub> = k)

    ![image-20201222135900492](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\30.png)

    * Logistic regression에서의 정규화 용어가 추가된 cost function의 형태는 다음과 같다. 우리는 인공신경망의 Cost function으로 

      이 logistic regression의 cost function을 일반화 한 것을 쓸 것이다

      ![image-20201222140128019](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\31.png)

      * Neural net work 에서 k개의 class 에 대한 출력을 각각 표시하기 위해 (h<sub>$\theta$</sub>(x))<sub>i</sub>  로 i의 첨자로 각각 k개의 output을 표현한다

  ## Backpropagation Algorithm

  * Cost function을 최소화 하는 알고리즘 = Backpropagation Algorithmn(역전파) 에 대해서 알아보겠다
  * 우선 $\delta$<sub>j</sub><sup>l</sup> - this is the error of node j in layer l 즉, layer l에 노드 j의 error 값이라고 정의하겠다
  * $\delta$<sup>4</sup> = a<sup>4</sup> - y 
  * $\delta$<sup>(3)</sup> = ($\theta$<sup>(3)</sup>)<sup>T</sup>$\delta$<sup>(4)</sup>.*g<sup>/</sup>(z<sup>(3)</sup>)

  ![image-20201222211747682](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\32.png)

  * Backpropagation은 각각 node의 error 값을 위의 계산처럼 $\delta$ 로 구한다. 그 후에 각각 error가 미치는 영향을 업데이트 하며 가중치를 줄여

     학습하는 방법을 말한다.

## **Gradient checking**

역전파는 종종 버그가 생기게 되는데, 이러한 버그들은 경사하강법으로 적용해 보았을 때 cost가 정확히 감소하는 것 처럼 보일 수도 있기 때문에

역전파가 잘 진행되고 있는지 확인하는 Gradient Checking 방법이 필요하다.

![image-20201222212939939](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\33.png)

* 이러한 J($\theta$) 에 대한 그래프가 주어졌을 때, $\theta$ 값의 접선의 기울기를 그려보면 위에처럼 파란색 직선으로 그릴 수 있다.

  이러한 Partial Derivative Term 을 Checking 하기위해 $\theta-\epsilon$ , $\theta + \epsilon$  의 두 점 사이의 기울기를 구해서 체킹하고자 하는 편미분 항과

  유사한지 검사하는 방식이 바로 Gradient Checking이다

## **Random Initialization**

이제까지 Forward Propagation, Backward Propagation 을 이용하여 $\theta$ 를 학습하는 방법에 대해서 알아보았다.

이제 초반 $\theta$ 의 값을 정하는 방법에 대해서 알아보자

* Zero Initialization(0초기화) : 이전에 Logistic Regression을 학습할 땐 $\theta$ 의 값을 0으로 두고 학습하여도 문제없이 학습하였다.

   마찬가지로 NNs에 같은 방식을 적용해보자.

  ![image-20201222213912191](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\34.png)

모든 $\theta$ 에 대해 0으로 초기화 하였다고 생각하자

그림과 같이 나가는 weight가 동일하게 계산되기 때문에 $\delta$ 값도 동일하게 된다. 그럼 BP 진행후의 업데이트 되는 $\theta$ 들의 값도

동일하게 되게 된다. 이러한 문제를 Symmetry Problem 이라 부르며, 이러한 문제 때문에 Random initailization 방법을 사용한다.

## Random initialization

 각각의 $\theta$ 값을 [−ϵ,ϵ] 범위 안에 랜덤한 값에 대해서 초기화 하는 방식을 사용하며 Symmetry breaking 이라고도 한다.

* Initialize each Θ(l)ij to a random value in [−ϵ,ϵ] 

