# 06: Logistic Regression

 ## Classification

* 이메일 : 스팸문자인지 스팸문자가 아닌지 검출하는 문제

* 종양 : 양성종양인지 악성종양인지 판단하는 문제

  ![image-20201219151944378](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\1.png)  

* 이전에 사용하던 linear regression을 이욯하여 위와 같은 데이터 셋의 분류 문제를 풀어본다

   만약 h(x) >= 0.5, predict "y=1"

  ​		 h(x) < 0.5, predict "y=1" 이라고 한다면 위의 데이터 셋에는 정확히 동작하는 것으로 보인다.

![image-20201219153001682](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\2.png).

* 하지만 이렇게 데이터 셋이 추가되면 linear regression으로는 정확한 답을 예측하기가 쉽지 않기 때문에

   이런 분류 문제에 맞는 다른 가설함수를 사용해야 하는 것을 알 수있다

* 앞으로 배울 ==Logistic Regression== 은 0 과 1사이의 값을 가지는 가설함수를 사용할 것이다.

  ​														0 <= h<sub>$\theta$</sub>(x) <= 1

## Logistic Regression Model

* 가설 함수가 0 <= h<sub>$\theta$</sub>(x) <= 1 사이의 값을 가져야 한다.

* 원래 가설함수인 $\theta$<sup>T</sup>x 에 sigmoid 함수 g를 적용한 것이 Logistic Regression Model이다
  $$
  h(x) = g(\theta^Tx),\ g(z) = \frac 1 {1+e^{-z}},\ h(x) = \frac 1 {1+ e^{-\theta^Tx}}
  $$

* 

* 참고로 Sigmoid function = Logistic function 둘 다  같은 함수를 의미한다

![image-20201219154423145](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\3.png)

* Sigmoid 함수의 개형을 보면 z값의 수평축을 기준으로 z가 음의 무한대 방향으로 가면 g(z) = 0으로 수렴되고

  ​                                                                              z가 양의 무한대 방향으로 가면 g(z) =1 로 수렴되는 것을 알 수 있다 

## **Decision boundary**

* g(z) 그래프의 개형에 따라 z의 값이 양수이면 g(z) >= 0.5 의 값을 갖는다. 즉  $\theta$<sup>T</sup>x>=0 이면, h(x) >=0.5 라고 할 수 있다

  * predicts y=1 : $\theta$<sup>T</sup>x>=0

    predicts y=0 : $\theta$<sup>T</sup>x<=0

* h(x) = g($\theta$+$\theta$<sub>0</sub>x1+$\theta$<sub>1</sub>x2) 이고 $\theta$ 의 값이 각각 -3,1,1 이라 하자 

  앞에서 우리는 z>=0 일 때 y=1로 예측하고 z<=일 때 y=0로 예측한다고 배웠다

  즉 $\theta$<sup>T</sup>x = -3 + x1 + x2 의 값의 부호에 따라 예측값이 정해지는 것을 알 수 있다.

  <img src="C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\4.png" alt="image-20201219160953460" style="zoom:67%;" />

  위의 그림처럼 -3 + x1 + x2 의 값이 <0 이면 동그라미 영역에 속하는 것이고 >0이면 x영역에 속하는 직선이 만들어지는 것을

  알 수 있다. 즉 파라미터 $\theta$ 의 값에 의해 경계가 만들어 지고 이 경계는 class를 분류하는 경계 즉  ==Decision Boundary== 라고 한다.

### **Non-linear decision boundaries**

* 바로 위에서는 decision boundary가 직선형태로 나오는 데이터 집합들의 예시를 들었다

  만약 아래와 같이 데이터 셋들의 집합이 주어진다면 어떻게 해야할까?

  ![image-20201219162059905](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\5.png)

* 이렇게 주어진 경우에는 직선모양의 decision boundary로 해결할 수 없어 보인다.

​       이를 해결하기 위해서 가설함수를 h(x) = g(θ<sub>0</sub>+θ<sub>1</sub>x<sub>1</sub>+θ<sub>2</sub>x<sub>2</sub>+θ<sub>3</sub>x<sub>1</sub>$^2$+θ<sub>4</sub>x$^2$) 의 형태로 만들었다  

​       그리고 $\theta$ 가 -1,0,0,1,1 으로 주어졌다고 가정하자

​       다시한번 z>=0 일 때는 y=1 예측을 하고 z<=0일 때는 y=0을 예측하는 사실을 이용해보자

​       주어진 $\theta$ 들을 대입하여, -1 + x<sub>1</sub>$^2$ + x<sub>2</sub>$^2$  >= 이면 y=1, <=이면 y=0으로 예측할 수 있고 그에 따라서

​	 decision boundary는 circle 모양으로 표현할 수 있게 된다

* 즉 decision boundary는 직선,원 등의 형태 뿐만 아니라 여러 비선형 적인 모양을 가질 수 있는데 이러한 경우

   다항식의 차수를 높혀 Non-linear 한 Decision Boundary를 표현할 수 있다

## Cost Function

* 이전에 선형회귀법을 위한 Cost Function 을 아래와 같은 MSE(Mean Square Error) 방식을 이용했다
  $$
  Cost\ Function =J(\theta0,\theta1)=\frac 1 {2m}\sum\limits_{i=1}^{m}(h(x^i) - y^i)^2
  $$
  Logistic Regression문제를 이와 같은 Cost function을 적용해보자.

  ![image-20201219164017322](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\6.png)

  이와같은 그래프 개형으로 non- convex 형태가 된다. 즉 Global Minimum으로 수렴하는 것이 보장되지 않는 문제가 생긴다

  그렇다면 Logistic Regression을 위한 Cost function이 필요할 것이고 이것에 대해서 정의해 보겠다

![image-20201219164426962](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\7.png)

  

* Cost function을 이와같이 사용하는 이유를 알아보겠다.

  ![image-20201219165225048](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\8.png)

  우선 y=1 인 경우의 -log(h(x))의 그래프를 그려보면 h(x)는 0 과 1 사이의 값을 같기 때문에 이와 같은 개형이 나온다.

  만약 y=1 , h(x) =1 즉 예측값과 일치했을 때 Cost=0임을 알 수있고

  ​        y=1, h(x) =0 예측값과 불일치했을 떄 Cost 가 무한대로 증가하는 것을 알 수 있다.

  ![image-20201219165536982](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\9.png)

  반대로 y=0인 경우의 -log(1-h(x)) 의 그래프를 그려보면 위와 같은 개형이 그려진다.

  만약 y=0이고 h(x) =1 이면 Cost 가 무한대로 커지는 것을 알 수있고

  만약 y=0이고 h(x) =0 이면 Cost 가 0이 되는 것을 알 수있다.

* 그래프 개형적 특성 때문에 Logistic Function 의 Cost function 이 왜 이렇게 정의 되었는지를 알 수 있다.

## Simplified Cost Function and Gradient Descent

![image-20201219165929179](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\10.png)

* Logistic regression 의 cost function을 위와 같이 두 줄을 이용해서 표기하였는데 Gradient Descent 알고리즘을 유도하기 쉽게

   공식을 한 줄로 유도할 수 있다
  $$
  Cost(h(x),y) = -ylog(h(x))-(1-y)log(1-h(x))
  $$
  이 함수는 Convex개형으로 그려지기 때문에 Logistic Regression Model을 해결하기 적절한 함수로써 이용된다

* 또한 Cost Function을 minimize 하기 위해서 이번에도 선형회기법과 같은 방법으로 Gradient Descent 방식을 이용 할 것이다

  ![image-20201219171207805](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\11.png)

  이전의 선형회기때와 식은 같지만 h(x) 가설함수가 달라졌기 때문에 값은 다르게 나온다.

  이번에도 마찬가지로 $\theta$ 들의 값은 동시에 업데이트를 해줘야 한다는 것에 유의한다.

## Multi-class Classification

* 지금까지는 0과1, yes or no ... 둘 중에 하나로 분류 되는 binary classification 문제에 대해서 공부했었다

* 이러한 분류 문제 이외에도 ==여러개의 Class로 분류되는 문제==를 Multi-class Classification 문제라고 한다

  ![image-20201219172533840](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\12.png)

  * 이러한 문제를 해결하기 위해서는 One-vs-all(One-vs-rest) 알고리즘을 사용하여 문제를 해결 할 수 있다

    "n 개의 Class가 있는 문제를 n개의 Binary Classification 문제로 바꾸어 해결한다" 의 개념이다.

    * △ 가 y=1 , 나머지는 y=0
    * □  가 y=1, 나머지는 y=0
    * x  가 y=1, 나머지는 y=0

  이렇게 3개번의 Classification 과정을 통해, 그 중에 가장 최대의 확률을 갖는 Class를 선택하는 방식을 이용한다.

  





