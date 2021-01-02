# 16: Recommender Systems

Recommender system(추천시스템) 은 최근들어 많이 사용되고 있는 기술이다. 구글, 유튜브, 넷플릭스 등 사용자가 원하는 컨텐츠를 예측하여 추천해 주는 기술로 요새는 추천시스템이 실제 수익에 상당히 큰 비중을 차지하는 추세이다. 다음은 추천 시스템의 개념을 예시를 통해 확인해 보겠다.

![image-20210102193916679](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\131.png)

위의 예시는 영화 평점을 예측하는 문제로 주어진 영화는 총 5개, 평점을 내릴 사용자는 총 4명으로 구성되어 있는 데이터이다. 또한 이제부터 쓰일 용어에 대해 정의하면 다음과 같다.

* $n_u$ :  사용자의 숫자
* $n_m$ : 영화의 숫자
* r(i,j) : 사용자 j가 영화 i를 평가 했는지의 여부 (평가=1 / 미평가=0)
* $y^{(i,j)}$ : 사용자 j가 영화 i를 평가한 점수 (0~5점)

위의 데이터로 얻을 수 있는 사실을 정리해보자. 현재 총 5개의 영화중 상위 3개의 초록색 박스는 로맨스 영화이고 나머지 2개의 영화는 액션영화로 분류될 수 있을 것이다. 장르별 사용자의 평점을 통해 알수 있는 사실들을 정리해보면 다음과 같다.

* Alice 는 액션 영화보다 로맨스 영화에 더 좋은 평점을 주었다. 이 사실을 바탕으로 평점을 입력하지 않은 로맨스 영화도 높은 점수가 될 것을 예측해볼 수 있다
* Dave 는 Alice와는 반대로 액션영화에 높은 점수를 주었기 때문에 다른 액션 영화에도 높은 평점을 줄 것이라는 사실을 예측해 볼 수 있다

이제부터 우리는 이러한 추측을 바탕으로 추천시스템을 설계하는 법과 사용하는 방법에 대해서 공부할 것이다.

## **Content based recommendation**

추천시스템에서 사용하는 방법 중 하나인 Content based recommendation에 대해서 알아보겠다.

![image-20210102212645399](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\132.png)

위와 같은 데이터가 주어졌을 때 물음표(?) 에 대한 값을 어떤 방법으로 예측할 수 있을지 알아보겠다.

![image-20210102212801273](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\133.png)

* $\theta$ : 각 사용자의 theta값이다 크기는 feature의 크기와 같고 각각은 feature의 가중치를 나타내고 있다
* x : feature들의 벡터로 여기에서는 x1(romance), x2(action) 로맨스와 액션의 정도를 나타내는 feature의 값들을 나타낸다

위의 예측은 linear regression(선형 회귀) 방식을 사용한다. 만약 사용자 Alice의 영화 Cute puppies of love 의 대한 평점을 예측하려 할 때의 

예시를 통해 알아보겠다. x3=[1,0.99,0] (1 x 3 벡터) 이고 $\theta$ = [0 5 0] 으로 주어졌다고 가정하자(Alice는 로맨스 영화에 대한 평점을 5점으로 전부 주었고 액션 영화는 전부 0점을 주었기 때문에 이와 같은 값으로 가정한다). 이를 통해 선형회귀법으로 계산하면 5 x 0.99 = 4.95의 값이 나온다. 즉 Cute puppies of love 에 대한 Alice의 평점 예측 결과 4.5가 나온다는 의미이다.

다음으로는 CBR의 optimization object와 Gradient descent 에 대해서 정의하겠다.

![image-20210102213656493](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\134.png)

* cost는 mse 방식을 사용하여 계산하는 것으로 선형회귀와 동일하다
* 선형회귀와의 차이점은 1/m로 나누는 부분을 없애주었다

![image-20210102213901004](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\135.png)

Gradient descent 방식을 사용하여 $\theta$ 의 값을 update 해준다.

## **Collaborative** **filtering - overview**

이번에 배울 Collaborative filtering은 사용자의 action을 통해 예측을 하기 때문에 사용자와 머신러닝의 협동이 이루어 졌다고 하여 지어진 이름이다.

![image-20210102221303198](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\136.png)

위의 데이터는 이전의 CBR 를 적용할 때 사용한 데이터이다. 이 때 우리는 x1, x2 feature들의 정도를 알고 있고, 그 주어진 정보를 통해 $\theta$ 의 값을 계산했었다.

하지만 이러한 feature들을 계산하는데 많은 시간과 비용이 필요할 것이다. 이번에는 feature의 정도를 모른다고 가정하고 문제를 푸는 방식에 대해서 공부할 것이다.

![image-20210102221549973](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\137.png)

위와 같이 x1, x2에 대한 값을 모른다고 해보자. 대신에 우리는 사용자들에 대한 $\theta$ 를 사용자에게서 아래와 같이 받았다고 가정하자.

![image-20210102221752432](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\138.png)

주어진 parameter를 분석해보면 다음과 같이 정리할 수 있다.

* Alice : [로맨스 영화 5 , 액션 영화 0] 로맨스 영화를 더 선호한다.
* Bob :  [로맨스 영화 5 , 액션 영화 0] 로맨스 영화를 더 선호한다.
* Carol: [로맨스 영화 0, 액션 영화 5] 액션 영화를 더 선호한다.
* Dave: [로맨스 영화 0, 액션 영화 5] 액션 영화를 더 선호한다.

이 정보를 가지고 우리는 x의값을 추론할 것이다. 이전의 linear regression 식을 적용하면 다음과 같다.

![image-20210102222837831](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\139.png)

이렇게 사용자에게서 얻은 parameter를 통해 x를 계산할 수가 있다.

이전에 배운 CBR 알고리즘과 Collaborate filtering 방식을 비교하면 다음과 같다.

* CBR : x의 값이 주어져 있고 그 값을 통해 $\theta$ 를 구함
* CF : $\theta$ 의 값을 통해 x의 값을 구함

![image-20210102223356263](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\140.png)

CF의 Cost function은 다음과 같다. CBR과의 차이점은 정규화 부분이다. CBR은 $\theta$ 를 구하는 알고리즘이고 CF는 x를 구하는 알고리즘이기 때문에

정규화 부분이 x로 이루어진 것을 알 수가 있다.

![image-20210102223538050](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\141.png)

이렇게 배운 두 가지 방식의 알고리즘을 혼용해서 사용하면 더 좋은 효율을 얻을수가 있다. 예를들어 초기 $\theta$의 값을 랜덤하게 설정한 후 x의 값을 계산하고 또 x의 값을 이용해서 $\theta$ 의 값을 계산하고 이 과정을 여러번 반복하여 최적화시키는 방식을 사용하면 훨씬 좋은 성능의 예측을 할 수 있을 것이다. 이제부터 두 가지의 알고리즘을 합쳐 최적화를 계산하는 방법에 대해서 알아보겠다.

## **Collaborative filtering Algorithm**

위에서 두 가지의 알고리즘을 혼용해서 사용하면 더 좋은 효율을 얻을 수 있을 것이라고 설명하였다. 이 두 가지 알고리즘의 공식을 하나로 합칠수 있고 그 공식은 아래와 같다.

![image-20210102230518995](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\142.png)

이 공식은 결국 두 가지 알고리즘의 공식을 합하여 하나의 공식으로 만든 것이라고 보면 된다. 이제 Cost Function을 하나의 공식으로 합쳤으니 전체적인 알고리즘에 대해서 살펴보자.

![image-20210102230637862](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\143.png)

*  x와 $\theta$ 의 값을 랜덤한 작은값으로 초기화 시킨다.
* cost를 계산하여 gradient descent algorithms을 이용하여 cost function을 minimize시킨다.
* 이와 같은 방식으로 계산된 x와 $\theta$ 값을 이용해, linear regression을 계산하여 rating을 예측한다.

## **Vectorization: Low rank matrix factorization**

![image-20210102233627501](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\144.png)

위와 같이 주어진 사용자, 영화 데이터를 행렬 Y로 표현할 수 있다. 그리고 행렬 Y의 각각의 요소는 또 아래와 같은 식으로 대체할 수 있을 것이다.

![image-20210102233752492](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\145.png)

Y의 각 요소들을 예측값을 계산하는 식으로 대체한 행렬의 모습이다.

다음은 X와 $\theta$ 를 모두 행렬의 형태로 만들어 아래와 같이 표현할 수 있으며, X$\theta^T$ 의 행렬간의 연산을 통해 predict한 값을 구해낼 수 있을 것이다.

![image-20210102234021449](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\146.png)

이처럼 행렬로 표현하는 방식을 바로 Low Rank Matrix Factorization 알고리즘이라고 한다.

## **Recommending new movies to a user**

우리는 이제까지 Collaborative Filtering 알고리즘을 배웠다. 이 알고리즘을 통해 얻어낸 feature와 parameter을 이용하여 사용자들이 원하는 영화를 추천할 수 있을까? 알고리즘을 통해 학습시킨 feature의 값을 x1(romance), x2(action) 라고 가정하고 각각의 값의 의미하는 바는 로맨스 영화를 좋아하는 정도와 액션 영화를 좋아하는 정도라고 해석할 수 있을 것이다. 즉 이렇게 학습되어 optimal한 값을 가진 feature들은 사용자 원하는 영화 정보를 담고 있는 것이다. 이러한 사실을 바탕으로 아래와 같은 공식을 적용할 수 있다.

![image-20210102234446575](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\147.png)

영화 i의 feature x의 값과 영화 j의 feature x의 값의 차이를 계산하여 가장 작은 값을 가지는 영화가 바로 movie i 와 가장 유사한 movie j가 되는 것이다. 

이렇게해서 우리는 사용자에게 영화를 추천하는 방법까지 알아보았다.



## **Implementation detail: Mean Normalization**

마지막으로 알아볼 내용은 바로 Mean Normalization이다. 아래의 예시를 통해 개념과 적용방법에 대해 설명하겠다.

![image-20210103001311667](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\148.png)

예시를 보면 사용자 Eva는 아직 모든 영화에 평점을 내리지 않아 Y matrix에 ?(물음표)로 기록이 되어있는 것을 알 수 있다. 이러한 상태에서 Eva의 데이터로 CF알고리즘을 적용하면, $\theta$의 값은 모두 0으로 앞부분의 공식은 0으로 없어지고 남은 정규화 부분도 결국 0으로 수렴할 것이다.  즉 사용자 Eva에 대한 평점은 모두 0으로 예측이 되는 문제가 발생하는 것이다.

이러한 상황에서 사용하는 것이 바로 Mean Normalization 방법이며, 쉽게 말해 사용자들이 평가한 각 영화의 평점의 평균값을 구해 적용하겠다 라는 아이디어이다. 

![image-20210103001848130](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\149.png)

각 열의 영화들의 평균을 구해 나온 행렬 M (5 x 1)을 신규 사용자인 Eva에게 적용한다. 이렇게 하여 아무영화도 평점을 내리지 않은 사용자 Eva에게도 영화를 추천할 수 있는 방법에 대한 방법인 Mean Normalization 까지 알아보았다.