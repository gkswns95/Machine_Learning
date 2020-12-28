# 11  Machine Learning System Design

이번 챕터에서는 기계 학습 시스템 설계에 대해 몇 가지에 대해서 공부할 것이고, 이번 챕터에서 복잡한 기계 학습 시스템을 설계할 때

직면하는 문제들에 대해서, 시스템을 전략화하는 방법에 대한 것을 배울 것이다.

![image-20201223231319917](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\47.png)

## Building a spam classifier

Spam 문자를 걸러내는 Classifier를 만든다고 하고 하고 label 1 = spam, label 0 = Non-spam 으로 정했다고 하자. 이러한 메일들을 Supervised Learning

으로 구분하는 방법으로 여러가지 방법이 존재한다.

* Spam 과 Non-spam 을 나타내는 feature들을 100가지의 단어로 사용한다

   예를들어 deal,buy,now,discount 등의 단어를 feature로 사용하며, 이 feature들을 벡터의 형태로 만들어 

  메일에 이 단어가 존재하면 1, 아니면 0 으로 분류하여 분류기를 만들 수가 있다.

* 이러한 Classifier의 정확성을 올리려면 많은 양의 데이터를 수집하거나, 일부러 misspelling 하는 것을 감지하는 알고리즘을 더욱 개발하는 등의

  방법들이 존재한다. 

##  Error Analysis

에러를 분석하는 방법으로 일단 간단한 알고리즘으로 시스템을 구축하여, 이후 Cross-validation Data로 Testing을 하여 Learning curve를 plot해보고

문제가 어디서 발생하는지 정확히 진단한 후에 최적화 하는 방식을 추천한다. 처음부터 복잡한 시스템을 만들고 직관적으로 문제가 어디에서 발생하는 

지를 찾으려고 하면 시간낭비가 많이 될 수 있기 때문이다.

즉 Error Analysis 방법은  cv-data 를 이용하여 Testing을 하며 Learning curves를 plotting하여  문제를 진단하는 방법이다.



사용자가 작성한 스팸문자 분류 알고리즘이  500개의 Cross Validation Set 중 100개의 data를 잘 못 구분하였을때 이것을 수동으로 분석해야 한다.

* What type of email it is?
* What cues (features) you think would have helped the algorithm classify them correctly

![image-20201223235252491](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\48.png)

이처럼 어떤종류의 Email인지 종류별로 분류하고, 현재 Steal passwords에 대한 분류가 제대로 이루어지지 않는 것을 알 수 있다. 이 사실을 바탕으로

이 부분을 개선해야 알고리즘의 성능이 향상된다는 것을 알 수 있을 것이다. 두 번째로 오른쪽 분홍색의 항목들 처럼 어떤 Features 를 추가해야 알고리즘의

성능을 개선할 수 있는지를 생각해 봐야 할 것이다. 이러한 방법은 수동으로 직접 오류를 검사하고 개선해 나가는 방식이 Error Analysis 방법으로 사용할 수 있다.

### The importance of numerical evaluation

마지막으로 학습 알고리즘을 개발할 때 유용한 팁 중에 하나로 학습 알고리즘에 대한 수치적 평가가 있는지 확인하는 것이다.

![image-20201224000154472](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\49.png)



만일 discount/discounts/discounted/discounting 같은 비슷하게 보이는 단어들을 같은 단어로 취급해야 할지 안할지 결정해야 한다고 생각해보자.

실제 자연어 처리에서 이것을 수행하는 방법으로 "Stemming" software를 사용하는 것이다. 하지만 이것을 사용할 때의 단점도 분명 존재한다 예를들어

universe / university 와 같은 단어는 명백히 뜻이 다른 단어로 구분지어야 하는데 software가 이것을 같은 단어로 취급할 수 있기 때문이다.

이러한 단점들이 존재하는데도 불구하고 Stemming software를 사용할지 말지에 대한 판단은 실제로 적용해보고 error가 줄었는지 줄지 않았는지 비교하여

판단하면 된다.

![image-20201224001240734](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\50.png)

## **Error metrics for skewed classes**

Skewed classes 는 분류 문제에서 발생하는 문제로서 한 집단이 다른 집단보다 많이 작은경우 생기는 집단을 말한다.

![image-20201224003845159](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\51.png)

여기 이전에 공부하였던 암 분류 문제 예시가 있다. 만약 악성종양이면 y=1, 양성종양이면 y=0으로 분류하는 logistic regression model이 있었다.

우리가 세운 모델이 1%의 error를 가지고 있다고 한다면, 99% 는 진단에 성공한 것이기 때문에 꽤나 훌륭한 알고리즘이라고 생각할 수 있다.

하지만 실제로 악성종양이었던 환자가 0.5% 였다면 나머지 0.5% 는 잘못 예측 한 것이기 때문에 이러한 관점으로 보았을 때는 1% error가 썩

좋은 결과로 보이지는 않는다.  그리고 function y 의 함수를 만들어 모든 입력에 대해 악성종양이 아니라고 반환하는 알고리즘을 작성하였다고 하자.

이 알고리즘의 성능은 단 0.5% error 밖에 없는 것이다. 그럼 이것이 이전의 알고리즘을 개선한 더 좋은 알고리즘 이라고 할 수 있는 것은 아니다.

즉, 이렇게 한 집단이 다른 집단에 비해 쏠려 있는 경우 skewed classes 라고 한다.

### [Precision / recall]

이러한 skewed classes 문제에 직면하였을 때 다른 Error metrics를 생각해 내야 할 것이고 그러한 metrics 가 바로 precision 과 recall 이다.

![image-20201224011701634](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\52.png)

위와 같이 Actual class 와 predicted class 관계에 대해서 표를 그려서 총 4부분으로 나누어 볼 수있다.

* True positive = 예측 결과와 실제 결과가 동일한 영역
* False positive = 예측 결과는 true였지만 실제 class에서는 false인 영역
* True negative = 예측 결과 false였고 실제 결과도 false인 영역
* False negative = 예측 결과가 false였는데 실제 결과가 true 인 영역

이렇게 4개의 용어에 대해서 정의하였고 이제는 알고리즘의 성능을 평가하는 다른 방법인 Precision 과 Recall 에 대해서 정의해 보겠다.

* Precision : True positives / number of predicted positive 의 비율로 암 진단 문제를 예를 들어, 우리가 암환자라고 예측했던 환자가 실제로 암을 가진

  환자들에 대한 비율을 나타낸 것이다.

* Recall : 암에 걸린 사람들 중에 우리가 암에 걸린 사람을 맞춘 비율을 의미한다.

  ![image-20201224013610697](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\53.png)

우리가 Logistic regression 을 이용하여 분류 문제를 풀 때, threshold 값을 정하여 그 값이 0.5이상이면 1, 0.5 이하이면 0 으로 예측을 했었다.

마찬가지로 암 환자 분류 문제에 이것을 적용한다고 하고 암환자를 검사하는 것을 좀 더 확실하게 하고 싶다고 가정하자.

그렇다면 간단하게 threshold 값을 늘리면 된다. threshold 값을 늘림으로써 우리는 좀 더 많은 확실하게 암환자에 대한 정밀도가 올라갈 것이며

반대로 recall은 떨어지게 되어있다. 쉽게 말해 Higher precision, Lower recall 의 형태가 될 것이다.



우리가 암의 실제 사례를 너무 많이 놓치는 것을 피하기를 원할때를 가정하자.

![image-20201224013947438](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\54.png)

암을 가진 환자가 있는데 암이 없다고 예측을 하면 좋은 예측이 될 수 없다. 이러한 경우에는 반대로 threshold 값을 낮추어 가설함수를 적용하면 된다.

그렇다면 아까와는 반대로 Higher recall, lower precision 의 형태의 결과가 나올 것이다.

![image-20201224014241723](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\55.png)

Recall과 Precision의 관계를 그래프로 표현하면 위와 같이 서로 대조되는 모습을 갖는다. 이를 precision 과 recall 의 trading off라고 부른다

![image-20201224014538729](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\56.png)

우리에게 이러한 알고리즘 3개가 주어져 있다고 가정하자. 우리는 이 알고리즘들 중에서 어떤 것을 골라야 가장 좋은 성능을 가질 수 있는지 결정할 수 있을까?

![image-20201224014933398](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\57.png)

이와 같은 공식을 사용하여 어떤것이 최적의 알고리즘인지 판단할 수가 있게 된다.

## Data For Machine Learning

![image-20201224131021420](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\58.png)

문장 빈칸에서 어떠한 단어가 들어갈 지 고르는 알고리즘을 사용하려면 여러 알고리즘이 존재한다.

* Perceptron (Logistic regression)
* Winnow
* Memory-based
* Naive Bayes

여러 알고리즘이 존재하는데  오른쪽 그래프를 보면 실제로 알고리즘의 Accuracy는 거의 비슷하다고 볼 수 있다. 알고리즘의 성능을 향상시킬 수 있는

요소는 바로 Training set size 이다. 학습 데이터의 크기가 커질수록 알고리즘의 성능이 올라오는 것을 바탕으로 얻어낸 결과이다. 즉 데이터의 크기를

충분히 늘리는 것이 알고리즘의 성능을 향상시킬 수 있는 확실한 방법이라고 결론지을 수가 있다.

> "it's not who has the best algorithm that wins. It's who has the most data." 



