# 08: Neural Networks - Representation

* 만약에 아래 그림과 같이 복잡한 분류 문제가 있다고 하자.

  ![image-20201221230740976](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\15.png)

   이렇게 복잡한 분류 문제를 해결하기 위해서는 decision boundary가 Non-linear하게 그려져야 한다. Logistic regression을 이용하여 decision boundary를 구할 수 있지만, 다항식이 복잡해지고 심지어는 과적합 문제가 발생할 수 있는 가능성이 커지게 된다.

  ![image-20201221232106632](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\16.png)

  * 예를들어 위와 같이 자동차가 맞는지 아닌지 알아내는 알고리즘을 구현할 때

  * Decision Boundary는 Non-linear 형태가 될 것이고, pixel 값을 50 x 50 으로만 해도 2500

    RGB 값으로 하면 7500이 된다.

  즉, 이러한 Non-linear한 분류 문제에서 사용할 수 있는 Neural networks 가 이용되게 되었다

  ## Neurons and the brain

  * Neural networks(NNs) 는 사람의 뇌를 기능적으로 복제하여 기계가 사람의 뇌처럼 동작하도록 만들자 라는 아이디어에서 시작이 되었다
  * NNs 는 80~90 년 대에 인간의 뇌를 모방하는 알고리즘으로써 사용되었다가 perceptron 이란 책이 출간되어 NNs는 toy-problem 밖에 해결하지 못하고 real-world 의 문제를 해결할 수 없다라는 인식 때문에 한동안 주춤하였다가 최근에 들어 컴퓨터의 성능이 좋아짐에 따라 다시 한 번 주목을 받고 있는 알고리즘 이다.

  ## Model representation 1

  ![image-20201221233013280](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\17.png)

  * 위의 사진은 Neurone의 구조이고 크게 cell body, input wires(dendrite), Output wire(axon) 으로 구성된다

    <뉴런의 동작과정>

    * 뉴런은 input wire을 통해 전기적인 신호를 입력으로 받아
    * Cell body에서 해당 연산을 마친 후
    * Output 으로 또 다른 뉴런에게 정보를 전달한다

  ### Artificial neural network - representation of a neurone

  ![image-20201221233531346](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\18.png)

  

  * 여기 간단한 Artificial neural network 의 모델이 있다. 
  * input layer, computation layer, output layer 로 구성되어 있고
  * computation layer는 sigmoid 함수로 계산한다 이 함수를 activation function 이라고 부른다

  ### Neural networks - notation

  ![image-20201221234032209](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\19.png)

  * a<sub>i</sub><sup>(i)</sup> = activation of unit i in layer j
  * $\theta$<sup>(j)</sup> = matrix of parameters controlling the function mapping from layer j to layer j+1

  위의 NNs 에서는 input layer의 값과 가중치 $\theta$ 를 곱하고 그 값을 activation function을 통해 activation 시켜 

  hidden layer의 값이 결정되고 또 한 번 hidden layer - output layer 와 연결되어 있는 weight 값을 곱하여 

  output 값을 출력하는 과정으로 진행한다

  ### Forward Propagation(전방 전파)

  ![image-20201221234858502](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\20.png)

  * Hidden layer 의 값 $a^{(2)}$ 들은 그 전의 layer의 가중치를 x의 값과 곱하고 그 값을 activation 한 값을 저장한다. 이 때 activation function 을 활성화 시키기 전의 함수의 인자 값을 z 라고 표현한다면 위의 사진과 같이 g(z) 형식으로 표기할 수 있다

  * $z^{(2)}$ = $\theta$<sup>(1)</sup>x = $\theta$<sup>(1)</sup>a<sup>(1)</sup> 

  * a<sup>(2)</sup> = g(z<sup>(2)</sup>)

  * z<sup>(3)</sup> = $\theta$<sup>(2)</sup>a<sup>(2)</sup>

  * h<sub>$\theta$</sub>(x) = a<sup>(3)</sup> = g(z<sup>(3)</sup>)) 

    위와 같이 표현할 수 있으며 결국 h<sub>$\theta$</sub>(x) 를 계산하는 데 그 전 layer의 값들을 계산하면서 답을 얻어내기 때문에 이를 '전방 전파' 'Forward Propagation' 이라고 부른다.

  <Logistic regression 과 NNs의 차이점>

  ![image-20201221235802919](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\21.png)

  * NNs 의 input layer 층이 없다고 했을 때 위와 같은 형태의 모델이 된다.

  * Layer2 의 row data 들이 바로 activation function 즉 sigmoid 함수를 거쳐 output값을 출력한다

    이러한 동작과정은 logistic regression 과 똑같이 작동하는 것과 같다

  NNs는 input data를 바로 activation 시켜 출력하는 것이 아니라 row data를 최소 한 번의 학습과정을 거쳐서 출력으로 내보내는 것이 일반 logistic regression 과의 차이점이다. 일반 logistic regression의 모델로는 XOR문제와 같은 문제를 해결 할 수 없다. NNs을 이용하여 XOR문제를 푸는 것을 유도할 수 있다.

## **Neural network example - computing a complex, nonlinear function of the input**

![image-20201222001713305](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\22.png)

* Non-linear 한 문제가 주어졌을 때 어떻게 NNs를 이용하여 풀 수 있는지 증명을 할 수 있다

   ### Neural Network example 1: AND function

![image-20201222001929347](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\23.png)

* x<sub>1</sub>,x<sub>2</sub>는 binary 값을 가지며, y= x<sub>1</sub> AND x<sub>2</sub> 인 문제가 있다. 각 각 weight를 -30 +20 +20 을 주었을 때

  ![image-20201222002128173](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\24.png)

  위와 같이 x<sub>1</sub> x<sub>2</sub> 의 값들이 모두 1일 때만 1이 되는 AND문제를 sigmoid function 을 통해서 표현할 수 있다

### **Neural Network example 2: NOT function**

![image-20201222002328341](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\25.png)

* AND 문제와 마찬가지로 OR문제도 각 weight값을 적절히 이용하여 or 문제를 해결할 수 있다

**Neural Network example 3: XNOR function** 

위에서 만든 AND, OR 모델을 이용하여 Non-linear Decision Boundary 를 구할 수 있는 모델을 만들어 본다

![image-20201222002641055](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\26.png)

각각의 모델을 hidden layer를 통해 연결해 주면 아래와 같은 Non-linear 한 Decision Boundary를 구할 수 있게 된다.

![image-20201222002757478](C:\Users\Choi\AppData\Roaming\Typora\typora-user-images\27.png)

