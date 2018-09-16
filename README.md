# MaketsGA
Hands On Neural Network inside Metatrader

In this article, we attempt an approach to the use of architecture and modeling of deep neural networks inside metatrader 5 plataform "IN THE BOX" without external librarys.

Let's try to demonstrate how we can optimize a neural network of any dimension using genetic algorithms inside our trading platform.

We are NOT going to train a network to predict a single value (let's try our best) we are going to train a network to discover a complex trading strategy by itself without the limitation of human knowledge.


# Mandatory Literature
It is assumed that readers are familiar with the basic concepts of deep networks and financial markets. 

Attached are some interesting articles to familiarize the reader with the technology that is intended to be developed.

1.-[THIRD GENERATION NEURAL NETWORKS: DEEP NETWORKS](https://www.mql5.com/en/articles/1103).

2.-[EVALUATION AND SELECTION OF VARIABLES FOR MACHINE LEARNING MODELS](https://www.mql5.com/en/articles/2029).

3.-[RANDOM FORESTS PREDICT TRENDS](https://www.mql5.com/en/articles/1165)

4.-[MARKET THEORY](https://www.mql5.com/en/articles/1825)

# About Forex Makets and Metatrader
The Forex market is a decentralized global market of all currencies that are traded around the world. This market is the largest and most liquid in the world, with a daily trading volume in excess of 5 trillion dollars. The other stock markets in the world togethers not come close to this.

Metatrader is a popular software among individual and institutional traders. The main features that make it a reliable and popular software is that most Brokers, Market Makers (ECN) and Banks have acquired its license and offer their clients the possibility to trade with it. It also makes it possible to trade in the markets independently of the Borker without changing the platform.
Of course Metatrader incorporates a tecnical module and object-oriented programming language (in version 5) with you can develop  your own indicators and/or fully automatic trading robots, this makes Metatrader a good choice for both an individual trader and an institutional trader looking for a robust platform.

# About Quantitative Traders
Quantitative trading provides a new science or strategy to operate in the financial markets applying statistical models and in recent years models related to machine learning/deep learning.

What are we trying to do with quantitative trading? 

Usually quantitave traders download(csv,sql etc..) financial timeseries then apply a treatment to the data if necessary (normalization PCA etc...). Finally they train a network(python/R etc..) normally with the objective of reducing the error between the prediction and the real value and finally they take a decision(automatic or not) based on model single prediction value. (This sounds interesting python and r have hundreds of liberties for machine learning some of them interesting as recently google library TensorFlow)

In other words we have price for time T-1....T-n as independent variables and we try to predict a T+1 price as dependent variable.


# Motivations about MarketsGA and Quantitative Traders problems

The main problems about quantitative trader encounters when operating in real market is that they are exposed to market volatility, rapid market movements, price slippage and the constant noise produced by the randomness of the market structure itself.

This means that a quantitative trader who is dedicated to predicting a value based on past values is NOT enough to develop a successful FOREX trading system. You need a complex system that can manage the volume of the operation, the risk of it and the capital of the account (and others...).

Can you imagine to train a network of any input dimension any internal dimension and any output dimension to try to discover a full complex trading system INSIDE TRADING PLATFORM?


# Genetics algorithms basis optimization
As indicated in the introduction of the article we will try to optimize a neural network of which input dimension any internal dimension and any output dimension. However, to facilitate the exposure we will use the simplest NN structure the perception.

![alt text](https://github.com/nopaixx/MaketsGA/blob/master/perceptonbasis.jpg)

As you probably already know, we must then find the values W1..WN that comply with the following formula

![alt text](https://github.com/nopaixx/MaketsGA/blob/master/formulapercepton.jpg)



# Loss function vs fitness function


# Software arquitecture
-onthebox
-tradedoubles

# geneticsevolution.mqh


# speciestrader.mqh


# backtesting analisis

# Future improvements

# Conclusions















