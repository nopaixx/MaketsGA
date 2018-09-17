# MaketsGA
Hands On Neural Network inside Metatrader

In this article, we attempt an approach to the use of architecture and modeling of deep neural networks inside trading plataform "IN THE BOX" without external librarys.

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

# Basic concepts about GA
Genetic algorithms are based on the genetic process of biologically living organisms. Where in a closed environment of X species only the strongest (or the best adapted) to the environment survive in a new species to evolve to a new specie with a mixted gens of the survivors.

Genes->Parameter to be optimized

Species->Each species contains a certain amount of genes all together offer a solution to a specific problem.

Population-->Population is a group of species all together perform a generation

Evolution->Evolution is the step that is generated between one generation and another, where there is a mixture of genes that generates new species and new species are born with a certain mutation.

Fitness function--> The fitness function is the function that will tell us how good a species is and will be used to maximize the value.

![alt text](https://github.com/nopaixx/TensorFlow-GeneticsAlgo/blob/master/GA%20grafic.jpg)

# Genetics algorithms basis Neural Netowrk weigth optimization
As indicated in the introduction of the article we will try to optimize a neural network of any input dimension any internal dimension and any output dimension. However, to facilitate the exposure we will use the simplest Neural Network structure named percepton.

![alt text](https://github.com/nopaixx/MaketsGA/blob/master/percepton.jpg)

As you probably already know, we must then find the values W1..WN that comply with the following formula

![alt text](https://github.com/nopaixx/MaketsGA/blob/master/formulapercepton.jpg)

Now we need optimize all Wn weigth using Genetics Algorithms, and we apply a binary transformatión of all weigth where all group of gene in specie represent a Wn weigth(note we initialized weigth in random way)

![alt text](https://github.com/nopaixx/MaketsGA/blob/master/wtobin.jpg)


After applying the fitness function in each species we can obtain the best species of each generation and create new species that are born through the process of random crossover based on their progenitors (plus a small mutation).

![alt text](https://github.com/nopaixx/MaketsGA/blob/master/newspecieborn.jpg)


# Loss function vs fitness function
The loss function in a machine learning represents the error between the predicted value vs. the true value, usually the objective of the neural network is to minimize the error, understanding that a minor error come to a better final result. The most common loss functions are MAE, MSE for regression problems and crossentropy for classification problems. (It's not the purpose of this document to analyze the performance of these functions).

The main difference between the gradient descent method and the genetic algorithms to optimize the weights of a neural network, is that, while in the first we must process over and over again the loss function epoch to epoch to see how the error decreases, with the genetic algorithm we do not need to process the network any time of training simply we should run several networks (piloted by the genes of a species) and check how good is(or bad have done).

To know how good is a species we will process all the actions performed to the fitness function. The design and development of the fitness function is a more important key element for the proper functioning of the genetic algorithm and will determine the success of the complex trading system.

# Our Fitness function
The main objective of the document is the creation of a neural network optimized by genetic algorithm in order to discover a complex trading system capable of making any decision available on the platform. This includes both opening and closing trades as well as managing open trades at all times and making decisions regarding the performance of trades and capital management. For this reason our loss function will be based on:

![alt text](https://github.com/nopaixx/MaketsGA/blob/master/fitness_function.jpg)
[More info about ratio sharpe](https://es.wikipedia.org/wiki/Ratio_de_Sharpe).

In summary our fitness function tell us how good is a specie before to evolve a new population of species and our fitness function value is equal to final_balance * ratio_sharpe. I have chosen a ratio shape because in its base the ratio shape includes a risk renatability analysis based on a reference environment and we therefore want our fitness function to include a risk analysis of each investment. In summary, the more benefit based on ratio shape the better.
(sure fitness function could be improved in future)

```
//we use TesterStatistics MQL function to help calculations
fitness_function=TesterStatistics(STAT_PROFIT)*TesterStatistics(STAT_SHARPE_RATIO);

```

# Software arquitecture

![alt text](https://github.com/nopaixx/MaketsGA/blob/master/softarquitecture.jpg)

# Our Neural Network arquitecture with 2000 inputs 3 hidden layers(35 neuron each) and 40 outputs

![alt text](https://github.com/nopaixx/MaketsGA/blob/master/netarquitecture.jpg)

This is the neural network that we are going to train (with a small change in the software it could be of any dimension and have more hidden layers).

We send 2000 input variables this is (25 backbars * 5 symbols * 10 max slots orders )* 2 lots and profits.
After apply weigth calculation sytem can perform 40 ouputs operations at once this is 8 operation * 5 symbols.

In summary our genetic algorithm need perform optimization of 74.200 parameters this is (2000 * 35)+(35 * 35)+(35 * 35)+(35 * 50)



# GA_Manager.mq5
This library implement a genetic algorhitm to evolve each population. And we deploy a full conected network with 2000 inputs 3 hidden layers with 35 neurons each with 40 outputs

```
//all symbols abailable to trade we try to discober complex system with all symbols together
string symb[]={"EURUSD","GBPUSD","AUDUSD","USDJPY","USDCAD"};
//we send 25 orders back history
int backbars=25;
//maxtotal orders abailable per symbol and per direction(buy/sell)
int totalorders=10;
//First input layer size
int size_out1=35;
//Second input layer size
int size_out2=35;
//third input layer size
int size_out3=35;
//output layer
int out=40;//lot 
//total number of species in population
input int numpopulation=75;
//user for selection 2 best species
input double i_tournamentSize = 5;
//if we need add best species in new generation
input bool i_elitism = true;
//user for randome crossover 
input double i_uniformRate = 0.5;
//user for random mutation rate
input double   i_mutationRate = 0.015;
```

Now we can analize 2 most important metohds
Function crossover recibe two individual species and perform a random crossover and return  new specie completly evolved

```
// Crossover individuals
Individual* Algorithm::crossover(Individual &indiv1, Individual &indiv2) {
        Individual *newSol = new Individual();
        newSol.initIndividual(bitlength);        
        for (int i = 0; i < indiv1.size(); i++) {        
            Alg.HQRndRandomize(&state);
            double rand1=UniformValue(-1,1);
            
            if (rand1 <= uniformRate) {
                newSol.setGene(i, indiv1.getGene(i));
            } else {
                newSol.setGene(i, indiv2.getGene(i));
            }
        }
        return newSol;
    }
 
```

Function mutate perform a random mutation to some genes in specie
```
// Mutate an individual
void Algorithm::mutate(Individual &indiv) {
        // Loop through genes
        for (int i = 0; i < indiv.size(); i++) {
            Alg.HQRndRandomize(&state);
            double rand1=UniformValue(0,1);
            
            if (rand1 <= mutationRate) {
                // Create random gene                
                Alg.HQRndRandomize(&state);
                double rand1=UniformValue(0,2);                
                double gene = indiv.getGene(i)*rand1;
                indiv.setGene(i, gene);
            }
        }
    }
```



# speciesrun.mq5
This library deploy class CNET to perform a internal neural network calculation and perform trades bases on optimized weigth

```
class CNet
  {  
  private:
   double damepercept(int pid,double &values[]);
  public:      
   double xvalues[];
   int numinput;
   int numouts;
   int               CNet_OnInit(int inputs,int outputs);
   double applycalcs(double &values[],double &percepts[]);
   
  };
  ```
Function applycalcs(double &values[],double &percepts[]) recibe input values and perform percepton calculation and return a percepts as output.

Let's go to analize how we call a full conected network of 3 hidden layers with 35 neurons each 

```
      net1.applycalcs(ind1,ind2);
      net2.applycalcs(ind2,ind3);
      net3.applycalcs(ind3,ind4);
      out1.applycalcs(ind4,outs);
```

ind1 var contain all inputs, after apply 3 layers calculation outs contain all outputs optimized

Finally let go to analize go we can perform tradings using neural network outputs
````
      
      ArrayResize(profitbuy,numbarshistory+1);
      ArrayResize(profitsell,numbarshistory+1);
      
      ArrayResize(volbuy,numbarshistory+1);
      ArrayResize(volsell,numbarshistory+1);
      
      for (int x=0;x<ArraySize(symb);x++){
         profitbuy[numbarshistory][x]=getProfits(POSITION_TYPE_BUY,symb[x]);
         profitsell[numbarshistory][x]=getProfits(POSITION_TYPE_SELL,symb[x]);
         
         volbuy[numbarshistory][x]=getVolums(POSITION_TYPE_BUY,symb[x]);
         volsell[numbarshistory][x]=getVolums(POSITION_TYPE_SELL,symb[x]);
         orderscount+=volbuy[numbarshistory][x]+volsell[numbarshistory][x];
      }
      numbarshistory++;
      
      double ind1[];// INPUTS
      double ind2[];//output ind for first net
      double ind3[];//output ind for sec net;
      double ind4[];//output ind for sec net;
      double outs[];//siganles >0 buy <0sell    
      
      ArrayResize(ind1,4*backbars*ArraySize(symb));
      int cta=0;
      for (int x=numbarshistory-1;x>=numbarshistory-backbars;x--){
         if (x<=0){
            for (int y=0;y<ArraySize(symb);y++){
               ind1[cta]=1; //profitbu inc
               cta++;
               ind1[cta]=1;
               cta++;
               ind1[cta]=1;
               cta++;
               ind1[cta]=1;
               cta++;
            }
         }else{
            for (int y=0;y<ArraySize(symb);y++){
               ind1[cta]=profitbuy[x][y]/profitbuy[x-1][y];
               cta++;
               ind1[cta]=profitsell[x][y]/profitsell[x-1][y];
               cta++;
               /*ind1[cta]=volbuy[x][y]/volbuy[x-1][y];
               cta++;
               ind1[cta]=volsell[x][y]/volsell[x-1][y];*/
               ind1[cta]=volbuy[x][y];
               cta++;
               ind1[cta]=volsell[x][y];
               cta++;
               
            }
         }
         
      }
      
      //apply networks calcs
      net1.applycalcs(ind1,ind2);
      net2.applycalcs(ind2,ind3);
      net3.applycalcs(ind3,ind4);
      out1.applycalcs(ind4,outs);
      
      
      int symbid=0;
      // 4 actions available for each symbol + 4 lot values
      //out0=need close lot buy
      //out1=lot to close on out0
      //out2=need close lot sell
      //out3=lot to close on out2
      //out4=need open buy
      //out5=lot to open on out 4
      //out6=need open sell
      //out7=lot to open on out5
      for (int x=0;x<(ArraySize(symb)*8);x+=8){      
         
         if (outs[x]>0){
            //need close lotbuy outs[x+1]
            printf("CLOSE1");
            close(POSITION_TYPE_BUY,outs[x+1],symb[symbid]);
         }   
         if (outs[x+2]>0){
            //need close lotsell outs[x+3]
            printf("CLOSE2");
            close(POSITION_TYPE_SELL,outs[x+3],symb[symbid]);
         }   
         if (outs[x+4]>0){
            //need open lot buy outs[x+5]
            printf("OPEN1");
            open(POSITION_TYPE_BUY,outs[x+5],symb[symbid]);
         }
         if (outs[x+6]>0){
            //need open lot sell outs[x+7]
            printf("OPEN2");
            open(POSITION_TYPE_SELL,outs[x+7],symb[symbid]);
         }
                  
         symbid++;                        
      }
      //trading by outs outstrategy
````

In summary this robot can perform 8 operation for each symbol available open/close/buy/sell simultaneously



# Backtesting analisis
After running some evolution we can see optimization grah where each blue dot represent the result for each specie

![alt text](https://github.com/nopaixx/MaketsGA/blob/master/evolutionnet.jpg)

Nice!, each new evolution neural network weigth perform and optimization and improve previous generation results

We can test it with the best specie

![alt text](https://github.com/nopaixx/MaketsGA/blob/master/runbest.jpg)


# Future improvements

-Add GaussianNoise at each input layer to reduce overfitting and increas performance on test data

-I detected some output neuron witout activation we need to add dropout to force neurons activaction and reduce overfitting

-Add normalization at each layer output

-Add and deploy activations at each output layer

-Try optimization with Simulation Annelaning

-Implement Neuro Evolution Augmented Topologies (NEAT) to self optimized network arquitecture



# Conclusions

-In this article i demonstrated how we can develop our own technology to work with neural networks in our trading platform without the use of external librarys. In our case i take as a reference a network of 2000 inputs 3 hidden layers and 40 outputs(with small change we can use any dimension)

-I have developed a network that is capable of making 40 decisions at a time (8 for each financial instrument) this is as a main requiremnt of this document, network can deploy and discover a full complex trading strategy based on all instruments together.
This means Neural Network can take desicions as open/close/buy/sell per instrument simultanously and deploy a full strategy like hedging over instrument and hedging over correlated instrument

-This allows the start of a promising trading system(not value prediction) that could be discovered by a neural network.

-This library can also be the basis for the development of promising technical indicators based on operation (not prediction).

-As a chimera for any trader i have developed a system that does NOT use any technical indicator.









