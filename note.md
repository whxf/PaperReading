# Paper Note

[TOC]

## Skip N-grams and Ranking Functions for Predicting Script Events

原文链接：https://www.aclweb.org/anthology/E12-1034

***总结：*** 比较预测事件方式方法之间的差异与优劣，从三个方面入手：1）训练数据的选择；2）bigram方式；3）预测方法比较。最终评测的指标是：Average ranking和Recall@N。

***核心收获：*** 1）预测时不仅可以选择链的最后一个node作为预测目标，也可以选择其他链中node作为预测目标；2）评判标注可以参考Average ranking计算平均score。

***论文概述：*** 
* 需要比较的三个方面
    * 训练数据的选择
        * all：数据集形成的所有事件链
        * long：长度大于等于5的事件链
        * longest：每条数据中最长的事件链（根据actor不同划分事件链）
    * Bigram
        * Regular：相邻的trigger对
        * 1-Skip：regular+中间间隔1个的trigger对
        * 2-Skip：regula+1-Skip+中间间隔2个的trigger对
    * Method
        * PMI
        * Ordered PMI
        * Bigram

* Evaluation Method
    * Averge Ranking
    * Recall@N

公式1是```score```函数，可以理解为，在一条事件链中取第m个为```miss event```，也就是需要预测的，根据公式2计算待选event和链中其他node的```P```值，然后```log```求和，算出score，score小的更好。

公式3是average ranking，就是整合整个数据集所有链计算得到的```f(e,c)```进行平均，Recall@N就是计算数据集中```f(e,c)```小于某个阈值的事件占比。


```math
f(e,c)= \sum_{k=1}^{m}\log P(e|c_k) + \sum_{k=m+1}^{n}\log P(c_k|e)....(1)
```
```math
P(e_1|e_2) = \frac{C(e_1,e_2)}{C(e_2)}....(2)
```
```math
AverageRanking = \frac{1}{|C|}\sum_{c \in C} rank_{sys} (c)....(3)
```
```math
Recall@N = \frac{1}{|C|} |\lbrace c:c \in C \wedge rank_{sys} (c) \leq N \rbrace|....(4)
```


## A Structured Self-Attentive Sentence Embedding

原文链接：https://arxiv.org/abs/1703.03130

***总结：*** 介绍了LSTM+SelfAttention对句子进行分类的方法

***核心收获：*** 提供了方法构造的详细过程，主要通过学习这个来实现LSTM+SelfAttention进行句子分类的模型

***论文概述：***

* 词向量嵌入:(shape: n * d, d is embedding size, n is sentence length)
```math
S=(\bf w_1,w_2,...w_n)
```
* 双向LSTM：将前后向结果拼接(shape: n * 2u, u is hidden size)
```math
\overrightarrow{h_t}=\overrightarrow{LSTM}(w_t,\overrightarrow {h_{t-1}})
```
```math
\overleftarrow{h_t}=\overleftarrow{LSTM}(w_t,\overleftarrow {h_{t-1}})
```
```math
h_{t} = [\overrightarrow {h_{t}},\overleftarrow {h_{t}}]
```
```math
H=(\bf h_1,h_2,...h_n)
```
* attention weight：(shape: Ws1 da * 2u, Ws2 r * da  output r * 2 * hidden_size, linear wx+b)
```math
{\bf a}=softmax({\bf w_{s2}}tanh(W_{s1}H^T))
```
```math
A=softmax({ W_{s2}}tanh(W_{s1}H^T))
```
* last：之后拉平，fc layer全连接，之后label，+dropout batch norm
```math
M=AH
```
* penalization term：交叉熵
