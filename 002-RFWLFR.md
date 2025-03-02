# Random forest weighted local Frechet regression with random objects {#RFWLFR}

**《带有随机对象的随机森林加权局部 Frechet 回归》**这篇文章是中国人民大学统计学院2025年春季研究生课程广义线性模型的小组汇报论文。小组成员除了我还包括Huang Jinming & Yang Zhe.这是[论文原文](https://www.jmlr.org/papers/volume25/23-0811/23-0811.pdf)。

## 提出的方法 {#rfwlfr2}

在\@ref(Local constant method)节正式展示方法之前，需要进行一些初步的准备工作，包括提供Frechet回归的背景介绍，并解释Frechet树的构建过程，这是我们方法的基本组成部分。

### 预备知识 {#rfwlfr21}

(`#rfwlfr21`带数字是否可以？？)

#### Frechet回归  

设$(\Omega,d)$为一个配备特定度量$d$的度量空间，$\mathcal{R}^p$是一个$p$维的欧氏空间。我们考虑一个随机对$(X,Y)\sim F,$其中$X\sim\mathcal{R}^p,Y\sim \Omega$并且$F$是$(X,Y)$的联合分布。记$X,Y$的边际分布分别为$F_X,F_Y$,条件分布$F_{X|Y},F_{Y|X}$也假设存在。当$\Omega\subseteq \mathcal{R}$时，经典回归的目标是估计条件均值  

$$
m(x)=E(Y\mid X=x)=\mathop{\arg\min}_{y\in\mathcal{R}} E\left\{(Y-y)^2\mid X=x\right\}.
$$
通过将欧几里得距离替换成$\Omega$的内在度量$d,$条件Fr\'echet均值可以定义为  

$$
m_{\oplus}(x)=\mathop{\arg\min}_{y \in \Omega} M_{\oplus}(x,y)=\mathop{\arg\min}_{y \in \Omega} E\big\{d^{2}(Y, y) \mid X=x\big\}.
$$
给定一个独立同分布的训练样本$\mathcal{D}_n=\{(X_i,Y_i)\}_{i=1}^n$其中$(X_i,Y_i)\sim F$,Fr\'echet回归的目标是在样本层面估计$m_{\oplus}(x)$,为此Hein (2009) 将Nadaraya-Watson回归推广到Frechet版本：  

$$
\hat{m}_{\oplus}^{\text{NW}}(x)=\mathop{\arg\min}_{y \in \Omega} \frac{1}{n} \sum_{i=1}^{n} K_{h}\left(X_{i}-x\right) d^{2}\left(Y_{i}, y\right),
$$

其中$K$是一个平滑核（如Epanechnikov核或高斯核），$h$为带宽，$K_h(\cdot)=h^{-1}K(\cdot/h)$.Petersen和Müller（2019）将标准的多元线性回归和局部线性回归重新表述为加权Fréchet均值的函数，并提出了全局Fréchet回归和局部Fréchet回归：  

$$
\hat{m}_{\oplus}(x)=\mathop{\arg\min}_{y \in \Omega}  \frac{1}{n}\sum_{i=1}^{n} s_{i n}(x) d^{2}\left(Y_{i}, y\right),
$$
其中$s_{in}(x)$在全局和局部Fr\'echet回归中具有不同的表示形式。  
Nadaraya-Watson Fr\'echet回归和局部Fréchet回归都涉及核加权函数$K$，这在$p ≥ 3$时限制了它们的应用。为了解决这个问题，我们的目标是借助随机森林的力量，生成一个更强大的加权函数，以应对适度较大的$p$.  
图\@ref(fig:manhattan)展示了纽约曼哈顿黄色出租车交通流统计和预测结果。这个问题可以表述为一个Fréchet回归问题，其中响应变量是一个网络（矩阵），并且考虑了14个预测变量。特别地，当$p = 14$时，Nadaraya-Watson Fréchet回归和局部Fréchet回归方法显示出显著的局限性。在这里，我们使用Fréchet充分维数约简（Ying和Yu，2022）来实现局部Fréchet回归。尽管全局Fréchet回归不受维度限制，但它依赖于线性假设才能达到令人满意的效果。相反，从图\@ref(fig:manhattan)可以明显看出，我们将在本文中提出的两种方法的预测准确性高于全局Fréchet回归。


<div class="figure">
<img src="https://Go9entle.github.io/picx-images-hosting/1740492700639.102axqi287.webp" alt="The first plot illustrates the flow statistics of yellow taxis in ten distinct zones of Manhattan, New York, during a certain time period. The thickness of the edges connecting vertices corresponds to the level of inter-zone traffic, while the size of vertices represents the total traffic volume within each zone. The remaining five plots from left to right are the predictions given by the global Fr'echet regression, local Fr'echet regression after dimension reduction, single index Fr'echet regression, RFWLCFR and RFWLLFR."  />
<p class="caption">(\#fig:manhattan)The first plot illustrates the flow statistics of yellow taxis in ten distinct zones of Manhattan, New York, during a certain time period. The thickness of the edges connecting vertices corresponds to the level of inter-zone traffic, while the size of vertices represents the total traffic volume within each zone. The remaining five plots from left to right are the predictions given by the global Fr'echet regression, local Fr'echet regression after dimension reduction, single index Fr'echet regression, RFWLCFR and RFWLLFR.</p>
</div>

#### Frechet树  

回归树$T$从根节点（整个输入空间）递归地划分输入空间。在每次划分时，父节点根据某个特征方向和某个切割点被分成两个子节点，这些切割点由特定的划分标准决定。经过多次划分后，子节点变得足够小，形成叶节点，叶节点中的样本数据用于估计条件（Fréchet）均值。

在本文中，我们使用Fréchet树来指代处理度量空间值响应的回归树，不管其划分标准是什么。这里我们引入了一种自适应标准——方差减少划分标准，它利用预测变量$X=(X^{(1)},...,X^{(p)})$和响应变量$Y$的信息来决定节点的划分。$Y$来自一般度量空间的杂质(impurity)不再由欧几里得距离下的方差来度量而是使用Fr\'echet方差。一个内部节点$A$的分裂可以用一对$(j,c),j\in\{1,...,p\}$来表示，表示在特征$X^{(j)}$方向上，节点$A$在位置$c$被分裂，我们选取最优的$(j_n^*,c_n^*)$使得样本Frechet方差尽可能减小，以至于在同一子节点的样本具有高度的相似性。具体地，分裂准则如下  

$$
\mathcal{L}_n(j,c)=\frac{1}{N_n(A)}\left\{\sum_{i:X_i\in A}d^2(Y_i,\bar{Y}_A) -\sum_{i:X_i\in A_{j,l}}d^2(Y_i,\bar{Y}_{A_{j,l}})-\sum_{i:X\in A_{j,r}}d^2(Y_i,\bar{Y}_{A_{j,r}}) \right\},
$$
其中$A_{j,l}=\{ x\in A:x^{(j)}<c \},A_{j,r}=\{ x\in A:x^{(j)}\geq c \},N_n(A)$是落在节点$A$中的样本数量，并且$\bar{Y}_A=\mathop{\arg\min}_{y\in\Omega} \sum_{i:X_i\in A}d^2(Y_i,y)$也就是节点$A$中样本$Y_i$的Frechet均值，$\bar{Y}_{A_{j,l}},\bar{Y}_{A_{j,r}}$也是类似定义的。于是最优分裂对由下面的式子决定  

$$
(j_n^*,c_n^*)=\mathop{\arg\max}_{j,c} \mathcal{L}_n(j,c).
$$

### Local constant method {#rfwlfr22}  

单棵树模型可能会因为调参的不同而面临较大的偏差或方差。为了提高预测精度，我们可以聚合多棵树来构建随机森林。随机森林的预测误差与不同树之间的相关性密切相关。除了通过重采样训练数据集来生长个别树之外，通常还会引入额外的随机性，以进一步降低树之间的相关性，从而提高随机森林的性能。例如，每次分裂前会随机选择一个特征子集，分裂方向仅基于该子集来设计。在这里，我们用$\xi\sim\Xi$表示这种辅助随机性。  

我们首先考虑经典的欧几里得响应的随机森林。每棵树是在训练数据集$\mathcal{D}_n$的一个子样本$\mathcal{D}_n^b=\{(X_{i_{b,1}},Y_{i_{b,1}}),(X_{i_{b,2}},Y_{i_{b,2}}),...,(X_{i_{b,s_n}},Y_{i_{b,s_n}})\}$上训练的，其中$1\leq i_{b,1}< i_{b,2}<...<i_{b,s_n}\leq n.$本文中假设子样本大小$s_n\rightarrow \infty,s_n/n\rightarrow 0$当$n\rightarrow \infty$时。数据重采样是无放回的。由$\mathcal{D}_n^b$和随机抽取$\xi_b\sim\Xi$构造的第$b$棵树$T_b$给出了$m(x)$的估计：

$$
T_b(x;\mathcal{D}_n^b,\xi_b)=\frac{1}{N(L_b(x;\mathcal{D}_n^b,\xi_b))}\sum_{i:X_i\in L_b(x;\mathcal{D}_n^b,\xi_b)} Y_i,
$$









