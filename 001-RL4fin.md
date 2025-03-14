# (PART) 课程笔记 {-} 

# 强化学习在金融中的应用 {#rlforfin}
## Markov过程 {#Markov}

本书的主题是“序列不确定下的序列决策”，在本章中将暂时忽略“序列决策”方面而只关注”序列不确定性“。

### 过程中的状态概念

$S_t$是过程在时间$t$时的状态。特别地，我们对于下一时刻的状态$S_{t+1}$的概率感兴趣，如果已知现在的状态$S_t$和过去的状态$S_0,S_1,...,S_{t-1}$，我们对$P\{S_{t+1}|S_t,S_{t-1},...,S_0\}$感兴趣。

### 通过股票价格的例子理解Markov性

为了帮助理解，我们假设股票价格只取整数值，并且零或负股票价格是可以接受的。我们将时间$t$的股票价格表示为$X_t$.假设从时间$t$ 到下一个时间步骤 $t + 1$,股票价格可以上涨$1$或下跌$1$,即$X_{t+1}$的唯一两个结果是$X_t + 1$或$X_t − 1$.要了解股票价格随时间的随机演变，我们只需要量化上涨的概率 $P[X_{t+1} =X_t+ 1]$.我们将考虑股票价格演变的 3 个不同过程。

1. $P[X_{t+1}=X_t+1]=\frac{1}{1+e^{-\alpha_1(L-X_t)}}$.

   这意味着股票的价格倾向于均值回归(mean-reverting),均值即为参考水平$L,$拉力系数为$\alpha.$

   我们不妨设$S_t=X_t,$且可以看到下一时刻的状态$S_{t+1}$只与$S_t$有关而与$S_0,S_1,...,S_{t-1}$无关。即可写作
   $$
   P[S_{t+1}|S_t,S_{t-1},...,S_0]=P[S_{t+1}|S_t]\text{ for all }t\geq 0.
   $$
   这就被成为Markov性。

   书中还给出了相应的代码

   ```python
   from dataclasses import dataclass
   import numpy as np
   
   @dataclass
   class Process1:
       @dataclass
       class State:
           price: int
   
       level_param: int  # level to which price mean-reverts
       alpha1: float = 0.25  # strength of mean-reversion (non-negative value)
   
       def up_prob(self, state: State) -> float:
           return 1./(1+np.exp(-self.alpha1*(self.level_param-state.price)))
       def next_state(self, state:State) -> State:
           up_move: int = np.random.binomial(1, self.up_prob(state),1)[0] #生成随机移动 up_move = 0 or 1
           return Process1.State(price=state.price + up_move * 2 - 1) # 若up_move = 1, 则价格上升1，若为0价格下降1
   ```

   接下来，我们使用 Python 的生成器功能（使用`yield`）编写一个简单的模拟器，如下所示：

   ```python
   def simulation(process, start_state):
       state = start_state
       while True:
           yield state
           state = process.next_state(state)
   ```

   现在我们可以使用此模拟器函数生成采样轨迹。在下面的代码中，我们从 `start_price`的价格$X_0$​开始，在`time_steps`时间步长内生成`num_traces`个采样轨迹。使用 Python 的生成器功能，我们可以使用`itertools.islice`函数“懒惰地”执行此操作。

   ```python
   import itertools
   def process1_price_traces(
   	start_price: int,
       level_param: int,
       alpha1: float,
       time_steps: int,
       num_traces: int
   ) -> np.ndarray:
       process = Process1(level_param=level_param, alpha1=alpha1)
       start_state = Process1.State(price=start.price)
       return np.vstack([
           np.fromiter((s.price for s in itertools.islice(
   			simulation(process, start_state),
               time_steps + 1
   		)), float) for _ in range(num_traces)])
   ```

2. $$
   P[X_{t+1}=X_t+1]=\begin{cases}
   0.5(1-\alpha_2(X_t-X_{t-1}))&\text{  if }t>0\\
   0.5&\text{ if }t=0
   \end{cases}
   $$

   其中$\alpha_2$是一个“拉力强度”参数，取值在$[0,1]$之间。这里的直觉时下一步的移动的方向偏向于前一次移动的反方向。我们注意到如果依然按照前文建模则无法满足Markov性质，因为$X_{t+1}$取值的概率不仅依赖于$X_t,$还依赖于$X_{t-1}.$不过我们可以在这里做一个小技巧，即创建一个扩展状态$S_t$由一对$(X_t,X_{t-1})$组成。当$t=0$时状态$S_0$可以取值$(X_0,null)$,这里的null只是一个符号。通过将状态$S_t$视为$(X_t,X_t-X_{t-1})$建模可以发现Markov性质得到了满足。
   $$
   \begin{aligned}
   &P[(X_{t+1},X_{t+1}-X_t)|(X_t,X_t-X_{t-1}),...,(X_0,null)]\\
   =&P[(X_{t+1},X_{t+1}-X_t)|(X_t,X_t-X_{t-1})]\\
   =&0.5(1-\alpha_2(X_{t+1}-X_t)(X_t-X_{t-1}))
   \end{aligned}
   $$
   关于上面的式子deepseek给出了证明。

   人们自然会想知道，为什么状态不单单由$X_t - X_{t-1}$ 组成——换句话说，为什么 $X_t$ 也需要作为状态的一部分。确实，单独知道$X_t - X_{t-1}$可以完全确定 $X_{t+1}-X_t$的概率。因此，如果我们将状态设定为在任意时间步 $t$仅为 $X_t - X_{t-1}$，那么我们确实会得到一个只有两个状态 +1 和 -1 的马尔可夫过程（它们之间的概率转移）。然而，这个简单的马尔可夫过程并不能通过查看时间$t$的状态 $X_t - X_{t-1}$ 来告诉我们股票价格 $X_t$ 的值。在这个应用中，我们不仅关心马尔可夫状态转移概率，还关心从时间 $t$ 的状态中获取任意时间 $t$ 的股票价格信息。因此，我们将状态建模为对 $( X_t, X_{t-1} )$。

   请注意，如果我们将状态 $S_t$ 建模为整个股票价格历史 $( X_0, X_1,..., X_t )，$那么马尔可夫性质将显然得到满足，将$S_t$建模为对$(X_t,X_{t-1})$Markov性质也会得到满足。然而，我们选择 $S_t := (X_t, X_t - X_{t-1})$ 是“最简单”的内部表示。实际上，在整本书中，我们对各种过程建模状态的努力是确保马尔可夫性质，同时使用“最简单/最小”的状态表示。

3. Process3是Process2的扩展，其中下一个移动的概率不仅依赖于上一时刻的移动还依赖于过去所有的移动。具体来说，它依赖于过去上涨次数的数量记为$U_t=\sum_{i=1}^t\max(X_i-X_{i-1},0)$,与过去下跌次数的数量，记为$D_t=\sum_{i=1}^t\max(X_{i-1}-X_i,0)$之间的关系。表示为
   $$
   P[X_{t+1}=X_t+1]=\begin{cases}
   \frac{1}{1+(\frac{U_t+D_t}{D_t}-1)^{\alpha_3}}&\text{ if }t>0\\
   0.5&\text{ if }t=0
   \end{cases}
   $$
   其中$\alpha_3\in\mathbb{R}_{\geq0}$是一个拉力强度参数，将上述概率表达式视为$f(\frac{D_t}{U_t+D_t};\alpha_3)$其中$f:[0,1]\rightarrow[0,1]$是一个sigmoid型函数
   $$
   f(x;\alpha)=\frac{1}{1+(\frac{1}{x}-1)^{\alpha}}.
   $$
   下一个上涨移动的概率基本依赖$\frac{U_t}{U_t+D_t}$即过去时间步中下跌次数的比例。因此，如果历史上的下跌次数大于上涨次数，那么下一个价格移动$X_{t+1}-X_t$将会有更多的向上拉力，反之亦然。

   我们将$S_t$建模为由对$(U_t,D_t)$组成，这样$S_t$的Markov性质可以得到满足
   $$
   \begin{aligned}
   &P[(U_{t+1},D_{t+1})|(U_t,D_t),...,(U_0,D_0)]=P[(U_{t+1},D_{t+1})|(U_t,D_t)]\\
   &=\begin{cases}
   f(\frac{D_t}{U_t+D_t};\alpha_3)&\text{ if }U_{t+1}=U_t+1,D_{t+1}=D_t\\
   f(\frac{U_t}{U_t+D_t};\alpha_3)&\text{ if }U_{t+1}=U_t,D_{t+1}=D_t+1
   \end{cases}
   \end{aligned}
   $$
   重要的是与前面两个过程不同，股票价格$X_t$实际上并不是过程3中状态$S_t$的一部分，这是因为$U_t,D_t$共同包含了捕捉$X_t$的足够信息，因为$X_t=X_0+U_t-D_t.$

### Markov过程的正式定义

书中的定义和定理将由限制在离散时间和可数状态集合。

>**Def 3.3.1**
>
>Markov过程由以下组成
>
>- 一个可数状态集合$\mathcal{S}$（称为状态空间）和一个子集$\mathcal{T}\subset \mathcal{S}$​（称为终止状态集合）。
>- 一个时间索引的随即状态序列$S_t\in S,$时间步为$t=0,1,2,...$,每个状态转移都满足Markov性质:$P[S_{t+1}|S_t,...,S_0]=P[S_{t+1}|S_t],\text{for all }t\geq0.$
>- 终止：如果某个时间步$T$的结果$S_T$是集合$\mathcal{T}$中的一个状态，则该序列的结果在时间步$T$终止。
>
>将$P[S_{t+1}|S_t]$称为时间$t$的转移概率。
>
>**Def 3.3.2**
>
>一个时间齐次Markov过程是一个Markov过程且$P[S_{t+1}|S_t]$与$t$无关。

这意味着时间齐次Markov过程的动态可以通过下面的函数完全指定：
$$
P:(\mathcal{S}-\mathcal{T})\times\mathcal{S}\rightarrow[0,1]
$$
定义为$P(s',s)=P[S_{t+1}=s'|S_t=s]$使得$\sum_{s'\in S}P(s,s')=1,\text{for all}s\in\mathcal{S-T}.$​

注意上述规范中$P$的参数没有时间索引$t$（因此称为时间齐次）。此外注意到一个非时间齐次的Markov过程可以通过将所有状态和时间索引$t$来结合转换为齐次Markov过程。这意味着如果一个非时间齐次的Markov过程的原始状态空间是$\mathcal{S}$，那么对应的时间齐次Markov过程的状态空间是$\mathbb{Z}_{\geq0}\times\mathcal{S}.$

### Markov过程的稳态分布
>**Def 3.7.1**
>对于状态空间$\mathcal{S}=\mathbb{N}$的离散、时间齐次的Markov过程及其转移概率函数$P:\mathbb{N}\times\mathbb{N}\rightarrow [0,1]$,稳态分布是一个概率分布函数$\pi:\mathbb{N}\rightarrow [0,1]$,满足
>$$
>\pi(s')=\sum_{s\in\mathbb{N}}\pi(s)\cdot P(s,s'),\text{ for all }s'\in\mathbb{N}
>$$

稳态分布$\pi$的直观理解是，在特定条件下如果我们让Markov过程无限运行，那么在长期内，状态在特定步出现频率（概率）由分布$\pi$给出，该分布与时间步无关。

如果将稳态分布的定义专门化为有限状态、离散时间、时间齐次的Markov过程，状态空间为$S=\{s_1,...,s_2\}=\mathbb{N},$那么我们可以将稳态分布$\pi$​表示为
$$
\pi(s_j)=\sum_{i=1}^n\pi(s_i)\cdot P(s_i,s_j),\text{ for al }j=1,2,...,n
$$
下面使用粗体符号表示向量和矩阵。故$\boldsymbol{\pi}$是一个长度为$n$的列向量，$\boldsymbol{\mathcal{P}}$是$n\times n$的转移概率矩阵，其中行是原状态，列为目标状态，每行的和为1。那么上述定义的表述就可以简洁地表示为：
$$
\boldsymbol{\pi}^T=\boldsymbol{\pi}^T\cdot\boldsymbol{\mathcal{P}},\text{ or }\boldsymbol{\mathcal{P}}^T\cdot\boldsymbol{\pi}=\boldsymbol{\pi}
$$
后一个式子可以说明$\boldsymbol{\pi}$是矩阵$\boldsymbol{\mathcal{P}}$​的特征值为1对应的特征向量。

### Markov奖励过程的形式主义

我们之所以讲述Markov过程是因为希望通过为Markov过程添加增量特性来逐步进入Markov决策过程，也就是强化学习的算法框架。现在开始讲述介于二者之间的中间框架即Markov奖励过程。基本上我们只是为每次从一个状态转移到下一个状态时引入一个数值奖励的概念。这些奖励是随机的，我们需要做的就是在进行状态转移时指定这些奖励的概率分布。

Markov奖励过程的主要目的是计算如果让过程无限运行（期望从每个非终止状态获得的奖励总和）我们将累积多少奖励，考虑到未来的奖励需要适当地折现。

>**Def 3.8.1**
>
>Markov奖励过程是一个Markov过程以及一个时间索引序列的奖励随机变量$R_t\in\mathcal{D},\mathcal{D}$是$\mathbb{R}$中一个可数子集，$t=1,2,...,$满足Markov性质：
>$$
>P[(R_{t+1},S_{t+1})|S_{t},S_{t-1},...,S_0]=P[(R_{t+1},S_{t+1})|S_t]\text{ for all }t\geq0
>$$

我们将$P[(R_{t+1},S_{t+1})|S_t]$称为Markov Reward Process在时间$t$地转移概率。由于我们通常假设Markov的时间齐次性，我们将假设MRP具有时间齐次性，即$P[(R_{t+1},S_{t+1})|S_t]$与$t$​无关。

由时间齐次性的假设，MRP的转移概率可以表示为转移概率函数
$$
\mathcal{P}_R:\mathcal{N\times D\times S}\rightarrow[0,1]
$$
定义为  
$$
\begin{aligned}
&\mathcal{P}_R(s,r,s')=P[(R_{t+1}=r,S_{t+1}=s')|S_t=s]\text{ for }t=0,1,2,...,\\
&\text{for all }s\in\mathcal{N},r\in\mathcal{D},s'\in\mathcal{S},\text{ s.t. }\sum_{s'\in\mathcal{S}}\sum_{r\in\mathcal{D}}\mathcal{P}_R(s,r,s')=1,\text{ for all }s\in \mathcal{N} 

\end{aligned}
$$
当涉及模拟时我们需要单独指定起始状态的概率分布。

现在可以扩展更多理论。给定奖励转移函数$\mathcal{P}_R$，我们可以得到

- 隐式Markov过程的概率转移函数$P:\mathbb{N}\times S\rightarrow [0,1]$可以定义为  
  $$
  \mathcal{P}(s,s')=\sum_{r\in\mathcal{D}}\mathcal{P}_R(s,r,s')
  $$

- 奖励转移函数$\mathcal{R}_T:\mathcal{N\times S}\rightarrow \mathbb{R}$定义为  
  $$
  \mathcal{R}_T(s,s')=\mathbb{E}[R_{t+1}|S_{t+1}=s',S_t=s]=\sum_{r\in\mathcal{D}}\frac{\mathcal{P}_R(s,r,s')}{\mathcal{P}(s,s')}=\sum_{r\in\mathcal{D}}\frac{\mathcal{P}_R(s,r,s')}{\sum_{r\in\mathcal{D}}\mathcal{P}_R(s,r,s')}\cdot r
  $$

我们在实践中遇到的大多数MRP奖励规范可以直接表示为奖励转移函数$\mathcal{R}_T$.最后我们想强调的是，可以将$\mathcal{P}_R$或$\mathcal{R}_T$转换为一种更紧凑的奖励函数。该函数足以执行涉及MRP的关键计算，这个奖励函数$\mathcal{R}:\mathcal{N}\rightarrow \mathbb{R}$定义为  
$$
\mathcal{R}(s)=\mathbb{E}[R_{t+1}|S_t=s]=\sum_{s'\in\mathcal{S}}\mathcal{P}(s,s')\cdot\mathcal{R}_T(s,s')=\sum_{s'\in\mathcal{S}}\sum_{r\in\mathcal{D}}\mathcal{P}_R(s,r,s')\cdot r
$$

### Markov奖励过程的价值函数  
现在，我们准备正式定义涉及MRP的主要问题，我们希望计算从任何非终止状态出发的“期望累积奖励”。允许在奖励累积时使用贴现因子$\gamma\in[0,1]$,我们将回报$G_t$定义为时间$t$之后的“未来奖励的贴现累积”。形式上：  
$$
G_t=\sum_{i=t+1}^\infty \gamma^{i-t-1}\cdot R_i=R_{t+1}+\gamma \cdot R_{t+2}+\gamma^2\cdot R_{t+3}+....
$$
即使对于终止序列（例如$t=T$时终止，即$S_T\in\mathcal{T}$）我们只需将$i>T$的$R_i=0$.  
我们希望识别具有较大期望回报的非终止状态和具有较小期望回报的非终止状态。事实上，这是涉及MRP的主要问题——计算MRP中每个非终止状态的期望回报。形式上，我们感兴趣的是计算价值函数：  
$$
V:\mathcal{N}\rightarrow\mathbb{R}
$$
定义为  
$$
V(s)=\mathbb{E}[G_t|S_t=s]\text{ for all }s\in\mathcal{N},\text{ for all }t=0,1,2,...
$$
贝尔曼指出价值函数具有递归结构，具体来说  

\begin{align}
V(s)=&\mathbb{E}[R_{t+1}|S_t=s]+\gamma\cdot\mathbb{E}[R_{t+2}|S_t=s]+...\\
=&\mathcal{R}(s)+\gamma\cdot\sum_{s'\in\mathcal{N}}P[S_{t+1}=s'|S_t=s]\cdot\mathbb{E}[R_{t+2}|S_{t+1}=s']\\
&+\gamma^2\sum_{s'\in\mathcal{N}}P[S_{t+1}=s'|S_t=s]\sum_{s''\in\mathcal{N}}P[S_{t+2}=s''|S_{t+1}=s']\cdot\mathbb{E}[R_{t+3}|S_{t+2}=s'']\\
&+...\\
=&\mathcal{R}(s)+\gamma\cdot\sum_{s'\in\mathcal{N}}\mathcal{P}(s,s')\cdot\mathcal{R}(s')+\gamma^2\sum_{s'\in\mathcal{N}}\mathcal{P}(s,s')\sum_{s''\in\mathcal{N}}\mathcal{P}(s',s'')\mathcal{R}(s'')+...\\
=&\mathcal{R}(s)+\gamma\cdot\sum_{s'\in\mathcal{N}}\mathcal{P}(s,s')\cdot(R(s')+\gamma\cdot\sum_{s''\in\mathcal{N}}\mathcal{P}(s',s'')\cdot\mathcal{R}(s'')+...)\\
=&\mathcal{R}(s)+\gamma\cdot\sum_{s'\in\mathcal{N}}\mathcal{P}(s,s')\cdot V(s') \text{ for all }s\in\mathcal{N} (\#eq:bellman)
\end{align}


我们将这个价值函数的递归方程@ref(eq:bellman)称为**Markov奖励过程的Bellman方程**。


## Markov决策过程 {#MDP}
### 不确定性下的序列决策难题  
通常，MDP具有两个截然不同且相互依赖的高级特征  
1. 在每个时间步$t,$观察到状态$S_t$后，从指定的动作集合中选择一个动作$A_t.$
2. 给定观察到的状态$S_t$和执行的动作$A_t,$下一个时间步的状态$S_{t+1}$和奖励$R_{t+1}$的概率通常不仅取决于状态$S_t$还取决于动作$A_t.$  
我们的任务是最大化每个状态的期望回报（即最大化价值函数）。在一般情况下，这似乎是一个非常困难的问题，因为存在循环的相互作用。一方面，动作依赖于状态；另一方面，下一个状态/奖励的概率依赖于动作和状态。此外，动作可能会对奖励产生延迟影响，如何区分不同时间步的动作对未来奖励的影响也是一个挑战。如果没有动作和奖励之间的直接对应关系，我们如何控制动作以最大化期望累积奖励？为了回答这个问题，我们需要建立一些符号和理论。在我们正式定义马尔可夫决策过程框架及其相关（优雅的）理论之前，让我们先设定一些术语。

**人工智能视角下的MDP**  
使用人工智能的语言，我们说在每个时间步$t,$**智能体（Agent）**（我们设计的算法）观察到状态$S_t,$然后智能体执行动作$A_t,$之后环境（Environment）（在看到$S_t,A_t$后）生成一个随机对$(S_{t+1},R_{t+1})$。接着，智能体观察到下一个状态$S_{t+1}$，循环重复直到达到终止状态。这种循环相互作用如图\@ref(fig:mdp)所示


``` r
knitr::include_graphics("https://Go9entle.github.io/picx-images-hosting/image.3yekzoaheh.webp")
```

<div class="figure">
<img src="https://Go9entle.github.io/picx-images-hosting/image.3yekzoaheh.webp" alt="Markov Decision Process"  />
<p class="caption">(\#fig:mdp)Markov Decision Process</p>
</div>


### Markov决策过程的正式定义  
与Markov过程和Markov奖励过程的定义类似，为了便于阐述，以下Markov决策过程的定义和理论将针对离散时间、可数状态空间和可数的下一个状态与奖励转移对。  

**定理**

Markov决策过程包括以下内容：  

- 可数状态集合$\mathcal{S}$（称为状态空间），终止状态集合$\mathcal{T}\subset \mathcal{S}$,以及可数动作集合$\mathcal{A}$（称为动作空间）。
- 时间索引的环境生成随机状态序列$S_t\in\mathcal{S}$（时间步$t=0,1,2,...$），时间索引的环境生成奖励随机变量序列$R_t\in\mathcal{D}$（$\mathcal{D}$是$\mathbb{R}$的可数子集），以及时间索引的智能体可控动作序列$A_t\in\mathcal{A}$  

- Markov性  
  \begin{equation*}
  P[(R_{t+1},S_{t+1})|(S_t,A_t,S_{t-1},A_{t-1},...,S_0,A_0)]=P[   (R_{t+1},S_{t+1})|(S_t,A_t)]\text{ for all }t\geq 0
  \end{equation*}
  
- 终止：如果某个时间步$T$的状态$S_T\in\mathcal{T}$,则该序列结果在时间步$T$终止。



在更一般的情况下，如果状态或奖励是不可数的，相同的概念仍然适用，只是数学形式需要更加详细和谨慎。具体来说，我们将使用积分代替求和，使用概率密度函数（用于连续概率分布）代替概率质量函数（用于离散概率分布）。为了符号的简洁性，更重要的是为了核心概念的理解（而不被繁重的数学形式分散注意力），我们选择默认使用离散时间、可数$\mathcal{S}$、可数$\mathcal{A}$和可数$\mathcal{D}$.  

我们将$P[(R_{t+1},S_{t+1})|(S_t,A_t)]$称为Markov决策过程在时间$t$的转移概率。

与Markov过程和MRP一样，我们默认Markov决策过程是时间其次的，即$P[(R_{t+1},S_{t+1})|(S_t,A_t)]$与$t$无关。这意味着Markov决策过程的转移概率在最一般情况下可以表示为**状态-奖励转移概率函数**：   

$$
\mathcal{P}_R:\mathcal{N\times A\times D\times S}\rightarrow [0,1]
$$
定义为  

$$
\mathcal{P}_R(s,a,r,s')=P[(R_{t+1}=r,S_{t+1}=s')|(S_t=s,A_t=a)]
$$
对于时间步$t=0,1,2,...$，对于所有的$s,s'\in\mathcal{N},a\in\mathcal{A},r\in\mathcal{D}$满足  

$$
\sum_{s'\in\mathcal{S}}\sum_{r\in\mathcal{D}}\mathcal{P}_R(s,a,r,s')=1\text{ for all }s\in\mathcal{N},a\in\mathcal{A}
$$
这又可以通过状态-奖励转移概率函数$\mathcal{P}_R$来表征，给定$\mathcal{P}_R$的规范，我们可以构造  

- 状态转移概率函数：  
  $$
    \mathcal{P}:\mathcal{N\times A\times S}\rightarrow [0,1]
  $$
  定义为  
  $$
  \mathcal{P}(s,a,s')=\sum_{r\in\mathcal{D}}\mathcal{P}_R(s,a,r,s')
  $$
  
- 奖励转移函数：  
  $$
  \mathcal{R}_T:\mathcal{N\times A\times S}\rightarrow \mathbb{R}
  $$
  定义为   

  \begin{align}
  \mathcal{R}_T(s,a,s')&=\mathbb{E}[R_{t+1}|(S_{t+1}=s',S_t=s,A_t=a)]\\
  &=\sum_{r\in\mathcal{D}}\frac{\mathcal{P}_R(s,a,r,s')}{\mathcal{P}(s,a,s')}\cdot r\\
  &=\sum_{r\in\mathcal{D}}\frac{\mathcal{P}_R(s,a,r,s')}{\sum_{r\in\mathcal{D}}\mathcal{P}_R(s,a,r,s')}\cdot r
  
  \end{align}

  
在实践中，我们遇到的大多数Markov决策过程的奖励规范可以直接表示为奖励转移函数$\mathcal{R}_T$而不是更一般的$\mathcal{P}_R$.最后我们想强调的是可以将$\mathcal{P}_R$或$\mathcal{R}_T$转换为“更紧凑”的奖励函数，该函数足以执行设计MDP的关键计算，这个奖励函数为：

$$
\mathcal{R}:\mathcal{N\times A}\rightarrow \mathbb{R}
$$
定义为：  


\begin{aligned}
\mathcal{R}(s,a)&=\mathbb{E}[R_{t+1}|(S_t=s,A_t=a)]\\
&=\sum_{s\in\mathcal{S}}\mathcal{P}(s,a,s')\cdot\mathcal{R}_T(s,a,s')\\
&=\sum_{s\in\mathcal{S}}\sum_{r\in\mathcal{D}}\mathcal{P}_R(s,a,r,s')\cdot r
\end{aligned}


### 策略  
理解了MDP的动态后，我们现在转向智能体动作的规范，即作为当前状态函数的动作选择。在一般情况下，我们假设智能体将根据当i请安状态$S_t$的概率分布执行动作$A_t$,我们将此函数称为策略（Policy）。  
形式上，策略是一个函数：  

$$
\pi:\mathcal{N\times A}\rightarrow [0,1]
$$
定义为:  

$$
\pi(s,a)=P[A_t=a|S_t=s]\text{ for }t=0,1,2,...\text{ for all }s\in\mathcal{N},a\in\mathcal{A}
$$
使得  

$$
\sum_{a\in\mathcal{A}}\pi(s,a)=1\text{ for all }s\in\mathcal{N}
$$

需要注意的是，上述定义假设策略是Markov的，即动作概率仅依赖于当前状态，而不依赖于历史状态。上述定义还假设策略是平稳的，即$P[A_t=a|S_t=s]$在时间$t$上是不变的。如果我们遇到策略需要依赖于时间$t$的情况，我们可以简单地将$t$包含在状态中，从而使策略变得平稳（尽管这会增加状态空间的规模，从而导致计算成本的增加）。  
当策略对每个状态的动作概率集中在单个动作上（即只要到达一个状态，动作是确定的）时，我们称之为**确定性策略**。形式上，确定性策略$\pi_D:\mathcal{N}\rightarrow \mathcal{A}$具有以下性质：对所有的$s\in\mathcal{N},$

$$
\pi(s,\pi_D(s))=1\text{ and }\pi(s,a)=0, \text{ for all }a\ne \pi_D(s)
$$

我们将非确定性的策略称为**随机策略**（随机反映了智能体将根据$\pi$指定的概率分布执行随机动作的事实）。

### [Markov决策过程，策略]:= Markov奖励过程  
本节有一个重要的见解——如果我们用固定策略$\pi$（通常是一个固定的随机策略，注意与确定性策略区分）评估MDP，我们会得到一个由MDP和策略$\pi$共同隐含的MRP。我们可以用符号精确地澄清这一点，但首先MDP和MRP中存在一些符号冲突。我们使用  

- $\mathcal{P}_R$表示MRP转移概率函数，同时也表示MDP的状态-奖励转移概率函数；

- $\mathcal{P}$表示MRP中隐含的Markov过程的转移概率函数，同时也表示MDP的状态转移函数；

- $\mathcal{R}_T$表示MRP的奖励转移函数，同时也表示MDP的奖励转移函数；

- $\mathcal{R}$表示MRP的奖励函数，同时也表示MDP的奖励函数。

我们将在$\pi$隐含的MRP的函数$\mathcal{P}_R,\mathcal{P},\mathcal{R}_T,\mathcal{R}$加上上标$\pi$以区分这些函数MDP和$\pi$隐含的MRP中的使用。

假设我们给定一个固定策略$\pi$和一个由其状态-奖励转移概率函数$\mathcal{P}_R$指定的MDP，那么由MDP与策略$\pi$评估隐含的MRP的转移概率函数$\mathcal{P}_R^\pi$定义为：

$$
\mathcal{P}_R^\pi(s,r,s')=\sum_{a\in\mathcal{A}}\pi(s,a)\cdot\mathcal{P}_R(s,a,r,s')
$$
类似地有  

\begin{align}
\mathcal{P}^\pi(s,s')&=\sum_{a\in\mathcal{A}}\pi(s,a)\cdot\mathcal{P}(s,a,s')\\
\mathcal{R}_T^\pi(s,s')&=\sum_{a\in\mathcal{A}}\pi(s,a)\cdot\mathcal{R}_T(s,a,s')\\
\mathcal{R}^\pi(s)&=\sum_{a\in\mathcal{A}}\pi(s,a)\cdot\mathcal{R}(s,a)
\end{align}

因此，每当我们谈论用固定策略评估的 MDP时，你应该知道我们实际上是在谈论隐含的 MRP。

### 固定策略下的MDP价值函数  
现在我们准备讨论用固定策略$\pi$评估的MDP的价值函数（也称为MDP预测问题，“预测”指的是该问题涉及在智能体遵循特定策略时预测未来期望回报）。与MRP情况类似，我们定义  

$$
G_t=\sum_{i=t+1}^\infty \gamma^{i-t-1}\cdot R_i=R_{t+1}+\gamma \cdot R_{t+2}+\gamma^2\cdot R_{t+3}+....
$$
其中$\gamma\in[0,1]$是指定的贴现因子。即便对于终止序列我们也使用上述回报的定义。  
用固定策略$\pi$评估的MDP的价值函数为  

$$
V^\pi:\mathcal{N}\rightarrow \mathbb{R}
$$
定义为：  

$$
V^\pi(s)=\mathbb{E}_{\pi,\mathcal{P}_R}[G_t|S_t=s]\text{ for all }s\in\mathcal{N},\text{ for all }t=0,1,2,...
$$
我们假设每当我们讨论价值函数时，折扣因子$\gamma$是适当的，以确保每个状态的期望回报是有限的——特别是对于可能发散的连续（非终止）MDP，$\gamma<1$.  
我们将$V^\pi(s)=\mathbb{E}_{\pi,\mathcal{P}_R}[G_t|S_t=s]$展开如下：  

\begin{align}
&\mathbb{E}_{\pi,\mathcal{P}_R}[R_{t+1}|S_t=s]+\gamma\cdot\mathbb{E}_{\pi,\mathcal{P}_R}[R_{t+2}|S_t=s]+\gamma^2\cdot\mathbb{E}_{\pi,\mathcal{P}_R}[R_{t+3}|S_t=s]+...\\
=&\sum_{a\in\mathcal{A}}\pi(s,a)\cdot\mathcal{R}(s,a)+\gamma\cdot\sum_{a\in\mathcal{A}}\pi(s,a)\sum_{s'\in\mathcal{N}}\mathcal{P}(s,a,s')\sum_{a'\in\mathcal{A}}\pi(s',a')\cdot\mathcal{R}(s',a')\\
&+\gamma^2\cdot\sum_{a\in\mathcal{A}}\pi(s,a)\sum_{s'\in\mathcal{N}}\mathcal{P}(s,a',s')\sum_{a'\in\mathcal{A}}\pi(s',a')\sum_{s''\in\mathcal{N}}\mathcal{P}(s',a'',s'')\sum_{a''\in\mathcal{A}}\pi(s'',a'')\cdot\mathcal{R}(s'',a'')\\
&+...\\
=& \mathcal{R}^\pi(s)+\gamma\cdot\sum_{s'\in\mathcal{N}}\mathcal{P}^\pi(s,s')\cdot\mathcal{R}^\pi(s')+\gamma^2\cdot\sum_{s'\in\mathcal{N}}\mathcal{P}^\pi(s,s')\sum_{s''\in\mathcal{N}}\mathcal{P}^\pi(s',s'')\cdot\mathcal{R}^\pi(s'')+...
\end{align}

最后一个表达式等于$\pi$隐含的MRP的状态$s$的价值函数。因此，用固定策略$\pi$评估的MDP的价值函数$V^\pi$与$\pi$隐含的MRP的价值函数完全相同，因此我们可以将MRP的Bellman方程应用于$V^\pi$，即  

\begin{align}
V^\pi(s)=&\mathcal{R}^\pi(s)+\gamma\cdot\sum_{s'\in\mathcal{N}}\mathcal{P}^\pi(s,s')\cdot V^\pi(s')\\
=&\sum_{a\in\mathcal{A}}\pi(s,a)\cdot\mathcal{R}(s,a)+\gamma\cdot\sum_{a\in\mathcal{A}}\pi(s,a)\sum_{s'\in\mathcal{N}}\mathcal{P}(s,a,s')\cdot V^\pi(s')\\
=&\sum_{a\in\mathcal{A}}\pi(s,a)\cdot(\mathcal{R}(s,a)+\gamma\cdot\sum_{s'\in\mathcal{N}}\mathcal{P}(s,a,s')\cdot V^\pi(s'))\text{ for all }s\in\mathcal{N} (\#eq:4-1)
\end{align}

对于状态空间不太大的有限MDP，方程 \@ref(eq:4-1) 可以通过线性代数求解$V^\pi$。更一般地，方程 \@ref(eq:4-1)  将成为本书其余部分开发各种动态规划和强化学习算法以解决MDP预测问题的关键方程。  
然而，另一个价值函数在开发MDP算法时也至关重要——它将（状态、动作）对映射到从该（状态，动作）对出发的期望回报，当用固定策略评估时。这被称为用固定策略评估的MDP地动作—价值函数：  

$$
Q^\pi:\mathcal{N\times A}\rightarrow \mathbb{R}
$$
定义为：  

$$
Q^\pi(s,a)=\mathbb{E}_{\pi,\mathcal{P}_R}[G_t|(S_t=s,A_t=a)]\text{ for all }s\in\mathcal{N},a\in\mathcal{A} \text{ for all }t=0,1,2,...
$$

为了避免术语混淆，我们将$V^\pi$称为策略$\pi$的**状态-价值函数**（尽管通常简称为价值函数），以区别于**动作-价值函数**$Q^\pi$.解释$Q^\pi(s,a)$的方式是，它是从给定非终止状态$s$出发，首先采取动作$a$，然后遵循策略$\pi$的期望回报。通过这种解释，我们可以将$V^\pi(s)$视作$Q^\pi(s,a)$的“加权平均”（对于所有从非终止状态$s$出发的所有可能动作$a$）,权重等于给定状态$s$的动作$a$的概率（即$\pi(s,a)$）。具体来说：  

\begin{equation}
V^\pi(s)=\sum_{a\in\mathcal{A}}\pi(s,a)\cdot Q^\pi(s,a)\text{ for all }s\in\mathcal{N} (\#eq:4-2)
\end{equation}

将$Q^\pi(s,a)$展开后得到  

\begin{equation}
Q^\pi(s,a)=\mathcal{R}(s,a)+\gamma\cdot\sum_{s'\in\mathcal{N}}\mathcal{P}(s,a,s')\cdot V^\pi(s') \text{ for all }s\in\mathcal{N},a\in\mathcal{A} (\#eq:4-3)
\end{equation}

结合方程\@ref(eq:4-2)和方程\@ref(eq:4-3)我们得到   

\begin{equation}
Q^\pi(s,a)=\mathcal{R}(s,a)+\gamma \cdot\sum_{s'\in\mathcal{N}}\mathcal{P}(s,a,s')\sum_{a'\in\mathcal{A}}\pi(s',a')\cdot Q^\pi(s',a')\text{ for all }s\in\mathcal{N},a\in\mathcal{A} (\#eq:4-4)
\end{equation}

方程\@ref(eq:4-1)被称为**MDP状态-价值函数贝尔曼策略方程**，方程\@ref(eq:4-4)被称为**MDP动作-价值函数贝尔曼策略方程**。方程\@ref(eq:4-1)、\@ref(eq:4-2)、\@ref(eq:4-3)和\@ref(eq:4-4)统称为**MDP贝尔曼策略方程**。

### 最优价值函数和最优策略

最后，我们要达到Markov决策问题的主要目的——识别能够产生最有价值函数的策略，即从每个非终止状态出发的最佳可能期望回报。我们说，当我们识别出MDP的最优价值函数（及相关的最优策略，即产生最优价值函数的策略）时，MDP就被“解决”了。识别最优价值函数及其相关最优策略被称为**MDP控制问题**。“控制”指的是该问题涉及通过策略的迭代修改来引导动作，以推动价值函数向最优性发展。  
形式上，最优价值函数定义为  

$$
V^*:\mathcal{N}\rightarrow \mathbb{R}
$$
定义为  

$$
V^*(s)=\max_{\pi\in\Pi} V^\pi(s) \text{ for all }s\in\mathcal{N}
$$
其中$\Pi$是$\mathcal{N,A}$空间上的平稳随机策略集合。  
上述定义的解释是，对于每个非终止状态$s$，我们考虑所有可能的随机平稳策略并在这些$\pi$中选择最大化$V^\pi(s).$需要注意的是，$\pi$的选择是针对每个$s$单独进行的，因此可以想象，不同的$\pi$的选择可能会为不同的$s\in\mathcal{N}$最大化$V^\pi(s).$因此，从上述$V^*$的定义中，我们还不能谈论“最优策略”的概念。因此，现在让我们只关注上述定义的最有价值函数。  
同样，最优动作-价值函数定义为  

$$
Q^*:\mathcal{N\times A}\rightarrow \mathbb{R}
$$
定义为  

$$
Q^*(s,a)=\max_{\pi\in\Pi} Q^\pi(s,a) \text{ for all }s\in\mathcal{N},a\in\mathcal{A}
$$
$V^*$通常被称为最优状态-价值函数，以区别于最优动作-价值函数$Q^*$（尽管为了简洁，$V^*$通常也被简称为最优价值函数）。需要明确的是，最优价值函数默认情况下值得就是最优状态-价值函数$V^*$.  

正如固定策略的价值函数具有递归公式一样，贝尔曼指出我们可以为最优价值函数创建递归公式。让我们从展开给定非终止状态$s$的最优状态-价值函数$V^*(s)$开始——我们考虑从状态$s$出发可以采取的所有可能动作$a\in\mathcal{A},$并选择能够产生最佳动作-价值的动作$a,$即选择出能够产生最优$Q^*(s,a)$的动作$a$。形式上给出了以下方程  

$$
V^*(s)=\max_{a\in\mathcal{A}}Q^*(s,a)\text{ for all }s\in\mathcal{N} (\#eq:4-5)
$$
同样让我们思考从给定非终止状态和动作对$(s,a)$出发的最优性意味着什么，也就是展开$Q^*(s,a)$.首先我们获得即时的期望奖励$\mathcal{R}(s,a)$.接下来，考虑所有可能的我们可以转到的状态$s'\in\mathcal{S}$并从每个非终止状态$s'$出发递归地采取最优动作。形式上，这给出了以下方程：  

$$
Q^*(s,a)=\mathcal{R}(s,a)+\gamma\cdot\sum_{s'\in\mathcal{N}}\mathcal{P}(s,a,s')\cdot V^*(s')\text{ for all }s\in\mathcal{N},a\in\mathcal{A} (\#eq:4-6)
$$
将\@ref(eq:4-6)中的$Q^*(s,a)$代入\@ref(eq:4-5),可以得到  

$$
V^*(s)=\max_{a\in\mathcal{A}}\{ \mathcal{R}(s,a)+\gamma\cdot\sum_{s'\in\mathcal{N}}\mathcal{P}(s,a,s')\cdot V^*(s') \} \text{ for all }s\in\mathcal{N} (\#eq:4-7)
$$
方程\@ref(eq:4-7)被称为**MDP状态-价值函数贝尔曼最优性方程**。  
将方程\@ref(eq:4-5)代入方程\@ref(eq:4-6)可以得到  

$$
Q^*(s,a)=\mathcal{R}(s,a)=\gamma\cdot\sum_{s'\in\mathcal{N}}\mathcal{P}(s,a,s')\cdot \max_{a'\in\mathcal{A}} Q^*(s',a')\text{ for all }s\in\mathcal{A},a\in\mathcal{A} (\#eq:4-8)
$$
方程\@ref(eq:4-8)被称为**MDP动作-价值函数贝尔曼最优性方程**。  
方程\@ref(eq:4-7)、\@ref(eq:4-5)、\@ref(eq:4-6)和\@ref(eq:4-8)统称为**MDP贝尔曼最优性方程**。我们应该强调，当有人说 MDP贝尔曼方程或简称为贝尔曼方程时，除非他们明确说明，否则他们指的是 MDP 贝尔曼最优性方程（通常是 MDP 状态-价值函数贝尔曼最优性方程）。这是因为 MDP 贝尔曼最优性方程解决了马尔可夫决策过程的最终目的——识别最优价值函数和实现最优价值函数的相关策略（即使我们能够解决 MDP 控制问题）。  
我们需要强调的是，贝尔曼最优性方程并没有直接给出计算最优质函数或实现最优质函数的策略的具体方法——它们只是阐述了最优值函数的一个强大数学性质，这一性质帮助我们提出动态规划或者强化学习的算法来计算最优值函数及其相关的策略。  

我们一直在使用“实现最优值函数的策略/策略组合”这个词，但我们还没有给出这样的策略的明确定义。事实上，正如之前提到的，从$V^*$的定义来看并不清楚是否存在这样的策略能实现$V^*$（因为可以设想不同的策略$\pi$对于不同的状态$s\in\mathcal{N}$实现$V^\pi(s)$最大化）。因此我们定义最优策略$\pi^*:\mathcal{N\times A}\rightarrow [0,1]$主导所有其他策略的策略，在价值函数上优于所有其他策略。形式化地说  

$$
\pi^*\in\Pi\text{ is an Optimal Policy if } V^{\pi^*}(s)\geq V^\pi(s)\text{ for all }\pi\in\Pi \text{ and for all states }s\in\mathcal{N}. 
$$

最优策略$\pi^*$的定义表明，它是一个“优于或等于”所有其他静态策略的策略，且适用于所有非终止状态（注意可能存在多个最优策略）。将这个定义与最优值函数$V^*$的定义结合，接下来的自然问题是：是否存在一个最优策略$\pi^*$,对所有的$s\in\mathcal{N}$最大化$V^\pi(s)$,也就是是否存在一个$\pi^*$使得$V^*(s)=V^{\pi^*}(s)$对所有$s\in\mathcal{N}.$下面的定理和证明是针对我们默认的MDP设置（离散时间、可数空间、时间齐次）的。

**定理**  

对于任何（离散时间、可数空间、时间齐次）的MDP：

- 存在一个最优策略$\pi^*\in\Pi$.

- 所有最优策略都实现最优值函数。

- 所有最优策略实现最优动作-价值函数，即对于所有的$s\in\mathcal{N},a\in\mathcal{A},Q^{\pi^*}(s,a)=Q^*(s,a)$对于所有的最优策略$\pi^*$.


定理的证明略去。

我们的确定性如下定义  

$$
\pi^*_D(s)=\mathop{\arg\max}_{a\in\mathcal{A}} Q^*(s,a)\text{ for all }s\in\mathcal{N} (\#eq:4-9)
$$

方程\@ref(eq:4-9)是一个关键构造，它与贝尔曼最优性方程紧密结合，在涉及各种动态规划和强化学习算法以解决MDP控制问题（即求解$V^*,Q^*,\pi^*$）时起到重要作用。最后，值得注意的是，不像预测问题在小状态空间中有一个直接的线性代数求解器，控制问题是非线性的，因此没有类似的直接线性代数求解器。控制问题的最简单解法（即使是在小状态空间中）就是我们将在下一章中讨论的动态规划算法。

## 动态规划算法 {#dynamicprogram}  

### Planning vs Learning 

在本书中，我们将从人工智能的角度探讨预测与控制（并且我们将特别使用人工智能的术语）。我们将区分没有马尔可夫决策过程（MDP）环境模型的算法（没有访问概率转移（$\mathcal{P}_R$）函数）与有马尔可夫决策过程环境模型的算法（意味着我们可以通过显式的概率分布表示或仅通过采样模型访问$\mathcal{P}_R$）。前者（没有模型访问的算法）被称为学习算法，以反映人工智能代理需要与真实世界环境互动（例如，一个机器人学习在实际森林中导航）并从其通过与环境互动获得的数据（遇到的状态、采取的行动、观察到的奖励）中学习价值函数的事实。后者（有MDP环境模型的算法）被称为规划算法，以反映人工智能代理不需要与真实世界环境互动，实际上是通过模型预测各种行动选择的未来状态/奖励的概率场景，并基于预测的结果求解所需的价值函数。在学习和规划中，贝尔曼方程是驱动算法的基本概念，但算法的细节通常会使它们看起来相当不同。本章将仅关注**规划算法**，实际上将只关注一种称为*动态规划*的规划算法子类。

### 不动点理论

**定义**  
一个函数$f:\mathcal{X\rightarrow X}$的不动点是指一个$x\in\mathcal{X},$使得满足方程$x=f(x).$


### 贝尔曼策略算子以及策略评估算法

本节介绍第一个动态规划算法即*策略评估算法*。策略评估算法解决了在固定策略$\pi$下计算有限Markov决策过程（MDP）的价值函数的问题（即有限MDP的预测问题）。我们知道这等价于计算$\pi$隐式的有限Markov奖励过程（MRP）的价值函数。为了避免符号混淆，注意对符号的$\pi$上标表示它是指$\pi$隐式MRP的符号。预测问题的具体定义如下

设MDP（以及$\pi$隐式的MRP）的状态集为$\mathcal{S}=\{s_1,...,s_n\}$,不失一般性，设$\mathcal{N}=(s_1,...,s_m)$为非终止状态。我们给定一个固定策略$\pi:\mathcal{N\times A}\rightarrow [0,1]$.我们还给定$\pi$隐式的MRP转移概率函数：

$$
\mathcal{P}_R^\pi:\mathcal{N\times D\times S}\rightarrow [0,1]
$$
该函数以数据结构的形式提供（因为状态是有限的，且每个非终止状的下一状态和奖励转移对也是有限的）。预测问题就是计算在固定策略$\pi$下评估的MDP的价值函数（等价于$\pi$隐式MRP的价值函数），我们用$V^\pi:\mathcal{N}\rightarrow \mathbb{R}$来表示。

根据前面的内容，通过从$\mathcal{P}_R^\pi$中提取隐式Markov过程的转移概率函数$\mathcal{P}^\pi:\mathcal{N\times S}\rightarrow [0,1]$和奖励函数$\mathcal{R}^\pi:\mathcal{N}\rightarrow \mathbb{R},$我们可以对价值函数$V^\pi:\mathcal{N}\rightarrow \mathbb{R}$(表示为列向量$\boldsymbol{V}^\pi\in\mathbb{R}^m$)执行以下计算来求解这个预测问题：

$$
\boldsymbol{V}^\pi=(\boldsymbol{I}_m-\gamma \boldsymbol{\mathcal{P}}^\pi)^{-1}\cdot \boldsymbol{\mathcal{R}}^\pi
$$
其中$\boldsymbol{I}_m$是$m$阶的单位矩阵，列向量$\boldsymbol{\mathcal{R}}^\pi\in\mathbb{R}^m$表示$\mathcal{R}^\pi,\boldsymbol{\mathcal{P}}^\pi$是一个$m$阶的矩阵代表$\mathcal{P}^\pi$（其中的行和列对应非终止状态）。然而当$m$很大时这种计算方式不能很好地扩展，因此我们需要寻找一个数值算法来解这个MRP贝尔曼方程

$$
\boldsymbol{V}^\pi=\boldsymbol{\mathcal{R}}^\pi+\gamma \boldsymbol{\mathcal{P}}^\pi\cdot \boldsymbol{V}^\pi
$$
我们定义**贝尔曼策略算子**$\boldsymbol{B}^\pi:\mathbb{R}^m\rightarrow \mathbb{R}^m$为

$$
\boldsymbol{B}^\pi(\boldsymbol{V})=\boldsymbol{\mathcal{R}}^\pi+\gamma \boldsymbol{\mathcal{P}}^\pi\cdot \boldsymbol{V}^\pi
\text{ for any vector }\boldsymbol{V} \text{ in the vector space }\mathbb{R}^m (\#eq:5-1) 
$$
因此，MRP贝尔曼方程就可以表示为

$$
\boldsymbol{V}^\pi=\boldsymbol{B}^\pi(\boldsymbol{V}^\pi)
$$
这意味着$\boldsymbol{V}^\pi$是贝尔曼策略算子$\boldsymbol{B}^\pi$地一个不动点！注意，贝尔曼策略算子可以推广到非有限MDP地情况，并且$\boldsymbol{V}^\pi$仍然是各种感兴趣的推广的不动点。然而，由于本章重点是开发有限MDP算法因此仍然使用上述狭义的（方程\@ref(eq:5-1)）定义。此外，为了证明本章基于不动点的动态规划算法的正确性，我们假设折扣因子$\gamma<1$.

我们希望提出一种度量使得$\boldsymbol{B}^\pi$是一个压缩映射从而能够利用Banach不动点定理，通过反复应用贝尔曼策略算子$\boldsymbol{B}^\pi$来解决这个预测问题。对于任何值函数$\boldsymbol{V}\in\mathbb{R}^m$（这表示$V:\mathcal{N}\rightarrow \mathbb{R}$），我们将表达任何状态$s\in\mathcal{N}$的值为$\boldsymbol{V}(s)$.

我们的度量$d:\mathbb{R}^m\times \mathbb{R}^m\rightarrow\mathbb{R} $将是$L^\infty$范数，定义为：

$$
d(\boldsymbol{X},\boldsymbol{Y})=\Vert\boldsymbol{X}-\boldsymbol{Y} \Vert_{\infty}=\max_{s\in\mathcal{N}}\vert(\boldsymbol{X}-\boldsymbol{Y})(s)\vert
$$
$\boldsymbol{B}^\pi$是在无穷范数下的压缩映射，这是因为对于所有的$\boldsymbol{X},\boldsymbol{Y}\in\mathbb{R}^m,$我们有

$$
\max_{s\in\mathcal{N}}\vert \boldsymbol{B}^\pi(\boldsymbol{X})-\boldsymbol{B}^\pi(\boldsymbol{Y})(s)\vert=\gamma\cdot\max_{s\in\mathcal{N}}\vert(\boldsymbol{P}^\pi\cdot(\boldsymbol{X}-\boldsymbol{Y})(s))\vert\leq \gamma  \cdot\max_{s\in\mathcal{N}}\vert(\boldsymbol{X}-\boldsymbol{Y})(s)\vert
$$
因此调用Banach不动点定理就证明了下面的定理

**定理（策略评估收敛定理）**：  
对于一个有限的MDP，若$\vert\mathcal{N}\vert=m,\gamma<1,$如果$\boldsymbol{V}^\pi\in\mathbb{R}^m$是在固定策略$\pi:\mathcal{N\times A}\rightarrow [0,1]$下评估的价值函数，则$\boldsymbol{V}^\pi$是贝尔曼策略算子$\boldsymbol{B}^\pi$的唯一不动点，并且：

$$
\lim_{i\rightarrow \infty}(\boldsymbol{B}^\pi)^i(\boldsymbol{V}_0)\rightarrow \text{ for all starting Value Functions }\boldsymbol{V}_0\in\mathbb{R}^m
$$


这给我们提供了以下的迭代算法（称为固定策略$\pi$下的策略评估算法）：

- 从任意$\boldsymbol{V}_0\in\mathbb{R}^m$开始

- 对于每次迭代$i=0,1,...,$计算：  
    $$
    \boldsymbol{V}_{i+1}=\boldsymbol{B}^\pi(\boldsymbol{V}_{i})=\boldsymbol{\mathcal{R}}^\pi+\gamma \boldsymbol{\mathcal{P}}^\pi\cdot \boldsymbol{V}_i
    $$

- 当$d(\boldsymbol{V}_{i},\boldsymbol{V}_{i+1})=\max_{s\in\mathcal{N}}$\vert (\boldsymbol{V}_i-\boldsymbol{V}_{i+1})(s) \vert$足够小时停止算法。


请注意，尽管我们将贝尔曼策略算子$\boldsymbol{B}^\pi$定义为作用于$\pi$隐式的MRP值函数，但我们也可将其看作作用于MDP的值函数。为了支持MDP的视角，我们将方程\@ref(eq:5-1)重新表达为MDP转移/奖励规范如下所示

$$
\boldsymbol{B}^\pi(s)=\sum_{a\in\mathcal{A}}\pi(s,a)\cdot\mathcal{R}(s,a)+\gamma\sum_{a\in\mathcal{A}} \pi(s,a)\sum_{s'\in\mathcal{N}}\mathcal{P}(s,a,s')\cdot \boldsymbol{V}(s')\text{ for all }s\in\mathcal{N} (\#eq:5-2)
$$
如果给定MRP的非终止状态数$m,$则每次迭代的运行时间为$O(m^2).$注意，要从给定的MDP和给定的策略构建MRP需要$O(m^2\cdot k)$次运算，其中$k=|\mathcal{A}|.$

### 贪心策略

我们之前提到过要展示三种动态规划算法。第一种策略评估如上一节所见。解决了MDP预测问题。接下来两节中介绍的两种将会解决MDP控制问题，本节是从预测到控制的一个过渡，在这一节中，我们定义了一个函数，该函数通过“贪心”技术来改进值函数或者策略。形式上，贪心策略函数

$$
G:\mathbb{R}^m\rightarrow (\mathcal{N\rightarrow A})
$$
将一个值函数$\boldsymbol{V}$(表示为向量)映射到一个确定性策略$\pi':$

$$
G(\boldsymbol{V})(s):\pi_D'(s)=\mathop{\arg\max}_{a\in\mathcal{A}}\{\mathcal{R}(s,a)+\gamma\cdot\sum_{s'\in\mathcal{N}}\mathcal{P}(s,a,s')\cdot\boldsymbol{V}(s')  \} \text{ for all }s\in\mathcal{N} (\#eq:5-3)
$$
请注意，对于任何特定的$s,$如果两个或多个动作$a$实现了$\mathcal{R}(s,a)+\gamma\cdot\sum_{s'\in\mathcal{N}}\mathcal{P}(s,a,s')\cdot\boldsymbol{V}(s')$的最大化,那我们将使用任意一个动作打破平局并分配一个单一的动作$a$来作为上述$\arg\max$操作的输出。下面使用一个等效的表达式(来指导代码)

$$
G(\boldsymbol{V})(s)=\mathop{\arg\max}_{a\in\mathcal{A}}\left\{\sum_{s'\in\mathcal{S}}\sum_{r\in\mathcal{D}}\mathcal{P}_R(s,a,r,s')\cdot(r+\gamma\cdot \boldsymbol{W}(s'))\right\}\text{ for all }s\in\mathcal{N} (\#eq:5-4)
$$
其中$\boldsymbol{W}\in\mathbb{R}^n$定义为


\begin{equation}
\boldsymbol{W}(s)=\begin{cases}
\boldsymbol{V}(s')&\text{ if }s'\in\mathcal{N}\\
0&\text{ if }s'\in\mathcal{T=S-N}
\end{cases}
\end{equation}

注意在方程\@ref(eq:5-4)中必须使用$\mathcal{P}_R$，我们需要考虑到所有状态$s'\in\mathcal{S}$的转移而不是方程\@ref(eq:5-3)中$s'\in\mathcal{N}$的状态。因此，我们需要小心处理$s'\in\mathcal{T}$的转移。

“贪心”一次来源于“贪心算法”，意味着一种算法通过在局部做出最优选择期望能接近全局最优解。在这里，贪心策略意味着如果我们有一个策略$\pi$及其对应的值函数$\boldsymbol{V}^\pi$(假设通过策略评估算法获得)，那么应用贪心策略函数$G$到$\boldsymbol{V}^\pi$将得到一个确定性策略$\pi_D',$它预期在某种意义上比$\pi$更好，具体而言即$\boldsymbol{V}^{\pi'_D}$要优于$\boldsymbol{V}^\pi$.

### 策略提升

像“更好”、“提升”这样的术语指的是值函数或者策略（后者指的是评估给定策略的MDP的值函数）。那么，什么叫做值函数$X:\mathcal{N}\rightarrow \mathbb{R}$比$Y:\mathcal{N}\rightarrow \mathbb{R}$更好呢？下面的定义给出了答案

**定义（值函数的比较）**  
我们说对于一个有限的MDP，值函数$X$比值函数$Y$更好，记作$X\geq Y,$当且仅当

$$
X(s)\geq Y(s)\quad \forall s\in\mathcal{N}
$$
如果我们处理的是有限的MDP（具有$m$个非终止状态），我们会将值函数表示为向量的形式$\boldsymbol{X},\boldsymbol{Y}\in\mathbb{R}^m.$

因此，每当你听到更好的值函数或者改进的值函数这样的术语时，应该理解为值函数在每个状态下都不比它所比较的值函数更差。

那么，什么是$\pi_D'=G(\boldsymbol{V}^\pi)$比$\pi$更好呢？下面是Richard Bellman的一个重要定理，给出了明确的解释：

**定理（策略改进定理）**：对于一个有限的MDP，对于任意策略$\pi$,都有：

$$
\boldsymbol{V}^{\pi_D'}=\boldsymbol{V}^{G(\boldsymbol{V}^{\pi})}\geq \boldsymbol{V}^{\pi}
$$
这个证明基于应用贝尔曼策略算子在给定MDP的值函数上的作用（注意这种MDP视角下的贝尔曼策略算子在\@ref(eq:5-2)中表示）。我们首先注意到，反复应用贝尔曼策略算子$B^{\pi_D'}$从值函数$\boldsymbol{V}^\pi$开始，最终会收敛到$\boldsymbol{V}^{\pi_D'}.$形式上，

$$
\lim_{i\rightarrow \infty}(B^{\pi_D'})^i(\boldsymbol{V}^\pi)=\boldsymbol{V}^{\pi_D'}
$$
因此证明的关键是证明

$$
(B^{\pi_D'})^{i+1}(\boldsymbol{V}^\pi)\geq (B^{\pi_D'})^i(\boldsymbol{V}^\pi)\quad \forall i=0,1,2,...
$$
这意味着通过反复应用贝尔曼策略算子得到一个不下降的值函数序列，随着反复应用，会不断改善直到收敛到值函数$\boldsymbol{V}^{\pi_D'}$.

策略改进定理为我们提供了用来解决MDP控制问题的第一个动态规划算法（称为策略迭代）。策略迭代算法是Ronald Howard（1960）提出的。

### 策略迭代算法

策略提升定理的证明向我们展示了怎么从一个关于策略$\pi$的值函数$\boldsymbol{V}^\pi$出发，通过贪婪策略改进生成策略$\pi_D'=G(\boldsymbol{V})$,然后以$\boldsymbol{V}^\pi$为起始价值函数进行策略评估（使用策略$\pi_D'$），得到改进后的价值函数$\boldsymbol{V}^{\pi_D'}$,该价值函数优于我们最初的价值函数$\boldsymbol{V}^\pi.$现在需要注意的是，我们可以重复这一过程，从$\pi_D',\boldsymbol{V}^{\pi_D'}$出发，进一步改进策略$\pi_D''$及其相关的改进价值函数$\boldsymbol{V}^{\pi_D''}$,我们可以继续这种方式，生成进一步改进的策略及其相关的价值函数，直到无法再改进为止。这种将策略改进与使用改进策略进行策略评估相结合的方法被称为**策略迭代算法**。

- 从任意价值函数$\boldsymbol{V}_0\in\mathbb{R}^m$开始；

- 迭代$j=0,1,2,...,$在每次迭代中计算：  
  - 确定性策略:$\pi_{j+1}=G(\boldsymbol{V}_j)$  
  - 价值函数：$\boldsymbol{V}_{j+1}=\lim_{i\rightarrow \infty}(B^{\pi_{j+1}})(\boldsymbol{V}_j)$  
- 当$d(\boldsymbol{V}_i,\boldsymbol{V}_{i+1})=\max_{s\in\mathcal{N}} |(\boldsymbol{V}_i-\boldsymbol{V}_{i+1})(s)|$足够小，停止算法。

因此，当价值函数无法进一步改进时，算法终止，当这种情况发生时，以下等式应成立：

$$
\boldsymbol{V}_j=(B^{G(\boldsymbol{V_j})})^i(\boldsymbol{V}_j)=\boldsymbol{V}_{j+1}\text{ for all }i=0,1,2,...
$$
特别地，当$i=1$时有：

$$
\boldsymbol{V}_j(s)=B^{G(\boldsymbol{V_j})}(\boldsymbol{V}_j)(s)=\mathcal{R}(s,G(\boldsymbol{V}_j)(s))+\gamma\sum_{s'\in\mathcal{N}}\mathcal{P}(s,G(\boldsymbol{V_j})(s),s')\cdot \boldsymbol{V_j}(s')\text{ for all }s\in\mathcal{N}
$$
由方程\@ref(eq:5-3)可知，我们对于每个$s\in\mathcal{N},\pi_{j+1}(s)=G(\boldsymbol{V_j})(s)$是最大化$\{ \mathcal{R}(s,a)+\gamma\sum_{s'}\mathcal{P}(s,a,s')\cdot \boldsymbol{V_j}(s') \}$的动作，因此

$$
\boldsymbol{V_j}(s)=\max_{a\in\mathcal{A}}\left\{ \mathcal{R}(s,a)+\gamma\sum_{s'\in\mathcal{N}}\mathcal{P}(s,a,s')\cdot\boldsymbol{V_j}(s') \right\}\text{ for all }s\in\mathcal{N}
$$
但这实际上是MDP状态-价值函数贝尔曼最优性方程，这意味着$\boldsymbol{V}_j=\boldsymbol{V}^*,$即当$\boldsymbol{V}_{j+1}=\boldsymbol{V}_j$时，策略迭代算法已收敛到最优价值函数。策略迭代算法收敛时的确定性策略$\pi_j:\mathcal{N\rightarrow A}$是一个最优策略，因为$\boldsymbol{Y}_{\pi_j}=\boldsymbol{V}_j\approx \boldsymbol{V}^*$,这意味着用确定性策略$\pi_j$评估MDP可以实现最优价值函数。这表明策略迭代算法解决了MDP控制问题。这证明了以下定理：

**定理（策略迭代收敛定理）：** 对于具有$|\mathcal{N}|=m,\gamma<1$的有限MDP，策略迭代算法收敛到最优价值函数$\boldsymbol{V}^*\in\mathbb{R}^m$以及一个确定性最优策略$\pi_D^*:\mathcal{N\rightarrow A}$，无论我们从哪个价值函数$\boldsymbol{V}_0$开始算法。



### 贝尔曼最优性算子与值迭代算法

通过对方程\@ref(eq:5-3)进行微调（将$\arg\max$改为$\max$）,我们定义贝尔曼最优性算子：

$$
B^*:\mathbb{R}^m\rightarrow \mathbb{R}^m
$$
作为向量空间$\mathbb{R}^m$中向量（表示价值函数）的以下非线性变换：

$$
B^*(\boldsymbol{V})(s)=\max_{a\in\mathcal{A}}\left\{ \mathcal{R}(s,a)+\gamma\sum_{s\in\mathcal{N}}\mathcal{P}(s,a,s')\cdot \boldsymbol{V}(s')\right\}\text{ for all }s\in\mathcal{N} (\#eq:5-5)
$$
我们将在数学推导中使用方程\@ref(eq:5-5),但我们需要一个不同但等价的表达式来指导代码，使用接口操作的是$\mathcal{P}_R$并非$\mathcal{P,R}$,等价表达式如下：

$$
B^*(\boldsymbol{V})(s)=\max_{a\in\mathcal{A}}\left\{ \sum_{s\in\mathcal{N}}\sum_{r\in\mathcal{D}}\mathcal{P}_R(s,a,r,s')\cdot 
(r+\gamma\boldsymbol{W}(s'))
\right\}\text{ for all }s\in\mathcal{N} (\#eq:5-6)
$$
其中$\boldsymbol{W}\in\mathbb{R}^n$定义为


\begin{equation}
\boldsymbol{W}(s)=\begin{cases}
\boldsymbol{V}(s')&\text{ if }s'\in\mathcal{N}\\
0&\text{ if }s'\in\mathcal{T=S-N}
\end{cases}
\end{equation}

注意在方程\@ref(eq:5-6)中，由于我们需要考虑到所有状态$s'\in\mathcal{S}$的转移而不是方程\@ref(eq:5-3)中$s'\in\mathcal{N}$的状态。因此，我们需要小心处理$s'\in\mathcal{T}$的转移。

对于每一个$s\in\mathcal{N},$在\@ref(eq:5-5)中产生的最大化动作$a\in\mathcal{A}$是由确定性策略$\pi_D$在\@ref(eq:5-3)中规定的动作。因此，如果我们使用贪婪策略$G(\boldsymbol{V})$在任何价值函数上应用贝尔曼策略算子，它应该与应用贝尔曼最优性算子相同，因此：

$$
B^{G(\boldsymbol{V})}(\boldsymbol{V})=B^*(\boldsymbol{V})\text{ for all }\boldsymbol{V}\in\mathbb{R}^m (\#eq:5-7)
$$
特别地，通过将$\boldsymbol{V}$转化为策略$\pi$的价值函数$\boldsymbol{V}^\pi$,我们得到：

$$
B^{G(\boldsymbol{V}^\pi)}(\boldsymbol{V}^\pi)=B^*(\boldsymbol{V}^\pi)
$$
这是策略评估第一阶段的一个简洁表示，其中使用了改进的策略$G(\boldsymbol{V}^\pi)$（注意贝尔曼策略算子、贝尔曼最优性算子和贪婪策略函数如何在这个方程中结合在一起）。

正如贝尔曼策略算子$B^\pi$是由MDP贝尔曼策略方程（等价于MRP贝尔曼方程）所驱动的，贝尔曼最优性算子$B^*$是由MDP状态-价值函数贝尔曼最优性方程（重新陈述如下）所驱动的

$$
\boldsymbol{V}^*(s)=\max_{a\in\mathcal{A}}\left\{\mathcal{R}(s,a)+\gamma\sum_{s'\in\mathcal{N}}\mathcal{P}(s,a,s')\cdot\boldsymbol{V}^*(s')\right\}\text{ for all }s\in\mathcal{N}
$$
因此，我们可以简洁地将MDP状态-价值函数贝尔曼最优性方程表示为：

$$
\boldsymbol{V^*}=B^*(\boldsymbol{V^*})
$$
这意味着$\boldsymbol{V^*}$是贝尔曼最优性算子$B^*$地一个不动点。

需要注意的是，我们提供的贪婪策略函数和贝尔曼最优性算子的定义可以推广到非有限MDP，因此我们可以推广方程\@ref(eq:5-7)，并且$\boldsymbol{V^*}$是贝尔曼最优性算子的不动点的陈述仍然成立。然而，在本章中由于我们专注于开发有限 MDP 的算法，因此我们将坚持为有限 MDP 提供的定义。

正如我们证明$B^\pi$是一个压缩函数一样，我们希望证明$B^*$也是一个压缩函数（在$L^\infty$范数下），以便我们可以利用Banach不动点定理，并通过迭代应用贝尔曼最优性算子$B^*$来解决控制问题。因此，我们需要证明对于所有$\boldsymbol{X},\boldsymbol{Y}\in\mathbb{R}^m:$

$$
\max_{s\in\mathcal{N}}|(B^*(\boldsymbol{X})-B*(\boldsymbol{Y}))(s)|\leq \gamma\cdot \max_{s\in\mathcal{N}}|(\boldsymbol{X}-\boldsymbol{Y})(s)|
$$
这个证明比之前为$B^\pi$所做的证明要难一些，这里需要利用$B^*$的两个关键性质，单调性和常数平移性：对于所有的$\boldsymbol{X}\in\mathbb{R}^m,c\in\mathbb{R},B^*(\boldsymbol{X}+c)(s)=B^*(\boldsymbol{X})(s)+\gamma c$对于所有的$s\in\mathcal{N}$.证明暂时略去，调用Banach不动点定理证明下面的定理：

**定理（值迭代收敛定理）：**对于具有$|\mathcal{N}|=m,\gamma<1$的有限MDP，如果$\boldsymbol{V}^*\in\mathbb{R}^m$是最优值函数，则$\boldsymbol{V}^*$是贝尔曼最优性算子$B^*$的唯一不动点，并且：

$$
\lim_{i\rightarrow \infty}(B^*)^i(\boldsymbol{V}_0)\rightarrow \boldsymbol{V}^*\text{ for all staring Value Function }\boldsymbol{V}_0\in\mathbb{R}^m
$$

这为我们提供了以下迭代算法，称为**值迭代算法**，有Richard Bellman提出：

- 从任意价值函数$\boldsymbol{V}_0$开始;

- 迭代$i=0,1,2,...,$在每次迭代中计算：  
  $$
  \boldsymbol{V}_{i+1}(s)=B^*(\boldsymbol{V}_i)(s)\text{ for all }s\in\mathcal{N}
  $$

- 当$d(\boldsymbol{V}_i,\boldsymbol{V}_{i+1})=\max_{s\in\mathcal{N}} |(\boldsymbol{V}_i-\boldsymbol{V}_{i+1})(s)|$足够小，停止算法。

### 从最优值函数到最优策略

需要注意的是，策略迭代算法在每次迭代中都会生成一个策略及其对应的价值函数。因此，最终当我们收敛到最优值函数$\boldsymbol{V}_j=\boldsymbol{V}^*$时，策略迭代算法总会有一个确定性策略$\pi_j$与$\boldsymbol{V}_j$相关联，使得：

$$
\boldsymbol{V}_j=\boldsymbol{V}^{\pi_j}=\boldsymbol{V}^*
$$
我们将$\pi_j$称为最优策略$\pi^*,$即产生最优值函数$\boldsymbol{V}^*$的策略，即

$$
\boldsymbol{V}^{\pi^*}=\boldsymbol{V}^*
$$
然而，值迭代算法没有与之相关联的策略，因为整个算法缺乏策略表示，仅操作价值函数。因此，现在的问题是：当值迭代手来难道最优价值函数$\boldsymbol{V}_i=\boldsymbol{V}^*$,我们如何获得一个最优策略使得：

$$
\boldsymbol{V}^{\pi^*}=\boldsymbol{V}_i=\boldsymbol{V}^*
$$
答案在于贪婪策略函数$G.$方程\@ref(eq:5-7)告诉我们：

$$
B^{G(\boldsymbol{V})}(\boldsymbol{V})=B^*(\boldsymbol{V})\text{ for all }\boldsymbol{V}\in\mathbb{R}^m
$$
将$\boldsymbol{V}$特化为$\boldsymbol{V}^*$,我们得到

$$
B^{G(\boldsymbol{V}^*)}(\boldsymbol{V}^*)=B^*(\boldsymbol{V}^*)
$$
但我们又指导$\boldsymbol{V}^*$是贝尔曼算子$B^*$的不动点，因此

$$
B^{G(\boldsymbol{V}^*)}(\boldsymbol{V}^*)=\boldsymbol{V}^*
$$
这表明，用确定性贪婪策略$G(\boldsymbol{V}^*)$(使用贪婪策略函数从最优价值函数创建的策略)评估MDP，实际上实现了最优价值函数$\boldsymbol{V}^*.$换句话说，$G(\boldsymbol{V}^*)$是我们一直在寻找的确定性最优策略$\pi^*.$

### 广义策略迭代

本节中将深入探讨策略迭代算法的结构，并展示如何将其结构推广到更一般的情况。让我们首先从二维布局的角度来看策略迭代中价值函数从初始值函数$\boldsymbol{V}_0$到最终价值函数$\boldsymbol{V}^*$的演进过程。

**策略迭代的二维布局**

策略迭代的演进过程可以用以下二维布局表示：

$$
\begin{align}
\pi_1=G(\boldsymbol{V}_0),\boldsymbol{V}_0\rightarrow B^{\pi_1}(\boldsymbol{V}_0)\rightarrow &... (B^{\pi_1})^i(\boldsymbol{V}_0)\rightarrow... \boldsymbol{V}^{\pi_1}=\boldsymbol{V}_1\\
\pi_2=G(\boldsymbol{V}_1),\boldsymbol{V}_1\rightarrow B^{\pi_2}(\boldsymbol{V}_1)\rightarrow &... (B^{\pi_2})^i(\boldsymbol{V}_1)\rightarrow... \boldsymbol{V}^{\pi_2}=\boldsymbol{V}_2\\
&...\\
&...\\
\pi_{j+1}=G(\boldsymbol{V}_j),\boldsymbol{V}_j\rightarrow B^{\pi_{j+1}}(\boldsymbol{V}_j)\rightarrow &... (B^{\pi_{j+1}})^i(\boldsymbol{V}_j)\rightarrow... \boldsymbol{V}^{\pi_{j+1}}=\boldsymbol{V}_j\\


\end{align}
$$
每一行代表在特定策略下价值函数的演进过程。每一行从使用贪婪策略函数$G$创建策略开始，其余部分是通过对该策略应用贝尔曼策略算子$B^\pi$指导收敛到该策略的价值函数。因此每一行以策略改进开始，其余部分是策略评估。注意，每一行的结束通过贪婪策略函数$G$与下一行的开始无缝衔接！

**策略迭代的三重循环**

策略迭代算法实际上包含三重循环：

1. 最外层循环：遍历二维布局中的每一行（每次迭代生成一个改进的策略）。

2. 中间层循环：遍历每一行中的列（每次迭代应用贝尔曼策略算子，即策略评估的迭代）。

3. 最内层循环：遍历所有状态$s\in\mathcal{N},$因为在应用贝尔曼策略算子更新价值函数时需要遍历所有状态（在应用贪婪策略函数改进策略时也需要遍历所有状态）。

**策略迭代的高层次视角**

从更高层次来看，策略迭代时策略评估和策略改进交替进行的过程：

- **策略评估**：根据当前策略生成价值函数。

- **策略改进**：根据当前价值函数生成贪婪策略（相对于前一个策略有所改进）。

这种交替过程使得价值函数和策略逐渐趋于一致，直到最终收敛。

**策略迭代的可视化**  

下图展示了策略迭代中价值函数和策略的演进过程。图中：  

- 下部的线（策略线）：表示策略的演进。

- 上部的线（价值函数线）：表示价值函数的演进。

- 指向价值函数线的箭头：表示对给定策略$\pi$的策略评估，生成价值函数$\boldsymbol{V}^\pi$.

- 指向策略线的箭头：表示从价值函数$\boldsymbol{V}^\pi$生成贪婪策略$\pi'=G(\boldsymbol{V}^\pi)$.

策略评估和策略改进是“竞争”的——它们“朝不同方向推动”，但最终目标是使价值函数和策略趋于一致。

**广义策略迭代**

广义策略迭代（Generalized Policy Iteration, GPI）是 Sutton 和 Barto 在其强化学习书中强调的一个重要概念，它统一了所有动态规划（DP）和强化学习（RL）算法的变体。GPI 的核心思想是：

- 策略评估：可以使用任何策略评估方法

- 策略改进：可以使用任何策略改进方法（不一定是经典策略迭代算法中的方法）。

GPI 的关键在于，策略评估和策略改进不需要完全达到它们各自追求的一致性目标。例如：

- 策略评估：可以只进行几次贝尔曼策略评估，而不是完全收敛到$V^{\pi}$.

- 策略改进：可以只更新部分状态的策略，而不是所有状态。

**值迭代作为GPI的实例**

值迭代是 GPI 的一个具体实例。在值迭代中，每次迭代只应用一次贝尔曼策略算子，然后进行策略改进。

在值迭代中，策略改进步骤保持不变，但策略评估简化为仅应用一次贝尔曼策略算子。

广义策略迭代是强化学习中最核心的概念之一。几乎所有强化学习控制算法都可以视为 GPI 的特例。例如，在某些简单的强化学习控制算法中：

- 策略评估：只对单个状态进行。

- 策略改进：也只对单个状态进行。

这些算法本质上是单状态策略评估和单状态策略改进的交替序列。

**总结**

- 贝尔曼方程和广义策略迭代是强化学习中最重要的两个概念。

- GPI 的核心思想是交替进行某种形式的策略评估和策略改进。

- GPI 统一了动态规划和强化学习的各种算法，是理解强化学习控制问题的基础。

### 异步动态规划

我们在本章中描述的经典动态规划算法被称为同步动态规划算法。同步一词指的是两个方面：

- 在每次迭代中所有状态的值都会被更新；

- 算法的数学描述中，所有状态的值更新是同时发生的。然而，在实际编码时（比如在`Python` 中，计算是串行的而非并行的），这种“同时更新”将通过创建一个新的值函数向量，并通过遍历所有状态将旧值向量中的值赋给新向量来实现。

实际上，动态规划算法通常实现为异步（Asynchronous）算法，其中上述两个约束（所有状态同时更新）被放宽。异步一词赋予了极大的灵活性——我们可以在每次迭代中更新一个子集的状态，且可以按任意顺序更新状态。

这种放宽同步约束的自然结果是，我们可以仅维护一个值函数向量，并就地更新这些值。这带来了显著的好处——状态的更新值可以立即用于更新其他状态（注意：在同步情况下，由于需要旧值向量和新值向量，在整个状态遍历完成之前不能用更新后的状态值来更新其他状态）。实际上，在实际的 MDP 控制问题求解算法中，值函数的就地更新是常见的做法。

另一个异步算法的特点是，我们可以优先更新某些状态值的顺序。有多种方法可以对算法进行优先级排序，这里我们仅突出一种简单但有效的状态值更新优先级排序方法——优先级遍历（Prioritized Sweeping）。我们维护一个按照“值函数间隙”（value function gaps）$g:\mathcal{N}\rightarrow\mathbb{R}$排序的状态队列（下面的例子展示了值迭代的情况）：

$$
g(s)=\left \vert V(s)-\max_{a\in\mathcal{A}}\left\{ \mathcal{R}(s,a)+\gamma\cdot\sum_{s'\in\mathcal{N}}\mathcal{P}(s,a,s')\cdot V(s') \right\}\right \vert\text{ for all }s\in\mathcal{N}
$$
在每次状态值更新后，我们更新所有由该状态值更新而导致值函数间隙变化的状态的值函数间隙。那些状态是从中有概率转移到更新后的状态的状态。这个过程意味着我们需要在数据结构中维护反向转移动态。因此，每次状态值更新后，队列会根据值函数间隙重新排序。我们总是从队列顶部取出值函数间隙最大的状态，并更新该状态的值函数。这样，我们就优先更新值函数间隙较大的状态，并确保我们很快会达到一个所有值函数间隙都足够小的状态。

另一种值得提到的异步动态规划形式是实时动态规划（Real Time Dynamic Programming, RTDP）。RTDP 是指在 AI 代理与环境实时交互的过程中运行动态规划算法。当一个状态在实时交互过程中被访问时，我们更新该状态的值。然后，在由于实时交互而转移到另一个状态时，我们更新新状态的值，以此类推。请注意，在 RTDP 中，动作选择是由 AI 代理执行的实时动作，环境根据该动作做出反应。这个动作选择由遇到状态时，值函数隐含的策略决定。

最后，我们需要强调的是，通常 MDP 的某些特殊结构可以通过特定定制的动态规划算法（通常是异步的）获得更好的效果。一个这样的特殊化是，当每个状态在 AI 代理执行 MDP 时，每个随机状态序列中最多仅被遇到一次，且所有随机序列都终止时。这种结构可以被概念化为一个有向无环图（Directed Acylic Graph, DAG），其中每个非终结节点表示一个非终结状态和动作对，而每个终结节点表示一个终结状态（图的边表示 MDP 的概率转移）。在这种特殊化中，MDP 的预测和控制问题可以通过简单的方式解决——通过从终结节点向后遍历 DAG，使用贝尔曼最优性方程（对于控制）或贝尔曼策略方程（对于预测）设置经过的状态的值函数。在这里，我们不需要像策略评估、策略迭代或值迭代那样的“迭代到收敛”方法。相反，所有这些动态规划算法本质上都简化为在 DAG 上 的值函数反向传播。这意味着，状态按照 DAG 上拓扑排序的逆序被访问（并设置其值函数）。我们将在下一个章节中详细讲解这种 DAG 反向传播动态规划算法——有限时域 MDP——在该类型的 MDP 中，所有的随机序列都将在固定的时间步内终止，每个时间步都有一个独立的状态集。这个有限时域 MDP 的特殊情况在金融应用中相当常见，因此我们将在下一节详细讨论。

### 有限时域的动态规划：反向推导

本节中我们讨论上一节提到的DAG结构化马尔可夫决策过程的一个特化版本——我们称之为有限时域的MDP，其中每个序列在固定的有限时间步$T$内终止，并且每个时间步都有一组独立于其他时间步的可数状态。因此$T$时刻的所有状态都是终止状态，且某些$T$之前的状态也可能是终止状态。对于所有的$t=0,1,...,T,$定义在$t$时刻的状态集为$\mathcal{S}_t,$第$t$时刻的终止状态集$\mathcal{T}_t,$非终止状态集为$\mathcal{N}_t=\mathcal{S}_t-\mathcal{T}_t$(注意$\mathcal{N}_T=\emptyset$).如前所述，当MDP不是时间齐次时，我们将每个状态扩展为包括时间步的索引，因此在时间步$t$的扩展状态为$(t,s_t),$其中$s_t\in\mathcal{S}_t.$整个MDP的扩展的状态空间$S$为

$$
\{(t,s_t)\mid t=0,1,...,T,s_t\in\mathcal{S}_t\}
$$
终止状态集$\mathcal{T}$为

$$
\{(t,s_t)\mid t=0,1,...,T,s_t\in\mathcal{T}_t\}
$$
在时间步$t$时，AI代理可获得的奖励集合记为$\mathcal{D}_t$($\mathbb{R}$中的可数子集)，而在非终止状态的可选动作记为$\mathcal{A}_t.$在更一般的设置中，我们将在代码中表示每个非终止状态$(t,s_t)$有自己独立的可选动作集$\mathcal{A}(s_t)$,然而为了简化讨论，这里我们假设所有时间步下的所有非终止状态具有相同的动作集$\mathcal{A}_t,$我们将整个动作空间$\mathcal{A}$表示为所有时间步$t=0,1,...,T-1$下的$\mathcal{A}_t$的并集。

状态-奖励转移概率函数

$$
\mathcal{P}_R:\mathcal{N\times A\times D\times S}\rightarrow [0,1]
$$
由下面的式子给出


\begin{align}
\mathcal{P}_R((t,s_t),a_t,r_{t'},(t',s_{t'}))=
\begin{cases}
(\mathcal{P}_R)_t(s_t,a_t,r_{t'},s_{t'})&\text{ if }t'=t+1\text{ and } s_t\in\mathcal{S}_{t'} \text{ and }r_{t'}\in\mathcal{D}_{t'}\\
0&\text{ otherwise}
\end{cases}
\end{align}

对于所有$t=0,1,...,T-1,s_t\in \mathcal{N}_t,a_t\in\mathcal{A}_t,t'=0,1,...,T$其中

$$
(\mathcal{P}_R)_t:\mathcal{N_t\times A_t\times }\mathcal{D}_{t+1}\times \mathcal{S}_{t+1}\rightarrow [0,1]
$$
是每个时间步$t$的独立状态-奖励转移概率函数，满足

$$
\sum_{s_{t+1}\in\mathcal{S}_{t+1}}\sum_{r_{t+1}\in\mathcal{D}_{t+1}}(\mathcal{P}_R)_t(s_t,a_t,r_{t+1},s_{t+1})=1
$$
对于所有$t=0,1,...,T-1,s_t\in \mathcal{N}_t,a_t\in\mathcal{A}_t.$

因此，使用每个时间步$t$的独立转移概率函数表示有限时域MDP是很方便的。同样地，MDP的任何策略$\pi:\mathcal{N\times A}\rightarrow[0,1]$

$$
\pi((t,s_t),a_t)=\pi_t(s_t,a_t)
$$
也可以很方便地表示为每个时间步$t$的独立策略,其中

$$
\pi_t:\mathcal{N_t\times A_t}\rightarrow [0,1]
$$
因此我们可以将$\pi$解释为由序列$(\pi_0,...,\pi_{T-1})$组成。

因此对于给定的策略$\pi,$值函数$V^\pi:\mathcal{N}\rightarrow \mathbb{R}$可以方便地表示为每个时间步$t$的值函数

$$
V^\pi_t:\mathcal{N_t}\rightarrow \mathbb{R}
$$
对于每一个时间步$t=0,1,...,T-1$,定义为

$$
V^\pi((t,s_t))=V_t^\pi(s_t)\text{ for all }t=0,1,...,T-1,s_t\in\mathcal{N}_t
$$

然后贝尔曼策略方程可以写为

$$
V_t^\pi(s_t)=\sum_{s_{t+1}\in\mathcal{S}_{t+1}}\sum_{r_{t+1}\in\mathcal{D}_{t+1}}(\mathcal{P}_R^{\pi_t})_t(s_t,r_{t+1},s_{t+1})\cdot (r_{t+1}+\gamma\cdot W_{t+1}^\pi(s_{t+1}))\\
\text{ for all }t=0,1,...,T-1,s_T\in\mathcal{N_t} (\#eq:5-8)
$$
其中


\begin{align}
W_t^\pi(s_t)=
\begin{cases}
V_t^\pi(s_t)&\text{ if }s_t\in\mathcal{N}_t\\
0&\text{ if }s_t\in\mathcal{T}_t
\end{cases}
\text{ for all }t=1,2,...,T
\end{align}

其中$(\mathcal{P}_R^{\pi_t})_t:\mathcal{N}_t\times\mathcal{D}_{t+1}\times\mathcal{S}_{t+1}$对所有$t=0,1,...,T-1$表示$\pi$隐含的MRP状态-奖励转移概率函数，定义为

$$
(\mathcal{P}_R^{\pi_t})_t(s_t,r_{t+1},s_{t+1})=\sum_{a_t\in\mathcal{A}_t}\pi_t(s_t,a_t)\cdot (\mathcal{P}_R)_t(s_t,a_t,r_{t+1},s_{t+1})\text{ for all }t=0,1,...,T-1
$$

所以对于有限时域的MDP，这提供了一个简单的算法，通过从$t=T-1$向$t=0$倒推，使用方程\@ref(eq:5-8)计算$V_t^\pi$,并从已知的$W_{t+1}^\pi$计算出所有的$t=0,1,...,T-1$的$V^\pi_t.$由于我们是倒推时间索引$t.$这就是已知的向后推到（Backward Induction）方法。

接下来，我们转向控制问题——计算最优值函数和最优策略。类似之前的模式，最优值函数$V^*:\mathcal{N}\rightarrow \mathbb{R}$可以方便地表示为每个时间步$t$的值函数$V_t^*:\mathcal{N}_t\rightarrow \mathbb{R}$的序列，定义为

$$
V^*((t,s_t))=V_t^*(s_t)\text{ for all }t=0,1,...,T-1,s_t\in\mathcal{N}_t
$$

因此，MDP状态-价值函数的贝尔曼最优性方程可以写成

$$
V_t^*(s_t)=\max_{a_t\in\mathcal{A}_t}\left\{ \sum_{s_{t+1}\in\mathcal{S}_{t+1}}\sum_{r_{t+1}\in\mathcal{D}_{t+1}}(\mathcal{P}_R)_t(s_t,a_t,r_{t+1},s_{t+1})\cdot (r_{t+1}+\gamma\cdot W_{t+1}^*(s_{t+1})) \right\}\\
\text{ for all }t=0,1,...,T-1,s_t\in\mathcal{N}_t (\#eq:5-9)
$$
其中

\begin{align}
W_t^*(s_t)=
\begin{cases}
V_t^*(s_t)&\text{ if }s_t\in\mathcal{N}_t\\
0&\text{ if }s_t\in\mathcal{T}_t
\end{cases}
\text{ for all }t=1,2,...,T
\end{align}

相关的最优（确定性）策略$(\pi^*_D)_t:\mathcal{N}_t\rightarrow \mathcal{A}_t$
定义为

$$
(\pi^*_D)_t(s_t)=\mathop{\arg\max}_{a_t\in\mathcal{A}_t}\left\{ \sum_{s_{t+1}\in\mathcal{S}_{t+1}}\sum_{r_{t+1}\in\mathcal{D}_{t+1}}(\mathcal{P}_R)_t(s_t,a_t,r_{t+1},s_{t+1})\cdot (r_{t+1}+\gamma\cdot W_{t+1}^*(s_{t+1}) \right\}\\
\text{ for all }t=0,1,...,T-1,s_t\in\mathcal{N}_t (\#eq:5-10)
$$

因此，对于有限MDP，通过从$t=T-1$倒推到$t=0,$使用方程\@ref(eq:5-9)计算$V^*_t,$并使用方程\@ref(eq:5-10)计算$(\pi^*_D)_t$的最优策略，对于所有的$t=0,1,...,T-1,$从已知的$W_{t+1}^*$值可以计算出最优值函数和最优策略。这个算法是将值迭代方法应用到有限时域情况下的“向后推导”技术。

请注意，这些有限视界有限 MDP 算法不需要像常规策略评估和值迭代那样进行任何“迭代收敛”。相反，在这些算法中，我们只是回顾过去，并立即从下一个时间步骤的值函数（由于我们回顾过去，因此已经知道）获得每个时间步骤的值函数。这种“值函数反向传播”技术被称为后向归纳算法，在许多金融应用中非常常见（我们将在本书后面看到）。

### 产品停产/停季时的动态定价

现在我们考虑一个相当重要的商业应用——动态定价。我们考虑一个问题，即在产品接近生命周期末期或季节结束时，如何进行动态定价，此时我们不再希望继续销售该产品。我们需要根据产品库存量、距离生命周期结束/季节结束还有多少天以及预期的客户需求（作为价格调整的函数）动态地调整价格。为了让问题更具体，假设你经营一个超市，距离万圣节还有$T$天。你刚从供应商那里收到了$M$个万圣节面具，并且在这$T$天内将不会再收到任何库存。你希望在每天开始时动态地设定万圣节面具的销售价格，以最大化从今天到万圣节的预期总销售收入（假设万圣节之后没有人再购买面具）。

假设在$T$天中的每一天，当天开始时需要从$N$个价格$P_1,...,P_N$中选择一个价格$P_i,$该价格将作为当天所有面具的销售价格。假设当天客户需求量服从一个Poisson分布，如果选择了$P_i$作为当天价格那么需求的平均值为$\lambda_i.$注意，在任何给定的日子，需求可能会超过你店里拥有的面具数量，这时当天的销售数量将等于当天开始时你店里拥有的面具数量。

这个MDP的状态由一对$(t,I_t)$组成，其中$t\in\{0,1,...,T\}$表示时间索引，$I_t\in\{0,1,...,M\}$表示时间$t$时的库存。使用前面章节的记号$\mathcal{S}_t=\{0,1,...,M\},I_t\in \mathcal{S}_t.$当$t=0,1,...,T-1$时$ \mathcal{N}_t=\mathcal{S}_t,\mathcal{N}_T=\emptyset.$在时间$t$时的动作选择可以通过从$1\sim N$的整数选择来表示，因此$\mathcal{A}_t=\{1,2,...,N\}.$

注意$I_0=M,I_{t+1}=\max(0,I_t-d_t),0\leq t\leq T.$其中$d_t$是第$t$天的随机需求，服从Poisson分布均值为$\lambda_i,$如果第$t$天的动作选择$i\in\mathcal{A}_t.$另外请注意，第$t$天的销售收入等于$\min(I_t,d_t)\cdot P_i.$因此，时间$t$的状态-奖励转移概率函数$(\mathcal{P}_R)_t:\mathcal{N}_t\times\mathcal{A}_t\times\mathcal{D}_{t+1}\times\mathcal{S}_{t+1}\rightarrow [0,1]$
定义为

$$
(\mathcal{P}_R)_t(I_t,i,r_{t+1},I_t-k)=
\begin{cases}
\frac{e^{-\lambda_i}\lambda_i^k}{k!}&\text{ for }k<I_t\text{ and } r_{t+1}=k\cdot P_i\\
\sum_{j=I_t}^\infty\frac{e^{-\lambda_i}\lambda_i^j}{j!}&\text{ for }k=I_t\text{ and } r_{t+1}=k\cdot P_i\\
0&\text{ otherwise}
\end{cases}
$$
对于所有的$0\leq t<T$.

利用$(\mathcal{P}_R)_t$的定义和边界条件$W_T^*(I_T)=0$,我们可以执行逆向推导算法来计算$V_t^*$及其相应的最优确定性策略$(\pi^*_D)_t,0\leq t<T.$

现在让我们编写代码来表示这个动态规划问题，作为一个有限MDP过程，确定其最优策略，即在任意库存水平$I_t$下时间步$t$的最优动态价格。

下面介绍如何用Colab和Github不配置环境即可复现书里的代码。

首先在Colab里面新建一个笔记本，在代码块里输入


``` python
from google.colab import drive
drive.mount('/content/drive')
```

上述已经挂载到我们的Google Drive上了，随后git clone本书仓库

``` python
!git clone https://github.com/TikhonJelvis/RL-book.git
```
当然也不要忘记更改工作目录！

``` python
import sys
sys.path.append('/content/RL-book')
```
然后可以写书中的例子了！相关的函数在书的仓库中可以找到。

``` python
from rl.markov_decision_process import (
    FiniteMarkovDecisionProcess, FiniteMarkovRewardProcess)
from rl.policy import FiniteDeterministicPolicy, FinitePolicy
from rl.finite_horizon import WithTime
from typing import Sequence, Tuple, Iterator
from scipy.stats import poisson
from rl.distribution import Categorical
from rl.finite_horizon import (
    finite_horizon_MRP, unwrap_finite_horizon_MRP, evaluate,
    finite_horizon_MDP, unwrap_finite_horizon_MDP, optimal_vf_and_policy)
from rl.dynamic_programming import V


class ClearancePricingMDP:

    initial_inventory: int
    time_steps: int
    price_lambda_pairs: Sequence[Tuple[float, float]]
    single_step_mdp: FiniteMarkovDecisionProcess[int, int]
    mdp: FiniteMarkovDecisionProcess[WithTime[int], int]

    def __init__(
        self,
        initial_inventory: int,
        time_steps: int,
        price_lambda_pairs: Sequence[Tuple[float, float]]
    ):
        self.initial_inventory = initial_inventory
        self.time_steps = time_steps
        self.price_lambda_pairs = price_lambda_pairs
        distrs = [poisson(l) for _, l in price_lambda_pairs]
        prices = [p for p, _ in price_lambda_pairs]
        self.single_step_mdp: FiniteMarkovDecisionProcess[int, int] =\
            FiniteMarkovDecisionProcess({
                s: {i: Categorical(
                    {(s - k, prices[i] * k):
                     (distrs[i].pmf(k) if k < s else 1 - distrs[i].cdf(s - 1))
                     for k in range(s + 1)})
                    for i in range(len(prices))}
                for s in range(initial_inventory + 1)
            })
        self.mdp = finite_horizon_MDP(self.single_step_mdp, time_steps)

    def get_vf_for_policy(
        self,
        policy: FinitePolicy[WithTime[int], int]
    ) -> Iterator[V[int]]:
        mrp: FiniteMarkovRewardProcess[WithTime[int]] \
            = self.mdp.apply_finite_policy(policy)
        return evaluate(unwrap_finite_horizon_MRP(mrp), 1.)

    def get_optimal_vf_and_policy(self)\
            -> Iterator[Tuple[V[int], FiniteDeterministicPolicy[int, int]]]:
        return optimal_vf_and_policy(unwrap_finite_horizon_MDP(self.mdp), 1.)


if __name__ == '__main__':
    from pprint import pprint
    ii = 12
    steps = 8
    pairs = [(1.0, 0.5), (0.7, 1.0), (0.5, 1.5), (0.3, 2.5)]
    cp: ClearancePricingMDP = ClearancePricingMDP(
        initial_inventory=ii,
        time_steps=steps,
        price_lambda_pairs=pairs
    )
    print("Clearance Pricing MDP")
    print("---------------------")
    print(cp.mdp)

    def policy_func(x: int) -> int:
        return 0 if x < 2 else (1 if x < 5 else (2 if x < 8 else 3))

    stationary_policy: FiniteDeterministicPolicy[int, int] = \
        FiniteDeterministicPolicy({s: policy_func(s) for s in range(ii + 1)})

    single_step_mrp: FiniteMarkovRewardProcess[int] = \
        cp.single_step_mdp.apply_finite_policy(stationary_policy)

    vf_for_policy: Iterator[V[int]] = evaluate(
        unwrap_finite_horizon_MRP(finite_horizon_MRP(single_step_mrp, steps)),
        1.
    )

    print("Value Function for Stationary Policy")
    print("------------------------------------")
    for t, vf in enumerate(vf_for_policy):
        print(f"Time Step {t:d}")
        print("---------------")
        pprint(vf)

    print("Optimal Value Function and Optimal Policy")
    print("------------------------------------")
    prices = []
    for t, (vf, policy) in enumerate(cp.get_optimal_vf_and_policy()):
        print(f"Time Step {t:d}")
        print("---------------")
        pprint(vf)
        print(policy)
        prices.append(
            [pairs[policy.action_for[s]][0]
             for s in range(ii + 1)])

    import matplotlib.pyplot as plt
    from matplotlib import cm
    import numpy as np
    heatmap = plt.imshow(np.array(prices).T, origin='lower')
    plt.colorbar(heatmap, shrink=0.5, aspect=5)
    plt.xlabel("Time Steps")
    plt.ylabel("Inventory")
    plt.show()
```

以下是输出的可视化结果
![](https://Go9entle.github.io/picx-images-hosting/image.58hirk8d2t.webp)<!-- -->


### 推广至非表格算法

本章中介绍的有限马尔可夫决策过程（MDP）算法被称为“表格”算法。这里的“表格”指的是MDP被指定为一个有限的数据结构，值函数也被表示为一个包含非终止状态和值的有限“表格”。这些表格算法通常在每次迭代中遍历所有非终止状态，更新值函数。但对于大状态空间或无限状态空间，无法使用这种方法，我们需要使用函数逼近来表示值函数。好消息是，我们可以修改每个表格算法，使得在每一步中，不是遍历所有非终止状态，而是仅从一个合适的非终止状态子集进行采样，使用适当的贝尔曼计算（就像在表格算法中一样）来计算这些采样状态的值，然后通过这些采样状态的计算值来创建/更新值函数的函数逼近。重要的一点是，从这些表格算法到基于函数逼近的算法，我们的算法基本结构和基本原理（固定点和贝尔曼算子）仍然保持不变。在下一章中，我们将讨论如何将这些动态规划算法从表格方法推广到函数逼近方法。我们称这些算法为“近似动态规划”。

## 动态资产配置和消费 {#dynassall}  
### 个人财务的优化  

个人财务对某些人来说可能非常简单（每月领取工资，花光所有工资）而对另一些人来说可能非常复杂（例如，那些在多个国家拥有多家企业并拥有复杂资产和负债的人）。在这里，我们将考虑一个相对简单但包含足够细节的情况，以便提供动态资产配置和消费问题的基本要素。假设你的个人财务包括以下几个方面：

- 收入：这可能包括你的定期工资，通常在一段时间内保持不变，但如果你获得晋升或找到新工作，工资可能会发生变化。这也包括你从投资组合中变现的资金，例如如果你卖出一些股票并决定不再投资于其他资产。此外，还包括你从储蓄账户或某些债券中获得的利息。还有许多其他收入来源，有些是固定的定期支付，有些则在支付金额和时间上具有不确定性，我们不会一一列举所有不同的收入方式。我们只想强调，在生活中的各个时间点获得收入是个人财务的关键方面之一。

- 消费：这里的“消费”指的是“支出”。需要注意的是，人们需要定期消费以满足基本需求，如住房、食物和衣物。你支付的房租或按揭贷款就是一个例子——它可能是每月固定金额，但如果你的按揭利率是浮动的，它可能会有所变化。此外，如果你搬到新房子，房租或按揭可能会有所不同。你在食物和衣物上的支出也构成了消费。这通常每月相对稳定，但如果你有了新生儿，可能需要额外的支出用于婴儿的食物、衣物甚至玩具。此外，还有超出“必需品”的消费——比如周末在高级餐厅用餐、夏季度假、购买豪华汽车或昂贵的手表。人们从这种消费中获得“满足感”或“幸福感”（即效用）。这里的关键点是，我们需要定期决定每周或每月花多少钱（消费）。在动态决策中，人们面临着消费（带来消费效用）和储蓄（将钱投入投资组合，希望钱能增值，以便未来能够消费更多）之间的张力。

- 投资：假设你可以投资于多种资产——简单的储蓄账户提供少量利息、交易所交易的股票（从价值股到成长股，各自有不同的风险-收益权衡）、房地产（你购买并居住的房子确实被视为一种投资资产）、黄金等大宗商品、艺术品等。我们将投资于这些资产的资金组合称为投资组合（关于投资组合理论的简要介绍见附录B）。我们需要定期决定是否应该将大部分资金投入储蓄账户以保安全，还是应该将大部分投资资金分配于股票，或者是否应该更具投机性，投资于早期初创公司或稀有艺术品。审查投资组合的构成并可能重新分配资金（称为重新平衡投资组合）是动态资产配置的问题。还需要注意的是，我们可以将部分收入投入投资组合（意味着我们选择不立即消费这笔钱）。同样，我们可以从投资组合中提取部分资金用于消费。将资金投入或提取出投资组合的决策本质上是我们所做的动态消费决策，它与动态资产配置决策密切相关。

以上描述希望为你提供了资产配置和消费的双重动态决策的基本概念。最终，我们的个人目标是在一生中最大化消费的期望总效用（可能还包括在你去世后为配偶和子女提供的消费效用）。由于投资组合本质上是随机的，并且我们需要定期做出资产配置和消费决策，你可以看到这具备了随机控制问题的所有要素，因此可以建模为马尔可夫决策过程（尽管通常相当复杂，因为现实生活中的财务有许多细节）。以下是该MDP的粗略和非正式草图（请记住，我们将在本章后面为简化的情况形式化MDP）：

- **状态**：状态通常可能非常复杂，但主要包括年龄（用于跟踪达到MDP时间范围的时间）、投资于每种资产的资金数量、所投资资产的估值，以及可能还包括其他方面，如工作/职业状况（用于预测未来工资的可能性）。

- **动作**：动作是双重的。首先，它是每个时间步骤中选择的投资金额向量（时间步骤是我们审查投资组合以重新分配资金的时间周期）。其次，它是选择消费的灵活/可选的资金数量（即超出我们承诺支付的固定支出，如房租）。

- **奖励**：奖励是我们视为灵活/可选的消费效用——它对应于动作的第二部分。

- **模型**：模型（给定当前状态和动作的下一个状态和奖励的概率）在大多数现实生活情况中可能相当复杂。最困难的方面是预测我们生活和职业中明天可能发生的事情（我们需要这种预测，因为它决定了我们未来获得收入、消费和投资的可能性）。此外，投资资产的不确定性运动也需要由我们的模型捕捉。

现在，我们准备采用这个MDP的一个简单特例，它去除了许多现实世界的摩擦和复杂性，但仍保留了关键特征（特别是双重动态决策方面）。这个简单的特例是默顿投资组合问题（Merton 1969）的主题，他在1969年的一篇里程碑论文中提出并解决了这个问题。他公式的一个关键特征是时间是连续的，因此状态（基于资产价格）演化为连续时间随机过程，而动作（资产配置和消费）是连续进行的。我们在下一节中介绍他论文的重要部分。

### Merton投资组合问题及其解决  

现在，我们描述默顿投资组合问题并推导其解析解，这是数学经济学中最优雅的解决方案之一。该解决方案的结构将为我们提供关于资产配置和消费决策如何不仅依赖于状态变量，还依赖于问题输入的深刻直觉。

我们将时间记为$t$，并假设当前时间为$t=0$.假设你刚刚退休并且你将再活$T$年。因此，用上一节的语言来说，你余生将不会获得任何收入，除了从投资组合中提取资金的选项。再假设你还没有固定支出，如按揭贷款、订阅费等等，这意味着你所有的消费都是灵活/可选的，即你可以在任何时间点选择消费任何非负实数。以上所有假设都是不合理的（<span style="color: orange;">但如果在养老金的资产配置中，这是合理的！</span>），但有助于保持问题的简单性，以便于解析处理。  

我们将任何时间$t$的财富（记为$W_t$）定义为你的投资资产的总市场价值。请注意，由于没有外部的收入，并且所有消费都是可选的，$W_t$就是你的净资产。假设有固定数量的$n$个风险资产和1个无风险资产。如前所述，目标是通过在任何时间点的双重动作——资产配置和消费（消费等于在任何时间点从投资组合中提取的资金）——最大化你一生中消费的期望总效用。请注意，由于没有外部资金来源，并且所有从投资组合中提取的资金都会立即被消费，因此你永远不会向投资组合中添加资金。投资组合的增长只能来自投资组合中资产市场价值的增长。最后，我们假设消费效用函数是恒定相对风险厌恶（CRRA）的。  

为了便于阐述，我们将问题形式化，并针对$n=1$（即只有1个风险资产）的情况推导出Merton的优美的解析解。该解可以直观地推广到$n>1$个风险资产的情况。  
由于我们在连续时间中进行操作，风险资产遵循一个随机过程$S$,具体来说是一个几何布朗运动  
$$
dS_t=\mu S_t+\sigma S_t dZ_t,
$$
其中$\mu\in\mathbb{R},\sigma\in\mathbb{R}_+$是固定常数（注意，对于$n$个资产则分别为向量和矩阵）。  
无风险资产没有不确定性，并且在连续时间内有固定的增长率，因此在时间$t$时无风险资产$R_t$的估值由下式给出  

$$
dR_t=rR_tdt,
$$
其中$r\in\mathbb{R}$是一个固定常数。我们将单位时间内财富的消费记为$c(t,W_t)\geq0$，以明确消费决策通常取决于$t,W_t.$将时间$t$时分配给风险资产的财富比例记为$\pi(t,W_t).$注意，$c(t,W_t),\pi(t,W_t)$共同构成了时间$t$时的决策（MDP动作）。为了保持简洁，将$c(t,W_t),\pi(w,W_t)$分别写为$c_t,\pi_t$,但请在整个推导过程中认识到两者都是时间$t$和财富$W_t$的函数。  
最后，我们假设消费的效用函数为  

$$
U(x)=\frac{x^{1-\gamma}}{1-\gamma},
$$
其中风险厌恶参数$\gamma\neq1$.$\gamma$是CRRA系数，等于$\frac{-xU''(x)}{U'(x)}.$我们不会讨论$\gamma=1$时的CRRA效用函数即$U(x)=\log x$.  
由于我们假设没有向投资组合中添加资金，且没有买卖任何分数量的风险和无风险资产的交易成本，财富的时间演化应被概念化为分配比例$\pi_t$的连续调整和从投资组合中的连续提取（等于连续消费$c_t$）.因此从时间$t$到$t+dt$的财富变化$dW_t$由下式给出：  

\begin{equation}
dW_t=((r+\pi_t\cdot (\mu-r))\cdot W_t-c_t)dt+\pi_t\sigma W_t dZ_t. (\#eq:8-1)
\end{equation}

这是一个确定财富随机演化的伊藤过程。  
我们的目标是确定在任何时间$t$时的最优$(\pi(t,W_t),c(t,W_t))$,以最大化  

$$
\mathbb{E}\left[\int_t^T\frac{e^{-\rho(s-t)}\cdot c_s^{1-\gamma}}{1-\gamma} ds+\frac{e^{-\rho (T-t)}\cdot B(T)\cdot W_T^{1-\gamma}}{1-\gamma}|W_t \right],
$$
其中$\rho\geq0$是效用贴现率，用于考虑未来消费效用可能低于当前效用的事实，$B(\cdot)$被称为遗赠函数，可以视为你在时间$T$去世时留给家人的钱。我们可以为任意遗赠函数$B(T)$解决这个问题，但为了简单起见，我们考虑$B(T)=\epsilon^\gamma,0<\epsilon \ll 1$,意味着无遗赠。出于技术原因我们不将其设为0，这将在后面变得明显。  
我们应该将这个问题视为一个连续时间的随机控制问题，其中MDP定义如下：

- 时间$t$的状态为$(t,W_t)$  

- 时间$t$的动作为$(\pi_t,c_t)$  

- 时间$t<T$时的单位时间奖励为：  
  $$
  U(c_t)=\frac{c_t^{1-\gamma}}{1-\gamma}
  $$
  在终端时刻$T$的奖励为   
  $$
  B(T)\cdot U(W_T)=\epsilon^\gamma\cdot \frac{W_T^{1-\gamma}}{1-\gamma}
  $$
  时间$t$时的回报是累积贴现奖励  
  
  $$
  \int_t^T\frac{e^{-\rho(s-t)\cdot c_s^{1-\gamma}}}{1-\gamma} ds+\frac{e^{-\rho (T-t)}\cdot \epsilon^\gamma\cdot W_T^{1-\gamma}}{1-\gamma}
  $$
  
我们的目标是找到策略：$(t,W_t)\rightarrow (\pi_t,c_t)$,以最大化期望汇报。  
我们第一步是写出Hamilton-Jacobi-Bellman(HJB)方程，这是连续时间中的Bellman最优性方程的类比。我们将最优的价值函数记为$V^*$,使得时间$t$时财富$W_t$的最优价值为$V^*(t,W_t).$这里的HJB方程可以特化为下面的式子  

\begin{equation}
\max_{\pi_t,c_t}\left\{\mathbb{E}_t[dV^*(t,W_t)+\frac{c_t^{1-\gamma}}{1-\gamma}] dt\right\}=\rho V^*(t,W_t)dt (\#eq:8-2)
\end{equation}

现在对于$dV^*$使用伊藤引理，移走$dZ_t$的部分因为这是一个鞅，并且将等式两边都除以$dt$,以产生任意$0\leq t<T$的HJB方程的偏微分形式  

$$
\max_{\pi_t,c_t}\left\{ \frac{\partial V^*}{\partial t}+\frac{\partial V^*}{\partial W_t} \cdot ((\pi_t(\mu-r)+r)W_t-c_t)+\frac{\partial^2 V^*}{\partial W_t^2}\cdot \frac{\pi_t^2\sigma^2W_t^2}{2}+\frac{c_t^{1-\gamma}}{1-\gamma}\right\}=\rho\cdot V^*(t,W_t)  (\#eq:8-3)
$$
这个HJB方程由下面的终端条件  

$$
V^*(T,W_T)=\epsilon^\gamma \cdot\frac{W_T^{1-\gamma}}{1-\gamma}
$$

可以将\@ref(eq:8-3)写得更简洁一些：  

$$
\max_{\pi_t,c_t} \Phi(t,W_t;\pi_t,c_t)=\rho\cdot V^*(t,W_t) (\#eq:8-4)
$$
需要强调的是，我们处理的约束条件是$W_t>0,c_t\geq0,0\leq t<T.$  
为了找到最优的$\pi_t^*,c_t^*$，我们求得$\Phi$的一阶条件得  

\begin{align}
\pi_t^*&=\frac{-\frac{\partial V^*}{\partial W_t}\cdot (\mu-r)}{\frac{\partial^2 V^*}{\partial W_t^*}\cdot\sigma^2\cdot W_t} (\#eq:8-5)\\
c_t^*&=\left( \frac{\partial V^*}{\partial W_t} \right)^{-\frac{1}{\gamma}} (\#eq:8-6)
\end{align}

下面将\@ref(eq:8-5)和\@ref(eq:8-6)代入到\@ref(eq:8-3)中得到下面的式子，这就给了我们最优价值函数的PDE(显然下面的$w$应为$W_t$)：

$$
V^*_t-\frac{(\mu-r)^2}{2\sigma^2}\cdot\frac{(V^*_{w})^2}{V^*_{ww}}+V_w^*\cdot r\cdot w+\frac{\gamma}{1-\gamma}\cdot(V_w^*)^{\frac{\gamma-1}{\gamma}}=\rho\cdot V^* (\#eq:8-7)
$$
终值条件仍是

$$
V^*(T,W_T)=\epsilon^\gamma \cdot\frac{W_T^{1-\gamma}}{1-\gamma}
$$

$\Phi$的二阶条件在以下假设下得以满足$c^*_t>0,W_t>0,\frac{\partial^2V^*}{\partial W_t^2}<0$对于所有的$0\leq t<T$,稍后将会证明这些条件在我们推导的解中得到满足，并且$U(\cdot)$是凹函数，即$\gamma>0.$

接下来我们希望将PDE\@ref(eq:8-7)简化为ODE，这样可以进行简单的求解。为此我们假设解的形式是一个关于时间的确定性函数$f(t):$

$$
V^*(t,W_t)=f(t)^\gamma\cdot\frac{W_t^{1-\gamma}}{1-\gamma}
$$

将猜解的形式代入PDE当中，可以得到


\begin{align}
&f'(t)=\nu f(t)-1,\\
\text{where }&\nu=\rho-(1-\gamma)\left(\frac{(\mu-r)^2}{2\sigma^2\gamma}+r  \right)
\end{align}

此时我们注意到遗赠函数$B(T)=\epsilon^\gamma$对拟合猜测解非常有用，因此该ODE的边界条件为$f(T)=\epsilon$.由此可以解得

\begin{equation}
f(t)=
\begin{cases}
\frac{1+(\nu\epsilon-1)e^{-\nu(T-t)}}{\nu}&\text{ for }\nu\neq 0\\
T-t+\epsilon&\text{ for }\nu=0
\end{cases}
\end{equation}

于是最优分配策略和消费策略如下


\begin{align}
\pi^*(t,W_t)&=\frac{\mu-r}{\sigma^2\gamma} (\#eq:8-14)\\
c^*(t,W_t)&=
\begin{cases}
\frac{\nu W_t}{1+(\nu\epsilon-1)e^{-\nu(T-t)}} &\text{ for }\nu\neq0\\
\frac{W_t}{T-t+\epsilon}&\text{ for }\nu=0
\end{cases}
(\#eq:8-15)
\end{align}

同时有了$f(t)$的表达式。$V^*(t,W_t)$的表达式也就迎刃而解了。注意$f(t)>0$对于所有$0\leq t<T,\forall \nu,$确保了$W_t>0,c_t^*>0,\frac{\partial^2V^*}{\partial W_t^2}<0$.这确保了约束条件$W_t>0,c_t\geq0$得以满足，同时二阶条件得以满足。解决Merton的投资组合问题的一个非常重要的经验教训是HJB公式是关键，这种解法为类似的连续时间随机控制问题提供了模板。

### Merton投资组合问题解的直觉

$\pi^*$是个常数，代表无论财富如何年龄如何，都该将相同的财富比例投资于风险资产。
而$c^*$中风险资产的超额回报$(\mu-r)$出现在分子上，$\sigma,\gamma$出现在坟墓上，波动率越大或者风险厌恶程度更高，自然会减少投资于风险资产，而当我们还年轻时我们希望消费的少一些，但是快死了的时候会增加消费（因为最优策略是死得一贫如洗，假设没有遗产）。

将最优策略\@ref(eq:8-14)和\@ref(eq:8-15)代入财富过程\@ref(eq:8-1)，可以得到进行最优资产配置和最优消费的时候，财富过程为

$$
dW_t^*=(r+\frac{(\mu-r)^2}{\sigma^2\gamma}-\frac{1}{f(t)})\cdot W_t^* \cdot dt+\frac{\mu-r}{\sigma\gamma}\cdot W_t^* \cdot dZ_t (\#eq:8-17)
$$
是一个对数正态过程，对数正态波动率是常数，对数正态飘逸与财富无关但依赖于时间，我们可以解得

$$
\mathbb{E}[W_t^*]=W_0\cdot e^{(r+\frac{(\mu-r)^2}{\sigma^2\gamma})t}\cdot e^{-\int_0^t\frac{du}{f(u)}}
$$


### 离散时间资产配置

在这一节中，将会讨论问题的离散时间版本，这使得问题具有解析可解性。类似于Merton的连续时间投资组合问题，我们在时间$t=0$是拥有财富$W_0,$在每个离散时间步长$t=0,1,...,T-1$时我们可以在没有约束没有交易成本的情况下，将财富$W_t$分配到风险资产和无风险资产的投资组合中。风险资产的回报在每个时间步长内为常数$r.$

我们假设在任何$t<T$时没有消费财富并且在时间$T$时会清算并消费财富$W_T.$因此我们的目标是通过在每个$t=0,1,...,T-1$时动态地分配风险资产$x_t\in\mathbb{R}$和剩余的$W_t-x_t$无风险资产，最大化最终时间步$t=T$时的期望财富效用。假设单步折现因子为$\gamma$,最终时间步$T$的财富效用由以下CARA函数给出  

$$
U(W_T)=\frac{1-e^{-aW_T}}{a}\text{ for some fixed } a\neq 0
$$
因此问题变成了在每个$t=0,1,...,T_1$时通过选择$x_t$来最大化

$$
\mathbb{E} \left[\gamma^{T-t}\cdot \frac{1-e^{-aW_T}}{a}\mid (t,W_t)\right]
$$
等价于最大化  

$$
\mathbb{E} \left[-\frac{e^{-aW_T}}{a}\mid (t,W_t)\right] (\#eq:8-19)
$$
我们将这个问题表述为一个连续状态和连续动作的离散时间有限时域MDP，并准确指定其转移状态、奖励和折现因子，然后我们的目标是求解MDP控制问题找到最优策略。

有限时域MDP的终止时间为$T$,因此所有时间$t=T$的状态都是终止状态。时间步$t=0,1,...,T$的状态$s_t\in\mathcal{S}_t$包含财富$W_t.$决策$a_t\in\mathcal{A}_t$是对风险投资的投资量$x_t,$因此每个时间步对无风险投资的投资量为$W_t-x_t.$时间步$t$的确定性策略$\pi_t$记为$\pi_t(W_t)=x_t,$同样，时间步$t$时最优确定性策略$\pi_t^*$记为$\pi_t^*(W_t)=x^*_t.$

我们将$t$到$t+1$的风险资产单步回报的随机变量记为$Y_t\sim N(\mu,\sigma^2)$对于所有的$t=0,1,...,T-1.$因此

$$
W_{t+1}=x_t\cdot(1+Y_t)+(W_t-x_t)\cdot (1+r)=x_t\cdot(Y_t-r)+W_t\cdot(1+r) (\#eq:8-20)
$$

MDP的奖励在每个$t=0,1,...,T-1$时为0，因此基于上述简化的目标\@ref(eq:8-19),MDP在$t=T$时的奖励设置为随机量$\frac{-e^{aW_T}}{a}.$我们将MDP的折现因子设置为$\gamma=1,$在时间$t$时的价值函数（给定策略$\pi=(\pi_0,\pi_1,...,\pi_{T-1})$）记为

$$
V_t^\pi(W_t)=\mathbb{E}_\pi\left[ -\frac{e^{-aW_T}}{a}\mid (t,W_t) \right]
$$
在时间$t$时的最优价值函数记为

$$
V_t^*(W_t)=\max_{\pi} V_t^\pi(W_t)=\max_\pi\left\{\mathbb{E}_\pi\left[ -\frac{e^{-aW_T}}{a}\mid (t,W_t) \right] \right\}
$$
贝尔曼最优方程为(当$t=0,1,...,T-2$时)

$$
V_t^*(W_t)=\max_{x_t}Q_t^*(W_t,x_t)=\max_{x_t}\left\{\mathbb{E}_{Y_t\sim N(\mu,\sigma^2)}\left[ V_{t+1}^*(W_{t+1}) \right] \right\}
$$
且

$$
V_{T-1}^*(W_{T-1})=\max_{x_{T-1}}Q_{T-1}^*(W_{T-1},x_{T-1})=\max_{x_{T-1}}\left\{\mathbb{E}_{Y_{T-1}\sim N(\mu,\sigma^2)}\left[ \frac{-e^{-aW_T}}{a} \right] \right\}
$$

其中$Q_t^*$是时间$t$时的最优动作价值函数。

我们通过对最优价值函数的形式进行合理猜测，得到

$$
V_t^*(W_t)=-b_t e^{-c_tW_t} (\#eq:8-21)
$$
其中$b_t,c_t$与财富$W_t$无关，接下来我们使用这个最优价值函数的形式来表达贝尔曼最优性方程

$$
V_t^*(W_t)=\max_{x_t}\left\{\mathbb{E}_{Y_t\sim N(\mu,\sigma^2)}\left[ -b_{t+1}e^{-c_{t+1}W_{t+1}} \right] \right\}
$$
利用公式\@ref(eq:8-20)可以将其改写为

$$
V_t^*(W_t)=\max_{x_t}\left\{\mathbb{E}_{Y_t\sim N(\mu,\sigma^2)}\left[ -b_{t+1}e^{-c_{t+1}(x_t(Y_t-r)+W_t(1+r))} \right] \right\}
$$

这个指数形式的期望值在正态分布下计算结果为

$$
V_t^*(W_t)=\max_{x_t}\left\{-b_{t+1}\exp\left\{-c_{t+1}(1+r)W_t-c_{t+1}(\mu-r)x_t+c_{t+1}^2 \frac{\sigma^2}{2} x_{t}^2\right\} \right\} (\#eq:8-22)
$$

又由于$V_{t}^*(W_t)=\max_{x_t}Q^*_t(W_t,x_t),$从上面的公式中可以推断出最优动作价值函数$Q^*_t(W_t,x_t)$的函数形式为

$$
Q^*_t(W_t,x_t)=-b_{t+1}\exp\left\{-c_{t+1}(1+r)W_t-c_{t+1}(\mu-r)x_t+c_{t+1}^2 \frac{\sigma^2}{2} x_{t}^2\right\} (\#eq:8-23)
$$

由于贝尔曼最优性方程\@ref(eq:8-22)右侧涉及对$x_t$的最大化操作，我们可以认为最大化操作内的项对于$x_t$的偏导数是0，这使得我们能够将最优分配$x_t^*$表示为$c_{t+1}$的函数，如下所示

$$
-c_{t+1}(\mu-r)+\sigma^2c_{t+1}^2x_t^*=0
$$
即

$$
x_t^*=\frac{\mu-r}{\sigma^2 c_{t+1}} (\#eq:8-24)
$$
代入贝尔曼最优性方程\@ref(eq:8-22)可得

$$
V_{t}^*(W_t)=-b_{t+1}\exp\left\{ -c_{t+1}(1+r)W_t-\frac{(\mu-r)^2}{2\sigma^2}\right\}
$$
但由于

$$
V_{t}^*(W_t)=-b_t\exp\{-c_tW_t\}
$$
就可以得到如下递推方程

$$
b_t=b_{t+1}\exp\left\{-\frac{(\mu-r)^2}{2\sigma^2}\right\},\quad  c_t=c_{t+1}(1+r)
$$
我们还可以通过知道在$t+T$终端时刻的MDP的奖励$\frac{-e^{-aW_T}}{a}$即终端时刻的财富效用来计算$b_{T-1},c_{T-1},$然后递推得到$b_t,c_t$的值。

在$t=T-1$时：

$$
V_{T-1}^*(W_{T-1})=\max_{x_{T-1}}\left\{\mathbb{E}_{Y_{T-1}\sim N(\mu,\sigma^2)}\left[ \frac{-e^{-aW_T}}{a} \right] \right\}
$$
通过公式\@ref(eq:8-20)可以写为

$$
V_{T-1}^*(W_{T-1})=\max_{x_{T-1}}\left\{\mathbb{E}_{Y_{T-1}\sim N(\mu,\sigma^2)}\left[ \frac{-\exp\{-a(x_{T-1}(Y_{T-1}-r)+W_{T-1}(1+r))\}}{a} \right] \right\}
$$

利用矩母函数的相关知识可以将其改写为

$$
V_{T-1}^*(W_{T-1})=-e^{-\frac{(\mu-r)^2}{2\sigma^2}}-\frac{a(1+r)W_{T-1}}{a}
$$
因此

$$
b_{T-1}=\frac{e^{-\frac{(\mu-r)^2}{2\sigma^2}}}{a},\quad c_{T-1}=a(1+r)
$$
递推得到当$t=0,1,...,T-2$时（当然$t=T-1$也满足下面的递推式）

$$
b_t=\frac{e^{-\frac{(\mu-r)^2(T-t)}{2\sigma^2}}}{a},\quad c_t=a(1+r)^{T-t}
$$

将$c_{t+1}$带入到\@ref(eq:8-24)即可得到最优策略的解

$$
\pi_t^*(W_t)=x_t^*=\frac{\mu-r}{\sigma^2a(1+r)^{T-t-1}},\text{ for all }t=0,1,...,T-1  (\#eq:8-25)
$$
请注意最优策略在时间步$t$时不依赖于状态$W_t.$因此对于固定时间的最优策略$\pi_t^*$是一个常数确定性的策略函数。

将$b_t,c_t$的解代入公式\@ref(eq:8-21)即可得到最优价值函数的解：

$$
V_t^*(W_t)=\frac{-e^{-\frac{(\mu-r)^2(T-t)}{2\sigma^2}}}{a}\cdot e^{-a(1+r)^{T-t}W_t} \text{ for all }t=0,1,...,T-1 (\#eq:8-26)
$$
将$b_{t+1},c_{t+1}$代入到公式\@ref(eq:8-23)可以得到最优动作价值函数的解（对于所有的$t=0,1,...,T-1$）

$$
Q^*_t(W_t,x_t)=\\
\frac{-e^{\frac{-(\mu-r)^2(T-t-1)}{2\sigma^2}}}{a}\exp\left\{-a(1+r)^{T-t}W_t-a(\mu-r)(1+r)^{T-t-1}x_t+\frac{(a\sigma(1+r)^{T-t-1})^2}{2}x_t^2  \right\}
$$

### 现实世界的应用

上面的讨论和设置通过简化假设提供了分析的可解性。使得问题具有解析可解性的具体简化假设包括：

- 资产回报的正态分布

- CRRA/CARA假设

- 无摩擦市场交易（没有交易成本没有约束，价格/分配数量/消费都是连续的）

但是，现实世界中的动态资产配置和消费问题并不像我们讨论的那么简单和清晰。实际的资产价格波动更加复杂，效用函数不一定符合简单的 CRRA/CARA 形式。实际上，交易通常发生在离散空间中——资产价格、分配数量和消费量往往是离散的。此外，当我们改变资产配置或清算部分投资组合以进行消费时，会产生交易成本。更重要的是，交易并不总是在连续时间内进行——通常会有特定的交易窗口或交易限制。最后，许多投资是流动性差的（如房地产），或者在特定的时间之前无法清算（如退休基金），这对从投资组合中提取资金进行消费构成了重大约束。因此，尽管价格、分配数量和消费可能接近连续变量，但上述摩擦使得我们无法在简化示例中那样利用微积分的便利。

考虑到上述现实世界的因素，我们需要利用动态规划——更具体地说，利用近似动态规划，因为现实世界的问题通常涉及大规模的状态空间和行动空间（即使这些空间不是连续的，它们通常接近连续）。对价值函数的适当函数逼近是解决这些问题的关键。实现一个完整的现实世界投资和消费管理系统超出了本书的范围，但我们可以实现一个具有足够理解的示例，展示如何实现一个完整的现实世界应用。我们要实现的设置包括：

- 一个风险资产和一个无风险资产；

- 有限时间步数（类似于上一节的离散时间设置）；

- 在有限时间结束之前不进行消费（即不从投资组合中提取资金），因此折现因子设置为1；

- 风险资产回报的任意分布；

- 随时间变化的无风险资产回报；

- 任意效用函数；

- 在每个时间步，有限数量的风险资产投资选择；

- 初始财富的任意分布。






