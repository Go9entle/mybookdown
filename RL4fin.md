# Markov过程 {# Markov}

本书的主题是“序列不确定下的序列决策”，在本章中将暂时忽略“序列决策”方面而只关注”序列不确定性“。

## 过程中的状态概念

$S_t$是过程在时间$t$时的状态。特别地，我们对于下一时刻的状态$S_{t+1}$的概率感兴趣，如果已知现在的状态$S_t$和过去的状态$S_0,S_1,...,S_{t-1}$，我们对$P\{S_{t+1}|S_t,S_{t-1},...,S_0\}$感兴趣。

## 通过股票价格的例子理解Markov性

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

## Markov过程的正式定义

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

注意上述规范中$P$的参数没有时间索引$t$（因此称为时间齐次）。此外注意到一个非时间齐次的Markov过程可以通过将所有状态和时间索引$t$来结合转换为齐次Markov过程。这意味着一个非时间齐次的Markov过程的原始状态空间是$\mathcal{S}$那么对应的时间齐次Markov过程的状态空间是$\mathbb{Z}_{\geq0}\times\mathcal{S}.$
