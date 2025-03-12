# 最优多维再保险与多元风险厌恶效用 {#mulreinsurance } 

本章开始于2025年2月26日。

## 共同冲击依赖结构下的最优多维再保险政策 {#OMR }  

>论文原文*Optimal multidimensional reinsurance policies under a common shock dependency structure*点击[这里](https://go9entle.github.io/myresearch/OMR.pdf)查看。


**再保险**是指以下保险公司为避免自己遭遇损失和风险而向另一家保险公司投保的过程。


在本文中，我们考虑一家在多个相关业务线上运营的保险公司。我们假设每条业务线的风险过程都服从Cramér–Lundberg风险模型，并采用共同冲击（common shock）依赖结构来描述不同业务线可能会同时发生索赔的情况。
作者考虑一种共同冲击依赖结构用于对再保险公司的盈余过程进行建模，目标是针对再保险策略向量最大化目标函数，即随时间积分的预期折现盈余水平。  

保险公司通过再保险策略向再保险公司转移部分风险，本文的目标是利用动态规划方法，最大化某一目标函数（即随时间积分的期望贴现盈余）。我们将最优目标函数（价值函数）描述为满足相应Hamilton–Jacobi–Bellman（HJB）方程及一定边界条件的唯一解。此外文章还提出了一种数值算法，用于计算目标函数的最优解，并得到相应的最优再保险策略。

随机控制是一个重要的研究领域，在保险业中有广泛的应用。特别是，随机控制被广泛用于控制保险公司基于再保险策略的风险过程。再保险策略是保险公司用来将部分风险转移给其他保险公司的方式。保险公司面临的一个重要挑战是优化其再保险策略。文献中广泛使用的一种解决此问题的方法是最小化与再保险策略相关的破产概率。这一方法最早由Schmidli 研究，该研究假设了Cramér–Lundberg风险过程和比例再保险策略。Schmidli 提出的方法被Hipp和Vogt 扩展到损失超额再保险策略。Hipp和Vogt 研究的相同问题也在Schmidli 和Taksar与Markussen 的研究中被考虑，但其假设风险过程是扩散过程。近年来，Cani和Thonhauser 以及Cani 使用不同的目标函数来找到最优的再保险策略。在这些工作中，Højgaard和Taksar 以及Højgaard和Taksar 提出的期望折现盈余水平函数被作为目标函数。此外，还研究了各种优化技术，如Beveridge、Dickson和Wu ，Meng和Siu ，Azcue和Muler ，Eisenberg和Schmidli ，Tamturk和Utev，Tan、Wei、Wei和Zhuang ，Preischl和Thonhauser 以及Salah和Garrido 等，来找到最优的再保险策略。

然而，所有上述工作仅关注活跃于单一业务领域的保险公司，这些公司在其所有风险上使用相同类型的再保险策略。然而，在实践中，大多数保险公司通常不仅活跃于一个业务领域。这些领域之间可能是相关的，单独控制每个领域将无法得到全局最优结果。最近，Masoumifard和Zokaei 使用生存概率作为目标函数，找到一个保险公司在多个独立领域中操作时的最优动态再保险策略向量。

在这项工作中，我们考虑了一种常见冲击依赖结构来建模再保险公司的盈余过程。与Cani和Thonhauser 相似，我们将期望折现盈余水平随时间的积分作为目标函数。我们的目标是根据再保险策略向量最大化该目标函数。

### Common shock model  

考虑一家运营于$n$个相互依赖保险业务线的保险公司。事实上，索赔可能同时发生在多个业务线上，例如，一次车祸可能对汽车和司机造成损害。因此，我们假设有$m$个源，其中每个源的发生会导致一个或多个业务线发生索赔。图\@ref(fig:csm)

<div class="figure">
<img src="https://Go9entle.github.io/picx-images-hosting/1740566499491.7snclf65t8.webp" alt="An example of common shock model with m = 4 sources and n = 3 lines."  />
<p class="caption">(\#fig:csm)An example of common shock model with m = 4 sources and n = 3 lines.</p>
</div>
需要注意的是源1和源4都会在第一条业务线发生索赔，但这些索赔的概率分布可能不同。

更准确地说，考虑一个概率空间$(\Omega,\mathcal{E},P)$,在此空间中我们考虑具有参数$\beta_i$的独立Poisson过程$\{N_i(t):t\geq0\}$,用于描述源$i\in\{1,2,...,m\}$中事件的发生频率。将源$i$在业务线$j$上的第$k$次索赔大小表示为随机变量$Y_{ijk}$.假设$\{Y_{ijk}:i\in\{1,2,...,m\},j\in A_i,k\in\mathbb{N}\}$对于第$k$次是独立的随机变量，其中$A_i\subset\{1,2,...,n\}$是源$i$影响的业务线的集合，$Y_{ijk}$具有累积分布函数$F^Y_{ij}$和有限的均值$\mu_{ij}$。因此源$i$(固定$i$)在时间$t$之前造成的总索赔金额为

$$
\sum_{k=1}^{N_i(t)}\sum_{j\in A_i}Y_{ijk},\quad i=1,2,...,m
$$

给定初始资本$x$,保险公司在时间$t$的盈余为  
$$
X(t)=x+pt-\sum_{i=1}^m\sum_{k=1}^{N_i(t)}\sum_{j\in A_i}Y_{ijk}, (\#eq:omr1)
$$
其中$p$是保费率，并使用带有相对安全载荷因子$\eta_j>0$的期望值原则来计算  
$$
p=\sum_{j=1}^np_j,\quad p_j=(1+\eta_j)\sum_{i=1}^m\beta_i\mu_{ij}I_{A_i}(j), (\#eq:omr2)
$$
其中$I_{A_i}(x)$是示性函数。上面这个式子也许需要更进一步理解：$\beta_i$是源$i$发生的平均次数,$\mu_{ij}$是所有$k$次索赔的平均索赔大小。

- $p_j$是业务线$j$的保费；

- $\eta_j$是与业务线$j$相关的相对安全载荷因子，表示对风险的附加费用；

- $\beta_i$是源$i$的平均发生次数；

- $\mu_{ij}$是源$i$在业务线$j$上的平均索赔大小。

- $I_{A_i}(j)$是示性函数，表示如果业务线 $j$ 受源 $i$ 影响，则 $I_{A_i}(j)=1$，否则为 $0$.


#### Common Shock Model2

考虑另一类多维风险过程的保险。

假设有$M$类业务的索赔到达被假设为具有共同冲击的相关的泊松过程。第$i$类业务的盈余过程$\{U_i(t)\}_{t\geq 0}$描述为，对于$i=1,2,...,M$,

$$
U_i(t)=u_i+c_it-\sum_{k=1}^{N_{ii}(t)}Y_{i,k}-\sum_{k=1}^{N_c(t)}Z_{i,k}
$$
其中初始资本和保费率分别用$u_i=U_i(0)\geq0,c_i\geq0$表示，采用常规约定对于任何$j>k,\sum_{i=j}^k=0$,并且由于我们关心的是多维过程，$m>1.$

这些计数过程$N_{11}(t),...,N_{MM}(t),N_c(t)$是具有速率$\lambda_{11},...,\lambda_{MM},\lambda_{c}$的泊松过程。

对于每个$i=1,2,...,M,\{Y_{i,k}\}_{k=1}^{\infty}$是独立同分布的正随机变量序列，具有共同的密度函数$f_{ii}(x_i).$

此外,$\{(Z_{i,k},...,Z_{M,k})\}_{k=1}^\infty$是一个具有独立同分布的$M$维正随机向量序列，具有共同的联合密度函数$f_C(x_1,...,x_M).$进一步假设$\{N_{ii}(t)\}_{t\geq0},\{N_c(t)\}_{t\geq0},\{Y_{i,k}\}_{k=1}^\infty(i=1,2,...,M)$和$\{(Z_{i,k},...,Z_{M,k})\}_{k=1}^\infty$互相独立。对于第$i$个风险过程破产时间定义为$\tau_i=\inf\{t\geq 0\mid U_i(t)<0\}$对于$i=1,2,...,M.$此外，保证破产不是确定事件的正安全载荷因子条件是$\theta_i>0,$其中$c_i=(1+\theta_i)(\lambda_{ii}\mathbb{E}[Y_{i,1}]+\lambda_c\mathbb{E}[Z_{i,1}])$

该模型具有以下解释

1. 对于每个固定的$i=1,2,...,M,$过程$\{N_{ii}(t)\}_{t\geq0}$计算第$i$类业务到时刻$t$为止所面临的通常索赔次数，这些索赔只会在某一条业务线产生但不会影响其他业务线，这些索赔发生的情况下，结果的索赔大小由序列$\{Y_{i,k}\}_{k=1}^\infty$给出。

2. 过程$\{N_c(t)\}_{t\geq0}$计算$M$条业务线所面临的“共同冲击”的次数，共同冲击可以解释为一种自然灾害导致不同种类的索赔或是影响所有$M$条业务线的索赔，第$k$次共同冲击导致第$i$条业务线的索赔大小为$Z_{i,k},$并且这些索赔大小$Z_{1,k},...,Z_{M,k}$可能是相关变量。




#### Reinsurance  
在本小节中，将会定义一个拥有$n$条业务线的公司再保险策略。首先，考虑滤波$\mathcal{F}=\{\mathcal{F}_t:t\geq0\}$其中$\mathcal{F_t}$是由$\{X(s):0\leq s\leq t\}$生成的$\sigma$-代数。

再保险策略是一个多维的随机过程$\boldsymbol{U}=\{\boldsymbol{U}(t)=(U_1(t),...,U_n(t)):t\geq0\}.$如果在时刻$t=t_1$发生源$i$的索赔，大小为$Y=\sum_{j\in A_i}Y_{ijk}$,那么再保险公司将承担$Y-\sum_{j\in A_i} r_j(U_j(t_1), Y_{ijk})$,其中函数$0\leq r_j(u,y)\leq y$是连续且关于$y$是递增的。我们说$\boldsymbol{U}$是可接受的，如果对于$j=1,2,...,n$和$t\geq0,y\geq0$函数$(\omega,t,y)\rightarrow r_j(U_j(\omega,t),y)$是$\mathcal{E\times B\times B}$可测的，并且函数$\omega\rightarrow \sum_{j\in A_i}r_j(U_j(\omega,t),y)$是$\mathcal{F_t}$可测的。我们将所有可接受策略记为$\boldsymbol{\mathcal{R}}.$本文将考虑下面两种再保险合约：  

1. 比例再保险：再保险公司承担比例 $u$ 的索赔金额，即$r^P(u,y)=uy,\quad u\in \mathcal{U}^P=[0,1]$.

2. 损失超额再保险（XL）：再保险公司承担至 $u$ 的索赔金额，但最多只承担 $y$，即$r^{XL}(u,y)=\min(u,y),\quad u\in\mathcal{U}^{XL}=[0,\infty].$

由公式\@ref(eq:omr1),由再保险策略$\boldsymbol{U}$控制的盈余过程就写为

$$
X_{\boldsymbol{U}}(t)=x+\int_0^tp(\boldsymbol{U}(s))ds-\sum_{i=1}^m\sum_{k=1}^{N_i(t)}\sum_{j\in A_i} r_j(U_j(\tau_{ik}^-),Y_{ijk}), (\#eq:omr3)
$$
其中$\tau_{ik}$是源$i$第$k$次发生索赔的时间。使用期望值原则计算保费

$$
p(\boldsymbol{u})=p-\sum_{j=1}^nq_j(u_j),\quad \boldsymbol{u}\in \mathcal{U}
$$
并且业务线$j$的再保险保费是$q_j(u_j)$

$$
q_j(u_j)=(1+\theta_j)\sum_{i=1}^m\beta_i\mathbb{E}(Y_{ijk}-r_j(u_j,Y_{ijk}))I_{A_i}(j)
$$

其中$\theta_j>\eta_j$是安全载荷因子，$p$在\@ref(eq:omr2)中定义，$r_j(u_j,Y_{ijk})$是保险公司按照再保险策略需要承担的索赔金额，且$\mathcal{U}=\mathcal{U}_1\times\mathcal{U}_2\times ...\times\mathcal{U_n}$是$\mathbb{R}^n$中一个子集。对于比例再保险$U_j=[0,1]$而对于损失超额再保险$U_j=[0,\infty].$

#### The value function

给定一个再保险策略$\boldsymbol{U}$和初始财富盈余$x\geq0$,类似于Cani & Thonhauser, 如下定义目标函数


\begin{align}
V_{\boldsymbol{U}}(x)&=\mathbb{E}\left(\int_0^{\tau_{\boldsymbol{U}}} e^{-\delta x}X_{\boldsymbol{U}}(s)ds \mid X_{\boldsymbol{U}}(0)=x  \right)\\
&=\mathbb{E}_x\left(\int_0^{\tau_{\boldsymbol{U}}} e^{-\delta x}X_{\boldsymbol{U}}(s)ds  \right), (\#eq:omr4)
\end{align}

其中$\tau_{\boldsymbol{U}}$是破产时刻

$$
\tau_{\boldsymbol{U}}=\inf\{t\geq0:X_{\boldsymbol{U}}(t)<0\},
$$
$\delta>0$是折现率。该函数称为随时间积分的预期折现盈余水平，价值函数则由式\@ref(eq:omr5)给出  

$$
V(x)=\sup_{\boldsymbol{U}\in\boldsymbol{R}} V_{\boldsymbol{U}}(x), (\#eq:omr5)
$$
由Cani & Thonhauser的研究可以得知，价值函数$V(x)$是严格递增、局部Lipschitz连续，因此是绝对连续的，并且对于$x\geq0,$有

$$
\frac{x}{\delta}<V(x)\leq \frac{x}{\delta}+\frac{P}{\delta^2}. (\#eq:omr6)
$$

### HJB equation  

为了能够使用动态规划的方法解决\@ref(eq:omr5),并进一步确保解的最优性，我们首先需要找到与$V(x)$相关的Hamilton-Jacobi-Bellman方程。

**引理1**  

价值函数$V(x)$也就是下面方程的解：

$$
\sup_{\{\boldsymbol{u}\in\mathcal{U}:p(\boldsymbol{u})\geq0\}} H_g(x,\boldsymbol{u})=0, (\#eq:omr7)
$$
其中

$$
H_g(x,\boldsymbol{u})=x+p(\boldsymbol{u})g'(x)-(\delta+\beta)g(x)+\sum_{i=1}^m\beta_i\int_0^x g(x-z)d F_i^{\boldsymbol{u}}(z).
$$
且$\beta=\sum_{i=1}^m\beta_i$,$F_i^{\boldsymbol{u}}$是$\sum_{j\in A_i}r_j(u_j,Y_{ijk})$的累积分布函数。




## 应用于ESG投资的多元风险厌恶效用 {#mrau}

>论文原文*Multivariate risk aversion utility, application to ESG investments*点击[这里](https://go9entle.github.io/myresearch/MRAU.pdf)查看。

**ESG**(Environmental, social, and governance) 是优先考虑环境问题、社会问题和公司治理的投资原则的简写。考虑ESG的投资有时被称为*负责任的投资*（responsible investing）或者*影响力投资*（impact investing）。

本文使用多变量或多属性效用的概念，将不同的风险规避水平与不同的财富来源（例如行业、股票、资产类别）联系起来。在此背景下，我们从对绿色和棕色股票具有不同风险规避水平的投资者的角度探讨环境、社会和公司治理 (ESG) 投资的主题。

期望效用理论（EUT）的基础自从Von Neumann和Morgenstern于20世纪50年代初的开创性工作以来就存在了，Neumann 和 Morgenstern（1953）。该理论迅速成为投资组合优化的主要分支之一。Merton在70年代基于几何布朗运动并使用其中一个广泛的效用函数家庭——超双曲绝对风险厌恶（HARA）效用，提出了连续时间下的直观解，Merton（1971）。我们在EUT框架下使用一类多元效用，类似于Cobb-Douglas效用函数（参见Rasmussen（2011）第3.4节，Campi和Owen（2011）），我们称之为多元常数相对风险厌恶（M-CRRA）。我们认为投资者在财富的不同来源上可能不会以相同的风险厌恶水平来考虑或对待所有的风险来源。市场中的某些风险来源（例如投资组合中的行业或资产类别）可能受到更大的关注、反感或带来较低的满意度；因此，投资者应具备为任何给定的风险来源附加合适风险厌恶水平的灵活性。有关各种资产类别的实证风险厌恶水平，可以参考Conine、McDonald和Tamarkin（2017）表1中的研究。多属性效用函数中对不同来源和满意度水平的研究也已被探讨，参见Kihlstrom和Mirman（1974）以及Dorfleitner和Krapp（2007）。在这种背景下，我们可以将资产类别（或一组股票）的风险视为一个属性。

尽管我们的假设较为简单，但多元效用也可以解释为：一方面，投资者对资产类别的偏好；另一方面，对风险的偏好。这两者是不同的概念，但在效用理论中，风险偏好常常超越了资产类别偏好，这使得后者只是简单地在分析中包括或排除资产类别，这通常对于投资者来说并不理想。将资产类别偏好与风险偏好结合的一个有趣例子是环境、社会和公司治理（ESG）投资主题。在过去20年中，ESG驱动的投资规模已增长到超过30万亿美元的资产管理规模，全球可持续投资资产在2018年飙升至30万亿美元（2019年），并且未来呈现出明显的增长趋势。为了ESG的目的，企业在三个方面进行评估：首先是它们在环境目标方面的工作，其次是它们对某些社会运动的支持，最后是公司是否按照多样性、公平性和包容性运动进行治理，即ESG（环境、社会和治理）。许多方法已经被设计用来衡量企业对ESG目标的对齐情况，这被称为“ESG评分”，参见Whelan（2021）对超过1000项不同评分研究的回顾。除了企业/公司/股票的“ESG评分”，我们还应该考虑投资者的“ESG口味”（资产类别偏好，属性偏好），这可以解释为投资者对其投资组合中股票的ESG评分的喜好或反感。

本文使用多变量或多属性效用的概念，将不同的风险规避水平与不同的财富来源（例如行业、股票、资产类别）联系起来。

本文的目标是通过风险厌恶水平将投资者的ESG偏好纳入投资组合优化分析中。在ESG文献中，资产类别偏好和风险厌恶这两个话题通常被分开处理，本文提供一种更简化的替代方法，不仅具有闭式解还具备潜在的推广性。为了简化问题，将股票分为<span style="color: green;">绿色股票</span>和<span style="color: brown;">棕色股票</span>（这两种属性的股票对投资者的吸引程度不同），因此有两种可能的风险厌恶水平。这里的“绿色”指的是公司为社会创造积极的外部效益（即具有良好的ESG评分），而“棕色”则带来负面外部效益，即低ESG评分。本文中的结果可以适应ESG评分与风险厌恶水平之间更灵活的关系。

关于ESG投资组合优化主要有两条研究路线，一条依赖于预期效用理论（EUT），另一条则基于均值方差理论（MVT）。于ESG投资组合优化这一主题的新颖性以及其不断发展，下面简要提及一些当前的研究。

在EUT分支中，作者假设代理人在喜欢财富的同时，也从持有绿色股票中获得效用，从持有棕色股票中获得负效用。该分支的一个例子是P\'astor等人（2021）的研究，在离散时间中，作者考虑$N$股票和形如$u(x)=-\exp(-\gamma x-b'\pi)$的指数效用函数。这里的$x$代表财富，$\gamma$是绝对风险厌恶系数，$\pi$是投资于风险资产的财富比例向量，$b$是代理人从股票持有中获得的非金钱收益向量。收益向量$b$是代理人特定和公司特定的组成部分$b=dg$，其中$g$是每个公司特定的（即ESG评分）$N\times 1$向量，$d\geq0$是衡量代理人ESG偏好的标量。类似地，Dorfleitner and Nguyen（2017）使用了一个混合效用函数：$u(x,s)=(1-\alpha)u_F(x)+\alpha u_S(s),$其中$\alpha\in[0,1]$可以解释为相比于财务方面投资者的ESG目标的重要性，$x$是财富，$s$是ESG回报，计算方式为$s=x_0(\sum\pi_i(1+s_i)-1)$,其中$s_i$是公司$i$的ESG评分。

至于MVT，Schmidt（2020）考虑了一个目标函数:$\frac{1}{2}\lambda\sigma^2_p-\mu_p+\gamma\delta'\pi$,其中$\mu_p,\sigma_p$分别是投资组合的均值和标准差，$\gamma$捕捉了ESG对投资者的重要性，而$\delta$是一个包含投资组合成分ESG的向量。在Gasser et al. (2017) 中，作者使用了类似的目标函数:$-\beta \sigma_P^2+\alpha\mu_P+\gamma\theta,$其中$\theta=\sum\pi_i\theta_i$表示投资组合的社会责任评分，而$\gamma$表示投资者的ESG偏好。De Spiegeleer et al. (2021) 中，作者将ESG偏好作为优化的约束条件，形式为$\gamma\cdot\pi<\gamma_P$,其中$\gamma$
是投资组合成分的ESG评分向量。Pedersen et al. (2021) 则通过目标函数$\frac{1}{2}\lambda\sigma^2_p+\mu_p+xf(s)$获得了ESG的有效前沿，其中$f(s)$是ESG偏好函数，依赖于风险资产头寸中的平均ESG评分，$f(s)=f(\frac{s'\pi}{1'\pi})$.

### Setting and theoretical results 

为了简便起见，我们假设有两只股票$S_1,S_2$其动态过程如下：

\begin{equation}
\begin{split}
\frac{dS_{1,t}}{S_{1,t}}&=(r+\lambda_1\sigma_1^2)dt+\sigma_1dW_{1,t} \\
\frac{dS_{2,t}}{S_{2,t}}&=\left(r+\lambda_1\sigma_1\sigma_2\rho+\lambda_2\sigma_2^2\sqrt{1-\rho^2}\right)dt+\sigma_2\left(\rho dW_{1,t}+\sqrt{1-\rho^2}dW_{2,t}\right)\\
&=(r+\lambda_{22}\sigma^2_2)dt+\sigma_2dW_{3,t} 
\end{split}
(\#eq:mrau1)
\end{equation}

其中$W_{1,t},W_{2,t}$是无关的布朗运动，$\rho\in(-1,1),\sigma_i>0,i=1,2;\lambda_1\sigma_1$是与风险因子$W_1$相关的单位波动率的市场风险价格（Market Price of Risk, MPR），而$\lambda_2\sigma_2$是与风险因子$W_2$相关的市场风险价格。通过下面的式子可以将$W_1,W_2$的MPR与$W_3$的MPR^[注意Market Price of Risk，MPR 市场风险价格与Market Risk Premium 风险溢价进行区分，前者实际上是夏普比率？]相联系：

$$
\lambda_{22}\sigma_2=\lambda_1\sigma_1\rho+\lambda_2\sigma_2\sqrt{1-\rho^2}.
$$

假设投资者财富为$X_t,$投资者将财富的比例$\pi_{1,t}$分配给$S_{1,t};\pi_{2,t}$分配给$S_{2,t}$,其余比例$(1-\pi_{1,t}-\pi_{2,t})$存入银行账户$B_t,$银行利率为常数$r.$投资者的自融条件为：


\begin{align}
\frac{dX_t}{X_t}=&\pi_{1,t}\frac{dS_{1,t}}{S_{1,t}}+\pi_{2,t}\frac{dS_{2,t}}{S_{2,t}}+(1-\pi_{1,t}-\pi_{2,t})\frac{dB_t}{B_t}\\
=&\left(r+\pi_{1,t}\lambda_1\sigma_1^2+\pi_{2,t}\left(\lambda_1\sigma_1\sigma_2\rho+\lambda_2\sigma_2^2\sqrt{1-\rho^2}\right)  \right)dt\\
&+\pi_{1,t}\sigma_1dW_{1,t}+\pi_{2,t}\sigma_2\left(\rho dW_{1,t}+\sqrt{1-\rho^2}dW_{2,t}\right)
\end{align}


投资者根据风险的来源,即$W_{1,t},W_{2,t}$表现出不同的偏好，即不同的风险厌恶水平。例如$W_{1,t}$代表绿色股票$S_1$的风险，投资者对该风险的厌恶程度较低，$W_{2,t}$是棕色股票$S_2$相应的与$S_1$不相关的风险，投资者对其表现出较高的风险厌恶，以捕捉投资者ESG偏好。

接下来在构建财富过程和效用函数时，将会把这两种风险来源分开。


\begin{align}
d \log X_t&=rdt+d\log X_{1,t}+d\log X_{2,t}\\
d\log X_{1,t}&=\left( \pi_{1,t}\lambda_1\sigma_1^2+\pi_{2,t}\lambda_1\sigma_1\sigma_2\rho-\frac{1}{2}(\pi_{1,t}\sigma_1+\pi_{2,t}\sigma_2\rho)^2 \right)dt+(\pi_{1,t}\sigma_1+\pi_{2,t}\sigma_2\rho)dW_{1,t}\\
d\log X_{2,t}&=\left(\pi_{2,t}\lambda_2\sigma_2^2\sqrt{1-\rho^2}-\frac{1}{2}\pi_{2,t}^2\sigma_2^2(1-\rho^2) \right)dt+\pi_{2,t}\sigma_2\sqrt{1-\rho^2}dW_{2,t}
\end{align}

通过以上设置，$X_{1,t}$捕捉了由绿色投资驱动的财富演化，而$X_{2,t}$ 则代表与非跨期的棕色投资相关的财富，这些 $X_i$ 可以被解释为导致投资者满意度不同的属性。具体地，我们有：

\begin{align}
\frac{dX_{1,t}}{X_{1,t}}&=(\pi_{1,t}\lambda_1\sigma_1^2+\pi_{2,t}\lambda_1\sigma_1\sigma_2\rho)dt+(\pi_{1,t}\sigma_1+\pi_{2,t}\sigma_2\rho)dW_{1,t}\\
&=(\pi_{1,t}\sigma_1+\pi_{2,t}\sigma_2\rho)(\lambda_1\sigma_1 dt+dW_{1,t})\\
\frac{dX_{2,t}}{X_{2,t}}&=\pi_{2,t}\lambda_2\sigma_2^2\sqrt{1-\rho^2} dt+\pi_{2,t}\sigma_2\sqrt{1-\rho^2} dW_{2,t}\\
&=\pi_{2,t}\sigma_2\sqrt{1-\rho^2}(\lambda_2\sigma_2 dt+dW_{2,t})
\end{align}


与以下关系：

\begin{align}
\log\frac{X_T}{X_0}&=rT+\log\frac{X_{1,T}}{X_{1,0}}\log\frac{X_{2,T}}{X_{2,0}}\\
X_0&=X_{1,0}X_{2,0}\\
X_T&=\exp(rT)X_{1,T}X_{2,T}
\end{align}


为了区分投资者的风险偏好，我们使用以下多重常数相对风险厌恶效用：

$$
u(X_{1,T},X_{2,T})=sign(\alpha_1)\frac{(X_{1,T})^{\alpha_1}}{\alpha_1}\frac{(X_{2,T})^{\alpha_2}}{\alpha_2}
$$
其中$0<\alpha_1\leq\alpha_2<1$或者$\alpha_2\leq\alpha_1<0.$注意，$u(X_{1,T}, X_{2,T})$ 是凹的（Hessian 矩阵是半负定的），并且在每个变量上是递增的. Arrow–Pratt系数用于衡量绝对风险厌恶度。其依赖于变量的财富水平：

- 对绿色投资的风险厌恶系数为$-\frac{\frac{\partial^2u}{\partial X_1^2}}{\frac{\partial u}{\partial X_1}}=\frac{1-\alpha_1}{X_1}$;

- 对棕色投资的风险厌恶系数为$-\frac{\frac{\partial^2u}{\partial X_2^2}}{\frac{\partial u}{\partial X_2}}=\frac{1-\alpha_2}{X_2}.$


这个问题的价值函数为：

$$
V(t,X_{1,0},X_{2,0})=\max_{\pi_{1,t},\pi_{2,t}}\mathbb{E}[u(X_{1,T},X_{2,T})]=\max_{\pi_{1,t},\pi_{2,t}}\mathbb{E} \left[sign(\alpha_1)\frac{(X_{1,T})^{\alpha_1}}{\alpha_1}\frac{(X_{2,T})^{\alpha_2}}{\alpha_2}\right] (\#eq:mrau2)
$$


在不失一般性的情况下，我们选择初始值 $X_{1,0} = X_0$, $X_{2,0} = 1$。接下来，我们给出论文的主要结果。

**Proposition**  
假设$0<\alpha_1\leq\alpha_2<1$或者$\alpha_2\leq\alpha_1<0,$则方程\@ref(eq:mrau2)的最优配置和价值函数为：


\begin{align}
\pi_{2}^*&=\frac{\lambda_2}{(1-\alpha_2)\sqrt{1-\rho^2}},\\
\pi_1^*&=\frac{\lambda_1}{1-\alpha_1}-\frac{\lambda_2\sigma_2\rho}{\sigma_1(1-\alpha_2)\sqrt{1-\rho^2}};\\
V(X_{1,t},X_{2,t})&=sign(\alpha_1)\frac{X_{1,t}^{\alpha_1}X_{2,t}^{\alpha_2}}{\alpha_1\alpha_2}\exp\{b(T-t)\},\\
\text{where }b&=\frac{1}{2}\left( \frac{\sigma_1^2\lambda_1^2\alpha_1}{1-\alpha_1}+\frac{\sigma_2^2\lambda_2^2\alpha_2}{1-\alpha_2}\right).
\end{align}

**Proof**

<div class="figure">
<img src="https://Go9entle.github.io/picx-images-hosting/IMG_0751.7sncx0qmuk.webp" alt="Proof of Proposition"  />
<p class="caption">(\#fig:unnamed-chunk-1)Proof of Proposition</p>
</div>


**Remark**


我们接下来强调一下这一结果和背景中的几个重要方面：

1. 如果$\rho$趋近于1（或 -1），$\pi_2$会趋向于$\infty$，而 $\pi_1$ 会趋向于$-\infty$（或 $\infty$）。这与$\alpha_1=\alpha_2$的情况相同。其背后的逻辑是，在极限情况下，可以创造出比现金账户更优的无风险投资组合。这也解释了协方差矩阵可逆的假设。

2. 该工作可以扩展到 $N$ 个资产，假设风险厌恶结构为$0 < \alpha_1 \leq ⋯ \leq \alpha_n < 1 $或$\alpha_n \leq ⋯ \leq \alpha_1< 0，$并且效用函数为：

  $$
  u(X_{1,T}, … , X_{n,T}) =
   \begin{cases}
   \text{sign}(\alpha₁) \prod_{i=1}^{N}\frac{ (X_{i,T})^{\alpha_i}}{ \alpha_i}, & N\text{ even} \\
   \quad\quad\prod_{i=1}^{N} \frac{(X_{i,T})^{\alpha_i}} { \alpha_i}, & N\text{ odd}
   \end{cases}
  $$

  对于股票，最方便的表示方法是使用下三角矩阵，以分离独立的风险因子$W_i,i=1,2,...,N$，这些因子驱动着各个属性 （记为$X_i,i=1,2,...,N$），其方程为：

  $$
    \frac{dS_{i,t}}{S_{i,t}} = \left( r + \sum_{j=1}^{i} \sigma_i \lambda_j \sigma_j \rho_{ij} \right) dt + \sigma_i \left( \sum_{j=1}^{i} \rho_{ij} dW_{j,t} \right)
  $$
 	用矩阵表示形式为：

$$
   dS_t = diag(S_t) \left( (r + diag(\sigma) diag(\lambda) A) dt + diag(\sigma) A dW_t \right)
$$

  其中 $S_t,W_t,\lambda$和$\sigma$是向量，$A$代表协方差矩阵$\rho=A'A$的下三角分解。这导致了一个方便的自融资条件的表示：

  \begin{align}
    \frac{dX_t}{X_t} &= \sum_{i=1}^{N} \pi_{i,t} \frac{dS_{i,t}}{S_{i,t}} + \left( 1 -    \sum_{i=1}^{N} \pi_{i,t} \right) \frac{dB_t}{B_t}\\
   &=\sum_{i=1}^N\pi_{i,t}\sum_{j=1}^i\sigma_i\rho_{ij}(\lambda_j\sigma_j dt+d     W_{j,t})+\frac{dB_t}{B_t}
  \end{align}

  通过使用 $d\log X_t=rdt+\sum_{j=1}^N d\log X_{j,t}$ₜ，可以表示为各个独立财富（属性）$X_j:j=1,...,N$：

  $$
   \frac{dX_{j,t}}{X_{j,t}} = \bar{\pi}_j (\lambda_j \sigma_j dt + dW_{j,t})
  $$

  其中 $\bar{\pi}_j = \sum_{i=j}^{N} \pi_{i,t} \sigma_i \rho_{ij}$。  
  该问题现在可以通过 $\bar{\pi} = (\bar{\pi}_1, … , \bar{\pi}_N)' $ 轻松求解，且 $\bar{\pi}_j = \frac{\lambda_j}{(1 - \alpha_j) \sigma_j},j=1,..,N$，然后通过矩阵表示法将其转换回$\pi$，即 $\bar{\pi} = A' diag(\sigma) \pi$。

3. 在存在两组股票的情况下：$S_1,...,S_{N_1}$和$S_{N_1+1},...,S_N$，每组具有不同的风险厌恶系数 $\alpha_A$和$\alpha_B$，分别对应之前提到的符号表示方法，我们可以写出以下方程：

  \begin{align}
    &d \log X_{A,t} = \sum_{j=1}^{N₁} \log X_{j,t}\\
    &d \log X_{B,t} = \sum_{j=N₁+1}^{N} \log X_{j,t}\\
    &d \log X_t = r dt + d \log X_{A,t} + d \log X_{B,t}
  \end{align}

  解决方法将与之前类似，通过分组$\alpha_1=...=\alpha_{N_1}=\alpha_A$ 和$\alpha_{N_1+1}=...=\alpha_N=\alpha_B$。

4. 有趣的是，我们可以重写\@ref(eq:mrau1)使用“棕色股票”作为“绿色股票”的驱动因素，模型为：

  \begin{equation}
  \begin{split}
    \frac{dS_{2,t}}{S_{2,t}}& = (r + \lambda_{22} \sigma_2^2) dt + \sigma_2 dW_{3,t}\\
    \frac{dS_{1,t}}{S_{1,t}} &= \left( r + \lambda_{22} \sigma_1 \sigma_2 \rho +     \lambda_{11} \sigma_1^2 \sqrt{1 - \rho^2} \right) dt + \sigma_1 \left( \rho dW_{3,t} +    \sqrt{1 - \rho^2} dW_{4,t} \right)\\
    &= (r + \lambda_1 \sigma_1^2) dt + \sigma_1 dW_{5,t}\\ 
  \end{split}
  (\#eq:mrau3)
  \end{equation}

  其中关系为：
  
  $$
   \lambda_1 \sigma_1 = \lambda_{22} \sigma_2 \rho + \lambda_{11} \sigma_1 \sqrt{1 - \rho^2}
  $$
  理论同样成立，但解决方法有所不同。两者的区别在于，投资者如何在风险厌恶的角度解释绿色股票和棕色股票之间的共同/共享风险（相关部分）。模型\@ref(eq:mrau1)将其视为绿色股票的较低风险厌恶，而模型\@ref(eq:mrau3)则假设它与棕色股票的风险厌恶相同（即更高的风险厌恶）。


### Numerical analysis and discussion

本节研究不同风险厌恶程度和相关性对最优资产配置的影响，以及使用流行的次优策略所造成的财富等价损失（CEL）。我们假设标准的年化参数设置为$\sigma_1=0.35,\sigma_2=0.4,\rho=0.5,r=0.01,\lambda_1=0.8,\lambda_2=0.5$.这意味着资产1和资产2的预期回报分别为$\mu_1=r+\lambda_1\sigma_1^2=0.108,\mu_2=r+\lambda_1\sigma_1\sigma_2\rho+\lambda_2\sigma_2^2\sqrt{1-\rho^2}=0.1353.$

<div class="figure">
<img src="https://Go9entle.github.io/picx-images-hosting/image.2obo7r6g6q.webp" alt="Optimal allocation versus changes in $\alpha_2$, left uses $\alpha_1$ = 0.6, right with $\alpha_1$ = −1." width="50%" /><img src="https://Go9entle.github.io/picx-images-hosting/image.99thysar6u.webp" alt="Optimal allocation versus changes in $\alpha_2$, left uses $\alpha_1$ = 0.6, right with $\alpha_1$ = −1." width="50%" />
<p class="caption">(\#fig:mraufig1)Optimal allocation versus changes in $\alpha_2$, left uses $\alpha_1$ = 0.6, right with $\alpha_1$ = −1.</p>
</div>

图\@ref(fig:mraufig1)左侧展示了固定$\alpha_1=0.6$时$\alpha_2$的变化对股票配置的影响，右侧则是固定$\alpha_1=-1,\alpha_2$从-1变动至-5.在这两种情况下，将更高的风险厌恶水平赋予棕色股票会显著增加对绿色股票的配置，同时棕色股票的投资大幅下降。

<div class="figure">
<img src="https://Go9entle.github.io/picx-images-hosting/image.2obo7r6g6q.webp" alt="Optimal allocation versus changes in correlation for, left uses $\alpha_1$ = 0.6, $\alpha_2$= 0.3, right with $\alpha_1$= −1, $\alpha_2$= −5." width="50%" /><img src="https://Go9entle.github.io/picx-images-hosting/image.99thysar6u.webp" alt="Optimal allocation versus changes in correlation for, left uses $\alpha_1$ = 0.6, $\alpha_2$= 0.3, right with $\alpha_1$= −1, $\alpha_2$= −5." width="50%" />
<p class="caption">(\#fig:mraufig2)Optimal allocation versus changes in correlation for, left uses $\alpha_1$ = 0.6, $\alpha_2$= 0.3, right with $\alpha_1$= −1, $\alpha_2$= −5.</p>
</div>

图\@ref(fig:mraufig2)显示了两只股票之间的相关性对最优配置的影响。左侧固定$\alpha_10=0.6,\alpha_2=0.1,$右侧固定$\alpha_1=-1,\alpha_2=-5.$我们可以看到，负相关会导致绿色投资的配置显著增加。这一点特别重要，因为气候变化可能会导致绿色股票和棕色股票的表现出现负相关。

接下来，我们研究了由于缺乏构建最优解决方案的知识，投资者在保持相同的风险厌恶水平下，使用次优策略所产生的财富等价损失（CEL）。由于使用相同的风险厌恶水平得到的配置是次优的，因此会导致效用损失。我们将使用次优策略得到的价值函数记作$V^s$，然后定义 CEL 为满足以下方程的标量$q$：

$$
V(t,X_0(1-q),1)=V^s(t,X_0,1)
$$
我们使用HJB方程利用$\pi_1,\pi_2$作为次优的常数策略，得到：

$$
V^s(t,X,1)=sign(\alpha_1)\frac{X^{\alpha_1}v^s(t)}{\alpha_1\alpha_2}\\
v^s(t)=\exp\{b^s(T-t)\}
$$
其中

$$
\begin{align}
b^s=&(\pi_1\lambda_1\sigma_1^2+\pi_2\lambda_1\sigma_1\sigma_2\rho)\alpha_1+\frac{1}{2}(\pi_1\sigma_1+\pi_2\sigma_2\rho)^2(\alpha_1-1)\alpha_1\\
&+\pi_2\lambda_2\sigma_2\sqrt{1-\rho^2}\alpha_2+\frac{1}{2}\pi_2^2\sigma_2^2(1-\rho^2)(\alpha_2-1)\alpha_2
\end{align}
$$
于是我们得到：

$$
q=1-\exp\left\{ \frac{1}{\alpha_1}(b^s-b)(T-t)\right\}.
$$
其中$b=\frac{1}{2}\left(\frac{\alpha_1\sigma_1^2\lambda_1^2}{1-\alpha_1}+\frac{\alpha_2\sigma_2^2\lambda_2^2}{1-\alpha_2} \right)$

图(暂时未能复现)显示，由于保持相同风险厌恶策略，投资者可能面临高达 65% 的财富损失。这可以从右侧图中看到，在考虑棕色投资的风险厌恶度为$\alpha_2=-6$ 时，投资者需要 65% 更少的初始财富来匹配一个风险厌恶度为$\alpha_2=-6$ 的投资者的表现，而后者使用的是由$\alpha_1=-1$ 推导出的配置。

### Conclusion

本研究使用多重风险厌恶效用的概念，也称为多属性效用，开辟了 ESG 投资的新方向。在假设投资者对棕色投资赋予更高的风险厌恶而对绿色投资赋予较低风险厌恶的前提下，我们得出了封闭形式的直观解，解决了预期效用设置下的最优配置和价值函数。这使我们能够探索两种风险厌恶设置在 ESG 投资中的含义，且选取了合理的股票参数进行研究。
该研究可以扩展到多个方向，不仅可以考虑其他多变量效用、每个风险厌恶下的多资产情况以及更丰富、更现实的模型来描述基础资产，还可以将 ESG 应用扩展到其他提出的考虑 ESG 偏好和评分的形式。

