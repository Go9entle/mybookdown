# 最优多维再保险与多元风险厌恶效用 {#mulreinsurance } 

本章开始于2025年2月26日。

## 共同冲击依赖结构下的最优多维再保险政策 {#OMR }  

>论文原文*Optimal multidimensional reinsurance policies under a common shock dependency structure*点击[这里](https://go9entle.github.io/myresearch/OMR.pdf)查看。


**再保险**是指以下保险公司为避免自己遭遇损失和风险而向另一家保险公司投保的过程。

本文中，作者考虑一种共同冲击依赖结构用于对再保险公司的盈余过程进行建模，目标是针对再保险策略向量最大化目标函数，即随时间积分的预期折现盈余水平。  

### Common shock model  

考虑一家运营于$n$个相互依赖保险业务线的保险公司。事实上，索赔可能同时发生在多个业务线上，例如，一次车祸可能对汽车和司机造成损害。因此，我们假设有$m$个源，其中每个源的发生会导致一个或多个业务线发生索赔。图\@ref(fig:csm)


``` r
knitr::include_graphics('https://Go9entle.github.io/picx-images-hosting/1740566499491.7snclf65t8.webp')
```

<div class="figure">
<img src="https://Go9entle.github.io/picx-images-hosting/1740566499491.7snclf65t8.webp" alt="An example of common shock model with m = 4 sources and n = 3 lines."  />
<p class="caption">(\#fig:csm)An example of common shock model with m = 4 sources and n = 3 lines.</p>
</div>
需要注意的是源1和源4都会在第一条业务线发生索赔，但这些索赔的概率分布可能不同。

更准确地说，考虑一个概率空间$(\Omega,\mathcal{E},P)$,在此空间中我们考虑具有参数$\beta_i$的独立Poisson过程$\{N_i(t):t\geq0\}$,用于描述源$i\in\{1,2,...,m\}$中事件的发生频率。将源$i$在业务线$j$上的第$k$次索赔大小表示为随机变量$Y_{ijk}$.假设$\{Y_{ijk}:i\in\{1,2,...,m\},j\in A_i,k\in\mathbb{N}\}$是独立的随机变量，其中$A_i\subset\{1,2,...,n\}$是源$i$影响的业务线的集合，$Y_{ijk}$具有累积分布函数$F^Y_{ij}$和有限的均值$\mu_{ij}$。因此源$i$(固定$i$)在时间$t$之前造成的总索赔金额为

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

#### Reinsurance  
在本小节中，将会定义一个拥有$n$条业务线的公司再保险策略。首先，考虑滤波$\mathcal{F}=\{\mathcal{F}_t:t\geq0\}$其中$\mathcal{F_t}$是由$\{X(s):0\leq s\leq t\}$生成的$\sigma$-代数。

再保险策略是一个多维的随机过程$\boldsymbol{U}=\{\boldsymbol{U}(t)=(U_1(t),...,U_n(t)):t\geq0\}.$如果在时刻$t=t_1$发生源$i$的索赔，大小为$Y=\sum_{j\in A_i}Y_{ijk}$,那么再保险公司将承担$Y-\sum_{j\in A_i} r_j(U_j(t_1), Y_{ijk})$,其中函数$0\leq r_j(u,y)\leq y$是连续且关于$y$是递增的。我们说$\boldsymbol{U}$是可接受的，如果对于$j=1,2,...,n$和$t\geq0,y\geq0$函数$(\omega,t,y)\rightarrow r_j(U_j(\omega,t),y)$是$\mathcal{E\times B\times B}$可测的，并且函数$\omega\rightarrow \sum_{j\in A_i}r_j(U_j(\omega,t),y)$是$\mathcal{F_t}$可测的。我们将所有可接受策略记为$\boldsymbol{\mathcal{R}}.$本文将考虑下面两种再保险合约：  

1. 比例再保险：$r^P(u,y)=uy,\quad u\in \mathcal{U}^P=[0,1]$.

2. 损失超额再保险（XL）：$r^{XL}(u,y)=\min(u,y),\quad u\in\mathcal{U}^{XL}=[0,\infty].$

由公式\@ref(eq:omr1),由再保险策略$\boldsymbol{U}$控制的盈余过程就写为

$$
X_{\boldsymbol{U}}(t)=x+\int_0^tp(\boldsymbol{U}(s))ds-\sum_{i=1}^m\sum_{k=1}^{N_i(t)}\sum_{j\in A_i} r_j(U_j(\tau_{ik}^-),Y_{ijk}), (\#eq:omr3)
$$
其中$\tau_{ik}$是源$i$第$k$次发生索赔的时间。使用期望值原则计算保费

$$
p(\boldsymbol{u})=p-\sum_{j=1}^nq_j(u_j),\quad \boldsymbol{u}\in \mathcal{U}
$$
并且

$$
q_j(u_j)=(1+\theta_j)\sum_{i=1}^m\beta_i\mathbb{E}(Y_{ijk}-r_j(u_j.Y_{ijk}))I_{A_i}(j)
$$
其中$\theta_j>\eta_j$是安全载荷因子，$p$在\@ref(eq:omr2)中定义，且$\mathcal{U}=\mathcal{U}_1\times\mathcal{U}_2\times ...\times\mathcal{U_n}$是$\mathbb{R}^n$中一个子集。对于比例再保险$U_j=[0,1]$而对于损失超额再保险$U_j=[0,\infty].$

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

## 应用于ESG投资的多元风险厌恶效用 {#mrau}

>论文原文*Optimal multidimensional reinsurance policies under a common shock dependency structure*点击[这里](https://go9entle.github.io/myresearch/MRAU.pdf)查看。

**ESG**(Environmental, social, and governance) 是优先考虑环境问题、社会问题和公司治理的投资原则的简写。考虑ESG的投资有时被称为*负责任的投资*（responsible investing）或者*影响力投资*（impact investing）。

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
\frac{dS_{1,t}}{S_{1,t}}&=(r+\lambda_1\sigma_1^2)dt+\sigma_1dW_{1,t} (\#eq:mrau1)\\
\frac{dS_{2,t}}{S_{2,t}}&=\left(r+\lambda_1\sigma_1\sigma_2\rho+\lambda_2\sigma_2^2\sqrt{1-\rho^2}\right)dt+\sigma_2\left(\rho dW_{1,t}+\sqrt{1-\rho^2}dW_{2,t}\right)\\
&=(r+\lambda_{22}\sigma^2_2)dt+\sigma_2dW_{3,t}
\end{split}
\end{equation}


其中$W_{1,t},W_{2,t}$是无关的布朗运动，$\rho\in(-1,1),\sigma_i>0,i=1,2;\lambda_1\sigma_1$是与风险因子$W_1$相关的市场风险溢价（Market Risk Premium, MPR），而$\lambda_2\sigma_2$是与风险因子$W_2$相关的市场风险溢价。通过下面的式子可以将$W_1,W_2$的MRP与$W_3$的MRP相联系：

$$
\lambda_{22}\sigma_2=\lambda_1\sigma_1\rho+\lambda_2\sigma_2\sqrt{1-\rho^2}.
$$

假设投资者财富为$X_t,$投资者将财富的比例$\pi_{1,t}$分配给$S_{1,t};\pi_{2,t}$分配给$S_{2,t}$,其余比例$(1-\pi_{1,t}-\pi_{2,t})$存入银行账户$B_t,$银行利率为常数$r.$投资者的自融条件为：


\begin{align}
\frac{dX_t}{X_t}=&\pi_{1,t}\frac{dS_{1,t}}{S_{1,t}}+\pi_{2,t}\frac{dS_{2,t}}{S_{2,t}}+(1-\pi_{1,t}-\pi_{2,t})\frac{dB_t}{B_t}\\
=&\left(r+\pi_{1,t}\lambda_1\sigma_1^2+\pi_{2,t}\left(\lambda_1\sigma_1\sigma_2\rho+\lambda_2\sigma_2^2\sqrt{1-\rho^2}\right)  \right)dt\\
&+\pi_{1,t}\sigma_1dW_{1,t}+\pi_{2,t}\sigma_2\left(\rho dW_{1,t}+\sqrt{1-\rho^2}dW_{2,t}\right)
\end{align}


投资者根据风险的来源,即$W_{1,t},W_{2,t}$表现出不同的偏好，即不同的风险厌恶水平。例如$W_{1,t}$代表绿色股票$S_1$的风险，投资者对该风险的厌恶程度较低，$W_{2,t}$是棕色股票$S_2$相应的与$S_1$不相关的风险吗，投资者对其表现出较高的风险厌恶，以捕捉投资者ESG偏好。

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
其中$0<\alpha_1\leq\alpha_2<1$或者$\alpha_2\leq\alpha_1<0.$注意，$u(X_{1,T}, X_{2,T})$ 是凹的（Hessian 矩阵是半负定的），并且在每个变量上是递增的，而 Arrow–Pratt 的绝对风险厌恶系数依赖于变量/属性：

$$
-\frac{\frac{\partial^2u}{\partial X_1^2}}{\frac{\partial u}{\partial X_1}}=\frac{1-\alpha_1}{X_1},\quad \frac{\frac{\partial^2u}{\partial X_2^2}}{\frac{\partial u}{\partial X_2}}=\frac{1-\alpha_2}{X_2}
$$
这个问题的价值函数为：

$$
V(X_{1,0},X_{2,0})=\max_{\pi_{1,t},\pi_{2,t}}\mathbb{E}[u(X_{1,T},X_{2,T})]=\max_{\pi_{1,t},\pi_{2,t}}\mathbb{E} \left[sign(\alpha_1)\frac{(X_{1,T})^{\alpha_1}}{\alpha_1}\frac{(X_{2,T})^{\alpha_2}}{\alpha_2}\right] (\#eq:mrau2)
$$

::?:: *是否应该写为*

$$
V(t,X_1,X_2)=\max_{\pi_{1,t},\pi_{2,t}}\mathbb{E}[u(X_{1,T},X_{2,T})]=\max_{\pi_{1,t},\pi_{2,t}}\mathbb{E} \left[sign(\alpha_1)\frac{(X_{1,T})^{\alpha_1}}{\alpha_1}\frac{(X_{2,T})^{\alpha_2}}{\alpha_2}\right] 
$$

在不失一般性的情况下，我们选择初始值 $X_{1,0} = X_0$, $X_{2,0} = 1$。接下来，我们给出论文的主要结果。

**Proposition 1.**  
假设$0<\alpha_1\leq\alpha_2<1$或者$\alpha_2\leq\alpha_1<0,$则方程\@ref(eq:mrau2)的最优配置和价值函数为：


\begin{align}
\pi_{2}^*&=\frac{\lambda_2}{(1-\alpha_2)\sqrt{1-\rho^2}},\\
\pi_1^*&=\frac{\lambda_1}{1-\alpha_1}-\frac{\lambda_2\sigma_2\rho}{\sigma_1(1-\alpha_2)\sqrt{1-\rho^2}};\\
V(X_{1,t},X_{2,t})&=sign(\alpha_1)\frac{X_t^{\alpha_1}}{\alpha_1\alpha_2}\exp\{b(T-t)\},\\
\text{where }b&=\frac{1}{2}\left( \frac{\sigma_1^2\lambda_1^2\alpha_1}{1-\alpha_1}+\frac{\sigma_2^2\lambda_2^2\alpha_2}{1-\alpha_2}\right).
\end{align}





