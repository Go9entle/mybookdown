# 随机分析 {#stochastic}

##  Introduction

一般地，我们称系数可以是随机的微分方程为**随机微分方程（Stochastic Differential Equation）**，显然随机微分方程的解一定具有随机性，因此我们只期望得到关于解的概率分布。

### 1.4 最优停时（Optimal Stopping）  

  假定某人计划卖掉一个资产，在开放的市场中，他的资产在$t$时刻的价格$X_t$满足随机微分方程  
$$
\frac{dX_t}{dt}=rX_t+\alpha X_t\cdot [\text{noise}],
$$
这里的$r,\alpha$为已知的常数，折现率为已知常数$\rho$，那么在什么时候卖掉该资产为最好？

  假设知道现在时刻$t$以前的资产表现$X_s(s<t)$但是由于系统中的噪声，当然无法确信选择卖的时间是否为最优，因此要找一个停时策略，在长期运行中它应该是最好的结果，即把通胀考虑进去以后的最大化期望利润。这是一个最优停时问题。

### 1.5 随机控制（Stochastic Control）  

  假设某人有两个投资可能性：  

1. 无风险投资（如债券）.在$t$时刻每单位的价格$X_0(t)$按指数增长：  
   $$
   \frac{dX_0}{dt}=\rho X_0
   $$
   这里的$\rho$为大于0的常数

2. 有风险投资（如股票）.在$t$时刻每单位的价格$X_1(t)$满足随机微分方程：  
   $$
   \frac{dX_1}{dt}=(\mu+\sigma\cdot[\text{noise}])X_1
   $$

在每个时刻$t$，该投资者选择他的财富$V_t$中多大比例$u_t$用于风险投资，从而另一部分用于无风险投资，给定效用函数$U$和终端时刻$T$,该投资问题是找到一个最优证券组合，使得终端时刻财富$V_T$的期望效用最大  
$$
\mathop{max}\limits_{0\leq u_t\leq1}\{E[U(V_T^{(u)})]\}
$$

## 预备知识

###  概率空间 随机变量 随机过程

  **定义2.1.1**  

给定全集合$\Omega$，那么$\Omega$上的$\sigma$代数$\mathcal{F}$是由$\Omega$的某些子集构成的集合族且具有下列性质  

1. $\varnothing\in\mathcal{F}$;
2. $F\in\mathcal{F}\Rightarrow F^C\in\mathcal{F}$;
3. $A_1,A_2,\dots\in\mathcal{F}\Rightarrow A:=\mathop{\cup}\limits_{i=1}^{\infty}A_i\in\mathcal{F}$.

称$(\Omega,\mathcal{F})$为一个可测空间，这个可测空间的概率测度$P$是一个实值函数，将$\mathcal{F}$映射到$[0,1]$上，满足概率测度的两个条件  

1. $P(\varnothing)=0,P(\Omega)=1$;

2. 若$A_1,A_2,\dots\in\mathcal{F}$且$\{A_i\}_{i=1}^{\infty}$是互不相交，那么  
   $$
   P(\mathop{\cup}_{i=1}^{\infty}A_i)=\mathop{\sum}_{i=1}^{\infty}P(A_i).
   $$

称$(\Omega,\mathcal{F},P)$为一个概率空间。我们假定所有概率空间都是完备的，即$\mathcal{F}$包括了$\Omega$中$P$外测度为零的所有子集。

对$\Omega$中的某一子集$F$，如果$F\in\mathcal{F}$,则称$F$为$\mathcal{F}$可测集，在概率上称为事件，$P(F)$就称为事件$F$发生的概率。特别地，$P(F)=1$则说事件$F$为依概率1发生或者几乎必然(a.s.)发生。

  对给定的$\Omega$的一个集合族$\mathcal{U}$,存在一个包含$\mathcal{U}$的最小$\sigma$代数$\mathcal{H}_\mathcal{U}$,即  
$$
\mathcal{H}_\mathcal{U}=\cap\{\mathcal{H}:\mathcal{H}为\Omega上的\sigma代数，\mathcal{U}\subset\mathcal{H}\}
$$
  ，称$\mathcal{H}_{\mathcal{U}}$是由$\mathcal{U}$生成的$\sigma$代数（包含$\mathcal{U}$的最小集合，所以做了交集）。

  例如，$\mathcal{U}$是拓扑空间$\Omega$的所有开子集构成的集合(如$\Omega=\mathbf{R}^n$)，那么$\mathcal{B}=\mathcal{H}_\mathcal{U}$称为$\Omega$上的Borel $\sigma$代数。对任意元素$B\in\mathcal{B}$称为Borel可测集。$\mathcal{B}$包含所有的开子集、所有的闭子集、所有的可数个闭子集的并集以及所有的可数个这种并集的交集等等。

  设$(\Omega,\mathcal{F},P)$是给定的概率空间，如果  
$$
Y^{-1}(U):=\{\omega\in\Omega;Y(\omega)\in U\}\in\mathcal{F}
$$
对所有开集$U\in\mathbf{R}^n$（或等价地，对所有Borel集$U\in\mathbf{R}^n$）均成立，那么函数$Y:\Omega\rightarrow\mathbf{R}^n$称为$\mathcal{F}$可测的。（这实际上有点像 Y是一个随机变量？）

  若$X:\Omega\rightarrow \mathbf{R}^n$是任意一个函数，那么由$X$生成的$\sigma$代数$\mathcal{H}_X$是$\Omega$上的包含所有形如$X^{-1}(U)$（$U\in\mathbf{R}^n$为开集）的最小$\sigma$代数。不难证明  
$$
\mathcal{H}_X=\{X^{-1}(B);B\in\mathcal{B}\},
$$
这里$\mathcal{B}$是$\mathbf{R}^n$上的Borel $\sigma$代数。显然$X$是$\mathcal{H}_X$可测的，而$\mathcal{H}_X$是具有上述性质的最小$\sigma$代数。（可以当作事件域？）

  **引理2.1.2**

如果$X,Y:\Omega\rightarrow\mathbf{R}^n$是两个给定的函数，$Y$为$\mathcal{H}_X$可测的充要条件是存在一个Borel可测函数$g:\mathbf{R}^n\rightarrow\mathbf{R}^n$使得$Y=g(X)$.

  下面，设$(\Omega,\mathcal{F},P)$是一个给定的完备概率空间，一个随机变量$X$是一个$\mathcal{F}$可测函数$X:\Omega\rightarrow\mathbf{R}^n$.每个随机变量诱导了$\mathbf{R}^n$上的概率测度$\mu_X$,定义为  
$$
\mu_X(B)=P(X^{-1}(B)),
$$
$\mu_X$称为$X$的分布（！！！）。（$B$是随机变量映射的像，$X:\Omega\rightarrow\mathcal{B}$）

  如果$\int_{\Omega}|X(\omega)|dP(\omega)<\infty$,那么  
$$
E[X]:=\int_{\Omega}X(\omega)dP(\omega)=\int_{\mathbf{R}^n}xd\mu_X(x)
$$
称为$X$的期望。更一般地，如果$f:\mathbf{R^n}\rightarrow\mathbf{R}$是Borel可测的，且$\int_{\Omega}|f(X(\omega))|dP(\omega)<\infty$,那么  
$$
E[f(X)]:=\int_{\Omega}f(X(\omega))dP(\omega)=\int_{\mathbf{R}^n}f(x)d\mu_X(x).
$$

#### $L^p$空间  

  如果$X:\Omega \rightarrow\mathbf{R}^n$是一个随机变量，$p\in[1,\infty)$是一个常数，定义$X$上的$L^p$范数$||X||_p:$  
$$
||X||_p=||X||_{L^p(P)}=(\int_\Omega|X(\omega)|^pdP(\omega))^{\frac{1}{p}}.
$$
如果$p=\infty$,定义  
$$
||X||_{\infty}=||X||_{L^{\infty}(P)}=\text{inf}\{N\in\mathbf{R};|X(\omega)|\leq N\, a.s.\},
$$
相应的$L^p$空间定义为  
$$
L^p(P)=L^p(\Omega)=\{X:\Omega\rightarrow\mathbf{R}^n;||X||_p<\infty\},
$$
在该范数定义下，$L^p$空间是Banach空间即完备的赋范空间。如果$p=2$,空间$L^2(P)$是一个Hilbert空间，即完备的内积空间，其中内积  
$$
(X,Y)_{L^2(P)}:=E[X\cdot Y],\quad X,Y\in L^2(P).
$$
  **定义2.1.3**  

  两个子集$A,B\in\mathcal{F}$称为独立的，如果  
$$
P(A\cap B)=P(A)\cdot P(B).
$$
集族$\mathcal{A}=\{\mathcal{H_i};i\in I\}$,如果  
$$
P(H_{i_1}\cap \dots \cap H_{i_k})=P(H_{i_1}) \cdots P(H_{i_k}),
$$
对$\forall H_{i_1}\in \mathcal{H_{i_1}},...,H_{i_k}\in\mathcal{H_{i_k}}$成立，$i_1,i_2,...i_k$互不相同。

  如果由随机变量族$X_i,i\in I$生成的$\sigma$代数$\mathcal{H}_{X_i}$构成的集族是独立的，那么随机变量族也是独立的。

  度过两个随机变量$X,Y:\omega\rightarrow\mathbf{R}$是独立的，假设$E[|X|]<\infty,E[|Y|]<\infty$，则$E[XY]=E[X]E[Y]$.

  **定义2.1.4**  

  随机过程是带参数的一族随机变量：$\{X_t\}_{t\in T}$定义于概率空间$(\Omega,\mathcal{H},P)$上，取值于$\mathbf{R}^n$中。

  参数空间$T$通常是射线$[0,\infty).$注意对每个固定的$t\in T$,有随机变量  
$$
\omega\rightarrow X_t(\omega);\quad \omega\in \Omega.
$$
另一方面，固定$\omega\in \Omega$,可以考虑函数  
$$
t\rightarrow X_t(\omega);\quad t\in T,
$$
称之为$X_t$的路径。

  一般地，可以直观地把$t$当作时间，而每个$\omega$可认为单个的“质子”或者“实验”。$X_t(\omega)$表示在时刻$t$时质子（实验）$\omega$的位置（或结果）。有时可以用$X(t,\omega)$代替$X_t(\omega)$，因此可以把随机过程看作一个从$T\times\Omega$到$\mathbf{R}^n$的函数，随机过程关于$(t,\omega)$二元可测的。

  最后注意到，可以认为，对每个$\omega$，函数$t\rightarrow X_t(\omega)$是从$T$到$\mathbf{R}^n$的函数，因此可认为$\Omega$是空间$\widetilde{\Omega}=(\mathbf{R}^n)^T$(即从$T$到$\mathbf{R}^n$的所有的函数全体集合)的子集。此时，$\sigma$代数$\mathcal{F}$将包含下述形式集合生成的$\sigma$代数$\mathcal{B}$:  
$$
\{\omega;\omega(t_1)\in F_1,...,\omega(t_k)\in F_k\},\quad F_i\subset \mathbf{R}^n为Borel集
$$
因此，我们可以将随机过程视为可测空间$((\mathbf{R}^n)^T,\mathcal{B})$上的一个概率测度$P$.

  过程$X=\{X_t\}_{t\in T}$的有限维分布是定义在$\mathbf{R}^{nk},k=1,2,...$上的测度$\mu_{t_1,...,t_k}$,其中  
$$
\mu_{t_1,...,t_k}(F_1\times F_2\times \dots F_k)=P[X_{t_1}\in F_1,...,X_{t_k}\in F_k];t_i\in T,
$$
$F_1,...,F_k$定义为$\mathbf{R}^n$中的Borel集。

  反之，给定$\mathbf{R}^{nk},k=1,2,...$上的概率测度$\nu_{t_1,...,t_k}$以后，能否构造一个随机过程$\{Y_t\}_{t\in T}$使得$\nu_{t_1,...,t_k}$作为它的有限维分布？

  **定理2.1.5（Kolmogorov存在定理）**  

  对任意的$t_1,...,t_k\in T,k\in\mathbf{N},$设$\nu_{t_1,...,t_k}$为$\mathbf{R}^{nk}$上的概率测度，满足

- (K1) $\nu_{t_{\sigma(1)},...,t_{\sigma(k)}}(F_1\times...\times F_k)=\nu_{t_1,...,t_k}(F_{\sigma^{-1}(1)}\times...\times F_{\sigma^{-1}(k)})$其中$\sigma$为$\{1,2,...,k\}$的任意一个排列。

- (K2) $\nu_{t_1,...,t_k}(F_1\times ...\times F_k)=\nu_{t_{\sigma(1)},...,t_{\sigma(k)},t_{\sigma_{k+1}},...,t_{\sigma_{k+m}}}(F_1\times...\times F_k\times \mathbf{R}^n\times...\mathbf{R}^n)$,对任意$m\in\mathbf{N},$此处右边总共有$k+m$个因素。

  则存在一个概率空间$(\Omega,\mathcal{F},P)$和$\Omega$上的随机过程$\{X_t\},X_t:\Omega\rightarrow \mathbf{R}^n$.对任意的$t_i\in T,k\in\mathbf{N}$及任意的Borel集$F_i$满足  
$$
\mu_{t_1,...,t_k}(F_1\times F_2\times \dots F_k)=P[X_{t_1}\in F_1,...,X_{t_k}\in F_k].
$$

### 布朗运动（Brownian Motion）

为了构造$\{B_t\}_{t\geq0}$,由Kolmogorov存在定理，只需要指定一族概率测度$\{\nu_{t_1,...,t_k}\}$满足条件(K1),(K2)且这些测度与观察到的花粉表现一致：固定$x\in\mathbf{R}^n$,定义
$$
p(t,x,y)=(2\pi t)^{-\frac{n}{2}}\cdot \exp \Big ( -\frac{|x-y|^2}{2t}\Big),\quad y\in\mathbf{R}^n,t>0.
$$
如果$0\leq t_1\leq t_2\leq ... \leq t_k$，在$\mathbf{R}^{nk}$上定义一个测度$\nu_{t_1,...,t_k}$使得
$$
\begin{aligned}
&\nu_{t_1,...,t_k}(F_1\times ...\times F_k)\\
=&\int_{F_1\times...\times F_k}p(t_1,x,x_1)p(t_2-t_1,x_2,x_2)\cdots p(t_k-t_{k-1},x_{k-1},x_k)dx_1\cdots dx_k,
\end{aligned}
$$
此处$dy=dy_1\cdots dy_k$为Lebesgue测度，$p(0,x,y)dy=\delta_x(y)$是在$x$处的单位质点。

​	利用(K1)，把它延拓到所有$t_i$的有限序列。由于对$\forall t\geq0,\int_{\mathbf{R}^n}p(t,x,y)dy=1,$故(K2)满足，由Kolmogorow定理，存在一个概率空间$(\Omega,\mathcal{F},P^x)$和一个$\Omega$上的随机过程$\{B_t\}_{t\geq0},$使得$B_t$的有限维分布为上式，即
$$
\begin{aligned}
&P^x(B_{t_1}\in F_1,...,B_{t_k}\in F_k)\\
=&\int_{F_1\times...\times F_k}p(t_1,x,x_1)p(t_2-t_1,x_2,x_2)\cdots p(t_k-t_{k-1},x_{k-1},x_k)dx_1\cdots dx_k.
\end{aligned}
$$


​	**定义2.2.1**

​	上述过程称为初值为$x$的布朗运动的修正，注意$P^x(B_0=x)=1$.

​	下面叙述布朗运动的基本性质：

1. $B_t$是一个Gauss过程，即对所有的$0\leq t_1\leq...t_k,$随机变量$Z=(B_{t_1},...,B_{t_k})\in \mathbf{R}^{nk}$是服从多重正态分布的，即存在一个向量$M\in \mathbf{R}^{nk}$和一个半正定矩阵$C=[c_{jm}]\in\mathbf{B}^{nk\times nk}$使得
   $$
   E^x\Big[\exp\Big(i\sum_{j=1}^{nk}u_jZ_j\Big)\Big]=\exp\Big(-\frac{1}{2}\sum_{j,m}u_jc_{jm}u_m+i\sum_{j}u_jM_j\Big)
   $$
   对所有的$u=(u_1,...,u_{nk})\in \mathbf{R}^{nk}$成立，这里的$i$是虚数单位，$E^x$表示关于概率$P^x$​所取的数学期望。
   $$
   M=E^x[Z]
   $$
   是$Z$的均值，
   $$
   c_{jm}=E^x[(Z_j-M_j)(Z_m-M_m)]
   $$
   是$Z$的协方差矩阵。

   ​	经过计算可知，
   $$
   M=E^x[Z]=(x,x,...,x)\in\mathbf{R}^{nk},
   $$

   $$
   \begin{aligned}
   C=
   \begin{pmatrix}
   t_1I_n & t_1I_n &...&t_1I_n\\
   t_1I_n & t_2I_n &...&t_2I_n\\
   \vdots & \vdots& &\vdots\\
   t_1I_n&t_2I_n&...&t_kI_n
   \end{pmatrix}，
   \end{aligned}
   $$

   

因此
$$
E^x[B_t]=x, \quad \text{for all}\,\,t\geq0
$$
及
$$
E^x[(B_t-x)^2]=nt,\quad E^x[(B_t-x)(B_s-x)]=n\,\min(s,t),
$$
而且，如果$t\geq s,$则有
$$
E^x[(B_t-B_s)^2]=n(t-s).
$$

2. $B_t$具有独立增量，即对任意的$0\leq t_1<t_2<...<t_k,$
   $$
   B_{t_1},B_{t_2}-B_{t_1},...,B_{t_k}-B_{t_{k-1}}是独立的.
   $$
