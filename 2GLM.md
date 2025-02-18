# 广义线性模型 {#glm}

## 广义线性模型的指数族分布 {#glmexp}

假设我们观察到数据$\{y_i\}$是独立随机变量$\{Y_i\}$的一个值，广义线性模型由以下成分组成：

- 随机成分
  $$
  Y_i\sim f(y_i|\theta_i,\phi),
  $$
  其中$f(\cdot)$是指数分布族的概率密度函数或概率质量函数。

- 链接函数$g(\cdot)$满足
  $$
  \eta_i=g(\mu_i),
  $$
  其中$\mu_i=E(Y_i)$.

- 线性预测器成分
  $$
  \eta_i=x_i^T\beta.
  $$

**指数族**  

一族概率密度函数或概率质量函数被称为指数族，如果满足形式
$$
f(y|\theta)=a(\theta)b(y)\exp\left( \sum_{i=1}^kw_i(\theta)t_i(y) \right),y\in A,
$$
其中我们假设：

- $A$不依赖于某个$k$维向量$\boldsymbol{\theta}$​.
- $a(\theta)>0$是不依赖于$y$的实值函数。
- $w_i(\theta),i=1,...,k$是不依赖于$y$的实值函数。
- $b(y)\geq0$是不依赖于$\theta$的实值函数。
- $t_i(y),i=1,2,...,k$是不依赖于$\theta$的实值函数

**典型形式**  

与其说$\theta$是参数不如令$\eta_i=w_i(\theta)$称为典型或自然参数，概率密度或质量函数就会变为
$$
f(y|\boldsymbol{\eta})=a^*(\boldsymbol{\eta})b(y)\exp\left( \sum_{i=1}^k\eta_it_i(y) \right),y\in A.
$$
这种方程的形式不是唯一的。



**指数族的示例**  

所有这些分布都是指数族：

- 正态分布
- Gamma分布
- Beta分布
- 逆高斯分布
- 二项分布
- Poisson分布
- 负二项分布

**示例**  

- 正态分布：
  $$
  f(y|\mu,\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left( -\frac{(y-\mu)^2}{2\sigma^2} \right).
  $$

- 二项分布
  $$
  f(y|p)=\binom{m}{y}p^y(1-p)^{m-y}.
  $$

**反例**  
$$
f(y|\theta)=\theta^{-1}\exp(1-(y/\theta)),0<\theta<y<\infty
$$
不是一个指数族。

**\boldsymbol{\eta}的充分统计量**  
假设我们有一组随机变量$\{Y_i\}$且满足$Y_i\sim f(y_i|\eta)$（在规范形势下）。  
于是
$$
\begin{aligned}
&\prod_{i=1}^nf(y_i|\boldsymbol{\eta})\\
=&\prod_{i=1}^n\left[b(y_i)a^*(\boldsymbol{\eta})\exp\left( \sum_{j=1}^k\eta_jt_j(y_i) \right) \right]\\
=&\left(\prod_{i=1}^nb(y_i)  \right)(a^*(\boldsymbol{\eta}))^n\exp\left(\sum_{j=1}^k\eta_j\sum_{i=1}^nt_j(y_i)\right).

\end{aligned}
$$
因此$\{ t_j(\boldsymbol{y})=\sum_{i=1}^nt_j(y_i):j=1,...,n \}$是$\{\eta_j\}$的充分统计量。

**指数分布族的似然函数**  
$$
\begin{aligned}
L(\boldsymbol{\eta}|\boldsymbol{y})&=\prod_{i=1}^nf(y_i|\boldsymbol\eta)\\
&=\left(\prod_{i=1}^nb(y_i)  \right)(a^*(\boldsymbol{\eta}))^n\exp\left(\sum_{j=1}^k\eta_j\sum_{i=1}^nt_j(y_i)\right).
\end{aligned}
$$
于是对数似然函数为
$$
l(\boldsymbol{\eta}|\boldsymbol{y})=\sum_{i=1}^n\log b(y_i)+n\log a^*(\boldsymbol{\eta})+\sum_{j=1}^k\eta_jt_j(\boldsymbol{y}).
$$

**GLM setup**    
利用Aitkin et al. (1989) 提出的指数分布族的特定形式来定义  
假设$\{Y_i:i=1,...,n\}$是一组随机变量且它们的质量分布函数或质量密度函数为
$$

f(y_i|\theta_i,\phi)=\exp\left( \frac{y_i\theta_i-b(\theta_i)}{a(\phi)}+c(y_i,\phi) \right).\tag{2.1}\label{aitkin}
$$
其中$\theta_i$是规范参数，$\phi$是尺度参数。  
- 假设有一固定的函数$g(\cdot)$使得$\eta_i=g(\mu_i)=\boldsymbol{x}_i^T\beta$其中$\mu_i=E(Y_i|X_i).$下面将会证明$\theta_i$与$\mu_i$是相关的。  

**如何说明$f$是指数分布族？**  
- 如果$\phi$是已知的，则$f(\cdot)$是指数分布族。
- 如果$\phi$未知，则$f$可能不是指数分布族。  

**广义“线性模型”**  
令$a(\phi)=\phi,b(\theta_i)=\frac{\theta_i^2}{2},c(y,\phi)=-\frac{1}{2}(y^2/\phi+\log (2\pi\phi))$.  
于是
$$
\begin{aligned}
f(y_i|\theta_i,\phi)&=\exp\left( \frac{y_i\theta_i-\theta_i^2/2}{\phi}-\frac{1}{2}\left[ \frac{y^2}{\phi}+\log(2\pi\phi) \right] \right)\\
&=(2\pi\phi)^{-1/2}\exp\left( -\frac{(y_i-\theta_i)^2}{2\phi} \right),


\end{aligned}
$$
是正态分布的密度函数。如果我们令
$$
\eta_i=\theta_i=\mu_i=E(Y_i)=\boldsymbol{x}_i^T\beta,\text{ and }\phi=\sigma^2

$$
我们就得到了线性模型。
**$Y_i$的均值与方差**  
设$Y_i\sim f(y_i|\theta_i,\phi)$,接下来可以证明
$$
\begin{aligned}
E(Y_i)&=b'(\theta_i)\\
Var(Y_i)&=b''(\theta_i)a(\phi)
\end{aligned}
$$

由$(\ref{aitkin})$提出的指数分布族的特定形式可以证明
$$
\begin{aligned}
l(\theta_i|y_i)=\frac{y_i\theta_i-b(\theta_i)}{a(\phi)}+c(y_i,\phi)，\\
\frac{\partial}{\partial \theta_i}l(\theta_i|y_i)=\frac{1}{a(\phi)}[y_i-b'(\theta_i)].
\end{aligned}
$$
由恒等式$E\left[ \frac{\partial}{\partial \theta_i}l(\theta_i|y_i) \right]=0,$得
$$
\begin{aligned}
0&=\frac{1}{a(\phi)}[E(Y_i)-b'(\theta_i)],a(\phi)>0\\
\Rightarrow & E(Y_i)=b'(\theta).
\end{aligned}
$$
又因为$\frac{\partial^2}{\partial \theta_i^2}l(\theta_i|y_i)=-\frac{b''(\theta_i)}{a(\phi)}$,利用恒等式
$$
-E\left[ \frac{\partial^2}{\partial \theta_i^2}l(\theta_i|y_i) \right]=E\left[  \frac{\partial}{\partial \theta_i}l(\theta_i|y_i)\right]^2

$$
可以得到
$$
\begin{aligned}
\frac{b''(\theta_i)}{a(\phi)}=\frac{E[Y_i-b'(\theta_i)]^2}{a(\phi)^2}&\\
i.e.\; b''(\theta_i)=\frac{E[Y_i-b'(\theta_i)]^2}{a(\phi)}&\\
i.e.\;Var(Y_i)=b''(\theta_i)a(\phi)&.
\end{aligned}
$$

::: {.example #unnamed-chunk-1}
（二项分布）  

:::
**例2.1 二项分布作为指数分布族**  
设$Y_i,i=1,...,n$是一列独立且分别服从于二项分布$b(n,p_i)$的随机变量序列，则
$$
\begin{aligned}
f(y_i|p_i)&=\binom{n}{y_i}p_i^{y_i}(1-p_i)^{n-y_i}\\
&=\exp\left(y_i\log\left( \frac{p_i}{1-p_i} \right)+n_i\log(1-p_i)+\log\binom{n}{y_i}  \right)
\end{aligned}
$$  
令$\theta_i=\log\left( \frac{p_i}{1-p_i} \right)$,即
$$
p_i=\frac{e^{\theta_i}}{1+e^{\theta_i}}  
$$
又因为
$$
  \log(1-p_i)=\log\left(\frac{1}{1+e^{\theta_i}}  \right),
$$
于是我们有
$$
  f(y_i|p_i,\phi)=\exp\left(\frac{y_i\theta_i-b(\theta_i)}{a(\phi)}+c(y_i,\phi)  \right),
$$
其中
$$
\begin{aligned}
&a(\phi)=\phi=1,\\
&b(\theta_i)=n\log(1+e^{\theta_i}),\\
&c(y_i,\phi)=\log\binom{n}{y_i}.
\end{aligned}
$$
二项分布的期望和方差也可以通过上面提到的规范形式导出，正是我们很熟悉的结果。

**Poisson分布作为指数分布族**  
设$Y_i,i=1,...,n$是一列独立且分别服从于Poisson分布$P(\lambda_i)$的随机变量序列，则
$$
\begin{aligned}
f(y_i|\lambda_i,\phi)&=\frac{\lambda_i^{y_i}e^{-\lambda_i}}{y_i!}\\
&=\exp(y_i\log(\lambda_i)-\lambda_i-\log(y_i!)),
\end{aligned}
$$
令$\theta_i=\log\lambda_i,\lambda_i=e^{\theta_i}$,于是有
$$
\begin{aligned}
f(y_i|\lambda_i,\phi)&=\exp(y_i\theta_i-e^{\theta_i}-\log(y_i!)),\\
&=\exp\left(\frac{y_i\theta_i-b(\theta_i)}{a(\phi)}+c(y_i,\phi)  \right),
\end{aligned}
$$
其中
$$
\begin{aligned}
&a(\phi)=\phi=1,\\
&b(\theta_i)=e^{\theta_i},\\
&c(y_i,\phi)=\log(y_i!).
\end{aligned}
$$
也可以通过$b(\theta_i)$的$1,2$阶导数导出其期望与方差。

**链接函数**  
链接函数是一个函数$g(\cdot)$使得$\eta_i=g(\mu_i),$其中$\mu_i=E(Y_i).$我们在上面已经证明了$\mu_i=E(Y_i)=b'(\theta_i).$因此$g(\cdot)$将均值$\mu_i$或典型参数$\theta_i$与$\eta_i$连接起来。我们通常假设$g(\cdot)$是一个双射且连续可微的函数。

**典型链接函数**  
典型链接函数是指$g(\cdot)$使得$\eta_i=theta_i$。
这表明了
$$
\eta_i=g(b'(\theta_i))=\theta_i,
$$
因此$g(b'(\cdot))$必是恒等函数。

下面是典型链接函数的例子：
- 正态分布中$\eta_i=\theta_i=\mu_i$,于是典型链接函数就是恒等函数。
- 二项分布中$\eta_i=\theta_i=\log(\frac{p_i}{1-p_i})=\log(\frac{\mu_i}{1-\mu_i})$,于是典型连接函数是对数几率比函数(logit link function: log odds ratio).
- Poisson分布中$\eta_i=\theta_i=\log(\lambda_i)=\log(\mu_i)$,于是典型链接函数是对数函数。
- 
