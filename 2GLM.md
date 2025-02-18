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
