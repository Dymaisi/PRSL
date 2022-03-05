# 朴素贝叶斯法
### 基本概念

- 模式状态（**随机变量**）： $\omega_j$

- 概率（**先验**）: $P\left(\omega_{j}\right)$

  模式属于 $\omega_{j}$ 类的频率值

- 概率密度函数（**证据**）: $p(x)$

  对模式的某一特征 $x$ 进行测量，出现的频率值

- 类条件概率密度（**似然**）： $p\left(x \mid \omega_{j}\right)$

  在模式属于 $\omega_{j}$ 类的条件下，对模式的某一特征 $x$ 进行测量，出现的频率值

- 条件概率（**后验**）：$P\left(\omega_{j} \mid x\right)$

  在给点给特征 $x$ 测量值的条件下，模式属于 $\omega_{j}$ 类的可能性

### 基本方法

- 训练数据集： $T=\left\{\left(\boldsymbol{x}_{1}, y_{1}\right),\left(\boldsymbol{x}_{2}, y_{2}\right), \cdots,\left(\boldsymbol{x}_{N}, y_{N}\right)\right\}$，由 $X$ 和 $Y$ 的联合概率分布 $P(X, Y)$ 独立同分布产生

- 朴素贝叶斯通过训练数据集学习联合概率分布 $P(X, Y)$

  通过计算先验概率分布：$P\left(Y=c_{k}\right), k=1,2, \cdots, K$

  及条件概率分布：$P\left(X=x \mid Y=c_{k}\right)=P\left(X^{(1)}=x^{(1)}, \cdots, X^{(n)}=x^{(n)} \mid Y=c_{k}\right), k=1,2, \cdots, K$ 进而得到联合分布概率

- 条件概率为指数级别的参数: $K \prod_{j=1}^{n} S_{j}$

------



- 条件独立性假设：

$$
P\left(X=x \mid Y=c_{k}\right)=P\left(X^{(1)}=x^{(1)}, \cdots, X^{(n)}=x^{(n)} \mid Y=c_{k}\right)=\prod_{j=1}^{n} P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right)
$$

”朴素“贝叶斯名字由来，牺牲分类准确性

------



- 贝叶斯定理：
  $$
  P\left(Y=c_{k} \mid X=\boldsymbol{x}\right)=\frac{P\left(X=\boldsymbol{x} \mid Y=c_{k}\right) P\left(Y=c_{k}\right)}{\sum_{k} P\left(X=\boldsymbol{x} \mid Y=c_{k}\right) P\left(Y=c_{k}\right)}
  $$
  

  由条件独立性假设，有

   
  $$
  P\left(Y=c_{k} \mid X=\boldsymbol{x}\right)=\frac{P\left(Y=c_{k}\right) \prod_{j}{ }^{P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right)}}{\sum_{k} P\left(Y=c_{k}\right) \prod_{j} P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right)}
  $$

------



- 贝叶斯分类器

  朴素贝叶斯法将实例分到后验概率最大的类中，等价于期望风险最小化：
  $$
  y=f(x)=\arg \max _{c_{k}} \frac{P\left(Y=c_{k}\right) \prod_{j} P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right)}{\sum_{k} P\left(Y=c_{k}\right) \prod_{j} P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right)}
  $$
  分母对所有 $c_{k}$ 都相同，所以只需要最大化似然即可
  $$
  y=\arg \max _{c_{k}} P\left(Y=c_{k}\right) \prod_{j} P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right)
  $$

### 参数估计

- **先验**概率 $P\left(Y=c_{k}\right)$ 的极大似然估计：
  $$
  P\left(Y=c_{k}\right)=\frac{\sum_{i=1}^{N} I\left(y_{i}=c_{k}\right)}{N}, k=1,2, \cdots, K
  $$
  

- 设第 $j$ 个特征 $x^{(j)}$ 可能取值的集合为：$\left\{a_{j 1}, a_{j 2}, \cdots, a_{j s_{j}}\right\}$，条件概率（**似然**）的极大似然估计：
  $$
  P\left(X^{(j)}=a_{j l} \mid Y=c_{k}\right)=\frac{\sum_{i=1}^{N} I\left(x_{i}^{(j)}=a_{j l}, y_{i}=c_{k}\right)}{\sum_{i=1}^{N} I\left(y_{i}=c_{k}\right)}\\
  j=1,2, \cdots, n ; l=1,2, \cdots, S_{j} ; k=1,2, \cdots, K
  $$

### 算法流程

1. 计算先验概率和条件概率
   $$
   \begin{gathered}
   P\left(Y=c_{k}\right)=\frac{\sum_{i=1}^{N} I\left(y_{i}=c_{k}\right)}{N}, k=1,2, \cdots, K \\
   P\left(X^{(j)}=a_{j l} \mid Y=c_{k}\right)=\frac{\sum_{i=1}^{N} I\left(x_{i}^{(j)}=a_{j l}, y_{i}=c_{k}\right)}{\sum_{i=1}^{N} I\left(y_{i}=c_{k}\right)} \\
   j=1,2, \cdots, n ; \quad l=1,2, \cdots, S_{j} ; \quad k=1,2, \cdots, K
   \end{gathered}
   $$
   

2. 对于给定的实例 $\boldsymbol{x}=\left(x^{(1)}, x^{(2)}, \cdots, x^{(n)}\right)^{T}$，计算
   $$
   P\left(Y=c_{k}\right) \prod_{j=1}^{n} P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right), k=1,2, \cdots, K
   $$
   

3. 确定 $\boldsymbol{x}$ 的类别

$$
y=\arg \max _{c_{k}} P\left(Y=c_{k}\right) \prod_{j=1}^{n} P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right)
$$