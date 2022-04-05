# 朴素贝叶斯法
## 基本概念 

离散情况下

- 模式状态（**随机变量**）： $\omega_j$

- 概率（**先验**）: $P\left(\omega_{j}\right)$

  模式属于 $\omega_{j}$ 类的频率值

- 概率密度函数（**证据**）: $P(x)$

  模式的某一特征 $x$ 的各取值的频率值

- 类条件概率密度（**似然**）： $P\left(x \mid \omega_{j}\right)$

  在模式属于 $\omega_{j}$ 类的条件下，模式的某一特征 $x$ 各取值的频率值

- 条件概率（**后验**）：$P\left(\omega_{j} \mid x\right)$

  在给点给特征 $x$ 测量值的条件下，模式属于 $\omega_{j}$ 类的频率值

## 最小化分类错误率

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
  P\left(Y=c_{k} \mid X=\boldsymbol{x}\right)=\frac{P\left(Y=c_{k}\right) \prod_{j}{ }P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right)}{\sum_{k} P\left(Y=c_{k}\right) \prod_{j} P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right)}
  $$

------



- 贝叶斯分类器

  朴素贝叶斯法将实例分到后验概率最大的类中，等价于$0-1$ 损失函数下的期望风险最小化：
  $$
  y=f(x)=\arg \max _{c_{k}} \frac{P\left(Y=c_{k}\right) \prod_{j} P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right)}{\sum_{k} P\left(Y=c_{k}\right) \prod_{j} P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right)}
  $$
  分母对所有 $c_{k}$ 都相同，所以只需要最大化分子即可
  $$
  y=\arg \max _{c_{k}} P\left(Y=c_{k}\right) \prod_{j} P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right)
  $$

### 参数估计

#### 极大似然估计



- **先验**概率 $P\left(Y=c_{k}\right)$ 的极大似然估计：
  $$
  P\left(Y=c_{k}\right)=\frac{\sum_{i=1}^{N} I\left(y_{i}=c_{k}\right)}{N}, k=1,2, \cdots, K
  $$
  
- 设第 $j$ 个特征 $x^{(j)}$ 可能取值的集合为：$\left\{a_{j 1}, a_{j 2}, \cdots, a_{j s_{j}}\right\}$，条件概率（**似然**）的极大似然估计：
  $$
  P\left(X^{(j)}=a_{j l} \mid Y=c_{k}\right)=\frac{\sum_{i=1}^{N} I\left(x_{i}^{(j)}=a_{j l}, y_{i}=c_{k}\right)}{\sum_{i=1}^{N} I\left(y_{i}=c_{k}\right)}\\
  j=1,2, \cdots, n ; l=1,2, \cdots, S_{j} ; k=1,2, \cdots, K
  $$

#### 贝叶斯估计

考虑用极大似然估计可能会出现所要估计的概率值为 0 的情况, 这时会影响到后验概率的计算结果, 使分类产生偏差。解决这一问题的方法是采用贝叶斯估计，即在随机变量各个取值的频数上加上一个正数 $\lambda\gt0$，可以验证这样得到的 $P_{\lambda}\left(X^{(j)}=a_{j l} \mid Y=c_{k}\right)$ 仍然是概率分布。

- **先验**概率的贝叶斯估计：
  $$
  P_{\lambda}\left(Y=c_{k}\right)=\frac{\sum_{i=1}^{N} I\left(y_{i}=c_{k}\right)+\lambda}{N+K \lambda}
  $$
  
- 条件概率的（**似然**）贝叶斯估计：
  $$
  P_{\lambda}\left(X^{(j)}=a_{j l} \mid Y=c_{k}\right)=\frac{\sum_{i=1}^{N} I\left(x_{i}^{(j)}=a_{j l}, y_{i}=c_{k}\right)+\lambda}{\sum_{i=1}^{N} I\left(y_{i}=c_{k}\right)+S_{j} \lambda}
  $$

### 算法流程

1. 通过极大似然估计或者贝叶斯估计得到先验概率和条件概率
   
2. 对于给定的实例 $\boldsymbol{x}=\left(x^{(1)}, x^{(2)}, \cdots, x^{(n)}\right)^{T}$，计算后验：
   $$
   P\left(Y=c_{k} \mid X=\boldsymbol{x}\right)=\frac{P\left(X=\boldsymbol{x} \mid Y=c_{k}\right) P\left(Y=c_{k}\right)}{\sum_{k} P\left(X=\boldsymbol{x} \mid Y=c_{k}\right) P\left(Y=c_{k}\right)}
   $$
   
3. 最大化后验（分子）确定 $\boldsymbol{x}$ 的类别：

$$
y=\arg \max _{c_{k}} P\left(Y=c_{k}\right) \prod_{j=1}^{n} P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right)
$$

## 最小化风险

### 基本方法

上述算法本质上默认采用了 $0-1$ 损失函数，目的是最小化分类错误率，分类错误的损失即为1。但现实情况下，不同的分类错误导致的后果通常是不同的。

更一般地，考虑损失 $ \lambda_{ij} $ 表示将一个真实标记为 $c_{j}$ 的样本误分类为 $c_{i}$ 所产生的损失。基于后验概率 $P\left(c_{i} \mid \boldsymbol{x}\right)$ 可获得将样本 $\boldsymbol{x}$ 分类为 $c_{i}$ 所产生的期望损失(expected loss), 即在样本 $\boldsymbol{x}$ 上的 “条件风险” (conditional risk)
$$
R\left(Y=c_{i} \mid X=\boldsymbol{x}\right)=\sum_{j=1}^{K} \lambda_{i j} P\left(Y=c_{j} \mid X=\boldsymbol{x}\right)
$$
我们的任务是寻找一个判定准则 $y: X \mapsto Y$ 以最小化期望风险
$$
R(y)=\mathbb{E}_{\boldsymbol{x}}[R(Y=y(\boldsymbol{x}) \mid X=\boldsymbol{x})] .
$$
显然，对每个样本 $\boldsymbol{x}$，若 $y$ 能最小化条件风险 $R(Y=y(\boldsymbol{x}) \mid X=\boldsymbol{x})$, 则期望风险 $R(y)$ 也将被最小化。这就产生了贝叶斯判定准则(Bayes decision rule)：为最小化总体风险，只需在每个样本上选择那个能使条件风险 $R(Y=c_{i} \mid X=\boldsymbol{x})$ 最小的类别标记 $c_i$，即
$$
y^{*}(\boldsymbol{x})=\underset{c_i}{\arg \min }\ R(Y=c_{i} \mid X=\boldsymbol{x})
$$
此时，$y^{*}$ 称为贝叶斯最优分类器(Bayes optimal classifier)，与之对应的总体风险 $R\left(y^{*}\right)$ 称为贝叶斯风险(Bayes risk). $1-R\left(y^{*}\right)$ 反映了分类器所能达到的最好性能，即通过机器学习所能产生的模型精度的理论上限。

### 算法流程

1. 参数估计得到先验和似然，然后计算后验：
   $$
   P\left(Y=c_{i} \mid X=\boldsymbol{x}\right)=\frac{P\left(X=\boldsymbol{x} \mid Y=c_{i}\right) P\left(Y=c_{i}\right)}{\sum_{k} P\left(X=\boldsymbol{x} \mid Y=c_{i}\right) P\left(Y=c_{i}\right)}
   $$
   
2. 计算条件风险：
   $$
   R\left(Y=c_{i} \mid X=\boldsymbol{x}\right)=\sum_{j=1}^{K} \lambda_{i j} P\left(Y=c_{j} \mid X=\boldsymbol{x}\right)
   $$

3. 选择风险最小的类别：
   $$
   y=\underset{c_i}{\arg \min }\ R(Y=c_{i} \mid X=\boldsymbol{x})
   $$

