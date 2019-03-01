
## abs_energy(x)
* 译：绝对能量值
* 返回时序数据的绝对能量（平方和）

$$E=\sum_{i=1}^n x_i^2$$
* 参数：$x$   (pandas.Series)计算时序特征的数据对象
* 返回值：绝对能量值（浮点数）
* 函数类型：简单
* 代码示例：

```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae = tsf.feature_extraction.feature_calculators.abs_energy(ts)
```
* 注释：描述时序数据的离原点的平方波动情况（能量）
——————————————————————————————————————————————————————————

## absolute_sum_of_changes(x) ##
* 译：一阶差分绝对和
* 返回时序数据的一阶差分结果的绝对值之和

![](http://latex.codecogs.com/gif.latex?\sum_{i=1}^{n-1} |x_{i+1}-x_i|

* 参数：$x$   (pandas.Series)计算时序特征的数据对象
* 返回值：一阶差分绝对和（非负浮点数）
* 函数类型：简单
* 代码示例：

```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae = tsf.feature_extraction.feature_calculators.absolute_sum_of_changes(ts)
```
* 注释：描述时序数据相邻观测值之间的绝对波动情况
——————————————————————————————————————————————————————————

## agg_autocorrelation(x, param) ##
* 译：各阶自相关系数的聚合统计特征
* 返回时序数据的各阶差分值之间的聚合（方差、均值）统计特征

$$R(l)=\frac{1}{(n-l)\sigma^2}\sum_{i=1}^{n-l}(x_i-\mu)(x_{i+l}-\mu)$$
* 参数：
  *  $x$   (pandas.Series)计算时序特征的数据对象
  * parma(list)   包含一个字典{“f_agg”: x, “maxlag”, n} 其中x为聚合函数名，n为最大差分阶数
* 返回值：各阶自相关系数的聚合统计特征（浮点数）
* 函数类型：组合
* 代码示例：

```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
param = [{'f_agg': 'mean', 'maxlag':2}]
ae = tsf.feature_extraction.feature_calculators.agg_autocorrelation(ts, param)
```
* 注释：统计时序数据各界差分间的聚合特征，依据不同的聚合函数，刻画时序数据的不同特征
——————————————————————————————————————————————————————————

## agg_linear_trend(x, param) ##
* 译：基于分块时序聚合值的线性回归
* 返回时序数据的分块聚合后的线性回归（基于OLS）


* 参数：
  *  $x$   (pandas.Series)计算时序特征的数据对象
  * parma(list)   包含一个字典{“attr”: x, “chunk_len”: l, “f_agg”: f}其中“f_agg”为聚合函数名，“chunk_len”指定每块数据量，“attr”为线性回归结果参数属性： “pvalue”, “rvalue”, “intercept”, “slope”, “stderr”。
* 返回值：指定的线性回归属性值（zip，可用list读取）
* 函数类型：简单
* 代码示例：

```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
param = [{'f_agg': 'mean','attr': 'pvalue', 'chunk_len': 2}]
ae=tsf.feature_extraction.feature_calculators.agg_linear_trend(ts,param)
print(ae,list(ae))
```
* 注释：略

——————————————————————————————————————————————————————————

## approximate_entropy(x, m, r) ##
* 译：近似熵
* 衡量时序数据的的周期性、不可预测性和波动性


* 参数：
  *  $x$   (pandas.Series)计算时序特征的数据对象
  * m (int)对照运行数据长度
  * r (float) 过滤阈值（非负数）
* 返回值：近似熵（浮点数）
* 函数类型：简单
* 代码示例：

```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae=tsf.feature_extraction.feature_calculators.approximate_entropy(ts, 10, 0.1)
```
* 注释：相邻熵值间的比值，是一个相对量

——————————————————————————————————————————————————————————

## ar_coefficient(x, param) ##
* 译：自回归系数
* 衡量时序数据的的周期性、不可预测性和波动性

$$X_t=\psi_0+\sum_{i=1}^k \psi_i X_{t-i} + \varepsilon_t$$
* 参数：
  *  $x$   (pandas.Series)计算时序特征的数据对象
  * paramm (lsit) {“coeff”: x, “k”: y}其中“coeff”自回归中第X项系数，“k”为自回归阶数
* 返回值：自回归系数（pandas.Series）
* 函数类型：组合
* 代码示例：

```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
param = [{'coeff': 0, 'k': 10}]
ae=tsf.feature_extraction.feature_calculators.ar_coefficient(ts, param)
```
* 注释：自回归方程的各阶系数$\psi_i$
