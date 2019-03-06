## max_langevin_fixed_point(x, r, m)
- 译：langevin模型的最大定点
- 数学解释：从$argmax_x{h(x)=0}$多项式中估计$h(x)$，它已经能适应Langevin模型的确定性动力学

![](http://latex.codecogs.com/gif.latex?(x)(t)=h[x(t)]+R(N)(0,1))

这被下述的文章描述：
`Friedrich et al. (2000): Physics Letters A 271, p. 217-222 Extracting model equations from experimental data`
对于短时间序列，这个方法高度依赖于参数。
- 参数：
  - $x$(pandas.Series)计算时序特征的数据对象
  - $m$(int)适合估计动力学固定点的多项式的阶数
  - $r$(float)用于平均的分位数
- 返回：最大的确定性动力学定点（float浮点型）
- 函数类型：简单
- 代码示例：
```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae = tsf.feature_extraction.feature_calculators.max_langevin_fixed_point(ts, m, r)
```
---
## mean(x)
- 译：计算x序列的平均值
- 参数：$x$(pandas.Series)计算时序特征的数据对象
- 返回：这个特征的值（float浮点数）
- 函数类型：简单
- 代码示例：
```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae = tsf.feature_extraction.feature_calculators.mean(ts)
```
---
## mean_abs_change(x)
- 译：时间序列连续两点值的变化的绝对值的平均值
- 返回后续时间序列值之间的绝对差值的平均值：
![](http://latex.codecogs.com/gif.latex?\frac{1}{n}{\sum_{i=1,...,n-1}}|x_{i+1}-x_{i}|)
- 参数：$x$(pandas.Series)计算时序特征的数据对象
- 返回：这个特征的值（float浮点数）
- 函数类型：简单
- 代码示例：
```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae = tsf.feature_extraction.feature_calculators.mean_abs_change(ts)
```
---
## mean_change(x)
- 译：时间序列连续两点值的变化的平均值
- 返回后续时间序列值之间的差值的平均值：
![](http://latex.codecogs.com/gif.latex?\frac{1}{n}{\sum_{i=1,...,n-1}}x_{i+1}-x_{i})
- 参数：$x$(pandas.Series)计算时序特征的数据对象
- 返回：这个特征的值（float浮点数）
- 函数类型：简单
- 代码示例：
```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae = tsf.feature_extraction.feature_calculators.mean_change(ts)
```
---
## mean_second_derivative_central（x）
- 译：二阶导数的中心的均值
- 返回二阶导数的中心近似的平均值：
![](http://latex.codecogs.com/gif.latex?\frac{1}{n}{\sum_{i=1,...,n-1}}{\frac{1}{2}(x_{i+2}-2·x_{i+1}+x_i)})
- 参数：$x$(pandas.Series)计算时序特征的数据对象
- 返回：这个特征的值（float浮点数）
- 函数类型：简单
- 代码示例：
```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae = tsf.feature_extraction.feature_calculators.mean_second_derivative_central(ts)
```
---
## median(x)
- 译：计算$x$序列的中位数
- 返回$x$序列的中位数
- 参数：$x$(pandas.Series)计算时序特征的数据对象
- 返回：这个特征的值（float浮点数）
- 函数类型：简单
- 代码示例：
```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae = tsf.feature_extraction.feature_calculators.median(ts)
```
---
## minimum(x)
- 译：计算$x$序列的最小值
- 参数：$x$(pandas.Series)计算时序特征的数据对象
- 返回：这个特征的值（float浮点数）
- 函数类型：简单
- 代码示例：
```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae = tsf.feature_extraction.feature_calculators.minimum(ts)
```
---
## number_crossing_m(x, m)
- 译：计算$m$上的$x$的交叉数。交叉数被定义为两个连续值，其中第一个值小于$m$而下一个值更大，反之亦然。如果将$m$设置为零，则将获得零交叉的数量。
- 参数：
  - $x$(pandas.Series)计算时序特征的数据对象
  - $m$(float)交叉项的阈值
- 返回：这个特征的值（int整数型）
- 函数类型：简单
- 代码示例：
```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae = tsf.feature_extraction.feature_calculators.number_crossing_m(ts, m)
```
---
## number_cwt_peaks(x, n)
- 译：此特征计算器搜索$x$中的不同峰值。为此，$x$由ricker小波平滑，宽度范围从1到n。此特征计算器返回在足够宽度范围内出现的峰值数量，并具有足够高的信噪比（SNR）。
- 参数：
  - $x$(pandas.Series)计算时序特征的数据对象
  - $n$(int)考虑的最大宽度
- 返回：这个特征的值（整数型）
- 函数类型：简单
- 代码示例：
```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae = tsf.feature_extraction.feature_calculators.number_cwt_peaks(ts, n)
```
---
## number_peaks(x, n)
- 译：计算时间序列x中至少支持n的峰值数。支持n的峰值被定义为x的子序列，其中出现值大于其左边和右边的n个邻居。

因此在序列中:
>     >>> x = [3,0,0,4,0,0,13]
4是支持1和2的一个峰值，因为在子序列中：
>     >>> [0,4,0]
>     >>> [0,0,4,0,0]
4仍然是最大值。但是在这里，4不是支持3的峰值，因为13是4右边的第三个邻居并且比4大。
- 参数：
  - $x$(pandas.Series)计算时序特征的数据对象
  - $n$(int)峰的支持数
- 返回：这个特征的值（整数型）
- 函数类型：简单
- 代码示例：
```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae = tsf.feature_extraction.feature_calculators.number_cwt_peaks(ts, n)
```
---
## partial_autocorrelation(x, param)
- 译：计算给定滞后处的部分自相关函数的值。时间序列{${x_t,t=1...T}$}的滞后k部分自相关等于$x_t$和$x_{t-k}$适应中间变量{$x_{t-1},...,x_{t-k+1}$}([1])的部分相关。根据[2]之后，它可以定义为：

![](http://latex.codecogs.com/gif.latex?\alpha_k=\frac{Cov(x_t,x_{t-k}|x_{t-1},...,x_{t-k+1})}{\sqrt{Var(x_t|x_{t-1},...,x_{t-k+1})Var(x_{t-k}|x_{t-1},...,x_{t-k+1})}})

（a）$x_t=f(x_{t-1},...,x_{t-k+1})$和（b）$x_{t-k}=f(x_{t-1},...,x_{t-k+1})$是可以由OLS拟合的AR（k-1）模型。请注意，在（a）中，回归是对过去的值进行预测$x_t$。而在（b）中，未来的值用于计算过去的值$x_{t-k}$。在[1]中说“对于AR（p），部分自相关[$\alpha_k$]对于$k<=p$将是非零且对于$k>p$将为零"。使用此属性，它用于确定AR-过程的滞后。
参考：
[1] Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). Time series analysis: forecasting and control. John Wiley & Sons.  
[2] https://onlinecourses.science.psu.edu/stat510/node/62
- 参数：
  - $x$(pandas.Series)计算时序特征的数据对象
  - 参数（list列表）包含多个字典{“lag”: val}，用整数（val）显示返回的滞后值
- 返回：这个特征的值（float浮点型）
- 函数类型：组合器
- 代码示例：
```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae = tsf.feature_extraction.feature_calculators.partial_autocorrelation(ts, param)
```
---
## percentage_of_reoccurring_datapoints_to_all_datapoints(x)
- 译：返回多次出现在时间系列中的唯一值的百分比
- 计算公式：出现多于一次的不同值的个数 / 不同值的个数
这意味着该百分比标准化为惟一值的数量，而不是重复出现的值占所有值的百分比。
- 参数：$x$(pandas.Series)计算时序特征的数据对象
- 返回：这个特征的值（float浮点数）
- 函数类型：简单
- 代码示例：
```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae = tsf.feature_extraction.feature_calculators.percentage_of_reoccurring_datapoints_to_all_datapoints(ts)
```
---
## percentage_of_reoccurring_values_to_all_values(x)
- 译：返回多次出现在时间序列中的唯一值的比率
- 计算公式：出现多于一次的数据点的个数 / 所有数据点的个数
这意味着这个比率与时间序列中数据点的数量是标准化的，相比`the percentage_of_reoccurring_datapoints_to_all_datapoints`
- 参数：$x$(pandas.Series)计算时序特征的数据对象
- 返回：这个特征的值（float浮点数）
- 函数类型：简单
- 代码示例：
```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae = tsf.feature_extraction.feature_calculators.percentage_of_reoccurring_values_to_all_values(ts)
```
---
## quantile(x, q)
- 译：计算$x$的q分位数。这是大于$x$的有序值的前$q\%$的$x$值。
- 参数：
  - $x$(pandas.Series)计算时序特征的数据对象
  - $q$(float)计算中位数
- 返回：这个特征的值（浮点型）
- 函数类型：简单
- 代码示例：
```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae = tsf.feature_extraction.feature_calculators.quantile(ts, n)
```
---
## range_count(x, min, max)
- 译：计算区间[min，max]内的观测值的个数。
- 参数：
  - $x$(pandas.Series)计算时序特征的数据对象
  - min(int or float)范围包含下限
  - max(int or float)范围包含上限
- 返回：范围内值的个数（整型）
- 函数类型：简单
- 代码示例：
```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae = tsf.feature_extraction.feature_calculators.range_count(ts, min, max)
```
---
## ratio_beyond_r_sigma(x, r)
- 译：偏离x的平均值大于r * std(x)(so r sigma)的值的比率。
- 参数：$x$(iterable)计算时序特征的数据对象
- 返回：这个特征的值（浮点型）
- 函数类型：简单
- 代码示例：
```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ae = tsf.feature_extraction.feature_calculators.ratio_beyond_r_sigma(ts, r)
```
---
## ratio_value_number_to_time_series_length(x)
- 译：如果时间序列中的所有值仅出现一次，则返回1，如果不是这样，则小于1。原则上，它只是返回：
- 计算公式：单一的值 / 所有的值
- 参数：$x$(pandas.Series)计算时序特征的数据对象
- 返回：这个特征的值（float浮点数）
- 函数类型：简单
- 代码示例：
```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae = tsf.feature_extraction.feature_calculators.ratio_value_number_to_time_series_length(ts)
```
---
## sample_entropy(x)
- 译：计算和返回序列$x$的样本熵
参考：
[1] http://en.wikipedia.org/wiki/Sample_Entropy
[2] https://www.ncbi.nlm.nih.gov/pubmed/10843903?dopt=Abstract
- 参数：$x$(pandas.Series)计算时序特征的数据对象
- 返回：这个特征的值（float浮点数）
- 函数类型：简单
- 代码示例：
```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae = tsf.feature_extraction.feature_calculators.sample_entropy(ts)
```
---
## set_property(key, value)
该方法返回一个装饰器，该装饰器将函数的属性键设置为value。

---
## skewness(x)
- 译：返回x的样本偏度（使用调整后的Fisher-Pearson标准化力矩系数G1计算）
- 参数：$x$(pandas.Series)计算时序特征的数据对象
- 返回：这个特征的值（float浮点数）
- 函数类型：简单
- 代码示例：
```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae = tsf.feature_extraction.feature_calculators.skewness(ts)
```
---
## spkt_welch_density(x, param)
- 译：该特征计算器估计不同频率下时间序列x的交叉功率谱密度。为此，首先将时间序列从时域转移到频域。
特征计算器返回不同频率的功率谱。
- 参数：
  - $x$(pandas.Series)计算时序特征的数据对象
  - param(list)包括多个字典{“coeff”: x}($x$为整型)
- 返回：不同的特征值(pandas.Series)
- 函数类型：组合器
- 代码示例：
```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x) # 数据x假设已经获取
ae = tsf.feature_extraction.feature_calculators.spkt_welch_density(ts, param)
```
---
## standard_deviation(x)
- 译：返回$x$的标准偏差
- 参数：$x$(pandas.Series)计算时序特征的数据对象
- 返回：这个特征的值（float浮点数）
- 函数类型：简单
- 代码示例：
```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae = tsf.feature_extraction.feature_calculators.standard_deviation(ts)
```
---
## sum_of_reoccurring_data_points(x)
- 译：返回时间序列中出现超过一次的所有数据点的个数总和
- 参数：$x$(pandas.Series)计算时序特征的数据对象
- 返回：这个特征的值（float浮点数）
- 函数类型：简单
- 代码示例：
```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae = tsf.feature_extraction.feature_calculators.standard_deviation(ts)
```
---
## sum_of_reoccurring_values(x)
- 译：返回时间序列中出现超过一次的所有数据点的值总和
- 参数：$x$(pandas.Series)计算时序特征的数据对象
- 返回：这个特征的值（float浮点数）
- 函数类型：简单
- 代码示例：
```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae = tsf.feature_extraction.feature_calculators.sum_of_reoccurring_values(ts)
```
---
## sum_values(x)
- 译：计算时间序列值的总和
- 参数：$x$(pandas.Series)计算时序特征的数据对象
- 返回：这个特征的值（bool布尔型）
- 函数类型：简单
- 代码示例：
```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae = tsf.feature_extraction.feature_calculators.sum_values(ts)
```
---
## symmetry_looking(x, param)
- 译：布尔变量标识$x$的分布是否对称。这是一个案例如果：
![](http://latex.codecogs.com/gif.latex?|mean(X)-median(X)|<r*[max(X)-min(X)])
- 参数：
  - $x$(pandas.Series)计算时序特征的数据对象
  - $r$(float)对比的范围的比例
- 返回：这个特征的值（bool布尔型）
- 函数类型：组合器
- 代码示例：
```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae = tsf.feature_extraction.feature_calculators.symmetry_looking(ts, r)
```
---
## time_reversal_asymmetry_statistic(x, lag)
- 译：这个函数计算下式的值：
![](http://latex.codecogs.com/gif.latex?\frac{1}{n-2lag}{\sum_{i=0}^{n-2lag}}x^2_{i+2·lag}·x_{i+lag}-x_{i+lag}·x^2_i)
它是：
![](http://latex.codecogs.com/gif.latex?E[L^2(X)^2·L(X)-L(X)·X^2])
其中$E$是均值且$L$是滞后算子。它在[1]中被提出，作为一个从序列中提出的有用的特征。
参考：
[1] Fulcher, B.D., Jones, N.S. (2014). Highly comparative feature-based time-series classification. Knowledge and Data Engineering, IEEE Transactions on 26, 3026–3037.
- 参数：
  - $x$(pandas.Series)计算时序特征的数据对象
  - $lag$(int)这个值应该被特征计算使用
- 返回：这个特征的值（float浮点型）
- 函数类型：简单
- 代码示例：
```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae = tsf.feature_extraction.feature_calculators.time_reversal_asymmetry_statistic(ts, lag)
```
---
## value_count(x, value)
- 译：计算时间序列$x$中$value$出现的次数
- 参数：
  - $x$(pandas.Series)计算时序特征的数据对象
  - $value$(int or float)被计算的值
- 返回：计数(int整型)
- 函数类型：简单
- 代码示例：
```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae = tsf.feature_extraction.feature_calculators.value_count(ts, value)
```
---
## variance(x)
- 译：返回序列$x$的方差
- 参数：$x$(pandas.Series)计算时序特征的数据对象
- 返回：这个特征的值（float浮点型）
- 函数类型：简单
- 代码示例：
```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae = tsf.feature_extraction.feature_calculators.variance(ts)
```
---
## variance_larger_than_standard_deviation(x)
- 译：布尔变量，表示x的方差是否大于其标准差。是表示x的方差大于1
- 参数：$x$(pandas.Series)计算时序特征的数据对象
- 返回：这个特征的值（bool布尔型）
- 函数类型：简单
- 代码示例：
```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae = tsf.feature_extraction.feature_calculators.variance_larger_than_standard_deviation(ts)
```
---