
## abs_energy(x)
* 译：绝对能量值
* 返回时序数据的绝对能量（平方和）

![](http://latex.codecogs.com/gif.latex?E=\sum_{i=1}^nx_i^2)
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

![](http://latex.codecogs.com/gif.latex?\sum_{i=1}^{n-1}|x_{i+1}-x_i|)
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

![](http://latex.codecogs.com/gif.latex?R(l)=\frac{1}{(n-l)\sigma^2}\sum_{i=1}^{n-l}(x_i-\mu)(x_{i+l}-\mu))
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

![](http://latex.codecogs.com/gif.latex?X_t=\psi_0+\sum_{i=1}^k\psi_{i}X_{t-i}+\varepsilon_t)
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
——————————————————————————————————————————————————————————

## augmented_dickey_fuller(x, param) ##
* 译：扩展迪基-福勒检验（ADF检验）
* 测试一个自回归模型是否存在单位根，衡量时序数据的平稳性

* 参数：
  *  $x$   (pandas.Series)计算时序特征的数据对象
  * paramm (lsit)  {“attr”: x} 其中x是字符串，包含“teststat”, “pvalue” 和“usedlag”
* 返回值：ADF检验统计值（浮点数）
* 函数类型：组合
* 代码示例：

```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
param = [{'attr': 'pvalue'}]
ae=tsf.feature_extraction.feature_calculators.augmented_dickey_fuller(ts, param)
```
* 注释：返回ADF检验统计值
——————————————————————————————————————————————————————————

## autocorrelation(x, lag) ##
* 译：lag阶自相关性
* 计算lag阶滞后时序数据的自相关性（浮点数）

* 参数：
  *  $x$   (pandas.Series)计算时序特征的数据对象
  * lag（int）时序数据滞后阶数
* 返回值：自相关性值（浮点数）
* 函数类型：简单
* 代码示例：

```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae=tsf.feature_extraction.feature_calculators.autocorrelation(ts, 2)
```
* 注释：lag阶自相关性值
——————————————————————————————————————————————————————————

## binned_entropy(x, max_bins) ##
* 译：分组熵
* 把整个序列按值均分成max_bins个桶，然后把每个值放进相应的桶中，然后求熵（浮点数）

* 参数：
  *  $x$   (pandas.Series)计算时序特征的数据对象
  * max_bins（int）分组数
* 返回值：分组熵（浮点数）
* 函数类型：简单
* 代码示例：

```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae=tsf.feature_extraction.feature_calculators.binned_entropy(ts, 10)
```
* 注释：时序数据等距分组求熵
——————————————————————————————————————————————————————————

## c3(x, lag) ##
* 译：时序数据非线性度量
* 基于物理学的时序数据非线性度量（浮点数）

![](http://latex.codecogs.com/gif.latex?\frac{1}{n-2lag}\sum_{i=0}^{n-2lag}x_{i+2lag}^2x_{i+lag}x_i)
等同于计算
$$\mathbb{E}[L^2(X)L(X)X]$$
其中L为时滞算子。
* 参数：
  *  $x$   (pandas.Series)计算时序特征的数据对象
  * lag（int）时滞阶数
* 返回值：非线性度（浮点数）
* 函数类型：简单
* 代码示例：

```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae=tsf.feature_extraction.feature_calculators.c3(ts, 2)
```
* 注释：时序数据非线性度
——————————————————————————————————————————————————————————

## change_quantiles(x, ql, qh, isabs, f_agg) ##
* 译：给定区间的时序数据描述统计
* 先用ql和qh两个分位数在x中确定出一个区间，然后在这个区间里计算时序数据的均值、绝对值、连续变化值。（浮点数）

* 参数：
  *  $x$   (pandas.Series)计算时序特征的数据对象
  * ql（float）区间下界
  * qh（float）区间上界
  * isabs（bool）是否采用绝对差值
  * f_agg（str）numpy聚合函数，如mean,std等
* 返回值：区间内描述统计量（浮点数）
* 函数类型：简单
* 代码示例：

```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd
import numpy as np

ts = pd.Series(x)  #数据x假设已经获取
ae=tsf.feature_extraction.feature_calculators.change_quantiles(ts, 0.05, 0.95, False, 'mean')
```
* 注释：时序数据区间内描述统计量
——————————————————————————————————————————————————————————

## cid_ce(x, normalize) ##
* 译：时序数据复杂度
* 用来评估时间序列的复杂度，越复杂的序列有越多的谷峰。 （浮点数）

* 参数：
  *  $x$   (pandas.Series)计算时序特征的数据对象
  * normalize (bool) 是否对数据进行z标准化

* 返回值：时序数据复杂度（浮点数）
* 函数类型：简单
* 代码示例：

```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae=tsf.feature_extraction.feature_calculators.cid_ce(ts, True)
```
* 注释：时序数据复杂度
——————————————————————————————————————————————————————————

## count_above_mean(x) ##
* 译：高于均值个数
* 统计高于时序数据均值的个数 （整数）

* 参数：
  *  $x$   (pandas.Series)计算时序特征的数据对象

* 返回值：高于均值个数（整数）
* 函数类型：简单
* 代码示例：

```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae=tsf.feature_extraction.feature_calculators.count_above_mean(ts)
```
* 注释：高于均值个数
——————————————————————————————————————————————————————————

## count_below_mean(x) ##
* 译：低于均值个数
* 统计低于时序数据均值的个数 （整数）

* 参数：
  *  $x$   (pandas.Series)计算时序特征的数据对象

* 返回值：低于均值个数（整数）
* 函数类型：简单
* 代码示例：

```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae=tsf.feature_extraction.feature_calculators.count_below_mean(ts)
```
* 注释：低于均值个数
——————————————————————————————————————————————————————————

## cwt_coefficients(x, param) ##
* 译：Ricker小波分析
* 连续的小波分析，ricker子波是地震勘探中常用的子波类型，ricker子波是基于波动方程严格推导得到的。（pandas.Series）

![](http://latex.codecogs.com/gif.latex?\frac{2}{\sqrt{3a}\pi^{\frac{1}{4}}}(1-\frac{x^2}{a^2})\exp(-\frac{x^2}{2a^2}))
其中，a是小波变换函数中的宽度参数。
* 参数：
  *  $x$   (pandas.Series)计算时序特征的数据对象
  * param (list)  {“widths”:x, “coeff”: y, “w”: z} 中x为整数值构成的数组，y,z都是整数值。

* 返回值：小波分析（pandas.Series）
* 函数类型：组合
* 代码示例：

```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
param = [ {'widths':tuple([2,2,2]), 'coeff': 2, 'w': 2}]
ae=tsf.feature_extraction.feature_calculators.cwt_coefficients(ts, param)
print(list(ae))
```
* 注释：width参数需要可hash对象，最后返回结果可用list查看
——————————————————————————————————————————————————————————

## energy_ratio_by_chunks(x, param) ##
* 译：分块局部熵比率
* 将时序数据分块后，计算目标块数据的熵与全体的熵比率。当数据不够均分时，会将多余的数据在前面的块中散布。（浮点数）

* 参数：
  *  $x$   (pandas.Series)计算时序特征的数据对象
  * param –  {“num_segments”: N, “segment_focus”: i} with N, i 取值均为整数

* 返回值：分块局部熵比率（list）
* 函数类型：组合
* 代码示例：

```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
param = [{'num_segments': 10, 'segment_focus': 5} ]
ae=tsf.feature_extraction.feature_calculators.energy_ratio_by_chunks(ts, param)
```
* 注释：segment_focus是从0开始计数的，返回值为一个列表包含元组
——————————————————————————————————————————————————————————

## fft_aggregated(x, param) ##
* 译：绝对傅里叶变换的谱统计量
* 返回绝对傅里叶变换后的光谱质心、峰度、偏度等值（pandas.Series）

* 参数：
  *  $x$   (pandas.Series)计算时序特征的数据对象
  * param (list)  {“aggtype”: s} s为字符串取值于 [“centroid”, “variance”, “skew”, “kurtosis”]
* 返回值：绝对傅里叶变换的谱统计量（pandas.Series）
* 函数类型：组合
* 代码示例：

```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
param = [{'aggtype': 'skew'}]
ae=tsf.feature_extraction.feature_calculators.fft_aggregated(ts, param)
print(list(ae))
```
* 注释：略
——————————————————————————————————————————————————————————

## fft_coefficient(x, param) ##
* 译：傅里叶变换系数
* 基于快速傅里叶变换算法计算一维离散傅里叶序列的系数（pandas.Series）

![](http://latex.codecogs.com/gif.latex?A_k=\sum_{m=0}^{n-1}a_m\exp{(-2\pi i\frac{mk}{n})})

* 参数：
  *  $x$   (pandas.Series)计算时序特征的数据对象
  * param (list) {“coeff”: x, “attr”: s} x正整数, s为字符串取值于 [“real”, “imag”, “abs”, “angle”]
* 返回值：傅里叶变换系数（pandas.Series）
* 函数类型：组合
* 代码示例：

```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
param = [{'coeff': 2, 'attr': 'angle'}]
ae=tsf.feature_extraction.feature_calculators.fft_coefficient(ts, param)
print(list(ae))
```
* 注释： [“real”, “imag”, “abs”, “angle”]分别对应系数的实值部、虚值部、绝对值、角度值。
——————————————————————————————————————————————————————————

## first_location_of_maximum(x) ##
* 译：最大值位置
* 基于时序数据长度的相对最大值位置

* 参数：
  *  $x$   (pandas.Series)计算时序特征的数据对象

* 返回值：相对最大值位置（float）
* 函数类型：简单
* 代码示例：

```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae=tsf.feature_extraction.feature_calculators.first_location_of_maximum(ts)
```
* 注释：返回值为pandas.Series，每个值都是最大值位置（在整个序列中的位置）与序列长度的比值
——————————————————————————————————————————————————————————

## first_location_of_minimum(x) ##
* 译：最小值位置
* 基于时序数据长度的相对最小值位置

* 参数：
  *  $x$   (pandas.Series)计算时序特征的数据对象

* 返回值：相对最小值位置（float）
* 函数类型：简单
* 代码示例：

```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae=tsf.feature_extraction.feature_calculators.first_location_of_minimum(ts)
```
* 注释：返回值为pandas.Series，每个值都是最小值位置（在整个序列中的位置）与序列长度的比值
——————————————————————————————————————————————————————————

## friedrich_coefficients(x, param) ##
<font face="黑体" color=red size=5>调用接口未成功</font>
* 译：Langevin模型拟合的多项式系数
* 基于确定动力学模型Langevin拟合的多项式系数（pandas.Series）

![](http://latex.codecogs.com/gif.latex?\dot{x}(t)=h(x(t))+\mathcal{N}(0,R))
* 参数：
  *  $x$   (pandas.Series)计算时序特征的数据对象
  * param (list)  {“m”: x, “r”: y, “coeff”: z} x为正整数，是多项式拟合的最高阶数，y是正实数，用于计算均值的分位数，z为正整数，多项式的第几项。

* 返回值：多项式拟合系数（pandas.Series）
* 函数类型：组合
* 代码示例：

```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
param = [{'m':3, 'r': 0.11, 'coeff': 2}]
ae=tsf.feature_extraction.feature_calculators.friedrich_coefficients(ts, param)
```
* 注释：略
——————————————————————————————————————————————————————————

## has_duplicate(x) ##

* 译：重复记录检验
* 检查时序数据是否有重复记录（bool）

* 参数：
  *  $x$   (pandas.Series)计算时序特征的数据对象

* 返回值：重复值存在与否（bool）
* 函数类型：简单
* 代码示例：

```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
param = [{'m':3, 'r': 0.11, 'coeff': 2}]
ae=tsf.feature_extraction.feature_calculators.has_duplicate(ts)
```
* 注释：略
——————————————————————————————————————————————————————————

## has_duplicate_max(x) ##

* 译：最大值记录重复检验
* 检查时序数据最大记录是否有重复记录（bool）

* 参数：
  *  $x$   (pandas.Series)计算时序特征的数据对象

* 返回值：最大记录重复存在与否（bool）
* 函数类型：简单
* 代码示例：

```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
param = [{'m':3, 'r': 0.11, 'coeff': 2}]
ae=tsf.feature_extraction.feature_calculators.has_duplicate_max(ts)
```
* 注释：略
——————————————————————————————————————————————————————————

## has_duplicate_min(x) ##

* 译：最小值记录重复检验
* 检查时序数据最小记录是否有重复记录（bool）

* 参数：
  *  $x$   (pandas.Series)计算时序特征的数据对象

* 返回值：最小记录重复存在与否（bool）
* 函数类型：简单
* 代码示例：

```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
param = [{'m':3, 'r': 0.11, 'coeff': 2}]
ae=tsf.feature_extraction.feature_calculators.has_duplicate_min(ts)
```
* 注释：略
——————————————————————————————————————————————————————————

## index_mass_quantile(x, param) ##

* 译：分位数索引
* 计算某分位数对应的索引值（pandas.Series）

* 参数：
  *  $x$   (pandas.Series)计算时序特征的数据对象
  * param (list) {“q”: x} x为分位数值

* 返回值：分位数索引（pandas.Series）
* 函数类型：组合
* 代码示例：

```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
param = [{'q':50}]
ae=tsf.feature_extraction.feature_calculators.index_mass_quantile(ts, param)
```
* 注释：略
——————————————————————————————————————————————————————————

## kurtosis(x) ##

* 译：峰度
* 计算基于修正的Fisher-Pearson矩统计量的峰度（float）

* 参数：
  *  $x$   (pandas.Series)计算时序特征的数据对象

* 返回值：修正的峰度（float）
* 函数类型：组合
* 代码示例：

```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae=tsf.feature_extraction.feature_calculators.kurtosis(ts)
```
* 注释：表征概率密度分布曲线在平均值处峰值高低的特征数。
——————————————————————————————————————————————————————————

## large_standard_deviation(x, r) ##

* 译：标准差是否倍于极差
* 标准差是否为数据范围的r倍（bool）

![](http://latex.codecogs.com/gif.latex?std(x)>r*(max(x)-min(x)))
* 参数：
  *  $x$   (pandas.Series)计算时序特征的数据对象
  * r (float) – 比率值
* 返回值：标准差是否倍于极差（bool）
* 函数类型：组合
* 代码示例：

```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae=tsf.feature_extraction.feature_calculators.large_standard_deviation(ts, 0.2)
```
* 注释：根据经验法则，标准偏差应该是数值范围的四分之一。
——————————————————————————————————————————————————————————

## last_location_of_maximum(x) ##

* 译：最大值最近位置
* 基于时序数据长度的相对最大值最近位置（float）

* 参数：
  *  $x$   (pandas.Series)计算时序特征的数据对象

* 返回值：相对最大值最近位置（float）
* 函数类型：简单
* 代码示例：

```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae=tsf.feature_extraction.feature_calculators.last_location_of_maximum(ts)
```
* 注释：返回值为一个相对位置和时序数据长度的比值
——————————————————————————————————————————————————————————

## last_location_of_minimum(x) ##

* 译：最小值最近位置
* 基于时序数据长度的相对最小值最近位置（float）

* 参数：
  *  $x$   (pandas.Series)计算时序特征的数据对象

* 返回值：相对最小值最近位置（float）
* 函数类型：简单
* 代码示例：

```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae=tsf.feature_extraction.feature_calculators.last_location_of_minimum(ts)
```
* 注释：返回值为一个相对位置和时序数据长度的比值
——————————————————————————————————————————————————————————

## length(x) ##

* 译：数据记录数
* 统计时序数据的总记录数（int）

* 参数：
  *  $x$   (pandas.Series)计算时序特征的数据对象

* 返回值：数据总行数（int）
* 函数类型：简单
* 代码示例：

```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
ae=tsf.feature_extraction.feature_calculators.length(ts)
```
* 注释：略
——————————————————————————————————————————————————————————

## linear_trend(x, param) ##
<font face="黑体" color=red size=5>调用接口未成功</font>
* 译：线性回归分析
* 基于最小二乘，自变量为索引（0-len(x)-1)的线性回归，认为数据是简单采样所得（pandas.Series）

* 参数：
  *  $x$   (pandas.Series)计算时序特征的数据对象
  * param (list) {“attr”: x} x为线性回归中的结果变量名，如'pvale'。

* 返回值：线性回归分析（pandas.Series）
* 函数类型：组合
* 代码示例：

```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
param = [{'attr': 'pvalue'}]
ae=tsf.feature_extraction.feature_calculators.linear_trend(ts, param)
```
* 注释：略
——————————————————————————————————————————————————————————

## longest_strike_above_mean(x) ##

* 译：均值上的最长连续自列长度
* 返回x中大于x平均值的最长连续子序列的长度（int）

* 参数：
  *  $x$   (pandas.Series)计算时序特征的数据对象

* 返回值：最长连续子列长度（int）
* 函数类型：简单
* 代码示例：

```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
param = [{'attr': 'pvalue'}]
ae=tsf.feature_extraction.feature_calculators.longest_strike_above_mean(ts)
```
* 注释：略
——————————————————————————————————————————————————————————

## longest_strike_below_mean(x) ##

* 译：均值下的最长连续自列长度
* 返回x中小于x平均值的最长连续子序列的长度（int）

* 参数：
  *  $x$   (pandas.Series)计算时序特征的数据对象

* 返回值：最长连续子列长度（int）
* 函数类型：简单
* 代码示例：

```python
#!/usr/bin/python3
import tsfresh as tsf
import pandas as pd

ts = pd.Series(x)  #数据x假设已经获取
param = [{'attr': 'pvalue'}]
ae=tsf.feature_extraction.feature_calculators.longest_strike_below_mean(ts)
```
* 注释：略

