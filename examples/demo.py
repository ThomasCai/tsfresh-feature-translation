# 这个demo参考 
# https://blog.csdn.net/takethevow/article/details/80171440
# https://www.smwenku.com/a/5b7ea8b82b717767c6aafc71/zh-cn

# 1. 首先加载数据
import tsfresh
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, load_robot_execution_failures

download_robot_execution_failures()
timeseries, y = load_robot_execution_failures()

# 2. 看一下数据的形式
print(timeseries.head())
print(y.head())

# 3. 抽取特征
from tsfresh import extract_features
extracted_features = extract_features(timeseries, column_id='id', column_sort='time')
print(extracted_features.head())

# 4. 特征过滤
# 由上一步操作得到的特征中存在空值(NaN)，这些没有意义的值需要去掉，选择有用的特征进行保留。从结果可以看出，数据的维度减少了很多。
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute

impute(extracted_features)
features_filtered = select_features(extracted_features, y)
print(features_filtered.head())

# 5. 特征抽取与过滤同时进行（一步到位，省去多余计算）
from tsfresh import extract_relevant_features

features_filtered_direct = extract_relevant_features(timeseries, y, column_id='id', column_sort='time')
print(features_filtered_direct.head())