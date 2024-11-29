---
title: "머신러닝을 통한 자전거 대여 수요 예측: (1) EDA"
excerpt: "EDA를 잘 하자"

categories:
  - Practices
tags:
  - [EDA, Machine Learning]

permalink: /practices/bike-sharing-demand-1/

toc: true
toc_sticky: true

date: 2024-11-28
last_modified_at: 2024-11-28
---

자전거 대여 시스템 데이터를 이용해 자전거 대여 패턴을 분석하고, 대여 수요를 가장 잘 예측하는 모델을 찾아내고자 한다.

데이터는 Kaggle의 <a href="https://www.kaggle.com/competitions/bike-sharing-demand" target="_blank">Bike Sharing Demand</a> 대회 데이터를 이용했다.

## 데이터 불러오기 및 확인

```python
# 데이터 불러오기

df_train = pd.read_csv('./src/train.csv')
df_test = pd.read_csv('./src/test.csv')
```

```python
df_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10886 entries, 0 to 10885
    Data columns (total 12 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   datetime    10886 non-null  object 
     1   season      10886 non-null  int64  
     2   holiday     10886 non-null  int64  
     3   workingday  10886 non-null  int64  
     4   weather     10886 non-null  int64  
     5   temp        10886 non-null  float64
     6   atemp       10886 non-null  float64
     7   humidity    10886 non-null  int64  
     8   windspeed   10886 non-null  float64
     9   casual      10886 non-null  int64  
     10  registered  10886 non-null  int64  
     11  count       10886 non-null  int64  
    dtypes: float64(3), int64(8), object(1)
    memory usage: 1020.7+ KB
    
```python
df_test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6493 entries, 0 to 6492
    Data columns (total 9 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   datetime    6493 non-null   object 
     1   season      6493 non-null   int64  
     2   holiday     6493 non-null   int64  
     3   workingday  6493 non-null   int64  
     4   weather     6493 non-null   int64  
     5   temp        6493 non-null   float64
     6   atemp       6493 non-null   float64
     7   humidity    6493 non-null   int64  
     8   windspeed   6493 non-null   float64
    dtypes: float64(3), int64(5), object(1)
    memory usage: 456.7+ KB
    
train 데이터와 test 데이터 모두 결측치가 존재하지 않는다.

> `datetime` 변수가 object(string) type으로 되어 있어 이를 datetime type으로 변환하고 파생변수들을 생성하였다.

## datetime 변수 형변환 및 파생변수 생성

```python
# string to datetime

df_train['datetime'] = pd.to_datetime(df_train['datetime'])
df_test['datetime'] = pd.to_datetime(df_test['datetime'])
```

```python
# year, month, day, hour, weekday 컬럼 생성

df_train['year'] = df_train['datetime'].dt.year
df_train['month'] = df_train['datetime'].dt.month
df_train['day'] = df_train['datetime'].dt.day
df_train['hour'] = df_train['datetime'].dt.hour
df_train['weekday'] = df_train['datetime'].dt.day_name()

df_test['year'] = df_test['datetime'].dt.year
df_test['month'] = df_test['datetime'].dt.month
df_test['day'] = df_test['datetime'].dt.day
df_test['hour'] = df_test['datetime'].dt.hour
df_test['weekday'] = df_test['datetime'].dt.day_name()
```

```python
df_train.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>count</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>hour</th>
      <th>weekday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-01 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>81</td>
      <td>0.0</td>
      <td>3</td>
      <td>13</td>
      <td>16</td>
      <td>2011</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>Saturday</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-01 01:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>8</td>
      <td>32</td>
      <td>40</td>
      <td>2011</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Saturday</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-01 02:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>5</td>
      <td>27</td>
      <td>32</td>
      <td>2011</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>Saturday</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-01 03:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>3</td>
      <td>10</td>
      <td>13</td>
      <td>2011</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>Saturday</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-01 04:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2011</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>Saturday</td>
    </tr>
  </tbody>
</table>
</div>

## 로그 변환

```python
# train 데이터 변수들 히스토그램

# 사용할 변수 목록
columns_to_plot = [
    "season", "holiday", "workingday", "weather",
    "temp", "atemp", "humidity", "windspeed",
    "casual", "registered", "count", "year"
]

def plot_selected_histograms(df, columns):
    num_columns = len(columns)
    rows = (num_columns + 3) // 4
    
    plt.figure(figsize=(16, rows * 4))
    
    for i, column in enumerate(columns, 1):
        plt.subplot(rows, 4, i)
        plt.hist(df[column], bins=30, color='blue', alpha=0.7, edgecolor='black')
        plt.title(column)
        plt.tight_layout()
    
    plt.show()

plot_selected_histograms(df_train, columns_to_plot)
```

![](/assets/images/posts_img/practices-bike-sharing-demand-1/001.png)

모델에서 종속변수로 사용될 `casual`, `registered`, `count` 변수의 데이터의 분포가 왼쪽으로 치우쳐져 있다.

> 따라서 로그 변환을 통해 데이터 정규화를 진행하였다.

```python
# 로그 변환

def apply_log_transformation(df, columns):
    for column in columns:
        df[column] = np.log1p(df[column])
    return df

columns_to_log_transform = ["casual", "registered", "count"]
df_train = apply_log_transformation(df_train, columns_to_log_transform)
```

```python
columns_to_plot = ["casual", "registered", "count"]

plot_selected_histograms(df_train, columns_to_plot)
```

![](/assets/images/posts_img/practices-bike-sharing-demand-1/002.png)

## season 변수 재정의

```python
# season별 casual, registered, count 박스플롯

def plot_boxplots_by_season_horizontal(df, variables, group_by_column):
    num_variables = len(variables)
    plt.figure(figsize=(18, 6))
    
    for i, var in enumerate(variables, 1):
        plt.subplot(1, num_variables, i)
        grouped_data = [df[df[group_by_column] == season][var] for season in sorted(df[group_by_column].unique())]
        
        plt.boxplot(
            grouped_data, 
            tick_labels=sorted(df[group_by_column].unique()),
            patch_artist=True,
            boxprops=dict(facecolor='skyblue', color='blue'),
            medianprops=dict(color='red'),
            whiskerprops=dict(color='blue')
        )
        plt.title(f"{var} by {group_by_column}")
        plt.xlabel(group_by_column.capitalize())
        plt.ylabel(var.capitalize())
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

variables_to_plot = ["casual", "registered", "count"]
plot_boxplots_by_season_horizontal(df_train, variables_to_plot, "season")
```

![](/assets/images/posts_img/practices-bike-sharing-demand-1/003.png)

`season` 변수의 1이 봄이고 4가 겨울인데, 봄의 자전거 대여 수가 겨울보다 적게 나타나는 것에 대해 의문이 들었다.

```python
# season이 1(봄)일 때 month

df_train[df_train['season'] == 1]['month'].unique()
```

    array([1, 2, 3], dtype=int32)

```python
# season이 4(겨울)일 때 month

df_train[df_train['season'] == 4]['month'].unique()
```

    array([10, 11, 12], dtype=int32)

확인해보니 1월~3월이 `season` 변수의 1(봄)로 할당되어 있었고, 10월~12월이 `season` 변수의 4(겨울)로 할당되어 있었다.  
보통 12월~2월을 겨울, 3월~5월을 봄으로 보기 때문에 `season` 변수를 재정의할 필요가 있는지 확인하기 위해 월별 온도를 시각화했다.

```python
# 월별 평균 기온 시각화

month_avg_temp = df_train.groupby("month")["temp"].mean()

plt.figure(figsize=(10, 6))
plt.plot(month_avg_temp.index, month_avg_temp.values, marker='o', linestyle='-', color='blue')

plt.title("월별 평균 기온", fontsize=16)
plt.xlabel("월", fontsize=12)
plt.ylabel("기온", fontsize=12)
plt.xticks(month_avg_temp.index)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```

![](/assets/images/posts_img/practices-bike-sharing-demand-1/004.png)

> 예상대로 12월~2월에 연중 온도가 가장 낮게 나타나 `season` 변수를 재정의하였다.

```python
# season 변수 재정의

conditions = [
    df_train['month'].isin([3, 4, 5]),
    df_train['month'].isin([6, 7, 8]),
    df_train['month'].isin([9, 10, 11]),
    df_train['month'].isin([12, 1, 2])
]
choices = [1, 2, 3, 4]

df_train['season'] = np.select(conditions, choices)
```

```python
# season별 casual, registered, count 박스플롯

def plot_boxplots_by_season_horizontal(df, variables, group_by_column):
    num_variables = len(variables)
    plt.figure(figsize=(18, 6))
    
    for i, var in enumerate(variables, 1):
        plt.subplot(1, num_variables, i)
        grouped_data = [df[df[group_by_column] == season][var] for season in sorted(df[group_by_column].unique())]
        
        plt.boxplot(
            grouped_data, 
            tick_labels=sorted(df[group_by_column].unique()),
            patch_artist=True,
            boxprops=dict(facecolor='skyblue', color='blue'),
            medianprops=dict(color='red'),
            whiskerprops=dict(color='blue')
        )
        plt.title(f"{var} by {group_by_column}")
        plt.xlabel(group_by_column.capitalize())
        plt.ylabel(var.capitalize())
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

variables_to_plot = ["casual", "registered", "count"]
plot_boxplots_by_season_horizontal(df_train, variables_to_plot, "season")
```

![](/assets/images/posts_img/practices-bike-sharing-demand-1/005.png)

재정의 후 `casual`, `registered`, `count` 모두 겨울에 가장 적게 나타난다.

## 데이터 시각화

실제 값의 뚜렷한 추이를 보기 위해 시각화할 때는 로그 변환 전의 데이터를 사용하였다.

```python
# 시각화를 위해 스케일링(로그 변환)되지 않은 train 데이터 준비

df_train_unscaled = pd.read_csv('./src/train.csv')

# string to datetime
df_train_unscaled['datetime'] = pd.to_datetime(df_train['datetime'])

# year, month, day, hour 컬럼 생성
df_train_unscaled['year'] = df_train_unscaled['datetime'].dt.year
df_train_unscaled['month'] = df_train_unscaled['datetime'].dt.month
df_train_unscaled['day'] = df_train_unscaled['datetime'].dt.day
df_train_unscaled['hour'] = df_train_unscaled['datetime'].dt.hour
df_train_unscaled['weekday'] = df_train_unscaled['datetime'].dt.day_name()

# season 변수 재정의
conditions = [
    df_train_unscaled['month'].isin([3, 4, 5]),
    df_train_unscaled['month'].isin([6, 7, 8]),
    df_train_unscaled['month'].isin([9, 10, 11]),
    df_train_unscaled['month'].isin([12, 1, 2])
]
choices = [1, 2, 3, 4]

df_train_unscaled['season'] = np.select(conditions, choices)
```

```python
# 월별 평균 대여 수

monthly_avg = df_train_unscaled.groupby('month')['count'].mean()

plt.figure(figsize=(10, 5))
monthly_avg.plot(kind='bar', color='skyblue')
plt.title('월별 평균 대여 수')
plt.ylabel('평균 대여 수')
plt.xlabel('월')
plt.xticks(ticks=range(12), labels=range(1, 13), rotation=0)
plt.tight_layout()
plt.show()
```

![](/assets/images/posts_img/practices-bike-sharing-demand-1/006.png)

> 여름철에 자전거 대여 수가 증가하고 겨울철에 자전거 대여 수가 감소한다.

```python
# 계절에 따른 시간대별 평균 대여 수

season_hour_avg = df_train_unscaled.groupby(['season', 'hour'])['count'].mean().reset_index()

season_map = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
season_hour_avg['season'] = season_hour_avg['season'].map(season_map)

plt.figure(figsize=(10, 5))
sns.lineplot(data=season_hour_avg, x='hour', y='count', hue='season', marker='o')
plt.title('계절에 따른 시간대별 평균 대여 수')
plt.ylabel('평균 대여 수')
plt.xlabel('시간대(하루)')
plt.xticks(range(0, 24))
plt.grid(axis='y')
plt.tight_layout()
plt.show()
```

![](/assets/images/posts_img/practices-bike-sharing-demand-1/007.png)

> 여름, 가을, 봄, 겨울 순으로 자전거 대여 수가 많다.  
출퇴근 시간대에 자전거 대여 수가 급증한다.

```python
# 요일에 따른 시간대별 평균 대여 수

weekday_hour_avg = df_train_unscaled.groupby(['weekday', 'hour'])['count'].mean().reset_index()

weekday_hour_avg['weekday'] = pd.Categorical(
    weekday_hour_avg['weekday'], 
    categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    ordered=True
)

plt.figure(figsize=(10, 5))
sns.lineplot(data=weekday_hour_avg, x='hour', y='count', hue='weekday', marker='o')
plt.title('요일에 따른 시간대별 평균 대여 수')
plt.ylabel('평균 대여 수')
plt.xlabel('시간대(하루)')
plt.xticks(range(0, 24))
plt.grid(axis='y')
plt.tight_layout()
plt.show()
```

![](/assets/images/posts_img/practices-bike-sharing-demand-1/008.png)

> 평일에는 출퇴근 시간대에 자전거 대여 수가 급증하고, 주말에는 오후 시간대에 자전거 대여 수가 많게 유지된다.  
평일에는 실용적으로, 주말에는 여가용으로 대여가 많이 발생한다고 볼 수 있다.

```python
# 사용자 타입에 따른 시간대별 평균 대여 수

user_type_hour_avg = df_train_unscaled.groupby(['hour']).agg({'casual': 'mean', 'registered': 'mean'}).reset_index()

plt.figure(figsize=(10, 5))
sns.lineplot(data=user_type_hour_avg, x='hour', y='casual', label='Casual', marker='o')
sns.lineplot(data=user_type_hour_avg, x='hour', y='registered', label='Registered', marker='o')
plt.title('사용자 타입에 따른 시간대별 평균 대여 수')
plt.ylabel('평균 대여 수')
plt.xlabel('시간대(하루)')
plt.xticks(range(0, 24))
plt.legend(title='User Type')
plt.grid(axis='y')
plt.tight_layout()
plt.show()
```

![](/assets/images/posts_img/practices-bike-sharing-demand-1/009.png)

> 주로 등록 사용자는 출퇴근 용도로, 미등록 사용자는 여가 용도로 자전거를 대여함을 알 수 있다.  
등록 사용자의 대여 수가 미등록 사용자의 대여 수보다 많은 것도 볼 수 있다.
