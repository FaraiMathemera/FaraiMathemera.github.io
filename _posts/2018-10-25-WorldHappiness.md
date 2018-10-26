---
title: "Data Analysis: World Happiness Report (Fun)"
date: 2018-10-24
tags: [data analysis, data science, kaggle, Game of Thrones]
header:
  image: "/images/worldhappiness/header.jpg"
excerpt: "data analysis, Data Science, Game of Thrones"
mathjax: "true"
---
# World Happiness

Some really interesting features are contained in these data sets.
It would be interesting to see how these features relate to our day to day lives.


```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sn # data visualization
from matplotlib import pyplot as plt
#import plotly.plotly as py
#import plotly.tools as tls


sn.set(color_codes = True, style="white")
import matplotlib.pyplot as ml # data visualisation as well
import warnings
warnings.filterwarnings("ignore")
stats17 = pd.read_csv("2017.csv", sep=",", header=0)
stats16 = pd.read_csv("2016.csv", sep=",", header=0)
stats15 = pd.read_csv("2015.csv", sep=",", header=0)
```

## Exploration


```python
stats16.describe()
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
      <th>Happiness.Rank</th>
      <th>Happiness.Score</th>
      <th>Whisker.high</th>
      <th>Whisker.low</th>
      <th>Economy..GDP.per.Capita.</th>
      <th>Family</th>
      <th>Health..Life.Expectancy.</th>
      <th>Freedom</th>
      <th>Generosity</th>
      <th>Trust..Government.Corruption.</th>
      <th>Dystopia.Residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>155.000000</td>
      <td>155.000000</td>
      <td>155.000000</td>
      <td>155.000000</td>
      <td>155.000000</td>
      <td>155.000000</td>
      <td>155.000000</td>
      <td>155.000000</td>
      <td>155.000000</td>
      <td>155.000000</td>
      <td>155.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>78.000000</td>
      <td>5.354019</td>
      <td>5.452326</td>
      <td>5.255713</td>
      <td>0.984718</td>
      <td>1.188898</td>
      <td>0.551341</td>
      <td>0.408786</td>
      <td>0.246883</td>
      <td>0.123120</td>
      <td>1.850238</td>
    </tr>
    <tr>
      <th>std</th>
      <td>44.888751</td>
      <td>1.131230</td>
      <td>1.118542</td>
      <td>1.145030</td>
      <td>0.420793</td>
      <td>0.287263</td>
      <td>0.237073</td>
      <td>0.149997</td>
      <td>0.134780</td>
      <td>0.101661</td>
      <td>0.500028</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>2.693000</td>
      <td>2.864884</td>
      <td>2.521116</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.377914</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>39.500000</td>
      <td>4.505500</td>
      <td>4.608172</td>
      <td>4.374955</td>
      <td>0.663371</td>
      <td>1.042635</td>
      <td>0.369866</td>
      <td>0.303677</td>
      <td>0.154106</td>
      <td>0.057271</td>
      <td>1.591291</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>78.000000</td>
      <td>5.279000</td>
      <td>5.370032</td>
      <td>5.193152</td>
      <td>1.064578</td>
      <td>1.253918</td>
      <td>0.606042</td>
      <td>0.437454</td>
      <td>0.231538</td>
      <td>0.089848</td>
      <td>1.832910</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>116.500000</td>
      <td>6.101500</td>
      <td>6.194600</td>
      <td>6.006527</td>
      <td>1.318027</td>
      <td>1.414316</td>
      <td>0.723008</td>
      <td>0.516561</td>
      <td>0.323762</td>
      <td>0.153296</td>
      <td>2.144654</td>
    </tr>
    <tr>
      <th>max</th>
      <td>155.000000</td>
      <td>7.537000</td>
      <td>7.622030</td>
      <td>7.479556</td>
      <td>1.870766</td>
      <td>1.610574</td>
      <td>0.949492</td>
      <td>0.658249</td>
      <td>0.838075</td>
      <td>0.464308</td>
      <td>3.117485</td>
    </tr>
  </tbody>
</table>
</div>




```python
stats16.head()
```




    Index(['Country', 'Region', 'Happiness Rank', 'Happiness Score',
           'Standard Error', 'Economy (GDP per Capita)', 'Family',
           'Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)',
           'Generosity', 'Dystopia Residual'],
          dtype='object')




```python
stats16.isna().sum()
```




    Country                          0
    Region                           0
    Happiness Rank                   0
    Happiness Score                  0
    Lower Confidence Interval        0
    Upper Confidence Interval        0
    Economy (GDP per Capita)         0
    Family                           0
    Health (Life Expectancy)         0
    Freedom                          0
    Trust (Government Corruption)    0
    Generosity                       0
    Dystopia Residual                0
    dtype: int64



We see what our data set looks  like.
We also conducted a summary to get a rough idea if we are missing anything, which we can see that we dont.

After having a look at the head we can see the top countries are in Europe. we need to see if this hunch carries.

### Region


```python
Lets see how many countries we have in each region
```


```python
stats16.groupby('Region').agg('count')[['Happiness Rank']].plot(kind='bar', figsize=(25, 7), stacked=True, color=['b', 'tab:pink']);
```


![alt]({{ site.url }}{{ site.baseurl }}/images/worldhappiness/output_11_0.png)


We can see that Africa and Central and Eastern Europe have the highest numder of countries.


```python
ml.figure(figsize=(15,10)) 
sn.stripplot(x="Region", y="Happiness Rank", data=stats16, jitter=True)
ml.xticks(rotation=-45)
```




    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), <a list of 10 Text xticklabel objects>)




![alt]({{ site.url }}{{ site.baseurl }}/images/worldhappiness/output_13_1.png)


We can see that Africa has the highest mean, with Australia/New Zealand and North America having the lowest.
We must remember that Australia/New Zealand and North America have 2 countries each and that with these 2 regions the means do not really mean much.

We need to see if our data set has any features that correlate to each other.


```python
ml.figure(figsize=(20,10)) 
sn.heatmap(stats16.corr(), annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0xdc35a58>




![alt]({{ site.url }}{{ site.baseurl }}/images/worldhappiness/output_16_1.png)


There is a high correlation between Economy and Happiness score. Life Expectancy and Economy.
There is also a moderate correlation between Freedom and government trust.


```python
sn.pairplot(stats16[['Happiness Score','Economy (GDP per Capita)','Family','Health (Life Expectancy)']])
```




    <seaborn.axisgrid.PairGrid at 0xdc66dd8>




![alt]({{ site.url }}{{ site.baseurl }}/images/worldhappiness/output_18_1.png)

