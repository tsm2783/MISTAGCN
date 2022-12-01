# Summary

The city area is divided into $M_{lat} \times N_{lng}$ regions, each one is of the same size. In order to calculate $A_{DG}$, distance between regions are calculated based on region centers; to calculate $A_{IG}$, both flow_out and flow_in between regions are culmulated; to calculate $A_{CG}$, the sum of flow_out and flow_in in a region is used instead of the total population, because of missing such data.

# Data description

At time t, the record on region i is $X_{t}^{i}$, with attributes:
flow_out (the number of people moving out the region),
flow_in (the number of people moving into the region),
holiday (whether the day is an official holiday (1) or not (0)),
temperature_min,
temperature_max,
weather,
wind,
air_quality

# Weather coding

```
晴 sun 1
晴~多云 sun-cloudy 2
多云 cloudy 3
晴~阴 sun-overcast 4
多云~阴 cloudy-overcast 5
阴 overcast 6
晴~小雨 sun-light rain 7
多云~小雨 cloudy-light rain 8
阴~小雨 cloudy-light rain 9
小雨 light rain 10
晴~阵雨 sun-shower 11
多云~阵雨 cloudy-shower 12
阴~阵雨 overcast-shower 13
小雨~阵雨 light rain-shower 14
阵雨 shower 15
晴~雷阵雨 sun-thundershower 16
多云~雷阵雨 cloudy-thundershower 17
阴~雷阵雨 overcast-thundershower 18
小雨~雷阵雨 light rain-thundershower 19
阵雨~雷阵雨 shower-thundershower 20
雷阵雨 thundershower 21
晴~中雨 sun-moderate rain 22
多云~中雨 cloudy-moderate rain 23
阴~中雨 overcast-moderate rain 24
小雨~中雨 light rain-moderate rain 25
阵雨~中雨 shower-moderate rain 26
雷阵雨~中雨 thundershower-moderate rain 27
中雨 moderate rain 28
晴~大雨 sun-heavy rain 29
多云~大雨 cloudy-heavy rain 30
阴~大雨 overcast-heavy rain 31
小雨~大雨 light rain-heavy rain 32
阵雨~大雨 shower-heavy rain 33
雷阵雨~大雨 thundershower-heavy rain 34
中雨~大雨 moderate rain-heavy rain 35
大雨 heavy rain 36
晴~暴雨 sun-intense fall 37
多云~暴雨 cloudy-intense fall 38
阴~暴雨 overcast-intense fall 39
小雨~暴雨 light rain-intense fall 40
阵雨~暴雨 shower-intense fall 41
雷阵雨~暴雨 thundershower-intense fall 42
中雨~暴雨 moderate rain-intense fall 43
大雨~暴雨 heavy rain-intense fall 44
暴雨 intense fall 45
晴~大暴雨 sun-rainstorm 46
多云~大暴雨 cloudy-rainstorm 47
阴~大暴雨 overcast-rainstorm 48
小雨~大暴雨 light-rainstorm 49
阵雨~大暴雨 shower-rainstorm 50
雷阵雨~大暴雨 thundershower-rainstorm 51
中雨~大暴雨 moderate rain-rainstorm 52
大雨~大暴雨 heavy rain-rainstorm 53
暴雨~大暴雨 intense fall-rainstorm 54
大暴雨 rainstorm 55
```
