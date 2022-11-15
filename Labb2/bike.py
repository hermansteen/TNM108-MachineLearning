from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
counts = pd.read_csv("FremontBridge.csv", index_col="Date", parse_dates=True)
weather = pd.read_csv("BicycleWeather.csv", index_col="DATE", parse_dates=True)

daily = counts.resample("d").sum()
daily["Total"] = daily.sum(axis=1)
daily = daily[["Total"]]

days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
for i in range(7):
    daily[days[i]] = (daily.index.dayofweek == i).astype(float)

from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()
holidays = cal.holidays("2012", "2016")
daily = daily.join(pd.Series(1, index=holidays, name="holiday"))
daily['holiday'].fillna(0, inplace=True)

def hours_of_daylight(date, axis=23.44, latitude=47.61):
    "Compute the hours of daylight for a given date"
    days = (date - pd.datetime(2000, 12, 21)).days
    m = (1. - np.tan(np.radians(latitude)) * np.tan(np.radians(axis) * np.cos(days * 2 * np.pi / 365.25)))
    return 24. * np.degrees(np.arccos(1 - np.clip(m, 0, 2))) / 180.

daily["daylight_hrs"] = list(map(hours_of_daylight, daily.index))
daily[['daylight_hrs']].plot()
plt.ylim(8, 17)
plt.show()

#temperatures are in tenths of degrees C; convert to C
weather["TMIN"] /= 10
weather["TMAX"] /= 10
#compute daily temperatures
weather["Temp (C)"] = 0.5 * (weather["TMIN"] + weather["TMAX"])
#compute daily precipitation
weather["PRCP"] /= 254 # 1/10mm -> inches
weather["dry day"] = (weather["PRCP"] == 0).astype(int)
daily = daily.join(weather[['PRCP', 'Temp (C)', 'dry day']])

daily['annual'] = (daily.index - daily.index[0]).days / 365.
daily.head()

daily.dropna(axis=0, how='any', inplace=True)
column_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", "holiday", "daylight_hrs", "PRCP", "Temp (C)", "dry day", "annual"]
X = daily[column_names]
y = daily["Total"]
model = LinearRegression(fit_intercept=False)
model.fit(X, y)
daily["predicted"] = model.predict(X)
daily[["Total", "predicted"]].plot(alpha=0.5)
plt.show()

params = pd.Series(model.coef_, index=X.columns)
params

from sklearn.utils import resample
np.random.seed(1)
err = np.std([model.fit(*resample(X, y)).coef_ for i in range(1000)], 0)
print(pd.DataFrame({"effect": params.round(0), "error": err.round(0)}))