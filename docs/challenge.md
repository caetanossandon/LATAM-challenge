exploration.ipnyb:

Changes:

- **Reading CSV**: Added error handling parameter when reading data.
- **`get_period_day`**: Used `pd.to_datetime` and `dt.hour` for efficient date handling and period determination.
- **`is_high_season`**: Transitioned to pandas for date range checks.
- **Date Difference**: Vectorized time difference calculation for entire columns.
- **Rate Calculation**: Optimized rate computation using `value_counts`.
- **Seaborn Plots**: Specified x and y arguments in barplot functions.

Based on the results provided in the notebook, model XGBoost with Feature Importance and Balance was chosen. The reason being that while it's more complex and harder to interpret, as far as the challenge is concerned, the end goal is to return to the user whether a flight is expected to be delayed or not, assuming precision, accuracy and reliability against new data are top priorities.

Logical Regression was considered, given that both model performed similarly on the range of data provided. However its strengths lay more on the side of interpretability and, when comparing to XGBoost, it has a better potential for higher accuracy predictions and the ability to capture complex data patterns.

model.py:

trained models are stored in data/models, where it can be accessed when calling .predict without having to call .fit in the same instance.

test_model.py:

file path to data.csv was modified in order to streamline debug and testing, since the original path wasn't working in my machine. Changing it proved to be the fastest solution.

Dockerfile:

created dockerfile

api.py:
