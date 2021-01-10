# TimeSeriesKaggle

# Baseline Case
In this model, we use last month's sales as the prediction for this month's value (Can be thought as Moving average with window size of 1).
Steps involved include:
1. Aggregating daily data to monthly data.
2. Joining test data and train data to get all possible pairs.
3. Filling null values, clipping monthly sales to [0,20] and extracting only last month sales.