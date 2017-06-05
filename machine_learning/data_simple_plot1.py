import matplotlib.pylab as plt
import numpy as np
import pandas as pd
data = pd.read_csv('./data/AAPL.csv')[150:0:-1]
close_price = data.ix[:, 'Adj Close'].tolist()
print(close_price)
plt.plot(close_price)
plt.show()