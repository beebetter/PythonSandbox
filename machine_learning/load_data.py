import pandas as pd
data = pd.read_csv('./data/AAPL.csv')[10:0:-1]
col = ['Open','High','Low','Close','Adj Close','Volume']
X = data.as_matrix(col)
y = data.ix[:, 'Adj Close'].tolist()
print(X)
print(y)