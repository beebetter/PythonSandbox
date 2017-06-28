import datetime
import pandas as pd
import numpy as np
from pandas_datareader import data, wb

start = datetime.datetime(2016, 1, 1)
end = datetime.datetime(2017, 1, 1)

a = data.DataReader("AAPL", "google", start, end)
print (a)
