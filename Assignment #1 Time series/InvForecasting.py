import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    data = pd.read_excel('service-data.xlsx', skipinitialspace=True)
    requiredData = data[['Job Card Date', 'INVOICE LINE TEXT']]
    column_name = ['Date','Stock']
    requiredData.columns = column_name
    print(requiredData[-10:-2])

    test = individualStock(requiredData, 'ENGINE OIL')
    # print(test[-10:-1])

def individualStock(data,stockName):
    data.loc[data.Stock == stockName,'Stock'] = 1
    data.loc[data.Stock != 1, 'Stock'] = 0
    data  = data[:-2]

    # Converting date (yyyy-mm-dd to yyyy-mm-01 and yyyy-mm-15)


    return data

if __name__ == "__main__":
    main()