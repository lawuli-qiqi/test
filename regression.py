import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from itertools import combinations


global file_path

try:
    file_path = input("请输入CSV文件路径:\n ")
    df = pd.read_csv(file_path)
    df_copy = df.copy()

    dependent_variables = df.columns[df.isnull().any()].tolist()

    pred_data = df.isnull().any(axis=1)

    df.dropna(inplace=True)

    correlation_threshold_high = 0.8
    correlation_threshold_low = 0.5

    high_corr_columns = []
    medium_corr_columns = []

    for dependent_variable in dependent_variables:

        for column in df.columns[:-len(dependent_variables)]:  # 不包括缺值变量列
            correlation = np.corrcoef(df[column], df[dependent_variable], rowvar=False)[0, 1]

            if abs(correlation) > correlation_threshold_high:
                high_corr_columns.append(column)
            elif abs(correlation) > correlation_threshold_low:
                medium_corr_columns.append(column)

        if high_corr_columns:
            X = df[high_corr_columns]
            y = df[dependent_variable]

            if len(X.shape) == 1:
                X = X.values.reshape(-1, 1)

            model = LinearRegression()
            model.fit(X, y)

            predictions = model.predict(X)
            mse = mean_squared_error(y, predictions)
            print(f'找到高相关性自变量{high_corr_columns}，用于缺值变量{dependent_variable}')
            print(f"线性回归均方误差: {mse}\n")

            df_copy.loc[pred_data, dependent_variable] = model.predict(df_copy.loc[pred_data, high_corr_columns])

        elif medium_corr_columns:
            for i in range(1, len(medium_corr_columns) + 1):
                for subset in combinations(medium_corr_columns, i):
                    X = df[list(subset)]
                    y = df[dependent_variable]

                    if len(X.shape) == 1:
                        X = X.values.reshape(-1, 1)

                    model = LinearRegression()
                    model.fit(X, y)

                    predictions = model.predict(X)
                    mse = mean_squared_error(y, predictions)
                    print(f'找到中相关性自变量{medium_corr_columns}，用于缺值变量{dependent_variable}')
                    print(f"线性回归均方误差: {mse}\n")

                    df_copy.loc[pred_data, dependent_variable] = model.predict(
                        df_copy.loc[pred_data, medium_corr_columns])
                    break

        else:
            print(f'未找到明显相关自变量，用于缺值变量{dependent_variable}\n')

    df_copy.to_csv('filled_data.csv', index=False)
    print("已完成回归填充")

except FileNotFoundError:
    print(f"找不到文件：{file_path}")
except pd.errors.EmptyDataError:
    print(f"文件为空：{file_path}")
except pd.errors.ParserError as e:
    print(f"解析文件时发生错误：{e}")
except Exception as e:
    print(f"发生错误：{e}")
