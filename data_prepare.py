from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def run():
    # Load datasets
    # df1 = pd.read_csv('iran_stock/dollar_tjgu_from_2012.csv')
    df2 = pd.read_csv('iran_stock/Iran.Khodro_from_2001.csv')
    df3 = pd.read_csv('iran_stock/Maskan.Invest_from_2014.csv')
    df4 = pd.read_csv('iran_stock/S_Parsian.Oil&Gas_from_2012.csv')
    df5 = pd.read_csv('iran_stock/Lotus.Gold.Com.ETF_from_2017.csv')
    # Load other datasets as needed
    dfs = [df2, df3, df4, df5]  # Add all your datasets to this list
    # print(dfs)
    date_col = "<DTYYYYMMDD>"
    combined_df = pd.concat(dfs, ignore_index=True)
    print(combined_df)

    combined_df = combined_df.sort_values(by=date_col)
    print(combined_df)
    # Assuming 'date' is the column containing date in integer format
    combined_df[date_col] = pd.to_datetime(combined_df[date_col].astype(str), format='%Y%m%d')
    print(combined_df)
    combined_df = combined_df.set_index(date_col)
    combined_df = combined_df.resample('D').ffill()

    start_date = '2017-01-01'
    end_date = '2022-01-01'
    combined_df = combined_df.loc[start_date:end_date]

    print(combined_df)


def run2():
    # Load datasets
    df1 = pd.read_csv('iran_stock/dollar_tjgu_from_2012.csv')
    df2 = pd.read_csv('iran_stock/Iran.Khodro_from_2001.csv')
    df3 = pd.read_csv('iran_stock/Maskan.Invest_from_2014.csv')
    df4 = pd.read_csv('iran_stock/S_Parsian.Oil&Gas_from_2012.csv')
    df5 = pd.read_csv('iran_stock/Lotus.Gold.Com.ETF_from_2017.csv')
    # Load other datasets as needed
    dfs = [df1]  # Add all your datasets to this list
    date_col = "<DTYYYYMMDD>"
    for dataset in dfs:
        dataset = dataset.sort_values(by=date_col)
        print(dataset)

        # Assuming 'date' is the column containing date in integer format
        # dataset[date_col] = pd.to_datetime(dataset[date_col], format='%Y%m%d')
        dataset[date_col] = pd.to_datetime(dataset[date_col], format='mixed')
        # Convert object columns to strings
        object_columns = dataset.select_dtypes(include='object').columns
        dataset[object_columns] = dataset[object_columns].astype(str)
        print(dataset.dtypes)
        # Identify and exclude object columns
        non_object_columns = dataset.select_dtypes(exclude='object').columns

        # Create a new DataFrame without object columns
        dataset = dataset[non_object_columns]

        print(dataset)
        dataset = dataset.set_index(date_col)
        print(dataset.dtypes)

        dataset = dataset.resample('W-Sat').mean().ffill()
        print(dataset)
        start_date = '2017-01-01'
        # end_date = datetime.now().date().isoformat()
        end_date = '2022-12-03'
        dataset = dataset.loc[start_date:end_date]
        print(dataset)
        plotting(dataset)




def data_preprocessing(dataset, date_col = "<DTYYYYMMDD>"):
    dataset = dataset.sort_values(by=date_col)
    print(dataset)

    # Assuming 'date' is the column containing date in integer format
    # dataset[date_col] = pd.to_datetime(dataset[date_col], format='%Y%m%d')
    dataset[date_col] = pd.to_datetime(dataset[date_col], format='mixed')
    # Convert object columns to strings
    object_columns = dataset.select_dtypes(include='object').columns
    dataset[object_columns] = dataset[object_columns].astype(str)
    print(dataset.dtypes)
    # Identify and exclude object columns
    non_object_columns = dataset.select_dtypes(exclude='object').columns

    # Create a new DataFrame without object columns
    dataset = dataset[non_object_columns]

    print(dataset)
    dataset = dataset.set_index(date_col)
    print(dataset.dtypes)

    dataset = dataset.resample('W-Sat').mean().ffill()
    print(dataset)
    start_date = '2017-01-01'
    # end_date = datetime.now().date().isoformat()
    end_date = '2022-12-03'
    dataset = dataset.loc[start_date:end_date]
    print(dataset)
    return dataset

def plotting(dataset):
    # Plotting
    plt.figure(figsize=(12, 8))

    # Plotting open, high, low, close prices
    plt.plot(dataset.index, dataset['<OPEN>'], label='Open', linestyle='-', color='blue')
    plt.plot(dataset.index, dataset['<HIGH>'], label='High', linestyle='-', color='green')
    plt.plot(dataset.index, dataset['<LOW>'], label='Low', linestyle='-', color='red')
    plt.plot(dataset.index, dataset['<CLOSE>'], label='Close', linestyle='-', color='purple')

    # Adding grid
    plt.grid(True, linestyle='--', alpha=0.6)

    # Adding labels and title
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Prices Over Time')

    # Formatting x-axis with a readable date format
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # plt.gca().xaxis.set_major_locator(mdates.DayLocator())

    # Rotating x-axis labels for better readability
    plt.gcf().autofmt_xdate()

    # Adding legend with a shadow
    plt.legend(shadow=True)

    # Adding a background color to the plot
    plt.gca().set_facecolor('#f4f4f4')

    # Adding horizontal lines for reference
    plt.axhline(y=100, color='gray', linestyle='--', linewidth=1, label='Reference Line')

    # Display the plot
    plt.show()

if __name__ == '__main__':
    run2()
