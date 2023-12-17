class Config:
    colab = True
    plotting = False
    date_col = "<DTYYYYMMDD>"
    start_date = '2017-06-10'
    end_date = '2022-12-03'
    epochs_for_multivariate_series = 100
    epochs_for_univariate_series = 100
    n_steps = 3
    prediction_col = '<CLOSE>'
    folder_path = '../iran_stock/'
    # 1. set csv dataset file names
    dollar_file_name = 'dollar_tjgu_from_2012.csv'
    car_file_name = 'Iran.Khodro_from_2001.csv'
    oil_file_name = 'S_Parsian.Oil&Gas_from_2012.csv'
    home_file_name = 'Maskan.Invest_from_2014.csv'
    gold_file_name = 'Lotus.Gold.Com.ETF_from_2017.csv'

    titles = ['Dollar', 'Home', 'Oil', 'Car', 'Gold']