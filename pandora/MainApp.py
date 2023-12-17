import pandas as pd

from pandora.Config import Config
from pandora.DataLoader import DataLoader
from pandora.Multivariate import Multivariate
from pandora.ResultPlotter import ResultPlotter
from pandora.Runner import Runner
from pandora.Univariate import Univariate


class PandoraApp:
    @staticmethod
    def run_and_plot(dataset, title, cnn, b_lstm, s_lstm, mo_cnn):
        labels = ['CNN', 'U-CNN', 'BLSTM', 'U-BLSTM', 'LSTM', 'U-LSTM', 'MO CNN', 'REAL']

        # 10. run algorithms on univariate dollar
        u_cnn, u_lstm, u_b_lstm, u_last = Runner.run_for_univariate_series_ir(dataset, Config.prediction_col,
                                                                              Config.n_steps,
                                                                              Univariate.epochs)
        data = [cnn, u_cnn, b_lstm, u_b_lstm, s_lstm, u_lstm, mo_cnn, u_last]
        # 13. plot all results of algorithms on car
        ResultPlotter.plot_result(labels, data, title)

    @staticmethod
    def attack():
        # 3. get each dataset according to env
        if Config.colab:
            dfs = DataLoader.read_csv_files_from_drive_in_colab()

            ir_dollar = dfs[Config.dollar_file_name]
            ir_home = dfs[Config.home_file_name]
            ir_oil = dfs[Config.oil_file_name]
            ir_car = dfs[Config.car_file_name]
            ir_gold = dfs[Config.gold_file_name]
        else:
            ir_dollar = pd.read_csv(Config.folder_path + Config.dollar_file_name)
            ir_home = pd.read_csv(Config.folder_path + Config.home_file_name)
            ir_oil = pd.read_csv(Config.folder_path + Config.oil_file_name)
            ir_car = pd.read_csv(Config.folder_path + Config.car_file_name)
            ir_gold = pd.read_csv(Config.folder_path + Config.gold_file_name)

        # 5. preprocess each dataset (sort, date col check, set index, resample)
        ir_dollar = DataLoader.data_preprocessing(ir_dollar, Config.date_col, Config.start_date, Config.end_date,
                                                  format=False)
        ir_home = DataLoader.data_preprocessing(ir_home, Config.date_col, Config.start_date, Config.end_date)
        ir_oil = DataLoader.data_preprocessing(ir_oil, Config.date_col, Config.start_date, Config.end_date)
        ir_car = DataLoader.data_preprocessing(ir_car, Config.date_col, Config.start_date, Config.end_date)
        ir_gold = DataLoader.data_preprocessing(ir_gold, Config.date_col, Config.start_date, Config.end_date)

        # 6. plot each dataset
        if Config.plotting:
            ResultPlotter.plotting(ir_dollar, Config.titles[0])
            ResultPlotter.plotting(ir_home, Config.titles[1])
            ResultPlotter.plotting(ir_oil, Config.titles[2])
            ResultPlotter.plotting(ir_car, Config.titles[3])
            ResultPlotter.plotting(ir_gold, Config.titles[4])

        # 7. array datasets for horizontally stack in multivariate prediction
        datasets = [ir_dollar, ir_home, ir_oil, ir_car, ir_gold]

        # 9. run algorithms on multivariate series
        cnn, b_lstm, s_lstm, mo_cnn, last = Runner.run_for_multivariate_series_ir(datasets, Config.prediction_col,
                                                                                  Config.n_steps,
                                                                                  Multivariate.epochs)
        for i in range(len(datasets)):
            PandoraApp.run_and_plot(datasets[i], Config.titles[i], cnn[i], b_lstm[i], s_lstm[i], mo_cnn[i])
