import numpy as np
import pandas as pd

class DatasetReader:
    defaultDataset = 0

    def __init__(self):
        self.dataset = self.defaultDataset
        self.options = {
            0: self._breast_dataset,
            1: self._blod_transfusion_dataset,
            2: self._wine_quality_red_dataset,
            3: self._backnote_dataset,
            4: self._shill_bidding_dataset,
            5: self._spambase_dataset,
            6: self._balance_scale_dataset,
            7: self._seoul_bike_dataset,
            8: self._customer_churn_database,
            9: self._facebook_live_database
        }

    def _read_csv(self, filepath, columns, labels_column=None, drop_columns=None, sep=","):
        data = pd.read_csv(filepath, names=columns, sep=sep)
        print(data.shape)
        labels = np.array(data[labels_column]) if labels_column else None
        data = np.array(data.drop(columns=drop_columns)) if drop_columns else data

        return data, labels

    def _breast_dataset(self):
        filepath = 'datasets/breast_cancer/breast-cancer-wisconsin.data'
        columns = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size',
                   'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size',
                   'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
        drop_columns = ['Sample code number', 'Bare Nuclei', 'Class']

        return self._read_csv(filepath, columns, "Class", drop_columns)

    def _blod_transfusion_dataset(self):
        filepath = 'datasets/blood+transfusion+service+center/transfusion.data'
        columns = ['Recency', 'Frequency', 'Monetary', 'Time', 'Donated']
        drop_columns = ['Donated']

        return self._read_csv(filepath, columns, "Donated", drop_columns)

    def _wine_quality_red_dataset(self):
        filepath_red = 'datasets/wine+quality/winequality-red.csv'
        filepath_white = 'datasets/wine+quality/winequality-white.csv'
        columns = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']
        
        data_red, l = self._read_csv(filepath_red, columns, sep=";")
        data_red['wine type'] = 0

        data_white, _ = self._read_csv(filepath_white, columns, sep=";")
        data_white['wine type'] = 1

        data = pd.concat([data_red, data_white], axis=0)
        labels = np.array(data['wine type'])
        return np.array(data.drop(columns=['wine type'])), labels

    def _backnote_dataset(self):
        filepath = 'datasets/banknote+authentication/data_banknote_authentication.data'
        columns = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
        drop_columns = ['class']

        return self._read_csv(filepath, columns, "class", drop_columns)

    def _shill_bidding_dataset(self):
        filepath = 'datasets/shill+bidding+dataset/Shill Bidding Dataset.csv'
        columns = ['Record_ID','Auction_ID','Bidder_ID','Bidder_Tendency','Bidding_Ratio','Successive_Outbidding','Last_Bidding','Auction_Bids','Starting_Price_Average','Early_Bidding','Winning_Ratio','Auction_Duration','Class']
        drop_columns = ['Record_ID','Auction_ID','Bidder_ID','Class']

        return self._read_csv(filepath, columns, "Class", drop_columns)
    
    def _spambase_dataset(self):
        filepath = 'datasets/spambase/spambase.data'
        columns = columns = ['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our',
           'word_freq_over', 'word_freq_remove', 'word_freq_internet', 'word_freq_order', 'word_freq_mail',
           'word_freq_receive', 'word_freq_will', 'word_freq_people', 'word_freq_report', 'word_freq_addresses',
           'word_freq_free', 'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit',
           'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money', 'word_freq_hp',
           'word_freq_hpl', 'word_freq_george', 'word_freq_650', 'word_freq_lab', 'word_freq_labs',
           'word_freq_telnet', 'word_freq_857', 'word_freq_data', 'word_freq_415', 'word_freq_85',
           'word_freq_technology', 'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct',
           'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project', 'word_freq_re',
           'word_freq_edu', 'word_freq_table', 'word_freq_conference', 'char_freq_;', 'char_freq_(',
           'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#', 'capital_run_length_average',
           'capital_run_length_longest', 'capital_run_length_total', 'class']

        drop_columns = ['class']

        return self._read_csv(filepath, columns, "class", drop_columns)
    
    def _balance_scale_dataset(self):
        filepath = 'datasets/balance+scale/balance-scale.data'
        columns = ['Class', 'Left-Weight', 'Left-Distance', 'Right-Weight', 'Right-Distance']
        drop_columns = ['Class']

        return self._read_csv(filepath, columns, "Class", drop_columns)

    def _seoul_bike_dataset(self):
        filepath = 'datasets/seoul+bike+sharing+demand/SeoulBikeData.csv'
        columns = ['Date', 'Rented Bike count', 'Hour', 'Temperature', 'Humidity', 'Windspeed', 'Visibility', 
           'Dew point temperature', 'Solar radiation', 'Rainfall', 'Snowfall', 'Seasons', 'Holiday', 'Functional Day']


        drop_columns = ['Date', 'Seasons', 'Holiday', 'Functional Day']
        return self._read_csv(filepath, columns, "Seasons", drop_columns)
        
    def _customer_churn_database(self):
        filepath = 'datasets/iranian+churn+dataset/Customer Churn.csv'
        columns = ['Call Failure', 'Complains', 'Subscription Length', 'Charge Amount', 'Seconds of Use', 'Frequency of use',
           'Frequency of SMS', 'Distinct Called Numbers', 'Age Group', 'Tariff Plan', 'Status', 'Age', 'Customer Value', 'Churn']

        drop_columns = ['Churn']
        return self._read_csv(filepath, columns, "Churn", drop_columns)

    def _facebook_live_database(self):
        filepath = 'datasets/facebook+live+sellers+in+thailand/Live_20210128.csv'
        columns = ['status_id', 'status_type', 'status_published', 'num_reactions', 'num_comments', 'num_shares', 'num_likes',
           'num_loves', 'num_wows', 'num_hahas', 'num_sads', 'num_angrys', 'Column1', 'Column2', 'Column3', 'Column4']


        drop_columns = ['status_id', 'status_type', 'status_published', 'Column1', 'Column2', 'Column3', 'Column4']
        return self._read_csv(filepath, columns, "status_type", drop_columns)

    def _default_read_data(self):
        return None, None

    def read_data(self, opt):
        return self.options.get(opt, self._default_read_data)()
