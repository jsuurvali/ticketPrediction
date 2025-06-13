import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import matplotlib.pyplot as plt
from prophet import Prophet
import xgboost as xgb

class RedemptionModel:

    def __init__(self, X, target_col):
        '''
        Args:
        X (pandas.DataFrame): Dataset of predictors, output from load_data()
        target_col (str): column name for target variable
        '''
        self._predictions = {}
        self.X = X
        self.target_col = target_col
        self.results = {} # dict of dicts with model results

    def score(self, truth, preds):
        # Score our predictions - modify this method as you like
        return MAPE(truth, preds)


    def run_models(self, n_splits=4, test_size=365):
        '''Run the models and store results for cross validated splits in
        self.results.
        '''
        # Time series split
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        cnt = 0 # keep track of splits
        for train, test in tscv.split(self.X):
            X_train = self.X.iloc[train]
            X_test = self.X.iloc[test]
            # Base model - please leave this here
            preds = self._base_model(X_train, X_test)
            if 'Base' not in self.results:
                self.results['Base'] = {}
            self.results['Base'][cnt] = self.score(X_test[self.target_col],
                                preds)
            # self.plot(preds, 'Base')
            # Other models...
            # self._my-new-model(train, test) << Add your model(s) here
            
            prophet_preds = self.prophet_base(X_train, X_test)
            if 'Prophet' not in self.results:
                self.results['Prophet'] = {}
            self.results['Prophet'][cnt] = self.score(X_test[self.target_col],
                                prophet_preds) 
            # self.plot(prophet_preds, 'Prophet')

            tweedy_preds = self.xgb_tweedy(X_train, X_test)
            if 'XGB_tweedy' not in self.results:
                self.results['XGB_tweedy'] = {}
            self.results['XGB_tweedy'][cnt] = self.score(X_test[self.target_col],
                                tweedy_preds) 
            # self.plot(tweedy_preds, 'XGB_Tweedy')

            log_preds = self.xgb_lognormal(X_train, X_test)
            if 'XGB_lognormal' not in self.results:
                self.results['XGB_lognormal'] = {}
            self.results['XGB_lognormal'][cnt] = self.score(X_test[self.target_col],
                                log_preds) 
            
            self.plot2(preds, prophet_preds, tweedy_preds, "Base", "Prophet", "XGB_tweedy")
            cnt += 1


    def _base_model(self, train, test):
        '''
        Our base, too-simple model.
        Your model needs to take the training and test datasets (dataframes)
        and output a prediction based on the test data.

        Please leave this method as-is.

        '''
        res = sm.tsa.seasonal_decompose(train[self.target_col],
                                        period=365)
        res_clip = res.seasonal.apply(lambda x: max(0,x))
        res_clip.index = res_clip.index.dayofyear
        res_clip = res_clip.groupby(res_clip.index).mean()
        res_dict = res_clip.to_dict()
        return pd.Series(index = test.index, 
                         data = map(lambda x: res_dict[x], test.index.dayofyear))


    def is_cold(self, timestamps, first_warm_month = 5, last_warm_month = 10):
        date = pd.to_datetime(timestamps)
        return (date.month < first_warm_month or date.month > last_warm_month)

    
    def data_to_prophet(self, data_in):
        df_tmp = data_in.reset_index()
        df_tmp = df_tmp[['Timestamp', self.target_col]]
        df_tmp = df_tmp.rename(
            mapper = {'Timestamp':'ds',
                      self.target_col:'y'},
            axis = 1)
        df_tmp['not_winter'] = ~df_tmp['ds'].apply(self.is_cold)
        df_tmp['winter'] = df_tmp['ds'].apply(self.is_cold)
        return df_tmp

    
    def prophet_base(self, train, test):
        '''
        This is based on the Prophet library.
        It consider multiple types of seasonal variation:
        
        '''

        event_df = pd.DataFrame({
            'holiday': 'worst COVID',
            'ds': pd.to_datetime(['2020-03-17']),
            'ds_upper': pd.to_datetime(['2022-03-16'])})  
        
        df_all = self.data_to_prophet(self.X)
        df_train = self.data_to_prophet(train)
        df_test = self.data_to_prophet(test)

        
        m0 = Prophet(holidays = event_df, weekly_seasonality=False)
        m0.add_seasonality(name='not_winter', period=6, fourier_order=4,
                          condition_name='not_winter')
        m0.add_seasonality(name='winter', period=6, fourier_order=4,
                          condition_name='winter')
        m0 = m0.fit(df_all)
        future0 = m0.make_future_dataframe(periods=365)
        future0['not_winter'] = ~future0['ds'].apply(self.is_cold)
        future0['winter'] = future0['ds'].apply(self.is_cold)
        forecast0 = m0.predict(future0)
        forecast0['yhat'] = forecast0['yhat'].apply(lambda x: max(0,x))
        fig0 = m0.plot(forecast0)

        m = Prophet(holidays = event_df, weekly_seasonality=False)
        m.add_seasonality(name='not_winter', period=6, fourier_order=4,
                          condition_name='not_winter')
        m.add_seasonality(name='winter', period=6, fourier_order=4,
                          condition_name='winter')
        m = m.fit(df_train)
        future = df_test
        future['not_winter'] = ~future['ds'].apply(self.is_cold)
        future['winter'] = future['ds'].apply(self.is_cold)
        forecast = m.predict(future)
        forecast['yhat'] = forecast['yhat'].apply(lambda x: max(0,x))
        # fig = m.plot(forecast)
        
        forecast.set_index('ds', inplace=True)
        forecast = forecast.reindex(test.index)
        prophet_preds_out = pd.Series(index=test.index,
                                      data=forecast['yhat'])
        return prophet_preds_out



    def prepare_data_for_xgb(self, input_df):
        # Converting all cyclical features in data to actual cycles with sinus / cosinus transformation
        cycX = input_df
        cycX['month_sin'] = np.sin(2 * np.pi * cycX['month'] / 12)
        cycX['month_cos'] = np.cos(2 * np.pi * cycX['month'] / 12)
        
        cycX['weekday_sin'] = np.sin(2 * np.pi * cycX['weekday'] / 7)
        cycX['weekday_cos'] = np.cos(2 * np.pi * cycX['weekday'] / 7)
        
        # Dropping columns that are no longer needed
        cycX = cycX.drop(['_id', 'month', 'weekday', 'day of year'],
                     axis=1)

        # adding two new columns: covid yes/no and summer yes/no
        event_df = pd.DataFrame({
        'holiday': 'worst COVID',
        'ds': pd.to_datetime(['2020-03-17']),
        'ds_upper': pd.to_datetime(['2022-03-16'])}) 

        cycX = cycX.reset_index()
        cycX['is_covid'] = (cycX['Timestamp'] > pd.to_datetime('2020-03-17')) & (cycX['Timestamp'] < pd.to_datetime('2022-03-16'))
        date_tmp = pd.to_datetime(cycX['Timestamp'])
        cycX['is_cold'] = (date_tmp.dt.month < 5) | (date_tmp.dt.month > 10)
        cycX = cycX.set_index('Timestamp')
        cycX['is_cold'] = cycX['is_cold'].astype('int64')
        cycX['is_covid'] = cycX['is_covid'].astype('int64')
        return cycX

    
    def xgb_tweedy(self, train, test):
        # read in data
        cycX = self.prepare_data_for_xgb(self.X)
        mtrain = cycX.loc[cycX.index.intersection(train.index)]
        mtest = cycX.loc[cycX.index.intersection(test.index)]
        
        X_mtrain = mtrain[['month_sin', 'month_cos', 'weekday_sin', 'weekday_cos', 'is_cold', 'is_covid']]
        X_mtest = mtest[['month_sin', 'month_cos', 'weekday_sin', 'weekday_cos', 'is_cold', 'is_covid']]
        y_mtrain = mtrain[['Redemption Count']]
        y_mtest = mtest[['Redemption Count']]
    
        # Defining a parameter grid
        param_grid = {'n_estimators': [60, 80, 100, 120, 140],
                      'max_depth': [2, 3, 4, 5, 6],
                      'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
                      'colsample_bytree': [0.2, 0.4, 0.6, 0.8, 1.0]}

        # Model and random search. Assuming the output to have a tweedie distribution (informative zeroes and a strong tail)
        # Random state is not really random the way it is used here, for reproducibility the method uses a specific value "17"
        
        model = xgb.XGBRegressor(objective='reg:tweedie', random_state=17, n_jobs=-1)
        random_search = RandomizedSearchCV(estimator=model,
                                           param_distributions=param_grid,
                                           n_iter=100,
                                           scoring='neg_mean_squared_error',
                                           cv=3,
                                           verbose=0,
                                           n_jobs=-1,
                                           random_state=17)
        random_search.fit(X_mtrain, y_mtrain)
        best_params = random_search.best_params_
        model = xgb.XGBRegressor(
            objective='reg:tweedie',
            tweedie_variance_power = 1.2,
            random_state=17,
            **best_params)

        model.fit(X_mtrain, y_mtrain)
        y_mpred = model.predict(X_mtest)
        res = pd.DataFrame({'Date' : X_mtest.index, 'Predicted' : y_mpred})
        res.sort_values(by = 'Date', inplace=True)
        res.set_index('Date', inplace=True)
        res = res.reindex(mtest.index)
        res = pd.Series(index=mtest.index, data=res['Predicted'])
        return res


    def xgb_lognormal(self, train, test):
        # read in data
        cycX = self.prepare_data_for_xgb(self.X)

        # Adding 0.01 to all outcomes values so that logarithms could be used on those
        cycX['Redemption Count'] = np.log(cycX['Redemption Count'] + 0.01)
        cycX['Sales Count'] = np.log(cycX['Sales Count'] + 0.01)

        mtrain = cycX.loc[cycX.index.intersection(train.index)]
        mtest = cycX.loc[cycX.index.intersection(test.index)]
        
        X_mtrain = mtrain[['month_sin', 'month_cos', 'weekday_sin', 'weekday_cos', 'is_cold', 'is_covid']]
        X_mtest = mtest[['month_sin', 'month_cos', 'weekday_sin', 'weekday_cos', 'is_cold', 'is_covid']]
        y_mtrain = mtrain[['Redemption Count']]
        y_mtest = mtest[['Redemption Count']]
    
        # Defining a parameter grid
        param_grid = {'n_estimators': [60, 80, 100, 120, 140],
                      'max_depth': [2, 3, 4, 5, 6],
                      'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
                      'colsample_bytree': [0.2, 0.4, 0.6, 0.8, 1.0]}

        # Model and random search. Assuming the output to have a tweedie distribution (informative zeroes and a strong tail)
        # Random state is not really random the way it is used here, for reproducibility the method uses a specific value "17"
        
        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=17, n_jobs=-1)
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=100,
            scoring='neg_mean_squared_error',
            cv=3,
            verbose=0,
            n_jobs=-1,
            random_state=17
        )
        random_search.fit(X_mtrain, y_mtrain)
        best_params = random_search.best_params_
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=17,
            **best_params)

        model.fit(X_mtrain, y_mtrain)
        y_mpred = model.predict(X_mtest)
        res = pd.DataFrame({'Date' : X_mtest.index, 'Predicted' : y_mpred})
        res.sort_values(by = 'Date', inplace=True)
        res.set_index('Date', inplace=True)
        res = res.reindex(mtest.index)
        res = pd.Series(index=mtest.index, data=res['Predicted'])
        return res
    
    
    def plot(self, preds, label):
        # plot out the forecasts
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.scatter(self.X.index, self.X[self.target_col], s=0.4, color='grey',
            label='Observed')
        ax.plot(preds, label = label, color='red', alpha = 0.5)
        plt.legend()

    def plot2(self, preds, preds2, preds3, label1, label2, label3):
        # plot out the forecasts
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.scatter(self.X.index, self.X[self.target_col], s=0.4, color='grey',
            label='Observed')
        ax.plot(preds, label = label1, color='red', alpha = 0.5)
        ax.plot(preds2, label = label2, color='blue', alpha = 0.5)
        ax.plot(preds3, label = label3, color='yellow', alpha = 0.5)
        plt.legend()
