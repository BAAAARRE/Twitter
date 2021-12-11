import pandas as pd
import numpy as np
import mysql.connector
import streamlit as st
import plotly.graph_objects as go


class LoadData:
    @staticmethod
    def init_connection():
        return mysql.connector.connect(**st.secrets["mysql"])

    @staticmethod
    def keywords(conn, list_key_words):
        list_key_words = list_key_words.replace(',', '|')
        query = """SELECT
            date_format(created_at, '%Y-%m-%d %H:%i') as date,
            count(id) as nb_tweet
            FROM tweet
            WHERE tweet REGEXP '{}' 
            GROUP BY date
            ORDER BY date
            """.format(list_key_words)
        df = pd.read_sql_query(query, conn)
        return df

    @staticmethod
    def prepare_keywords(df, tz, granu):
        df.date = pd.to_datetime(df.date)
        df.date = df.date.dt.tz_localize('UTC').dt.tz_convert(tz).dt.tz_localize(None)
        df = df.resample(granu, on='date').agg({'nb_tweet': 'sum'})
        df = df.reset_index()
        df.columns = ['date', 'nb_tweets']
        return df

    @staticmethod
    def price(conn):
        query = "SELECT crypto, dateTime, open FROM binance"
        df = pd.read_sql_query(query, conn)
        return df

    @staticmethod
    def prepare_price(df, tz, granu):
        df.dateTime = df.dateTime.dt.tz_localize('UTC').dt.tz_convert(tz).dt.tz_localize(None)
        df = df.resample(granu, on='dateTime').agg({'open': 'first'})
        df = df.reset_index()
        df.columns = ['date', 'price']
        return df


class Calculation:
    @staticmethod
    def ema(df, col_name, window):
        df[col_name + '_EMA_' + str(window)] = df[col_name].ewm(span=window).mean()
        return df

    @staticmethod
    def signal_ema(df, col_short, col_long):
        df['signal'] = 0.0
        df['signal'] = np.where(df[col_short] > df[col_long], 1.0, 0.0)
        df['position'] = df['signal'].diff()
        return df

    @staticmethod
    def merge_tweet_price(df_keywords_1, df_keywords_2, df_price, min_date, max_date, period_ema_1, period_ema_2):
        df_keywords = df_keywords_1.merge(df_keywords_2, on='date', suffixes=('_up', '_down'))
        df_keywords['ratio_up_down'] = df_keywords['nb_tweets_up'] / df_keywords['nb_tweets_down']

        df_keywords = Calculation.ema(df_keywords, 'ratio_up_down', period_ema_1)
        df_keywords = Calculation.ema(df_keywords, 'ratio_up_down', period_ema_2)
        df_keywords = Calculation.signal_ema(df_keywords, 'ratio_up_down_EMA_' + str(period_ema_1),
                                             'ratio_up_down_EMA_' + str(period_ema_2))

        df = df_keywords.merge(df_price, on='date')
        df = df[df['date'] > min_date]
        df = df[df['date'] <= max_date]
        return df

    @staticmethod
    def price_share(df):
        res = df.copy()
        price_share = []
        for i in range(0, res.shape[0]):
            if res['position'].iloc[i] == -1:
                price_share.append(res['price'].iloc[i])
            elif res['position'].iloc[i] == 1:
                price_share.append(res['price'].iloc[i])
            else:
                price_share.append(0)

        res['price_share'] = price_share
        return res

    @staticmethod
    def back_test(df, ptf_start, placed_start, perc_to_buy, perc_to_sold, commission_rate):
        """
        :param ptf_start: Money in the portfolio at the beginning of the backtest
        :param placed_start: Money placed at the beginning of the backtest
        :param perc_to_buy: Percentage of the portfolio to be invested in a purchase
        :param perc_to_sold: Percentage of the portfolio to be placed in a sale
        :param commission_rate: Commission fee for each transaction
        """
        df_backtest = df.copy()
        df_backtest = df_backtest.reset_index().drop('index', axis=1)

        ptf = [ptf_start]
        placed = [placed_start]
        nb_shares = [placed_start / df_backtest.price[0]]

        for i in range(1, df_backtest.shape[0]):
            if df_backtest['price_share'].iloc[i] != 0:

                # Buy
                if df_backtest['position'].iloc[i] == 1:
                    new_money_available = ptf[-1] * perc_to_buy  # Money available to invest before the commission fee
                    commission_fee = new_money_available * commission_rate  # Commission fee
                    new_money_placed = new_money_available - commission_fee  # Money available to invest after the commission fee
                    new_nb_shares = new_money_placed / df_backtest.price[i]  # Number of shares to buy
                    nb_shares_owned = nb_shares[-1] + new_nb_shares  # Total number of shares owned after purchase
                    money_ptf = ptf[-1] - new_money_available  # Money remaining in the portfolio after the purchase
                    money_placed = placed[-1] + new_money_placed  # Total money invested after purchase

                    placed.append(money_placed)
                    ptf.append(money_ptf)
                    nb_shares.append(nb_shares_owned)

                # Sell
                elif df_backtest['position'].iloc[i] == -1:
                    money_sold = nb_shares[-1] * df_backtest.price[i - 1] * perc_to_sold  # Money for sale
                    commission = money_sold * commission_rate  # Commission fee
                    money_placed = placed[-1] - money_sold  # Remaining invested money
                    money_ptf = ptf[-1] + money_sold - commission  # Money in the portfolio after the sale
                    nb_shares_owned = nb_shares[-1] - (
                            nb_shares[-1] * perc_to_sold)  # Number of shares remaining after the sale

                    placed.append(money_placed)
                    ptf.append(money_ptf)
                    nb_shares.append(nb_shares_owned)

                else:
                    placed.append(nb_shares[-1] * df_backtest.price[i])
                    ptf.append(ptf[-1])
                    nb_shares.append(nb_shares[-1])

            else:
                placed.append(nb_shares[-1] * df_backtest.price[i])
                ptf.append(ptf[-1])
                nb_shares.append(nb_shares[-1])

        df_backtest['ptf'] = ptf
        df_backtest['placed'] = placed
        df_backtest['nb_shares'] = nb_shares
        df_backtest['total'] = df_backtest['ptf'] + df_backtest['placed']
        df_backtest['benef'] = df_backtest['total'] - ptf_start

        ptf_end = ptf_start + df_backtest.iloc[-1]['benef']
        benef_backtest = round((ptf_end - ptf_start) / ptf_start * 100, 2)
        benef_hold = round(
            (df_backtest.iloc[-1]['price'] - df_backtest.iloc[0]['price']) / df_backtest.iloc[-0]['price'] * 100, 2)

        return df_backtest, benef_backtest, benef_hold


class Graph:
    @staticmethod
    def ema_line(df, x_var, y_value, col_ema_1, col_ema_2):
        st.title('Ratio between up and down tweets about bitcoin')
        col1, col2, col3 = st.columns(3)
        bool_value = col1.checkbox(y_value, value=False)
        bool_ema_1 = col2.checkbox(col_ema_1, value=True)
        bool_ema_2 = col3.checkbox(col_ema_2, value=True)

        fig = go.Figure()
        if bool_value:
            fig.add_trace(go.Scatter(x=x_var, y=df[y_value], name=y_value, mode='lines'))
        if bool_ema_1:
            fig.add_trace(go.Scatter(x=x_var, y=df[col_ema_1], name=col_ema_1, mode='lines'))
        if bool_ema_2:
            fig.add_trace(go.Scatter(x=x_var, y=df[col_ema_2], name=col_ema_2, mode='lines'))
        fig.update_layout(plot_bgcolor='white')

        st.write(fig)

    @staticmethod
    def price_signal(df):
        st.title('Bitcoin prices and signals')
        buy = df[df['position'] == 1]
        sell = df[df['position'] == -1]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['date'], y=df['price'], mode='lines', name='Price'))
        fig.add_trace(go.Scatter(x=sell['date'], y=sell['price'], mode='markers', name='Sell'))
        fig.add_trace(go.Scatter(x=buy['date'], y=buy['price'], mode='markers', name='Buy'))

        fig.update_traces(marker_size=10)
        fig.update_layout(plot_bgcolor='white')

        st.write(fig)

    @staticmethod
    def plot_back_test(df, benef_backtest, benef_hold):
        st.title(
            'Evolution of the profit during the back test : {}% VS {}% if we hold'.format(benef_backtest, benef_hold))

        buy = df[df['position'] == 1]
        sell = df[df['position'] == -1]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['date'], y=df['benef'], mode='lines', name='profit'))
        fig.add_trace(go.Scatter(x=sell['date'], y=sell['benef'], mode='markers', name='sell'))
        fig.add_trace(go.Scatter(x=buy['date'], y=buy['benef'], mode='markers', name='buy'))

        fig.update_yaxes(ticksuffix=' %')
        fig.update_traces(marker_size=10)
        fig.update_layout(plot_bgcolor='white')

        st.write(fig)
