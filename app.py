import streamlit as st
import datetime
import pytz

from tools import LoadData, Calculation, Graph


def main():
    # Set configs
    st.set_page_config(
        layout="centered",  # Can be "centered" or "wide". In the future also "dashboard", etc.
        initial_sidebar_state="expanded",  # Can be "auto", "expanded", "collapsed"
        page_title='Twitter',  # String or None. Strings get appended with "• Streamlit".
        page_icon=None,  # String, anything supported by st.image, or None.
    )

    # Set init
    all_timezones = pytz.all_timezones
    list_granularity = ['1min', '5min', '30min', '2H', '1H', '3H', '6H', '12H', '1D', '1W']
    connection = LoadData.init_connection()

    # Set sidebar
    with st.sidebar:
        st.sidebar.title('General settings')

        timezone = st.selectbox('Timezone', all_timezones, index=all_timezones.index('Europe/Paris'))
        granularity = st.selectbox('Granularity', list_granularity, index=list_granularity.index('2H'))

        min_date = st.date_input('Min date', datetime.date.today() - datetime.timedelta(days=40))
        min_date_time = datetime.datetime.combine(min_date, datetime.datetime.min.time())
        max_date = st.date_input('Max date', datetime.date.today() + datetime.timedelta(days=1))
        max_date_time = datetime.datetime.combine(max_date, datetime.datetime.min.time())

        st.title('EMA Periods')
        period_ema_short = st.number_input('EMA period short', value=20)
        period_ema_long = st.number_input('EMA period long', value=70)

    # Display
    df_price_raw = LoadData.price(connection)
    df_price = LoadData.prepare_price(df_price_raw, timezone, granularity)

    st.title('Keywords choice')
    with st.form(key='keywords'):
        st.write("Write list separate by comma and without space")
        list_key_words_1 = st.text_input('Suggested list for up :', "bull,rise,risi,advanc,up,boom,expand,crack,moon")

        list_key_words_2 = st.text_input('Suggested list for down :', "bear,down,crash,depression,recession")
        st.form_submit_button(label='Submit')

    if list_key_words_1 != '' and list_key_words_2 != '':
        df_keywords_1_raw = LoadData.keywords(connection, list_key_words_1)
        df_keywords_1 = LoadData.prepare_keywords(df_keywords_1_raw, timezone, granularity)

        df_keywords_2_raw = LoadData.keywords(connection, list_key_words_2)
        df_keywords_2 = LoadData.prepare_keywords(df_keywords_2_raw, timezone, granularity)

        df_final = Calculation.merge_tweet_price(df_keywords_1, df_keywords_2, df_price, min_date_time, max_date_time,
                                                 period_ema_short, period_ema_long)
        df_final = Calculation.price_share(df_final)
        df_back_test, benef_backtest, benef_hold = Calculation.back_test(df=df_final, ptf_start=100, placed_start=0,
                                                                         perc_to_buy=1,
                                                                         perc_to_sold=1, commission_rate=0.001)

        Graph.ema_line(df_final, df_final['date'], 'ratio_up_down', 'ratio_up_down_EMA_' + str(period_ema_short),
                       'ratio_up_down_EMA_' + str(period_ema_long))
        Graph.price_signal(df_final)
        Graph.plot_back_test(df_back_test, benef_backtest, benef_hold)

        st.title('Data')
        st.write(df_back_test)

    # Bottom page
    st.write("\n")
    st.write("\n")
    st.info("""By : [Linkedin](https://www.linkedin.com/in/florent-barre-a25921194/) / 
            Ligue des Datas [Instagram](https://www.instagram.com/ligueddatas/)  |
            Data source : [Twitter](https://twitter.com/home)""")


if __name__ == "__main__":
    main()
