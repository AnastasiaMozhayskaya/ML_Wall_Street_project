import streamlit as st
import pandas as pd
import yfinance as yf
import base64
import io
import os
import torch
from datetime import datetime, timedelta
from PIL import Image
from plotly import graph_objs as go
from datetime import date

st.set_page_config(layout='wide', initial_sidebar_state='expanded')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('ML Wall Street')
st.image('images/img.png')

# Функция для получения данных о ценах акций
def get_stock_data():
    dow_tickers = ['UNH', 'MSFT', 'GS', 'HD', 'AMGN', 'MCD', 'CAT', 'CRM', 'V', 'BA', 'HON', 'TRV', 'AAPL', 'AXP', 'JPM', 'IBM', 'JNJ', 'WMT', 'PG', 'CVX', 'MRK', 'MMM', 'NKE', 'DIS', 'KO', 'DOW', 'CSCO', 'INTC', 'VZ', 'WBA']
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    dow_data = yf.download(dow_tickers, start=start_date, end=end_date)
    return dow_data

data = get_stock_data() 
latest_date = data.index[-1].strftime('%Y-%m-%d')
data = data.loc[latest_date, 'Close'].reset_index()
data.columns = ['Ticker', 'Close']
data['Close'] = data['Close'].round(2)

# Добавляем кнопку обновления данных
# if st.button("Обновить данные", type="primary"):
#     data = get_stock_data()
#     latest_date = data.index[-1].strftime('%Y-%m-%d')
#     data = data.loc[latest_date, 'Close'].reset_index()
#     data.columns = ['Ticker', 'Close']
#     data['Close'] = data['Close'].round(2)
#     st.success("Данные успешно обновлены!")

st.markdown(f"<h3 style='text-align: center;'>Цены актуальны на последнюю дату закрытия торгов {latest_date}</h3>", unsafe_allow_html=True)

col3, col1, col2 = st.columns([0.2, 5.3, 1.8]) 
with col2:
    def image_to_base64(img_path, output_size=(64, 64)):
        if os.path.exists(img_path):
            with Image.open(img_path) as img:
                img = img.resize(output_size)
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
        return ""

    if 'Logo' not in data.columns:
        output_dir = 'downloaded_logos'
        data['Logo'] = data['Ticker'].apply(lambda name: os.path.join(output_dir, f'{name}.png'))

    # Convert image paths to Base64
    data["Logo"] = data["Logo"].apply(image_to_base64)
    image_column = st.column_config.ImageColumn(label="")
    ticker_column = st.column_config.TextColumn(label="Ticker 💬", help="📍**Тикеры компаний Индекса Dow Jones**")
    price_column = st.column_config.TextColumn(label=f"Close 💬", help="📍**Цена за последний день (в USD)**")

    data.reset_index(drop=True, inplace=True)
    data.index = data.index + 1

    data = data[['Logo', 'Ticker', 'Close']]
    st.write('')
    st.write('')
    st.markdown('**Компании Индекса Dow Jones**')
    st.dataframe(data, height=1088, column_config={"Logo": image_column, "Ticker":ticker_column, 'Close':price_column})

with col1:
    START = "1920-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")
    # @st.cache_data
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data

    def plot_raw_data(data, text):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Цена закрытия"))
        fig.update_layout(title_text=text, xaxis_rangeslider_visible=True)
        fig.update_traces(showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    data = load_data('^DJI')
    last_DJI = data['Close'].iloc[-1]
    diff_DJI = data['Close'].iloc[-1] - data['Close'].iloc[-2]
    pr_DJI = 100 * diff_DJI / last_DJI
    text_DJI = f'🇺🇸 Dow Jones Industrial Average (^DJI) \
        <span style="font-size: 1.5em;">{last_DJI:.2f}</span> <span style="font-size: 1em; color: crimson;">{diff_DJI:.2f}</span><span style="font-size: 1em; color: crimson;">({pr_DJI:.2f}%)</span>' \
        '<br><span style="font-size: 0.7em; color: grey;">DJI - DJI Real Time Price. Currency in USD</span>'
    plot_raw_data(data, text_DJI)
    check1 = st.checkbox("Исторические данные Dow Jones Industrial Average")
    if check1:
        st.write(data)

    data_500 = load_data('^GSPC')
    last_500 = data_500['Close'].iloc[-1]
    diff_500 = data_500['Close'].iloc[-1] - data_500['Close'].iloc[-2]
    pr_500 = 100 * diff_500 / last_500
    text_500 = f'🇺🇸 S&P 500 (^GSPC) \
        <span style="font-size: 1.5em;">{last_500:.2f}</span> <span style="font-size: 1em; color: crimson;">{diff_500:.2f}</span><span style="font-size: 1em; color: crimson;">({pr_500:.2f}%)</span>' \
        '<br><span style="font-size: 0.7em; color: grey;">SNP - SNP Real Time Price. Currency in USD</span>'
    plot_raw_data(data_500, text_500)
    check4 = st.checkbox("S&P 500")
    if check4:
        st.write(data_500)
    
    data_SSE = load_data('000001.SS')
    last_SSE = data_SSE['Close'].iloc[-1]
    diff_SSE = data_SSE['Close'].iloc[-1] - data_SSE['Close'].iloc[-2]
    pr_SSE = 100 * diff_SSE / last_SSE
    text_SSE = f'🇨🇳 SSE Composite Index (000001.SS) \
        <span style="font-size: 1.5em;">{last_SSE:.2f}</span> <span style="font-size: 1em; color: crimson;">{diff_SSE:.2f}</span><span style="font-size: 1em; color: crimson;">({pr_SSE:.2f}%)</span>' \
        '<br><span style="font-size: 0.7em; color: grey;">Shanghai - Shanghai Delayed Price. Currency in CNY</span>'
    plot_raw_data(data_SSE, text_SSE)
    check2 = st.checkbox("Исторические данные SSE Composite Index")
    if check2:
        st.write(data_SSE)

    data_IMOEX = load_data('IMOEX.ME')
    last_IMOEX  = data_IMOEX['Close'].iloc[-1]
    diff_IMOEX  = data_IMOEX['Close'].iloc[-1] - data_IMOEX['Close'].iloc[-2]
    pr_IMOEX = 100 * diff_IMOEX / last_IMOEX
    text_IMOEX= f'🇷🇺 MOEX Russia Index (IMOEX.ME) \
        <span style="font-size: 1.5em;">{last_IMOEX:.2f}</span> <span style="font-size: 1em; color: crimson;">{diff_IMOEX:.2f}</span><span style="font-size: 1em; color: crimson;">({pr_IMOEX:.2f}%)</span>' \
        '<br><span style="font-size: 0.7em; color: grey;">MCX - MCX Real Time Price. Currency in RUB</span>'
    plot_raw_data(data_IMOEX, text_IMOEX)
    check3 = st.checkbox("Исторические данные MOEX Russia Index")
    if check3:
        st.write(data_IMOEX)
