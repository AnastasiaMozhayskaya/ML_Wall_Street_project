import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from prophet.make_holidays import make_holidays_df
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(layout='wide', initial_sidebar_state='expanded')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('ML Wall Street') 
st.image('images/img.png')

START = "2021-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

period = st.slider('Количество дней прогноза:', 7, 14, 14)

# @st.cache_data
def load_data():
    dji = yf.download('^DJI', START, TODAY)
    dji.reset_index(inplace=True)
    data_500 = yf.download('^GSPC', START, TODAY)
    data_500.reset_index(inplace=True)
    sse = yf.download('000001.SS', START, TODAY)
    sse.reset_index(inplace=True)
    imoex = yf.download('IMOEX.ME', START, TODAY)
    imoex.reset_index(inplace=True)
    return dji, data_500, sse, imoex

dji, data_500, sse, imoex = load_data()
latest_date = dji['Date'].iloc[-1].strftime('%Y-%m-%d')
st.markdown(f"<h3 style='text-align: center;'>Цены актуальны на последнюю дату закрытия торгов {latest_date}</h3>", unsafe_allow_html=True)

def evaluate_trend_first_day(predicted_value, actual_value):
    # Разница между первым днем прогноза и последним днем тестовых данных
    forecast_diff_first_last = predicted_value - actual_value

    # Оценка тренда на первый день: рост, падение, стабильность
    if forecast_diff_first_last > 0:
        return "Тенденция на первый день: Рост"
    elif forecast_diff_first_last < 0:
        return "Тенденция на первый день: Падение"
    else:
        return "Тенденция на первый день: Стабильность"

def evaluate_trend_period(forecast_14_days):
    # Разница между первым и последним значением прогноза
    forecast_diff = forecast_14_days['yhat'].iloc[-1] - forecast_14_days['yhat'].iloc[0]

    # Оценка тренда на весь период прогноза: рост, падение, стабильность
    print("Разница между первым и последним значением прогноза:", forecast_diff)
    
    if forecast_diff > 0:
        return "Тенденция на период прогноза: Рост"
    elif forecast_diff < 0:
        return "Тенденция на период прогноза: Падение"
    else:
        return "Тенденция на период прогноза: Стабильность"


# Формирование календаря для США
year_2023 = 2023
# Создаем DataFrame с встроенными праздниками для 2023
holidays_df_2023 = make_holidays_df(year_list=[year_2023], country='US')
# Преобразуем входные строки в datetime
holidays_df_2023['ds'] = pd.to_datetime(holidays_df_2023['ds'])
# Создаем DataFrame с кастомными праздниками для 2024 года
custom_holidays_2024 = pd.DataFrame({
    'holiday': 'custom',
    'ds': pd.to_datetime([
        '2024-01-01',  # Новый год   
        '2024-01-15',  # День Мартина Лютера Кинга младшего
        '2024-02-19',  # День рождения Дж. Вашингтона (Washington’s Birthday)
        '2024-03-29',  # Страстная пятница (Good Friday)
        '2024-05-27',  # День Памяти (Memorial Day)
        '2024-06-19',  # День национальной независимости | Juneteenth
        '2024-07-04',  # День независимости
        '2024-09-02',  # День труда
        '2024-11-28',  # День Благодарения
        '2024-12-24',  # Рождество
    ]),
    'lower_window': 0,
    'upper_window': 1,
})
# Объединяем все DataFrame с праздниками
all_holidays_US = pd.concat([holidays_df_2023, custom_holidays_2024]).drop_duplicates(subset=['ds']).sort_values(by=['ds'])
# Создаем DataFrame с датами праздников
holidays_df_US = pd.DataFrame({
    'ds': all_holidays_US['ds'],
    'holiday': 'holiday',
})

# Формирование календаря для Китая
year_2023 = 2023
holidays_ch_2023 = make_holidays_df(year_list=[year_2023], country='China')
# Преобразуем входные строки в datetime
holidays_ch_2023['ds'] = pd.to_datetime(holidays_ch_2023['ds'])
# Создаем DataFrame с кастомными праздниками для 2024 года
custom_holidays_2024_ch = pd.DataFrame({
    'holiday': 'custom',
    'ds': pd.to_datetime([
        '2024-01-01',  # Новый год   
        '2024-02-12',  # Китайский Новый Год
        '2024-02-13',  # Китайский Новый Год
        '2024-03-29',  # Страстная пятница (Good Friday)
        '2024-04-01',  # Пасхальный понедельник
        '2024-04-04',  # День ухода за могилами 
        '2024-05-01',  # Праздник Труда
        '2024-05-15',  # День рождения Будды
        '2024-06-10',  # Фестиваль Туен Нг 
        '2024-07-01'  # Праздник Специального Административного района
    ]),
    'lower_window': 0,
    'upper_window': 1,
})
# Объединяем все DataFrame с праздниками
all_holidays_China = pd.concat([holidays_ch_2023, custom_holidays_2024_ch]).drop_duplicates(subset=['ds']).sort_values(by=['ds'])
# Создаем DataFrame с датами праздников
holidays_df_China = pd.DataFrame({
    'ds': all_holidays_China['ds'],
    'holiday': 'holiday',
})

# Формирование календаря для России
year_2023 = 2023
holidays_r_2023 = make_holidays_df(year_list=[year_2023], country='Russia')
# Преобразуем входные строки в datetime
holidays_r_2023['ds'] = pd.to_datetime(holidays_r_2023['ds'])
# Создаем DataFrame с кастомными праздниками для 2024 года
custom_holidays_2024_r = pd.DataFrame({
    'holiday': 'custom',
    'ds': pd.to_datetime([
        '2024-01-01', 
        '2024-01-02',  
        '2024-02-23',  
        '2024-03-8',  
        '2024-05-01',  
        '2024-05-09',  
        '2024-06-12',  
        '2024-11-04' 
    ]),
    'lower_window': 0,
    'upper_window': 1,
})
# Объединяем все DataFrame с праздниками
all_holidays_r = pd.concat([holidays_r_2023, custom_holidays_2024_r]).drop_duplicates(subset=['ds']).sort_values(by=['ds'])
# Создаем DataFrame с датами праздников
all_holidays_r = pd.DataFrame({
    'ds': all_holidays_China['ds'],
    'holiday': 'holiday',
})

def index(ind, holidays_df, text1, text2, k=0):
    data = ind
    data = data.rename(columns={'Date': 'ds', 'Adj Close': 'y'})
    # Сортируем данные по дате
    data = data.sort_values(by='ds')
    # Определяем индекс для разделения
    split_index = len(data) - period

    # Разделяем данные на обучающую и тестовую выборки
    full_train_data3 = data.iloc[:split_index].copy()
    full_test_data3 = data.iloc[split_index:].copy()

    # Удаляем временную зону из столбца ds
    full_train_data3['ds'] = full_train_data3['ds'].dt.tz_localize(None)
    full_test_data3['ds'] = full_test_data3['ds'].dt.tz_localize(None)

    # Создаем модель Prophet
    model = Prophet(interval_width=0.95)
    model.fit(full_train_data3)
    # Создаем фрейм для прогноза на тестовых данных, исключая даты праздников
    last_date = full_test_data3['ds'].max()
    future = model.make_future_dataframe(periods=full_test_data3.shape[0]+k, freq='B')
    future = future[~future['ds'].isin(holidays_df['ds'])]
    forecast_test = model.predict(future)
    # latest_date2 = forecast_test['ds'].iloc[-1]
    # st.write(latest_date2)
    # Создаем фрейм для прогноза на +14 дней после последней даты
    future_14_days = model.make_future_dataframe(periods=period, freq='B', include_history=False)
    future_14_days['ds'] = pd.date_range(start=last_date + pd.DateOffset(1), periods=period, freq='B')
    forecast_14_days = model.predict(future_14_days)

    # Отрисовка графика 
    fig = go.Figure()
    fig = plot_plotly(model, forecast_test)
    # full_test_data3 = full_test_data3.loc[full_test_data3['ds'].isin(forecast_test['ds'])]
    fig.add_trace(go.Scatter(x=full_test_data3['ds'], 
                            y=full_test_data3['y'], 
                            mode='markers',
                            marker=dict(color='orchid'),
                            name='Факт тест'))
    fig.add_trace(go.Scatter(x=forecast_test['ds'].iloc[-period:], 
                             y=forecast_test['yhat'].iloc[-period:], 
                             mode='lines+markers', 
                             marker=dict(color='blue'),
                             name='Прогноз тест'))
    fig.add_trace(go.Scatter(x=forecast_14_days['ds'], y=forecast_14_days['yhat'], mode='lines+markers', name='Прогноз будущее'))
    fig.update_layout(title_text=text1, xaxis_rangeslider_visible=True, xaxis_title='', yaxis_title='')
    fig.update_traces(showlegend=True)
    st.plotly_chart(fig, use_container_width=True, range_slider_visible=True)  
    # Расчет метрик на тестовой выборке
    actual_values_test = full_test_data3['y'].values
    predicted_values_test = forecast_test['yhat'].iloc[-period:].values
    mape_test = np.mean(np.abs((actual_values_test - predicted_values_test) / actual_values_test)) * 100
    rmse_test = np.sqrt(mean_squared_error(actual_values_test, predicted_values_test))
    check = st.checkbox(text2)
    if check:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.write(f"**Информация.** \
                    Горизонт планирования {period} дней. Валидация модели на тестовой выборке ({period} дней).")
            st.markdown("**Метрики для тестовой выборки:**")
            st.write(f"MAPE: {mape_test:.2f}%")
            st.write(f"RMSE: {rmse_test:.2f}")
            # Оценка тренда на первый день
            trend_evaluation_first_day = evaluate_trend_first_day(forecast_14_days['yhat'].iloc[0], full_test_data3['y'].iloc[-1])
            st.write(trend_evaluation_first_day)
            # Оценка тренда на период прогноза
            trend_evaluation_period = evaluate_trend_period(forecast_14_days[['ds', 'yhat']])
            st.write(trend_evaluation_period)
        with col2:
            forecast_results = pd.DataFrame({
                'Дата': forecast_14_days['ds'].iloc[-period:].values,
                'Прогноз': forecast_14_days['yhat'].iloc[-period:].values.round(2)
            })
            st.dataframe(forecast_results.set_index('Дата'))

text1_dji = f'График прогноза для {period} дней по индексу Dow Jones, USD 🇺🇸'
text2_dji = f"Результаты прогноза по Dow Jones Industrial Average"
index(dji, holidays_df_US, text1_dji, text2_dji, 1)

text1_500 = f'График прогноза для {period} дней по индексу S&P 500, USD 🇺🇸'
text2_500 = f"Результаты прогноза по S&P 500"
index(data_500, holidays_df_US, text1_500, text2_500, 1)

text1_sse = f'График прогноза для {period} дней по индексу SSE Composite, CNY 🇨🇳'
text2_sse = f"Результаты прогноза по SSE Composite Index"
index(sse, holidays_df_China, text1_sse, text2_sse)

text1_imoex = f'График прогноза для {period} дней по индексу MOEX Russia, RUB 🇷🇺'
text2_imoex = f"Результаты прогноза по MOEX Russia Index"
index(imoex, all_holidays_r, text1_imoex, text2_imoex, 0)
