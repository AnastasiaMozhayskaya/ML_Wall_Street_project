import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from sklearn.model_selection import train_test_split
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from prophet.make_holidays import make_holidays_df
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(layout='wide', initial_sidebar_state='expanded')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('ML Wall Street')
st.image('images/img.png')

START = "2020-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

stocks = ('AAPL', 'UNH', 'MSFT', 'GS', 'HD', 'AMGN', 'MCD', 'CAT', 'CRM', 'V', 'BA', 'HON', 'TRV', 'AXP', 'JPM', 'IBM', 'JNJ', 'WMT', 'PG', 'CVX', 'MRK', 'MMM', 'NKE', 'DIS', 'KO', 'DOW', 'CSCO', 'INTC', 'VZ', 'WBA')
selected_stock = st.selectbox('Выберите тикер из индекса Dow Jones', stocks)

period = st.slider('Количество дней прогноза:', 7, 14, 14)

# @st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data = load_data(selected_stock)
latest_date = data['Date'].iloc[-1].strftime('%Y-%m-%d')
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

def ticker(data, holidays_df, text1, text2, k):
    data = data
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
    # st.write(full_test_data3)

    # Создаем модель Prophet
    model = Prophet(interval_width=0.95)
    model.fit(full_train_data3)
    # Создаем фрейм для прогноза на тестовых данных, исключая даты праздников
    last_date = full_test_data3['ds'].max()
    future = model.make_future_dataframe(periods=full_test_data3.shape[0]+k, freq='B')
    future = future[~future['ds'].isin(holidays_df['ds'])]
    forecast_test = model.predict(future)
    # Создаем фрейм для прогноза на +14 дней после последней даты
    future_14_days = model.make_future_dataframe(periods=period, freq='B', include_history=False)
    future_14_days['ds'] = pd.date_range(start=last_date + pd.DateOffset(1), periods=period, freq='B')
    forecast_14_days = model.predict(future_14_days)

    # Отрисовка графика 
    fig = go.Figure()
    fig = plot_plotly(model, forecast_test)
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
    # Рассчитываем кастомную метрику точности прогноза для 1 дня с учетом весов предыдущих 7 дней
    forecast_values = forecast_test['yhat'].tail(period).values
    predicted_value = forecast_values[0]  # Выбираем первый предсказанный день из последних 14
    actual_values = full_test_data3['y'].tail(7).values
    weights = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    custom_mape = np.dot(weights, np.abs((actual_values - predicted_value) / actual_values)) / np.sum(weights) * 100
    check = st.checkbox(text2)
    if check:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.write(f"**Информация.** \
                    Горизонт планирования {period} дней. Валидация модели на тестовой выборке ({period} дней).")
            st.markdown("**Метрики для тестовой выборки:**")
            st.write(f"RMSE: {rmse_test:.2f}")
            st.write(f"MAPE: {mape_test:.2f}%")
            st.write(f"Weighted MAPE: {custom_mape:.2f}%")
            # Оценка тренда на первый день
            trend_evaluation_first_day = evaluate_trend_first_day(forecast_14_days['yhat'].iloc[0], full_test_data3['y'].iloc[-1])
            st.write(trend_evaluation_first_day)
            # Оценка тренда на период прогноза
            trend_evaluation_period = evaluate_trend_period(forecast_14_days[['ds', 'yhat']])
            st.write(trend_evaluation_period)
            st.info("📌 Кастомная метрика “weighted MAPE’’  - \
                    взвешенное среднее абсолютных процентных ошибок 1 дня прогноза по отношению к значениям крайних 7 дней.")
        with col2:
            forecast_results = pd.DataFrame({
                'Дата': forecast_14_days['ds'].iloc[-period:].values,
                'Прогноз': forecast_14_days['yhat'].iloc[-period:].values.round(2)
            })
            st.dataframe(forecast_results.set_index('Дата'))
        # fig2 = data.plot_components(future_14_days)
        # st.write(fig2)

text1 = f'График прогноза для {period} дней по акции {selected_stock}, USD 🇺🇸'
text2 = f"Результаты прогноза {selected_stock}"
ticker(data, holidays_df_US, text1, text2, 1)
