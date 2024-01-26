import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from datetime import date, datetime, timedelta
from model.BiLSTM_model import BiLSTM
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
import plotly.graph_objects as go

st.set_page_config(layout='wide', initial_sidebar_state='expanded')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('ML Wall Street')
st.image('images/img.png')

START = "2021-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
period = st.slider('Количество дней прогноза:', 7, 14, 14)

df = yf.download('BTC-USD', START, TODAY)
df.reset_index(inplace=True)

latest_date = df['Date'].iloc[-1].strftime('%Y-%m-%d')
st.markdown(f"<h3 style='text-align: center;'>Цены актуальны на последнюю дату закрытия торгов {latest_date}</h3>", unsafe_allow_html=True)

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def mape(predictions, targets):
    return np.mean(np.abs((targets - predictions) / targets)) * 100

def weighted_mape(predictions, targets, weights):
    errors = np.abs(targets - predictions)
    weighted_errors = errors * weights
    weighted_mape = np.sum(weighted_errors) / np.sum(np.abs(targets) * weights) * 100
    return weighted_mape

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Adj Close'].values.reshape(-1,1))

def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data)-time_step-1):
        a = data[i:(i+time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

X, y = create_dataset(scaled_data, time_step=period)

# Разделяем данные на обучающую и тестовую выборки
test_size = period  # 14 дней для теста
train_size = len(X) - test_size
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train = torch.Tensor(X_train).unsqueeze(-1)  # Добавляем размерность
X_test = torch.Tensor(X_test).unsqueeze(-1)    # Для совместимости с LSTM
y_train = torch.Tensor(y_train)
y_test = torch.Tensor(y_test)

input_size = 1  # Входной размер равен 1, так как мы рассматриваем одну фичу - цену
hidden_size = 128
num_layers = 3
output_size = 1
model = BiLSTM(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()

model = BiLSTM(1, 128, 3, 1)
model.load_state_dict(torch.load('model/model_weights.pth'))
model.eval()

# Получение весов модели из сессионного состояния
model_weights = st.session_state.model_weights

# Загрузка весов в модель
model.load_state_dict(model_weights)
with torch.no_grad():
    test_predictions = model(X_test)
    test_loss = criterion(test_predictions, y_test.view(-1, 1))

    test_predictions = scaler.inverse_transform(test_predictions.cpu().numpy())
    y_test = scaler.inverse_transform(y_test.view(-1, 1).cpu().numpy())
    
    test_rmse = rmse(test_predictions, y_test)
    test_mape = mape(test_predictions, y_test)
    weights = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    test_weighted_mape = weighted_mape(test_predictions, y_test, weights)

    # st.write(f'Test Loss: {test_loss.item():.4f}')
    # st.write(f'Test RMSE: {test_rmse:.4f}')
    # st.write(f'Test MAPE: {test_mape:.4f}%')
    # st.write(f'Test Weighted MAPE: {test_weighted_mape:.4f}%')

# Генерация дат для будущих предсказаний
future_dates = [df['Date'].values[-1] + np.timedelta64(i+1, 'D') for i in range(period)]
# Расчет тренда на первый день
trend_first_day = test_predictions[0] - df['Adj Close'].iloc[-1]
# Расчет тренда на последний день
trend_last_day = test_predictions[-1] - test_predictions[0]
# Прогнозы с учетом трендов
adjusted_future_predictions = [df['Adj Close'].iloc[-1] + trend_first_day + (trend_last_day / period) * i for i in range(period)]

adjusted_future_pred = pd.Series([x.item() for x in adjusted_future_predictions])

# Форматирование каждой даты в нужный формат
formatted_dates = np.datetime_as_string(future_dates)

# fig = go.Figure()
# fig.add_trace(go.Scatter(x=df['Date'], 
#                          y=df['Adj Close'], 
#                          mode='lines',
#                          name='Актуальные данные',
#                          line=dict(width=2)))

# fig.add_trace(go.Scatter(x=formatted_dates, 
#                          y=adjusted_future_pred, 
#                          mode='lines+markers',
#                          name='Прогноз будущее',
#                          line=dict(color='red', width=2, dash='dash')))

# fig.update_layout(title=f'График прогноза для {period} дней по Bitcoin, USD 🇺🇸',
#                   xaxis=dict(tickangle=45), xaxis_rangeslider_visible=True, showlegend=True)
# st.plotly_chart(fig, use_container_width=True)

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], 
                         y=df['Adj Close'], 
                         mode='lines',
                         name='Актуальные данные',
                         line=dict(width=2)))

fig.add_trace(go.Scatter(x=formatted_dates, 
                         y=adjusted_future_pred, 
                         mode='lines+markers',
                         name='Прогноз будущее',
                         line=dict(color='red', width=2, dash='dash')))

fig.update_layout(title=f'График прогноза для {period} дней по Bitcoin, USD 🇺🇸',
                  xaxis=dict(tickangle=45),
                  xaxis_range=[datetime.strptime(latest_date, '%Y-%m-%d') - timedelta(days=12), \
                               datetime.strptime(latest_date, '%Y-%m-%d') +  timedelta(days=18)], 
                               xaxis_rangeslider_visible=True, showlegend=True)

st.plotly_chart(fig, use_container_width=True)

# Генерация дат для будущих педсказаний
future_dates = [df['Date'].values[-1] + np.timedelta64(i+1, 'D') for i in range(period)]

# Расчет тренда на первый день
trend_first_day = adjusted_future_predictions[0] - df['Adj Close'].iloc[-1]

# Расчет тренда на последний день
trend_last_day = adjusted_future_predictions[-1] - adjusted_future_predictions[0]

# Вывод тренда на первый и последний день
trend_first_day_text = "Рост" if trend_first_day > 0 else "Падение" if trend_first_day < 0 else "Нет тренда"
trend_last_day_text = "Рост" if trend_last_day > 0 else "Падение" if trend_last_day < 0 else "Нет тренда"

check = st.checkbox('Результаты прогноза Bitcoin')
if check:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write(f"**Информация.** \
                 Горизонт планирования {period} дней. Валидация модели на тестовой выборке ({period} дней).")
        st.markdown("**Метрики для тестовой выборки:**")
        st.write(f"RMSE: {test_rmse:.2f}")
        st.write(f"MAPE: {test_mape:.2f}%")
        st.write(f"Weighted MAPE: {test_weighted_mape:.2f}%")
        # Оценка тренда на первый день
        st.write(f'Тенденция на первый день: {trend_first_day_text}')
        # Оценка тренда на период прогноза
        st.write(f'Тенденция на период прогноза: {trend_last_day_text}')
        st.info("📌 Кастомная метрика “weighted MAPE’’  - \
                 взвешенное среднее абсолютных процентных ошибок 1 дня прогноза по отношению к значениям крайних 7 дней.")
    with col2:
        results = pd.DataFrame({
            'Дата': future_dates,
            'Прогноз': adjusted_future_pred.values.round(2)
        })
        st.dataframe(results.set_index('Дата'))
