'''
기계학습과 응용 프로젝트
2020131013 정준혁
목적 : 다음 날 주가(종가)를 예측하는 모델
'''
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn.linear_model import Lasso
# 종목 딕셔너리
tickers = {
    'Samsung': '005930.KS',
    'Kakao': '035720.KS',
    'Naver': '035420.KS',
    'SKHynix': '000660.KS'
}

# Download Parameters
period = '1y'
interval = '1h'

models = {}  # model per ticker를 저장하기 위함
test_indices = None  # for alignment

for name, ticker in tickers.items():
    print(f"=== {name} ({ticker}) ===")

    # 데이터 다운로드
    stock_datas = yf.download(ticker, interval=interval, period=period)

    # 이동평균 계산
    stock_datas['moving_avgs_5'] = stock_datas['Close'].rolling(window=5).mean()
    stock_datas['moving_avgs_10'] = stock_datas['Close'].rolling(window=10).mean()
    stock_datas['moving_avgs_50'] = stock_datas['Close'].rolling(window=50).mean()

    # NaN 제거
    stock_datas = stock_datas.dropna()

    # Feature, Target 정의
    feat_set = stock_datas[['Close', 'moving_avgs_5', 'moving_avgs_10', 'moving_avgs_50']]
    close_price = stock_datas['Close'].shift(-1).dropna()
    feat_set = feat_set[:-1]

    # train_test_split 사용
    # 70% : train set
    # 30% : test set
    X_train, X_test, y_train, y_test = train_test_split(
        feat_set, close_price, test_size=0.3, random_state=10, shuffle=False
    )

    # 모델 훈련
    prediction_model = Lasso().fit(X_train, y_train)
    models[name] = prediction_model  # Store the trained model

    # 예측
    predictions = prediction_model.predict(X_test)

    # Store test indices
    if test_indices is None:
        test_indices = y_test.index

    # 평가
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mse)
    print(f'Mean Absolute Error (MAE): {mae:.4f}')
    print(f'Mean Squared Error: {mse}')
    print(f'R² Score: {r2}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')

    # 가장 최신의 특성을 사용하여 다음 날의 주가 예측
    latest_feat = stock_datas[['Close', 'moving_avgs_5', 'moving_avgs_10', 'moving_avgs_50']].iloc[-1].values.reshape(1, -1)
    predicted_next_close = prediction_model.predict(latest_feat)

    # 오늘의 실제 종가
    latest_close = stock_datas['Close'].iloc[-1]

    # 예측값과 실제값의 차이 계산
    difference = predicted_next_close[0] - latest_close

    # difference가 Series인지 확인하고 스칼라로 변환
    if isinstance(difference, (pd.Series, np.ndarray)):
        difference = difference.item()

    # 상승/감소/변동 없음 여부 판단
    if difference > 0:
        change_status = "상승"
    elif difference < 0:
        change_status = "감소"
    else:
        change_status = "변동 없음"

    # 결과 출력
    print(f'Predicted Next Closing Price for {name}: {predicted_next_close[0]:.2f}')
    print(f'Difference (Prediction - Today\'s Close): {difference:.2f} ({change_status})')

    # Plot: 실제 vs 예측
    plt.figure(figsize=(14, 7))
    plt.plot(y_test.index, y_test.values, label='Actual Price')
    plt.plot(y_test.index, predictions, label='Predicted Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'{name} - Actual vs. Predicted Stock Prices')
    plt.legend()
    plt.show()

    # Plot: 이동평균 + 종가
    plt.figure(figsize=(14, 7))
    plt.plot(stock_datas.index, stock_datas['Close'], label='Close Price')
    plt.plot(stock_datas.index, stock_datas['moving_avgs_5'], label='5-Period MA')
    plt.plot(stock_datas.index, stock_datas['moving_avgs_10'], label='10-Period MA')
    plt.plot(stock_datas.index, stock_datas['moving_avgs_50'], label='50-Period MA')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'{name} - Close Price Over Time')
    plt.legend()
    plt.show()
