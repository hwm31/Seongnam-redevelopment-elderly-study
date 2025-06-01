import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick # Y축 퍼센트 포맷팅
from sklearn.linear_model import LinearRegression # 선형 회귀 모델
import numpy as np # 숫자 배열 처리

# --- 기본 설정 ---
START_YEAR = 2017
END_YEAR = 2024 # 실제 데이터가 있는 마지막 연도
FUTURE_YEARS_TO_PREDICT = 5 # 예측할 미래 연도 수
BASE_FILE_PATH = "/Users/wooyoungcho/Desktop/Univ/Data Science Term Project/Seongnam-redevelopment-elderly-study/성남시/Data/성남시 연도별 인구지표"

# 노인으로 간주할 연령 그룹 (65세 이상)
ELDERLY_AGE_GROUPS = [
    '65~69세', '70~74세', '75~79세', '80~84세',
    '85~89세', '90~94세', '95~99세', '100세이상'
]

# 주어진 단일 CSV 파일에서 전체 인구, 노인 인구, 노인 인구 비율을 분석
def analyze_single_year_population(csv_filepath, year):
    try:
        # CSV 파일 읽기
        df = pd.read_csv(csv_filepath, index_col=0, thousands=',', encoding='utf-8-sig')

        # 1. 전체 인구수 추출
        if '계' in df.index and '계' in df.columns:
            total_population = df.loc['계', '계']
        else:
            # print(f"오류 알림 (함수 내부): {year}년도 파일('{os.path.basename(csv_filepath)}')에서 전체 인구수 정보를 찾을 수 없습니다.")
            return None # 오류 발생 시 None 반환하여 호출부에서 처리

        # 2. 노인 인구수 계산 (65세 이상)
        elderly_population = 0
        for age_group in ELDERLY_AGE_GROUPS:
            if age_group in df.index:
                elderly_population += df.loc[age_group, '계']
            # else:
                # 개별 파일에 대한 경고는 너무 많을 수 있어 주석 처리
                # print(f"경고 알림 (함수 내부): {year}년도 파일('{os.path.basename(csv_filepath)}')에 '{age_group}' 그룹이 없습니다.")

        # 3. 노인 인구 비율 계산
        if total_population > 0:
            elderly_ratio = (elderly_population / total_population) * 100
        else:
            elderly_ratio = 0 # 전체 인구가 0이면 비율도 0

        return {
            '연도': year, # 컬럼명은 내부적으로 한글 유지 (그래프에서 영어로 변경)
            '전체인구': total_population,
            '노인인구': elderly_population,
            '노인인구비율(%)': elderly_ratio
        }
    except FileNotFoundError:
        # 이 오류는 호출부에서 파일 존재 여부 체크로 먼저 걸러지므로, 여기서 특별히 메시지 안띄워도 됨
        return None
    except Exception as e:
        # print(f"오류 알림 (함수 내부): {year}년도 파일('{os.path.basename(csv_filepath)}') 처리 중 예외 발생 - {e}")
        return None

print("셀 3: 데이터 분석 함수 'analyze_single_year_population' 정의 완료.")

all_years_data = [] # 모든 연도의 분석 결과를 저장할 리스트

print(f"데이터 로드 및 분석 시작: {START_YEAR}년 ~ {END_YEAR}년")
print(f"데이터 검색 경로: {BASE_FILE_PATH}")
print("-" * 70)

for year in range(START_YEAR, END_YEAR + 1):
    year_short = str(year)[-2:] # 예: 2017 -> "17"

    # 파일명 후보 (NFD '년' / 일반 '년' 순으로 확인)
    filename_nfd_part = f"{year_short}년_연령별_인구.csv" # '년' (NFD)
    filename_nfc_part = f"{year_short}년_연령별_인구.csv" # '년' (NFC)

    # 전체 경로 조합
    filepath_candidate_nfd = os.path.join(BASE_FILE_PATH, filename_nfd_part)
    filepath_candidate_nfc = os.path.join(BASE_FILE_PATH, filename_nfc_part)

    filepath_to_process = None

    if os.path.exists(filepath_candidate_nfd):
        filepath_to_process = filepath_candidate_nfd
    elif os.path.exists(filepath_candidate_nfc):
        filepath_to_process = filepath_candidate_nfc

    if filepath_to_process:
        print(f"  {year}년 데이터 처리 중 (파일: {os.path.basename(filepath_to_process)})... ", end="")
        yearly_result = analyze_single_year_population(filepath_to_process, year)
        if yearly_result:
            all_years_data.append(yearly_result)
            print("성공")
        else:
            print("실패 (데이터 분석 함수에서 오류)")
    else:
        print(f"  {year}년도 인구 파일을 찾을 수 없습니다.")
        # print(f"    (시도1: {filepath_candidate_nfd})") # 디버깅 시 주석 해제
        # print(f"    (시도2: {filepath_candidate_nfc})") # 디버깅 시 주석 해제

results_df = pd.DataFrame() # 빈 DataFrame으로 초기화
if all_years_data:
    results_df = pd.DataFrame(all_years_data)
    results_df = results_df.set_index('연도') # '연도'를 인덱스로 설정
    print("\n--- 연도별 인구 데이터 요약 ---")

    # DataFrame을 영어 캡션과 함께 포맷팅하여 출력
    formatted_df = results_df.style.format({
        '전체인구': "{:,.0f}",        # 내부 컬럼명은 한글
        '노인인구': "{:,.0f}",
        '노인인구비율(%)': "{:.2f}%"
    }).set_caption("Population Data Summary by Year (Raw Data)")
    print(formatted_df)

    print(f"\n셀 4: 총 {len(results_df)}개 연도의 데이터 로드 및 DataFrame 생성 완료.")
else:
    print("\n셀 4: 분석할 수 있는 데이터 파일이 없습니다. 파일 경로와 파일명을 확인해주세요.")


if not results_df.empty:
    print("\n--- 과거 데이터 시각화 ---")
    # 1. 연도별 노인 인구 수 변화 (선 그래프) - 영어 레이블
    plt.figure(figsize=(10, 6))
    plt.plot(results_df.index, results_df['노인인구'], marker='o', linestyle='-', color='skyblue', label='Elderly Population Count')
    plt.title('Historical Trend of Elderly Population (Age 65+)')
    plt.xlabel('Year')
    plt.ylabel('Number of Elderly People')
    plt.xticks(results_df.index)
    plt.grid(True, linestyle='--', alpha=0.7)
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
    for i, txt in enumerate(results_df['노인인구']):
        plt.annotate(f"{int(txt):,}", (results_df.index[i], results_df.iloc[i]['노인인구']), # iloc 사용
                     textcoords="offset points", xytext=(0,10), ha='center')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2. 연도별 노인 인구 비율 변화 (선 그래프) - 영어 레이블
    plt.figure(figsize=(10, 6))
    plt.plot(results_df.index, results_df['노인인구비율(%)'], marker='s', linestyle='--', color='salmon', label='Elderly Population Ratio')
    plt.title('Historical Trend of Elderly Population Ratio (Age 65+)')
    plt.xlabel('Year')
    plt.ylabel('Elderly Population Ratio (%)')
    plt.xticks(results_df.index)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100.0))
    plt.grid(True, linestyle='--', alpha=0.7)
    for i, txt in enumerate(results_df['노인인구비율(%)']):
        plt.annotate(f"{txt:.2f}%", (results_df.index[i], results_df.iloc[i]['노인인구비율(%)']), # iloc 사용
                     textcoords="offset points", xytext=(0,10), ha='center')
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("셀 5: 과거 데이터 시각화 완료.")
else:
    print("셀 5: 시각화할 과거 데이터가 없습니다.")


# predict
if not results_df.empty and len(results_df) >= 2: # 최소 2개의 데이터 포인트 필요
    print(f"\n--- 선형 회귀 분석 및 미래 예측 (향후 {FUTURE_YEARS_TO_PREDICT}년) ---")
    
    # 1. 데이터 준비
    X_train = results_df.index.values.reshape(-1, 1) # 연도 (독립 변수)
    y_train = results_df['노인인구비율(%)'].values    # 노인 인구 비율 (종속 변수)

    # 2. 선형 회귀 모델 훈련
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 모델 계수 및 절편 가져오기
    slope = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(X_train, y_train)

    print(f"\nLinear Regression Model Details:")
    print(f"  Coefficient (slope, m): {slope:.4f} (Rate of change in ratio per year)")
    print(f"  Intercept (c): {intercept:.4f} (Estimated ratio if Year=0, interpret with caution)")
    print(f"  R-squared (Goodness of fit): {r_squared:.4f}")
    print(f"  Regression Equation: Elderly_Ratio(%) = {slope:.4f} * Year + {intercept:.4f}")


    # 3. 미래 연도 생성 및 예측
    last_historical_year = results_df.index.max()
    future_years_predict = np.array(
        [last_historical_year + i for i in range(1, FUTURE_YEARS_TO_PREDICT + 1)]
    ).reshape(-1, 1)
    
    predicted_ratios_future = model.predict(future_years_predict)

    future_predictions_df = pd.DataFrame({
        'Year': future_years_predict.flatten(),
        'Predicted_Elderly_Ratio(%)': predicted_ratios_future
    })
    future_predictions_df = future_predictions_df.set_index('Year')
    
    print("\nPredicted Elderly Population Ratios for Future Years:")
    print(future_predictions_df.style.format({'Predicted_Elderly_Ratio(%)': "{:.2f}%"}))

    # 4. 결과 시각화 (과거 데이터 + 회귀선 + 미래 예측 + 회귀식)
    plt.figure(figsize=(14, 8)) # 그래프 크기 약간 조정
    
    # 실제 과거 데이터 포인트
    plt.scatter(X_train.flatten(), y_train, color='skyblue', label='Actual Historical Data', zorder=5)
    
    # 과거 데이터에 대한 회귀선
    plt.plot(X_train.flatten(), model.predict(X_train), color='green', linestyle='-', linewidth=2, label='Linear Regression Fit (Historical)')
    
    # 예측된 미래 값 포인트
    plt.scatter(future_years_predict.flatten(), predicted_ratios_future, color='red', marker='x', s=100, label=f'Predicted Data (Next {FUTURE_YEARS_TO_PREDICT} Years)', zorder=5)
    
    # 회귀선을 미래까지 연장하여 표시
    all_years_for_line = np.concatenate([X_train.flatten(), future_years_predict.flatten()]).reshape(-1,1)
    all_years_for_line.sort(axis=0) # x축 값을 정렬
    all_predicted_ratios_for_line = model.predict(all_years_for_line)
    plt.plot(all_years_for_line.flatten(), all_predicted_ratios_for_line, color='lightcoral', linestyle='--', linewidth=1.5, label='Extended Trend Line')

    # 회귀 직선의 방정식 텍스트로 추가
    equation_text = f'Regression Line: y = {slope:.4f}x + {intercept:.4f}\n$R^2 = {r_squared:.4f}$'
    # 텍스트 위치를 그래프 내 적절한 곳에 배치 (예: 왼쪽 상단)
    # transform=plt.gca().transAxes를 사용하면 축 기준으로 상대 위치 지정 (0,0)은 왼쪽 하단, (1,1)은 오른쪽 상단
    plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))


    plt.title('Elderly Population Ratio: Historical, Regression Fit, and Future Predictions')
    plt.xlabel('Year (x)') # x로 표시
    plt.ylabel('Elderly Population Ratio (%) (y)') # y로 표시
    
    x_ticks_combined = np.unique(np.concatenate([X_train.flatten(), future_years_predict.flatten()]))
    plt.xticks(x_ticks_combined.astype(int))
    
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100.0))
    plt.legend(loc='lower right') # 범례 위치 조정 (회귀식 텍스트와 겹치지 않도록)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # 제목과 그래프 영역 간 간격 확보
    plt.suptitle('Seongnam City Elderly Population Analysis', fontsize=16) # 전체 부제목 추가
    plt.show()
    print("\n셀 6: 선형 회귀 분석 및 미래 예측 시각화 (회귀식 포함) 완료.")

elif results_df.empty:
    print("\n셀 6: 선형 회귀 분석을 위한 데이터가 없습니다.")
else:
    print("\n셀 6: 선형 회귀 분석을 수행하려면 최소 2개 연도의 데이터가 필요합니다.")