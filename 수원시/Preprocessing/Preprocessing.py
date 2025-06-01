# -*- coding: utf-8 -*-
#수원시 사회조사.ipynb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 전체 데이터 조회

social17 = pd.read_csv("/Users/wooyoungcho/Desktop/Univ/Data Science Term Project/Seongnam-redevelopment-elderly-study/수원시/Data(before_preprocessing)/수원시_사회조사_17년.csv", encoding='cp949')
social19 = pd.read_csv("/Users/wooyoungcho/Desktop/Univ/Data Science Term Project/Seongnam-redevelopment-elderly-study/수원시/Data(before_preprocessing)/수원시_사회조사_19년.csv", encoding='cp949')
social23 = pd.read_csv("/Users/wooyoungcho/Desktop/Univ/Data Science Term Project/Seongnam-redevelopment-elderly-study/수원시/Data(before_preprocessing)/수원시_사회조사_23년.csv", encoding='cp949')

datasets = {
    '2017': social17,
    '2019': social19,
    '2023': social23
}

for year, df in datasets.items():
    null_rows = df[df.isnull().any(axis=1)]
    print(f"=== {year}년 데이터 ===")
    print(f"전체 데이터 수: {len(df)}")
    print(f"컬럼 수: {len(df.columns)}")
    print(f"총 null 값: {df.isnull().sum().sum()}")
    print(f"null이 있는 컬럼 수: {(df.isnull().sum() > 0).sum()}")
    print(f"null이 있는 행 수: {len(null_rows)}")

    if len(null_rows) > 0:
        print("null이 있는 컬럼들:")
        null_columns = df.columns[df.isnull().any()].tolist()
        print(null_columns)
    else:
        print("null이 있는 행 없음")

    print(f"=== {year}년 null 비율 ===")
    null_percent = (df.isnull().sum() / len(df) * 100).round(2)
    null_info = null_percent[null_percent > 0]  # null이 있는 컬럼만
    if len(null_info) > 0:
        print(null_info.to_string())
    else:
        print("null 값 없음")
    print("-" * 50)

"""# Column 명 통일"""

#
new_social23 = social23[['만나이',
                         '시군거주년수',
                         '시군향후거주의향정도코드1_1',
                         '정주의식코드_1',
                         '거주지소속감정도코드1_1',
                         '지역생활만족도코드',
                         '주택만족도코드1_1',
                         '월평균가구소득유형코드_1',
                         '부채여부_1',
                         '삶만족도코드',
                         '지하철만족도코드1_1',
                         '택시만족도코드1_1',
                         '기차만족도코드1_1',
                         '시내버스만족도코드1_1',
                         '시외버스만족도코드1_1',
                         '수원도시공원이용만족도코드1_1']].copy()

# 대중교통만족도 6(해당없음)->3(보통이다)
cols = ['지하철만족도코드1_1', '택시만족도코드1_1', '기차만족도코드1_1', '시내버스만족도코드1_1', '시외버스만족도코드1_1']
new_social23[cols] = new_social23[cols].replace(6, 3)
# 소득구간 통일
new_social23['월평균가구소득유형코드_1'] = new_social23['월평균가구소득유형코드_1'].map({1:1, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:8})

# 1) 대중교통 만족도 평균 계산 및 불필요한 열 삭제
new_social23['지하철만족도코드1_1'] = new_social23[['지하철만족도코드1_1', '택시만족도코드1_1', '기차만족도코드1_1', '시내버스만족도코드1_1', '시외버스만족도코드1_1']].mean(axis=1)
new_social23 = new_social23.drop(columns=['택시만족도코드1_1', '기차만족도코드1_1', '시내버스만족도코드1_1', '시외버스만족도코드1_1'])
# 3) 최종 컬럼 이름(총 13개) 한 번에 지정
new_social23.columns = [
    '만나이',             # 기존 '만나이'
    '지역거주기간',       # 기존 '시군거주년수'
    '향후거주의향',       # 기존 '시군향후거주의향정도코드1_1'
    '정주의식',           # 기존 '정주의식코드_1'
    '거주지소속감',       # 기존 '거주지소속감정도코드1_1'
    '지역생활만족도',     # 기존 '지역생활만족도코드'
    '주거만족도',         # 기존 '주택만족도코드1_1'
    '월평균가구소득',     # 기존 '월평균가구소득유형코드_1'
    '부채유무',           # 기존 '부채여부_1'
    '삶의만족도',         # 기존 '삶만족도코드'
    '대중교통만족도',     # 기존 '지하철만족도코드1_1' (다섯 개 평균)
    '공원이용만족도', # 기존 '수원도시공원이용만족도코드1_1'
]


new_social19 = social19[[
    '만나이',
    '11.지역거주기간',
    '11-1.향후거주의향',
    '12.정주의식',
    '15.거주지소속감',
    '16.거주지만족도',
    '13.주택만족도',
    '25-1.월평균가구소득',
    '30.부채유무',
    '33.삶의만족도',
    '20-1.대중교통만족도_1지하철',
    '20-2.대중교통만족도_2기차',
    '20-3.대중교통만족도_3택시',
    '20-4.대중교통만족도_4버스',
    '47-2.도시공원_이용만족도_수원',
]].copy()

# 대중교통만족도 6(해당없음)->3(보통이다)
cols = ['20-1.대중교통만족도_1지하철', '20-2.대중교통만족도_2기차', '20-3.대중교통만족도_3택시', '20-4.대중교통만족도_4버스']
new_social19[cols] = new_social19[cols].replace(6, 3)
# 1) 대중교통 만족도 평균 계산 및 불필요한 열 삭제
new_social19['20-1.대중교통만족도_1지하철'] = new_social19[
    ['20-1.대중교통만족도_1지하철',
     '20-2.대중교통만족도_2기차',
     '20-3.대중교통만족도_3택시',
     '20-4.대중교통만족도_4버스']
].mean(axis=1)
new_social19 = new_social19.drop(columns=[
    '20-2.대중교통만족도_2기차',
    '20-3.대중교통만족도_3택시',
    '20-4.대중교통만족도_4버스'
])
# 2) 최종 컬럼 이름(총 13개) 한 번에 지정
new_social19.columns = [
    '만나이',             # 기존 '만나이'
    '지역거주기간',       # 기존 '시군거주년수'
    '향후거주의향',       # 기존 '시군향후거주의향정도코드1_1'
    '정주의식',           # 기존 '정주의식코드_1'
    '거주지소속감',       # 기존 '거주지소속감정도코드1_1'
    '지역생활만족도',     # 기존 '지역생활만족도코드'
    '주거만족도',         # 기존 '주택만족도코드1_1'
    '월평균가구소득',     # 기존 '월평균가구소득유형코드_1'
    '부채유무',           # 기존 '부채여부_1'
    '삶의만족도',         # 기존 '삶만족도코드'
    '대중교통만족도',     # 기존 '지하철만족도코드1_1' (다섯 개 평균)
    '공원이용만족도', # 기존 '47-2.도시공원_이용만족도_수원'
]


new_social17 = social17[[
    '만나이',
    '11.지역거주기간',
    '11-1.향후거주의향',
    '12.정주의식',
    '14.거주지소속감',
    '15.거주지만족도',
    '13.주택만족도',
    '24-1.월평균가구소득',
    '29.부채유무',
    '30.삶의만족도',
    '19-1.대중교통만족여부_지하철',
    '19-2.대중교통만족여부_기차',
    '19-3.대중교통만족여부_택시',
    '19-4.대중교통만족여부_버스',
    '47-2.도시공원이용만족도',
]].copy()

# 대중교통만족도 6(해당없음)->3(보통이다)
cols = ['19-1.대중교통만족여부_지하철', '19-2.대중교통만족여부_기차', '19-3.대중교통만족여부_택시', '19-4.대중교통만족여부_버스',]
new_social17[cols] = new_social17[cols].replace(6, 3)

# 1) 대중교통(지하철·기차·택시·버스) 만족도 평균
new_social17['19-1.대중교통만족여부_지하철'] = new_social17[
    ['19-1.대중교통만족여부_지하철',
     '19-2.대중교통만족여부_기차',
     '19-3.대중교통만족여부_택시',
     '19-4.대중교통만족여부_버스']
].mean(axis=1)
new_social17 = new_social17.drop(columns=[
    '19-2.대중교통만족여부_기차',
    '19-3.대중교통만족여부_택시',
    '19-4.대중교통만족여부_버스'
])
new_social17.columns = [
    '만나이',
    '지역거주기간',
    '향후거주의향',
    '정주의식',
    '거주지소속감',
    '지역생활만족도',
    '주거만족도',
    '월평균가구소득',
    '부채유무',
    '삶의만족도',
    '대중교통만족도',
    '공원이용만족도',
]


print('전처리 후 17년 : ')
print(new_social17)
print('='*50)

print('전처리 후 19년 : ')
print(new_social19)
print('='*50)

print('전처리 후 23년 : ')
print(new_social23)
print('='*50)

print("social research 2017")
print(social17.head())


print("\nsocial research 2019")
print(social19.head())


print("\nsocial research 2023")
print(social23.head())


new_social17 = new_social17[new_social17['만나이'] >= 65].dropna()
new_social19 = new_social19[new_social19['만나이'] >= 65].dropna()
new_social23 = new_social23[new_social23['만나이'] >= 65].dropna()

print("\nsocial research 2017")
print(new_social17.head())
print("\n17 data count:")
print(len(new_social17))

print("\nsocial research 2019")
print(new_social19.head())
print("\n19 data count:")
print(len(new_social19))

print("\nsocial research 2023")
print(new_social23.head())
print("\n23 data count:")
print(len(new_social23))

# 재개발 전 (17,19년도) / 재개발 후 (23년도) 데이터 합치기



def merge_preprocessed_data(data_dict, save_path=None):

    merged_data_list = []

    for year, data in data_dict.items():
        # 각 데이터에 년도 컬럼 추가
        data_copy = data.copy()
        data_copy['year'] = year
        merged_data_list.append(data_copy)

        print(f"{year}년 데이터 형태: {data_copy.shape}")
        print(f"{year}년 컬럼: {list(data_copy.columns)}")
        print("-" * 30)

    # 모든 데이터 합치기
    merged_data = pd.concat(merged_data_list, axis=0, ignore_index=True)

    print(f"\n=== 합친 결과 ===")
    print(f"전체 데이터 형태: {merged_data.shape}")
    print(f"년도별 데이터 수:")
    print(merged_data['year'].value_counts().sort_index())

    # 결측치 확인
    print(f"\n결측치 현황:")
    missing_info = merged_data.isnull().sum()
    if missing_info.sum() > 0:
        print(missing_info[missing_info > 0])
    else:
        print("결측치 없음")

    # CSV 저장
    if save_path:
        merged_data.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"\n파일 저장 완료: {save_path}")

    return merged_data

# 1. 재개발 이전 데이터 (17, 19) 합치기
redevelopment_before = {
    2017: new_social17,
    2019: new_social19
}

print("=== 재개발 이전 데이터 합치기 (2017, 2019) ===")
before_data = merge_preprocessed_data(
    redevelopment_before,
    save_path='redevelopment_before_2017_2019.csv'
)

print("\n" + "="*60 + "\n")

# 2. 재개발 이후 데이터 (23년만)
print("=== 재개발 이후 데이터 (2023) ===")
after_data = new_social23.copy()
after_data['year'] = 2023

print(f"2023년 데이터 형태: {after_data.shape}")
print(f"2023년 컬럼: {list(after_data.columns)}")

# 재개발 이후 데이터도 CSV로 저장
after_data.to_csv('redevelopment_after_2023.csv', index=False, encoding='utf-8-sig')
print("재개발 이후 파일 저장: redevelopment_after_2023.csv")

print("\n" + "="*60 + "\n")

# 3. 데이터 기본 통계 확인
def basic_statistics(data, data_name):
    #기본 통계 정보 출력
    print(f"=== {data_name} 기본 통계 ===")
    print(f"전체 행 수: {len(data)}")
    print(f"컬럼 수: {len(data.columns)}")

    # 년도별 분포
    if 'year' in data.columns:
        print("\n년도별 분포:")
        year_counts = data['year'].value_counts().sort_index()
        for year, count in year_counts.items():
            print(f"  {year}년: {count}개")

    # 목표 변수 기본 통계
    if '거주지만족도' in data.columns:
        print(f"\n거주지만족도 기본 통계:")
        print(data['거주지만족도'].describe())

    print("-" * 50)

# 기본 통계 확인
basic_statistics(before_data, "재개발 이전")
basic_statistics(after_data, "재개발 이후")

# 4. 컬럼 정보 최종 확인
print("=== 최종 컬럼 정보 ===")
print("재개발 이전 컬럼:", list(before_data.columns))
print("재개발 이후 컬럼:", list(after_data.columns))

# 컬럼이 모두 동일한지 확인
before_cols = set(before_data.columns)
after_cols = set(after_data.columns)

if before_cols == after_cols:
    print("\n✅ 재개발 이전/이후 컬럼이 모두 동일합니다!")
else:
    print(f"\n⚠️ 컬럼 차이 발견:")
    print(f"이전에만 있음: {before_cols - after_cols}")
    print(f"이후에만 있음: {after_cols - before_cols}")
