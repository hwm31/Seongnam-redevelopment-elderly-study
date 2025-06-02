import pandas as pd

def print_basic_stats(df: pd.DataFrame, period: str, year_col: str, target_col: str):
    print(f"\n=== {period} 기본 통계 ===")
    print(f"전체 행  수: {len(df)}")
    print(f"컬럼  수: {df.shape[1]}\n")

    # 연도별 분포
    year_counts = df[year_col].value_counts().sort_index()
    print("연도별 분포:")
    for y, cnt in year_counts.items():
        print(f" {y}년: {cnt}개")
    print()

    # 대상 컬럼의 기술통계
    print(f"{target_col} 기본 통계:")
    print(df[target_col].describe())     # count‧mean‧std‧min‧quartile‧max
    print("-" * 35)

# ---------- 경로와 인코딩만 바꿔 주십시오 ----------
pre_df  = pd.read_csv("redevelopment_before_2017_2019.csv",  encoding="utf-8-sig")
post_df = pd.read_csv("redevelopment_after_2023.csv", encoding="utf-8-sig")

# 분석 대상 컬럼명도 필요에 따라 수정
YEAR_COL   = "year"
TARGET_COL = "지역생활만족도"

print_basic_stats(pre_df,  "재개발 이전", YEAR_COL, TARGET_COL)
print_basic_stats(post_df, "재개발 이후", YEAR_COL, TARGET_COL)
