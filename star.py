import streamlit as st
import pandas as pd
import pickle
import folium
from streamlit_folium import st_folium

# 1. 페이지 기본 설정
st.set_page_config(page_title="세계 지진 위험도 예측 대시보드", layout="wide", page_icon="🌍")

st.title("🌍 세계 지진 데이터 기반 위험도 예측 웹 앱")
st.markdown("코랩에서 학습한 K-Means 군집 모델과 데이터를 활용하여 특정 위치의 지진 위험도를 예측합니다.")

# 2. 데이터 및 모델 로드 (캐싱을 이용해 속도 최적화)
@st.cache_data
def load_data():
    df = pd.read_csv('earthquake_clustered.csv')
    return df

@st.cache_resource
def load_models():
    with open('earthquake_kmeans_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('earthquake_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    df_new = load_data()
    model, scaler = load_models()
except FileNotFoundError as e:
    st.error(f"📁 필요한 파일이 부족합니다. 다운로드한 파일 3개를 현재 폴더에 넣어주세요. (에러: {e})")
    st.stop()

# 3. 사이드바 - 사용자 입력 받기 (노트북의 input() 기능 대체)
st.sidebar.header("📍 분석 위치 입력")
lat = st.sidebar.number_input("위도(Latitude) 입력", value=0.0, min_value=-90.0, max_value=90.0, step=0.1)
lon = st.sidebar.number_input("경도(Longitude) 입력", value=0.0, min_value=-180.0, max_value=180.0, step=0.1)

# 위험도 매핑 딕셔너리
risk_dict = {0: '🔴 높음 (규모가 크고 진원깊이가 얕음)', 
             1: '🔵 낮음 (규모와 깊이 모두 낮음)', 
             2: '🟢 중간 (규모가 크나 깊이가 깊음)'}

# 4. 메인 화면 - 예측 및 시각화 레이아웃 분할
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🔮 위험도 예측 결과")
    
    # 노트북의 주변 지진 탐색 로직 적용 (주변 +- 5도 범위)
    near_df = df_new[
        (df_new['위도'] >= lat - 5) & (df_new['위도'] <= lat + 5) &
        (df_new['경도'] >= lon - 5) & (df_new['경도'] <= lon + 5)
    ]
    
    if not near_df.empty:
        # 주변 군집의 비율 계산 후 가장 많은 군집 찾기
        cluster_ratio = near_df['cluster'].value_counts(normalize=True)
        main_cluster = cluster_ratio.idxmax()
        
        # 결과 출력
        st.metric(label="선택 구역 최종 위험도", value=risk_dict[main_cluster].split(" ")[1])
        st.write(f"**상세 상태:** {risk_dict[main_cluster]}")
        
        # 반경 내 데이터 통계 정보 제공
        st.write(f"ℹ️ 입력하신 위치 주변 ($\pm 5^\circ$) 데이터 추출 결과, 총 **{len(near_df)}건**의 지진 이력이 존재합니다.")
    else:
        st.warning("⚠️ 입력한 위치 주변 $\pm 5^\circ$ 이내에 최근 지진 데이터가 존재하지 않아 예측이 어렵습니다. 다른 위치를 입력해 주세요.")

with col2:
    st.subheader("🗺️ 지진 분포 및 입력 위치 지도")
    
    # 지도 생성 (입력된 위치 기준 또는 0,0 기준)
    m = folium.Map(location=[lat, lon], zoom_start=3)
    
    # 웹 로딩 속도를 위해 기존 노트북처럼 5000개 샘플링하여 지도에 표시
    df_sample = df_new.sample(n=min(5000, len(df_new)), random_state=42)
    colors = {0: 'red', 1: 'blue', 2: 'green'}
    
    for i in range(len(df_sample)):
        cluster_val = df_sample.iloc[i]['cluster']
        folium.CircleMarker(
            location=[df_sample.iloc[i]['위도'], df_sample.iloc[i]['경도']],
            radius=2,
            color=colors[cluster_val],
            fill=True,
            fill_color=colors[cluster_val],
            fill_opacity=0.4
        ).add_to(m)
    
    # 사용자가 입력한 위치 검은색 별표 마커로 추가
    folium.Marker(
        location=[lat, lon],
        popup=f"입력 위치 (위도: {lat}, 경도: {lon})",
        icon=folium.Icon(color='black', icon='star')
    ).add_to(m)
    
    # 스트림릿에 folium 지도 렌더링
    st_folium(m, width=800, height=500)