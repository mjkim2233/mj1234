import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from datetime import datetime

# --- 페이지 설정 ---
st.set_page_config(page_title="팀 예산 관리 대시보드", page_icon="📊", layout="wide")

# --- 메인 로직 ---
st.title("📊 팀 예산 관리 시스템")
st.markdown("부장님 보고용 월별 예산 취합 및 대시보드 (Google Sheets 연동)")

# Google Apps Script Web App URL (Streamlit Secrets에서 가져오기)
try:
    WEB_APP_URL = st.secrets["apps_script_url"]
except:
    st.error("⚠️ Streamlit Secrets에 `apps_script_url`을 설정해주세요. (README 참조)")
    st.stop()

# 구글 시트 데이터 로드 (Apps Script GET 요청)
try:
    response = requests.get(WEB_APP_URL)
    data = response.json()
    
    if not data:
        df = pd.DataFrame(columns=["ID", "연월", "팀원", "항목", "금액"])
    else:
        df = pd.DataFrame(data)
except Exception as e:
    st.error(f"⚠️ 구글 시트 연결에 실패했습니다.\n\n에러 내용: {e}")
    st.stop()

# 탭 구성
tab1, tab2 = st.tabs(["📝 데이터 입력", "📈 전체 대시보드"])

# --- TAB 1: 데이터 입력 ---
with tab1:
    st.subheader("새로운 예산 내역 입력")
    with st.form("budget_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            member = st.selectbox("팀원 선택", ["부장님", "팀원1", "팀원2", "팀원3", "팀원4"])
            month = st.date_input("해당 월 (날짜 선택)", datetime.today()).strftime("%Y-%m")
        with col2:
            category = st.selectbox("예산 항목", ["수선유지비", "비품", "개량공사"])
            amount = st.number_input("사용 금액 (원)", min_value=0, step=1000)

        submitted = st.form_submit_button("기록 저장하기")

        if submitted:
            new_id = int(datetime.now().timestamp())
            
            # Apps Script로 전송할 페이로드 데이터
            payload = {
                "ID": new_id, 
                "연월": month, 
                "팀원": member, 
                "항목": category, 
                "금액": amount
            }
            
            # POST 요청으로 데이터 전송
            try:
                res = requests.post(WEB_APP_URL, json=payload)
                if res.status_code == 200 and res.json().get("status") == "success":
                    st.success("✅ 구글 스프레드시트에 성공적으로 저장되었습니다!")
                    st.rerun() # 화면 새로고침
                else:
                    st.error(f"❌ 저장에 실패했습니다: {res.text}")
            except Exception as e:
                st.error(f"❌ 통신 중 오류가 발생했습니다: {e}")

    st.subheader("📂 최근 입력 내역 (Google Sheets)")
    st.dataframe(df.tail(10).sort_index(ascending=False), use_container_width=True)

# --- TAB 2: 전체 대시보드 ---
with tab2:
    if df.empty:
        st.info("아직 입력된 데이터가 없습니다.")
    else:
        # 금액 데이터를 숫자형으로 변환
        df['금액'] = pd.to_numeric(df['금액'])
        
        # 상단 통계 수치
        total_amount = df['금액'].sum()
        top_category = df.groupby('항목')['금액'].sum().idxmax()
        top_category_amount = df.groupby('항목')['금액'].sum().max()
        data_count = len(df)

        col1, col2, col3 = st.columns(3)
        col1.metric("전체 누적 사용액", f"{total_amount:,.0f}원")
        col2.metric("최대 사용 항목", f"{top_category}", f"{top_category_amount:,.0f}원")
        col3.metric("데이터 건수", f"{data_count}건")

        st.divider()

        # 차트 영역
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.markdown("**🏠 항목별 예산 분포**")
            cat_df = df.groupby('항목')['금액'].sum().reset_index()
            fig_cat = px.pie(cat_df, values='금액', names='항목', hole=0.4,
                             color_discrete_sequence=['#3b82f6', '#10b981', '#8b5cf6'])
            st.plotly_chart(fig_cat, use_container_width=True)

        with col_chart2:
            st.markdown("**👥 팀원별 누적 사용액**")
            mem_df = df.groupby('팀원')['금액'].sum().reset_index()
            fig_mem = px.bar(mem_df, x='팀원', y='금액', text_auto='.2s',
                             color_discrete_sequence=['#60a5fa'])
            st.plotly_chart(fig_mem, use_container_width=True)

        st.divider()
        
        # 월별/항목별 요약 피벗 테이블
        st.markdown("**📅 월별/항목별 요약 테이블 (취합본)**")
        pivot_df = df.pivot_table(index='연월', columns='항목', values='금액', aggfunc='sum', fill_value=0)
        pivot_df['합계'] = pivot_df.sum(axis=1)
        st.dataframe(pivot_df.style.format("{:,.0f}"), use_container_width=True)
