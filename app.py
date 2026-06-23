import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as pd_st  # 이름 충돌 방지
import streamlit as st
from scipy import stats
from scipy.fft import rfft, rfftfreq
from scipy.io import loadmat
import kagglehub

# -----------------------------------------------------------------------------
# 1. 스트림릿 페이지 설정 및 스타일 (한글 폰트 포함)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="베어링 진동 기반 CBM 진단 시스템", layout="wide")

# Matplotlib 한글 깨짐 방지 설정
plt.rcParams["axes.unicode_minus"] = False
# 로컬 환경에 설치된 한글 폰트 지정 (예: Windows: Malgun Gothic, Mac: AppleGothic)
import platform
if platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
elif platform.system() == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
else:
    plt.rcParams["font.family"] = "DejaVu Sans" # 리눅스 기본 (필요시 NanumBarunGothic 지정)

st.title("⚙️ 베어링 진동 신호 기반 상태 모니터링 (CBM) 시스템")
st.markdown("Kaggle의 **CWRU 베어링 데이터셋**을 활용하여 정상/이상 상태를 분석하고 진단합니다.")

# -----------------------------------------------------------------------------
# 2. 사이드바 - 데이터 및 하이퍼파라미터 설정
# -----------------------------------------------------------------------------
st.sidebar.header("🛠️ 시스템 설정")

# 2-1. 데이터셋 다운로드 및 로드 모드 선택
data_source = st.sidebar.radio("데이터 소스 선택", ["Kaggle 자동 다운로드 (CWRU)", "임시 예제 신호 사용"])
FS = st.sidebar.number_input("샘플링 주파수 (Hz)", value=12000, step=1000)

# 2-2. 진단 임계치(Threshold) 설정
st.sidebar.subheader("🚨 진단 기준치 설정")
kurtosis_threshold = st.sidebar.slider("Kurtosis 주의 기준", 2.0, 7.0, 5.0, 0.1)
crest_threshold = st.sidebar.slider("Crest Factor 주의 기준", 2.0, 7.0, 4.0, 0.1)

# -----------------------------------------------------------------------------
# 3. 핵심 연산 함수들
# -----------------------------------------------------------------------------
@st.cache_data
def load_kaggle_data():
    """Kaggle에서 베어링 데이터를 다운로드하고 정상/이상 신호를 추출합니다."""
    try:
        path = kagglehub.dataset_download("vinayak123tyagi/bearing-dataset")
        mat_files = glob.glob(os.path.join(path, "**/*.mat"), recursive=True)
        
        # CWRU 예시 파일 필터링 (정상: 97.mat, 외륜 결함: 105.mat 예시)
        normal_path = [f for f in mat_files if "97.mat" in f or "97" in os.path.basename(f)]
        fault_path = [f for f in mat_files if "105.mat" in f or "105" in os.path.basename(f)]
        
        if normal_path and fault_path:
            mat_n = loadmat(normal_path[0])
            mat_f = loadmat(fault_path[0])
            
            # 드라이브 엔드(DE) 타임 데이터 키 찾기
            n_key = [k for k in mat_n.keys() if "DE_time" in k][0]
            f_key = [k for k in mat_f.keys() if "DE_time" in k][0]
            
            # 2초 분량 샘플링
            max_samples = FS * 2
            ns = np.asarray(mat_n[n_key]).ravel()[:max_samples]
            fs = np.asarray(mat_f[f_key]).ravel()[:max_samples]
            return ns, fs, "CWRU Bearing Dataset", True
    except Exception as e:
        st.sidebar.error(f"Kaggle 데이터 로드 실패: {e}")
    return None, None, "", False

def generate_mock_data():
    """데이터 다운로드가 안 되거나 임시 모드일 때 사용할 가상 데이터 생성"""
    duration = 2.0
    t = np.arange(0, duration, 1 / FS)
    ns = 0.8 * np.sin(2 * np.pi * 60 * t) + 0.08 * np.random.randn(len(t))
    fs = ns.copy()
    impact_positions = np.arange(0, len(t), int(FS / 90))
    for pos in impact_positions:
        if pos + 40 < len(fs):
            fs[pos:pos+40] += np.hanning(40) * np.random.uniform(1.5, 2.2)
    return ns, fs, "임시 가상 데이터셋"

# 데이터 확보
if data_source == "Kaggle 자동 다운로드 (CWRU)":
    with st.spinner("Kaggle에서 베어링 데이터를 다운로드 중입니다..."):
        normal_signal, fault_signal, dataset_name, success = load_kaggle_data()
    if not success:
        st.warning("Kaggle 데이터를 가져오지 못해 가상 데이터로 대체합니다.")
        normal_signal, fault_signal, dataset_name = generate_mock_data()
else:
    normal_signal, fault_signal, dataset_name = generate_mock_data()

# 특징값 계산 함수
def calculate_features(signal):
    signal = np.asarray(signal).ravel()
    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))
    kurtosis = stats.kurtosis(signal, fisher=False)
    skewness = stats.skew(signal)
    crest_factor = peak / rms if rms > 0 else np.nan
    return {
        "mean": np.mean(signal),
        "std": np.std(signal),
        "rms": rms,
        "peak": peak,
        "kurtosis": kurtosis,
        "skewness": skewness,
        "crest_factor": crest_factor,
        "mean_abs": np.mean(np.abs(signal)),
    }

# 구간별 특징값 추출 함수
def window_features(signal, fs, window_sec=0.2, step_sec=0.1):
    signal = np.asarray(signal).ravel()
    window = int(fs * window_sec)
    step = int(fs * step_sec)
    rows = []
    for start in range(0, len(signal) - window + 1, step):
        seg = signal[start:start + window]
        rows.append({"time_sec": start / fs, **calculate_features(seg)})
    return pd.DataFrame(rows)

# FFT 연산 함수
def compute_fft(signal, fs):
    signal = np.asarray(signal).ravel()
    signal = signal - np.mean(signal)
    n = len(signal)
    window = np.hanning(n)
    spectrum = np.abs(rfft(signal * window)) / n
    freq = rfftfreq(n, 1 / fs)
    return freq, spectrum

# -----------------------------------------------------------------------------
# 4. 웹 화면 대시보드 구성
# -----------------------------------------------------------------------------
tabs = st.tabs(["📊 데이터 개요 및 시각화", "📈 주파수 분석 (FFT)", "🚨 실시간 트렌드 및 고장 진단"])

# -----------------------------------------------------------------------------
# Tab 1: 데이터 개요 및 시각화
# -----------------------------------------------------------------------------
with tabs[0]:
    st.subheader("📋 분석 데이터 정보")
    col1, col2, col3 = st.columns(3)
    col1.metric("데이터셋 이름", dataset_name)
    col2.metric("샘플링 주파수", f"{FS} Hz")
    col3.metric("데이터 길이", f"{len(normal_signal)} 샘플")

    # 시간 도메인 파형 시각화
    st.subheader("⏱️ 시간 영역 진동 신호 파형 (0.2초 구간)")
    seconds_to_show = 0.2
    n_samples = min(len(normal_signal), int(FS * seconds_to_show))
    time_axis = np.arange(n_samples) / FS

    fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    ax[0].plot(time_axis, normal_signal[:n_samples], color='#1f77b4')
    ax[0].set_title("정상 진동 신호")
    ax[0].grid(alpha=0.3)
    
    ax[1].plot(time_axis, fault_signal[:n_samples], color='#d62728')
    ax[1].set_title("이상 진동 신호")
    ax[1].set_xlabel("Time (s)")
    ax[1].grid(alpha=0.3)
    st.pyplot(fig)

    # 전체 특징값 비교 테이블 및 바차트
    st.subheader("📊 통계적 특징값(Features) 비교")
    feature_df = pd.DataFrame([
        {"state": "정상 (Normal)", **calculate_features(normal_signal)},
        {"state": "이상 (Fault)", **calculate_features(fault_signal)},
    ])
    st.dataframe(feature_df.style.format(precision=4))

    # 바 차트 시각화
    plot_cols = ["rms", "peak", "kurtosis", "crest_factor"]
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    feature_df.set_index("state")[plot_cols].T.plot(kind="bar", ax=ax2)
    plt.title("정상/이상 특징값 비교")
    plt.ylabel("Feature value")
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.3)
    st.pyplot(fig2)

# -----------------------------------------------------------------------------
# Tab 2: 주파수 분석 (FFT)
# -----------------------------------------------------------------------------
with tabs[1]:
    st.subheader("🔍 Fast Fourier Transform (FFT) 분석")
    max_freq = st.slider("시각화할 최대 주파수 범위 (Hz)", 100, int(FS/2), 1000)

    fig_fft, ax_fft = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    # 정상 FFT
    freq_n, spec_n = compute_fft(normal_signal, FS)
    mask_n = freq_n <= max_freq
    ax_fft[0].plot(freq_n[mask_n], spec_n[mask_n], color='#1f77b4')
    ax_fft[0].set_title("정상 신호 주파수 스펙트럼")
    ax_fft[0].grid(alpha=0.3)
    
    # 이상 FFT
    freq_f, spec_f = compute_fft(fault_signal, FS)
    mask_f = freq_f <= max_freq
    ax_fft[1].plot(freq_f[mask_f], spec_f[mask_f], color='#d62728')
    ax_fft[1].set_title("이상 신호 주파수 스펙트럼")
    ax_fft[1].set_xlabel("Frequency (Hz)")
    ax_fft[1].grid(alpha=0.3)
    
    st.pyplot(fig_fft)

# -----------------------------------------------------------------------------
# Tab 3: 실시간 트렌드 및 고장 진단
# -----------------------------------------------------------------------------
with tabs[2]:
    st.subheader("📈 시계열 구간별 특징값 추세 (Trend)")
    
    # 구간 계산
    normal_win = window_features(normal_signal, FS)
    fault_win = window_features(fault_signal, FS)
    normal_win["state"] = "normal"
    fault_win["state"] = "fault"
    trend_df = pd.concat([normal_win, fault_win], ignore_index=True)

    # 정상 데이터의 베이스라인 통계치 계산 후 RMS 임계치 자동 설정
    normal_baseline = normal_win[["rms", "kurtosis", "crest_factor"]].agg(["mean", "std"])
    rms_threshold = normal_baseline.loc["mean", "rms"] + 3 * normal_baseline.loc["std", "rms"]

    # 동적 룰 기반 진단 함수
    def diagnose(row):
        reasons = []
        if row["rms"] > rms_threshold: reasons.append("RMS 증가")
        if row["kurtosis"] > kurtosis_threshold: reasons.append("충격성 증가")
        if row["crest_factor"] > crest_threshold: reasons.append("Crest Factor 증가")

        if len(reasons) >= 2: return "위험", ", ".join(reasons)
        if len(reasons) == 1: return "주의", reasons[0]
        return "정상", "-"

    # 추세 그래프 그리기
    for col in ["rms", "kurtosis", "crest_factor"]:
        fig_t, ax_t = plt.subplots(figsize=(12, 2.5))
        for state, group in trend_df.groupby("state"):
            ax_t.plot(group["time_sec"], group[col], label="정상 구간 시퀀스" if state=='normal' else "이상 발생 구간 시퀀스")
        
        # 가이드라인(임계치) 표시
        if col == "rms":
            ax_t.axhline(rms_threshold, color='r', linestyle='--', label=f'임계치 ({rms_threshold:.4f})')
        elif col == "kurtosis":
            ax_t.axhline(kurtosis_threshold, color='r', linestyle='--', label=f'임계치 ({kurtosis_threshold:.1f})')
        elif col == "crest_factor":
            ax_t.axhline(crest_threshold, color='r', linestyle='--', label=f'임계치 ({crest_threshold:.1f})')
            
        ax_t.set_title(f"시간 흐름에 따른 {col.upper()} 지표 추세")
        ax_t.set_xlabel("Time (s)")
        ax_t.legend()
        ax_t.grid(alpha=0.3)
        st.pyplot(fig_t)

    # 실시간 진단 결과 리포트 테이블
    st.subheader("🚨 이상 신호 시퀀스에 대한 실시간 진단 결과")
    diagnosis = fault_win.copy()
    diagnosis[["진단 결과", "원인"]] = diagnosis.apply(
        lambda row: pd.Series(diagnose(row)), axis=1
    )
    
    # 결과 요약 표시
    counts = diagnosis["진단 결과"].value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 정상 판정 구간", counts.get("정상", 0))
    c2.metric("🟡 주의 필요 구간", counts.get("주의", 0))
    c3.metric("🔴 위험 경보 구간", counts.get("위험", 0))

    st.dataframe(
        diagnosis[["time_sec", "rms", "kurtosis", "crest_factor", "진단 결과", "원인"]].style.map(
            lambda val: 'background-color: #ffcccc' if val == '위험' else ('background-color: #fff2cc' if val == '주의' else ''),
            subset=['진단 결과']
        ),
        use_container_width=True
    )
