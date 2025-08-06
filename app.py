import streamlit as st
import pandas as pd
import numpy as np
import firebase_admin
from firebase_admin import credentials, db
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from windrose import WindroseAxes
import matplotlib.pyplot as plt
from scipy import stats
from fpdf import FPDF
from PIL import Image 
import os
import io
import tempfile

# --- 1. CONFIGURAÇÃO DA PÁGINA E ESTILO ---
st.set_page_config(page_title="Análise Meteorológica", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #1a1c20; }
    div[data-testid="stMetric"] { background-color: #25282e; border-radius: 10px; padding: 15px; text-align: center; }
    h2, h3 { border-bottom: 2px solid #00bcd4; padding-bottom: 5px; }
    .stButton>button { background-color: #00bcd4; color: #1a1c20; font-weight: bold; width: 100%;}
    div[data-testid="stDownloadButton"] > button {
        background-color: #333a44; color: #e1e1e1; border: 1px solid #00bcd4;
        width: auto; padding: 4px 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛰️ Ferramenta de Análise Meteorológica Avançada")

# --- 2. FUNÇÕES DE CONEXÃO E PREPARAÇÃO DE DADOS ---
@st.cache_resource
def conectar_ao_firebase():
    try:
        creds_dict = dict(st.secrets.firebase_credentials)
        database_url = creds_dict.get("databaseURL", 'https://dados-meteorologicos-ca4f9-default-rtdb.firebaseio.com/')
        if not firebase_admin._apps:
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred, {'databaseURL': database_url})
        return db
    except Exception as e:
        st.sidebar.error(f"Erro de credenciais do Firebase: Verifique seu secrets.toml")
        return None

@st.cache_data(ttl=600)
def carregar_dados_firebase(_db_connection):
    if _db_connection is None: return pd.DataFrame()
    ref = _db_connection.reference('dados')
    dados = ref.get()
    if not dados: return pd.DataFrame()
    
    df = pd.DataFrame.from_dict(dados, orient='index')
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    colunas_numericas = ['temperatura', 'umidade', 'pressao', 'velocidade_vento']
    for col in colunas_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True)
    df['velocidade_ms'] = df['velocidade_vento'] * 0.27778
    df.sort_values('timestamp', inplace=True)
    return df

# --- 3. FUNÇÃO DE GERAÇÃO DE PDF ---
def criar_relatorio_pdf(caminho_pasta, resumo_df, data_inicio, data_fim, graficos_info):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('helvetica', 'B', 18)
    pdf.cell(0, 10, 'Relatório de Análise Meteorológica', ln=True, align='C')
    pdf.set_font('helvetica', '', 12)
    pdf.cell(0, 10, f"Período: {data_inicio.strftime('%d/%m/%Y')} a {data_fim.strftime('%d/%m/%Y')}", ln=True, align='C')
    pdf.ln(10)

    if "Resumo Estatístico" in graficos_info and not resumo_df.empty:
        pdf.set_font('helvetica', 'B', 12); pdf.cell(0, 10, 'Resumo Estatístico', ln=True)
        pdf.set_font('helvetica', '', 8); col_width = pdf.epw / (len(resumo_df.columns) + 1); line_height = pdf.font_size * 2
        pdf.set_font('helvetica', 'B', 8); pdf.cell(col_width, line_height, '', border=1)
        for col in resumo_df.columns: pdf.cell(col_width, line_height, col, border=1, align='C')
        pdf.ln()
        pdf.set_font('helvetica', '', 8)
        for index, row in resumo_df.iterrows():
            pdf.set_font('helvetica', 'B', 8); pdf.cell(col_width, line_height, index, border=1)
            pdf.set_font('helvetica', '', 8)
            for col in resumo_df.columns: pdf.cell(col_width, line_height, str(row[col]), border=1, align='C')
            pdf.ln()
    
    for titulo, nome_arquivo in graficos_info.items():
        if titulo == "Resumo Estatístico": continue
        caminho_imagem = os.path.join(caminho_pasta, nome_arquivo)
        if os.path.exists(caminho_imagem):
            pdf.add_page(orientation='L'); pdf.set_font('helvetica', 'B', 14)
            pdf.cell(0, 10, titulo, ln=True, align='C'); pdf.ln(5)
            
            page_width = pdf.epw; page_height = pdf.eph
            with Image.open(caminho_imagem) as img: img_width, img_height = img.size
            aspect_ratio = img_height / img_width
            
            new_width = page_width
            new_height = new_width * aspect_ratio
            if new_height > page_height:
                new_height = page_height
                new_width = new_height / aspect_ratio

            x = (page_width - new_width) / 2
            y = pdf.get_y()
            pdf.image(caminho_imagem, x=x, y=y, w=new_width, h=new_height)
    
    caminho_final_pdf = os.path.join(caminho_pasta, "relatorio_final.pdf")
    pdf.output(caminho_final_pdf)
    return caminho_final_pdf

# --- 4. LÓGICA DE AUTENTICAÇÃO ---
def check_password():
    if "password_correct" not in st.session_state: st.session_state.password_correct = False
    if not st.session_state.password_correct:
        with st.form("login"):
            st.header("Acesso Restrito")
            password = st.text_input("Digite a senha para acessar:", type="password")
            submitted = st.form_submit_button("Entrar")
            if submitted:
                if password == st.secrets["app_password"]:
                    st.session_state.password_correct = True
                    st.rerun()
                else:
                    st.error("Senha incorreta.")
        return False
    return True

# --- INÍCIO DA EXECUÇÃO DO DASHBOARD ---
if check_password():
    database = conectar_ao_firebase()
    if database:
        df_completo = carregar_dados_firebase(database)

        if not df_completo.empty:
            VENTO_MAX_MS = 20.0
            registros_antes = len(df_completo)
            df_sem_outliers = df_completo[df_completo['velocidade_ms'] <= VENTO_MAX_MS].copy()
            registros_depois = len(df_sem_outliers)
            
            st.sidebar.header("🎛️ Controles de Análise")
            if registros_antes > registros_depois:
                st.sidebar.warning(f"{registros_antes - registros_depois} registro(s) com vento > {VENTO_MAX_MS} m/s foram removidos.")

            data_min, data_max = df_sem_outliers['timestamp'].min().date(), df_sem_outliers['timestamp'].max().date()
            data_inicio = st.sidebar.date_input('Data de Início', data_min, min_value=data_min, max_value=data_max)
            data_fim = st.sidebar.date_input('Data de Fim', data_max, min_value=data_min, max_value=data_max)
            df_filtrado = df_sem_outliers[(df_sem_outliers['timestamp'].dt.date >= data_inicio) & (df_sem_outliers['timestamp'].dt.date <= data_fim)].copy()

            st.sidebar.subheader("Análises para Gerar")
            opcoes = {
                "Resumo Estatístico": st.sidebar.checkbox("Resumo Estatístico", value=True),
                "Análise Diária": st.sidebar.checkbox("Análise Diária (Temp/Umid/Pressão)", value=True),
                "Distribuição de Weibull": st.sidebar.checkbox("Distribuição de Weibull (Vento)"),
                "Rosa dos Ventos": st.sidebar.checkbox("Rosa dos Ventos")
            }
            
            st.sidebar.markdown("---")
            st.sidebar.subheader("Filtros de Vento (m/s)")
            df_vento_completo = df_filtrado[df_filtrado['velocidade_ms'] > 0.1]['velocidade_ms'].dropna()
            if len(df_vento_completo) >= 20:
                vento_min_val, vento_max_val = float(df_vento_completo.min()), float(df_vento_completo.max())
                vento_selecionado_ms = st.sidebar.slider(
                    "Refinar faixa de velocidade para análise de vento:", 
                    value=(vento_min_val, vento_max_val), 
                    min_value=vento_min_val, 
                    max_value=vento_max_val,
                    step=0.1
                )
            else:
                st.sidebar.info("Dados de vento insuficientes no período para ativar o filtro de velocidade.")
                vento_selecionado_ms = (0, VENTO_MAX_MS)

            st.markdown(f"### 🔎 Análise para o período de **{data_inicio.strftime('%d/%m/%Y')}** a **{data_fim.strftime('%d/%m/%Y')}**")
            st.info(f"Analisando {len(df_filtrado)} registros.")
            st.markdown("---")

            if opcoes["Resumo Estatístico"]:
                st.subheader("📊 Resumo Estatístico")
                resumo_df = df_filtrado[['temperatura', 'umidade', 'pressao', 'velocidade_vento']].describe().round(2)
                st.dataframe(resumo_df)

            if opcoes["Análise Diária"]:
                st.subheader("📈 Análise Diária Agregada")
                df_diario = df_filtrado.set_index('timestamp').resample('D').agg(temp_media=('temperatura', 'mean'), temp_max=('temperatura', 'max'), temp_min=('temperatura', 'min'), umid_media=('umidade', 'mean'), pressao_media=('pressao', 'mean')).dropna()
                if len(df_diario) >= 2:
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_trace(go.Scatter(x=df_diario.index, y=df_diario['temp_media'], name='Temp. Média', line=dict(color='orangered')), secondary_y=False)
                    fig.add_trace(go.Scatter(x=df_diario.index, y=df_diario['umid_media'], name='Umid. Média', line=dict(color='royalblue')), secondary_y=False)
                    if 'pressao_media' in df_diario.columns and not df_diario['pressao_media'].dropna().empty:
                        fig.add_trace(go.Scatter(x=df_diario.index, y=df_diario['pressao_media'], name='Pressão Média', line=dict(color='lightgreen', dash='dot')), secondary_y=True)
                    fig.update_layout(template="plotly_dark", title_text='Análise Diária de Temp, Umid e Pressão')
                    fig.update_xaxes(title_text="Data", tickformat="%d/%m/%Y")
                    fig.update_yaxes(title_text="<b>Temp (°C)</b> / <b>Umid (%)</b>", secondary_y=False)
                    if 'pressao' in df_diario.columns: fig.update_yaxes(title_text="<b>Pressão (hPa)</b>", secondary_y=True, showgrid=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Selecione pelo menos 2 dias de dados para a análise diária.")
            
            if opcoes["Distribuição de Weibull"]:
                st.subheader("💨 Distribuição de Weibull (Vento)")
                df_vento_analise = df_vento_completo[df_vento_completo.between(vento_selecionado_ms[0], vento_selecionado_ms[1])]
                if len(df_vento_analise) >= 20:
                    shape_k, loc, scale_c = stats.weibull_min.fit(df_vento_analise, floc=0)
                    col1, col2 = st.columns(2)
                    col1.metric("Parâmetro de Forma (k)", f"{shape_k:.2f}")
                    col2.metric("Parâmetro de Escala (c)", f"{scale_c:.2f} m/s")
                    fig_weibull = go.Figure()
                    hist_data, bin_edges = np.histogram(df_vento_analise, bins=30, density=True)
                    fig_weibull.add_trace(go.Bar(x=bin_edges[:-1], y=hist_data, name='Frequência Real', marker_color='#00bcd4'))
                    x_axis = np.linspace(0, df_vento_analise.max() * 1.1, 100)
                    pdf_fitted = stats.weibull_min.pdf(x_axis, shape_k, scale=scale_c)
                    fig_weibull.add_trace(go.Scatter(x=x_axis, y=pdf_fitted, mode='lines', name='Curva de Weibull', line=dict(color='red')))
                    fig_weibull.update_layout(template="plotly_dark", title_text="Distribuição de Weibull vs. Dados Reais")
                    st.plotly_chart(fig_weibull, use_container_width=True)
                else:
                    st.warning("A faixa de vento selecionada não contém dados suficientes para a análise.")

            if opcoes["Rosa dos Ventos"]:
                st.subheader("🧭 Rosa dos Ventos")
                df_rosa_base = df_filtrado[df_filtrado['velocidade_vento'] > 0.1].copy()
                df_rosa_base['velocidade_ms_rosa'] = df_rosa_base['velocidade_vento'] * 0.27778
                df_rosa = df_rosa_base[df_rosa_base['velocidade_ms_rosa'].between(vento_selecionado_ms[0], vento_selecionado_ms[1])]
                if len(df_rosa) >= 20:
                    mapa_direcao_graus = {'N': 0, 'NE': 45, 'E': 90, 'SE': 135, 'S': 180, 'SW': 225, 'W': 270, 'NW': 315}
                    df_rosa['direcao_graus'] = df_rosa['direcao_vento'].map(mapa_direcao_graus).dropna()
                    fig_rosa = plt.figure(figsize=(6, 6), facecolor='#1a1c20')
                    ax = WindroseAxes.from_ax(fig=fig_rosa)
                    ax.bar(df_rosa['direcao_graus'], df_rosa['velocidade_vento'], normed=True, opening=0.8, edgecolor='white')
                    ax.set_legend(title="Velocidade (km/h)", prop={'size': 'small'}, facecolor='#25282e', labelcolor='white')
                    plt.setp(ax.get_xticklabels(), color="white"); plt.setp(ax.get_yticklabels(), color="white")
                    st.pyplot(fig_rosa)
                    plt.close(fig_rosa)
                else:
                    st.warning("Dados de vento e direção insuficientes para gerar a Rosa dos Ventos.")

            st.sidebar.markdown("---")
            st.sidebar.subheader("📥 Relatório Consolidado")
            if st.sidebar.button("Preparar Relatório PDF"):
                with st.spinner("Gerando relatório PDF... Por favor, aguarde."):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        imagens_a_incluir = {}
                        resumo_df_pdf = pd.DataFrame()
                        
                        if opcoes["Resumo Estatístico"]:
                            resumo_df_pdf = df_filtrado[['temperatura', 'umidade', 'pressao', 'velocidade_vento']].describe().round(2)
                            imagens_a_incluir['Resumo Estatístico'] = 'resumo.png'
                        
                        if opcoes["Análise Diária"]:
                            df_diario_pdf = df_filtrado.set_index('timestamp').resample('D').agg(temp_media=('temperatura', 'mean'), temp_max=('temperatura', 'max'), temp_min=('temperatura', 'min'), umid_media=('umidade', 'mean')).dropna()
                            if len(df_diario_pdf) >= 2:
                                fig, ax1 = plt.subplots(figsize=(12, 6))
                                ax1.plot(df_diario_pdf.index, df_diario_pdf['temp_media'], color='orangered', marker='o', label='Temp. Média')
                                ax1.fill_between(df_diario_pdf.index, df_diario_pdf['temp_min'], df_diario_pdf['temp_max'], color='lightcoral', alpha=0.3, label='Range Temp.')
                                ax1.set_ylabel('Temperatura (°C)', color='orangered'); ax1.tick_params(axis='y', labelcolor='orangered'); ax1.grid(True)
                                ax2 = ax1.twinx()
                                ax2.plot(df_diario_pdf.index, df_diario_pdf['umid_media'], color='royalblue', marker='.', linestyle='--', label='Umidade Média')
                                ax2.set_ylabel('Umidade Média (%)', color='royalblue'); ax2.tick_params(axis='y', labelcolor='royalblue')
                                plt.title('Análise Diária de Temperatura e Umidade'); fig.legend()
                                plt.savefig(os.path.join(temp_dir, 'analise_diaria.png'), dpi=150, bbox_inches='tight')
                                plt.close(fig)
                                imagens_a_incluir['Análise Diária'] = 'analise_diaria.png'
                        
                        if opcoes["Distribuição de Weibull"]:
                            df_vento_pdf_base = df_filtrado[df_filtrado['velocidade_ms'] > 0.1]['velocidade_ms'].dropna()
                            df_vento_pdf = df_vento_pdf_base[df_vento_pdf_base.between(vento_selecionado_ms[0], vento_selecionado_ms[1])]
                            if len(df_vento_pdf) >= 20:
                                shape_k, loc, scale_c = stats.weibull_min.fit(df_vento_pdf, floc=0)
                                plt.figure(figsize=(10, 6))
                                plt.hist(df_vento_pdf, bins=30, density=True, color='deepskyblue', edgecolor='black', alpha=0.7, label='Frequência Real')
                                x_axis = np.linspace(0, df_vento_pdf.max() * 1.1, 200)
                                pdf_fitted = stats.weibull_min.pdf(x_axis, shape_k, scale=scale_c)
                                plt.plot(x_axis, pdf_fitted, 'r-', lw=2, alpha=0.8, label=f'Ajuste (k={shape_k:.2f}, c={scale_c:.2f})')
                                plt.xlabel('Velocidade do Vento (m/s)'); plt.ylabel('Densidade de Probabilidade')
                                plt.title('Distribuição de Weibull para Velocidade do Vento'); plt.legend(); plt.grid(True)
                                plt.savefig(os.path.join(temp_dir, 'weibull_distribuicao.png'), dpi=150, bbox_inches='tight')
                                plt.close()
                                imagens_a_incluir['Distribuição de Weibull'] = 'weibull_distribuicao.png'

                        if opcoes["Rosa dos Ventos"]:
                             df_rosa_pdf_base = df_filtrado[df_filtrado['velocidade_vento'] > 0.1].copy()
                             df_rosa_pdf_base['velocidade_ms_rosa'] = df_rosa_pdf_base['velocidade_vento'] * 0.27778
                             df_rosa_pdf = df_rosa_pdf_base[df_rosa_pdf_base['velocidade_ms_rosa'].between(vento_selecionado_ms[0], vento_selecionado_ms[1])]
                             if len(df_rosa_pdf) >= 20:
                                 mapa_direcao_graus = {'N': 0, 'NE': 45, 'E': 90, 'SE': 135, 'S': 180, 'SW': 225, 'W': 270, 'NW': 315}
                                 df_rosa_pdf['direcao_graus'] = df_rosa_pdf['direcao_vento'].map(mapa_direcao_graus).dropna()
                                 fig_rosa_pdf = plt.figure(figsize=(8, 8))
                                 ax_rosa = WindroseAxes.from_ax(fig=fig_rosa_pdf)
                                 ax_rosa.bar(df_rosa_pdf['direcao_graus'], df_rosa_pdf['velocidade_vento'], normed=True, opening=0.8, edgecolor='white')
                                 ax_rosa.set_legend(title="Velocidade (km/h)")
                                 plt.title('Rosa dos Ventos')
                                 plt.savefig(os.path.join(temp_dir, 'rosa_dos_ventos.png'), dpi=150, bbox_inches='tight')
                                 plt.close(fig_rosa_pdf)
                                 imagens_a_incluir['Rosa dos Ventos'] = 'rosa_dos_ventos.png'
                        
                        caminho_pdf = criar_relatorio_pdf(temp_dir, resumo_df_pdf, data_inicio, data_fim, imagens_a_incluir)
                        
                        with open(caminho_pdf, "rb") as f:
                            st.session_state.pdf_bytes_download = f.read()
            
            if "pdf_bytes_download" in st.session_state:
                st.sidebar.download_button(
                    label="Clique para Baixar o PDF",
                    data=st.session_state.pdf_bytes_download,
                    file_name=f"Relatorio_{data_inicio.strftime('%d-%m-%Y')}_{data_fim.strftime('%d-%m-%Y')}.pdf",
                    mime="application/pdf"
                )
        else:
            st.warning("Nenhum dado encontrado no Firebase.")
    else:
        st.error("Falha na conexão com o Firebase.")
else:
    st.info("⬆️ Acesso restrito. Por favor, digite a senha na barra lateral para continuar.")