import streamlit as st
import pandas as pd
import json
import io
import unicodedata
from datetime import datetime
from openai import OpenAI

# ─────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Análise Jornada NPS",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6B7280;
        margin-bottom: 1.5rem;
    }
    .detrator-badge {
        background: #FEE2E2;
        color: #991B1B;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .neutro-badge {
        background: #FEF3C7;
        color: #92400E;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .promotor-badge {
        background: #D1FAE5;
        color: #065F46;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* OS Number box - big and copyable */
    .os-number-box {
        background: linear-gradient(135deg, #1E3A5F 0%, #2563EB 100%);
        color: white;
        padding: 18px 24px;
        border-radius: 12px;
        text-align: center;
        margin: 8px 0;
    }
    .os-number-box .os-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.85;
        margin-bottom: 4px;
    }
    .os-number-box .os-value {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: 2px;
        font-family: 'Courier New', monospace;
    }
    .os-number-box .os-hint {
        font-size: 0.7rem;
        opacity: 0.7;
        margin-top: 6px;
    }

    /* Step indicator */
    .step-indicator {
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 12px 0 8px 0;
    }
    .step-number {
        background: #2563EB;
        color: white;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.85rem;
        flex-shrink: 0;
    }
    .step-text {
        font-size: 1rem;
        font-weight: 600;
        color: #1E3A5F;
    }

    /* Paste area highlight */
    .paste-area-wrapper {
        border: 2px dashed #2563EB;
        border-radius: 12px;
        padding: 16px;
        background: #EFF6FF;
        margin: 8px 0;
    }
    .paste-instructions {
        background: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0 12px 0;
        font-size: 0.85rem;
        color: #92400E;
    }

    /* Case info card */
    .case-info-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
    }
    .case-info-card .info-label {
        font-size: 0.72rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .case-info-card .info-value {
        font-size: 0.95rem;
        font-weight: 600;
        color: #1E293B;
    }

    div[data-testid="stExpander"] {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────
def normalize_text(text):
    if not isinstance(text, str):
        text = str(text)
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_text = nfkd.encode("ascii", "ignore").decode("ascii")
    return ascii_text.lower().strip()


def find_sheet(sheet_names, *candidates):
    for candidate in candidates:
        if candidate in sheet_names:
            return candidate
    for candidate in candidates:
        nc = normalize_text(candidate)
        for sheet in sheet_names:
            if normalize_text(sheet) == nc:
                return sheet
    for candidate in candidates:
        nc = normalize_text(candidate)
        for sheet in sheet_names:
            ns = normalize_text(sheet)
            if nc in ns or ns in nc:
                return sheet
    return None


def find_column(df_columns, *candidates):
    for candidate in candidates:
        if candidate in df_columns:
            return candidate
    for candidate in candidates:
        nc = normalize_text(candidate)
        for col in df_columns:
            if normalize_text(col) == nc:
                return col
    for candidate in candidates:
        nc = normalize_text(candidate)
        for col in df_columns:
            if nc in normalize_text(col):
                return col
    return None


def get_openai_client():
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        return OpenAI(api_key=api_key)
    except Exception:
        return None


# ─────────────────────────────────────────────────
# Column Mapper
# ─────────────────────────────────────────────────
class ColumnMapper:
    def __init__(self, df):
        cols = df.columns.tolist()
        self.pedido = find_column(cols, "Pedido", "OrderId", "PESCOD", "pedido")
        self.cliente = find_column(cols, "Cliente ", "Cliente", "Nome", "nome")
        self.afiliado = find_column(cols, "Afiliado", "afiliado")
        self.jornada = find_column(cols, "Nome  Jornada", "Nome Jornada", "Jornada")
        self.nota = find_column(cols, "Nota:", "Nota", "nota")
        self.classificacao = find_column(cols, "Classificação Nota", "Classificacao Nota", "Classificação", "Classificacao")
        self.comentario = find_column(cols, "Comentário", "Comentario", "comentario")
        self.motivo_pesq_1 = find_column(cols, "Motivo Pesquisa 1", "Motivo 1", "motivo_pesquisa_1")
        self.motivo_pesq_2 = find_column(cols, "Motivo Pesquisa 2", "Motivo 2", "motivo_pesquisa_2")
        self.sro = find_column(cols, "SRO", "sro")
        self.tipo_sro = find_column(cols, "Tipo do SRO", "Tipo SRO", "tipo_sro")
        self.cidade = find_column(cols, "CIDADE", "Cidade")
        self.estado = find_column(cols, "ESTADO", "Estado")

        # Output columns
        self.motivo_1 = find_column(cols, "Motivo 1") or "Motivo 1"
        self.motivo_2 = find_column(cols, "Motivo 2") or "Motivo 2"
        self.motivo_3 = find_column(cols, "Motivo 3") or "Motivo 3"
        self.motivo_4 = find_column(cols, "Motivo 4") or "Motivo 4"
        self.notas_caso = find_column(cols, "Notas sobre o caso", "Notas") or "Notas sobre o caso"
        self.acao_1 = find_column(cols, "Ação 1", "Acao 1") or "Ação 1"
        self.acao_2 = find_column(cols, "Ação 2", "Acao 2") or "Ação 2"
        self.acao_3 = find_column(cols, "Ação 3", "Acao 3") or "Ação 3"
        self.acao_4 = find_column(cols, "Ação 4", "Acao 4") or "Ação 4"
        self.acao_5 = find_column(cols, "Ação 5", "Acao 5") or "Ação 5"

    def get(self, row, attr, default="N/A"):
        col = getattr(self, attr, None)
        if col and col in row.index:
            val = row[col]
            if pd.notna(val):
                return val
        return default

    def get_pedido_str(self, row):
        val = self.get(row, "pedido", "?")
        if isinstance(val, (int, float)) and pd.notna(val):
            return str(int(val))
        return str(val)


# ─────────────────────────────────────────────────
# OpenAI
# ─────────────────────────────────────────────────
def build_analysis_prompt(case_info, historico, motivos_ref, acoes_ref):
    motivos_text = "\n".join(f"  - {m}" for m in motivos_ref) if motivos_ref else "  (nenhum de referência — crie motivos apropriados)"
    acoes_text = "\n".join(f"  - {a}" for a in acoes_ref) if acoes_ref else "  (nenhuma de referência — crie ações apropriadas)"

    system_prompt = f"""Você é um especialista em Qualidade e Experiência do Cliente (CX) no setor de seguros automotivos.
Analise o histórico de atendimento de um pedido/sinistro e determine:
1. Os MOTIVOS raiz do problema (de 1 a 4 motivos)
2. Notas sobre o caso (resumo analítico factual)
3. As AÇÕES corretivas recomendadas (de 1 a 5 ações)

REGRAS:
- Use PREFERENCIALMENTE motivos e ações da lista de referência.
- Se nenhum se encaixar, crie um personalizado no mesmo estilo.
- Seja específico e factual nas notas.
- Ações devem ser concretas e executáveis.
- Retorne EXCLUSIVAMENTE JSON válido, sem texto extra, sem markdown.

MOTIVOS DE REFERÊNCIA:
{motivos_text}

AÇÕES DE REFERÊNCIA:
{acoes_text}

FORMATO (JSON puro):
{{
  "motivo_1": "texto",
  "motivo_2": "texto ou null",
  "motivo_3": "texto ou null",
  "motivo_4": "texto ou null",
  "notas_caso": "resumo analítico",
  "acao_1": "texto",
  "acao_2": "texto ou null",
  "acao_3": "texto ou null",
  "acao_4": "texto ou null",
  "acao_5": "texto ou null"
}}"""

    user_prompt = f"""Analise o caso:

DADOS:
- Pedido/OS: {case_info.get('pedido', 'N/A')}
- Cliente: {case_info.get('cliente', 'N/A')}
- Afiliado: {case_info.get('afiliado', 'N/A')}
- Seguradora/Jornada: {case_info.get('jornada', 'N/A')}
- Nota NPS: {case_info.get('nota', 'N/A')}
- Classificação: {case_info.get('classificacao', 'N/A')}
- Comentário do cliente: {case_info.get('comentario', 'N/A')}
- Motivo Pesquisa 1: {case_info.get('motivo_pesq_1', 'N/A')}
- Motivo Pesquisa 2: {case_info.get('motivo_pesq_2', 'N/A')}
- Cidade/Estado: {case_info.get('cidade', 'N/A')}/{case_info.get('estado', 'N/A')}
- SRO: {case_info.get('sro', 'N/A')} | Tipo: {case_info.get('tipo_sro', 'N/A')}

HISTÓRICO COMPLETO DO PEDIDO (copiado do sistema):
{historico}

Identifique motivos raiz, escreva notas analíticas e recomende ações corretivas."""

    return system_prompt, user_prompt


def analyze_with_openai(client, system_prompt, user_prompt, model="gpt-4o-mini"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=2000,
        )
        content = response.choices[0].message.content.strip()
        if "```" in content:
            parts = content.split("```")
            for part in parts:
                clean = part.strip()
                if clean.startswith("json"):
                    clean = clean[4:].strip()
                if clean.startswith("{"):
                    content = clean
                    break
        return json.loads(content)
    except json.JSONDecodeError:
        st.error("Erro ao interpretar resposta da IA. Tente novamente.")
        st.code(content, language="json")
        return None
    except Exception as e:
        st.error(f"Erro na API: {str(e)}")
        return None


# ─────────────────────────────────────────────────
# Excel Export
# ─────────────────────────────────────────────────
def generate_output_excel(df_original, analyses, col_map):
    df = df_original.copy()

    output_cols = {
        "motivo_1": col_map.motivo_1, "motivo_2": col_map.motivo_2,
        "motivo_3": col_map.motivo_3, "motivo_4": col_map.motivo_4,
        "notas_caso": col_map.notas_caso,
        "acao_1": col_map.acao_1, "acao_2": col_map.acao_2,
        "acao_3": col_map.acao_3, "acao_4": col_map.acao_4,
        "acao_5": col_map.acao_5,
    }

    for key, col_name in output_cols.items():
        if col_name not in df.columns:
            df[col_name] = None

    for idx_str, analysis in analyses.items():
        if analysis is None:
            continue
        idx = int(idx_str)
        if idx not in df.index:
            continue
        for json_key, col_name in output_cols.items():
            val = analysis.get(json_key)
            if val and str(val).lower() != "null":
                df.at[idx, col_name] = val

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Base Analitica", index=False)
    return output.getvalue()


# ─────────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────────
if "analyses" not in st.session_state:
    st.session_state.analyses = {}
if "historicos" not in st.session_state:
    st.session_state.historicos = {}

# ─────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configurações")
    model_choice = st.selectbox(
        "Modelo OpenAI",
        ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"],
        index=0,
    )

    st.divider()
    st.markdown("### 📖 Fluxo de trabalho")
    st.markdown("""
    1. 📁 **Upload** do arquivo NPS
    2. 📄 **Selecione** as abas corretas
    3. 🔍 **Selecione** um caso (detrator)
    4. 📋 **Copie o nº do Pedido/OS**
    5. 🖥️ **No sistema**, busque a OS e copie o histórico
    6. 📝 **Cole o histórico** no campo indicado
    7. 🤖 **Clique em Analisar** com IA
    8. ✏️ **Revise e edite** os resultados
    9. 📥 **Baixe** o Excel final
    """)

    st.divider()
    if "df_base" in st.session_state and st.session_state.get("df_base") is not None:
        total = len(st.session_state.df_base)
        analyzed = len(st.session_state.analyses)
        st.metric("Progresso", f"{analyzed}/{total} analisados")
        if total > 0:
            st.progress(min(analyzed / total, 1.0))


# ─────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────
st.markdown('<div class="main-header">📊 Análise Jornada NPS — Motivos & Plano de Ação</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Análise assistida por IA para identificação de causas raiz e ações corretivas</div>', unsafe_allow_html=True)

client = get_openai_client()
if client is None:
    st.warning('⚠️ Chave da OpenAI não configurada. Em **Settings → Secrets** adicione:\n\n```\nOPENAI_API_KEY = "sk-..."\n```')

# ═══════════════════════════════════════════════════
# STEP 1: FILE UPLOAD
# ═══════════════════════════════════════════════════
st.markdown("""<div class="step-indicator">
    <span class="step-number">1</span>
    <span class="step-text">Upload do arquivo Jornada NPS</span>
</div>""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Arraste ou selecione o arquivo .xlsx",
    type=["xlsx"],
    help="Arquivo Excel com dados dos casos NPS",
    label_visibility="collapsed",
)

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    xl = pd.ExcelFile(io.BytesIO(file_bytes))
    sheet_names = xl.sheet_names

    # ═══════════════════════════════════════════════
    # STEP 2: SELECT SHEETS
    # ═══════════════════════════════════════════════
    st.markdown("""<div class="step-indicator">
        <span class="step-number">2</span>
        <span class="step-text">Confirme as abas do arquivo</span>
    </div>""", unsafe_allow_html=True)

    base_sheet_auto = find_sheet(sheet_names, "Base Analítica", "Base Analitica", "Base", "Sheet1")
    ref_sheet_auto = find_sheet(sheet_names, "Motivos e Plano de Ação", "Motivos e Plano de Acao", "Motivos")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        selected_base_sheet = st.selectbox(
            "📄 Aba com dados dos casos",
            options=sheet_names,
            index=sheet_names.index(base_sheet_auto) if base_sheet_auto and base_sheet_auto in sheet_names else 0,
        )
    with col_s2:
        has_ref = st.checkbox("Possui aba de referência (Motivos e Ações)", value=(ref_sheet_auto is not None))
        selected_ref_sheet = None
        if has_ref:
            selected_ref_sheet = st.selectbox(
                "📄 Aba de referência",
                options=sheet_names,
                index=sheet_names.index(ref_sheet_auto) if ref_sheet_auto and ref_sheet_auto in sheet_names else 0,
            )

    # Load data
    try:
        df_base = pd.read_excel(io.BytesIO(file_bytes), sheet_name=selected_base_sheet)
    except Exception as e:
        st.error(f"Erro ao ler aba '{selected_base_sheet}': {e}")
        st.stop()

    motivos_ref, acoes_ref = [], []
    if selected_ref_sheet:
        try:
            df_ref = pd.read_excel(io.BytesIO(file_bytes), sheet_name=selected_ref_sheet)
            para_col = find_column(df_ref.columns.tolist(), "PARA (Gerais)", "PARA")
            if para_col:
                motivos_ref = [str(m).strip() for m in df_ref[para_col].dropna().unique() if str(m).strip() and str(m).strip() != "---"]
            acoes_col = find_column(df_ref.columns.tolist(), "Ações", "Acoes")
            if acoes_col:
                acoes_ref = [str(a).strip() for a in df_ref[acoes_col].dropna().unique() if str(a).strip() and str(a).strip() != "---"]
        except Exception:
            pass

    st.session_state.df_base = df_base
    col_map = ColumnMapper(df_base)

    # ═══════════════════════════════════════════════
    # STEP 3: SELECT CASE
    # ═══════════════════════════════════════════════
    st.markdown("---")
    st.markdown("""<div class="step-indicator">
        <span class="step-number">3</span>
        <span class="step-text">Selecione o caso para análise</span>
    </div>""", unsafe_allow_html=True)

    # Filters
    filter_col1, filter_col2 = st.columns(2)

    classif_values = []
    if col_map.classificacao and col_map.classificacao in df_base.columns:
        classif_values = df_base[col_map.classificacao].dropna().unique().tolist()

    with filter_col1:
        class_filter = st.multiselect("Filtrar por classificação", options=classif_values,
                                       default=["Detrator"] if "Detrator" in classif_values else [])
    jornada_values = []
    if col_map.jornada and col_map.jornada in df_base.columns:
        jornada_values = df_base[col_map.jornada].dropna().unique().tolist()
    with filter_col2:
        jornada_filter = st.multiselect("Filtrar por jornada", options=jornada_values)

    df_filtered = df_base.copy()
    if class_filter and col_map.classificacao:
        df_filtered = df_filtered[df_filtered[col_map.classificacao].isin(class_filter)]
    if jornada_filter and col_map.jornada:
        df_filtered = df_filtered[df_filtered[col_map.jornada].isin(jornada_filter)]

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total filtrado", len(df_filtered))
    det_n = len(df_filtered[df_filtered[col_map.classificacao] == "Detrator"]) if col_map.classificacao and col_map.classificacao in df_filtered.columns else 0
    m2.metric("Detratores", det_n)
    m3.metric("Motivos ref.", len(motivos_ref))
    m4.metric("Ações ref.", len(acoes_ref))

    # Case selector
    case_options = []
    for idx in df_filtered.index:
        row = df_base.loc[idx]
        pedido_str = col_map.get_pedido_str(row)
        afiliado = col_map.get(row, "afiliado", "?")
        classif = col_map.get(row, "classificacao", "?")
        status = "✅" if str(idx) in st.session_state.analyses else "⬜"
        case_options.append((idx, f"{status} OS {pedido_str} — {afiliado} [{classif}]"))

    if not case_options:
        st.info("Nenhum caso encontrado. Ajuste os filtros acima.")
        st.stop()

    selected_label = st.selectbox("Selecione o caso", options=[l for _, l in case_options], index=0)
    selected_idx = next(idx for idx, l in case_options if l == selected_label)
    row = df_base.loc[selected_idx]
    hist_key = str(selected_idx)

    # ═══════════════════════════════════════════════
    # STEP 4: CASE INFO + OS NUMBER TO COPY
    # ═══════════════════════════════════════════════
    st.markdown("---")
    st.markdown("""<div class="step-indicator">
        <span class="step-number">4</span>
        <span class="step-text">Copie o número da OS e busque no sistema</span>
    </div>""", unsafe_allow_html=True)

    pedido_str = col_map.get_pedido_str(row)

    col_os, col_info = st.columns([1, 2])

    with col_os:
        # Big OS number box
        st.markdown(f"""
        <div class="os-number-box">
            <div class="os-label">Número do Pedido / OS</div>
            <div class="os-value">{pedido_str}</div>
            <div class="os-hint">📋 Copie este número e busque no sistema</div>
        </div>
        """, unsafe_allow_html=True)

        # Copy-friendly text input
        st.text_input(
            "OS (selecione e copie)",
            value=pedido_str,
            key=f"os_copy_{hist_key}",
            help="Selecione o texto neste campo e copie (Ctrl+C) para buscar no sistema",
        )

    with col_info:
        st.markdown('<div class="case-info-card">', unsafe_allow_html=True)

        ci1, ci2 = st.columns(2)
        with ci1:
            st.markdown(f"**Cliente:** {col_map.get(row, 'cliente', 'N/A')}")
            st.markdown(f"**Afiliado:** {col_map.get(row, 'afiliado', 'N/A')}")
            st.markdown(f"**Jornada:** {col_map.get(row, 'jornada', 'N/A')}")

            cidade = col_map.get(row, "cidade", "")
            estado = col_map.get(row, "estado", "")
            if cidade and str(cidade) != "N/A":
                st.markdown(f"**Local:** {cidade}/{estado}")

        with ci2:
            nota_val = col_map.get(row, "nota", "N/A")
            classif_val = col_map.get(row, "classificacao", "N/A")
            badge = "detrator-badge" if classif_val == "Detrator" else ("promotor-badge" if classif_val == "Promotor" else "neutro-badge")
            st.markdown(f"**Nota:** {nota_val} <span class='{badge}'>{classif_val}</span>", unsafe_allow_html=True)

            sro = col_map.get(row, "sro", "")
            tipo_sro = col_map.get(row, "tipo_sro", "")
            if sro and str(sro) not in ("N/A", "", "nan"):
                st.markdown(f"**SRO:** {sro} ({tipo_sro})")

            mp1 = col_map.get(row, "motivo_pesq_1", "")
            mp2 = col_map.get(row, "motivo_pesq_2", "")
            if mp1 and str(mp1) not in ("N/A", "", "nan"):
                st.markdown(f"**Motivo Pesq. 1:** {mp1}")
            if mp2 and str(mp2) not in ("N/A", "", "nan"):
                st.markdown(f"**Motivo Pesq. 2:** {mp2}")

        st.markdown('</div>', unsafe_allow_html=True)

        # Show comment
        comentario = col_map.get(row, "comentario", "")
        if comentario and str(comentario) not in ("N/A", "", "nan"):
            st.info(f"💬 **Comentário do cliente:** {comentario}")

    # ═══════════════════════════════════════════════
    # STEP 5: PASTE HISTORY
    # ═══════════════════════════════════════════════
    st.markdown("---")
    st.markdown("""<div class="step-indicator">
        <span class="step-number">5</span>
        <span class="step-text">Cole o histórico do pedido copiado do sistema</span>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class="paste-instructions">
        <strong>📋 Instruções:</strong><br>
        1. Copie o número <strong>{pedido_str}</strong> acima<br>
        2. Abra o sistema e busque esta OS<br>
        3. Copie todo o histórico do pedido (atendimentos, ligações, movimentações, anotações)<br>
        4. Cole no campo abaixo
    </div>""", unsafe_allow_html=True)

    default_hist = st.session_state.historicos.get(hist_key, "")

    st.markdown('<div class="paste-area-wrapper">', unsafe_allow_html=True)

    historico_text = st.text_area(
        f"📝 Histórico do Pedido/OS {pedido_str}",
        value=default_hist,
        height=350,
        key=f"hist_{hist_key}",
        placeholder=f"Cole aqui o histórico completo da OS {pedido_str} copiado do sistema...\n\n"
                    "O histórico pode incluir:\n"
                    "• Timeline de atendimentos e ligações\n"
                    "• Movimentações do pedido (abertura, vistoria, liberação, execução)\n"
                    "• Anotações dos atendentes\n"
                    "• Interações com o cliente\n"
                    "• Registros de reclamações ou SROs\n\n"
                    "Quanto mais detalhado o histórico, melhor será a análise da IA.",
    )

    st.markdown('</div>', unsafe_allow_html=True)

    st.session_state.historicos[hist_key] = historico_text

    # Word/char count
    if historico_text.strip():
        word_count = len(historico_text.split())
        st.caption(f"📊 {word_count} palavras | {len(historico_text)} caracteres colados")

    # ═══════════════════════════════════════════════
    # STEP 6: ANALYZE
    # ═══════════════════════════════════════════════
    st.markdown("---")
    st.markdown("""<div class="step-indicator">
        <span class="step-number">6</span>
        <span class="step-text">Analisar com IA e revisar resultados</span>
    </div>""", unsafe_allow_html=True)

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    with col_btn1:
        analyze_btn = st.button(
            "🤖 Analisar com IA",
            type="primary",
            use_container_width=True,
            disabled=(not historico_text.strip() or client is None),
        )
    with col_btn2:
        if hist_key in st.session_state.analyses:
            if st.button("🗑️ Limpar análise", use_container_width=True):
                del st.session_state.analyses[hist_key]
                st.rerun()

    if not historico_text.strip():
        st.warning(f"⬆️ Cole o histórico da OS **{pedido_str}** no campo acima para habilitar a análise.")

    if analyze_btn and historico_text.strip() and client:
        with st.spinner("🤖 Analisando o histórico com IA... Aguarde."):
            case_info = {
                "pedido": col_map.get(row, "pedido", "N/A"),
                "cliente": col_map.get(row, "cliente", "N/A"),
                "afiliado": col_map.get(row, "afiliado", "N/A"),
                "jornada": col_map.get(row, "jornada", "N/A"),
                "nota": col_map.get(row, "nota", "N/A"),
                "classificacao": col_map.get(row, "classificacao", "N/A"),
                "comentario": col_map.get(row, "comentario", "N/A"),
                "motivo_pesq_1": col_map.get(row, "motivo_pesq_1", "N/A"),
                "motivo_pesq_2": col_map.get(row, "motivo_pesq_2", "N/A"),
                "cidade": col_map.get(row, "cidade", "N/A"),
                "estado": col_map.get(row, "estado", "N/A"),
                "sro": col_map.get(row, "sro", "N/A"),
                "tipo_sro": col_map.get(row, "tipo_sro", "N/A"),
            }
            sys_p, usr_p = build_analysis_prompt(case_info, historico_text, motivos_ref, acoes_ref)
            result = analyze_with_openai(client, sys_p, usr_p, model=model_choice)
            if result:
                st.session_state.analyses[hist_key] = result
                st.success("✅ Análise concluída!")
                st.rerun()

    # ─────────────────────────────────────────────
    # RESULTS (editable)
    # ─────────────────────────────────────────────
    if hist_key in st.session_state.analyses:
        analysis = st.session_state.analyses[hist_key]

        st.markdown("#### 📌 Resultado da Análise")

        with st.expander("**Motivos identificados**", expanded=True):
            ed_m1 = st.text_input("Motivo 1", value=analysis.get("motivo_1") or "", key=f"m1_{hist_key}")
            ed_m2 = st.text_input("Motivo 2", value=analysis.get("motivo_2") or "", key=f"m2_{hist_key}")
            ed_m3 = st.text_input("Motivo 3", value=analysis.get("motivo_3") or "", key=f"m3_{hist_key}")
            ed_m4 = st.text_input("Motivo 4", value=analysis.get("motivo_4") or "", key=f"m4_{hist_key}")

        with st.expander("**Notas sobre o caso**", expanded=True):
            ed_notas = st.text_area("Notas", value=analysis.get("notas_caso") or "", height=120, key=f"notas_{hist_key}", label_visibility="collapsed")

        with st.expander("**Ações recomendadas**", expanded=True):
            ed_a1 = st.text_input("Ação 1", value=analysis.get("acao_1") or "", key=f"a1_{hist_key}")
            ed_a2 = st.text_input("Ação 2", value=analysis.get("acao_2") or "", key=f"a2_{hist_key}")
            ed_a3 = st.text_input("Ação 3", value=analysis.get("acao_3") or "", key=f"a3_{hist_key}")
            ed_a4 = st.text_input("Ação 4", value=analysis.get("acao_4") or "", key=f"a4_{hist_key}")
            ed_a5 = st.text_input("Ação 5", value=analysis.get("acao_5") or "", key=f"a5_{hist_key}")

        if st.button("💾 Salvar edições", key=f"save_{hist_key}"):
            st.session_state.analyses[hist_key] = {
                "motivo_1": ed_m1 or None, "motivo_2": ed_m2 or None,
                "motivo_3": ed_m3 or None, "motivo_4": ed_m4 or None,
                "notas_caso": ed_notas or None,
                "acao_1": ed_a1 or None, "acao_2": ed_a2 or None,
                "acao_3": ed_a3 or None, "acao_4": ed_a4 or None,
                "acao_5": ed_a5 or None,
            }
            st.success("✅ Edições salvas!")

    # ═══════════════════════════════════════════════
    # DOWNLOAD
    # ═══════════════════════════════════════════════
    if st.session_state.analyses:
        st.markdown("---")
        st.markdown("""<div class="step-indicator">
            <span class="step-number">7</span>
            <span class="step-text">Download do arquivo com análises</span>
        </div>""", unsafe_allow_html=True)

        analyzed_count = len(st.session_state.analyses)
        st.markdown(f"**{analyzed_count} caso(s)** analisados prontos para exportação.")

        summary_rows = []
        for idx_str, analysis in st.session_state.analyses.items():
            if analysis is None:
                continue
            idx = int(idx_str)
            if idx in df_base.index:
                r = df_base.loc[idx]
                summary_rows.append({
                    "OS/Pedido": col_map.get_pedido_str(r),
                    "Afiliado": col_map.get(r, "afiliado", "?"),
                    "Motivo 1": analysis.get("motivo_1", ""),
                    "Ação 1": analysis.get("acao_1", ""),
                })
        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        excel_bytes = generate_output_excel(df_base, st.session_state.analyses, col_map)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        st.download_button(
            label="⬇️ Baixar Excel com Análises",
            data=excel_bytes,
            file_name=f"Jornada_NPS_Analise_{timestamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.document",
            type="primary",
            use_container_width=True,
        )

else:
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color: #9CA3AF;">
        <p style="font-size: 3rem;">📂</p>
        <p style="font-size: 1.1rem;">Faça upload do arquivo <strong>Jornada NPS Porto</strong> para começar</p>
        <p style="font-size: 0.85rem;">O sistema detecta automaticamente as abas e colunas do seu arquivo</p>
    </div>
    """, unsafe_allow_html=True)
