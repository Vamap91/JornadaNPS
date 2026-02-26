import streamlit as st
import pandas as pd
import json
import io
import copy
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
    .case-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 8px;
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
    .stTextArea textarea {
        font-size: 0.85rem;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────
# OpenAI Client
# ─────────────────────────────────────────────────
def get_openai_client():
    """Initialize OpenAI client from Streamlit secrets."""
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        return OpenAI(api_key=api_key)
    except Exception:
        return None


# ─────────────────────────────────────────────────
# Reference data extraction
# ─────────────────────────────────────────────────
@st.cache_data
def load_reference_data(file_bytes):
    """Load reference motivos and ações from the 'Motivos e Plano de Ação' sheet."""
    try:
        df_ref = pd.read_excel(io.BytesIO(file_bytes), sheet_name="Motivos e Plano de Ação")

        motivos = []
        if "PARA (Gerais)" in df_ref.columns:
            motivos = (
                df_ref["PARA (Gerais)"]
                .dropna()
                .unique()
                .tolist()
            )
            motivos = [m.strip() for m in motivos if m.strip() and m.strip() != "---"]

        acoes = []
        if "Ações" in df_ref.columns:
            acoes = (
                df_ref["Ações"]
                .dropna()
                .unique()
                .tolist()
            )
            acoes = [a.strip() for a in acoes if a.strip() and a.strip() != "---"]

        return motivos, acoes
    except Exception:
        return [], []


@st.cache_data
def load_base_analitica(file_bytes):
    """Load the 'Base Analítica' sheet."""
    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name="Base Analítica")
    return df


# ─────────────────────────────────────────────────
# OpenAI Analysis
# ─────────────────────────────────────────────────
def build_analysis_prompt(case_info: dict, historico: str, motivos_ref: list, acoes_ref: list) -> str:
    """Build the system + user prompt for GPT analysis."""

    motivos_text = "\n".join(f"  - {m}" for m in motivos_ref)
    acoes_text = "\n".join(f"  - {a}" for a in acoes_ref)

    system_prompt = f"""Você é um especialista em Qualidade e Experiência do Cliente (CX) no setor de seguros automotivos.
Sua tarefa é analisar o histórico de atendimento de um pedido/sinistro e, com base nos fatos, determinar:
1. Os MOTIVOS raiz do problema (de 1 a 4 motivos)
2. Notas sobre o caso (resumo analítico)
3. As AÇÕES corretivas recomendadas (de 1 a 5 ações)

REGRAS IMPORTANTES:
- Use PREFERENCIALMENTE os motivos e ações da lista de referência abaixo.
- Se nenhum motivo/ação da lista se encaixar perfeitamente, você pode criar um personalizado, mas mantenha o mesmo estilo e formato.
- Seja específico e factual nas notas sobre o caso.
- As ações devem ser concretas e executáveis.
- Retorne EXCLUSIVAMENTE um JSON válido, sem texto adicional, sem markdown.

LISTA DE MOTIVOS DE REFERÊNCIA:
{motivos_text}

LISTA DE AÇÕES DE REFERÊNCIA:
{acoes_text}

FORMATO DE RESPOSTA (JSON puro):
{{
  "motivo_1": "texto do motivo 1",
  "motivo_2": "texto do motivo 2 ou null",
  "motivo_3": "texto do motivo 3 ou null",
  "motivo_4": "texto do motivo 4 ou null",
  "notas_caso": "resumo analítico do caso",
  "acao_1": "texto da ação 1",
  "acao_2": "texto da ação 2 ou null",
  "acao_3": "texto da ação 3 ou null",
  "acao_4": "texto da ação 4 ou null",
  "acao_5": "texto da ação 5 ou null"
}}"""

    user_prompt = f"""Analise o caso abaixo:

DADOS DO CASO:
- Pedido: {case_info.get('pedido', 'N/A')}
- Cliente: {case_info.get('cliente', 'N/A')}
- Afiliado: {case_info.get('afiliado', 'N/A')}
- Seguradora/Jornada: {case_info.get('jornada', 'N/A')}
- Nota NPS: {case_info.get('nota', 'N/A')}
- Classificação: {case_info.get('classificacao', 'N/A')}
- Comentário do cliente: {case_info.get('comentario', 'N/A')}
- Motivo Pesquisa 1: {case_info.get('motivo_pesq_1', 'N/A')}
- Motivo Pesquisa 2: {case_info.get('motivo_pesq_2', 'N/A')}

HISTÓRICO COMPLETO DO PEDIDO:
{historico}

Com base no histórico acima, identifique os motivos raiz, escreva notas analíticas e recomende ações corretivas."""

    return system_prompt, user_prompt


def analyze_with_openai(client, system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini") -> dict:
    """Call OpenAI and parse the JSON response."""
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
        # Try to extract JSON from content (handle markdown code blocks)
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        result = json.loads(content)
        return result
    except json.JSONDecodeError:
        st.error("Erro ao interpretar resposta da IA. Tente novamente.")
        st.code(content, language="json")
        return None
    except Exception as e:
        st.error(f"Erro na chamada à API: {str(e)}")
        return None


# ─────────────────────────────────────────────────
# Excel Export
# ─────────────────────────────────────────────────
def generate_output_excel(df_original: pd.DataFrame, analyses: dict) -> bytes:
    """Generate the output Excel file with analyses merged in."""
    df = df_original.copy()

    # Ensure output columns exist
    for col in ["Motivo 1", "Motivo 2", "Motivo 3", "Motivo 4",
                 "Notas sobre o caso",
                 "Ação 1", "Ação 2", "Ação 3", "Ação 4", "Ação 5"]:
        if col not in df.columns:
            df[col] = None

    # Apply analyses
    for idx, analysis in analyses.items():
        if analysis is None:
            continue
        row_idx = int(idx)
        df.at[row_idx, "Motivo 1"] = analysis.get("motivo_1")
        df.at[row_idx, "Motivo 2"] = analysis.get("motivo_2")
        df.at[row_idx, "Motivo 3"] = analysis.get("motivo_3")
        df.at[row_idx, "Motivo 4"] = analysis.get("motivo_4")
        df.at[row_idx, "Notas sobre o caso"] = analysis.get("notas_caso")
        df.at[row_idx, "Ação 1"] = analysis.get("acao_1")
        df.at[row_idx, "Ação 2"] = analysis.get("acao_2")
        df.at[row_idx, "Ação 3"] = analysis.get("acao_3")
        df.at[row_idx, "Ação 4"] = analysis.get("acao_4")
        df.at[row_idx, "Ação 5"] = analysis.get("acao_5")

    # Write to Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Base Analítica", index=False)
    return output.getvalue()


# ─────────────────────────────────────────────────
# Session State init
# ─────────────────────────────────────────────────
if "analyses" not in st.session_state:
    st.session_state.analyses = {}
if "historicos" not in st.session_state:
    st.session_state.historicos = {}

# ─────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/graph-report.png", width=48)
    st.markdown("### Configurações")

    model_choice = st.selectbox(
        "Modelo OpenAI",
        ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"],
        index=0,
        help="gpt-4o-mini é mais rápido e econômico. gpt-4o é mais preciso.",
    )

    st.divider()

    st.markdown("### Como usar")
    st.markdown("""
    1. **Faça upload** do arquivo Jornada NPS (.xlsx)
    2. **Selecione** um caso para analisar
    3. **Cole o histórico** do pedido
    4. **Clique em Analisar** para a IA preencher motivos e ações
    5. **Revise e edite** se necessário
    6. **Baixe** o arquivo completo
    """)

    st.divider()

    # Progress summary
    if "df_base" in st.session_state and st.session_state.df_base is not None:
        total = len(st.session_state.df_base)
        analyzed = len(st.session_state.analyses)
        st.metric("Casos analisados", f"{analyzed}/{total}")
        if total > 0:
            st.progress(analyzed / total)

# ─────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────
st.markdown('<div class="main-header">📊 Análise Jornada NPS — Motivos & Plano de Ação</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Análise assistida por IA para identificação de causas e ações corretivas</div>', unsafe_allow_html=True)

# Check API Key
client = get_openai_client()
if client is None:
    st.warning("⚠️ Chave da OpenAI não configurada. Vá em **Settings → Secrets** no Streamlit Cloud e adicione:\n\n```\nOPENAI_API_KEY = \"sk-...\"\n```")

# ─────────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────────
st.markdown("### 📁 Upload do Arquivo")

uploaded_file = st.file_uploader(
    "Faça upload do arquivo Jornada NPS Porto (.xlsx)",
    type=["xlsx"],
    help="Arquivo Excel com a aba 'Base Analítica' e 'Motivos e Plano de Ação'",
)

if uploaded_file is not None:
    file_bytes = uploaded_file.read()

    # Load data
    df_base = load_base_analitica(file_bytes)
    motivos_ref, acoes_ref = load_reference_data(file_bytes)

    st.session_state.df_base = df_base
    st.session_state.motivos_ref = motivos_ref
    st.session_state.acoes_ref = acoes_ref

    # ─────────────────────────────────────────────
    # OVERVIEW TABLE
    # ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Visão Geral dos Casos")

    # Identify key columns
    col_pedido = "Pedido"
    col_cliente = "Cliente "
    col_afiliado = "Afiliado"
    col_jornada = "Nome  Jornada"
    col_nota = "Nota:"
    col_class = "Classificação Nota"
    col_comentario = "Comentário"

    # Show filter
    filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 2])
    with filter_col1:
        class_filter = st.multiselect(
            "Classificação",
            options=df_base[col_class].dropna().unique().tolist(),
            default=["Detrator"] if "Detrator" in df_base[col_class].values else [],
        )
    with filter_col2:
        jornada_filter = st.multiselect(
            "Jornada / Seguradora",
            options=df_base[col_jornada].dropna().unique().tolist(),
        )

    # Apply filters
    df_filtered = df_base.copy()
    if class_filter:
        df_filtered = df_filtered[df_filtered[col_class].isin(class_filter)]
    if jornada_filter:
        df_filtered = df_filtered[df_filtered[col_jornada].isin(jornada_filter)]

    # Show summary metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total de casos", len(df_filtered))
    with m2:
        detratores = len(df_filtered[df_filtered[col_class] == "Detrator"])
        st.metric("Detratores", detratores)
    with m3:
        st.metric("Motivos referência", len(motivos_ref))
    with m4:
        st.metric("Ações referência", len(acoes_ref))

    # Display compact table
    display_cols = [col_pedido, col_afiliado, col_jornada, col_nota, col_class, col_comentario]
    display_cols = [c for c in display_cols if c in df_filtered.columns]

    st.dataframe(
        df_filtered[display_cols].reset_index(drop=True),
        use_container_width=True,
        height=250,
    )

    # ─────────────────────────────────────────────
    # CASE ANALYSIS SECTION
    # ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 Análise Individual de Caso")

    # Case selector
    case_options = []
    for idx, row in df_filtered.iterrows():
        pedido = row.get(col_pedido, "?")
        afiliado = row.get(col_afiliado, "?")
        classif = row.get(col_class, "?")
        status = "✅" if str(idx) in st.session_state.analyses else "⬜"
        case_options.append(f"{status} Pedido {int(pedido) if pd.notna(pedido) else '?'} — {afiliado} [{classif}]")

    if not case_options:
        st.info("Nenhum caso encontrado com os filtros aplicados.")
    else:
        selected_case_label = st.selectbox(
            "Selecione o caso para analisar",
            options=case_options,
            index=0,
        )
        selected_idx_in_filtered = case_options.index(selected_case_label)
        selected_original_idx = df_filtered.index[selected_idx_in_filtered]
        row = df_base.loc[selected_original_idx]

        # Display case info
        with st.container():
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                pedido_val = row.get(col_pedido, "N/A")
                st.markdown(f"**Pedido:** {int(pedido_val) if pd.notna(pedido_val) else 'N/A'}")
            with c2:
                st.markdown(f"**Afiliado:** {row.get(col_afiliado, 'N/A')}")
            with c3:
                st.markdown(f"**Jornada:** {row.get(col_jornada, 'N/A')}")
            with c4:
                nota_val = row.get(col_nota, "N/A")
                classif_val = row.get(col_class, "N/A")
                badge_class = "detrator-badge" if classif_val == "Detrator" else (
                    "promotor-badge" if classif_val == "Promotor" else "neutro-badge"
                )
                st.markdown(
                    f"**Nota:** {nota_val} <span class='{badge_class}'>{classif_val}</span>",
                    unsafe_allow_html=True,
                )

            comentario = row.get(col_comentario, "")
            if pd.notna(comentario) and str(comentario).strip():
                st.info(f"💬 **Comentário do cliente:** {comentario}")

            motivo_p1 = row.get("Motivo Pesquisa 1", "")
            motivo_p2 = row.get("Motivo Pesquisa 2", "")
            if pd.notna(motivo_p1) and str(motivo_p1).strip():
                st.markdown(f"**Motivo Pesquisa 1:** {motivo_p1}")
            if pd.notna(motivo_p2) and str(motivo_p2).strip():
                st.markdown(f"**Motivo Pesquisa 2:** {motivo_p2}")

        # Historical text input
        st.markdown("#### 📝 Histórico do Pedido")
        st.caption("Cole abaixo o histórico completo do pedido (interações, timeline, observações).")

        hist_key = str(selected_original_idx)
        default_hist = st.session_state.historicos.get(hist_key, "")

        historico_text = st.text_area(
            "Histórico do pedido",
            value=default_hist,
            height=300,
            key=f"hist_{hist_key}",
            label_visibility="collapsed",
            placeholder="Cole aqui o histórico detalhado do pedido...\n\nExemplo:\n08-01-2026 – Abertura do sinistro\nCliente aciona seguro após colisão...\n\n09-01-2026 – Agendamento\nAtendimento liberado na oficina X...",
        )

        # Save historico to session
        st.session_state.historicos[hist_key] = historico_text

        # Analysis button
        col_btn1, col_btn2 = st.columns([1, 3])
        with col_btn1:
            analyze_btn = st.button(
                "🤖 Analisar com IA",
                type="primary",
                use_container_width=True,
                disabled=(not historico_text.strip() or client is None),
            )

        if analyze_btn and historico_text.strip() and client:
            with st.spinner("Analisando com IA..."):
                case_info = {
                    "pedido": int(row[col_pedido]) if pd.notna(row.get(col_pedido)) else "N/A",
                    "cliente": row.get(col_cliente, "N/A"),
                    "afiliado": row.get(col_afiliado, "N/A"),
                    "jornada": row.get(col_jornada, "N/A"),
                    "nota": row.get(col_nota, "N/A"),
                    "classificacao": row.get(col_class, "N/A"),
                    "comentario": row.get(col_comentario, "N/A"),
                    "motivo_pesq_1": row.get("Motivo Pesquisa 1", "N/A"),
                    "motivo_pesq_2": row.get("Motivo Pesquisa 2", "N/A"),
                }

                system_prompt, user_prompt = build_analysis_prompt(
                    case_info, historico_text, motivos_ref, acoes_ref
                )
                result = analyze_with_openai(client, system_prompt, user_prompt, model=model_choice)

                if result:
                    st.session_state.analyses[hist_key] = result
                    st.success("✅ Análise concluída!")
                    st.rerun()

        # ─────────────────────────────────────────
        # SHOW / EDIT RESULTS
        # ─────────────────────────────────────────
        if hist_key in st.session_state.analyses:
            analysis = st.session_state.analyses[hist_key]

            st.markdown("---")
            st.markdown("#### 📌 Resultado da Análise")

            with st.expander("**Motivos identificados**", expanded=True):
                edited_m1 = st.text_input("Motivo 1", value=analysis.get("motivo_1", "") or "", key=f"m1_{hist_key}")
                edited_m2 = st.text_input("Motivo 2", value=analysis.get("motivo_2", "") or "", key=f"m2_{hist_key}")
                edited_m3 = st.text_input("Motivo 3", value=analysis.get("motivo_3", "") or "", key=f"m3_{hist_key}")
                edited_m4 = st.text_input("Motivo 4", value=analysis.get("motivo_4", "") or "", key=f"m4_{hist_key}")

            with st.expander("**Notas sobre o caso**", expanded=True):
                edited_notas = st.text_area(
                    "Notas",
                    value=analysis.get("notas_caso", "") or "",
                    height=120,
                    key=f"notas_{hist_key}",
                    label_visibility="collapsed",
                )

            with st.expander("**Ações recomendadas**", expanded=True):
                edited_a1 = st.text_input("Ação 1", value=analysis.get("acao_1", "") or "", key=f"a1_{hist_key}")
                edited_a2 = st.text_input("Ação 2", value=analysis.get("acao_2", "") or "", key=f"a2_{hist_key}")
                edited_a3 = st.text_input("Ação 3", value=analysis.get("acao_3", "") or "", key=f"a3_{hist_key}")
                edited_a4 = st.text_input("Ação 4", value=analysis.get("acao_4", "") or "", key=f"a4_{hist_key}")
                edited_a5 = st.text_input("Ação 5", value=analysis.get("acao_5", "") or "", key=f"a5_{hist_key}")

            # Save edits button
            if st.button("💾 Salvar edições", key=f"save_{hist_key}"):
                st.session_state.analyses[hist_key] = {
                    "motivo_1": edited_m1 if edited_m1 else None,
                    "motivo_2": edited_m2 if edited_m2 else None,
                    "motivo_3": edited_m3 if edited_m3 else None,
                    "motivo_4": edited_m4 if edited_m4 else None,
                    "notas_caso": edited_notas if edited_notas else None,
                    "acao_1": edited_a1 if edited_a1 else None,
                    "acao_2": edited_a2 if edited_a2 else None,
                    "acao_3": edited_a3 if edited_a3 else None,
                    "acao_4": edited_a4 if edited_a4 else None,
                    "acao_5": edited_a5 if edited_a5 else None,
                }
                st.success("Edições salvas!")

    # ─────────────────────────────────────────────────
    # DOWNLOAD SECTION
    # ─────────────────────────────────────────────────
    if st.session_state.analyses:
        st.markdown("---")
        st.markdown("### 📥 Download do Arquivo Completo")

        analyzed_count = len(st.session_state.analyses)
        st.markdown(f"**{analyzed_count} caso(s) analisado(s)** prontos para exportação.")

        # Show summary table of all analyses
        summary_rows = []
        for idx_str, analysis in st.session_state.analyses.items():
            idx = int(idx_str)
            if idx < len(df_base):
                r = df_base.iloc[idx]
                pedido_v = r.get(col_pedido, "?")
                summary_rows.append({
                    "Pedido": int(pedido_v) if pd.notna(pedido_v) else "?",
                    "Afiliado": r.get(col_afiliado, "?"),
                    "Motivo 1": analysis.get("motivo_1", ""),
                    "Ação 1": analysis.get("acao_1", ""),
                })
        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        # Generate and offer download
        excel_bytes = generate_output_excel(df_base, st.session_state.analyses)
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
    # Empty state
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center; padding: 60px 20px; color: #9CA3AF;">
            <p style="font-size: 3rem;">📂</p>
            <p style="font-size: 1.1rem;">Faça upload do arquivo <strong>Jornada NPS Porto</strong> para começar</p>
            <p style="font-size: 0.85rem;">O arquivo deve conter as abas "Base Analítica" e "Motivos e Plano de Ação"</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
