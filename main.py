import streamlit as st
from agent import executar_agente
import os

# ==============================================
# 🔐 Configurar chave da API do Gemini
# ==============================================
# A chave vem automaticamente de st.secrets
if "GOOGLE_API_KEY" in st.secrets["general"]:
    os.environ["GOOGLE_API_KEY"] = st.secrets["general"]["GOOGLE_API_KEY"]
else:
    st.error("⚠️ A chave GOOGLE_API_KEY não foi encontrada em secrets.toml.")
    st.stop()

# ==============================================
# 🎯 Configurações iniciais do app
# ==============================================
st.set_page_config(page_title="Agente EDA Gemini - WSC", layout="centered")
st.title("🤖 Agente Autônomo EDA - WSC (CSV)")
st.write("Este agente realiza análise exploratória de dados com IA (LangChain + Gemini).")

# ==============================================
# 📂 Upload do CSV
# ==============================================
st.subheader("📂 Carregar Dataset (opcional)")
uploaded_file = st.file_uploader("Escolha um arquivo CSV", type=["csv"])

if uploaded_file:
    with open("/tmp/dataset.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("✅ Dataset carregado com sucesso!")
else:
    st.info("💡 Se não enviar um arquivo, o agente usará um dataset interno padrão.")

# ==============================================
# 💬 Chat com o Agente
# ==============================================
st.subheader("💬 Converse com o Agente")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

prompt = st.text_input("Digite um comando (ex: estatisticas_descritivas, gerar_histograma, detectar_outliers...)")

if st.button("Enviar") and prompt:
    with st.spinner("🧠 Pensando..."):
        resposta = executar_agente(prompt)
        st.session_state.chat_history.append(("Você", prompt))
        st.session_state.chat_history.append(("Agente", resposta))

# ==============================================
# 🧠 Histórico de Conversa
# ==============================================
if st.session_state.chat_history:
    st.subheader("🧠 Histórico")
    for remetente, mensagem in st.session_state.chat_history:
        if remetente == "Você":
            st.markdown(f"**🧑 {remetente}:** {mensagem}")
        else:
            st.markdown(f"**🤖 {remetente}:** {mensagem}")

# ==============================================
# 📄 Geração de Relatório
# ==============================================
st.subheader("📄 Gerar Relatório Final")
if st.button("Gerar Relatório Completo"):
    resposta = executar_agente("gerar_relatorio_completo")
    st.success("✅ Relatório gerado!")
    st.write(resposta)

    # Mostrar botão de download se o PDF existir
    caminho_pdf = "/tmp/Agentes_Autonomos_Relatorio_Atividade_Extra.pdf"
    if os.path.exists(caminho_pdf):
        with open(caminho_pdf, "rb") as f:
            st.download_button(
                label="📥 Baixar Relatório PDF",
                data=f,
                file_name="Agentes_Autonomos_Relatorio_Atividade_Extra.pdf",
                mime="application/pdf"
)
