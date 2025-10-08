"""
core.py
Configuração central do agente autônomo EDA (LangChain + Gemini)
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType

# Importa todas as ferramentas do módulo EDA
from eda_tools import (
    carregar_csv,
    estatisticas_descritivas,
    valores_nulos,
    gerar_histograma,
    gerar_mapa_correlacao,
    executar_kmeans,
    metodo_elbow,
    gerar_boxplot,
    gerar_dispersao,
    detectar_outliers,
)

# =====================================================
# 🧠 Configuração do Modelo Gemini
# =====================================================
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",   # ✅ Modelo estável e compatível
    temperature=0.3       # Controla criatividade
)

# =====================================================
# 💾 Memória de Conversa
# =====================================================
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# =====================================================
# 🧰 Lista de Ferramentas
# =====================================================
tools = [
    carregar_csv,
    estatisticas_descritivas,
    valores_nulos,
    gerar_histograma,
    gerar_mapa_correlacao,
    executar_kmeans,
    metodo_elbow,
    gerar_boxplot,
    gerar_dispersao,
    detectar_outliers,
]

# =====================================================
# 🤖 Criação do Agente
# =====================================================
agente = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# =====================================================
# 🚀 Função de Execução
# =====================================================
def executar_agente(prompt: str) -> str:
    """
    Executa o agente com base no comando fornecido pelo usuário.
    """
    try:
        resposta = agente.run(prompt)
        return resposta
    except Exception as e:
        return f"❌ Erro ao executar o agente: {e}"
