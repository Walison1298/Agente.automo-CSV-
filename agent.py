from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

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
# 🧰 Lista de ferramentas disponíveis
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
# 🧠 Configuração do modelo Gemini com LangChain
# =====================================================
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",      # ✅ Modelo suportado
    temperature=0.3          # Controla criatividade da resposta
)

# =====================================================
# 🧠 Memória de conversação (histórico)
# =====================================================
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# =====================================================
# 🤖 Criação do agente autônomo
# =====================================================
agente = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# =====================================================
# 🚀 Função de execução
# =====================================================
def executar_agente(prompt: str) -> str:
    """
    Executa o agente com o comando do usuário.
    """
    try:
        resposta = agente.run(prompt)
        return resposta
    except Exception as e:
        return f"❌ Erro ao executar o agente: {e}"
