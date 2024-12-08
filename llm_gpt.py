import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from stqdm import stqdm
import matplotlib.pyplot as plt  # Para gráficos gerados pela LLM

# LLM
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Função para consultar a LLM
def query_llm_analysis(df: pd.DataFrame, prompt: str):
    model = "llama3-70b-8192"
    chat = ChatGroq(model=model)

    column_info = ", ".join([f"{col} ({dtype})" for col, dtype in zip(df.columns, df.dtypes)])
    sample_data = df.head(5).to_dict(orient="records")
    
    full_prompt = f"""
Você é um analista de dados. Aqui estão as informações do DataFrame:

Colunas e tipos:
{column_info}

Exemplos de dados:
{sample_data}

Agora, baseado nessas informações, gere o código Python para realizar a seguinte análise:
{prompt}

O código deve ser funcional e conter apenas o código Python necessário. Não inclua explicações ou texto adicional, apenas o código.
"""
    response = chat.invoke(full_prompt).content
    return response

# Função para limpar o código gerado
def clean_code(code: str) -> str:
    if code.startswith("```"):
        code = code.replace("```python", "").replace("```", "")
    return code.strip()

def main():
    load_dotenv()
    st.title("Agente de IA")
    st.header("Analista de Sentimentos de Texto.")

    # Upload do arquivo
    uploaded_file = st.file_uploader("Escolha um arquivo", type=["csv"])
   
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            if "processed_df" not in st.session_state:
                st.session_state.processed_df = df

            df = st.session_state.processed_df

            st.write("## Dataframe")
            st.write(df.head())

            column_analise = st.text_input("Qual o nome da coluna que deseja analisar?")
            
            if column_analise:
                if column_analise not in df.columns:
                    st.error("A coluna especificada não existe no arquivo enviado.")
                    return

                if "Sentimento" not in df.columns:  # Processa sentimentos uma única vez
                    template = """Você é um analista de dados, trabalhando em um projeto de análise de sentimentos de textos.
Seu trabalho é escolher o sentimento dos textos entre as seguintes opções:
- POS, caso o texto seja positivo
- NEG, caso o texto seja negativo
- NEU, caso o texto seja neutro

Escolha o sentimento deste texto:
{text}

Responda apenas com uma das opções: POS, NEG ou NEU.
"""
                    prompt = PromptTemplate.from_template(template=template)
                    model = "llama3-70b-8192"
                    chat = ChatGroq(model=model)

                    chain = prompt | chat

                    category = []

                    for description in stqdm(df[column_analise], desc="Processando sentimentos..."):
                        aux_category = chain.invoke(description).content
                        category.append(aux_category)

                    df["Sentimento"] = category
                    st.session_state.processed_df = df  # Atualiza no session_state
                    st.write("### Sentimentos processados:")
                    st.write(df.head())

                csv = df.to_csv(index=False)
                st.download_button(
                    label="Baixar DataFrame como CSV",
                    data=csv,
                    file_name="dataframe_sentimento.csv",
                    mime="text/csv",
                )

                st.write("### Solicite uma análise personalizada")
                analysis_request = st.text_input("Descreva o tipo de análise que deseja realizar no DataFrame:")

                if analysis_request:
                    if "analysis_response" not in st.session_state:
                        st.session_state.analysis_response = query_llm_analysis(df, analysis_request)
                    
                    analysis_response = st.session_state.analysis_response
                    st.write("### Código Gerado pela LLM:")
                    st.code(analysis_response, language="python")

                    clean_analysis_response = clean_code(analysis_response)

                    if st.button("Executar Código Gerado"):
                        try:
                            local_namespace = {"df": df, "plt": plt, "st": st}
                            
                            if not clean_analysis_response.strip():
                                st.error("O código gerado pela LLM está vazio.")
                                return
                            
                            exec(clean_analysis_response, {}, local_namespace)

                            modified_keys = [key for key in local_namespace if key not in {"df", "plt", "st","pd"}]

                            if modified_keys:
                                for key in modified_keys:
                                    st.write("### Resultado:")
                                    st.write(local_namespace[key])
                            else:
                                st.warning("Nenhum resultado foi gerado pelo código executado.")

                            if "plt" in clean_analysis_response:
                                st.pyplot(plt.gcf())
                        
                        except SyntaxError as se:
                            st.error(f"Erro de Sintaxe no código gerado: {se}")
                        except Exception as e:
                            st.error(f"Erro ao executar o código gerado: {e}")

        except Exception as e:
            st.error(f"Ocorreu um erro ao processar o arquivo: {e}")

if __name__ == "__main__":
    main()
