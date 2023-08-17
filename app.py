import re
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
# import openai
from langchain.utilities import SQLDatabase
from langchain import PromptTemplate
# このAPIは デフォルトでtext-davinci-003 (fine-tuning model)モデルを使っていて、
# gpt-3.5-turboよりコストが高いので、使うときは要注意
# inputは20倍 outputは60倍
# from langchain.llms import OpenAI
from langchain_experimental.sql import SQLDatabaseChain
# from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains  import LLMChain
# from langchain.chains  import LLMChain, SimpleSequentialChain, ConversationChain
# from langchain.prompts import (
#     ChatPromptTemplate,
#     SystemMessagePromptTemplate,
#     HumanMessagePromptTemplate,
#     MessagesPlaceholder,
# )
from langchain.prompts import MessagesPlaceholder
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

app = Flask(__name__)
CORS(app)
app.debug = True

# system_template=""" あなたは、質問者からの質問を{language}で回答するAIです。
# 会話履歴: {history}
# """
# human_template="{question}"
# chat_prompt = ChatPromptTemplate(
#   messages=[
#     SystemMessagePromptTemplate.from_template(system_template),
#     # The `variable_name` here is what must align with memory
#     MessagesPlaceholder(variable_name="history"),
#     HumanMessagePromptTemplate.from_template(human_template),
#   ]
# )

# もしchainでmemoryを使う場合は、promptではinput_variablesは一つしかつかえない
# https://github.com/langchain-ai/langchain/issues/2013
chat_template = """
# 命令文:
1.与えられた質問の内容にOAILという単語が含まれている場合は、質問内容の最後に[forDB]を追加して答えとして返してください。
2.1以外の場合は以下の情報を参考にして、与えられた質問に対し、日本語で回答してください。ただし、情報が空の場合は、無視してください。

# 情報:
{context}

# 質問:
{question}

# 答え:
"""
# chat_prompt = PromptTemplate(template=chat_template, input_variables=["chat_history", "context", "question"])
chat_prompt = PromptTemplate(template=chat_template, input_variables=["context", "question"])

db = SQLDatabase.from_uri("postgresql://user:password@oail-postgresql:5432/oail")
chat_ai = ChatOpenAI(
     # デフォルトが gpt-3.5-turbo
     model_name="gpt-3.5-turbo",

     # 生成するトークンの最大数を指定します。GPT-4モデルの上限は8192トークン。
     max_tokens=1000,

     # デフォルトは1。サンプリング温度は0～2の間で指定します。0.8のような高い値は出力をよりランダムにし、0.2のような低い値は出力をより集中させて、決定論的にします。
     temperature=0.5,

     # デフォルトは1。温度によるサンプリングを代替する核サンプリングと呼ばれるもので、
     # 確率がtop_pのトークンの結果をモデルが考慮します。つまり、0.1は、確率が上位10%のトークンだけを考慮することを意味します。
     # 一般的には、この値か温度のどちらか一方のみを変更することをお勧めします。
     # top_p=0.95,

     # デフォルトは0。-2.0から2.0の間の数値。
     # 正の値は、これまでのテキストにおける頻度に基づいて新しいトークンにペナルティを与え、モデルが同じ行をそのまま繰り返す可能性を減少させます。
     # frequency_penalty=0,

     # デフォルトは0。-2.0から2.0の間の数値。
     # 正の値は、新しいトークンがこれまでのテキストに出現したトークンを繰り返すとペナルティを与え、新しいトピックについて話す可能性を高めます。
     # presence_penalty=0,
     verbose=True
)

# DB chainはagentのほうがもっとやりやすっぽい
# memoryも使える
# https://python.langchain.com/docs/use_cases/sql
db_chain = SQLDatabaseChain.from_llm(chat_ai, db, verbose=True)
tools = [
    Tool(
        name="OAIL-DB",
        func=db_chain.run,
        description="useful for when you need to answer questions about OAIL.",
    ),
]
db_agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
}
db_chain_memory = ConversationBufferMemory(memory_key="memory", return_messages=True)
db_agent = initialize_agent(
    tools,
    chat_ai,
    agent=AgentType.OPENAI_FUNCTIONS,
    agent_kwargs=db_agent_kwargs,
    memory=db_chain_memory,
    verbose=True,
)

# chat_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chat_chain = LLMChain(
  llm=chat_ai,
  prompt=chat_prompt,
  # memory=chat_memory,
  verbose=True,
)

# PDFをloadしてembedding(ベクトル化)してvectorstoreに保存する
loader = PyPDFLoader("https://blog.freelance-jp.org/wp-content/uploads/2023/03/FreelanceSurvey2023.pdf")
text_splitter = CharacterTextSplitter(
    separator = "\n\n",
    chunk_size = 1000,
    chunk_overlap = 0,
    length_function = len,
)
vector_index = VectorstoreIndexCreator(
    vectorstore_cls=Chroma,
    embedding=OpenAIEmbeddings(),
    text_splitter=text_splitter,
).from_loaders([loader])

# ConversationChainは、デフォルトプロンプトをオーバーライドすることができないみたい
# https://github.com/langchain-ai/langchain/issues/1800
# conversation_with_summary = ConversationChain(
#     llm=chat_ai, 
#     memory=ConversationSummaryMemory(
#       llm=OpenAI(model_name="gpt-3.5-turbo"),
#     ),
#     verbose=True
# )

# chains = SimpleSequentialChain(
#     chains=[conversation_with_summary, db_chain],
#     verbose=True,
#     memory=ConversationSummaryMemory(
#      llm=OpenAI(model_name="gpt-3.5-turbo"),
#     ),
# )

@app.route('/app/', methods=['POST'])
def chat_ai():
    data = request.json
    message = data.get('user_input')
    context = vector_index.query(message)
    if re.search("I don't know", str(context), re.IGNORECASE):
      context = ""
    qa_res = chat_chain.run(context=context, question=message)
    if re.search("forDB", str(qa_res), re.IGNORECASE):
        db_message = str(qa_res).replace("[forDB]", "").replace("OAIL", "")
        res = db_chain.run(db_message)
        return jsonify({"response": res})
    return jsonify({"response": qa_res})

    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages = [
    #         {"role": "user", "content": prompt},
    #     ],
    #     temperature=0.9,
    #     max_tokens=1000,
    #     top_p=0.95,
    #     frequency_penalty=0,
    #     presence_penalty=0,
    #     stop=None
    # )
    # return jsonify({"response": response['choices'][0]['message']['content']})

@app.route('/')
def index():
	return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
