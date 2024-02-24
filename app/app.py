from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import os
import json
import datetime as dt
import warnings

import arxiv
import openai
from openai import OpenAI
from dotenv import load_dotenv

from duckduckgo_search import DDGS, AsyncDDGS
import asyncio

# すべての警告を無視する
warnings.filterwarnings('ignore')

from contextlib import contextmanager
from time import time

class Timer:
    """処理時間を表示するクラス
    with Timer(prefix=f'pred cv={i}'):
        y_pred_i = predict(model, loader=test_loader)
    
    with Timer(prefix='fit fold={} '.format(i)):
        clf.fit(x_train, y_train, 
                eval_set=[(x_valid, y_valid)],  
                early_stopping_rounds=100,
                verbose=verbose)

    with Timer(prefix='fit fold={} '.format(i), verbose=500):
        clf.fit(x_train, y_train, 
                eval_set=[(x_valid, y_valid)],  
                early_stopping_rounds=100,
                verbose=verbose)
    """
    def __init__(self, logger=None, format_str='{:.3f}[s]', prefix=None, suffix=None, sep=' ', verbose=0):

        if prefix: format_str = str(prefix) + sep + format_str
        if suffix: format_str = format_str + sep + str(suffix)
        self.format_str = format_str
        self.logger = logger
        self.start = None
        self.end = None
        self.verbose = verbose

    @property
    def duration(self):
        if self.end is None:
            return 0
        return self.end - self.start

    def __enter__(self):
        self.start = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time()
        out_str = self.format_str.format(self.duration)
        if self.logger:
            self.logger.info(out_str)
        else:
            print(out_str)


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['UPLOAD_FOLDER'] = 'uploads'  # アップロードされたファイルを保存するフォルダ
socketio = SocketIO(app)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
# MODEL_NAME = "gpt-3.5-turbo-0125"
# MODEL_NAME = "gpt-3.5-turbo-instruct"
MODEL_NAME = "gpt-4-0125-preview"
TEMPERATURE = 0.7
# OpenAIクライアントの初期化
client = OpenAI()

# 各ノードの名称を定義
RESEARCH_NODE = "research"
RE_RESEARCH_NODE = "re-research"
QUALITY_ASSURANCE_NODE = "quality_assurance"
WRITER_NODE = "writer"
SUPERVISOR_NODE = "supervisor"
DECISION = "decision"
SPLIT_SEAECH_DECISION = "split_search_decision"
Q_LEN_SEAECH_DECISION = 50

# エージェントに追加するシステムプロンプト作成関数
def create_agent_system(
        system_prompt: list,  # システムからエージェントへの初期プロンプト
        team_members: str,   # メンバーの役割
):
    # システムプロンプトに自律的な働きに関する指示を追加
    system_prompt.append({"role" : "system", "content" : "Work autonomously according to your specialty, using the tools available to you."})
    system_prompt.append({"role" : "system", "content" : " Do not ask for clarification."})
    system_prompt.append({"role" : "system", "content" : " Your other team members (and other teams) will collaborate with you with their own specialties."})
    system_prompt.append({"role" : "system", "content" : f" You are chosen for a reason! You are one of the following team members: {team_members}."})
    """
    あなたの専門分野に従って自律的に働いてください。使用可能なツールを使ってください
    確認のために質問をしないでください
    あなたの他のチームメンバーや他のチームも、それぞれの専門分野であなたと協力します
    あなたが選ばれたのには理由があります！あなたは以下のチームメンバーの一人です: {team_members}
    """
    # エージェントを実行するsystem_promptを返す
    return system_prompt

# Decisionノードの定義
def search_decision_node(
        model_name: str,
        input_text: str,
):
    # 検索判断エージェントを呼び出し、結果を取得
    # あなたは、input_text を受け取り、それを自分が持つ知識で絶対に正しく回答できるかを思慮深く考えて判断する決定者です。
    prompt = [{'role': 'system', 'content': "You are the decider who receives input_text and judiciously considers whether you can respond with certainty on your own."}]
    prompt = [{'role': 'system', 'content': "If you receive input_text and determine that it's better to generate an answer by conducting an external search, please decide 'Search'. If you can answer with your own knowledge and the query does not pertain to recent content, then please reply 'Not Search'."}]
    
    decision_prompt = create_agent_system(prompt, DECISION)
    decision_prompt.append({"role": "system", "content": 'Please generate a JSON from the following input text. Use "decision_result" as the schema and "decision results" as the key, and generate it in the format of {"decision_result": "decision results"}.'})
    decision_prompt.append({"role": "user", "content": 'Generate a JSON from the following input text. Use "decision_result as the schema and the decision result as the key, creating it in the format of {"decision_result": "Search or Not Search"}.'})
    decision_prompt.append({"role": "user", "content": f"Input text: {input_text}"})
    """
    システム
    あなたは、input_text を受け取り、それを自分が持つ知識で絶対に正しく回答できるかを思慮深く考えて判断する決定者です。
    あなたは、input_text を受け取り、外部検索を行い回答を生成した方が良いと判断するのなら  Search と判断してください。あなた自身の知識で回答でき、最近の内容が聞かれているのでなければ　Not Search と回答してください。
    (これは入れてません。回答できるか自信が無くても Search と回答します。) 
    
    あなたの専門分野に従って自律的に働いてください。使用可能なツールを使ってください
    確認のために質問をしないでください
    あなたの他のチームメンバーや他のチームも、それぞれの専門分野であなたと協力します
    あなたが選ばれたのには理由があります！あなたは以下のチームメンバーの一人です: {team_members}
    以下の入力されたテキストからJSONを生成してください。スキーマとして「decision_result」、キーとして「decision results」を使用し、{"decision_result": decision results}の形式で生成してください。
    user
    以下の入力されたテキストからJSONを生成する。スキーマとして "decision_result "を使用し、キーとして判断結果を使用して、{"decision_result": Search or Not Search}というフォーマットで生成します。
    入力されたテキスト: {input_text}
    """
    
    # Research用のプロンプトテンプレートを作成
    response = client.chat.completions.create(
        model=model_name, # model = "deployment_name".
        messages=decision_prompt,
        response_format={ "type": "json_object" },
        temperature=TEMPERATURE,
    )
    decision_res_str = response.choices[0].message.content
    # print(decision_res_str)
    
    # JSON形式の文字列を辞書に変換
    search_res = json.loads(decision_res_str)
    
    # 出力と新しいメッセージをステートに反映
    return {
        "output": search_res["decision_result"],
        "messages": input_text
    }

# テキスト検索用の関数
def search_text(keywords, region='wt-wt', safesearch='moderate', timelimit=None, max_results=10):
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(keywords, region=region, safesearch=safesearch, timelimit=timelimit, max_results=max_results)]
    return results

# Researchノードの定義
def research_node(
        model_name: str,
        job_result: str, # search の結果
):
    # リサーチエージェントを呼び出し、結果を取得
    # あなたは、DuckDuckGo検索エンジンを使って、検索された情報を順番に確認し、ポイントを外さずに思慮深く説明するリサーチアシスタントです。
    prompt = [{'role': 'system', 'content': "You are a research assistant who uses the DuckDuckGo search engine to review searched information in order, explaining points carefully without missing anything."}]
    prompt.append({"role": "system", "content": "Explanation results must be in Japanese."})
    
    research_prompt = create_agent_system(prompt, RESEARCH_NODE)
    research_prompt.append({"role": "system", "content": "Please generate a JSON from the text of the following search results. Use 'explanation_result' as the schema and 'explanation results' as the key, and generate it in the format of {'explanation_result': 'explanation results'}."})
    research_prompt.append({"role": "user", "content": "Generate a JSON from the text of the following search results. Use 'explanation_result' as the schema and the results of explaining the search as the key, creating it in the format of {'explanation_result': the results of explaining the search}."})
    research_prompt.append({"role": "user", "content": f"Text of search job results: {job_result}"})
    """
    システム
    あなたは、DuckDuckGo検索エンジンを使って、検索された情報を順番に確認し、ポイントを外さずに思慮深く説明するリサーチアシスタントです。
    説明結果は日本語でなければならない。
    
    あなたの専門分野に従って自律的に働いてください。使用可能なツールを使ってください
    確認のために質問をしないでください
    あなたの他のチームメンバーや他のチームも、それぞれの専門分野であなたと協力します
    あなたが選ばれたのには理由があります！あなたは以下のチームメンバーの一人です: {team_members}
    
    以下の検索結果のテキストからJSONを生成してください。スキーマとして「explanation_result」、キーとして「explanation results」を使用し、{"explanation_result": explanation results}の形式で生成してください。
    user
    以下の検索結果のテキストからJSONを生成する。スキーマとして "explanation_result "を使用し、キーとして検索を説明した結果を使用して、{"explanation_result": 検索を説明した結果}というフォーマットで生成します。
    search ジョブ結果のテキスト: {job_result}
    """
    
    # Research用のプロンプトテンプレートを作成
    response = client.chat.completions.create(
        model=model_name, # model = "deployment_name".
        messages=research_prompt,
        response_format={ "type": "json_object" },
        temperature=TEMPERATURE,
    )
    
    print("response")
    print(response)
    search_res_str = response.choices[0].message.content
    
    # JSON形式の文字列を辞書に変換
    search_res = json.loads(search_res_str)
    
    # 出力と新しいメッセージをステートに反映
    return {
        "output": search_res["explanation_result"],
        "messages": job_result
    }

# 再検索用クエリを抽出
def make_re_search_query(
    model_name: str,
    question: str,    # 最初の質問
    before_query: str,   # 直前の検索クエリ 初回は question
):
    
    # questionから検索クエリを作成する。before_queryと違うものクエリを作成する
    # テキストのブロックが提供されます。あなたのタスクは、そこからキーワードのリストを抽出することです。
    prompt = [{'role': 'system', 'content': "You will be provided with a block of text, and your task is to extract a list of keywords from it."}]
    
    prompt.append({"role": "system", "content": "Use the following example to extract a list of keywords.\n###Example###\n\nuser's question: Black-on-black ware is a 20th- and 21st-century pottery tradition developed by the Puebloan Native American ceramic artists in Northern New Mexico. Traditional reduction-fired blackware has been made for centuries by pueblo artists. Black-on-black ware of the past century is produced with a smooth surface, with the designs applied through selective burnishing or the application of refractory slip. Another style involves carving or incising designs and selectively polishing the raised areas. For generations several families from Kha'po Owingeh and P'ohwhóge Owingeh pueblos have been making black-on-black ware with the techniques passed down from matriarch potters. Artists from other pueblos have also produced black-on-black ware. Several contemporary artists have created works honoring the pottery of their ancestors.\n\nresult query: {      'query_result': 'Black-on-black ware, pottery tradition, Puebloan Native American, ceramic artists, Northern New Mexico, reduction-fired blackware, pueblo artists, smooth surface, designs, selective burnishing, refractory slip, carving, incising designs, polishing, generations, families, Kha'po Owingeh, P'ohwhóge Owingeh pueblos, matriarch potters, contemporary artists, ancestors'}"})
    """
    Use the following example to extract a list of keywords.
    ###Example###
    
    user's question: Black-on-black ware is a 20th- and 21st-century pottery tradition developed by the Puebloan Native American ceramic artists in Northern New Mexico. Traditional reduction-fired blackware has been made for centuries by pueblo artists. Black-on-black ware of the past century is produced with a smooth surface, with the designs applied through selective burnishing or the application of refractory slip. Another style involves carving or incising designs and selectively polishing the raised areas. For generations several families from Kha'po Owingeh and P'ohwhóge Owingeh pueblos have been making black-on-black ware with the techniques passed down from matriarch potters. Artists from other pueblos have also produced black-on-black ware. Several contemporary artists have created works honoring the pottery of their ancestors.
    
    result query: 
    {
      "query_result": "Black-on-black ware, pottery tradition, Puebloan Native American, ceramic artists, Northern New Mexico, reduction-fired blackware, pueblo artists, smooth surface, designs, selective burnishing, refractory slip, carving, incising designs, polishing, generations, families, Kha'po Owingeh, P'ohwhóge Owingeh pueblos, matriarch potters, contemporary artists, ancestors"
    }
    """
    prompt.append({'role': 'system', 'content': "Generate queries that are as different as possible from the query used in the previous search for extracting the list of keywords."})
    prompt.append({'role': 'system', 'content': f"The query used in the previous search was '{before_query}'."})

    prompt.append({"role": "user", "content": "Generate JSON from search result text. Use 'query_result' as the schema, generate in the format {'query_result': Result extract keywords from a block of text.}, and key in the evaluation results, such as whether the generated search results describe the user's request."})
    prompt.append({"role": "user", "content": f"user's question:{question}. Result extract keywords from a block of text.:"})
    

    """
    system
    以下の例を参考にキーワードのリストを抽出してください。
    
    ###Example###
    
    user's question: 黒地に黒の陶器は、ニューメキシコ州北部のプエブロ人ネイティブ アメリカンの陶芸家によって発展した、20 世紀から 21 世紀にかけての陶器の伝統です。伝統的な還元焼成黒食器は、プエブロの芸術家によって何世紀にもわたって作られてきました。前世紀の黒地に黒の陶器は、選択的に磨きをかけたり、耐火物スリップを塗布したりすることによってデザインが施され、滑らかな表面で製造されています。別のスタイルには、デザインを彫刻または切り込み、盛り上がった領域を選択的に研磨することが含まれます。カポ オウィンゲとポホゲ オウィンゲ プエブロの数家族が、家長の陶芸家から受け継がれた技術を用いて、黒地に黒の陶器を何世代にもわたって作り続けてきました。他のプエブロ出身の芸術家も黒地に黒の陶器を制作しています。何人かの現代芸術家は、祖先の陶器に敬意を表して作品を制作しました。
    
    result query: 
    {
      'query_result': '黒地に黒の器、陶器の伝統、プエブロのネイティブ アメリカン、陶芸家、ニューメキシコ北部、還元焼成黒器、プエブロの芸術家、滑らかな表面、デザイン、選択的バニシング、耐火物スリップ、彫刻、切り込みデザイン、研磨、世代、家族、カポ・オウィンゲ、ポホゲ・オウィンゲ・プエブロス、女家長陶芸家、現代芸術家、先祖'
    }
    
    キーワードリストを抽出するために、前回の検索で使用したクエリとはできるだけ異なるクエリを生成する。
    前回の検索で使用したクエリは「{before_query}」です。
    user
    検索結果のテキストからJSONを生成する。スキーマとして "query_result"を使用し、{"query_result"： テキストブロックからキーワードを抽出した結果}の形式で生成する。
    
    前回の検索で使用したクエリ: "{before_query}".

    
    user's question:{}. result query:
    """
    
    # Research用のプロンプトテンプレートを作成
    response = client.chat.completions.create(
        model=model_name, # model = "deployment_name".
        messages=prompt,
        response_format={ "type": "json_object" },
        temperature=TEMPERATURE+TEMPERATURE,
    )
    res_str = response.choices[0].message.content
    # print(res_str)
    
    # JSON形式の文字列を辞書に変換
    res = json.loads(res_str)
    
    # 出力と新しいメッセージをステートに反映
    return {
        "output": res["query_result"],
        "before_query": before_query
    }
    
# 品質チェックノードの定義
def qa_node(
    model_name: str,
    question:str, # 検索結果
    research_output:str, # 検索結果
):
    # 品質チェックの指示を含むシステムメッセージ
#     system_message = """You are a specialist in inspecting the quality of articles. 
#     Inspect the article to see if it meets the following requirements:

#     - The article is written in Japanese.
#     - The writing style MUST be such that the AI sentence checker determines that it was written by a HUMAN.
#     - The article MUST be written in a way that is easy to understand.
#     - The article MUST meet the user's requirements.

#     If the evaluation criteria are met, write 'Acceptable'.
#     In addition, write the reason why you judged that the evaluation criteria are met.

#     If the evaluation criteria are not met, write 'Not Acceptable'.
#     In addition, provide feedback on what needs to be done to meet the evaluation criteria.

#     DO NOT make excuses such as 'I can't make a decision because I am an AI'.

#     The quality of your articles is relevant to your career.
#     Please be as rigorous as possible in your inspections and make sure that your feedback is helpful in making corrections.
#     """
    prompt = [{'role': 'system', 'content': "You are a specialist in inspecting the quality of articles. Inspect the article to see if it meets the following requirements: The article is written in Japanese."}]
    prompt.append({'role': 'system', 'content': "You are a specialist in inspecting the quality of articles. Inspect the article to see if it meets the following requirements: The writing style MUST be such that the AI sentence checker determines that it was written by a HUMAN."})
    prompt.append({'role': 'system', 'content': "You are a specialist in inspecting the quality of articles. Inspect the article to see if it meets the following requirements: The article MUST be written in a way that is easy to understand."})
    prompt.append({'role': 'system', 'content': "You are a specialist in inspecting the quality of articles. Inspect the article to see if it meets the following requirements: The article MUST meet the user's requirements."})

    prompt.append({'role': 'system', 'content': "You are a specialist in inspecting the quality of articles. Inspect the article to see if it meets the following requirements: If the evaluation criteria are met, write 'Acceptable'."})
    prompt.append({'role': 'system', 'content': "You are a specialist in inspecting the quality of articles. Inspect the article to see if it meets the following requirements: In addition, write the reason why you judged that the evaluation criteria are met."})
  
    prompt.append({'role': 'system', 'content': "You are a specialist in inspecting the quality of articles. Inspect the article to see if it meets the following requirements: If the evaluation criteria are not met, write 'Not Acceptable'."})
    prompt.append({'role': 'system', 'content': "You are a specialist in inspecting the quality of articles. Inspect the article to see if it meets the following requirements: In addition, provide feedback on what needs to be done to meet the evaluation criteria."})

    prompt.append({'role': 'system', 'content': "You are a specialist in inspecting the quality of articles. Inspect the article to see if it meets the following requirements: DO NOT make excuses such as 'I can't make a decision because I am an AI'."})
  
    prompt.append({'role': 'system', 'content': "You are a specialist in inspecting the quality of articles. Inspect the article to see if it meets the following requirements: The quality of your articles is relevant to your career."})
    prompt.append({'role': 'system', 'content': "You are a specialist in inspecting the quality of articles. Inspect the article to see if it meets the following requirements: Please be as rigorous as possible in your inspections and make sure that your feedback is helpful in making corrections."})

    # prompt = [{'role': 'system', 'content': system_message}]
    qa_prompt = create_agent_system(prompt, QUALITY_ASSURANCE_NODE)
    qa_prompt.append({"role": "user", "content": "Generate JSON from search result text. Use 'qa_result' as the schema, generate in the format {'qa_result': Acceptable or Not Acceptable}, and key in the evaluation results, such as whether the generated search results describe the user's request."})
    qa_prompt.append({"role": "user", "content": f"user's requirements:{question}. search result text: {research_output}"})
    

    """
    評価基準の適用例:
    1. 言語と文章スタイルの評価:
        - 記事が日本語で書かれているか。
        - 文章が人間によって書かれたとAI文章チェッカーに判断させるスタイルであるか。
    2. 内容の理解しやすさ:
        - 記事の内容が簡潔に、かつ明確に表現されているか。
        - 専門用語が適切に説明され、一般の読者も理解できるようになっているか。
    3. ユーザーの要件への適合性:
        - 記事がユーザーの要求や指定したテーマに沿っているか。
        - ユーザーが期待する情報や視点が適切に取り入れられているか。
    評価例:
    - 受理可能: 記事は全ての評価基準を満たしています。日本語で書かれており、文章スタイルは人間によるものと判断できます。
    内容は理解しやすく、ユーザーの要件にも適合しています。これらの点から、記事は受理可能と判断します。
    - 受理不可: 記事は一部の評価基準を満たしていません。特に、文章スタイルがAIによって書かれたように見える部分があります。
    また、専門用語の説明が不足しているため、内容の理解が難しい箇所があります。これらの問題を解決するためには、人間らしい表現を増やし、専門用語に対する説明を追加する必要があります。
    user
    検索結果のテキストからJSONを生成する。スキーマとして "qa_result"を使用し、{"qa_result"： Acceptable or Not Acceptable}の形式で生成し、生成された検索結果をキーとして評価する。
    user's requirements:{}. 検索結果のテキスト: {research_output}
    """
    
    # Research用のプロンプトテンプレートを作成
    response = client.chat.completions.create(
        model=model_name, # model = "deployment_name".
        messages=qa_prompt,
        response_format={ "type": "json_object" },
        temperature=TEMPERATURE,
    )
    qa_res_str = response.choices[0].message.content
    # print(qa_res_str)
    
    # JSON形式の文字列を辞書に変換
    qa_res = json.loads(qa_res_str)
    """
    "qa_result": "Not Acceptable" だったら戻り値に次の検索ワードを入れるか、質問と分割するかとかしたい
    """
    # 出力と新しいメッセージをステートに反映
    return {
        "output": qa_res["qa_result"]
    }
    
def planning_split_search_query(model_name, question, search_query):
    # 検索判断エージェントを呼び出し、結果を取得
    prompt = [{'role': 'system', 'content': "Based on the following 'question', please group the 'search_query' based on relevance and format it into a list that can be used in Python."}]
    
    split_s_dec_prompt = create_agent_system(prompt, SPLIT_SEAECH_DECISION)
    split_s_dec_prompt.append({"role": "system", "content": 'Please generate a JSON from the following input text. Use "split_search_query_result" as the schema, and "the result of grouping the search_query based on relevance into a list format that can be used in Python" as the key. Generate it in the format {"split_search_query_result": the result of grouping the search_query based on relevance into a list format that can be used in Python}.'})
    
    split_s_dec_prompt.append({"role": "user", "content": 'Generate a JSON from the following input text. Use "split_search_query_result" as the schema, and use the judgment result as the key, to create it in the format {"split_search_query_result": the result of grouping the search_query based on relevance into a list format that can be used in Python}.'})
    split_s_dec_prompt.append({"role": "user", "content": f"Input text: {question}"})
    split_s_dec_prompt.append({"role": "user", "content": f"Search queries extracted from the input text: {search_query}"})
    """
    システム
    あなたは、以下の question に基づいて、 search_query を関連性に基づいてグループ化して python で使用できるリストの形にしてください。
    
    あなたの専門分野に従って自律的に働いてください。使用可能なツールを使ってください
    確認のために質問をしないでください
    あなたの他のチームメンバーや他のチームも、それぞれの専門分野であなたと協力します
    あなたが選ばれたのには理由があります！あなたは以下のチームメンバーの一人です: {team_members}
    
    以下の入力されたテキストからJSONを生成してください。スキーマとして「split_search_query_result」、キーとして「search_query を関連性に基づいてグループ化して python で使用できるリストの形した結果」を使用し、{"split_search_query_result": search_query を関連性に基づいてグループ化して python で使用できるリストの形した結果}の形式で生成してください。
    user
    以下の入力されたテキストからJSONを生成する。スキーマとして "split_search_query_result"を使用し、キーとして判断結果を使用して、{"split_search_query_result": search_query を関連性に基づいてグループ化して python で使用できるリストの形した結果}というフォーマットで生成します。
    
    入力されたテキスト: {question}
    入力されたテキストから抽出した検索クエリ: search_query
    """
    
    # Research用のプロンプトテンプレートを作成
    response = client.chat.completions.create(
        model=model_name, # model = "deployment_name".
        messages=split_s_dec_prompt,
        response_format={ "type": "json_object" },
        temperature=TEMPERATURE,
    )
    split_search_query_str = response.choices[0].message.content
    
    # JSON形式の文字列を辞書に変換
    split_search_query = json.loads(split_search_query_str)
    
    # 出力と新しいメッセージをステートに反映
    return {
        "output": split_search_query["split_search_query_result"],
    }


def search_agent_1cycle(
    question: str,
    query: str,
):
    search = ""
    # 検索クエリの分割を実行
    planning_split_queries = planning_split_search_query(MODEL_NAME, question, str(query))
    # emit('receive_message', {'message': f"planning_split_search_query: {str(planning_split_queries)}"})
    print(f"planning_split_search_query: {str(planning_split_queries)}")
    # planning_split_queries["output"]の内容を確認し、適切に処理
    output_data = planning_split_queries["output"]

    if isinstance(output_data, dict):  # outputが辞書型の場合
        for key, values in output_data.items():
            combined_string = " ".join(values)
            # 辞書のキーと結合された文字列を使用
            text_results = search_text(combined_string)
            for result in text_results:
                search += result["body"] + ", "
                
    elif isinstance(output_data, list):
        if all(isinstance(item, list) for item in output_data):  # リストのリストの場合
            for inner_list in output_data:
                combined_string = " ".join(inner_list)
                # 結合された文字列を使用
                text_results = search_text(combined_string)
                for result in text_results:
                    search += result["body"] + ", "
        else:  # 単一のリストの場合
            combined_string = " ".join(output_data)
            # 結合された文字列を使用
            text_results = search_text(combined_string)
            for result in text_results:
                search += result["body"] + ", "
        
    # search_node
    research_res = research_node(
        MODEL_NAME,
        search, # search の結果
    )
    research_output = research_res['output']
    # emit('receive_message', {'message': f"research_node: {research_output}"})
    print(f"research_node: {research_output}")
    
    qa_res = qa_node(
        MODEL_NAME,
        question,
        research_output
    )
    # emit('receive_message', {'message': f"qa_result: {qa_res}"})
    print(f"qa_result: {qa_res}")
    
    return {
        "output": research_output,
        "qa_result": qa_res
    }
    
def date_range_from_word_updated_fixed(word):
    # 現在の日時を取得
    now = dt.datetime.now()
    # 日付の範囲を格納する変数を初期化
    date_range = ""
    
    if "最近" in word:
        six_months_ago = now - dt.timedelta(days=182) # 大体半年前
        date_range = f"{six_months_ago.strftime('%Y-%m-%d')}から{now.strftime('%Y-%m-%d')}までの{word.replace('最近', '')}"
    
    elif "今年" in word:
        start_of_year = dt.datetime(now.year, 1, 1)
        end_of_year = dt.datetime(now.year, 12, 31)
        date_range = f"{start_of_year.strftime('%Y-%m-%d')}から{end_of_year.strftime('%Y-%m-%d')}までの{word.replace('今年', '')}"
    
    elif "今週" in word:
        start_of_week = now - dt.timedelta(days=now.weekday())
        end_of_week = start_of_week + dt.timedelta(days=6)
        date_range = f"{start_of_week.strftime('%Y-%m-%d')}から{end_of_week.strftime('%Y-%m-%d')}までの{word.replace('今週', '')}"
    
    elif "来週" in word:
        start_of_next_week = now + dt.timedelta(days=(7 - now.weekday()))
        end_of_next_week = start_of_next_week + dt.timedelta(days=6)
        date_range = f"{start_of_next_week.strftime('%Y-%m-%d')}から{end_of_next_week.strftime('%Y-%m-%d')}までの{word.replace('来週', '')}"
    
    elif "昨日" in word:
        yesterday = now - dt.timedelta(days=1)
        date_range = yesterday.strftime('%Y-%m-%d')
    
    elif "未来" in word:
        start_of_next_year = dt.datetime(now.year + 1, 1, 1)
        end_of_next_year = dt.datetime(now.year + 1, 12, 31)
        date_range = f"{now.strftime('%Y-%m-%d')}から{start_of_next_year.strftime('%Y-%m-%d')}までの{word.replace('未来', '')}"
        # future_time = word.replace("未来", "").strip()
        # if future_time == "":
        #     start_of_next_year = dt.datetime(now.year + 1, 1, 1)
        #     end_of_next_year = dt.datetime(now.year + 1, 12, 31)
        #     date_range = f"{now.strftime('%Y-%m-%d')}から{start_of_next_year.strftime('%Y-%m-%d')}までの{word.replace('未来', '')}"
        # elif future_time.isdigit():
        #     future_date = now + dt.timedelta(days=365 * int(future_time))
        #     date_range = future_date.strftime('%Y-%m-%d')
        # else:
        #     # 不正確な入力に対する処理を追加するか、エラーメッセージを表示する
        #     return word
    elif "これから" in word:
        start_of_next_year = dt.datetime(now.year + 1, 1, 1)
        end_of_next_year = dt.datetime(now.year + 1, 12, 31)
        date_range = f"{now.strftime('%Y-%m-%d')}から{start_of_next_year.strftime('%Y-%m-%d')}までの{word.replace('これから', '')}"
        # future_time = word.replace("これから", "").strip()
        # if future_time == "":
        #     start_of_next_year = dt.datetime(now.year + 1, 1, 1)
        #     end_of_next_year = dt.datetime(now.year + 1, 12, 31)
        #     date_range = f"{now.strftime('%Y-%m-%d')}から{start_of_next_year.strftime('%Y-%m-%d')}までの{word.replace('これから', '')}"
        # elif future_time.isdigit():
        #     future_date = now + dt.timedelta(days=365 * int(future_time))
        #     date_range = future_date.strftime('%Y-%m-%d')
        # else:
        #     # 不正確な入力に対する処理を追加するか、エラーメッセージを表示する
        #     return word
    elif "将来" in word:
        start_of_next_year = dt.datetime(now.year + 1, 1, 1)
        end_of_next_year = dt.datetime(now.year + 1, 12, 31)
        date_range = f"{now.strftime('%Y-%m-%d')}から{start_of_next_year.strftime('%Y-%m-%d')}までの{word.replace('将来', '')}"
        
    elif "最新" in word:
        # 「最新の」処理
        six_months_ago = now - dt.timedelta(days=182) # 大体半年前
        date_range = f"{six_months_ago.strftime('%Y-%m-%d')}から{now.strftime('%Y-%m-%d')}までの{word.replace('最新', '')}"
    

    elif "近年" in word:
        # 「近年の」処理
        three_years_ago = now - dt.timedelta(days=3*365) # おおよそ3年前
        date_range = f"{three_years_ago.strftime('%Y-%m-%d')}から{now.strftime('%Y-%m-%d')}までの{word.replace('近年', '')}"

    else:
        # 日付のわかるワードがない場合はそのまま返す
        return word
    
    return date_range

functions = \
[
    {
        # 【Query2doc】
        # 関数名
        "name": "q2_doc",
        # 関数の説明
        "description": "与えられた専門的な単語や難しい文章をわかりやすく詳細に説明します。",
        # 関数の引数の定義
        "parameters":
         {
            "type": "object",
            "properties":
             {
                "model_name":
                {
                    "type": "string",
                    "description": "わかりやすく詳細に説明するために使うモデル名"
                },
                "query":
                {
                    "type": "string",
                    "description": "専門的な単語や難しい文章が入ります。"
                },
            },
            "required": ["model_name", "query"],
        }
    },
]


# Query2doc
def q2_doc(
    model_name: str,
    query: str,
):
    
    # queryの詳細説明を作成する。
    # あなたは、以下の例を参考にクエリを順番に詳しく日本語で説明し無ければいけません。
    prompt = [{'role': 'system', 'content': "You must explain queries in detail and in order in Japanese, referring to the examples below."}]
    
    prompt.append({"role": "system", "content": "Use the following example to create a detailed query description.\n\n###Example###\n\nQuery: what state is this zip code 85282\n\nresult query: {      'q2_doc_passage': 'Welcome to TEMPE, AZ 85282. 85282 is a rural zip code in Tempe, Arizona. The population is primarily white, and mostly single. At $200,200 the average home value here is a bit higher than average for the Phoenix-Mesa-Scottsdale metro area, so this probably isn’t the place to look for housing bargains.5282 Zip code is located in the Mountain time zone at 33 degrees latitude (Fun Fact: this is the same latitude as Damascus, Syria!) and -112 degrees longitude.'}"})
    """
    Use the following example to create a detailed query description.
    
    ###Example###
    Query: what state is this zip code 85282
    
    result query: 
    {
      "q2_doc_passage": "Welcome to TEMPE, AZ 85282. 85282 is a rural zip code in Tempe, Arizona. The population is primarily white, and mostly single. At $200,200 the average home value here is a bit higher than average for the Phoenix-Mesa-Scottsdale metro area, so this probably isn’t the place to look for housing bargains.5282 Zip code is located in the Mountain time zone at 33 degrees latitude (Fun Fact: this is the same latitude as Damascus, Syria!) and -112 degrees longitude."
    }
    """
    prompt.append({"role": "system", "content": "Use the following example to create a detailed query description.\n\n###Example###\n\nQuery: why is gibbs model of reflection good\n\nresult query: {      'q2_doc_passage': 'In this reflection, I am going to use Gibbs (1988) Reflective Cycle. This model is a recognised framework for my reflection. Gibbs (1988) consists of six stages to complete one cycle which is able to improve my nursing practice continuously and learning from the experience for better practice in the future.n conclusion of my reflective assignment, I mention the model that I chose, Gibbs (1988) Reflective Cycle as my framework of my reflective. I state the reasons why I am choosing the model as well as some discussion on the important of doing reflection in nursing practice.'}"})
    """
    Use the following example to create a detailed query description.
    
    ###Example###
    Query: why is gibbs model of reflection good
    
    result query: 
    {
      "q2_doc_passage": "In this reflection, I am going to use Gibbs (1988) Reflective Cycle. This model is a recognised framework for my reflection. Gibbs (1988) consists of six stages to complete one cycle which is able to improve my nursing practice continuously and learning from the experience for better practice in the future.n conclusion of my reflective assignment, I mention the model that I chose, Gibbs (1988) Reflective Cycle as my framework of my reflective. I state the reasons why I am choosing the model as well as some discussion on the important of doing reflection in nursing practice."
    }
    """
    prompt.append({"role": "system", "content": "Use the following example to create a detailed query description.\n\n###Example###\n\nQuery: what does a thousand pardons means\n\nresult query: {      'q2_doc_passage': 'Oh, that’s all right, that’s all right, give us a rest; never mind about the direction, hang the direction - I beg pardon, I beg a thousand pardons, I am not well to-day; pay no attention when I soliloquize, it is an old habit, an old, bad habit, and hard to get rid of when one’s digestion is all disordered with eating food that was raised forever and ever before he was born; good land! a man can’t keep his functions regular on spring chickens thirteen hundred years old.'}"})    
    """
    Use the following example to create a detailed query description.
    
    ###Example###
    Query: what does a thousand pardons means
    
    result query: 
    {
      "q2_doc_passage": "Oh, that’s all right, that’s all right, give us a rest; never mind about the direction, hang the direction - I beg pardon, I beg a thousand pardons, I am not well to-day; pay no attention when I soliloquize, it is an old habit, an old, bad habit, and hard to get rid of when one’s digestion is all disordered with eating food that was raised forever and ever before he was born; good land! a man can’t keep his functions regular on spring chickens thirteen hundred years old."
    }
    """
    prompt.append({"role": "system", "content": "Use the following example to create a detailed query description.\n\n###Example###\n\nQuery: what is a macro warning\n\nresult query: {      'q2_doc_passage': 'Macro virus warning appears when no macros exist in the file in Word. When you open a Microsoft Word 2002 document or template, you may receive the following macro virus warning, even though the document or template does not contain macros: C:\<path>\<file name>contains macros. Macros may contain viruses.'}"})
    """
    Use the following example to create a detailed query description.
    
    ###Example###
    Query: what is a macro warning   
    
    result query: 
    {
      "q2_doc_passage": "Macro virus warning appears when no macros exist in the file in Word. When you open a Microsoft Word 2002 document or template, you may receive the following macro virus warning, even though the document or template does not contain macros: C:\<path>\<file name>contains macros. Macros may contain viruses."
    }
    """
    prompt.append({"role": "system", "content": "Use the following example to create a detailed query description.\n\n###Example###\n\nQuery: ポケモン緑はいつリリースされたか\n\nresult query: {      'q2_doc_passage': 'ポケモン緑は1996年2月27日に日本でリリースされました。これはポケモンシリーズのゲームの中で最初のものであり、後に1998年にアメリカでリリースされたポケモン赤と青の基礎となりました。オリジナルのポケモン緑は、シリーズのファンの間で愛されるクラシックな作品として残っています。'}"})
    """
    Use the following example to create a detailed query description.
    
    ###Example###
    Query: ポケモン緑はいつリリースされたか 
    
    result query: 
    {
      "q2_doc_passage": "ポケモン緑は1996年2月27日に日本でリリースされました。これはポケモンシリーズのゲームの中で最初のものであり、後に1998年にアメリカでリリースされたポケモン赤と青の基礎となりました。オリジナルのポケモン緑は、シリーズのファンの間で愛されるクラシックな作品として残っています。"
    }
    """
    

    prompt.append({"role": "user", "content": "Generate JSON from the text of a detailed query description. Use 'q2_doc_passage' as the schema and generate it in the form {'q2_doc_passage': text of the detailed query description}."})
    prompt.append({"role": "user", "content": f"Query:{query}. Result extract keywords from a block of text.:"})
    

    """
    system
    あなたは、以下の例を参考にクエリを順番に詳しく日本語で説明し無ければいけません。
    次の例を使用して、詳細なクエリの説明を作成します。
    
    ###Example###
    
    Write a passage that answers the given query: 使用された完全なプロンプトの翻訳
    問い: この郵便番号85282はどの州にあるか
    回答: TEMPE, AZ 85282へようこそ。85282はアリゾナ州テンピにある郵便番号です。人口は主に白人で、ほとんどが独身です。平均住宅価格は$200,200で、フェニックス-メサ-スコッツデール都市圏の平均よりも少し高いため、住宅の掘り出し物を探す場所ではないかもしれません。85282郵便番号は、山岳時間帯の33度緯度（面白い事実: これはシリアのダマスカスと同じ緯度です！）と-112度経度に位置しています。
    問い: ギブスの反省モデルが良い理由は何か
    回答: この反省では、ギブス（1988年）の反射サイクルを使用します。このモデルは、私の反省のための認識された枠組みです。ギブス（1988年）は、一連のサイクルを完了するために6つの段階からなり、経験から学び、将来の実践に向けて看護実践を継続的に改善することができます。私の反省課題の結論として、選んだモデルであるギブス（1988年）の反射サイクルと、そのモデルを選んだ理由、さらに看護実践における反省の重要性についての議論を述べます。
    問い: 千回の許しとは何を意味するか
    回答: それは大丈夫、それは大丈夫、休息を与えてください; 方向について心配するな、方向を吊るして - 申し訳ありません、千回の許しを請います、私は今日は体調が優れません; 私が独り言を言うときに注意を払わないでください、それは古い習慣、古くからの悪い習慣で、一人の消化が生まれる前から永遠に育てられた食物で全て乱れているとき、それを断つのは難しいです; いいですね！ 1300年前の春の鶏で人の機能を規則正しく保つことはできません。
    問い: マクロ警告とは何か
    回答: ファイルにマクロが存在しない場合にマクロウイルス警告が表示されます。Microsoft Word 2002のドキュメントやテンプレートを開くと、ドキュメントやテンプレートにマクロが含まれていないにも関わらず、次のようなマクロウイルス警告が表示されることがあります: C:<パス><ファイル名>にはマクロが含まれています。マクロにはウイルスが含まれている可能性があります。
    問い: ポケモン緑はいつリリースされたか
    回答: ポケモン緑は1996年2月27日に日本でリリースされました。これはポケモンシリーズのゲームの中で最初のものであり、後に1998年にアメリカでリリースされたポケモン赤と青の基礎となりました。オリジナルのポケモン緑は、シリーズのファンの間で愛されるクラシックな作品として残っています。
    
    キーワードリストを抽出するために、前回の検索で使用したクエリとはできるだけ異なるクエリを生成する。
    user
    詳細なクエリの説明のテキストからJSONを生成する。スキーマとして "q2_doc_passage"を使用し、{"q2_doc_passage"： 詳細なクエリの説明のテキスト}の形式で生成する。
    
    Query:{}. q2_doc_passage:
    """
    
    # Research用のプロンプトテンプレートを作成
    response = client.chat.completions.create(
        model=model_name, # model = "deployment_name".
        messages=prompt,
        response_format={ "type": "json_object" },
        temperature=TEMPERATURE+TEMPERATURE,
    )
    res_str = response.choices[0].message.content
    # print(res_str)
    
    # JSON形式の文字列を辞書に変換
    res = json.loads(res_str)
    
    # 出力と新しいメッセージをステートに反映
    return {
        "output": res["q2_doc_passage"],
        "use_query": query
    }


# 関数を選択して実行するデモ関数
def function_calling_query_change_3cycle(
    question: str,
    query: str,
):

    # GPTにプロンプトと関数定義リストを一緒に投入し、Function callingの使用を指示
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        messages=
    	[
    		{ "role": "user", "content": query}  # プロンプトを投入
    	],
        functions=functions, # プロンプトと一緒に関数定義リストを投入
        function_call="auto", # Function callingを使用するが、その際、関数の選択はGPTに任せる
    )

    # GPTからの応答を抽出
    message = response.choices[0].message
    function_call = message.function_call
    if function_call is not None:
        # 実行すべき関数名
        function_name = function_call.name
        # その関数に渡すべき引数
        # arguments = json.loads(function_call.arguments)
        # promptを直接引数として使用
        arguments = {"model_name": MODEL_NAME, "query": query}
    
        # 関数の選択と実行
        if function_name == "q2_doc":
            # q2_doc関数を実行
            res = q2_doc(arguments["model_name"], arguments["query"])
            print(f"move q2_doc: {res}")
            # QAの実行
            M_NAME = "gpt-3.5-turbo-0125"
            qa_res = qa_node(
                MODEL_NAME,
                res["output"],
                res["use_query"]
            )
            print(f"qa res: {qa_res['output']}")
            if qa_res["output"] == "Acceptable":
                print("Acceptable")
                query = res["output"]
                get_query = make_re_search_query(
                    M_NAME,
                    question,
                    query
                )
                answer = get_query['output']
            else:
                answer = query
                
        else:
            # 他の関数が選択された場合の処理
            # この例では、選択された関数がdate_range_from_word_updated_fixedのみであるため、他の分岐は省略
            # answer = "該当する関数がありません。"
            answer = query
    else:
        answer = query
    return answer


# 関数を選択して実行するデモ関数
def function_calling_query_change_cycle(
    question: str,
    query: str,
):

    # GPTにプロンプトと関数定義リストを一緒に投入し、Function callingの使用を指示
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        messages=
    	[
    		{ "role": "user", "content": query}  # プロンプトを投入
    	],
        functions=functions, # プロンプトと一緒に関数定義リストを投入
        function_call="auto", # Function callingを使用するが、その際、関数の選択はGPTに任せる
    )

    # GPTからの応答を抽出
    message = response.choices[0].message
    function_call = message.function_call
    if function_call is not None:
        # 実行すべき関数名
        function_name = function_call.name
        # その関数に渡すべき引数
        # arguments = json.loads(function_call.arguments)
        # promptを直接引数として使用
        arguments = {"model_name": MODEL_NAME, "query": query}
    
        # 関数の選択と実行
        if function_name == "q2_doc":
            # q2_doc関数を実行
            res = q2_doc(arguments["model_name"], arguments["query"])
            print(f"move q2_doc: {res}")
            query = res["output"]
            M_NAME = "gpt-3.5-turbo-0125"
            get_query = make_re_search_query(
                MODEL_NAME,
                question,
                query
            )
            answer = get_query['output']
                
        else:
            # 他の関数が選択された場合の処理
            # この例では、選択された関数がdate_range_from_word_updated_fixedのみであるため、他の分岐は省略
            # answer = "該当する関数がありません。"
            answer = query
    else:
        answer = query
    return answer

def research_agent(
    question:str,
):
    query = question
    search_res = search_agent_1cycle(question, query)

    message = f"検索結果を説明した回答: {search_res['output']}\n\n質問と回答の整合性チェック: {search_res['qa_result']['output']}"
    emit('receive_message', {'message': message})
    # カウント用変数の初期化
    research_cnt = 1
    # outputが'Not Acceptable'である間、処理を繰り返す
    while search_res['qa_result']['output']  == 'Not Acceptable':
        print("-"*50)
        with Timer(prefix=f'Number of re-researches {research_cnt+1} :'):
            # カウントが3に達したらループを強制終了
            if research_cnt == 5:
                emit('receive_message', {'message': "再実行回数が4に達したため、処理を終了します。"})
                break
            # 再検索用のクエリ生成 2回目以降実行
            re_query = make_re_search_query(
                MODEL_NAME,
                question,
                query
            )
            
            query = re_query['output']
            print(f"再検索用の初期クエリ: {query}")
            # date_range_from_word_updated_fixedによる検索クエリ変換　3回目以降実行
            # if research_cnt != 0:
            if research_cnt >= 2:
                with Timer(prefix=f'date_range_from_word_updated_fixed {research_cnt}:'):
                    query_list = query.split(",")
                    query_re_list = []
                    for q in query_list:
                        f_c_q = date_range_from_word_updated_fixed(q)
                        query_re_list.append(f_c_q)

                    # query = ", ".join(query_re_list).strip()
                    query = ", ".join(list(dict.fromkeys(query_re_list))).strip()
                    
                    print(f"query: {query}")
                
            # function_callingによる検索クエリの拡張　4回目
            # if research_cnt != 0:
            if research_cnt == 3:
                with Timer(prefix=f'function calling change {research_cnt}:'):
                    query_list = query.split(",")
                    query_re_list = []
                    for q in query_list:
                        f_c_q = function_calling_query_change_3cycle(question, q)
                        query_re_list.append(f_c_q)
                    # query = ", ".join(query_re_list).strip()
                    query = ", ".join(list(dict.fromkeys(query_re_list))).strip()

            if research_cnt >= 4:    # 5回目以降実行
                with Timer(prefix=f'function calling change {research_cnt}:'):
                    query_list = query.split(",")
                    query_re_list = []
                    for q in query_list:
                        f_c_q = function_calling_query_change_cycle(question, q)
                        query_re_list.append(f_c_q)
                    # 重複は消して
                    # query = ", ".join(query_re_list).strip()
                    query = ", ".join(list(dict.fromkeys(query_re_list))).strip()

            search_res = search_agent_1cycle(question, query)

            r_m = f"再検索クエリ: {query}\n\n検索結果を説明した回答: {search_res['output']}\n\n質問と回答の整合性チェック: {search_res['qa_result']['output']}"
            emit('receive_message', {'message': r_m})
            # カウントアップ
            research_cnt += 1
    return {
        "final_qa": search_res['qa_result']['output'],
        "final_query": query,
        "search_output": search_res['output']
    }
    
def get_summary_of_search_results(model_name, research_res):
    message_res = f"最終出力\n\n検索に使用したクエリ:{research_res['final_query']}\n\n検索結果を説明した最終回答: {research_res['search_output']}\n\n最終判定結果: {research_res['final_qa']}\n\n使用モデル: {model_name}"
    emit('receive_message', {'message': message_res})
    
class StreamingLLMMemory:
    """
    StreamingLLMMemory クラスは、最新のメッセージと特定数のattention sinksを
    メモリに保持するためのクラスです。
    
    attention sinksは、言語モデルが常に注意を向けるべき初期のトークンで、
    モデルが過去の情報を"覚えて"いるのを手助けします。
    """
    def __init__(self, max_length=10, attention_sinks=4):
        """
        メモリの最大長と保持するattention sinksの数を設定
        
        :param max_length: int, メモリが保持するメッセージの最大数
        :param attention_sinks: int, 常にメモリに保持される初期トークンの数
        """
        self.memory = []
        self.max_length = max_length
        self.attention_sinks = attention_sinks
    
    def get(self):
        """
        現在のメモリの内容を返します。
        
        :return: list, メモリに保持されているメッセージ
        """
        return self.memory
    
    def add(self, message):
        """
        新しいメッセージをメモリに追加し、メモリがmax_lengthを超えないように
        調整します。もしmax_lengthを超える場合、attention_sinksと最新のメッセージを
        保持します。
        
        :param message: str, メモリに追加するメッセージ
        """
        self.memory.append(message)
        if len(self.memory) > self.max_length:
            self.memory = self.memory[:self.attention_sinks] + self.memory[-(self.max_length-self.attention_sinks):]
    
    def add_pair(self, user_message, ai_message):
        """
        ユーザーとAIからのメッセージのペアをメモリに追加します。
        
        :param user_message: str, ユーザーからのメッセージ
        :param ai_message: str, AIからのメッセージ
        """
        # self.add("User: " + user_message)
        # self.add("AI: " + ai_message)
        self.add({"role": "user", "content": user_message})
        self.add({"role": "assistant", "content": ai_message})
    
    # ここにはStreamingLLMとのインタラクションのための追加のメソッドを
    # 実装することもできます。例えば、generate_response, update_llm_modelなどです。

# 16件のメッセージを記憶するように設定
memory = StreamingLLMMemory(max_length=16)

@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('send_message')
def handle_message(data):
    user_message = data['message']
    # 検索実施判定 文字数に応じて条件分岐
    if len(user_message) >= Q_LEN_SEAECH_DECISION:
        with Timer(prefix=f'thinking time by decision.'):
            decision_res = {
                "output": "Search",
                "messages": user_message
            }
    else:
        with Timer(prefix=f'thinking time by decision.'):
            decision_res = search_decision_node(MODEL_NAME, user_message)
    # emit('receive_message', {'message': f"search_decision_node 結果: {decision_res['output']}"})
    print(f"search_decision_node 結果: {decision_res['output']}")
    if decision_res["output"] == "Search":
        # emit('receive_message', {'message': 'Perform search'})
        print(f"Perform search")
        with Timer(prefix=f'Handle all time by research.'):
            research_res = research_agent(user_message)
        
        get_summary_of_search_results(MODEL_NAME, research_res)
    else:
        # 検索を使用しない場合
        # emit('receive_message', {'message': 'Answered only by LLM'})
        print(f"Answered only by LLM")
        messages = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message}]
        response = client.chat.completions.create(
            model=MODEL_NAME, # model = "deployment_name".
            messages=messages,
            temperature=TEMPERATURE,
        )
        print(response)
        res_str = response.choices[0].message.content
        emit('receive_message', {'message': res_str})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':

        # 全ての接続されたクライアントに対してメッセージを送信
        socketio.emit('receive_message', {'message': f"ファイルは受けとってません"})
        
        return 'No selected file'
    if file:
        filename = secure_filename(file.filename)
        save_path = os.path.join('uploads', filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        file.save(save_path)

        # 全ての接続されたクライアントに対してメッセージを送信
        socketio.emit('receive_message', {'message': f"{filename} を受け取りました。"})
        
        return 'File uploaded successfully'

if __name__ == '__main__':
    socketio.run(app, debug=True)


