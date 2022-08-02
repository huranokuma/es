import streamlit as st
import requests

API_URL = "https://api-inference.huggingface.co/models/huranokuma/es"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

def main():
  st.title("AIによる自動ES作成")  

  max_length = st.slider(label='最大文字数',
                  min_value=30,
                  max_value= 500,
                  value=100,
                  )
  
  top_p = st.slider(label='top p',
                  min_value=0,
                  max_value=1.0,
                  value=0.95
  )

  top_k = st.slider(label='top k',
                min_value=0,
                max_value=1000,
                value=500
)
  temperature = st.slider(label='temperature',
                min_value=0.0,
                max_value=100.0,
                value=1.00
)
  

  prompt_text = st.text_area(
        label='ESの書き出しの文章', 
        value='私が御社を志望した理由は'
  )

  progress_num = 0
  status_text = st.empty()
  progress_bar = st.progress(progress_num)
  process_text = st.empty()

  process_text.text("ESの書き出しや、会社の質問を入力してください。それに続く文章を生成します。")

  if st.button('文章生成'):

    progress_num = 10 
    progress_bar = st.progress(progress_num)
    status_text.text(f'Progress: {progress_num}%')
    process_text.text("文章を生成しています...")

    # APIを使ってHuggingfaceから文章を取ってくる。
    output = query({"inputs": prompt_text,
                "parameters": {
                               "max_length":max_length,
                               "min_length":50,
                               "top_p":top_p,
                               "top_k":top_k,
                               "temperature":temperature,
                               },
                "options":{
                    "wait_for_model": True,
                }
                })

    process_text.text("ESの生成が終了しました。")
    st.info('生成結果')
    progress_num = 100
    status_text.text(f'Progress: {progress_num}%')
    st.write(output[0]['generated_text'])
    progress_bar.progress(progress_num)

if __name__ == '__main__':
  main()
