import streamlit as st
from transformers import T5Tokenizer, AutoModelForCausalLM

def cached_tokenizer():
    tokenizer = T5Tokenizer.from_pretrained("huranokuma/es")
    tokenizer.do_lower_case = True
    return tokenizer

def cached_model():
    model = AutoModelForCausalLM.from_pretrained("huranokuma/es")
    return model

def main():
  st.title("AIによるES作成")  

  num_of_output_text = st.slider(label='出力する文章の数',
                  min_value=1,
                  max_value=3,
                  value=1,
                  )

  max_length = st.slider(label='最大文字数',
                  min_value=30,
                  max_value=1000,
                  value=100,
                  )
  
  min_length = st.slider(label='最低文字数',
                  min_value=30,
                  max_value=1000,
                  value=30,
                  )
  

  PREFIX_TEXT = st.text_area(
        label='テキスト入力', 
        value='御社を志望した理由は'
  )

  progress_num = 0
  status_text = st.empty()
  progress_bar = st.progress(progress_num)
  process_text = st.empty()

  #モデルは一回だけ読み込む
  progress_num =  10
  status_text.text(f'Progress: {progress_num}%')
  progress_bar.progress(progress_num)

  process_text.text("モデルの読み込みに時間がかかります。根気よくお待ちください。")

  tokenizer = cached_tokenizer()

  progress_num =  50
  status_text.text(f'Progress: {progress_num}%')
  progress_bar.progress(progress_num)

  model = cached_model()

  progress_num =  100
  status_text.text(f'Progress: {progress_num}%')
  progress_bar.progress(progress_num)

  if st.button('文章生成'):

    process_text.text("文章を生成しています...")

    # 推論 
    input = tokenizer.encode(PREFIX_TEXT, return_tensors="pt",add_special_tokens=False) 
    progress_num = 50
    status_text.text(f'Progress: {progress_num}%')
    progress_bar.progress(progress_num)

    output = model.generate(
            input, do_sample=True, 
            max_length=max_length,
            min_length=min_length,
            top_k=500,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bad_word_ids=[[tokenizer.unk_token_id]],
            num_return_sequences=num_of_output_text
            )
    progress_num = 90
    status_text.text(f'Progress: {progress_num}%')
    progress_bar.progress(progress_num)

    output_text = tokenizer.batch_decode(output,skip_special_tokens=True)
    progress_num = 95
    status_text.text(f'Progress: {progress_num}%')
    progress_bar.progress(progress_num)

    st.info('生成結果')
    progress_num = 100
    status_text.text(f'Progress: {progress_num}%')
    for i in range(num_of_output_text):
      st.write(output_text[i])
    progress_bar.progress(progress_num)

if __name__ == '__main__':
  main()
