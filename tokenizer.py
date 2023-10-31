import os
import json
import pandas as pd
from transformers import DistilBertTokenizer
from tokenizers import BertWordPieceTokenizer

ckpt_path = './ckpts/tokenizer/ASSIST2009/'
dataset_path = os.path.join(
    "./datasets/ASSIST2009", "skill_builder_data.csv"
)

df = pd.read_csv(dataset_path, encoding='ISO-8859-1').dropna(subset=["skill_name"])\
    .dropna(subset=['answer_text'])\
    .sort_values(by=["order_id"])
    # .drop_duplicates(subset=["order_id", "skill_name"])\

# 셀프로 토크나이저 만들기 (아래)
# my_vocab_size = 30000
# my_limit_alphabet = 6000

# my_special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
# user_defined_symbols = ["[BOS]", "[EOS]"]
# my_special_tokens = my_special_tokens + user_defined_symbols

# answer_text = df['answer_text'].values

# bert_tokenizer = BertWordPieceTokenizer(
#     clean_text=True,
#     handle_chinese_chars=False,
#     strip_accents=True,
#     lowercase=True
# )

# bert_tokenizer.train_from_iterator(
#     answer_text,
#     limit_alphabet=my_limit_alphabet,
#     vocab_size=my_vocab_size,
#     show_progress=True,
#     min_frequency=5,
#     special_tokens=my_special_tokens
# )

# # save vocab file
# token_file = f"tok_added-ch-{my_limit_alphabet}-{my_vocab_size}"
# vocab_path = os.path.join(ckpt_path, token_file)
# bert_tokenizer.save(vocab_path, True)

# # vocab.txt to vocab.json
# vocab_file = os.path.join(ckpt_path, "vocab.txt")
# # write txt
# f = open(vocab_file, 'w', encoding='utf-8')
# with open(vocab_path) as json_file:
#     json_data = json.load(json_file)
#     for item in json_data["model"]["vocab"].keys():
#         f.write(item+'\n')
#     f.close()
    
# distil_tokenizer = DistilBertTokenizer(vocab_file, do_lower_case=True)
# distil_tokenizer.save_pretrained(os.path.join(ckpt_path, "DISTIL_A2009"))


# Let's increase the vocabulary of Bert model and tokenizer
# 여기에 map이나 lambda 적용해서 배열 한번에 빼야할 듯
ds_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

ats = df['answer_text'].values
print(ats)
for at in ats:
    new_tokens = ds_tokenizer.tokenize(at)
    num_added_toks = ds_tokenizer.add_tokens(new_tokens)
    
print(len(ds_tokenizer))
ds_tokenizer.save_pretrained(os.path.join(ckpt_path, "DISTIL_A2009"))    