import os

os.environ["OPENAI_API_KEY"] = "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"

def get_text_file_content(txt_files):
    text = ""
    for txt_file in txt_files:
        with open(txt_file, 'r') as file:
            text += file.read()
    return text

txt_files = ["/Users/vir/Documents/GitHub/chatCKB/premchand.txt"]
raw_text = get_text_file_content(txt_files)

print(raw_text)
