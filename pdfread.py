
from PyPDF2 import PdfReader
import os

os.environ["OPENAI_API_KEY"] = "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"

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        with open(pdf, 'rb') as file:
            pdf_reader = PdfReader(file)
            for page in range(len(pdf_reader.pages)):
                text += pdf_reader.getPage(page).extractText()
    return text

pdf_docs = ["/Users/vir/Documents/GitHub/chatCKB/premchand.pdf"]
raw_text = get_pdf_text(pdf_docs)

print(raw_text)