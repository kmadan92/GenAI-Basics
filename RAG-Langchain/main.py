from pdf_processor import process_all_pdfs_in_folder
from dotenv import load_dotenv
import os

# Load environment just in case
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Folder with your PDFs
pdf_folder_path = "./pdf"

retrievers = process_all_pdfs_in_folder(pdf_folder_path)

print(f"\nTotal retrievers created: {len(retrievers)}")
