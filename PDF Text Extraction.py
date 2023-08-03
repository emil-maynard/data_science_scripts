# This script aims to extract pages from large pdf files which contain certain keywords. Making it easier
# to comb through large reports for specific data.

from pypdf import *
from docx import Document
import os
from os import listdir
from os.path import isfile, join
import fitz
import time

main_input_directory = '/Users/Emilmaynard/Desktop/PDF Text Extraction/Banking PDFs/'

# Set the keywords to search for in the documents
keywords = ['Climate Risk', 'Sustainability']

# Create output folders if they dont already exist
output_directory = main_input_directory+'Initial_Results/'
output_directory2 = main_input_directory+'Final_Results/'
directories = [output_directory, output_directory2]

for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)

#Choose between pdf or word output
output_format = 'pdf'  

# Pulls all pdfs from input files folder for analysis
pdf_name = [file for file in listdir(main_input_directory) if isfile(join(main_input_directory, file))] 
pdf_name.remove('.DS_Store')

# Search pdfs for keywords, extracts the pages where keywords are found 
def extract_information(format):
    for file in pdf_name:     
        with open(main_input_directory+file, 'rb') as f:
            reader = PdfReader(f)
            writer = PdfWriter()
            pdf_length = len(reader.pages)
            print(pdf_name)
            print("Length of pdf: "+str(pdf_length)+ " pages")

            for page_number in range(pdf_length):
                print("Page "+str(page_number))
                page = reader.pages[page_number]
                information = page.extract_text()
                
                for word in keywords:
                    word = word.lower()
                    if word in information:
                        print(information)
                        if format == 'word':
                            document = Document()
                            document.add_paragraph(information)
                            document.save(output_directory+str(page_number)+' '+word+'.docx')
                        else:    
                            writer.add_page(reader.pages[page_number]) #needs error handling if theres no data
                            pdf_output_filename = output_directory+word+'_'+file+'.pdf'  
                            with open(pdf_output_filename, "wb") as out:
                                writer.write(out)          
                            
# Highlights keywords in the pages found from the previous step            
def highlights():
    if output_format == 'pdf':
        output_files = [file for file in listdir(output_directory) if isfile(join(output_directory, file))]
        for file in output_files:
            print(file)
            doc = fitz.open(output_directory+file)

            for page in doc:
                # Search
                for word in keywords:
                    text = word.lower()
                    text_instances = page.search_for(text)

                    # Highlight
                    for inst in text_instances:
                        highlight = page.add_highlight_annot(inst)
                        highlight.update()

            # Output
            doc.save(output_directory2+'_final_'+file, garbage=4, deflate=True, clean=True)

if __name__ == '__main__':
    extract_information(output_format)
    highlights()



