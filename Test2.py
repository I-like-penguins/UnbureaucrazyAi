from Lancedb_Manager import LawDB as LawManagerLance
import os
import ollama
import json
import time
import io
import re

from docling_core.types.io import DocumentStream
from markdown_it.rules_block import paragraph
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageOps
from docling.document_converter import DocumentConverter

import requests
import zipfile
import xml.etree.ElementTree as ET
from io import BytesIO

#MODEL_OCR = "deepseek-r1:7b"
MODEL_OCR = "qwen2.5:3b"
MODEL_GIST = "qwen2.5:7b"
MODEL_RESPONSE = "qwen2.5:7b" # change to 14b maybe
MODEL_RAG = "phi3.5:latest"
MODEL_EMBEDDING = "nomic-embed-text:latest"

model_list = [MODEL_OCR, MODEL_GIST, MODEL_RESPONSE, MODEL_EMBEDDING, MODEL_RAG]

def check_and_download_model(model_name) -> bool:
    """Checks if the model is already downloaded locally and downloads it if not.
     Returns True if successful, False otherwise.
     Uses the ollama library: https://github.com/olly-ai/ollama"""
    try:
        print(f"checking for model {model_name}...")
        local_models = ollama.list()
        if model_name not in local_models:
            print(f"Model {model_name} not found locally. Downloading...")
            current = ""
            for progress in ollama.pull(model_name, stream=True):
                status = progress.get("status", "")
                digest = progress.get("digest", "")
                if status != current:
                    print(f"Status: {status}")
                    current = status
            print(f"model {model_name} downloaded successfully.")
        else:
            print(f"model {model_name} already downloaded and ready to use.")
        return True
    except Exception as e:
        print(f"Error downloading model {model_name}: {e}")
        return False

def clean_json(raw_content):
    cleaned = re.sub(r'```json\s?|\s?```', '', raw_content).strip()
    match = re.search(r'(\{.*\})', cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1)

    return cleaned

def check_readability(text):
    """Checks a text against occurrences of high-probability German words or if it is empty. True if readable"""
    if len(text) < 20:  # Empty pages are okay
        return True
    word_list = ["der", "die", "das", "und", "oder", "nicht", "wir", "ihr", "sie", "ist", "mit", "von", "den"]
    count = sum(1 for word in word_list if word in text.lower())
    return count > 1

def ocr_enhanced_images(pdf_path, contrast_factor=2.0, sharpness_factor=2.0) -> str:
    """
    Gets the text from a PDF using OCR after enhancing the images for readability. Also rotates images 180 degrees if they are not readable.
    :param pdf_path: Path to the PDF file.
    :param contrast_factor: Contrast factor to adjust image contrast.
    :param sharpness_factor: Sharpness factor to adjust image sharpness.
    :return: A Markdown string with the OCR text of all pages in the PDF. Returns an empty string if the text is not readable.
    """
    pages = convert_from_path(pdf_path, 300)
    converter = DocumentConverter()
    enhanced_images = []
    full_text = ""
    for i, page in enumerate(pages):
        print(f"Processing page {i+1}/{len(pages)}")
        page = page.convert("L") # Greyscale image
        enhancer = ImageEnhance.Contrast(page)
        page = enhancer.enhance(contrast_factor)
        enhancer = ImageEnhance.Sharpness(page)
        page = enhancer.enhance(sharpness_factor)
        enhanced_images.append(page)

        img_bytes = io.BytesIO()
        page.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        doc_stream = DocumentStream(name=f"page_{i+1}.png", stream=img_bytes)
        result = converter.convert(doc_stream)
        markdown_text = result.document.export_to_markdown()
        if check_readability(markdown_text):
            full_text += f"\n---- Seite {i+1} ----\n {markdown_text} \n"
        else:
            print(f"Page {i + 1}/{len(pages)} maybe rotated?")
            j = 0
            while True:
                j += 1
                if j > 1:   # experience shows, docling reads 180 degree rotation
                    print(f"Page {i+1} is not readable. Skipping.")
                    full_text += f"\n---- Seite {i+1} ----\n Nicht lesbar. \n"
                    break
                img_bytes = io.BytesIO()
                page.rotate(90)
                page.save(img_bytes, format="PNG")
                img_bytes.seek(0)
                doc_stream = DocumentStream(name=f"page_{i+1}_{j}.png", stream=img_bytes)
                result = converter.convert(doc_stream)
                markdown_text = result.document.export_to_markdown()
                if check_readability(markdown_text):
                    full_text += f"\n---- Seite {i + 1} ----\n {markdown_text} \n"
                    break
    return full_text

def extract_pdf_data(pdf_path, model_name=MODEL_OCR):
    """The basic idea is to take an PDF and extract the text using OCR and then run it through the Ollama model, with the text in prompt, to get the metadata and a more readable and correct version of the text."""
    if not check_and_download_model(model_name):
        print("Error downloading model. Exiting.")
        return None

    markdown_text = ocr_enhanced_images(pdf_path)

    # First correct the original text with MODEL_OCR
    print(f"Starting Ollama chat with model {model_name}...")
    start_time = time.time()
    fulltext = ""
    prompt = f"""
    Correct this inconsistent text with OCR errors.
            Original Text to be corrected:
            {markdown_text}
            Correct this text, keep the markdown-format (!) and return it as a string. Focus on grammatical and semantical CORRECTNESS! Keep text to German. Attention! OCR text may contain letters for numbers, like "B" for 6 or 8 and "S" for 5. Correct numbers in context.
            Also correct names or jargon if those words occur multiple times.
    """

    response = ollama.chat(
        model=model_name,
        messages=[
                    {"role": "user",
                    "content": prompt,
                    }
                ],
        stream=True,
        keep_alive=0,
        options={
            "temperature": 0.1,
            "num_ctx": 32768,
            "num_predict": 32768
        },
    )
    # for debugging, to see if anything is happening
    for chunk in response:
        content = chunk["message"]["content"]
        print (content, end="", flush=True)
        fulltext += content
    #print(f"Ollama response: '{response['message']['content']}'")

    end_time = time.time()
    print(f"Ollama chat completed in {end_time - start_time:.2f} seconds.")
    return fulltext

def get_law_xml(law_name):
    """Downloads the xml-zip from gesetze-im-internet and extracts the law text from it
    and saves it locally. Needs the name of the law as a string."""
    if law_name == "":
        return ""
    law_name = law_name.lower().strip()
    xml_filename = f"./laws/{law_name}.xml"
    if not os.path.exists("./laws"):
        os.mkdir("./laws")
    if not os.path.exists(xml_filename):
        # Check for file existence and download if necessary
        print(f"Lade {law_name} herunter...")
        response = requests.get(f"https://www.gesetze-im-internet.de/{law_name}/xml.zip")
        if response.status_code == 200:
            print(f"Heruntergeladen.")
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                xml_files = [f for f in z.namelist() if f.endswith(".xml")]
                with z.open(xml_files[0]) as f:
                    with open(xml_filename, "wb") as out_file:
                        out_file.write(f.read())
                        print(f"Datei {law_name}.xml heruntergeladen und gespeichert.")
        else:
            print(f"Fehler beim Herunterladen von {law_name}: {response.status_code}")
            return ""
    # XML parsing after checking for existence and/or downloading
    print(f"XML-Datei {law_name}.xml wird geparst...")
    tree = ET.parse(xml_filename)
    root = tree.getroot()
    paragraphs = []
    for norm in root.findall("norm"):
        metadaten = norm.find("metadaten")
        enbez = metadaten.find("enbez").text if metadaten.find("enbez") is not None else ""
        title = metadaten.find("titel").text if metadaten.find("titel") is not None else ""

        text_data = norm.find("textdaten/text")
        if text_data is not None:
            text = ET.tostring(text_data, encoding="unicode", method="text").strip()
            paragraphs.append({
                "id": f"{law_name.upper()} {enbez}",
                "title": title,
                "content": text
            })
    return paragraphs

def write_response(text_to_respond, lawDB=None, model_fast=MODEL_GIST, model_final=MODEL_RESPONSE) -> str:
    # Do a first search for laws
    search_prompt = f"""
    {text_to_respond}
    
    Liste die genannten Rechtsgebiete als Liste von Strings auf. Zum Beispiel aus der Titelzeile oder 
    anhand der häufigsten Nennungen im Haupttext.

    Format: Rechtsgebiet1++ Rechtsgebiet2++ Rechtsgebiet3++...
                
    """
    fulltext = ""
    response = ollama.chat(
        model=model_fast,
        messages=[{"role": "user", "content": search_prompt}],
        stream=True,
        format="json",
        keep_alive=0,
        options={
            "temperature": 0.1,
            "num_ctx": 32768,
            "num_predict": 32768
        }
    )
    for chunk in response:
        content = chunk["message"]["content"]
        print(content, end="", flush=True)
        fulltext += content

    law_list = fulltext.split("++")
    laws = []
    for law in law_list:
        print(f"Law: {law}")
        laws.append(lawDB.search(law,limit=3))

    prompt = f"""
    "{text_to_respond}"
    Get the main arguments of the text above, which is in markdown from an corrected OCR pdf. Return them as a JSON 
    JSON-Format: 
    "from": name/address sender,
     "to": name/address recipient, 
     "subject": subject line,
     "deadlines": ["YYYY-MM-DD"], 
     "laws_cited": Every cited or named law as a list.
     "legal_area": find the main legal area of the text, maybe from the subject line above the main-text. Use list of laws below
     "arguments": ["first short title": argument text1, second short title: argument text2,....],
     "links": other mentioned documents or statements
     
     Laws:
     {laws}
     
    From your work another AI will write a response to this arguments with other relevant information not contained in the text.
    Arguments have to be complete: GET ALL ARGUMENTS!
    """
    start_time = time.time()
    fulltext = ""
    while fulltext == "" or None:
        response = ollama.chat(
            model=model_fast,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            format="json",
            keep_alive=0,
            options={
                "temperature": 0.1,
                "num_ctx": 32768,
                "num_predict": 32768
                }

        )
        for chunk in response:
            content = chunk["message"]["content"]
            print (content, end="", flush=True)
            fulltext += content
    end_time = time.time()
    print(f"Ollama chat completed in {end_time - start_time:.2f} seconds.")
    try:
        cleaned_text = clean_json(fulltext)
        tmp_json = json.loads(cleaned_text)
    except json.JSONDecodeError:
        print(f"Error decoding JSON. Raw text returned.")
        tmp_json = {"fulltext": fulltext}

    laws = []
    if lawDB is not None:
        for argument in tmp_json["arguments"]:
            laws.append(lawDB.search(argument[1],limit=3))
        for law in tmp_json["laws_cited"]:
            laws.append(lawDB.search(law,limit=3))
        for law in tmp_json["legal_area"]:
            laws.append(lawDB.search(law,limit=3))
    else:
        laws = ["No relevant laws were mentioned."]
    research = ["No Reasearch was done on this topic."]
    mentioned_docs = ["No other documents were mentioned."]
    response_prompt = f"""
    Du bist ein anonymer Fachanwalt für Sozialrecht. Schreibe als dieser eine Antwort GEGEN die Argumente:
    {tmp_json}.
    
    Sei förmlich und klar. Beziehe dich auf die hier relevanten Gesetze:
    {laws}
    
    Und auf die Recherche:
    {research}
    
    Beziehe dich außerdem auf die genannten Dokumente, sofern vorhanden:
    {mentioned_docs}
    
    For Debug-reasons some texts are empty. Just ignore them.
    """
    start_time = time.time()
    response = ollama.chat(
        model=model_final,
        messages=[{"role": "user", "content": response_prompt}],
        stream=True,
        options = {
            "temperature": 0.5,
            "num_ctx": 32768,
            "num_predict": 32768
        }
    )
    for chunk in response:
        content = chunk["message"]["content"]
        print(content, end="", flush=True)
        fulltext += content
    end_time = time.time()
    print(f"Ollama chat completed in {end_time - start_time:.2f} seconds.")
    return fulltext

def main():
    for model in model_list:
        check_and_download_model(model)
    extract_pdf_text = ""
    # extracted_pdf_text = extract_pdf_data("LRA_StellungnahmeSGD_250115.pdf")
    with open("tmp_text.txt", "r") as f:
        extracted_pdf_text = f.read()

    # initialize social law texts
    i = 1
    while True:
        law_data = get_law_xml(f"sgb_{i}")
        if law_data == "":
            break
        else:
            manager = LawManagerLance()
            manager.add_law(law_data)
        i += 1
    get_law_xml("sgb_9_2018")
    get_law_xml("sgb_10")
    get_law_xml("sgb_11")
    get_law_xml("sgb_12")
    get_law_xml("kfzhv")

    print(extracted_pdf_text)
    # print(extract_pdf_data("Stellungnahme250128.pdf")[1])
    print(write_response(extracted_pdf_text, LawManagerLance()))


if __name__ == "__main__":
    main()