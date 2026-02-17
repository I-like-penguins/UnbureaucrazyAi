from ddgs.exceptions import DDGSException
from tavily import TavilyClient

from Lancedb_Manager import BrainDB as ManagerLance
import os
import ssl
from pathlib import Path
import ollama
import json
import time
import io
import re

from docling_core.types.io import DocumentStream
from pdf2image import convert_from_path
from PIL import ImageEnhance
from docling.document_converter import DocumentConverter

import requests
import zipfile
import xml.etree.ElementTree as ET

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Falls das OS es nicht unterstützt
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# 2. Umgebungsvariablen für zugrundeliegende HTTP-Libraries (httpx, requests)
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['PYTHONHTTPSVERIFY'] = '0'

from ddgs import DDGS
import httpx
import trafilatura
from playwright.sync_api import sync_playwright

MODEL_OCR = "qwen2.5:7b"
MODEL_GIST = "gemma3:12b" #"qwen2.5:7b"
MODEL_RESPONSE = "gemma3:12b"
MODEL_RAG = "phi3.5:latest"
MODEL_EMBEDDING = "nomic-embed-text:latest"

model_list = [MODEL_OCR, MODEL_GIST, MODEL_RESPONSE, MODEL_EMBEDDING, MODEL_RAG]

def get_prompt_result(prompt: str, model_name: str, stream=True,) -> str:
    """Runs a prompt through a model and returns the result as a string."""
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        stream=stream,
        keep_alive=0,
        options={
            "temperature": 0.1,
            "num_ctx": 32768,
            "num_predict": 32768
        }
    )

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
    if raw_content is None or "":
        print("Error: raw_content is None.")
        return "{}"
    open_braces = raw_content.count('{') - raw_content.count('}')
    open_brackets = raw_content.count('[') - raw_content.count(']')

    raw_content += ']' * open_brackets
    raw_content += '}' * open_braces

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
    pages = convert_from_path(pdf_path, 400)
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
            full_text += f"{markdown_text}\n++++#\n"
        else:
            print(f"Page {i + 1}/{len(pages)} maybe rotated?")
            j = 0
            while True:
                j += 1
                if j > 1:   # experience shows, docling reads 180 degree rotation
                    print(f"Page {i+1} is not readable. Skipping.")
                    full_text += f"Nicht lesbar.\n++++#\n"
                    break
                img_bytes = io.BytesIO()
                page.rotate(90)
                page.save(img_bytes, format="PNG")
                img_bytes.seek(0)
                doc_stream = DocumentStream(name=f"page_{i+1}_{j}.png", stream=img_bytes)
                result = converter.convert(doc_stream)
                markdown_text = result.document.export_to_markdown()
                if check_readability(markdown_text):
                    full_text += f"{markdown_text}\n++++#\n"
                    break
    return full_text

def extract_pdf_data(pdf_path, model_name=MODEL_OCR, save_as_file=""):
    """The basic idea is to take an PDF and extract the text using OCR and then run it through the Ollama model,
     with the text in prompt, to get the metadata and a more readable and correct version of the text.

     :param pdf_path: Path to the PDF file.
     :param model_name: Name of the Ollama model to use for OCR.
     :param save_as_file: Path to save the corrected text to. If empty, the corrected text is not saved.
     :return: The corrected text as a string. Returns an empty string if an error occurs or the text is not readable.
     """
    if save_as_file != "":
        path = Path(f"./brain/{save_as_file}_ocr.txt")
        if path.exists():
            print(f"File {path} already exists. Skipping OCR.")
            return path.read_text()
    if not os.path.exists(pdf_path):
        print(f"PDF file {pdf_path} not found. Exiting.")
        return ""
    if not check_and_download_model(model_name):
        print("Error downloading model. Exiting.")
        return None

    path = Path(f"./brain/{save_as_file}_markdown.txt")
    if path.exists():
        print(f"File {path} already exists. Skipping first step of OCR.")
        markdown_text = path.read_text()
    else:
        markdown_text = ocr_enhanced_images(pdf_path)
        if save_as_file != "":
            if write_tmp_file(markdown_text, f"./brain/{save_as_file}_markdown.txt"):
                print(f"Raw OCR text saved to '{save_as_file}_markdown.txt'.")

    # First correct the original text with MODEL_OCR
    print(f"Starting Ollama chat with model {model_name}...")
    start_time = time.time()
    fulltext = ""
    for i, page in enumerate(markdown_text.split("++++#")):
        prompt = f"""
                Korrigiere diese fehlerhafte OCR-Seite. Nutze Markdown-Format (!) und gib den korrigierten Text als
                String zurück. Behalte das Format bei! Erfinde nichts dazu! Original Text:
                {page}
                
                ACHTUNG! Text enthält womöglich Zahlen für Buchstaben und andersherum, z.B. B für 6 oder 8 oder A für 4.
                Gib den Text ohne eigene Einleitung und ohne eigenen Schlusssatz einfach nur so wieder wie korrigiert!
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
        fulltext += "\n---- Seite " + str(i+1) + " ----\n"

    print(f"Ollama chat completed in {time.time() - start_time:.2f} seconds.")
    if save_as_file != "":
        if write_tmp_file(fulltext, f"./brain/{save_as_file}_ocr.txt"):
            print(f"OCR corrected text saved to {save_as_file}_ocr.txt.")
        else:
            print(f"Error writing OCR corrected text to {save_as_file}_ocr.txt.")
    return fulltext

def write_tmp_file(text, file_path="") -> bool:
    """Writes a string to a temporary file. Returns True if successful, False otherwise."""
    if file_path != "":
        try:
            output_file = Path(file_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(text)
        except Exception as e:
            print(f"Error writing output file: {e}. Just returning text.")
            return False
        return True
    return False # return False if no file_path

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

def generate_search_query(argument_inhalt, argument_titel = "", model=MODEL_GIST):
    prompt = f"""
    Analysiere dieses juristische Argument und erstelle EINE präzise Research-Anfrage, 
    um relevante Urteile oder Kommentare zu finden. Sei prägnant und sparsam! Verwende WENIGE Worte!

    TITEL: {argument_titel}
    INHALT: {argument_inhalt}

    ANTWORTE NUR MIT DER SUCHANFRAGE. Kein "Hier ist die Suche...", keine Anführungszeichen.
    """
    response = ollama.chat(model=model, messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content'].strip()

def perform_web_search(query, max_results=1):
    print(f"--- Suche im Web nach: {query} ---")
    sources = [
        "site:rehadat-recht.de",
        "site:sozialgerichtsbarkeit.de",
        "site:bundessozialgericht.de",
    ]
    js_heavy_domains = ["sozialgerichtsbarkeit.de", "bundessozialgericht.de"]

    results_formatted = []
    site_filter = "(" + " OR ".join(sources) + ")"

    full_query = f"{query} {site_filter}"
    results_formatted = []
    with DDGS(timeout=30) as ddgs:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
            )
            page = context.new_page()
            try:
                results = list(ddgs.text(full_query, max_results=max_results))
                web_context = ""
                for r in results:
                    url = r["href"]
                    print(f"Crawling {url}")
                    if any(domain in url for domain in js_heavy_domains):
                            try:
                                page.goto(url, wait_until="networkidle", timeout=30000)
                                page.wait_for_timeout(1000)
                                html_content = page.content()
                                full_text = trafilatura.extract(html_content)
                                browser.close()
                            except Exception as e:
                                print(f"Error loading page: {e}")
                                browser.close()
                                continue
                    else:
                        page_download = trafilatura.fetch_url(url)
                        full_text = trafilatura.extract(page_download)
                    print(f"Download:\n {full_text}")   #TODO: Delete this line.
                    if full_text is not None:
                        results_formatted.append(f"QUELLE: {r['title']}\nURL: {url}\nTEXT: {full_text}")
            except DDGSException as e:
                browser.close()
            except ddgs.exceptions.TimeoutException as te:
                print(f"Timeout loading page: {te}")
                browser.close()

        return "\n\n---\n\n".join(results_formatted)

def perform_tavily_search(query):
    print(f"--- Suche in Tavily nach: {query} ---")
    tavily_client = TavilyClient(api_key="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")
    response = tavily_client.search(query)
    return response

def page_to_json(path: Path, text, page=1, model=MODEL_GIST, lanceDB=None):
    prompt = f"""
                        {text}
                        Extrahiere folgende Informationen von dieser Seite, für einen späteren Denkprozess!
                        NUR DEUTSCH. NUR JSON-Format:
                        {{
                            "Seite: {page}
                            "Inhalt": {{
                                "Fristen": ["YYYY-MM-DD"], 
                                "Gesetze": Alle konkret benannten Gesetze als Liste.
                                "Rechtsgebiet": Finde das hauptsächliche Rechtsgebiet dieser Seite.
                                "Argument": Zusammenfassung des Inhalts dieser Seite. Sei AUSFÜHRLICH und FAKTENTREU.
                                "Verweise": Andere Dokumente, die genannt oder auf die verwiesen wurde.
                                }}
                        }}
                    """
    start_time = time.time()
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        format="json",
        options={
            "temperature": 0.1,
            "num_ctx": 16384,
            "num_predict": 16384 * 2
        }
    )
    page_json = ""
    for chunk in response:
        content = chunk["message"]["content"]
        print(content, end="", flush=True)
        page_json += content
    print(f"Ollama chat completed in {time.time() - start_time:.2f} seconds.")

    if lanceDB is not None:
        manager = lanceDB
        manager.add_page_json(str(path), page_json, page)
    return page_json

def write_response(text_to_respond, lanceDB=None, model_fast=MODEL_GIST, model_final=MODEL_RESPONSE, save_as_file="") -> str:
    # Do a first search for laws
    tmp_json = {}
    json_list = []
    global_search_results = ""
    arguments = {}      # Title: argument
    # split text into pages to reduce memory usage for AI - long texts destroy prompt
    # get JSON of every page for later processing
    page_before = ""
    fulltext = ""
    for i, page in enumerate(text_to_respond.split("\n---- ")):
        page_json = ""
        if i == 0:
            continue        # the first split is useless, skip it
        path = Path(f"./brain/{save_as_file}_page_{i}.json")
        if path.exists():
            print(f"File {path} already exists. Skipping creating JSON from page {i}.")
            json_list.append(path.read_text())
            page_before = path.read_text()
        else:
            page_json = page_to_json(path,f"### DIESE Seite: {page}\n### VORHERIGE Seite: {page_before}", i,
                                     model=model_fast, lanceDB=ManagerLance())
            if write_tmp_file(page_json, f"./brain/{save_as_file}_page_{i}.json"):
                print(f"JSON saved to '{save_as_file}_page_{i}.json'.")
        page_before = page
        json_list.append(page_json)

    # get all mentioned laws from the database
    laws = []
    laws_cited = []
    for json_page in json_list:
        try:
            tmp_json = json.loads(json_page)
            for law in tmp_json["Inhalt"]["Gesetze"]:
                if law not in laws_cited:
                    laws_cited.append(law)
                    laws.append(lanceDB.search(law, "law_table", limit=1))
            laws.append(lanceDB.search(tmp_json["Inhalt"]["Rechtsgebiet"], "law_table", limit=3))
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON. Error: {e}")
            try:
                tmp_json = json.loads(json_page)
            except json.JSONDecodeError as e:
                print(f"Not possibe to encode: '{json_page}'")
                continue
        except KeyError as e:
            print(f"Error decoding JSON. Error: {e}")
            continue
    blnExit = False
    i = 0
    thinking_process = ""
    task = "Analysiere das vorliegende Dokument, damit daraus später ein Antwortschreiben generiert werden kann."
    path = Path(f"./brain/{save_as_file}_page_1.json")
    doc_db = lanceDB.search(str(path), "doc_table", limit=5)
    while not blnExit:
        # Start thinking process until satisfied or loop-kill, do websearches or querries to database automatically.
        thinking_prompt = f"""
        Dein Auftrag:
        {task}
        
        Gefundene Dokumentauszüge:
        {doc_db}
        
        ### DENKPROZESS!!! ###
        ### Runde {i+1}: ###
        
        Wähle NUR EINES:
        Wenn Denken erfüllt, Schreibe nur "EXIT"
        Wenn Websuche erforderlich, Schreibe "WEBSEARCH:" gefolgt von der Suchanfrage. Halte dich kurz.
        Wenn Suche in Gesetze-Vektordatenbank erforderlich, Schreibe "GESETZE:" gefolgt von dem Paragraphen und dem Gesetzbuch
        Wenn Denkprozess, Schreibe "PROCESS:" gefolgt von einem Absatz an Gedanken zum Text.
        Wenn sich der Auftrag ändern soll, Schreibe "NEU:" gefolgt von einem neuen Auftrag.
        
        Nutze folgende Gesetze aus der Vektor-Datenbank:
        {laws}
        
        Letzte Denkprozesse (letzten 500 Zeichen):
        {thinking_process[:500]}
        
        Bisherige Suchergebnisse:
        {global_search_results}
        """
        start_time = time.time()
        response = ollama.chat(
            model=model_final,
            messages=[{"role": "user", "content": thinking_prompt}],
            stream=True,
            options={
                "temperature": 0.4,
                "num_ctx": 32768*1.5,
                "num_predict": 32768/2
            }
        )
        for chunk in response:
            content = chunk["message"]["content"]
            print(content, end="", flush=True)
            fulltext += content
        end_time = time.time()
        print(f"Ollama chat completed in {end_time - start_time:.2f} seconds.")
        if "EXIT" in fulltext or i > 29:
            print("Exiting thinking process.")
            blnExit = True
        elif "WEBSEARCH:" in fulltext:
            print("Starting web search.")
            query = fulltext.split("WEBSEARCH:")[1].strip()
            global_search_results.append(perform_tavily_search(query))
        elif "GESETZE:" in fulltext:
            print("Starting law search.")
            laws.append(lanceDB.search(fulltext.split("GESETZE:")[1].strip(), "law_table", limit=3))
        elif "PROCESS:" in fulltext:
            print("Starting process step.")
            thinking_process += f"Runde: {i+1}\n{fulltext.split("PROCESS:")[1].strip()}"
        elif "NEU:" in fulltext:
            task = fulltext.split("NEU:")[1].strip()
        i += 1

    # TODO: implement devil's advocat to get a more rounded analysis

    arg_prompt = f"""
        {arguments}
        Für die vorstehende LISTE an Argumenten (Titel: Argument), erstelle in DEUTSCH eine rechtliche vollständige Analyse. 
        Nenne zuerst das Argument gefolgt von einer detaillierten kritischen Analyse der Rechtslage.
                
        Nutze dafür die folgenden Gesetze aus der Vektor-Datenbank:
        {laws}
                
        Nutze die Ergebnisse der Websuche, falls relevant:
        {global_search_results}
    """
    start_time = time.time()
    response = ollama.chat(
        model=model_fast,
        messages=[{"role": "user", "content": arg_prompt}],
        stream=True,
        options={
            "temperature": 0.4,
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

    laws = []
    if lanceDB is not None:
        for law in tmp_json["laws_cited"]:
            laws.append(lanceDB.search(law,"law_table",limit=5))
        for law in tmp_json["legal_areas"]:
            laws.append(lanceDB.search(law,"law_table",limit=3))
        for text in global_search_results:
            laws.append(lanceDB.search(text,"law_table",limit=3))
    else:
        laws = ["No relevant laws were mentioned."]
    mentioned_docs = ["No other documents were mentioned."]
    response_prompt = f"""
    Du bist ein anonymer Fachanwalt für Sozialrecht. Du repräsentierst die Gegenseite zu den oben vorgelegten Texten.
    Deine Vorarbeiter haben die Argumente bereits analysiert. Fasse alles mit unten stehenden Informationen auf DEUTSCH
    zusammen:
    Sei förmlich und klar. Versuche zu überzeugen! 
    
    Beziehe dich auf die hier relevanten Gesetze:
    {laws}
    
    Und auf die Recherche:
    {global_search_results}
    
    Beziehe dich außerdem auf die genannten Dokumente, sofern vorhanden:
    {mentioned_docs}
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
    extracted_pdf_text = extract_pdf_data("Schreiben_SG_LRA_Widerspruchsbescheid-Urlaub25.pdf", save_as_file="Schreiben_SG_LRA_Widerspruchsbescheid-Urlaub25")

    # initialize social law texts
    i = 1
    manager = ManagerLance()
    while True:
        law_data = get_law_xml(f"sgb_{i}")
        if law_data == "":
            break
        else:
            manager.add_law(law_data)
        i += 1
    manager.add_law(get_law_xml("sgb_9_2018"))
    manager.add_law(get_law_xml("sgb_10"))
    manager.add_law(get_law_xml("sgb_11"))
    manager.add_law(get_law_xml("sgb_12"))
    manager.add_law(get_law_xml("kfzhv"))
    manager.add_law(get_law_xml("sgg"))
    manager.add_law(get_law_xml("gg"))
    manager.add_law(get_law_xml("bgb"))
    manager.add_law(get_law_xml("agg"))

    print(extracted_pdf_text)
    # print(extract_pdf_data("Stellungnahme250128.pdf")[1])
    print(write_response(extracted_pdf_text, save_as_file="Schreiben_SG_LRA_Widerspruchsbescheid-Urlaub25"))

if __name__ == "__main__":
    main()