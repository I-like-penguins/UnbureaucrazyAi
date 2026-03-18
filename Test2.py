from ddgs.exceptions import DDGSException
from ddgs import DDGS
import httpx
import trafilatura
from playwright.sync_api import sync_playwright
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
import fitz

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

# Umgebungsvariablen für zugrundeliegende HTTP-Libraries (httpx, requests)
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['PYTHONHTTPSVERIFY'] = '0'


MODEL_OCR = "qwen2.5:7b"
MODEL_GIST = "deepseek-r1:8b"
MODEL_RESPONSE = "gemma3:12b"
MODEL_RAG = "phi3.5:latest"
MODEL_EMBEDDING = "nomic-embed-text:latest"
MODEL_VLM = "glm-ocr"

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
    if raw_content is None or "":
        print("Error: raw_content is None.")
        return "{}"

    cleaned = re.sub(r'```json\s?|\s?```', '', raw_content).strip()
    match = re.search(r'(\{.*\})', cleaned, re.DOTALL)

    if match:
        cleaned = match.group(1)

    return cleaned

def check_readability(text):
    """Checks a text against occurrences of high-probability German words or if it is empty. True if readable"""
    if len(text) < 20:  # Empty pages are to be skipped
        return False
    word_list = ["der", "die", "das", "und", "oder", "nicht", "wir", "ihr", "sie", "ist", "mit", "von", "den"]
    count = sum(1 for word in word_list if word in text.lower())
    return count > 1

def is_scanned_pdf(pdf_path) -> bool:
    """Checks if a PDF is scanned or not. Returns True if scanned, False if not scanned."""
    doc = fitz.open(pdf_path)
    for page in doc:
        if page.get_text().strip() or page.get_fonts():
            return False
    return True

def extract_pdf_vlm(pdf_path, save_file=True, output_folder="./brain", model=MODEL_VLM):
    """VLM Test"""
    path_str = f"{output_folder}/{pdf_path.split(".")[0]}_ocr.txt"
    full_text = ""
    if save_file and output_folder:
        path = Path(path_str)
        if path.exists():
            print(f"OCR-file already exists at {path}. Using existing file.")
            return path.read_text()
    print(f"Starting OCR...")
    if not is_scanned_pdf(pdf_path):
        print("PDF is not scanned. Extracting text directly from PDF.")
        doc = fitz.open(pdf_path)
        full_text = f"\n ---- Ende der Seite ----#\n".join(page.get_text() for page in doc)
    else:
        images = convert_from_path(pdf_path, 250)
        full_text = ""
        for i, image in enumerate(images):
            image = image.convert("L")  # Greyscale image
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(2.0)
            image_path = os.path.abspath(f"tmp_img_{i}.png")
            image.save(image_path, "PNG")
            print(f"Processing page {i+1}...")
            start_time = time.time()
            with open(image_path, 'rb') as f:
                stream = ollama.chat(
                    model=model,
                    stream=True,
                    messages=[{
                        'role': 'user',
                        'content': f""""Extrahiere den Text dieser Seite. STARTE UNBEDINGT beim Briefkopf (Absender/Empfänger). Arbeite dich strikt von OBEN nach UNTEN vor. Ignoriere die Fußzeile vorerst!""",
                        'images': [f.read()]}
                    ],
                    options={'temperature': 0.0, 'top_k': 40, 'top_p': 0.9, 'seed': 42, 'num_predict': 4069, 'num_ctx': 8192*3, 'repeat_penalty': 1.5}
                )
                for chunk in stream:
                    content = chunk['message']['content']
                    print(content, end='', flush=True)
                    full_text += content
            os.remove(image_path)
            print(f"\nPage {i + 1} processed in {time.time() - start_time:.2f} seconds.")
            full_text += f"\n ---- Ende Seite {i+1} ----#\n"
            time.sleep(2)

    if save_file and output_folder:
        if write_tmp_file(full_text, path_str):
            print(f"Saved ocr-text to file {path_str}")
    return full_text

def write_tmp_file(text, file_path="") -> bool:
    """Writes a string to a temporary file. Returns True if successful, False otherwise."""
    if file_path != "":
        try:
            output_file = Path(file_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(text)
        except Exception as e:
            print(f"Error writing output file: {e}. Printing text instead.")
            print(text)
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
    tavily_client = TavilyClient()
    # response = tavily_client.search(query)
    # print(f"Tavily response: {response}")   # TODO: Delete this line.
    # return response
    return "Websuche noch nicht implementiert! FÜhre eine Abfrage in der Datenbank durch."

def page_to_json(path: Path, text, page=1, model=MODEL_GIST, lanceDB=None):
    prompt = f"""
                        {text}
                        Extrahiere folgende Informationen von dieser Seite, für einen späteren Denkprozess!
                        NUR DEUTSCH. NUR JSON-Format:
                        {{
                            "Seite": {page},
                            "Fristen": ["YYYY-MM-DD"], 
                            "Gesetze": Alle konkret benannten Gesetze als Liste.
                            "Rechtsgebiet": Finde das hauptsächliche Rechtsgebiet dieser Seite.
                            "Argument": Zusammenfassung des Inhalts dieser Seite. Sei AUSFÜHRLICH und FAKTENTREU.
                            "Verweise": Andere Dokumente, die genannt oder auf die verwiesen wurde.
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
            "num_ctx": 32768,
            "num_predict": 32768
        }
    )
    page_json = ""
    for chunk in response:
        content = chunk["message"]["content"]
        print(content, end="", flush=True)
        page_json += content
    print(f"Ollama chat completed in {time.time() - start_time:.2f} seconds.")
    try:
        if lanceDB is not None:
            manager = lanceDB
            manager.add_page_json(str(path), page_json, page)
        return page_json
    except:
        print()

def write_response(text_to_respond, lanceDB=None, model_fast=MODEL_GIST, model_final=MODEL_RESPONSE, save_as_file="") -> str:
    json_list = []
    global_search_results = []
    # split text into pages to reduce memory usage for AI - long texts destroy prompt
    # get JSON of every page for later processing
    page_before = ""
    fulltext = ""
    for i, page in enumerate(text_to_respond.split("----#")):
        path = Path(f"./brain/{save_as_file}_page_{i}.json")
        if path.exists():
            print(f"File {path} already exists. Skipping creating JSON of page {i}.")
            json_list.append(path.read_text())
            page_before = path.read_text()
        else:
            page_json = page_to_json(path,f"### DIESE Seite: {page}\n### VORHERIGE Seite: {page_before}", i,
                                     model=model_fast)
            if write_tmp_file(page_json, f"./brain/{save_as_file}_page_{i}.json"):
                print(f"JSON saved to '{save_as_file}_page_{i}.json'.")
            page_before = page
            json_list.append(page_json)
    i = 0
    thinking_process = ""
    parts_of_answer = []
    # doc_db = lanceDB.search(f"./brain/{save_as_file}_page_*.json", "page_table", limit=8)
    path = Path(f"./brain/{save_as_file}_ocr.txt")
    doc_length = len(path.read_text().split("----#"))
    for current_page in path.read_text().split("----#"):
        # Start analyzing text until satisfied or loop-kill, do websearches or querries to database automatically.
        print(f"Seite: {i} - Denkprozess.......")     # TODO: Delete this line
        thinking_prompt = f"""
                    #### Analysiere die vorliegende Dokumentenseite, damit eine KI damit arbeiten kann. Stelle auch FRAGEN! ####
                    SEITE: {i}/{doc_length}
                    +++++++
                    Zu analysierende Seite: "{current_page}"
                    +++++++
                    #### HALTE DICH KURZ!! RELEVANTE INFORMATIONEN MÜSSEN VOLLSTÄNDIG SEIN!!! ####
                    #### WICHTIG!! - Seiten ohne juristischen Inhalt sind zu ignorieren!! ####
                """
        start_time = time.time()
        response = ollama.chat(
            model=model_fast,
            messages=[{"role": "user", "content": thinking_prompt}],
            stream=True,
            think=True,
            options={
                "temperature": 0.35,
                "num_ctx": 32768*1.5,
                "num_predict": 32768
            }
        )
        fulltext = ""
        is_thinking = False
        for chunk in response:
            if chunk.message.thinking and not is_thinking:
                is_thinking = True
                print("Thinking...")
            if chunk.message.thinking:
                print(chunk.message.thinking, end='')
            elif chunk.message.content:
                if is_thinking:
                    print('\n\nAnswer:\n', end='')
                    is_thinking = False
                print(chunk.message.content, end='')
            fulltext += chunk.message.content
        end_time = time.time()
        print("")
        print(f"Ollama chat completed in {end_time - start_time:.2f} seconds.")
        thinking_process += f"Runde: {i + 1}|{fulltext.strip()}"

        path = Path(f"./brain/{save_as_file}_page_{i}.json")
        if not path.exists():
            i += 1
            continue
        json_page = path.read_text()
        laws = []
        laws_cited = []
        try:
            tmp_json = json.loads(json_page)
            for law in tmp_json["Gesetze"]:
                if law not in laws_cited:
                    laws_cited.append(law)
                    laws.append(lanceDB.search_law_id(law))
                    laws.append(lanceDB.search(law, "law_table", limit=2))
            laws.append(lanceDB.search(tmp_json["Rechtsgebiet"], "law_table", limit=1))
            laws.append(lanceDB.search(tmp_json["Argument"], "law_table", limit=1))
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON. Error: {e}")
            i += 1
            continue
        except KeyError as e:
            print(f"Error decoding JSON. Error: {e}")
            i += 1
            continue
        control_prompt = f"""
            #### Prüfe die vorliegende Seite!! Verfasse ggf. eine Antwort GEGEN die vorgebrachten falschen Argumente! Beziehe Gesetze mit ein! ####
            ### HALTE DICH KURZ! Keine Anrede oder Schlussformel! Kein Fazit!! ###
            Jetzige Runde: {i}/{doc_length}
            Bisherige Ausgaben der KI (inkl. diese Runde):
            {thinking_process.split("Runde: ")[-3] if len(thinking_process.split("Runde: ")) > 3 else thinking_process}
            
            Gesetze:
            {laws}
            
            #### IGNORIERE SEITEN OHNE JURISTISCHE RELEVANZ!!! ####
        """
        print("Kontrollprozess....")
        start_time = time.time()
        response = ollama.chat(
            model=model_fast,
            messages=[{"role": "user", "content": control_prompt}],
            stream=True,
            think=True,
            options={
                "temperature": 0.2,
                "num_ctx": 32768,
                "num_predict": 32768}
        )
        fulltext = ""
        is_thinking = False
        for chunk in response:
            if chunk.message.thinking and not is_thinking:
                is_thinking = True
                print("Thinking...")
            if chunk.message.thinking:
                print(chunk.message.thinking, end='')
            elif chunk.message.content:
                if is_thinking:
                    print('\n\nAnswer:\n', end='')
                    is_thinking = False
                print(chunk.message.content, end='')
            fulltext += chunk.message.content
        end_time = time.time()
        print(f"Ollama chat completed in {end_time - start_time:.2f} seconds.")
        parts_of_answer.append(fulltext)
        i += 1

    # TODO: implement devil's advocat to get a more rounded analysis

    arg_prompt = f"""
        {parts_of_answer}
        Du bist ein anonymer FACHANWALT für Sozialrecht. Aus der vorstehenden LISTE, erstelle in DEUTSCH eine 
        rechtlich vollständige Antwort. 
        Nenne zuerst das Argument gefolgt von einer detaillierten kritischen Analyse der Rechtslage.
                
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

    return fulltext

def main():
    for model in model_list:
        check_and_download_model(model)
    manager = ManagerLance()
    tmp_txt = extract_pdf_vlm("")

    # initialize social law texts
    i = 1

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

    print("test")
    print(tmp_txt)
    # print(extract_pdf_data("Stellungnahme250128.pdf")[1])
    if write_tmp_file(write_response(tmp_txt, manager, save_as_file=""),"_AW.txt"):
        print("Successfully saved response!")
if __name__ == "__main__":
    main()
