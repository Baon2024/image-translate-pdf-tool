


## get images
from pathlib import Path
import img2pdf
import ocrmypdf
from liteparse import LiteParse
import sys

from google import genai
from google.genai import types
from ollama import chat
from pydantic import Field, BaseModel
from typing import List, Any
from dotenv import load_dotenv
import json
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate, Spacer
import argparse
from rich.console import Console
from tqdm import tqdm


load_dotenv()

console = Console()

clientGemini = genai.Client()

def infer_page_numbers(ready_to_assemble):
    known = [
        (i, item[0])
        for i, item in enumerate(ready_to_assemble)
        if item[0] is not None
    ]

    inferred = []

    if not known:
        for i, item in enumerate(ready_to_assemble, start=1):
            _, content = item
            inferred.append((i, content))
        return inferred

    for i, item in enumerate(ready_to_assemble):
        page_number, content = item

        if page_number is not None:
            inferred.append((page_number, content))
            continue

        previous_known = [(idx, page) for idx, page in known if idx < i]

        if previous_known:
            idx, page = previous_known[-1]
            inferred_page_number = page + (i - idx)
        else:
            idx, page = known[0]
            inferred_page_number = page - (idx - i)

        inferred.append((inferred_page_number, content))

    return inferred


class Footnote(BaseModel):
    number: int | None = Field(
        default=None,
        description="The footnote number, if the footnote has one. If it doesn't have one, because it's continuing a footnote from a previous page, don't return this."
    )
    footnote_text: str = Field(
        ...,
        description="the text of the footnote"
    )
    continuation_previous_footnote: bool = Field(
        ...,
        description="If the footnote is continuing a previous footnote from a previous page, and doesn't have footnote number, return True. Otherwise, return False"
    )
    continuation_previous_footnote_number: int | None = Field(
        default=None,
        description="If the footnote is continuting a previous footnote, return what the footnote number must be - (whatever the next numbered footnote is, minus 1, will give you the footnote number it must be continuing)"
    )
    ## add in what the previous footnumber must have been - based on what the next one is

class TranslationResult(BaseModel):
    title: str | None = Field(
        default=None,
        description="if the title of the article or PhD is on this page in the image, return it. Otherwise, leave this blank. If you return a title, don't return it's text in the translated_main_text section. The title is usually found in the main part of the page, but separate to the main text body."
    )
    page_number: int | None = Field(
        ...,
        description="the page number of the article page in the image. If no page number is visible, return. The article name or author name at the top of the page does not count as main text."
    )
    translated_main_text: str = Field(
        default="",
        description="the main text of the image. Do not return the title in this."
        )
    translated_footnotes: List[Footnote] | None = Field(
        default_factory=list,
        description="the footnotes of the main text, if they exist in the image"
        )

class TranslationResultBatch(BaseModel):
    pages: list[TranslationResult] = Field(
        ...,
        description="pages are a list of TranslationResult class, one per image/page"
    )

def call_gemini_structured_output(image, response_model, model_name, language):

     ## how much cheaper can i get, while still getting sufficient quality?

    system_instruction = f""""You are a text extractor and translator of academic articles to English
    You need to extract all the details of the schema from the image you are given, and return a response that matches the schema.
    
    *Rules and Guidelines*
    - only translates from the specified target language ({language}) to English - return the rest of the text in the original language
    
    Footnote continuation rule:
  If there is footnote text at the bottom of the page before the first visible numbered footnote, and that text does not itself begin
  with a footnote number, treat it as a continuation of the previous page's footnote.

  Do not attach that unnumbered text to the first numbered footnote on the current page.

  For example, if the first visible numbered footnote on the page is 57, and there is unnumbered footnote text immediately before it,
  return that unnumbered text as:
  - number: null
  - footnote_text: <the text>
  - continuation_previous_footnote: true
  - continuation_previous_footnote_number: 56
    """ ## need to actually add this in, stupid

    mime_type = image.suffix.lower()

    mime_types_dict = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }

    contents = []
    contents.append(types.Part.from_bytes(data=image.read_bytes(), mime_type=mime_types_dict[mime_type]))
    contents.append(f"Extract the text from this image, translate from {language} into English, and return in the format specified.")
    

    response = clientGemini.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=response_model
            )
        )
    usage = response.usage_metadata

    print("Input tokens:", usage.prompt_token_count)
    print("Output tokens:", usage.candidates_token_count)
    print("Thoughts:", usage.thoughts_token_count)
    print("Total tokens:", usage.total_token_count)

    ## need to ensure i track token costs - so i get a baseline for now.
    
    response_parsed = response.parsed
    return { "response": response_parsed, "input_tokens": usage.prompt_token_count or 0, "output_tokens": usage.candidates_token_count or 0, "total_tokens": usage.total_token_count or 0, "thought_tokens": usage.thoughts_token_count or 0}   # should return correctly

def call_gemini_structured_output_batch(batch_waitlist, response_model, model_name, language):

     ## how much cheaper can i get, while still getting sufficient quality?

    system_instruction = f""""You are a text extractor and translator of academic articles to English
    You need to extract all the details of the schema from the image you are given, and return a response that matches the schema.
    
    *Rules and Guidelines*
    - only translates from the specified target language ({language}) to English - return the rest of the text in the original language
    
    Footnote continuation rule:
  If there is footnote text at the bottom of the page before the first visible numbered footnote, and that text does not itself begin
  with a footnote number, treat it as a continuation of the previous page's footnote.

  Do not attach that unnumbered text to the first numbered footnote on the current page.

  For example, if the first visible numbered footnote on the page is 57, and there is unnumbered footnote text immediately before it,
  return that unnumbered text as:
  - number: null
  - footnote_text: <the text>
  - continuation_previous_footnote: true
  - continuation_previous_footnote_number: 56
    """ ## need to actually add this in, stupid

    ## should the instructions for dealing with footnotes be different, when it can actually see what they continue??


    image_waitlist = []

    mime_types_dict = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }

    for image in batch_waitlist:
        mime_type = image.suffix.lower()
        image_waitlist.append(types.Part.from_bytes(data=image.read_bytes(), mime_type=mime_types_dict[mime_type]))
    

    contents = []
    contents.extend(image_waitlist)
    contents.append(f"Extract the text from these images, translate from {language} into English, and return in the format specified.")
    

    ## need to use a variant of translationResult, that handles multiple pages

    response = clientGemini.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=response_model
            )
        )
    usage = response.usage_metadata

    print("Input tokens:", usage.prompt_token_count)
    print("Output tokens:", usage.candidates_token_count)
    print("Thoughts:", usage.thoughts_token_count)
    print("Total tokens:", usage.total_token_count)

    ## need to ensure i track token costs - so i get a baseline for now.
    if response.parsed:
        response_parsed = response.parsed.pages
    else:
        response_parsed = None
    return { "response": response_parsed, "input_tokens": usage.prompt_token_count or 0, "output_tokens": usage.candidates_token_count or 0, "total_tokens": usage.total_token_count or 0, "thought_tokens": usage.thoughts_token_count or 0}  # should return correctly


def find_image_directory(imagesCollectionPath):
    dir = Path(imagesCollectionPath)
    
    console.print("[yellow].. Searching for image directory ..[/yellow]")
    dir_full_path = dir.resolve()

    
    if not dir_full_path.exists():
        console.print("[red].. Image directory not found ..[/red]")
        raise FileNotFoundError(f"folder of images to extract and translate not found: at {imagesCollectionPath}")
    
    if not dir_full_path.is_dir():
        console.print("[red].. ImageCollectionPath is not a directory ..[/yellow]")
        raise NotADirectoryError(f"--- url passed {imagesCollectionPath} is not a folder! ---")
    
    console.print("[green].. Image directory found ..[/green]")
    return dir_full_path


def core_batch_retry_handler(batch_waitlist, token_usage, current_batch_number, model, language, retries_allowed, run_usage):
    
    translated_text_local = []
    num_of_tries = retries_allowed
    failed_after_retries = False
    for attempt in range(1, num_of_tries + 1):
            if attempt > 1:
                print(f"text extraction and translation for batch {current_batch_number} return None - retrying for the {attempt} time.. ")
            extracted_and_translated_text_result = call_gemini_structured_output_batch(batch_waitlist, TranslationResultBatch, model, language)
            print(f"extracted and translated text for batch {current_batch_number} is: {extracted_and_translated_text_result['response']}")

            run_usage.append({
                "mode": "batch",
                "page": None,
                "batch": current_batch_number,
                "batch_size": len(batch_waitlist),
                "attempt": attempt,
                "success": extracted_and_translated_text_result["response"] is not None,
                "input_tokens": extracted_and_translated_text_result.get("input_tokens", 0),
                "output_tokens": extracted_and_translated_text_result.get("output_tokens", 0),
                "thought_tokens": extracted_and_translated_text_result.get("thought_tokens", 0),
                "total_tokens": extracted_and_translated_text_result.get("total_tokens", 0),
            })

            print(f"text extraction and translation for batch {current_batch_number} return None - retrying for the {attempt} time.. ")
            
            if extracted_and_translated_text_result["response"] is not None:
                translated_text_local.extend(extracted_and_translated_text_result["response"])
                print(f"extracted and translated text for batch {current_batch_number} is: {extracted_and_translated_text_result['response']}")
                return translated_text_local, failed_after_retries    
        
    print(f"failed to extract and translate batch {current_batch_number} after {num_of_tries} attempts: - ending translation run..")
    failed_after_retries = True
    return translated_text_local, failed_after_retries

def extract_text_and_translate_batch(images_to_convert, token_usage, batch_size, model, language, retries_allowed, run_usage):
    
    current_batch_number = 0
    current_batch_total = 0 ## count each
    batch_waitlist = [] ## store waiting batches here
    translated_text = []
    failed_after_retries = False 

    for i, image in enumerate(tqdm(images_to_convert)):
        if current_batch_total == batch_size:
            ## translate existing batch group
            current_batch_number += 1
            result, failed_after_retries = core_batch_retry_handler(batch_waitlist, token_usage, current_batch_number, model, language, retries_allowed, run_usage)
            batch_waitlist = []
            if failed_after_retries:
                return translated_text, failed_after_retries
            translated_text.extend(result)
            batch_waitlist.append(image)
            current_batch_total = 1
        else:
            batch_waitlist.append(image)
            current_batch_total += 1

        ## but last batch will always be short of batch number ??
    ## so, if current_batch_total ? 0 on last batch, call
    if current_batch_total > 0:
        ## translate existing batch group
        current_batch_number += 1
        result, failed_after_retries = core_batch_retry_handler(batch_waitlist, token_usage, current_batch_number, model, language, retries_allowed, run_usage)
        if failed_after_retries:
                return translated_text, failed_after_retries
        translated_text.extend(result)
        current_batch_total = 0 ## reset batches in waitlist, for next batch
    
    return translated_text, failed_after_retries


def extract_text_and_translate(images_to_convert, token_usage, model, language, retries_allowed, run_usage):
    translated_text = []
    failed_after_retries  = False

    for i, image in enumerate(tqdm(images_to_convert)):
        num_of_tries = retries_allowed
        for attempt in range(1, num_of_tries + 1):
                if attempt > 1:
                    console.print(f"[yellow].. text extraction and translation for page {i} return None - retrying for the {attempt} time.. [/yellow]")
                
                extracted_and_translated_text_result = call_gemini_structured_output(image, TranslationResult, model, language)
                print(f"extracted and translated text for image {i} is: {extracted_and_translated_text_result['response']}")
                
                run_usage.append({
                    "mode": "single",          # or "batch"
                    "page": i,
                    "batch": None,
                    "attempt": attempt,
                    "success": extracted_and_translated_text_result["response"] is not None,
                    "input_tokens": extracted_and_translated_text_result.get("input_tokens", 0),
                    "output_tokens": extracted_and_translated_text_result.get("output_tokens", 0),
                    "thought_tokens": extracted_and_translated_text_result.get("thought_tokens", 0),
                    "total_tokens": extracted_and_translated_text_result.get("total_tokens", 0),
                })

                if extracted_and_translated_text_result["response"] is not None:
                    translated_text.append(extracted_and_translated_text_result["response"])
                    print(f"extracted and translated text for image {i} is: {extracted_and_translated_text_result['response']}")
                    break
                else:
                    if attempt < 3:
                        console.print("[red].. GEMINI LLM call did not return a response - will re try ..[/red]")
        else:
            console.print("[red].. GEMINI LLM call did not return a response - ran out of retries[/red]")
            failed_after_retries = True
            return translated_text, failed_after_retries
                
    print(f"failed to extract and translate page {i} after {num_of_tries} attempts: {image} - ending translation run..")        
    return translated_text, failed_after_retries                    


import os
import keyring
import getpass

def get_gemini_key():
    
    ## otehrwise, fetch key if it exists, or save with getpass
    key = keyring.get_password('images_to_translated_pdf', "GEMINI_API_KEY")
    console.print("[green].. GEMINI_API_KEY already saved to keyring - will use this .. [/green]")
    if key:
        return key
    
    console.print("[yellow].. no existing GEMINI_API_KEY saved to keyring - you need to provide one .. [/yellow]")
    key = getpass.getpass("Enter your gemini api key: ").strip()
    save = input("Save api key to keyring? [Y/N] ").lower() == 'y'
    if save:
        keyring.set_password("images_to_translated_pdf", "GEMINI_API_KEY", key)
    console.print("[green].. GEMINI_API_KEY saved to keyring .. [/green]")
    
    return key

## and ability to change current api key


def main():


    ## add in all argument parser stuff:
    parser = argparse.ArgumentParser(description="Assemble images of a physical article or PhD into a .pdf, and translate into English")
    parser.add_argument("--model", default="gemini-2.5-flash", type=str)
    parser.add_argument("--image_directory", required=True)
    parser.add_argument("--pdf_path", required=True)
    parser.add_argument("--language", required=True, type=str)
    parser.add_argument("--batches", default=1, type=int)
    parser.add_argument("--retries_allowed", default=3, type=int)

    run_usage = [] ## track each run


    args = parser.parse_args()

    api_key = get_gemini_key() ## switch to call 
    if not api_key:
        console.print("[red].. failed to provide GEMINI_API_KEY - ending run ..[/red]")
        sys.exit(1)

    ## add in BATCHE_PAGES, conditional based on batch_size - default 1, if above 1 then should be True
    BATCH_PAGES = None
    if args.batches > 1:
        BATCH_PAGES = True
    else:
        BATCH_PAGES = False

    ## replace this wih a proper 'find_image_directory() function
    try:
        image_directory = find_image_directory(args.image_directory) #
    except (FileNotFoundError, NotADirectoryError) as e:
        print(e)
        sys.exit(1) ## exit CLI

    images_to_convert: list[Path] = []

    token_usage = []

    image_directory_files = list(image_directory.iterdir())
    
    console.print("[yellow].. Searching for images in image directory ..[/yellow]")
    for file in tqdm(image_directory_files, desc="..Finding images in image directory.."):
        print(f"file is: {file}")
        if file.is_file():
            if file.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                images_to_convert.append(file)

    if images_to_convert is None:
        console.print("[red].. No images found - ending run ..[/red]")
        sys.exit(1)
    console.print("[green].. Images found for image directory ..[/green]")
    
    ## sort images_to_convert by end of file name
    console.print("[green].. sorting images into correct page order by file name ..[/green]")
    images_to_convert.sort(key=lambda image: int(image.name.split("_")[2].split(".")[0]))


    ## add in split based on batches
    if BATCH_PAGES:
        # batch pages function
        console.print("[green].. User has chosen to batch images ..[/green]")
        translated_text, failed_after_retries = extract_text_and_translate_batch(images_to_convert, token_usage, args.batches, args.model, args.language, args.retries_allowed, run_usage) ## pass batch size
        if failed_after_retries:
            console.print("[red].. translation failed after retries ..[/red]")
            sys.exit(1)
        console.print("[green].. (Batch) Images extracted and translated ..[/green]")
    else:
        ## existing version
        console.print("[green].. User has chosen default batch size of 1 ..[/green]")
        translated_text, failed_after_retries = extract_text_and_translate(images_to_convert, token_usage, args.model, args.language, args.retries_allowed, run_usage)
        if failed_after_retries:
            console.print("[red].. translation failed after retries ..[/red]")
            sys.exit(1)
        console.print("[green].. Images extracted and translated ..[/green]")
        
    from decimal import Decimal
            
    total_input_tokens = sum(row.get("input_tokens" or 0) for row in run_usage)
    total_output_tokens = sum(row.get("output_tokens", 0) for row in run_usage)
    total_thought_tokens = sum(row.get("thought_tokens", 0) for row in run_usage)
    total_tokens = sum(row.get("total_tokens", 0) for row in run_usage)
    console.print(f"[yellow].. Total Input Tokens used: {total_input_tokens} ..[/yellow]")
    console.print(f"[yellow].. Total Output Tokens used: {total_output_tokens} ..[/yellow]")
    console.print(f"[yellow].. Total Thought Tokens used: {total_thought_tokens} ..[/yellow]")
    console.print(f"[yellow].. Total Tokens used: {total_tokens} ..[/yellow]")
    cost = (total_input_tokens * 0.0000003) + (total_output_tokens * 0.0000025) + (total_thought_tokens * 0.0000025) # $0.30 per 1m input, $2.50 per 1m output
    amount = Decimal(str(cost))
    console.print(f"[yellow].. Total Token Cost: ${amount:.2f} ..[/yellow]")

    total_attempts = len(run_usage)
    failed_attempts = sum(1 for row in run_usage if not row["success"])
    successful_attempts = sum(1 for row in run_usage if row["success"])

    retried_pages = sorted({
        row["page"]
        for row in run_usage
        if row["mode"] == "single" and row["attempt"] > 1
    })

    retried_batches = sorted({
        row["batch"]
        for row in run_usage
        if row["mode"] == "batch" and row["attempt"] > 1
    })

    console.print(f"[yellow].. Total Gemini attempts: {total_attempts} ..[/yellow]")
    console.print(f"[yellow].. Successful Gemini attempts: {successful_attempts} ..[/yellow]")
    console.print(f"[yellow].. Failed Gemini attempts: {failed_attempts} ..[/yellow]")
    console.print(f"[yellow].. Retried pages: {retried_pages or 'None'} ..[/yellow]")
    console.print(f"[yellow].. Retried batches: {retried_batches or 'None'} ..[/yellow]")


        # £0.00430 per page, roughly - for english pdfs - what about foreign languages?
        
        ## should be able to fix footnote continuations out of order, based purely on what correct footnumber should be
    for i, page in enumerate(translated_text):

        translated_footnotes = page.translated_footnotes or []
        if translated_footnotes: ##possibly a page might not have footnotes
            print(f"-- page {i} has translated_footnotes! --")
            any_footnote_continuations = any(footnote.continuation_previous_footnote for footnote in translated_footnotes )
            if any_footnote_continuations:
                ## need to create list of all footnotes
                footnote_continuations = [footnote for footnote in translated_footnotes if footnote.continuation_previous_footnote]

                for footnote_c in footnote_continuations:
                    found_match = False
                    target_footnote = footnote_c.continuation_previous_footnote_number
                    if target_footnote is None:
                        continue
                
                            ## go through all other pages
                    for ix, pageN in enumerate(translated_text): 
                        if ix == i:
                            continue ## don't want to loop through current page
                        footnotes = pageN.translated_footnotes or []
                        if footnotes:
                            for prev in footnotes:
                                if prev.number == target_footnote:
                                    prev.footnote_text = f"{prev.footnote_text.rstrip()} {footnote_c.footnote_text.lstrip()}"
                                    page.translated_footnotes.remove(footnote_c)
                                    found_match = True
                                    break

                        if found_match:
                            break
                        
                        

                        



        

    ## make page_number and rest of pydantic into equal-level tuple
    ready_to_assemble = [(chunk.page_number, (chunk.translated_main_text, chunk.translated_footnotes or "")) for chunk in translated_text]
    ## then sort by page number
    print("-- about to fill in any pdf page gaps --")
    ready_to_assemble = infer_page_numbers(ready_to_assemble) ## add in any page number gaps, on presumption that page order is correct
    ready_to_assemble.sort(key=lambda item: item[0])
    print("-- pdf pages ordered! --")

        

    ## then, fix any footnotes that are contiuations of previous ones

    ## order is determined by image file name - presumes images were taken in order from start to end of article/phd

    print(f"ready_to_assemble is: {ready_to_assemble}")

    ## find title
    title = ""
    for chunk in translated_text:
        if chunk.title:
            title += chunk.title
            break
        else:
            continue
    print(f"title of article or PhD is: {title}")

        ## then need to add in pdf creation code
    doc = SimpleDocTemplate(
        args.pdf_path,
        pagesize=A4,
        leftMargin=2.2 * cm,
        rightMargin=2.2 * cm,
        topMargin=2.2 * cm,
        bottomMargin=2.5 * cm,
        title=title,
        )
    
    styles = getSampleStyleSheet()
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=11,
        leading=15,
    )
        
    foot_style = ParagraphStyle(
        "Footnote",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=9,
        leading=12,
    )
    
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontName="Times-Roman",
        fontSize=16,
        leading=20,
        spaceAfter=12,
    )

    story: List[Any] = [Paragraph(title, title_style)]

    def add_paragraphs(story: List[Any], style: ParagraphStyle, text: str) -> None:
        for para in [p.strip() for p in text.split("\n\n")]:
            if not para:
                continue
            story.append(Paragraph(convert_markers_to_sup(para), style))
            story.append(Spacer(1, 6))
    
    def convert_markers_to_sup(text: str) -> str:
        out = []
        i = 0
        while i < len(text):
            if text.startswith("[^", i):
                j = text.find("]", i)
                if j != -1:
                    marker = text[i + 2:j]
                    out.append(f"<super>{marker}</super>")
                    i = j + 1
                    continue
            out.append(text[i])
            i += 1
        return "".join(out)


    def add_footnotes(story: List[Any], style: ParagraphStyle, footnotes: List[Footnote]) -> None:
        if not footnotes:
            return
        items = []
        story.append(Spacer(1, 6))
        for note in footnotes:
            number = note.number if note.number is not None else ""
            para = Paragraph(f"{number}. {note.footnote_text or ""}", style)
            story.append(para)
            story.append(Spacer(1, 4))

    for page_number, translated_chunks in ready_to_assemble:
            main_text, footnotes = translated_chunks
            add_paragraphs(story, body_style, main_text)
            add_footnotes(story, foot_style, footnotes)
            story.append(PageBreak())

    if story and isinstance(story[-1], PageBreak):
        story = story[:-1]

    print("Rendering PDF...")
    doc.build(story)

    ## show cost

    ## with batch size 2 - cost = $0.02

    ## with default batch of 1 - cost = $0.03

    ## gemini 2.5 flash lite batch 2 - $0.01
        

    #with open("test_result.md", "w", encoding="utf8") as file:
        #file.write(str(translated_text))
    #return

        #converted_images = img2pdf.convert([str(p) for p in images_to_convert])
        #print("-- images converted to pdf bytes --")
        # produces bytes in memory
        # 
        # ## ocrmypdf alternative
        #ocrmypdf.ocr(
        #"book.pdf",
        #"searchable.pdf",
        #force_ocr=True,
        #deskew=True,
        #clean=True,
        #rotate_pages=True,
        #)

        ## step 3. - try and obtain text with liteparse (OCR) first
        #parser = LiteParse(
           # ocr_enabled=True,
           # dpi=200,
        #)

        #result = parser.parse(converted_images) ## take straight from bytes
        #print("-- pdf bytes converted to pdf text")

        #text = result.text
        #print(f"text from LiteParse operation on converted_images is: {text}")

        ## then, translate and create pdf with existing workflow - ??

if __name__ == "__main__":
    main()

#main()
