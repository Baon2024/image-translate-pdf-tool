from pathlib import Path
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
from .basic_rate_limiter import rate_limiter
import keyring
import getpass
import asyncio


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

def call_gemini_structured_output(image, response_model, model_name, language, client):

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
    

    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=response_model
            )
        )
    usage = response.usage_metadata

    #print("Input tokens:", usage.prompt_token_count)
    #print("Output tokens:", usage.candidates_token_count)
    #print("Thoughts:", usage.thoughts_token_count)
    #print("Total tokens:", usage.total_token_count)

    ## need to ensure i track token costs - so i get a baseline for now.
    
    response_parsed = response.parsed
    return { "response": response_parsed, "input_tokens": usage.prompt_token_count, "output_tokens": usage.candidates_token_count, "total_tokens": usage.total_token_count, "thought_tokens": usage.thoughts_token_count }  # should return correctly

def call_gemini_structured_output_batch(batch_waitlist, response_model, model_name, language, client):

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

    

    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=response_model
            )
        )
    usage = response.usage_metadata

    #print("Input tokens:", usage.prompt_token_count)
    #print("Output tokens:", usage.candidates_token_count)
    #print("Thoughts:", usage.thoughts_token_count)
    #print("Total tokens:", usage.total_token_count)

    ## need to ensure i track token costs - so i get a baseline for now.
    if response.parsed:
        response_parsed = response.parsed.pages
    else:
        response_parsed = None
    return { "response": response_parsed, "input_tokens": usage.prompt_token_count, "output_tokens": usage.candidates_token_count, "total_tokens": usage.total_token_count, "thought_tokens": usage.thoughts_token_count }  # should return correctly


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


async def core_batch_retry_handler(index, batch_waitlist, token_usage, current_batch_number, model, language, retries_allowed, run_usage, client, free_tier_llm):
    
    translated_text_local = []
    num_of_tries = retries_allowed
    failed_after_retries = False
    for attempt in range(1, num_of_tries + 1):
            if attempt > 1:
                console.print(f"[yellow].. text extraction and translation for batch {current_batch_number} return None - retrying for the {attempt} time.. [/yellow]")
            
            if free_tier_llm:
                await rate_limiter(10)
            extracted_and_translated_text_result = await asyncio.to_thread(call_gemini_structured_output_batch, batch_waitlist, TranslationResultBatch, model, language, client)
            #print(f"extracted and translated text for batch {current_batch_number} is: {extracted_and_translated_text_result['response']}")

            run_usage.append({
                    "mode": "single",          # or "batch"
                    "page": None,
                    "batch": current_batch_number,
                    "attempt": attempt,
                    "success": extracted_and_translated_text_result["response"] is not None,
                    "input_tokens": extracted_and_translated_text_result.get("input_tokens", 0),
                    "output_tokens": extracted_and_translated_text_result.get("output_tokens", 0),
                    "thought_tokens": extracted_and_translated_text_result.get("thought_tokens", 0),
                    "total_tokens": extracted_and_translated_text_result.get("total_tokens", 0),
            })

            #print(f"text extraction and translation for batch {current_batch_number} return None - retrying for the {attempt} time.. ")
            
            if extracted_and_translated_text_result["response"] is not None:
                translated_text_local.extend(extracted_and_translated_text_result["response"])
                #print(f"extracted and translated text for batch {current_batch_number} is: {extracted_and_translated_text_result['response']}")
                return { "page_number": index, "translated_text": translated_text_local, "failed_after_retries": failed_after_retries }    
        
    #print(f"failed to extract and translate batch {current_batch_number} after {num_of_tries} attempts: - ending translation run..
    # ")
    failed_after_retries = True
    return { "page_number": index, "translated_text": translated_text_local, "failed_after_retries": failed_after_retries } 

async def extract_text_and_translate_batch(images_to_convert, token_usage, batch_size, model, language, retries_allowed, run_usage, client, free_tier_llm):
    
    current_batch_number = 0
    current_batch_total = 0 ## count each
    batch_waitlist = [] ## store waiting batches here
    translated_text = []
    failed_after_retries = False

    ## need to switch this to asyncio.gather
    batch_tasks = [] 

    for i, image in enumerate(tqdm(images_to_convert)):
        if current_batch_total == batch_size:
            ## translate existing batch group
            current_batch_number += 1
            ## add to batch_tasks instead
            batch_tasks.append(core_batch_retry_handler(i, batch_waitlist, token_usage, current_batch_number, model, language, retries_allowed, run_usage, client, free_tier_llm))
            batch_waitlist = []
            batch_waitlist.append(image)
            current_batch_total = 1
        else:
            batch_waitlist.append(image)
            current_batch_total += 1
    if current_batch_total > 0:
        ## translate existing batch group
        current_batch_number += 1
        batch_tasks.append(core_batch_retry_handler(i, batch_waitlist, token_usage, current_batch_number, model, language, retries_allowed, run_usage, client, free_tier_llm))
        current_batch_total = 0 ## reset batches in waitlist, for next batch
    

    ## then call the tasks
    #task_results = await asyncio.gather(*batch_tasks)
    task_results = [] #await asyncio.gather(*tasks)
    ## change so can track
    for finished_task in tqdm(asyncio.as_completed(batch_tasks), total=len(batch_tasks), desc="Translating pages"):
        result = await finished_task
        task_results.append(result)

    task_results.sort(key=lambda task: task["page_number"])    


    translated_text = [chunk for text in task_results for chunk in text.get("translated_text", [])]
    retry_failures_exist = any(task.get("failed_after_retries", None) is True for task in task_results)
    partial_translation = any(text is None for text in translated_text)

    if retry_failures_exist or partial_translation:
        failed_after_retries = True
        return translated_text, failed_after_retries
    else:
        failed_after_retries = False
        return translated_text, failed_after_retries

    



async def extract_text_and_translate_concurrent_functionality(i, image, TranslationResult, model, language, client, num_of_tries, free_tier_llm):
        
        

        for attempt in range(1, num_of_tries + 1):
                if attempt > 1:
                    console.print(f"[yellow].. text extraction and translation for page {i} return None - retrying for the {attempt} time.. [/yellow]")
                
                if free_tier_llm:
                    await rate_limiter(10)
                extracted_and_translated_text_result = await asyncio.to_thread(call_gemini_structured_output, image, TranslationResult, model, language, client)
                #print(f"extracted and translated text for image {i} is: {extracted_and_translated_text_result['response']}")
                
                run_usage = {
                    "mode": "single",          # or "batch"
                    "page": i,
                    "batch": None,
                    "attempt": attempt,
                    "success": extracted_and_translated_text_result["response"] is not None,
                    "input_tokens": extracted_and_translated_text_result.get("input_tokens", 0),
                    "output_tokens": extracted_and_translated_text_result.get("output_tokens", 0),
                    "thought_tokens": extracted_and_translated_text_result.get("thought_tokens", 0),
                    "total_tokens": extracted_and_translated_text_result.get("total_tokens", 0),
                }

                if extracted_and_translated_text_result["response"] is not None:
                    failed_after_retries = False
                    return { "page_number": i, "translated_text": extracted_and_translated_text_result["response"], "failed_after_retries": failed_after_retries, "run_usage": run_usage } 
                    #print(f"extracted and translated text for image {i} is: {extracted_and_translated_text_result['response']}")
                else:
                    if attempt < 3:
                        console.print("[red].. GEMINI LLM call did not return a response - will re try ..[/red]")
        else:
            console.print("[red].. GEMINI LLM call did not return a response - ran out of retries[/red]")
            failed_after_retries = True
            run_usage = None
            return {"page_number": i, "translated_text": None, "failed_after_retries": failed_after_retries, "run_usage": run_usage }


async def extract_text_and_translate(images_to_convert, token_usage, model, language, retries_allowed, run_usage, client, free_tier_llm):
    
    

    ## need to switch to concurrent

    tasks = []

    for i, image in enumerate(tqdm(images_to_convert)):
        tasks.append(extract_text_and_translate_concurrent_functionality(i, image, TranslationResult, model, language, client, retries_allowed, free_tier_llm))
    
    task_results = [] #await asyncio.gather(*tasks)
    ## change so can track
    for finished_task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Translating pages"):
        result = await finished_task
        task_results.append(result)
    
    ## asyncio.as_completed returns task in order of completion, not order of pages, so needs to be sorted afterwards
    task_results.sort(key=lambda task: task["page_number"])

    ## then extract each
    translated_text = [text.get("translated_text", []) for text in task_results]
    partial_translation = any(task.get("translated_text", None) is None for task in task_results)
    failed = any((task.get('failed_after_retries')) is True for task in task_results)

    if failed or partial_translation:
        failed_after_retries = True
        return translated_text, failed_after_retries
    else:
        failed_after_retries = False
        return translated_text, failed_after_retries

                    








def get_gemini_client(api_key):
    return genai.Client(api_key=api_key)

def validate_gemini(client):
    try:
        models = client.models.list()
        return True
    except Exception as e:
        return False


from rich.prompt import Confirm, Prompt

def validation_gemini_api_key_loop():
    console.print("[yellow].. Checking Gemini key is valid ..[/yellow]")


    api_key = keyring.get_password('images_to_translated_pdf', "GEMINI_API_KEY")


    while True:
        if api_key:
            console.print("[green].. GEMINI_API_KEY already saved to keyring - will use this .. [/green]")
        else:
            console.print("[yellow].. no existing GEMINI_API_KEY saved to keyring - you need to provide one .. [/yellow]")
            api_key = getpass.getpass("Enter your gemini api key: ").strip()
        client = get_gemini_client(api_key)
        validation_check = validate_gemini(client)
        if validation_check:
            ## update saved keychain and return authenticated client
            keyring.set_password("images_to_translated_pdf", "GEMINI_API_KEY", api_key)
            return client
        else:
            ## ask user for new key or to exist
            answer = Prompt.ask("The api_key you entered isn't a valid Gemini key. Please choose your option: ",
                     choices=["Retry", "Abort"])
            if answer == "Abort":
                console.print("[yellow].. Ending current run ..[/yellow]")
                sys.exit(1)
            else:
                api_key = Prompt.ask("Please enter a Gemini api key").strip()


def change_api_key():

    ## get current key, if it exists
    console.print("[yellow].. Checking if you have a current api key ..[/yellow]")
    curr_api_key = keyring.get_password('images_to_translated_pdf', "GEMINI_API_KEY")
    if curr_api_key:
        console.print(f"[yellow].. your current Gemini api key is: {curr_api_key} ..[/yellow]")
    
    ## get new api key
    api_key = Prompt.ask("Please enter a Gemini api key").strip()
    
    while True:
        client = get_gemini_client(api_key)
        validation_check = validate_gemini(client)
        if validation_check:
            ## update saved keychain and return authenticated client
            keyring.set_password("images_to_translated_pdf", "GEMINI_API_KEY", api_key)
            console.print("[yellow].. New Api Key saved! ..[/yellow]")
            return
        else:
            answer = Prompt.ask("The api_key you entered isn't a valid Gemini key. Please choose your option: ",
                     choices=["Retry", "Abort"])
            if answer == "Abort":
                console.print("[yellow].. Ending current run ..[/yellow]")
                sys.exit(1)
            else:
                api_key = Prompt.ask("Please enter a Gemini api key").strip()


async def async_main():


    ## add in all argument parser stuff:
    parser = argparse.ArgumentParser(description="Assemble images of a physical article or PhD into a .pdf, and translate into English")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--image_directory")
    parser.add_argument("--pdf_path")
    parser.add_argument("--language", type=str)
    parser.add_argument("--batches", default=1, type=int)
    parser.add_argument("--retries_allowed", default=3, type=int)
    parser.add_argument("--change_api_key", action="store_true")
    parser.add_argument("--free_tier_llm", default=True, type=bool)

    run_usage = []
    footnotes = []

    args = parser.parse_args()

    

    if args.change_api_key:
        change_api_key() ## do code for this, then return
        sys.exit(1)
    


    ## otherwise, check all fields that need to be required were provided
    required_args_missing = [name for name in ("image_directory", "language", "pdf_path") if getattr(args, name) is None]
    if required_args_missing:
        sys.exit(1)
        console.print(f"[red].. required flags for CLI were missing: {required_args_missing} ..[/red]")

    ## need to pass client down to LLM calls
    client = validation_gemini_api_key_loop()

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
        #print(f"file is: {file}")
        if file.is_file():
            if file.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                console.print(f"[yellow].. Valid file found: {file} ..[/yellow]")
                images_to_convert.append(file)

    if images_to_convert is None:
        console.print("[red].. No images found - ending run ..[/red]")
        sys.exit(1)
    console.print("[green].. Images found for image directory ..[/green]")
    
    ## sort images_to_convert by end of file name
    console.print("[green].. sorting images into correct page order by file name ..[/green]")
    images_to_convert.sort(key=lambda image: int(image.name.split("_")[2].split(".")[0]))
    
    ## need to pass down something that gets

    ## add in split based on batches
    if BATCH_PAGES:
        # batch pages function
        console.print("[green].. User has chosen to batch images ..[/green]")
        translated_text, failed_after_retries = await extract_text_and_translate_batch(images_to_convert, token_usage, args.batches, args.model, args.language, args.retries_allowed, run_usage, client, args.free_tier_llm) ## pass batch size
        if failed_after_retries:
            console.print("[red].. translation failed after retries ..[/red]")
            sys.exit(1)
        console.print("[green].. (Batch) Images extracted and translated ..[/green]")
    else:
        ## existing version
        console.print("[green].. User has chosen default batch size of 1 ..[/green]")
        translated_text, failed_after_retries = await extract_text_and_translate(images_to_convert, token_usage, args.model, args.language, args.retries_allowed, run_usage, client, args.free_tier_llm)
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
    console.print("[green].. Fixing any PDF footnotes that span pages ..[/green]")
    for i, page in enumerate(translated_text):

        translated_footnotes = page.translated_footnotes or []
        if translated_footnotes: ##possibly a page might not have footnotes
            #print(f"-- page {i} has translated_footnotes! --")
            any_footnote_continuations = any(footnote.continuation_previous_footnote for footnote in translated_footnotes )
            if any_footnote_continuations:
                ## need to create list of all footnotes
                footnote_continuations = [footnote for footnote in translated_footnotes if footnote.continuation_previous_footnote == True ]

                for footnote_c in footnote_continuations:
                    footnotes.append({
                        "footnote_contination": True,
                        "footnote_continuation_number": footnote_c.continuation_previous_footnote_number
                        })


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
                        
                        
    footnote_continuation_total = sum(1 for footnote in footnotes)
    console.print(f"[yellow].. No. of footnotes spanning multiple pages: {footnote_continuation_total} ..[/yellow]")



        

    ## make page_number and rest of pydantic into equal-level tuple
    ready_to_assemble = [(chunk.page_number, (chunk.translated_main_text, chunk.translated_footnotes or "")) for chunk in translated_text]
    ## then sort by page number
    ready_to_assemble = infer_page_numbers(ready_to_assemble) ## add in any page number gaps, on presumption that page order is correct
    console.print("[green].. Any PDF page gaps filled-in ..[/green]")
    ready_to_assemble.sort(key=lambda item: item[0])
    console.print("[green].. PDF ordered by pages ..[/green]")

        

    ## then, fix any footnotes that are contiuations of previous ones

    ## order is determined by image file name - presumes images were taken in order from start to end of article/phd

    #print(f"ready_to_assemble is: {ready_to_assemble}")

    ## find title
    title = ""
    for chunk in translated_text:
        if chunk.title:
            title += chunk.title
            break
        else:
            continue

    console.print(f"[green].. title of article or PhD is: {title} ..[/green]")

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

    console.print("[green]... Rendering PDF... [/green]")
    doc.build(story)
    console.print("[green]... PDF Ready... [/green]")
    console.print(f"[green]... New PDF saved to: {args.pdf_path}... [/green]")


def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()




