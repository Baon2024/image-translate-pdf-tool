from pathlib import Path
from google import genai
from google.genai import types
from ollama import chat
from pydantic import Field, BaseModel
from typing import List, Any
from dotenv import load_dotenv
from rich.console import Console
from tqdm import tqdm
from .basic_rate_limiter import rate_limiter
import asyncio
import fitz

load_dotenv()

console = Console()

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


def token_count(value):
    return value if value is not None else 0


def usage_token_count(usage, attribute):
    if usage is None:
        return 0
    return token_count(getattr(usage, attribute, 0))


def ensure_pydantic_schema(content, pydantic_model, data_type):

    def normalize(item):
        page = pydantic_model.model_validate(item)

        if hasattr(page, "translated_footnotes"):
            page.translated_footnotes = [
                note if isinstance(note, Footnote) else Footnote.model_validate(note)
                for note in (page.translated_footnotes or [])
            ]
            print("normalized footnote types:", [
                type(note) for note in (page.translated_footnotes or [])
            ])


        return page


    if data_type == "single":
        return normalize(content)
    elif data_type == "array":
        return [normalize(page) for page in content]
    else:
          raise ValueError(f"Unsupported data_type: {data_type}")




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

    mime_types_dict = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }

    contents = []

    if isinstance(image, dict):
        print("mode used for mime_type and data passing is pdf")
        mime_type = image["suffix"].lower()
        data = image["bytes"]
        page_num = image["page_num"]

    else:
        print("mode used for mime_type and data passing is images")
        mime_type = image.suffix.lower()
        data = image.read_bytes()

    contents.append(types.Part.from_bytes(data=data, mime_type=mime_types_dict[mime_type]))

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
    print(f"response.parsed before checking it matches pydantic schema: {response.parsed}")
    if isinstance(image, dict):
        print(f"for page number: {page_num}")
    
    response_parsed = ensure_pydantic_schema(response.parsed, TranslationResult, "single")
    return { "response": response_parsed, "input_tokens": usage_token_count(usage, "prompt_token_count"), "output_tokens": usage_token_count(usage, "candidates_token_count"), "total_tokens": usage_token_count(usage, "total_token_count"), "thought_tokens": usage_token_count(usage, "thoughts_token_count") }  # should return correctly

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

    ## need to check if it's pdf or image version
    for image in batch_waitlist:
        if isinstance(image, dict):
            print("mode used for mime_type and data passing is pdf")
            ## then it's images form a pdf
            mime_type = image["suffix"].lower()
            data = image["bytes"]
            image_waitlist.append(types.Part.from_bytes(data=data, mime_type=mime_types_dict[mime_type]))
        else:
            print("mode used for mime_type and data passing is images")
            mime_type = image.suffix.lower()
            data = image.read_bytes()
            image_waitlist.append(types.Part.from_bytes(data=data, mime_type=mime_types_dict[mime_type]))
    

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
        response_parsed = ensure_pydantic_schema(response.parsed.pages, TranslationResult, "array")
    else:
        response_parsed = None
    return { "response": response_parsed, "input_tokens": usage_token_count(usage, "prompt_token_count"), "output_tokens": usage_token_count(usage, "candidates_token_count"), "total_tokens": usage_token_count(usage, "total_token_count"), "thought_tokens": usage_token_count(usage, "thoughts_token_count") }  # should return correctly



def failed_gemini_result():
    return {
        "response": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "thought_tokens": 0,
        "total_tokens": 0,
    }

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

def find_pdf_file(pdf_path):
    file = Path(pdf_path)
    console.print("[yellow].. Searching for image directory ..[/yellow]")

    file_pdf_path = file.resolve()
    if not file_pdf_path.exists():
        console.print("[red].. input_pdf_path not found ..[/red]")
        raise FileNotFoundError(f"pdf_path_file extract and translate not found: at {pdf_path}")

    console.print("[green].. input_pdf_path found ..[/green]")
    return file_pdf_path

def get_images_from_pdf(pdf_path):

    #pdf_reader = PDFReader() ## or whatever the code is
    doc = fitz.open(pdf_path)

    ## get each page
    converted_pages = []

    ## will need to add something

    for pagenumber, page in enumerate(doc):
        pix = page.get_pixmap(dpi=150)
        #img = Image.frombytes("RGB", [pix.wdith, pix.height], pix.samples)
        #img.format = "JPEG"

        #buffer = io.BytesIO()
        #img.save(buffer, format="JPEG")
        #new_imag = Image.open(io.BytesIO(buffer))
        converted_pages.append({ "bytes": pix.tobytes("jpeg"), "filename": f"_page_{pagenumber}.jpeg", "suffix": ".jpeg", "page_num": pagenumber + 1 })
        ## stays as bytes - will need to change existing version to have something very similar

        ## for testing
        #re_img = Image.open(buffer)
        #print(re_img.format)
    return converted_pages



async def core_batch_retry_handler(index, batch_waitlist, token_usage, current_batch_number, model, language, retries_allowed, run_usage, client, free_tier_llm, semaphore):
    
    translated_text_local = []
    num_of_tries = retries_allowed
    failed_after_retries = False
    for attempt in range(1, num_of_tries + 1):
            if attempt > 1:
                console.print(f"[yellow].. text extraction and translation for batch {current_batch_number} return None - retrying for the {attempt} time.. [/yellow]")
            
            try:
                async with semaphore:
                    if free_tier_llm:
                        await rate_limiter(10)
                    extracted_and_translated_text_result = await asyncio.to_thread(call_gemini_structured_output_batch, batch_waitlist, TranslationResultBatch, model, language, client)
            except Exception as e:
                console.print(f"[red].. Gemini call failed for batch {current_batch_number} on attempt {attempt}: {e} ..[/red]")
                extracted_and_translated_text_result = failed_gemini_result()
            #print(f"extracted and translated text for batch {current_batch_number} is: {extracted_and_translated_text_result['response']}")

            run_usage.append({
                    "mode": "single",          # or "batch"
                    "page": None,
                    "batch": current_batch_number,
                    "attempt": attempt,
                    "success": extracted_and_translated_text_result["response"] is not None,
                    "input_tokens": token_count(extracted_and_translated_text_result.get("input_tokens", 0)),
                    "output_tokens": token_count(extracted_and_translated_text_result.get("output_tokens", 0)),
                    "thought_tokens": token_count(extracted_and_translated_text_result.get("thought_tokens", 0)),
                    "total_tokens": token_count(extracted_and_translated_text_result.get("total_tokens", 0)),
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
    semaphore = asyncio.Semaphore(5)

    for i, image in enumerate(tqdm(images_to_convert)):
        if current_batch_total == batch_size:
            ## translate existing batch group
            current_batch_number += 1
            ## add to batch_tasks instead
            batch_tasks.append(core_batch_retry_handler(i, batch_waitlist, token_usage, current_batch_number, model, language, retries_allowed, run_usage, client, free_tier_llm, semaphore))
            batch_waitlist = []
            batch_waitlist.append(image)
            current_batch_total = 1
        else:
            batch_waitlist.append(image)
            current_batch_total += 1
    if current_batch_total > 0:
        ## translate existing batch group
        current_batch_number += 1
        batch_tasks.append(core_batch_retry_handler(i, batch_waitlist, token_usage, current_batch_number, model, language, retries_allowed, run_usage, client, free_tier_llm, semaphore))
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

    




async def extract_text_and_translate_batch_sequential(images_to_convert, token_usage, batch_size, model, language, retries_allowed, run_usage, client, free_tier_llm):
    current_batch_number = 0
    current_batch_total = 0
    batch_waitlist = []
    task_results = []
    semaphore = asyncio.Semaphore(1)

    for i, image in enumerate(tqdm(images_to_convert)):
        if current_batch_total == batch_size:
            current_batch_number += 1
            result = await core_batch_retry_handler(i, batch_waitlist, token_usage, current_batch_number, model, language, retries_allowed, run_usage, client, free_tier_llm, semaphore)
            task_results.append(result)
            batch_waitlist = [image]
            current_batch_total = 1
        else:
            batch_waitlist.append(image)
            current_batch_total += 1

    if current_batch_total > 0:
        current_batch_number += 1
        result = await core_batch_retry_handler(i, batch_waitlist, token_usage, current_batch_number, model, language, retries_allowed, run_usage, client, free_tier_llm, semaphore)
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

async def extract_text_and_translate_concurrent_functionality(i, image, TranslationResult, model, language, client, num_of_tries, free_tier_llm, semaphore):
        
        

        for attempt in range(1, num_of_tries + 1):
                if attempt > 1:
                    console.print(f"[yellow].. text extraction and translation for page {i} return None - retrying for the {attempt} time.. [/yellow]")
                
                try:
                    async with semaphore:
                        if free_tier_llm:
                            await rate_limiter(10)
                        extracted_and_translated_text_result = await asyncio.to_thread(call_gemini_structured_output, image, TranslationResult, model, language, client)
                except Exception as e:
                    console.print(f"[red].. Gemini call failed for page {i} on attempt {attempt}: {e} ..[/red]")
                    extracted_and_translated_text_result = failed_gemini_result()
                #print(f"extracted and translated text for image {i} is: {extracted_and_translated_text_result['response']}")
                
                run_usage = {
                    "mode": "single",          # or "batch"
                    "page": i,
                    "batch": None,
                    "attempt": attempt,
                    "success": extracted_and_translated_text_result["response"] is not None,
                    "input_tokens": token_count(extracted_and_translated_text_result.get("input_tokens", 0)),
                    "output_tokens": token_count(extracted_and_translated_text_result.get("output_tokens", 0)),
                    "thought_tokens": token_count(extracted_and_translated_text_result.get("thought_tokens", 0)),
                    "total_tokens": token_count(extracted_and_translated_text_result.get("total_tokens", 0)),
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
    semaphore = asyncio.Semaphore(5)

    for i, image in enumerate(tqdm(images_to_convert)):
        tasks.append(extract_text_and_translate_concurrent_functionality(i, image, TranslationResult, model, language, client, retries_allowed, free_tier_llm, semaphore))
    
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

                    









async def extract_text_and_translate_sequential(images_to_convert, token_usage, model, language, retries_allowed, run_usage, client, free_tier_llm):
    semaphore = asyncio.Semaphore(1)
    task_results = []

    for i, image in enumerate(tqdm(images_to_convert, desc="Translating pages")):
        result = await extract_text_and_translate_concurrent_functionality(i, image, TranslationResult, model, language, client, retries_allowed, free_tier_llm, semaphore)
        task_results.append(result)

    task_results.sort(key=lambda task: task["page_number"])

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

