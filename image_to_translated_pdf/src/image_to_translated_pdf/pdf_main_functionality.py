from .shared_helper_functions import (extract_text_and_translate_batch, 
 extract_text_and_translate, 
 extract_text_and_translate_batch_sequential, 
 extract_text_and_translate_concurrent_functionality, 
 extract_text_and_translate_sequential, token_count, Footnote, TranslationResultBatch, TranslationResult )
import sys
from pydantic import Field, BaseModel
from typing import List, Any
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from dotenv import load_dotenv
from rich.console import Console
from reportlab.platypus import ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate, Spacer


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



async def rest_of_functionality(pdf_images, args, BATCH_PAGES, client):
    token_usage = []
    run_usage = []
    footnotes = []

    if BATCH_PAGES:
        # batch pages function
        console.print("[green].. User has chosen to batch images ..[/green]")
        if args.concurrent:
            console.print("[green].. Running batch translation concurrently ..[/green]")
            translated_text, failed_after_retries = await extract_text_and_translate_batch(pdf_images, token_usage, args.batches, args.model, args.language, args.retries_allowed, run_usage, client, args.free_tier_llm) ## pass batch size
        else:
            console.print("[green].. Running batch translation sequentially ..[/green]")
            translated_text, failed_after_retries = await extract_text_and_translate_batch_sequential(pdf_images, token_usage, args.batches, args.model, args.language, args.retries_allowed, run_usage, client, args.free_tier_llm) ## pass batch size
        if failed_after_retries:
            console.print("[red].. translation failed after retries ..[/red]")
            sys.exit(1)
        console.print("[green].. (Batch) Images extracted and translated ..[/green]")
    else:
        ## existing version
        console.print("[green].. User has chosen default batch size of 1 ..[/green]")
        if args.concurrent:
            console.print("[green].. Running page translation concurrently ..[/green]")
            translated_text, failed_after_retries = await extract_text_and_translate(pdf_images, token_usage, args.model, args.language, args.retries_allowed, run_usage, client, args.free_tier_llm)
        else:
            console.print("[green].. Running page translation sequentially ..[/green]")
            translated_text, failed_after_retries = await extract_text_and_translate_sequential(pdf_images, token_usage, args.model, args.language, args.retries_allowed, run_usage, client, args.free_tier_llm)
        if failed_after_retries:
            console.print("[red].. translation failed after retries ..[/red]")
            sys.exit(1)
        console.print("[green].. Images extracted and translated ..[/green]")
        

            
    from decimal import Decimal
            
    total_input_tokens = sum(token_count(row.get("input_tokens", 0)) for row in run_usage)
    total_output_tokens = sum(token_count(row.get("output_tokens", 0)) for row in run_usage)
    total_thought_tokens = sum(token_count(row.get("thought_tokens", 0)) for row in run_usage)
    total_tokens = sum(token_count(row.get("total_tokens", 0)) for row in run_usage)
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
    print(type(translated_text[0]))
    #print(type(translated_text[0].translated_footnotes[0]))
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
                        footnotes_local = pageN.translated_footnotes or []
                        if footnotes_local:
                            for prev in footnotes_local:
                                print(f"current prev footnote: {prev}")
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
    return