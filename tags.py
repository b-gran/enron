import asyncio
import re
import sys
import time
import click
import joblib
import pandas as pd
import tiktoken
import multiprocessing as mp
from typing import List, Any

from pydantic import BaseModel
from openai import AsyncOpenAI

import os
from dotenv import load_dotenv
import tqdm

env_path = '/Users/bill/experiments/enron-analysis/secrets.env'

load_dotenv(dotenv_path=env_path)
client = AsyncOpenAI(api_key=os.getenv("BILL_OPENAI_API_KEY"))


GPT_ENCODING = tiktoken.encoding_for_model('gpt-4o-mini')


system_prompt = '''
Tag the following email based on the tags below. The email is part of the Enron email dataset. It was sent in a corporate context.

Respond with a json array of tags. E.g. ["PERSONAL", "FUNNY"]
If the email doesn't fit any of the tags, respond with an empty array. E.g. []

TAGS:
* PERSONAL: contains significant non-work related text. E.g.: emails discussing personal plans at work event. Entirely personal emails should also be tagged PERSONAL.
* LOGISTICS: primarily discusses scheduling, meeting times, or other logistical matters.
* FUNNY: contains jokes, memes, or other humorous content. Also include emails that are not intended to be humorous, but are funny in hindsight knowning that the email is part of the Enron email dataset.
* FRAUD: contains evidence of fraud or other crimes. 
* GUILT: someone in the thread admits to significant wrongdoing.
* 911: discussion of the 9/11 attacks.
* MENTORSHIP: someone in the email thread is giving specific and actionable personal advice to another person.
* LEGAL_ADVICE: someone in the email thread is giving or receiving legal advice.
* ROMANCE: participants in the email thread are in a romantic relationship
'''

email_content = f"""EMAIL CONTENT:
{{content}}"""

ATTACHMENT_CONTENT_RE = re.compile(r'[a-zA-Z0-9+/=]{30,}')
JUNK_LINES = [
    'This e-mail, including any attachments, is intended for the',
    'receipt and use by the intended addressee(s), and may contain',
    'confidential and privileged information.  If you are not an intended',
    'recipient of this e-mail, you are hereby notified that any unauthorized',
    'use or distribution of this e-mail is strictly prohibited.',
]
seps = r'[\s\t\r\n\-]*'

def escape_chars(s):
    chars = ['(', ')', '.']
    for c in chars:
        s = s.replace(c, f'[{c}]')
    return s

text_lines = escape_chars(seps.join([seps.join(list(l)) for l in ' '.join(JUNK_LINES[:5]).split(' ')]))

JUNK_LINES_RE = re.compile(r'\**' + seps + text_lines, re.MULTILINE)

def remove_attachment_content(content):
    cnt = ATTACHMENT_CONTENT_RE.sub('', content)
    return JUNK_LINES_RE.sub('', cnt)

    
class ParseResult(BaseModel):
    tags: list[str]


async def parse_openai(system_prompt: str, user_prompt: str):
    try:
        completion = await client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=ParseResult,
        )
        return {
            'result': completion.choices[0].message.parsed,
            'error': None
        }
    except Exception as e:
        return {
            'result': None,
            'error': e
        }

        
async def process_batch(batch, system_prompt):
    tasks = []
    for prompt in batch['user_prompt']:
        task = asyncio.create_task(parse_openai(system_prompt, prompt))
        tasks.append(task)
    return await asyncio.gather(*tasks)


async def run_batches(
    df: pd.DataFrame,
    system_prompt: str,
    batch_size: int = 5000,
    concurrency: int = 100,
    final_file_template: str = 'final-{id}.joblib',
    checkpoint_file_template: str = 'checkpoint-{id}.joblib',
    tmp_file_template = 'tmp-{id}.joblib',
) -> List[Any]:
    run_id = int(time.time())
    tmp_file = tmp_file_template.format(id=run_id)
    final_file = final_file_template.format(id=run_id)
    checkpoint_file = checkpoint_file_template.format(id=run_id)
    print('Starting run', run_id)

    all_results = []
    total_tasks = len(df)
    
    for start in tqdm.tqdm(range(0, total_tasks, batch_size)):
        end = min(start + batch_size, total_tasks)
        batch = df[start:end]

        semaphore = asyncio.Semaphore(concurrency)

        async def process_batch_with_semaphore(batch: pd.DataFrame):
            async with semaphore:
                return await process_batch(batch, system_prompt)
        
        batch_results = await process_batch_with_semaphore(batch)
        all_results.extend(batch_results)

        num_errors = len([r for r in batch_results if r['error']])
        print(f'Batch {start}-{end} completed. Errors: {num_errors}')
        if num_errors:
            sample_error = [r for r in batch_results if r['error']][0]
            print('Sample error:')
            print(sample_error['error'])
        
        with open(tmp_file, 'wb') as f:
            joblib.dump(all_results, f)

        os.rename(tmp_file, checkpoint_file)

    print('Writing final results to', final_file)
    final_df = df.copy()
    final_df['result'] = all_results
    with open(final_file, 'wb') as f:
        joblib.dump(final_df, f)

    print('Run', run_id, 'completed.')
    
    return all_results



@click.command()
@click.option('-i', '--input', help='Path to input joblib containing the enron dataframe', required=True, type=click.Path(exists=True, dir_okay=False))
@click.option('--limit', type=int, help='Limit the number of rows to process for testing')
def cli(input: str, limit: int | None):
    from pandarallel import pandarallel
    pandarallel.initialize(progress_bar=True)

    try:
        df: pd.DataFrame = joblib.load(input)
        assert isinstance(df, pd.DataFrame), 'input is not a dataframe'
    except Exception as e:
        print(f"Error loading file {input}")
        print(e)
        return sys.exit(1)

    if limit:
        print('Limiting rows to', limit)
        df = df.head(limit)

    df['content_no_attachment'] = df['content'].parallel_apply(remove_attachment_content)
    df['user_prompt'] = df['content_no_attachment'].apply(lambda x: email_content.format(content=x))

    print()
    print()
    print('SYSTEM PROMPT:')
    print(system_prompt)

    print()
    print()
    print('SAMPLE USER PROMPT:')
    print(df['user_prompt'].sample(1).values[0])

    system_prompt_len = len(GPT_ENCODING.encode(system_prompt))
    df['token_count_total'] = df['content_no_attachment'].parallel_apply(lambda x: len(GPT_ENCODING.encode(x))) + system_prompt_len
    print()
    print()
    print(f'About to process {len(df)} emails with a total token count of {df["token_count_total"].sum()}')
    print(f'Estimated cost for input tokens: {df["token_count_total"].sum() * 0.15 / 1e6} USD')

    print()
    print()
    should_run = click.confirm('Do you want to continue?')
    if not should_run:
        print('Exiting.')
        return sys.exit(1)

    asyncio.run(run_batches(df, system_prompt))


if __name__ == '__main__':
    cli()