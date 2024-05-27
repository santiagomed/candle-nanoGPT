#!/usr/bin/env python
import click
from openai import OpenAI
import re


@click.command()
@click.argument("transcript_file", type=click.Path(exists=True))
@click.argument("code_file", type=click.Path(exists=True))
@click.argument("lang")
@click.argument("output_file", type=click.Path())
def generate_docs(transcript_file, code_file, lang, output_file):
    # Read transcript and code from files
    with open(transcript_file, "r") as tf:
        transcript_text = tf.read()

    with open(code_file, "r") as cf:
        code_text = cf.read()

    # Remove timestamps from transcript text
    cleaned_transcript = re.sub(r"\b\d{1,2}:\d{2}\b", "", transcript_text)

    # System prompt
    system_prompt = """You are a knowledgeable and helpful assistant. Your task is to create a tutorial-style documentation based on a provided YouTube transcript and a code snippet in {lang}. Carefully review both the transcript and the code to gain context and understand the concepts being discussed. Write the documentation in a clear, instructional manner, as if you are explaining the concept to someone learning it for the first time. Ensure the documentation is comprehensive, containing code snippets and covering all necessary details to help the reader understand and apply the code effectively. If needed, generate new code snippets to support the documentation.""".format(
        lang=lang
    )

    # User message prompt with the cleaned transcript and code text
    user_message = f"""Transcript:\n{cleaned_transcript}\n\nCode:\n{code_text}"""

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )

    print(response)

    documentation = response.choices[0].message.content

    # Write documentation to Markdown file
    with open(output_file, "w") as of:
        of.write(documentation)


if __name__ == "__main__":
    generate_docs()
