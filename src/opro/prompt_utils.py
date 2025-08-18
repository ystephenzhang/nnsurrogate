# Copyright 2023 The OPRO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The utility functions for prompting GPT and Google Cloud models."""

import time
from openai import OpenAI
import pdb

def call_local_server_func(
  inputs, model_path, max_tokens=50, temperature=0.8
):
  pass

def call_openai_server_single_prompt(
    messages, model="gpt-3.5-turbo", max_decode_steps=20, temperature=0.8
):
  """The function to call OpenAI server with an input string."""
  if model.startswith("deepseek"):
    client = OpenAI(base_url="https://api.deepseek.com")
  else:
    client = OpenAI()
  try:
      if model in ["o4-mini"]:
        '''completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )'''
        completion = client.chat.completions.create(
            model=model,
            messages=messages
        )
      else:
        '''completion = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_decode_steps,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )'''
        completion = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_decode_steps,
            messages=messages
        )
      return completion.choices[0].message.content

  except Exception as e:
    if "timeout" in str(e).lower():
      retry_time = 30
      print(f"Timeout error occurred. Retrying in {retry_time} seconds...")
      time.sleep(retry_time)
      return call_openai_server_single_prompt(
          prompt, model=model, max_decode_steps=max_decode_steps, temperature=temperature
      )
    elif "rate_limit_exceeded" in str(e).lower() or "rate limit" in str(e).lower():
      retry_time = 30
      print(f"Rate limit exceeded. Retrying in {retry_time} seconds...")
      time.sleep(retry_time)
      return call_openai_server_single_prompt(
          prompt, model=model, max_decode_steps=max_decode_steps, temperature=temperature
      )
    elif "api" in str(e).lower() and "error" in str(e).lower():
      retry_time = 30
      print(f"API error occurred: {e}. Retrying in {retry_time} seconds...")
      time.sleep(retry_time)
      return call_openai_server_single_prompt(
          prompt, model=model, max_decode_steps=max_decode_steps, temperature=temperature
      )
    elif "connection" in str(e).lower():
      retry_time = 30
      print(f"Connection error occurred: {e}. Retrying in {retry_time} seconds...")
      time.sleep(retry_time)
      return call_openai_server_single_prompt(
          prompt, model=model, max_decode_steps=max_decode_steps, temperature=temperature
      )
    elif "service unavailable" in str(e).lower():
      retry_time = 30
      print(f"Service unavailable: {e}. Retrying in {retry_time} seconds...")
      time.sleep(retry_time)
      return call_openai_server_single_prompt(
          prompt, model=model, max_decode_steps=max_decode_steps, temperature=temperature
      )
    else:
      raise e

  except OSError as e:
    retry_time = 5  # Adjust the retry time as needed
    print(
        f"Connection error occurred: {e}. Retrying in {retry_time} seconds..."
    )
    time.sleep(retry_time)
    return call_openai_server_single_prompt(
        prompt, max_decode_steps=max_decode_steps, temperature=temperature
    )


def call_openai_server_func(
    inputs, model="gpt-3.5-turbo", max_decode_steps=20, temperature=0.8
):
  """The function to call OpenAI server with a list of input strings."""
  if isinstance(inputs, str):
    inputs = [inputs]
  outputs = []
  for input_str in inputs:
    output = call_openai_server_single_prompt(
        input_str,
        model=model,
        max_decode_steps=max_decode_steps,
        temperature=temperature,
    )
    outputs.append(output)
  return outputs


def call_palm_server_from_cloud(
    input_text, model="text-bison-001", max_decode_steps=20, temperature=0.8
):
  """Calling the text-bison model from Cloud API."""
  assert isinstance(input_text, str)
  assert model == "text-bison-001"
  all_model_names = [
      m
      for m in palm.list_models()
      if "generateText" in m.supported_generation_methods
  ]
  model_name = all_model_names[0].name
  try:
    completion = palm.generate_text(
        model=model_name,
        prompt=input_text,
        temperature=temperature,
        max_output_tokens=max_decode_steps,
    )
    output_text = completion.result
    return [output_text]
  except:  # pylint: disable=bare-except
    retry_time = 10  # Adjust the retry time as needed
    print(f"Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return call_palm_server_from_cloud(
        input_text, max_decode_steps=max_decode_steps, temperature=temperature
    )
