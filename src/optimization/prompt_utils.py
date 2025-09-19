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
import anthropic
import time
from openai import OpenAI
import pdb
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

def call_local_server_function_parallel(
    messages, model, tokenizer, max_decode_steps=50, temperature=0.8
):
    """The function to call local model with a single message list.
    
    Args:
        messages: List of message dictionaries
        model: Either a model instance with invoke() method or a path string
        max_decode_steps: Maximum tokens to generate (not used for local models)
        temperature: Temperature for generation (not used for local models)
    """
    texts = [tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # Switches between thinking and non-thinking modes. Default is True.
    ) for message in messages]

    
    model_inputs = tokenizer(texts, return_tensors="pt").to(model.device)

    # Conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[:, len(model_inputs.input_ids[0]):].tolist() 
    
    # Parsing thinking content
    responses = []
    for output_id in output_ids:
      try:
          # rindex finding 151668 (</think>)
          index = len(output_id) - output_id[::-1].index(151668)
      except ValueError:
          index = 0
    
      # thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
      response = tokenizer.decode(output_id[index:], skip_special_tokens=True).strip("\n")
      responses.append(response)

    # print("thinking content:", thinking_content)
    # print("Qwen3 Response:", response)
    
    return responses

def call_openai_server_single_prompt(
    messages, model="gpt-3.5-turbo", max_decode_steps=20, temperature=0.8
):
  """The function to call OpenAI server with an input string."""
  if model.startswith("deepseek"):
    client = OpenAI(base_url="https://api.deepseek.com")
  elif model.startswith("anthropic"):
    client = client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],  # 只放 OpenRouter Key
    )
    check = client.models.list()
  else:
    client = OpenAI()
    check = client.models.list()
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
      elif model.startswith("anthropic"):
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            extra_body={"provider": {"only": ["amazon-bedrock"], "allow_fallbacks": False}},
        )
        print(completion.usage)
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
          messages, model=model, max_decode_steps=max_decode_steps, temperature=temperature
      )
    elif "rate_limit_exceeded" in str(e).lower() or "rate limit" in str(e).lower():
      retry_time = 30
      print(f"Rate limit exceeded. Retrying in {retry_time} seconds...")
      time.sleep(retry_time)
      return call_openai_server_single_prompt(
          messages, model=model, max_decode_steps=max_decode_steps, temperature=temperature
      )
    elif "api" in str(e).lower() and "error" in str(e).lower():
      retry_time = 30
      print(f"API error occurred: {e}. Retrying in {retry_time} seconds...")
      time.sleep(retry_time)
      return call_openai_server_single_prompt(
          messages, model=model, max_decode_steps=max_decode_steps, temperature=temperature
      )
    elif "connection" in str(e).lower():
      retry_time = 30
      print(f"Connection error occurred: {e}. Retrying in {retry_time} seconds...")
      time.sleep(retry_time)
      return call_openai_server_single_prompt(
          messages, model=model, max_decode_steps=max_decode_steps, temperature=temperature
      )
    elif "service unavailable" in str(e).lower():
      retry_time = 30
      print(f"Service unavailable: {e}. Retrying in {retry_time} seconds...")
      time.sleep(retry_time)
      return call_openai_server_single_prompt(
          messages, model=model, max_decode_steps=max_decode_steps, temperature=temperature
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
        messages, max_decode_steps=max_decode_steps, temperature=temperature
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


def call_local_server_func_parallel(
    messages_list, model, max_decode_steps=50, temperature=0.8, max_workers=1
):
    """The function to call local model server with a list of message lists in parallel.
    
    Note: max_workers is set to 1 by default for local models to avoid GPU memory conflicts.
    """
    if not isinstance(messages_list, list):
        raise ValueError("messages_list must be a list of message lists")
    
    if not messages_list:
        return []
    
    # If single message list is passed, wrap it in a list
    if isinstance(messages_list[0], dict):
        messages_list = [messages_list]
    
    outputs = [None] * len(messages_list)
    
    # For local models, we typically use max_workers=1 to avoid GPU memory conflicts
    # But we keep the ThreadPoolExecutor pattern for consistency
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(
                call_local_server_single_prompt,
                messages,
                model_path=model,
                max_decode_steps=max_decode_steps,
                temperature=temperature
            ): i for i, messages in enumerate(messages_list)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                outputs[index] = result
            except Exception as e:
                print(f"Error processing messages at index {index}: {e}")
                outputs[index] = None
    
    return outputs


def call_openai_server_func_parallel(
    messages_list, model="gpt-3.5-turbo", max_decode_steps=20, temperature=0.8, max_workers=10
):
  """The function to call OpenAI server with a list of message lists in parallel."""
  if not isinstance(messages_list, list):
    raise ValueError("messages_list must be a list of message lists")
  
  if not messages_list:
    return []
  
  # If single message list is passed, wrap it in a list
  if isinstance(messages_list[0], dict):
    messages_list = [messages_list]
  
  outputs = [None] * len(messages_list)
  
  with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Submit all tasks
    future_to_index = {
      executor.submit(
        call_openai_server_single_prompt,
        messages,
        model=model,
        max_decode_steps=max_decode_steps,
        temperature=temperature
      ): i for i, messages in enumerate(messages_list)
    }
    
    # Collect results as they complete
    for future in as_completed(future_to_index):
      index = future_to_index[future]
      try:
        result = future.result()
        outputs[index] = result
      except Exception as e:
        print(f"Error processing messages at index {index}: {e}")
        outputs[index] = None
  
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
