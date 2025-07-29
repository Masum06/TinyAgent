import os, json

## First set OpenAI API key
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

from openai import OpenAI
client = OpenAI()

import json
import tiktoken

class TinyAgent:
  def __init__(self, name="", system=""):
    self.name = name
    self.system = system
    self.messages = []
    self.temperature = 1
    self.max_tokens = 2048
    self.model = "gpt-4.1"
    if name:
      self.add_system_message(f"Your name is {name}.")
    if system:
      self.add_system_message(system)

  def token_count(self, string):
    encoding = tiktoken.encoding_for_model("gpt-4o") # 4.1 not in tiktoken yet
    num_tokens = len(encoding.encode(string))
    return num_tokens

  def prompt_token_count(self):
    prompt = ""
    for message in self.messages:
      prompt += message["content"]
    return self.token_count(prompt)

  def add_message(self,message_type, message):
    self.messages.append({"role": message_type, "content":message})

  def add_system_message(self, message):
    self.add_message("system", message)

  def add_user_message(self, message):
    self.add_message("user", message)

  def add_instruction(self, instruction):
    self.add_system_message(f"Follow this instruction: \n{instruction}\n\n")

  def add_example(self, input, output):
    self.add_system_message(f"Example Input: {input}\nExample Output: {output}\n\n")

  def add_data(self, data):
    self.add_user_message(f"Data: {data}\n\n")

  def set_temperature(self, temperature):
    self.temperature = temperature

  def set_max_tokens(self, max_tokens):
    self.max_tokens = max_tokens

  def set_model(self, model):
    self.model = model

  def call(self, prompt="", response_type="text", cache=True):
    temp_messages = self.messages.copy()
    if prompt:
      temp_messages.append({"role": "user", "content": prompt})
    if cache:
      self.add_message("user", prompt)
    response = client.chat.completions.create(
      model=self.model,
      messages=temp_messages,
      temperature=self.temperature,
      max_tokens=max(self.max_tokens, int(self.prompt_token_count()*2)),
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
      response_format={
        "type": response_type # "text" or "json_object"
      }
    )
    reply = response.choices[0].message.content
    if cache:
      self.add_message("assistant", reply)
    return reply

  def load_json(self,s):
    s = s.strip()
    if s[0] != "{" or s[-1] != "}":
      return None
    try:
        # Attempt to parse the matched JSON
        return json.loads(s)
    except json.JSONDecodeError:
        # Return None if JSON parsing fails
        return None

  def call_json(self, prompt="", cache=True):
    prompt += "\n\nOutput must be JSON format. Don't say anything else.\n\n"
    reply = self.call(prompt, response_type="json_object", cache=cache)
    try:
      reply = reply.replace("```json", "").replace("```", "").strip()
      if reply[-1] != "}":
        raise Exception("Incomplete JSON")
      reply_json = self.load_json(reply)
    except Exception as e:
      print(e)
      reply_json = None

    return reply_json
