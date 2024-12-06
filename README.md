
# TinyAgent

TinyAgent is a lightweight Python library for creating and managing conversational agents using a GPT-based model. The library provides an easy-to-use interface for building multi-turn conversations, defining system behavior, and interacting with a model API.

---

## Features

- **Customizable Agent Configuration**: Set agent names, behaviors, and system messages.
- **Conversation Management**: Add system, user, and assistant messages to simulate multi-turn conversations.
- **Instructions and Examples**: Add explicit instructions and examples to guide model responses.
- **Temperature and Token Management**: Fine-tune the generation behavior using temperature and token limits.
- **Flexible Output Formats**: Support for text and JSON object responses.
- **Lightweight and Intuitive API**: Designed for rapid prototyping and development.

---

## Installation

Clone the repository and install the required dependencies.

```bash
git clone https://github.com/Masum06/TinyAgent.git
cd TinyAgent
pip install -r requirements.txt
```

---

## Usage

### Initialize an Agent

```python
from tinyagent import TinyAgent

# Create an agent with a name and system behavior
agent = TinyAgent(name="TinyAgent", system="You are a helpful assistant.")
```

### Add Messages

```python
# Add a system message
agent.add_system_message("Always provide concise and accurate answers.")

# Add a user message
agent.add_user_message("What is the capital of France?")
```

### Add Instructions and Examples

```python
# Add explicit instructions
agent.add_instruction("Respond politely and professionally.")

# Add input-output examples
agent.add_example("What is 2+2?", "4")
```

### Configure Agent Behavior

```python
# Adjust the temperature (creativity)
agent.set_temperature(0.7)

# Set a maximum token limit
agent.set_max_tokens(1500)

# Change the model
agent.set_model("gpt-4o")
```

### Generate a Response

```python
# Generate a text response
response = agent.call(prompt="Can you tell me about the Eiffel Tower?")
print(response)

# Generate a JSON response
json_response = agent.call_json(prompt="Provide details about Python programming.")
print(json_response)
```

---

## API Reference

### Class: `TinyAgent`

#### `__init__(name="", system="")`
- **Parameters**:
  - `name` (str): Name of the agent.
  - `system` (str): Initial system message to define the agent's behavior.

#### `add_message(message_type, message)`
- **Description**: Adds a message to the conversation.
- **Parameters**:
  - `message_type` (str): Role of the message (`system`, `user`, or `assistant`).
  - `message` (str): Message content.

#### `add_system_message(message)`
- **Description**: Adds a system message to the conversation.

#### `add_user_message(message)`
- **Description**: Adds a user message to the conversation.

#### `add_instruction(instruction)`
- **Description**: Adds an instruction to guide the agent's responses.

#### `add_example(input, output)`
- **Description**: Adds an input-output example to the conversation.

#### `add_data(data)`
- **Description**: Adds data as part of the user message.

#### `set_temperature(temperature)`
- **Description**: Adjusts the creativity of responses.

#### `set_max_tokens(max_tokens)`
- **Description**: Sets the maximum number of tokens for responses.

#### `set_model(model)`
- **Description**: Changes the underlying model used for responses.

#### `call(prompt="", response_type="text")`
- **Description**: Generates a response from the agent.
- **Parameters**:
  - `prompt` (str): User input or question.
  - `response_type` (str): Response format (`text` or `json_object`).
- **Returns**: Model-generated response (str or JSON object).

#### `load_json(s)`
- **Description**: Parses a string as JSON.
- **Parameters**:
  - `s` (str): Input string.
- **Returns**: Parsed JSON object or `None`.

#### `call_json(prompt="")`
- **Description**: Generates a JSON response.
- **Parameters**:
  - `prompt` (str): User input or question.
- **Returns**: Parsed JSON response.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any bugs or feature requests.

---

## Contact

For questions or support, feel free to reach out at [m.hasan@rochester.edu](mailto:m.hasan@rochester.edu).
