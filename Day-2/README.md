# Day 2 · Prompts, Memory and the Python Agents Need

**AI Agents Workshop · Level 1 Fundamentals · Day 2 of 2**  
**Duration:** 2 hours · 1 session · **Format:** Trainer 90% / Students 10% · **Environment:** Google Colab · **Case Study:** Sweet Crumbs Home Bakery

Day 1 gave you one API call. Today you learn to control what the model says (prompts), to make it remember (history), and the five pieces of Python that every agent in this workshop is built from. By the end, Sweet Crumbs has a chatbot that knows its menu, remembers the customer and never crashes.

---

## Today at a Glance

- **01 · Messages and Roles:** `system`, `user`, `assistant` (10 min)
- **02 · The System Prompt:** Role · Facts · Rules · Format (15 min)
- **03 · Few-Shot Examples:** Show, don't just tell (10 min)
- **04 · Asking for JSON:** Answers your code can use (15 min)
- **05 · Memory by Hand:** The chat loop + trimming (15 min)
- **06 · Python for Agents ①:** Dicts and JSON (10 min)
- **07 · Python for Agents ②:** Functions, type hints and docstrings (10 min)
- **08 · Python for Agents ③:** Decorators, the `@` sign (10 min)
- **09 · Python for Agents ④:** `try / except`, failing politely (10 min)

---

## 1. Messages and Roles

**Every call sends a list of messages. Each message has a role: `system`, `user` or `assistant`.**

The messages list is the whole conversation the model sees, top to bottom. `system` contains your instructions to the model (hidden from the customer), `user` holds what the person typed, and `assistant` holds what the model replied earlier. The model reads the list and writes the next assistant reply.

```python
messages = [
    {"role": "system", "content": "You are Sweet Crumbs' assistant..."},
    {"role": "user", "content": "Eggless available?"},
    {"role": "assistant", "content": "Yes, all our cakes come eggless!"},
    {"role": "user", "content": "Price of plum cake?"},
]
```

- `system` = instructions · `user` = the person · `assistant` = the model's earlier replies.
- The model always writes the next `assistant` message.

---

## 2. The System Prompt: Role · Facts · Rules · Format

**A good system prompt is a job description in four parts. Leave one out and you can predict exactly what goes wrong.**

| Part | What It Says | Sweet Crumbs Example | Missing It Causes… |
| :--- | :--- | :--- | :--- |
| **Role** | Who the model is, and for whom | You are the assistant for Sweet Crumbs Home Bakery. | Generic, robotic answers |
| **Facts** | What it must know | Orders need 24 hours' notice. All cakes come eggless. Plum cake Rs 500. | Made-up prices and policies |
| **Rules** | What it must and must never do | Never promise a delivery time. If unsure, say *"I'll check with the owner."* | Risky promises |
| **Format** | How the answer should look | Under 50 words, friendly, match the customer's language. | Long, rambling replies |

---

## 3. Few-Shot Examples: Show, Don't Just Tell

**Put two or three worked examples in the messages and the model copies their pattern, format included.**

Instructions alone leave room for format drift. Adding example question–answer pairs as earlier `user`/`assistant` turns shows the model the exact output shape your code expects.

```python
messages = [
    {"role": "system", "content": "Label the customer message with one word: order, question or complaint."},
    {"role": "user", "content": "need 2 plum cakes for friday"},
    {"role": "assistant", "content": "order"},
    {"role": "user", "content": "do you have eggless?"},
    {"role": "assistant", "content": "question"},
    {"role": "user", "content": "the cake was dry yesterday"},
    {"role": "assistant", "content": "complaint"},
    {"role": "user", "content": new_message},
]
```

---

## 4. Asking for JSON: Answers Your Code Can Use

**People read sentences; programs read fields. Ask the model to reply in JSON, then turn it into a Python `dict` with `json.loads`.**

```python
import json

SYSTEM = """Extract the order from the customer's message.
Return ONLY JSON, no other text, with these keys:
name (text), item (text), qty (number), date (text), eggless (true/false)."""

reply = response.choices[0].message.content  # JSON string
order = json.loads(reply)  # string -> Python dict
print(order["item"], order["qty"])
```

- Name every key and its type in the prompt and ask for JSON only.
- Wrap `json.loads` in `try / except` because models occasionally add extra text or markdown fences.

---

## 5. Memory by Hand: The Chat Loop

**To make a chatbot remember, keep a list of messages and append every question and every reply to it.**

Because chat models are stateless per call, your code maintains the conversation list and trims old turns when needed to manage token cost and latency.

```python
history = [{"role": "system", "content": SYSTEM}]
MAX_TURNS = 6  # keep the last 6 messages after system

def chat(user_text: str) -> str:
    history.append({"role": "user", "content": user_text})
    recent = [history[0]] + history[1:][-MAX_TURNS:]
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=recent,
        temperature=0.3,
    )
    reply = response.choices[0].message.content
    history.append({"role": "assistant", "content": reply})
    return reply
```

---

## 6. Python for Agents ①: Dicts and JSON

**A `dict` stores data as `key -> value` pairs. It is how agents hold menus, orders and messages.**

```python
MENU = {"plum cake": 500, "chocolate cake": 650, "cupcake": 80}

print(MENU["plum cake"])  # 500
print(MENU.get("donut", "not on menu"))  # 'not on menu' (no KeyError)

order = {"name": "Ravi", "item": "plum cake", "qty": 2, "eggless": True}
total = order["qty"] * MENU[order["item"]]  # 1000

import json
text = json.dumps(order)  # dict -> JSON string
back = json.loads(text)   # JSON string -> dict
```

- Use `.get(key, default)` for safe lookups and normalize keys with `.lower().strip()`.

---

## 7. Python for Agents ②: Functions, Type Hints and Docstrings

**A function is a named machine: inputs in, output out. Type hints and a docstring say what goes in and what comes out.**

```python
def get_price(item: str, qty: int = 1) -> int:
    """Return the total price in rupees for qty of an item on the Sweet Crumbs menu."""
    return MENU[item.lower().strip()] * qty

print(get_price("cupcake", 6))  # 480
print(get_price("Plum Cake"))   # 500
```

- When functions become agent tools on Day 4, the model relies on the function name, type hints, and docstring to decide when and how to call them.

---

## 8. Python for Agents ③: Decorators, the `@` Sign

**A decorator wraps a function to add extra behaviour, such as logging, without changing the function's code.**

```python
import functools

def log_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"-> calling {func.__name__}{args}")
        result = func(*args, **kwargs)
        print(f"<- {func.__name__} returned {result!r}")
        return result
    return wrapper

@log_call
def get_price(item: str, qty: int = 1) -> int:
    """Return the total price in rupees for qty of an item on the Sweet Crumbs menu."""
    return MENU[item.lower().strip()] * qty
```

- Placing `@deco` above `def f(...)` is equivalent to `f = deco(f)`.

---

## 9. Python for Agents ④: `try / except`, Failing Politely

**Things will go wrong: no internet, rate limits, bad JSON, unknown cakes. `try / except` catches the error and lets you reply politely instead of crashing.**

```python
from openai import RateLimitError, AuthenticationError, APIConnectionError

try:
    response = client.chat.completions.create(model=MODEL, messages=history)
    reply = response.choices[0].message.content
except RateLimitError:
    reply = "We're a little busy. Please try again in a minute."
except AuthenticationError:
    reply = "Setup problem: please tell the owner the API key isn't working."
except APIConnectionError:
    reply = "I can't reach the server right now. Please try again shortly."

print(reply)
```

- Catch specific exceptions and return clear messages instead of using bare `except: pass`.

---

## Today's Build: A Sweet Crumbs Chatbot That Remembers

Run the Day 1 setup cell (install, key, client) first in Google Colab, then run this build cell:

```python
import json
import functools
from openai import RateLimitError, APIConnectionError

MODEL = "openai/gpt-oss-20b"
MENU = {"plum cake": 500, "chocolate cake": 650, "cupcake": 80}

SYSTEM = f"""ROLE:   You are the friendly assistant for Sweet Crumbs Home Bakery.
FACTS:  Menu with prices in rupees: {json.dumps(MENU)}
        Orders need 24 hours' notice. All cakes have eggless versions.
RULES:  Never promise a delivery time. If it isn't in FACTS, say "I'll check with the owner."
        Remember the customer's name and order during the chat.
FORMAT: Under 50 words, friendly, English or Telugu matching the customer."""

def log_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f"   [log] {func.__name__}{args} -> {result!r}")
        return result
    return wrapper

@log_call
def get_price(item: str, qty: int = 1) -> int:
    """Return the total price in rupees for qty of an item on the menu."""
    return MENU[item.lower().strip()] * qty

history = [{"role": "system", "content": SYSTEM}]
MAX_TURNS = 8

def chat(user_text: str) -> str:
    history.append({"role": "user", "content": user_text})
    recent = [history[0]] + history[1:][-MAX_TURNS:]
    try:
        response = client.chat.completions.create(model=MODEL, messages=recent, temperature=0.3)
        reply = response.choices[0].message.content
    except RateLimitError:
        reply = "We're a little busy. Please try again in a minute."
    except APIConnectionError:
        reply = "I can't reach the server right now. Please try again shortly."
    history.append({"role": "assistant", "content": reply})
    return reply

print("Total for 6 cupcakes:", get_price("cupcake", 6))

while True:
    text = input("You: ")
    if text.lower().strip() in ("bye", "exit", "quit"):
        print("Bot: Thank you for visiting Sweet Crumbs!")
        break
    print("Bot:", chat(text))
```

### Test It Like a Customer

- Say your name, ask for an eggless chocolate cake for tomorrow, ask the price for two, then ask *"what did I order?"*.
- Verify it remembers your name and order, quotes **Rs 1,300**, and never promises a delivery time.

---

## Your 10%: Try It Different Ways

1. **Prompt parts:** Delete the `RULES` lines and ask *"Can you deliver in 1 hour?"*. Restore them and ask again.
2. **Few-shot:** Build a classifier for your business with three examples and test it on five new messages.
3. **JSON:** Ask the bot to return the order as JSON with `name`, `item`, `qty`, `date`, and `eggless`, then parse it with `json.loads` inside `try / except`.
4. **Memory:** Set `MAX_TURNS = 2`, state your name, chat for three turns, then ask your name and observe what is forgotten.
5. **Your business:** Rebuild the chatbot cell for your own business with its catalogue dict, a function with a docstring, and its own `SYSTEM` prompt.

> **Before Level 2 (Day 3):** Bring 10 customer questions and your business's catalogue as a Python `dict`.

---

## Summary: Day 2 in One Table

| Idea | One Line | Analogy |
| :--- | :--- | :--- |
| **Messages and roles** | `system` = instructions, `user` = person, `assistant` = model's replies | A drama script |
| **System prompt** | Role · Facts · Rules · Format | First-day briefing |
| **Few-shot** | A few worked examples fix the format | Showing a cashier two bills |
| **JSON output** | Ask for fields, then `json.loads` | Application form, not a letter |
| **Memory by hand** | Append, send all, trim old turns | Reading the notebook aloud |
| **Dicts and JSON** | `key -> value`; `.get` doesn't crash | Phone contacts |
| **Functions** | One job, type hints, docstring | Labelled vending machine |
| **Decorators** | `@d` means `f = d(f)` | Gift wrapping |
| **`try / except`** | Catch specific errors, reply politely | Safety net |
