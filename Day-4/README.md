# Day 4 · Tools, Structured Output and Google Sheets

**AI Agents Workshop · Level 2 (Single Agent) · Day 4 (Session 2 of 3)**  
**Duration:** 2 hours · **Format:** Trainer 70% / Students 30% · **Environment:** Google Colab · **Case Study:** Sweet Crumbs Home Bakery

Yesterday the assistant could only talk. Today it gets **hands**: Python functions it can call by itself to check prices and opening hours, and to save orders. You also learn to turn a messy chat into a clean, checked record, and to write it into a real Google Sheet.

---

## Today's Agenda

- **01 · Why Agents Need Tools:** Live facts and real actions (10 min)
- **02 · How Function Calling Works:** The model asks, your code runs (15 min)
- **03 · Writing Good Tools:** `@tool`, names, docstrings (15 min)
- **04 · The Agent With Tools:** `create_agent` and the trace (15 min)
- **05 · Rate Limits and Tool Errors:** `429`, retries, polite failures (10 min)
- **06 · Structured Output With Pydantic:** A form with checks (15 min)
- **07 · Saving to Google Sheets:** `gspread` in Colab (10 min)
- **08 · Putting It Together:** Chat → order → sheet row (10 min)

---

## 1. Why Agents Need Tools

A model only knows what it learned and what is in the prompt. **Tools** let it fetch live facts and take real actions.

- An LLM cannot check today's stock, read your database, perform reliable arithmetic on big numbers, or save records on its own—it only produces text.
- A *tool* is a normal Python function you allow the model to use. The model decides *when* a tool is needed; your function does the actual work accurately and safely.

---

## 2. How Function Calling Works

The model never runs your code directly. It replies with a request (*"call `get_price` with these arguments"*), and your program runs the function and sends the result back.

1. You pass tool descriptions (name, purpose, and input schema) along with messages.
2. When needed, the model emits a **tool call** instead of a final text reply.
3. Your code executes the Python function, appends the output as a **tool** message, and invokes the model again.
4. This repeats until the model produces the final response.

---

## 3. Writing Good Tools With `@tool`

In LangChain, `@tool` turns a normal Python function into a tool. Its name, type hints, and docstring form the specification the model reads.

```python
from langchain.tools import tool

MENU = {"plum cake": 500, "chocolate cake": 650, "cupcake": 80}

@tool
def get_price(item: str, qty: int = 1) -> str:
    """Get the total price in rupees for a quantity of one menu item.

    Args:
        item: Menu item, e.g. "plum cake", "chocolate cake" or "cupcake"
        qty: How many the customer wants
    """
    price = MENU.get(item.lower().strip())
    if price is None:
        return f"'{item}' is not on the menu. We have: {', '.join(MENU)}."
    return f"{qty} x {item} = Rs {price * qty}"
```

| Rule | Why It Matters |
| :--- | :--- |
| **`snake_case` name** describing the action (`get_price`, not `tool1`) | First signal the model reads when selecting a tool |
| **Type hints** on every input | Used to generate the parameter schema |
| **Docstring** stating when to use it + input examples | Guides accurate tool selection |
| **Single responsibility** per tool | Easier to choose, reason about, and unit-test |
| **Graceful return messages** on invalid input instead of crashing | Lets the model ask the customer for clarification |

---

## 4. The Agent With Tools: `create_agent`

Pass a model, a list of tools, and a system prompt to `create_agent`. It manages the think → act → observe loop and returns the full message history (**trace**).

```python
from langchain.agents import create_agent

agent = create_agent(model=llm, tools=[get_price, is_open], system_prompt=SYSTEM)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Are you open on Sunday? And how much for 2 chocolate cakes?"}]
})

for m in result["messages"]:
    m.pretty_print()  # Full trace: user prompt -> tool calls -> tool outputs -> final reply

print(result["messages"][-1].content)  # Final answer
```

- Each tool call typically requires multiple model invocations (one to request the tool, another to synthesize the answer after receiving the tool result).
- Inspecting the trace shows whether the model picked the right tool, passed valid arguments, or misinterpreted a tool output.

---

## 5. Rate Limits and Tool Errors

Free API tiers enforce request and token-per-minute limits. Because an agent can make multiple LLM calls per user message, rate limits (`429 Too Many Requests`) can happen quickly.

```python
from langchain_groq import ChatGroq

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2, max_retries=3)

try:
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    answer = result["messages"][-1].content
except Exception as e:
    print("Agent error:", e)
    answer = "Sorry, I'm a little busy. Please try again in a minute."
```

Three habits to prevent failures:
1. **Use separate API keys per learner/user** rather than sharing one key across many concurrent users.
2. **Keep prompts concise and test in small batches** rather than firing large bursts at once.
3. **Wrap agent invocations in `try/except`** and return helpful strings from tools so errors never crash the chat.

---

## 6. Structured Output With Pydantic

Define the desired record schema as a **Pydantic** `BaseModel` and use `.with_structured_output(...)` so fields are typed and validated before your code uses them.

```python
from typing import Literal
from pydantic import BaseModel, Field

class Order(BaseModel):
    """A cake order taken from a customer's chat message."""
    name: str = Field(description="Customer's first name")
    item: Literal["plum cake", "chocolate cake", "cupcake"] = Field(description="Menu item")
    qty: int = Field(description="How many", ge=1)
    date: str = Field(description="Pickup date as the customer said it, e.g. '24 Dec'")
    eggless: bool = Field(description="True if the customer asked for eggless")

extractor = llm.with_structured_output(Order)
order = extractor.invoke("hi ravi here, need 2 plum cakes for 24th dec, eggless pls")
print(order)
print(order.qty * 500)  # Verified integer ready for arithmetic
```

---

## 7. Saving to Google Sheets

Google Sheets gives small business owners a familiar shared table that updates live. In Google Colab, authenticate and append rows with `gspread`:

```python
from google.colab import auth
import gspread
from google.auth import default

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

sh = gc.create("Sweet Crumbs Orders")  # Run once; later use gc.open("Sweet Crumbs Orders")
ws = sh.sheet1
ws.append_row(["name", "item", "qty", "date", "eggless", "total"])
ws.append_row(["Ravi", "plum cake", 2, "24 Dec", "yes", 1000])
```

- Keep column ordering consistent across `ws.append_row([...])` calls so every order maps cleanly to one row.

---

## 8. Putting It Together: The `log_order` Tool

Combine validation, total calculation, and sheet persistence into a single `log_order` tool, instructing the model via its docstring to call it only once required order details are present.

```python
@tool
def log_order(name: str, item: str, qty: int, date: str, eggless: bool) -> str:
    """Save a confirmed cake order to the Orders sheet.
    Use only when the customer has given their name, item, quantity and pickup date."""
    try:
        order = Order(name=name, item=item, qty=qty, date=date, eggless=eggless)
    except Exception as e:
        return f"Could not save the order: {e}. Ask the customer to correct it."
    total = MENU[order.item] * order.qty
    ws.append_row([order.name, order.item, order.qty, order.date, "yes" if order.eggless else "no", total])
    return f"Saved: {order.qty} x {order.item} for {order.name} on {order.date}, Rs {total}."
```

---

## Today's Build: Sweet Crumbs Agent That Looks Up and Saves

Run these cells in order in a new Google Colab notebook with `GROQ_API_KEY` stored in Colab Secrets.

### Cell 1 · Setup

```python
!pip install -q langchain langchain-groq gspread

import os
from google.colab import userdata, auth
import gspread
from google.auth import default
from langchain_groq import ChatGroq

os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2, max_retries=3)

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
sh = gc.create("Sweet Crumbs Orders")  # Run once; next time use gc.open("Sweet Crumbs Orders")
ws = sh.sheet1
ws.append_row(["name", "item", "qty", "date", "eggless", "total"])
print("Sheet ready:", sh.url)
```

### Cell 2 · Data, the `Order` Model and Three Tools

```python
from typing import Literal
from pydantic import BaseModel, Field
from langchain.tools import tool

MENU = {"plum cake": 500, "chocolate cake": 650, "cupcake": 80}
HOURS = {
    "monday": "9 am to 8 pm",
    "tuesday": "9 am to 8 pm",
    "wednesday": "9 am to 8 pm",
    "thursday": "9 am to 8 pm",
    "friday": "9 am to 8 pm",
    "saturday": "9 am to 8 pm",
    "sunday": "10 am to 6 pm",
}

class Order(BaseModel):
    """A cake order taken from a customer's chat message."""
    name: str = Field(description="Customer's first name")
    item: Literal["plum cake", "chocolate cake", "cupcake"] = Field(description="Menu item")
    qty: int = Field(description="How many", ge=1)
    date: str = Field(description="Pickup date as the customer said it")
    eggless: bool = Field(description="True if the customer asked for eggless")

@tool
def get_price(item: str, qty: int = 1) -> str:
    """Get the total price in rupees for a quantity of one menu item.

    Args:
        item: Menu item, e.g. "plum cake", "chocolate cake" or "cupcake"
        qty: How many the customer wants
    """
    price = MENU.get(item.lower().strip())
    if price is None:
        return f"'{item}' is not on the menu. We have: {', '.join(MENU)}."
    return f"{qty} x {item} = Rs {price * qty}"

@tool
def is_open(day: str) -> str:
    """Check the bakery's opening hours for a day of the week, e.g. "sunday"."""
    hours = HOURS.get(day.lower().strip())
    return f"Open {hours} on {day}." if hours else f"'{day}' is not a day of the week."

@tool
def log_order(name: str, item: str, qty: int, date: str, eggless: bool) -> str:
    """Save a confirmed cake order to the Orders sheet.
    Use only when the customer has given their name, item, quantity and pickup date."""
    try:
        order = Order(name=name, item=item.lower().strip(), qty=qty, date=date, eggless=eggless)
    except Exception as e:
        return f"Could not save the order: {e}. Ask the customer to correct it."
    total = MENU[order.item] * order.qty
    ws.append_row([order.name, order.item, order.qty, order.date, "yes" if order.eggless else "no", total])
    return f"Saved: {order.qty} x {order.item} for {order.name} on {order.date}, Rs {total}."
```

### Cell 3 · The Agent

```python
from langchain.agents import create_agent

SYSTEM = """ROLE:   You are the friendly order assistant for Sweet Crumbs Home Bakery.
FACTS:  Orders need 24 hours' notice. All cakes have eggless versions.
RULES:  Always use get_price for prices and is_open for timings; never guess them.
        Before log_order, make sure you have name, item, quantity and pickup date; ask if missing.
        After saving, read the order back with the total.
FORMAT: Under 60 words, friendly, English or Telugu matching the customer."""

agent = create_agent(model=llm, tools=[get_price, is_open, log_order], system_prompt=SYSTEM)

history = []

def chat(text: str) -> str:
    global history
    try:
        result = agent.invoke({"messages": history + [{"role": "user", "content": text}]})
        history = result["messages"]
        return history[-1].content
    except Exception as e:
        print("Agent error:", e)
        return "Sorry, I'm a little busy. Please try again in a minute."

print(chat("Are you open on Sunday? How much are 2 chocolate cakes?"))
print(chat("I'm Ravi. Please book 2 eggless chocolate cakes for 24 Dec."))
```

### Cell 4 · Inspect the Trace

```python
for m in history:
    m.pretty_print()
```

### Test It Like a Customer

- Ask for a `"donut"` (not on the menu), place an order without a pickup date, and try ordering `0` cupcakes.
- Confirm the assistant clarifies or explains without writing invalid rows to Google Sheets, then append these scenarios to `tests.csv`.

---

## Your 30% Checklist

1. **Your tools:** Convert two business helper functions into `@tool` functions with clear docstrings.
2. **Your record:** Define a Pydantic `BaseModel` for a domain record (e.g., enquiry, booking, order) and test `.with_structured_output(...)` on 5 messy messages.
3. **Your sheet:** Set up a Google Sheet and a `log_*` tool to persist validated records.
4. **Break it:** Temporarily rename a tool to `tool1` with docstring `"does stuff"`, observe selection behavior, and restore the proper name/docstring.
5. **Read the trace:** Count tool calls and model calls for a single prompt using `.pretty_print()`.

> **Prep for Day 5:** Write 15–20 business facts (policies, timings, delivery areas, FAQs) into a text file so the agent can retrieve them via RAG rather than stuffing everything into the system prompt.

---

## Summary Table

| Idea | One Line | Analogy |
| :--- | :--- | :--- |
| **Tools** | Functions for live facts, actions, and exact work | Phone and register for the exam student |
| **Function calling** | Model asks, your code runs, result returns as a message | Waiter and kitchen |
| **Good tools** | `snake_case` name, type hints, docstring with when to use | Labelled kitchen drawers |
| **`create_agent`** | Model + tools + prompt; trace shows every step | Delivery tracking timeline |
| **Rate limits** | `429` = slow down; own key, small tests, `try/except` | Toll plaza |
| **Pydantic** | Fields + types + rules; rejects invalid data | Deposit slip checked at the counter |
| **Google Sheets** | `append_row` = one order, one row | Self-updating order register |
| **`log_order`** | Validate → compute total → save → confirm when complete | Careful billing clerk |
