# Day 3 · Your First Agent: What an Agent Is, and LangChain

**AI Agents Workshop · Level 2 (Single Agent) · Day 3 (Session 1 of 3)**  
**Duration:** 2 hours · **Format:** Trainer 70% / Students 30% · **Environment:** Google Colab · **Case Study:** Sweet Crumbs Home Bakery

A chatbot **talks**. An agent **gets things done**. Today you learn what turns a chatbot into an agent, how an agent thinks in a loop, when you should (and shouldn't) use one, and how LangChain makes the code shorter. You finish with the Sweet Crumbs assistant rebuilt in LangChain, and a test file that checks it.

---

## Today's Agenda

- **01 · Chatbot vs Agent:** Talking vs doing (10 min)
- **02 · The Four Parts of an Agent:** Brain, instructions, tools, memory (10 min)
- **03 · The Agent Loop:** Think → act → observe → repeat (15 min)
- **04 · Three Levels of AI:** Traditional, generative, agentic (10 min)
- **05 · Agent or Not?:** Pick the simplest thing that works (10 min)
- **06 · Why LangChain:** One adapter for every model (10 min)
- **07 · LangChain Basics:** Models, templates, chains (20 min)
- **08 · Testing Your Assistant:** `tests.csv` and pass rate (15 min)

---

## 1. Chatbot vs Agent

**A chatbot answers with words. An agent is given a goal and takes actions until the goal is done.**

Ask a chatbot *"Cancel my order"* and it tells you *how* to cancel. Ask an agent and it *cancels it*: it looks up the order, calls the cancel system, checks it worked, and then tells you. The difference is not a smarter model; it's that the agent is **allowed to use tools** and **decides the steps itself**.

- **Chatbot:** Answers with instructions or explanations—the customer still has to do the work.
- **Agent:** Executes actions with tools, checks outcomes, and returns a confirmed result.

---

## 2. The Four Parts of an Agent

**Every agent has the same four parts: a brain, instructions, tools, and memory.**

| Part | What It Is | At Sweet Crumbs |
| :--- | :--- | :--- |
| **Brain** | The LLM that reads and decides | `openai/gpt-oss-20b` on Groq |
| **Instructions** | The system prompt: role, facts, rules, format | *"You are Sweet Crumbs' assistant…"* |
| **Tools** | Python functions it is allowed to call | `get_price`, `is_open`, `log_order` |
| **Memory** | What it keeps between steps and messages | Chat history; later, saved facts |

- Without **tools**, it can only talk.
- Without **instructions**, it guesses.
- Without **memory**, it forgets.

---

## 3. The Agent Loop: Think → Act → Observe → Repeat

**An agent works in a loop: decide the next step, do it, look at the result, and repeat until the goal is reached.**

Often called **ReAct** (*Reason* + *Act*), in each round the model either requests a tool call (*act*) or provides a final response (*stop*). Your code executes the tool and returns the result (*observe*).

- **The model decides; your code runs the tools.**
- Always set a limit on rounds so loops cannot run endlessly.

---

## 4. Three Levels of AI

**Traditional software follows fixed rules. Generative AI creates content. Agentic AI uses that ability to take actions.**

| Level | What It Does | Who Decides the Steps | Bakery Example |
| :--- | :--- | :--- | :--- |
| **Traditional software** | Follows rules a programmer wrote | The programmer, in advance | Bill calculator: price × quantity |
| **Generative AI** | Creates text, images, or code when asked | The person asking, one request at a time | *"Write a Diwali offer post"* → a draft |
| **Agentic AI** | Plans and acts with tools to reach a goal | The model, step by step | Reads an order, checks price, logs it, confirms |

- Higher levels bring more freedom, along with higher cost and risk.

---

## 5. Agent or Not? Pick the Simplest Thing That Works

**Use an agent only when the steps can't be known in advance. If you can write the steps down, plain code is cheaper, faster, and safer.**

Ask three questions in order:
1. **Can I write the exact steps?** → Write normal code (no AI).
2. **Do I just need text understood or written once?** → One LLM call.
3. **Must it decide which steps to take, using tools?** → An agent.

---

## 6. Why LangChain?

**LangChain is a Python library that gives every model the same simple interface, plus ready-made pieces for prompts, tools, memory, and agents.**

Switching providers (e.g., Groq, OpenAI, Google Gemini) only changes the model/provider setup and API key while keeping prompt templates, chains, and orchestration consistent.

---

## 7. LangChain Basics: Model, Template, Chain

Three core building blocks:
1. **Chat model:** Invoked with `.invoke(...)`, returning a response whose text lives in `.content`.
2. **Prompt template:** `ChatPromptTemplate.from_messages(...)` with `{curly_brace}` blanks filled at runtime.
3. **Chain:** Joins steps with the `|` pipe operator (e.g., `prompt | llm | StrOutputParser()`).

> **Common Trap:** Because `{...}` denotes template variables, raw curly braces (such as JSON) in a template can trigger missing-variable errors unless escaped (`{{ ... }}`) or formatted as plain text.

---

## 8. Testing Your Assistant: `tests.csv`

**Write down real customer questions and what a correct answer must contain. Run them all after every change.**

Each row in `tests.csv` defines a `question` and a required substring (`must_include`). Re-running the full suite after prompt or parameter edits prevents quiet regressions and tracks your **pass rate**.

```csv
question,must_include
How much is a plum cake?,500
Do you make eggless cakes?,eggless
Can I get a cake today evening?,24 hours
Do you deliver to Kakinada?,check with the owner
What is the price of 6 cupcakes?,480
```

---

## Today's Build: The Sweet Crumbs Assistant in LangChain

Run these cells in order in a new Google Colab notebook with `GROQ_API_KEY` stored in Colab Secrets.

### Cell 1 · Setup

```python
!pip install -q langchain langchain-groq

import os
from google.colab import userdata
from langchain_groq import ChatGroq

os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")  # LangChain reads it from here
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.3)
```

### Cell 2 · The Assistant: Template + Memory + Chain

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

SYSTEM = """ROLE:   You are the friendly assistant for Sweet Crumbs Home Bakery.
FACTS:  Plum cake Rs 500. Chocolate cake Rs 650. Cupcake Rs 80 each.
        Orders need 24 hours' notice. All cakes have eggless versions.
RULES:  Never promise a delivery time or place. If it isn't in FACTS,
        say "I'll check with the owner." Work out totals carefully.
FORMAT: Under 50 words, friendly, English or Telugu matching the customer."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    MessagesPlaceholder("history"),  # the chat so far goes here
    ("human", "{question}"),
])
chain = prompt | llm | StrOutputParser()

history = []

def ask(question: str) -> str:
    answer = chain.invoke({"history": history, "question": question})
    history.extend([("human", question), ("ai", answer)])
    return answer

print(ask("Hi, I'm Ravi. Do you make eggless cakes?"))
print(ask("What's my name?"))
```

### Cell 3 · Tests

```python
import csv

def run_tests(path: str = "tests.csv") -> None:
    rows = list(csv.DictReader(open(path)))
    passed = 0
    for row in rows:
        answer = chain.invoke({"history": [], "question": row["question"]})
        ok = row["must_include"].lower() in answer.lower()
        passed += ok
        print("PASS" if ok else "FAIL", "|", row["question"], "→", answer[:70])
    print(f"\n{passed}/{len(rows)} passed ({passed / len(rows):.0%})")

run_tests()  # create tests.csv first with the %%writefile cell from topic 8
```

### Cell 4 · Sneak Peek: The Same Assistant as a LangChain Agent

```python
from langchain.agents import create_agent

agent = create_agent(model=llm, tools=[], system_prompt=SYSTEM)  # tools come on Day 4
result = agent.invoke({"messages": [{"role": "user", "content": "Do you make eggless cakes?"}]})
print(result["messages"][-1].content)
```

### Test It Like a Customer

- Ask about eggless options, a same-day cake, delivery to your town, and the price of 6 cupcakes, then run the tests. A solid first prompt should pass at least 4 of 5.

---

## Your 30%: Do It Yourself

1. **Your business:** Rewrite `SYSTEM` for your chosen business and update the business details.
2. **Your tests:** Turn 10 customer questions into `tests.csv` with a `must_include` value for each, run them, and record the pass rate.
3. **Improve:** Change one part of the prompt, re-run the tests, and note whether the pass rate went up or down.
4. **Switch:** Set `temperature=1.0` and run the tests twice to see if results vary across runs.
5. **Agent or not:** List three tasks in your business and label each as plain code, one LLM call, or an agent.

> **Before Day 4:** Write two Python functions your business needs (such as a price lookup and an availability check) with type hints and docstrings so they can be turned into agent tools.

---

## Summary: Day 3 in One Table

| Idea | One Line | Analogy |
| :--- | :--- | :--- |
| **Chatbot vs agent** | Answers vs acts to reach a goal | Receptionist vs personal assistant |
| **Four parts** | Brain, instructions, tools, memory | A new shop assistant |
| **Agent loop** | Think → act → observe → repeat | Cooking and tasting |
| **Three levels** | Fixed rules → creates → creates and acts | Calculator, writer, intern |
| **Agent or not** | Use the simplest option that works | Don't hire a manager to switch on a fan |
| **LangChain** | One interface for many models + building blocks | Universal travel adapter |
| **Template and chain** | Blanks in `{}`; `prompt \| llm \| parser` | Form on a conveyor belt |
| **Tests** | Real questions + `must_include`; re-run every change | Practice test with answer key |
