# Day 1 · How LLMs Work, and Your First API Call

**AI Agents Workshop · Level 1 Fundamentals · Day 1 of 2**  
**Duration:** 2 hours · 1 session · **Format:** Trainer 90% / Students 10% · **Environment:** Google Colab · **Case Study:** Sweet Crumbs Home Bakery

Everything you need before building agents: what a large language model really does, why it behaves the way it does, and the lines of Python that let you call one. Over the workshop we build an AI team for **Sweet Crumbs Home Bakery**—which takes cake orders on chat, writes them in a notebook, and answers repeat questions—starting today with one API call that answers a bakery question.

---

## Today at a Glance

- **01 · Why AI Is Everywhere Now:** APIs changed everything (10 min)
- **02 · What an LLM Does:** Next-token prediction (10 min)
- **03 · Tokens:** The unit of cost and limits (10 min)
- **04 · Temperature:** The creativity dial (10 min)
- **05 · No Memory:** Why the app resends the chat (10 min)
- **06 · Context Window:** The token budget per call (10 min)
- **07 · Hallucination:** Confident but wrong, and the fix (15 min)
- **08 · The AI Engineer:** Your job, and the 4-step pattern (10 min)
- **09 · Your First API Call:** Line by line in Colab (25 min)

---

## 1. Why AI Is Everywhere Now

**AI is 70 years old. What is new is that anyone can use a powerful model with a few lines of code.**

For decades, building AI meant collecting data, hiring PhDs, and training models for months. Two shifts changed that:
- **General-purpose capability:** One model can write, summarize, translate, and explain code.
- **Collapsed barrier via APIs:** Providers host giant models on their servers and expose them through an API—your code sends a request, and their server returns an answer. Apps built this way are called *wrappers*.

At Sweet Crumbs, we rent a model through an API rather than training one from scratch.

---

## 2. What an LLM Does

**An LLM does one thing: it predicts the next piece of text (token), again and again, in a loop.**

Given input text, the model scores candidate next tokens, selects one, appends it to the sequence, and repeats until it stops. Chatting, coding, summarizing, and translating all run on this next-token loop—which is why streamed replies appear token by token.

---

## 3. Tokens: The Unit of Everything

**A model reads and writes tokens—pieces of words. Cost and context limits are measured in tokens.**

- Common words are often a single token; rare or long words split into sub-word pieces.
- **Rule of thumb:** `1 token ≈ ¾ of an English word` (so `300 words ≈ 400 tokens`).
- Scripts and languages vary in token density—for example, Telugu typically requires more tokens than English for the same meaning, increasing token usage and API cost.

---

## 4. Temperature: The Creativity Dial

**Temperature controls how adventurous the model is when sampling the next token.**

- **Low temperature (`0`–`0.3`):** Sharpens probability toward the highest-scoring tokens—focused, consistent, and repeatable (ideal for prices, facts, rules, and support).
- **Moderate/High temperature (`~0.7`–`1.2`):** Flattens the distribution—more varied and creative (useful for brainstorming or marketing captions).

---

## 5. The Model Has No Memory

**Between API calls, an LLM is stateless. Chat apps appear to remember because the client code resends the conversation history.**

Each API request is independent. If your code only sends the newest message without earlier turns, the model cannot recall names or details from earlier in the session—and sending longer history increases input tokens on every call.

---

## 6. The Context Window: The Token Budget

**The context window is the maximum number of tokens one call can hold: system instructions, chat history, reference text, and the generated reply combined.**

- Everything must fit within the model's token budget (for example, `128,000` tokens).
- Exceeding the limit causes errors or truncated output, and larger inputs increase latency and per-call cost.

---

## 7. Hallucination: Confident but Wrong, and the Fix

**Models are trained to generate plausible text, not to verify real-world facts—creating confident yet inaccurate answers known as hallucinations.**

When asked about unseen policies or specifics (such as a bakery's delivery rules), a model may guess smoothly if left ungrounded. Grounding replies in provided facts (**RAG**—supplying relevant context at query time and instructing the model to say when something is missing rather than guessing) and using low temperature help prevent fabricated answers.

---

## 8. The AI Engineer: Where You Fit

**An AI engineer builds reliable products on top of existing models rather than training foundation models from scratch.**

Key responsibilities include selecting models, crafting prompts and context, managing token cost, handling edge cases, and shipping the interface. Most AI products follow four core steps:

1. **Connect:** Make an API call to a model endpoint.
2. **Give context:** Provide relevant facts, rules, or data.
3. **Get the answer:** Read and process the model's response.
4. **Wrap in a UI:** Expose it via a web app, mobile app, or chat bot.

---

## 9. Your First API Call, Line by Line

### Step 0 · Store Your API Key in Colab Secrets

1. Create a key at [Groq Console](https://console.groq.com) (`API Keys`). Keep keys secret and never commit them to GitHub.
2. In Google Colab, open the **Secrets** (key icon) panel in the left sidebar, add `GROQ_API_KEY`, paste your key, and enable **Notebook access**.

### The Program

```python
!pip install -q openai  # once per Colab session
from google.colab import userdata  # reads Colab Secrets
from openai import OpenAI  # OpenAI-compatible client

client = OpenAI(
    api_key=userdata.get("GROQ_API_KEY"),
    base_url="[https://api.groq.com/openai/v1](https://api.groq.com/openai/v1)",  # route requests to Groq
)

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "system", "content": "You are a funny joker"},
        {"role": "user", "content": "tell me a joke"},
    ],
)

print(response.choices[0].message.content)
```

| Line / Concept | Role |
| :--- | :--- |
| `pip install openai` | Installs the OpenAI-compatible Python client library |
| `userdata.get("GROQ_API_KEY")` | Loads your API key from Colab Secrets at runtime |
| `OpenAI(api_key=..., base_url=...)` | Initializes the client object (no network call yet) |
| `client.chat.completions.create(...)` | Sends the request over HTTPS and waits for the response |
| `response.choices[0].message.content` | Extracts the generated reply text from the response object |

| Common Error | Cause | Fix |
| :--- | :--- | :--- |
| Missing `api_key` error | Secret not loaded or notebook access off | Verify `GROQ_API_KEY` name and toggle Notebook access |
| `401 AuthenticationError` | Invalid or expired key | Generate a fresh API key |
| `404` model error | Incorrect or unavailable model ID | Check available models with `client.models.list()` |
| `429 RateLimitError` | Request/token rate limit exceeded | Wait and retry; avoid sharing one key across many users |

---

## Today's Build: Sweet Crumbs' First Answer

Run this in Google Colab after setting up `client` with `GROQ_API_KEY`:

```python
SYSTEM = """ROLE:   You are the assistant for Sweet Crumbs Home Bakery.
FACTS:  We bake to order. Orders need 24 hours' notice. All cakes have eggless versions.
        Plum cake Rs 500. Chocolate cake Rs 650. Cupcake Rs 80.
RULES:  Never promise a delivery time. If the answer isn't in FACTS, say "I'll check with the owner."
FORMAT: Under 50 words, friendly, English or Telugu matching the customer."""

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    temperature=0.2,  # focused for facts and pricing
    messages=[
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Can I get an eggless cake today evening?"},
    ],
)

print(response.choices[0].message.content)
print(response.usage)
```

### Test It Like a Customer

- Ask `"Can I get an eggless cake today evening?"` and verify the assistant notes eggless cakes are available while explaining that orders require **24 hours' notice**, rather than promising same-day evening fulfillment.

---

## Your 10%: Try It Different Ways

1. **Temperature:** Ask *"Write a tagline for Sweet Crumbs"* three times at `0.2` and three times at `1.2`, then compare consistency vs. variety.
2. **Hallucination:** Remove the `FACTS` line and ask *"Do you deliver to Kakinada?"* to observe ungrounded guessing.
3. **No memory:** Make two separate calls (`"My name is Ravi"`, then `"What is my name?"`), then include both turns in a single `messages` list.
4. **Tokens:** Send the same prompt in English and Telugu and compare `response.usage.prompt_tokens`.
5. **Your business:** Write a `SYSTEM` prompt for your chosen business (5 facts, 3 rules) and test 3 customer questions.

> **Before Day 2:** List the 10 most common customer questions for your chosen business so you can turn them into a system prompt, conversational memory, and a test file.

---

## Summary: Day 1 in One Table

| Idea | One Line | Analogy |
| :--- | :--- | :--- |
| **Why now** | Ready-made models accessed through an API | Ordering via an app instead of running a kitchen |
| **LLM** | Predicts the next token iteratively in a loop | Autocomplete scaled up |
| **Token** | Roughly ¾ of an English word; unit of cost and limits | Building blocks of text |
| **Temperature** | Low = focused/deterministic; high = varied/creative | Strict grader vs. open-ended brainstorm |
| **No memory** | Client code must resend conversation history | Passing the full chart at each visit |
| **Context window** | Shared token budget for prompt, history, context, and reply | Fixed-capacity container |
| **Hallucination** | Fluent but unverified output; ground in facts/RAG | Open-book reference instead of guessing |
| **AI engineer** | Builds reliable products around existing models | Assembling the product around a prebuilt engine |
| **API call** | Construct client once; call `.create(...)` per turn | Configuring a phone, then placing a call |
