# Day 5 · Memory and RAG: An Agent That Remembers and Looks Things Up

**AI Agents Workshop · Level 2 (Single Agent) · Day 5 (Session 3 of 3)**  
**Case Study:** Sweet Crumbs Home Bakery · **Environment:** Google Colab

Today the Sweet Crumbs agent learns two core capabilities:
1. **Conversation Memory:** Automatically remembers each customer's chat history per thread.
2. **RAG (Retrieval-Augmented Generation):** Searches bakery policies and FAQs dynamically instead of stuffing all rules into the system prompt.

---

## Today's Agenda

- **01 · Three Kinds of Memory:** Conversation, customer, and knowledge (10 min)
- **02 · Conversation Memory: Threads:** `InMemorySaver` checkpointer + `thread_id` (15 min)
- **03 · Why Not Put Everything in the Prompt?:** Token cost, latency, and focus (10 min)
- **04 · Embeddings:** Representing meaning as numeric vectors (15 min)
- **05 · Vector Database: ChromaDB:** Semantic search over bakery facts (15 min)
- **06 · Chunking:** Splitting documents into focused pieces with overlap (10 min)
- **07 · RAG: Retrieve, Then Answer:** Building the `search_faq` tool and grounding replies (15 min)
- **08 · Single-Agent Sign-Off:** End-to-end verification before Level 3 multi-agent workflows (10 min)

---

## 1. Three Kinds of Memory

| Kind | What It Holds | Lifespan | Sweet Crumbs Example | Implementation |
| :--- | :--- | :--- | :--- | :--- |
| **Conversation memory** (short-term) | Messages within a single chat | One conversation | *"Ravi asked for 2 plum cakes a minute ago"* | Checkpointer + `thread_id` |
| **Customer memory** (long-term) | Persistent user traits/history | Weeks or months | *"Ravi always orders eggless"* | External store / sheet / DB |
| **Knowledge** | Business policies & documents | Until docs change | *"Delivery is free above Rs 1,000"* | ChromaDB + RAG tool |

---

## 2. Conversation Memory: Checkpointer and `thread_id`

Instead of manually passing full message lists every turn, attach a **checkpointer** to the agent and pass a `thread_id` in `config` so separate customer chats stay isolated.

```python
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model=llm,
    tools=[get_price, is_open, log_order],
    system_prompt=SYSTEM,
    checkpointer=InMemorySaver(),
)

def chat(thread_id: str, text: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"messages": [{"role": "user", "content": text}]}, config)
    return result["messages"][-1].content

chat("ravi", "Hi, I'm Ravi. I want 2 plum cakes.")
chat("sravani", "Hello! Do you make cupcakes?")
chat("ravi", "Make them eggless please.")  # Remembers Ravi's 2 plum cakes
```

> **Note:** `InMemorySaver` stores state in runtime memory, so threads reset when the Colab runtime restarts.

---

## 3. Why Not Put Everything in the Prompt?

With ~10 facts, putting everything in the system prompt works fine. With hundreds of policies, sending all of them on every call wastes input tokens, slows down responses, and buries the exact policy needed.

- **Token cost:** Agents can make multiple calls per customer turn; resending a large policy document every time multiplies input token costs.
- **Latency & focus:** Long prompts take longer to process and distract the model with irrelevant rules.
- **Better approach:** Retrieve only the top 2–3 relevant facts first and pass only those to the model.

---

## 4. Embeddings: Meaning as Numbers

An **embedding** converts text into a list of numbers (a vector) that captures semantic meaning rather than exact keywords.

- `"Do you deliver?"` and `"Can you send the cake to my house?"` share few literal words, yet map close together in vector space.
- Similar meaning produces nearby numbers across hundreds of dimensions, enabling semantic search across varied phrasing.

---

## 5. Vector Database: ChromaDB

**ChromaDB** stores text documents along with their embeddings and retrieves the closest matches in meaning to a query.

```python
import chromadb

client = chromadb.Client()  # In-memory database
faq = client.get_or_create_collection(name="sweet_crumbs_faq")

faq.add(
    ids=["delivery", "payment", "photo_cake"],
    documents=[
        "We deliver within 5 km of the bakery. Delivery is Rs 50, free for orders above Rs 1,000.",
        "We accept UPI and cash. Orders above Rs 1,000 need a 50% advance by UPI.",
        "Photo cakes and custom designs need 48 hours' notice.",
    ],
)

results = faq.query(query_texts=["Can you send the cake to my house?"], n_results=2)
print(results["documents"][0])
```

- `results["documents"][0]` returns the top matching texts for the first query.
- ChromaDB always returns the nearest `n_results` items, so the agent still judges whether the retrieved facts actually answer the question.

---

## 6. Chunking: Cutting Documents Into Focused Pieces

Long documents are split into smaller pieces (**chunks**) before indexing so each chunk holds roughly one idea or topic.

- **Too large:** Returning a huge block brings back mixed topics and wastes prompt space.
- **Too small:** Cutting mid-sentence can break meaning and strip context.
- **Overlap:** Repeating a small window of characters/tokens between neighboring chunks keeps boundary sentences intact.

```python
def chunk(text: str, size: int = 300, overlap: int = 50) -> list[str]:
    """Split text into pieces of about `size` characters that overlap by `overlap`."""
    pieces, start = [], 0
    while start < len(text):
        pieces.append(text[start:start + size])
        start += size - overlap
    return pieces
```

---

## 7. RAG: Retrieve, Then Answer

**RAG (Retrieval-Augmented Generation)** grounds responses in retrieved documents instead of guessing:

1. **Retrieve:** Query the vector database with the customer's question.
2. **Augment:** Add the top retrieved facts as context for the model.
3. **Generate:** Answer strictly from the retrieved facts—or state *"I'll check with the owner"* if the answer is not in the documents.

```python
@tool
def search_faq(question: str) -> str:
    """Search Sweet Crumbs' policies and FAQs: delivery, payment, cancellation,
    photo cakes, allergens, storage, candles and messages on cakes.
    Use for any question about rules or policies, not for prices or opening hours."""
    results = faq.query(query_texts=[question], n_results=3)
    return "\n".join(results["documents"][0])
```

---

## 8. Single-Agent Sign-Off

Before moving to **Level 3 (Multi-Agent Workflows)**, verify that your complete single agent handles every message type cleanly:

| Part | What You Built | Day |
| :--- | :--- | :--- |
| **Brain** | `ChatGroq`, low temperature | 3 |
| **Instructions** | System prompt (`ROLE`, `RULES`, `FORMAT`) with fallback when unknown | 2–3 |
| **Tools** | `get_price`, `is_open`, `log_order`, `search_faq` | 4–5 |
| **Memory** | Checkpointer (`InMemorySaver`) + `thread_id` per customer | 5 |
| **Knowledge** | ChromaDB collection of FAQs and policies (RAG) | 5 |
| **Checks** | Pydantic validation inside `log_order` + `tests.csv` test suite | 3–4 |

---

## Today's Build & Your 30%

### Build Cells (Add After Day 4 Notebook)

```python
# Cell 1 · Install and load the bakery's knowledge
# !pip install -q chromadb
import chromadb

FAQS = {
    "delivery": "We deliver within 5 km of the bakery. Delivery is Rs 50, free for orders above Rs 1,000.",
    "payment": "We accept UPI and cash. Orders above Rs 1,000 need a 50% advance by UPI.",
    "cancellation": "Cancellation is free up to 12 hours before pickup. After that, the advance is not refunded.",
    "photo_cake": "Photo cakes and custom designs need 48 hours' notice.",
    "notice": "Regular orders need 24 hours' notice.",
    "eggless": "All our cakes have eggless versions at the same price.",
    "allergens": "Plum cake contains nuts and dried fruit. Cupcakes and chocolate cake are nut-free.",
    "storage": "Keep cakes in the fridge and eat them within 3 days.",
    "message": "A message on the cake is free, up to 20 letters.",
    "candles": "Candles are free with every cake.",
    "discounts": "Bulk-order discounts need the owner's approval.",
    "pickup": "Pickup is during opening hours: Monday to Saturday 9 am to 8 pm, Sunday 10 am to 6 pm.",
}

client = chromadb.Client()
faq = client.get_or_create_collection(name="sweet_crumbs_faq")
faq.upsert(ids=list(FAQS.keys()), documents=list(FAQS.values()))
print(faq.count(), "facts stored")
print(faq.query(query_texts=["Can you send the cake to my house?"], n_results=2)["documents"][0])
```

```python
# Cell 2 · The search_faq tool
@tool
def search_faq(question: str) -> str:
    """Search Sweet Crumbs' policies and FAQs: delivery, payment, cancellation,
    photo cakes, allergens, storage, candles and messages on cakes.
    Use for any question about rules or policies, not for prices or opening hours."""
    results = faq.query(query_texts=[question], n_results=3)
    return "\n".join(results["documents"][0])
```

```python
# Cell 3 · The agent with memory
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

SYSTEM = """ROLE:   You are the friendly assistant for Sweet Crumbs Home Bakery.
RULES:  Use get_price for prices and is_open for timings; never guess them.
        For any policy question, use search_faq and answer ONLY from what it returns.
        If the answer isn't there, say "I'll check with the owner."
        Before log_order, make sure you have name, item, quantity and pickup date; ask if missing.
FORMAT: Under 60 words, friendly, English or Telugu matching the customer."""

agent = create_agent(
    model=llm,
    tools=[get_price, is_open, log_order, search_faq],
    system_prompt=SYSTEM,
    checkpointer=InMemorySaver(),
)

def chat(thread_id: str, text: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": text}]}, config)
        return result["messages"][-1].content
    except Exception as e:
        print("Agent error:", e)
        return "Sorry, I'm a little busy. Please try again in a minute."

print(chat("ravi", "Hi, I'm Ravi. Do you deliver? My order is about Rs 1,300."))
print(chat("ravi", "Does the plum cake have nuts?"))
print(chat("sravani", "Can I cancel tomorrow morning if I pick up tomorrow evening?"))
print(chat("ravi", "What's my name, and will I pay for delivery?"))
```

### Your 30% Checklist
1. **Your knowledge:** Load 15–20 facts for your business into a ChromaDB collection (one fact per chunk).
2. **Your search tool:** Write a `search_*` tool with a docstring listing covered policy topics.
3. **Your agent:** Attach a checkpointer and test with at least two `thread_id` values.
4. **Break it:** Ask out-of-scope questions missing from your documents and verify the agent does not hallucinate.
5. **Sign-off:** Run your test suite (`tests.csv`), record your pass rate (aim for 80%+), and sketch your customer request flow before Day 6.

---

## Summary Table

| Idea | One Line | Analogy |
| :--- | :--- | :--- |
| **Three kinds of memory** | Conversation, customer, knowledge | Notepad, register, policy file |
| **Threads** | Checkpointer + one `thread_id` per conversation | Email threads |
| **Not everything in the prompt** | Search first, send only what's relevant | An index, not the whole library |
| **Embeddings** | Text to numbers; similar meaning, nearby numbers | Supermarket shelves |
| **Vector database** | Store texts, find closest by meaning | A librarian who knows every book |
| **Chunking** | One idea per chunk, small overlap | Neat cake slices |
| **RAG** | Retrieve -> augment -> generate; "not found" is allowed | Open-book exam, no bluffing |
| **Sign-off** | Brain, instructions, tools, memory, knowledge, tests | Employee after probation |
