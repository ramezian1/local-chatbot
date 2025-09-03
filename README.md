# Local Chatbot (Bobo)

A fully local chatbot created by **Robert Mezian** that supports **multi-file Q&A** with TF-IDF search, persistent personal memory (SQLite + FTS5), task tracking, and colorful console output.  
No internet or external dependencies required. Everything runs locally.

---

## ✨ Features

### Document Q&A
- `load <file>` – load `.txt`, `.md`, `.log` (searches CWD, script folder, or `./docs`)
- `load folder <path>` – index all `.txt/.md/.log` in a folder
- `ask <question>` – cosine similarity over TF-IDF vectors
- `list docs` – list loaded files
- `clear docs` – clear all indexed docs
- **Colored output:** results styled by relevance (green = strong, yellow = medium, gray = weak).  
  Disable with `--no-color`.

### Persistent Personal Memory (SQLite)
- `remember X is Y` – save facts locally
- `what is X?` – recall saved facts
- `/remember X: Y` – slash alias to save
- `/recall X` – slash alias to recall
- `/memkeys` – list all stored keys
- `/memsearch <text>` – fuzzy search facts with FTS5
- `/forget <key>` – delete a saved memory

### To-Do Manager
- `add <task>` – add to-do  
- `list todos` – view tasks  
- `done <n>` – mark done  
- `clear todos` – reset list  

### Utilities
- `time` / `date` – local time & date  
- `echo <text>` – repeat input  
- `joke` – random programming joke  
- `help` – show all commands  

---

## 🚀 Example Usage

```bash
$ python chatbot.py --top-k 5
Bobo ready. Type 'help' to begin. Ctrl+C to quit.

> remember my name is Bobby
Got it. I’ll remember 'my name' = 'Bobby'.

> /memkeys
🧠 Keys (newest first):
- my name

> ask what is tf-idf?
Top matches for: what is tf-idf
- [notes.txt] (0.882) TF-IDF is a numerical statistic that reflects how important a word is...
```

---

## 📂 Project Structure

```
chatbot.py        # main script
memory.py         # SQLite memory store
docs/             # default document folder
data/             # persistent memory (memory.db, todos.json)
chat_logs/        # daily chat transcripts
```

---

## 🛠 Requirements
- Python 3.8+
- Pure standard library (no external pip installs)

---

## Author
Developed by **Robert Mezian**

---

## 📜 License
See [LICENSE.md](LICENSE.md) for details.
