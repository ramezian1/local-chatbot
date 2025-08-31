# Local Chatbot with Multi‑File Q&A (TF‑IDF)

This is a simple, local Python chatbot that can remember facts, manage to‑dos, tell jokes, and most importantly, **load multiple local text files and answer questions** from them using TF‑IDF + cosine similarity search.

## ✨ Features

- **Chatbot basics**:
  - Remember facts: `remember my name is Bobby`
  - Recall facts: `what is my name?`
  - To‑dos: `add buy milk`, `list todos`, `done 1`, `clear todos`
  - Jokes, time, date, echo, polite small talk

- **Q&A over local files**:
  - Load one file: `load <file>` (looks in current folder, script folder, then `./docs`)
  - Load a folder: `load folder <path>` (indexes all `.txt`, `.md`, `.log`)
  - List docs: `list docs`
  - Clear docs: `clear docs`
  - Ask questions: `ask <your question>` → returns top 3 relevant chunks with cosine scores

- **Docs folder**: Put files into the `docs/` directory and load them easily.

- **Logs & memory**:
  - Conversations saved in `./chat_logs`
  - Facts and todos persisted in `./data`

All data stays fully local — no internet or external services required.

## 🚀 Usage

### 1. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
# or
source venv/bin/activate # macOS/Linux
```

### 2. Run the chatbot
```bash
python chatbot.py
```

### 3. Example session
```
> remember favorite drummer is Danny Carey
Got it. I’ll remember 'favorite drummer' = 'Danny Carey'.

> load folder docs
Indexed 4 files, 12 chunks. Ask with: ask <question>

> ask what is linux
Top matches for: what is linux
- [linux.txt] (0.921) Linux is a family of open-source operating systems based on the Linux kernel.

> list docs
Docs:
   1. python.txt       (2 chunks)
   2. javascript.txt   (2 chunks)
   3. linux.txt        (2 chunks)
   4. networking.txt   (3 chunks)
```

### 4. Exit
Type `bye`, `exit`, or `quit`.

## 📂 Project Structure
```
chatbot.py       # main program
data/            # stores facts and todos
chat_logs/       # daily logs of chats
docs/            # place your text files here
venv/            # virtual environment (optional)
README.md        # this file
```

## 🛠 Requirements
- Python 3.8+
- No external libraries (pure standard library)

## 🔮 Future ideas
- Highlight matched words in answers
- Short synthesized answers above snippets
- Packaged `.exe` for Windows or `.app` for macOS
- Mini Tkinter GUI

---
Made by Robert Mezian for local tinkering and learning.
