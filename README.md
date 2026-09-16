# 📚 Notes to Mastery

An AI study companion that takes you from raw notes to genuine understanding — not just memorization. Built for the Nebius x NVIDIA Global AI Hackathon.

**Live app:** [notes-to-mastery.streamlit.app](https://notes-to-mastery.streamlit.app)

## What it does

1. **Upload notes** — paste text or upload up to 10 PDFs at once
2. **Get flashcards** — Nemotron extracts key concepts and automatically organizes them into topics, scaling the number of flashcards to match how much material you provide
3. **Browse by topic** — flashcards are grouped into clickable, collapsible topic folders so you can jump straight to what you want to review
4. **Choose your quiz style** — pick a specific topic (or retry all your weak topics at once) and take either an open-ended or multiple-choice quiz, with questions freshly reworded by the AI rather than copy-pasted from the flashcards
5. **Lenient, meaning-based grading** — open-ended answers are graded on whether they capture the right idea, not exact wording, with built-in "I don't know" and "stop quiz" options
6. **Teach-back mode** — for topics you got wrong, the AI plays a confused classmate and asks you to explain it back, testing whether you *actually* understand it. Follow-up questions get harder if you've struggled with a topic across multiple attempts
7. **Progress dashboard** — a sidebar tracks your overall accuracy and which topics you've mastered vs. still need work on, for the current session
8. **Class Mode** — generate a class code for a flashcard set so others can load the exact same set and study alongside you
9. **Save / load progress** — download your flashcards and progress as a JSON file, and reload it later to pick up where you left off

## Tech stack

- **Nebius Token Factory** — hosts the model and handles inference
- **NVIDIA Nemotron 3 Nano** (`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`) — powers flashcard generation, quiz question rewriting, answer grading, and the teach-back conversation
- **Streamlit** — web interface and hosting (Streamlit Community Cloud)
- **Python** (`requests`, `pypdf`, `python-dotenv`)

## Setup instructions

1. Clone this repo:
2. Install dependencies:
3. Create a `.env` file in the project root with your own Nebius Token Factory API key:
4. Run the app:
5. Open the local URL shown in your terminal (usually `http://localhost:8501`).

## Files

- `app.py` — the full application (this is what's deployed and what judges should run/visit). Includes multi-PDF upload, topic-grouped flashcards, choice of open-ended or multiple-choice quizzes, lenient AI grading, a weak-topics retry mode, an adaptive teach-back mode, a progress dashboard, Class Mode sharing, and save/load progress support.
- `main.py` — an early terminal-only prototype built first thing, before the Streamlit interface existed. Kept for reference to show the project's development process; it has the original core logic (flashcards → quiz → teach-back) but does not include any of the later features listed above.
- `requirements.txt` — Python dependencies for deployment

## Notes on Class Mode and progress tracking

Class Mode share codes and the progress dashboard are stored in the app's live memory rather than a database, so they persist as long as the hosted app stays running but reset if it restarts. The save/load progress feature (JSON export) is the recommended way to keep your data long-term across sessions.

## Team

[THE AI DUO/silentify and gitthegreat]