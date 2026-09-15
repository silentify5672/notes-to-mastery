# 📚 Notes to Mastery

An AI study companion that takes you from raw notes to genuine understanding — not just memorization. Built for the Nebius x NVIDIA Global AI Hackathon.

## What it does

1. **Upload notes** — paste text or upload a PDF
2. **Get flashcards** — Nemotron extracts key concepts, scaling the number of flashcards to match how much material you gave it
3. **Take a quiz** — real AI grading of your answers, with "I don't know" and "quit" options built in
4. **Teach-back mode** — for topics you got wrong, the AI plays a confused classmate and asks you to explain it back, testing whether you *actually* understand it (not just recognize it)

## Tech stack

- **Nebius Token Factory** — hosts the model and handles inference
- **NVIDIA Nemotron 3 Nano** (`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`) — powers flashcard generation, quiz grading, and the teach-back conversation
- **Streamlit** — web interface
- **Python** (`requests`, `pypdf`, `python-dotenv`)

## Setup instructions

1. Clone this repo: