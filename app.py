import streamlit as st
import os
import requests
from dotenv import load_dotenv
import json
import random
import string
import threading
import time
import re
import copy
import uuid
from datetime import datetime, timedelta
from pypdf import PdfReader

load_dotenv()

API_KEY = os.environ.get("NEBIUS_API_KEY")
API_URL = "https://api.tokenfactory.nebius.com/v1/chat/completions"
MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"

REQUEST_TIMEOUT = 30       # seconds per API call
MAX_RETRIES = 2            # extra attempts after the first, on timeouts/5xx/429
CLASS_CODE_EXPIRY_HOURS = 48  # class codes stop working after this long


# ---------- Custom errors ----------

class NemotronError(Exception):
    """Raised when the Nebius/Nemotron API call fails after retries."""
    pass


class FlashcardParseError(Exception):
    """Raised when the AI's response can't be parsed into the expected JSON shape."""
    pass


# ---------- Core API helpers ----------

def call_nemotron(system_prompt, user_prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    last_error = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise NemotronError(
                "Couldn't reach the Nebius Token Factory API (timed out or unreachable). Please try again."
            ) from e

        # Rate-limited or transient server error: worth a retry
        if response.status_code == 429 or response.status_code >= 500:
            last_error = f"HTTP {response.status_code}"
            if attempt < MAX_RETRIES:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise NemotronError(
                f"Nebius API returned an error ({response.status_code}) after retrying. Please try again shortly."
            )

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise NemotronError(
                f"Nebius API request failed ({response.status_code}). Check your API key and try again."
            ) from e

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise NemotronError("Nebius API returned an unexpected response format.") from e

    raise NemotronError(f"Couldn't reach Nemotron after {MAX_RETRIES + 1} attempts: {last_error}")


def safe_json_parse(raw_response):
    """
    Strip markdown code fences (models add these despite instructions not to) and parse JSON.
    Falls back to grabbing the first [...] or {...} block if the response has stray text around it.
    Raises FlashcardParseError with a friendly message if nothing works.
    """
    cleaned = raw_response.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"(\[.*\]|\{.*\})", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        raise FlashcardParseError(
            "The AI returned a response that wasn't valid JSON. This sometimes happens — try again."
        )


def run_with_progress(worker_fn, label="Working..."):
    """
    Runs worker_fn() in a background thread while showing a simulated
    progress bar (0-90% while waiting, snaps to 100% when actually done).
    worker_fn takes no arguments and returns whatever result you want back.
    Re-raises any exception worker_fn raised, in the main thread.
    """
    result = {"data": None, "error": None}

    def worker():
        try:
            result["data"] = worker_fn()
        except Exception as e:
            result["error"] = e

    thread = threading.Thread(target=worker)
    thread.start()

    progress_bar = st.progress(0)
    status_text = st.empty()
    progress = 0

    while thread.is_alive():
        progress = min(progress + 3, 90)
        progress_bar.progress(progress / 100)
        status_text.write(f"{label} {progress}% (estimated)")
        time.sleep(0.3)

    thread.join()
    progress_bar.progress(100)
    status_text.write(f"{label} 100%")
    time.sleep(0.3)
    progress_bar.empty()
    status_text.empty()

    if result["error"] is not None:
        raise result["error"]
    return result["data"]


# ---------- Flashcard generation ----------

def generate_flashcards(notes_text):
    word_count = len(notes_text.split())
    num_cards = max(3, min(word_count // 100, 40))

    system_prompt = (
        f"You are a study assistant. Extract key concepts from the given notes and turn them into "
        f"approximately {num_cards} flashcards, covering the material thoroughly without excessive repetition. "
        f"Group related flashcards under the SAME topic name — use consistent, clear topic names "
        f"(around 3-10 distinct topics total, depending on how much material there is), so flashcards on the "
        f"same concept always share the exact same topic string. "
        f"Respond ONLY with a JSON array like this: "
        f"[{{\"question\": \"...\", \"answer\": \"...\", \"topic\": \"...\"}}]. No extra text, just the JSON."
    )
    raw_response = call_nemotron(system_prompt, notes_text)
    flashcards = safe_json_parse(raw_response)

    if not isinstance(flashcards, list) or len(flashcards) == 0:
        raise FlashcardParseError("The AI didn't return any flashcards. Try again, or add more notes.")

    for card in flashcards:
        card["_id"] = str(uuid.uuid4())

    return flashcards


def regenerate_flashcard(card, source_notes):
    """Ask the AI for a fresh, differently-angled flashcard on the same topic."""
    if source_notes:
        context = f"Original source notes (for context):\n{source_notes}\n\n"
        user_prompt = context
    else:
        user_prompt = "No source notes are available; use your best judgment based on the topic and existing card."

    system_prompt = (
        f"You are a study assistant. The student wants a fresh version of ONE flashcard on the topic "
        f"'{card['topic']}'. Their current flashcard is:\n"
        f"Q: {card['question']}\nA: {card['answer']}\n\n"
        "Write a DIFFERENT flashcard covering the same topic (a different angle, question style, or "
        "level of detail), still grounded in the source notes if provided. "
        "Respond ONLY with a JSON object like this: {\"question\": \"...\", \"answer\": \"...\"}. No extra text."
    )

    raw_response = call_nemotron(system_prompt, user_prompt)
    result = safe_json_parse(raw_response)

    if not isinstance(result, dict) or "question" not in result or "answer" not in result:
        raise FlashcardParseError("The AI didn't return a valid replacement flashcard. Try again.")

    return result


def extract_text_from_pdf(uploaded_file):
    try:
        reader = PdfReader(uploaded_file)
    except Exception:
        return ""

    text = ""
    for page in reader.pages:
        try:
            page_text = page.extract_text()
        except Exception:
            page_text = None
        if page_text:
            text += page_text + "\n"
    return text


def read_all_pdfs(pdf_files):
    """Reads every uploaded PDF, returning combined text and a list of filenames that yielded nothing."""
    combined = ""
    empty_files = []
    for pdf_file in pdf_files:
        pdf_text = extract_text_from_pdf(pdf_file)
        if pdf_text.strip() == "":
            empty_files.append(pdf_file.name)
        else:
            combined += "\n" + pdf_text
    return combined, empty_files


def group_by_topic(flashcards):
    grouped = {}
    for card in flashcards:
        topic = card["topic"]
        grouped.setdefault(topic, []).append(card)
    return grouped


def get_difficulty_hint(topics, history):
    """Builds a prompt fragment nudging question difficulty based on past attempts on these topics."""
    max_attempts = max((history.get(t, {}).get("attempts", 0) for t in topics), default=0)
    if max_attempts >= 3:
        return (
            " These topics have been attempted several times before, so make the questions noticeably "
            "harder and more specific/nuanced than a first-pass question."
        )
    elif max_attempts >= 1:
        return " The student has seen this material before, so make questions moderately challenging rather than very basic."
    else:
        return " This is the student's first attempt at this material, so keep questions at a clear, approachable level."


def generate_quiz_questions(cards, quiz_type, difficulty_hint=""):
    cards_for_prompt = [{"question": c["question"], "answer": c["answer"]} for c in cards]

    if quiz_type == "mcq":
        system_prompt = (
            "You are a quiz writer. For each flashcard given (question and answer), create a multiple-choice "
            "quiz question: reword the question using different wording than the original while keeping the "
            "exact same meaning. Provide exactly 4 answer options where exactly one is correct (matching the "
            "meaning of the original answer) and the other 3 are plausible but clearly incorrect distractors. "
            "Respond ONLY with a JSON array, one entry per flashcard, in the SAME order as given, in this exact "
            "format: [{\"question\": \"...\", \"options\": [\"...\", \"...\", \"...\", \"...\"], "
            "\"correct_option\": \"...\"}]. The correct_option value must exactly match one of the 4 strings "
            "in options. No extra text, just the JSON." + difficulty_hint
        )
    else:
        system_prompt = (
            "You are a quiz writer. For each flashcard given (question and answer), reword ONLY the question "
            "using different wording than the original while keeping the exact same meaning and the same "
            "correct answer. Respond ONLY with a JSON array, one entry per flashcard, in the SAME order as "
            "given, in this exact format: [{\"question\": \"...\", \"answer\": \"...\"}]. No extra text." + difficulty_hint
        )

    raw_response = call_nemotron(system_prompt, json.dumps(cards_for_prompt))
    quiz_questions = safe_json_parse(raw_response)

    if not isinstance(quiz_questions, list) or len(quiz_questions) == 0:
        raise FlashcardParseError("The AI didn't return quiz questions in the expected format. Try again.")

    safe_length = min(len(quiz_questions), len(cards))
    for i in range(safe_length):
        quiz_questions[i]["topic"] = cards[i]["topic"]
    return quiz_questions[:safe_length]


# ---------- Spaced repetition helpers ----------

def review_interval_days(attempts, mastered):
    """How many days to wait before a mastered topic comes up for review again. Grows with attempts, caps at 30."""
    if not mastered:
        return 0  # unmastered topics are always "due"
    return min(2 ** min(attempts, 5), 30)


def is_due_for_review(topic_data):
    last_reviewed = topic_data.get("last_reviewed")
    if not last_reviewed:
        return True
    interval = review_interval_days(topic_data.get("attempts", 0), topic_data.get("mastered", False))
    due_date = datetime.fromisoformat(last_reviewed) + timedelta(days=interval)
    return datetime.now() >= due_date


# ---------- Misc state helpers ----------

def reset_quiz_state():
    keys_to_clear = [
        "quiz_topic", "quiz_type", "quiz_questions", "quiz_index", "score",
        "weak_topics", "quiz_done", "round_topic_results", "round_folded_in",
        "tb_topic_index", "tb_stage", "tb_explanation", "tb_followup_question", "tb_verdict", "tb_done"
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def ensure_history_initialized():
    if "history" not in st.session_state:
        st.session_state["history"] = {}
    if "overall_score" not in st.session_state:
        st.session_state["overall_score"] = {"correct": 0, "incorrect": 0, "idk": 0}
    if "active_class_code" not in st.session_state:
        st.session_state["active_class_code"] = None


@st.cache_resource
def get_class_store():
    return {}


def make_class_code():
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=5))


# ---------- Page setup ----------

st.title("📚 Notes to Mastery")

if not API_KEY:
    st.error(
        "⚠️ No Nebius API key found. Set the `NEBIUS_API_KEY` environment variable "
        "(in a `.env` file locally, or in your deployment's secrets) and restart the app."
    )
    st.stop()

ensure_history_initialized()

# ---------- Sidebar: Class Mode + Progress Dashboard ----------

with st.sidebar:
    st.header("🏫 Class Mode")
    st.caption("Join a set someone shared with you, or create your own below after generating flashcards.")
    join_code = st.text_input("Enter a class code:", key="join_code_input").strip().upper()
    if st.button("Join"):
        store = get_class_store()
        if join_code in store:
            created_at = store[join_code].get("created_at")
            expired = (
                created_at is not None
                and datetime.now() - datetime.fromisoformat(created_at) > timedelta(hours=CLASS_CODE_EXPIRY_HOURS)
            )
            if expired:
                del store[join_code]
                st.error("That class code has expired. Ask whoever shared it to create a new one.")
            else:
                st.session_state["flashcards"] = copy.deepcopy(store[join_code]["flashcards"])
                st.session_state["source_notes"] = ""
                st.session_state["active_class_code"] = join_code
                reset_quiz_state()
                st.success(f"Joined class code {join_code}! Scroll down to see the flashcards.")
        else:
            st.error("That code wasn't found. Double-check it with whoever shared it.")

    active_code = st.session_state.get("active_class_code")
    if active_code:
        store = get_class_store()
        class_data = store.get(active_code)
        if class_data:
            results = class_data.get("quiz_results", [])
            st.write(f"**Active class code: {active_code}**")
            if results:
                avg = round(sum(r["score_pct"] for r in results) / len(results))
                st.write(f"{len(results)} quiz attempt{'s' if len(results) != 1 else ''} so far — average score {avg}%")
            else:
                st.caption("No quiz attempts recorded yet for this set.")

    st.divider()
    st.header("📊 Your Progress Dashboard")
    history = st.session_state["history"]
    overall = st.session_state["overall_score"]
    total_answered = overall["correct"] + overall["incorrect"] + overall["idk"]

    if total_answered == 0:
        st.caption("Complete a quiz to start tracking your progress here.")
    else:
        overall_pct = round((overall["correct"] / total_answered) * 100)
        st.metric("Overall accuracy", f"{overall_pct}%", f"{overall['correct']}/{total_answered} answered")

        mastered_count = sum(1 for t in history.values() if t["mastered"])
        st.write(f"**{mastered_count} of {len(history)}** topics mastered")

        for topic, data in history.items():
            status = "✅ Mastered" if data["mastered"] else "🔁 Needs review"
            due_tag = " · 📅 due for review" if is_due_for_review(data) else ""
            st.write(f"- {topic}: {status} ({data['attempts']} attempt{'s' if data['attempts'] != 1 else ''}){due_tag}")

    st.divider()
    st.header("💾 Save / Load Progress")
    if "flashcards" in st.session_state:
        export_data = {
            "flashcards": st.session_state["flashcards"],
            "history": st.session_state["history"],
            "overall_score": st.session_state["overall_score"],
        }
        st.download_button(
            "Download progress (.json)",
            data=json.dumps(export_data, indent=2),
            file_name="notes_to_mastery_progress.json",
            mime="application/json"
        )

    uploaded_progress = st.file_uploader("Load a saved progress file:", type=["json"], key="progress_uploader")
    if uploaded_progress is not None and st.button("Load this file"):
        try:
            loaded = json.load(uploaded_progress)
        except json.JSONDecodeError:
            st.error("That doesn't look like a valid progress file.")
        else:
            loaded_flashcards = loaded.get("flashcards", [])
            for card in loaded_flashcards:
                if "_id" not in card:
                    card["_id"] = str(uuid.uuid4())
            st.session_state["flashcards"] = loaded_flashcards
            st.session_state["history"] = loaded.get("history", {})
            st.session_state["overall_score"] = loaded.get("overall_score", {"correct": 0, "incorrect": 0, "idk": 0})
            st.session_state["source_notes"] = ""
            st.session_state["active_class_code"] = None
            reset_quiz_state()
            st.success("Progress loaded!")
            st.rerun()

    st.divider()
    st.header("📤 Export to Anki")
    if "flashcards" in st.session_state and st.session_state["flashcards"]:
        anki_lines = []
        for card in st.session_state["flashcards"]:
            front = card["question"].replace("\t", " ").replace("\n", "<br>")
            back = card["answer"].replace("\t", " ").replace("\n", "<br>")
            anki_lines.append(f"{front}\t{back}")
        anki_text = "\n".join(anki_lines)
        st.download_button(
            "Download for Anki (.txt)",
            data=anki_text,
            file_name="notes_to_mastery_anki_import.txt",
            mime="text/plain"
        )
        st.caption("In Anki: File → Import, pick this file, set the field separator to Tab, and map columns to Front/Back.")


# ---------- Notes / PDF input ----------

st.write("Paste your notes below, or upload up to 10 PDFs, and I'll turn them into flashcards.")

notes_input = st.text_area("Your notes:", height=150)
uploaded_pdfs = st.file_uploader("Or upload up to 10 PDFs:", type=["pdf"], accept_multiple_files=True)

if uploaded_pdfs and len(uploaded_pdfs) > 10:
    st.warning("You can upload up to 10 PDFs at a time — only the first 10 will be used.")
    uploaded_pdfs = uploaded_pdfs[:10]

if st.button("Generate Flashcards"):
    combined_notes = notes_input.strip()
    empty_pdf_names = []

    if uploaded_pdfs:
        pdf_text, empty_pdf_names = run_with_progress(
            lambda: read_all_pdfs(uploaded_pdfs),
            label="Reading PDFs..."
        )
        combined_notes = (combined_notes + "\n" + pdf_text).strip()

    if empty_pdf_names:
        st.warning(
            f"No extractable text found in: {', '.join(empty_pdf_names)}. "
            "These might be scanned/image-only PDFs — try a text-based PDF or OCR them first."
        )

    if combined_notes == "":
        st.warning("Please paste some notes or upload at least one PDF with readable text!")
    else:
        try:
            flashcards = run_with_progress(
                lambda: generate_flashcards(combined_notes),
                label="Generating flashcards..."
            )
        except FlashcardParseError as e:
            st.error(f"⚠️ {e}")
        except NemotronError as e:
            st.error(f"⚠️ {e}")
        else:
            st.session_state["flashcards"] = flashcards
            st.session_state["source_notes"] = combined_notes
            st.session_state["active_class_code"] = None
            reset_quiz_state()
            st.success(f"Generated {len(flashcards)} flashcards!")

# ---------- Flashcard display, grouped by topic + Class Mode create ----------

if "flashcards" in st.session_state:
    flashcards = st.session_state["flashcards"]
    topics_grouped = group_by_topic(flashcards)

    st.subheader("Your Flashcards")
    st.write(f"{len(flashcards)} flashcards across {len(topics_grouped)} topics.")

    for topic, cards in topics_grouped.items():
        with st.expander(f"📁 {topic} ({len(cards)} card{'s' if len(cards) != 1 else ''})"):
            for card in cards:
                card_id = card["_id"]
                editing = st.session_state.get(f"editing_{card_id}", False)

                if editing:
                    new_q = st.text_area("Question:", value=card["question"], key=f"edit_q_{card_id}")
                    new_a = st.text_area("Answer:", value=card["answer"], key=f"edit_a_{card_id}")
                    col1, col2 = st.columns(2)
                    if col1.button("💾 Save", key=f"save_{card_id}"):
                        card["question"] = new_q
                        card["answer"] = new_a
                        st.session_state[f"editing_{card_id}"] = False
                        st.rerun()
                    if col2.button("Cancel", key=f"cancel_{card_id}"):
                        st.session_state[f"editing_{card_id}"] = False
                        st.rerun()
                else:
                    st.write(f"**Q:** {card['question']}")
                    st.write(f"**A:** {card['answer']}")
                    col1, col2, col3 = st.columns(3)
                    if col1.button("✏️ Edit", key=f"edit_btn_{card_id}"):
                        st.session_state[f"editing_{card_id}"] = True
                        st.rerun()
                    if col2.button("🔄 Regenerate", key=f"regen_{card_id}"):
                        try:
                            new_card_data = run_with_progress(
                                lambda: regenerate_flashcard(card, st.session_state.get("source_notes", "")),
                                label="Regenerating flashcard..."
                            )
                        except (FlashcardParseError, NemotronError) as e:
                            st.error(f"⚠️ {e}")
                        else:
                            card["question"] = new_card_data["question"]
                            card["answer"] = new_card_data["answer"]
                            st.rerun()
                    if col3.button("🗑️ Delete", key=f"delete_{card_id}"):
                        st.session_state["flashcards"] = [c for c in st.session_state["flashcards"] if c["_id"] != card_id]
                        st.rerun()

                st.divider()

    st.write("**Sharing this set with a class or study group?**")
    if st.button("Create a class code for this set"):
        store = get_class_store()
        code = make_class_code()
        store[code] = {
            "flashcards": copy.deepcopy(flashcards),
            "created_at": datetime.now().isoformat(),
            "quiz_results": []
        }
        st.session_state["active_class_code"] = code
        st.success(f"Share this code with others: **{code}** (they enter it in the sidebar 'Class Mode' box)")
        st.caption(f"Codes expire after {CLASS_CODE_EXPIRY_HOURS} hours.")


# ---------- Quiz section ----------

if "flashcards" in st.session_state:
    st.divider()
    st.subheader("Quiz Time")

    flashcards = st.session_state["flashcards"]
    topics_grouped = group_by_topic(flashcards)
    topic_names = list(topics_grouped.keys())

    idk_phrases = ["idk", "i don't know", "i dont know", "not sure", "no idea", ""]

    weak_pool_topics = [t for t, d in st.session_state["history"].items() if not d["mastered"] and d["attempts"] > 0]
    due_topics = [t for t, d in st.session_state["history"].items() if is_due_for_review(d)]
    review_due_topics = [t for t in due_topics if t not in weak_pool_topics]

    if "quiz_topic" not in st.session_state:
        st.write("What would you like to practice?")

        mode_options = ["Choose a specific topic"]
        if review_due_topics:
            mode_options.insert(0, f"Review due topics ({len(review_due_topics)})")
        if weak_pool_topics:
            mode_options.insert(0, f"Retry all weak topics ({len(weak_pool_topics)})")

        mode_choice = st.radio("Practice mode:", mode_options, key="practice_mode")

        if mode_choice.startswith("Retry all weak topics"):
            cards_for_quiz = [c for c in flashcards if c["topic"] in weak_pool_topics]
            chosen_label = "Weak Topics Review"
        elif mode_choice.startswith("Review due topics"):
            cards_for_quiz = [c for c in flashcards if c["topic"] in review_due_topics]
            chosen_label = "Spaced Repetition Review"
        else:
            selected_topic = st.selectbox("Topic:", topic_names, key="topic_selector")
            cards_for_quiz = topics_grouped[selected_topic]
            chosen_label = selected_topic

        quiz_type_label = st.radio("Quiz type:", options=["Open-ended", "Multiple Choice"], key="quiz_type_selector")
        quiz_type = "mcq" if quiz_type_label == "Multiple Choice" else "open"

        if st.button("Start Quiz"):
            topics_in_quiz = list({c["topic"] for c in cards_for_quiz})
            difficulty_hint = get_difficulty_hint(topics_in_quiz, st.session_state["history"])

            try:
                quiz_questions = run_with_progress(
                    lambda: generate_quiz_questions(cards_for_quiz, quiz_type, difficulty_hint),
                    label="Preparing your quiz..."
                )
            except (FlashcardParseError, NemotronError) as e:
                st.error(f"⚠️ {e}")
            else:
                st.session_state["quiz_topic"] = chosen_label
                st.session_state["quiz_type"] = quiz_type
                st.session_state["quiz_questions"] = quiz_questions
                st.session_state["quiz_index"] = 0
                st.session_state["score"] = {"correct": 0, "incorrect": 0, "idk": 0}
                st.session_state["weak_topics"] = {}
                st.session_state["round_topic_results"] = {}
                st.session_state["quiz_done"] = False
                st.rerun()

    else:
        quiz_questions = st.session_state["quiz_questions"]
        index = st.session_state["quiz_index"]
        quiz_label = st.session_state["quiz_topic"]
        quiz_type = st.session_state["quiz_type"]

        def record_result(topic, outcome):
            st.session_state["score"][outcome] += 1
            rtr = st.session_state["round_topic_results"]
            rtr.setdefault(topic, {"correct": 0, "wrong": 0})
            if outcome == "correct":
                rtr[topic]["correct"] += 1
            else:
                rtr[topic]["wrong"] += 1
                st.session_state["weak_topics"][topic] = st.session_state["weak_topics"].get(topic, 0) + 1

        if not st.session_state["quiz_done"] and index < len(quiz_questions):
            st.progress(index / len(quiz_questions))
            question_data = quiz_questions[index]
            st.write(f"**{quiz_label}** — Question {index + 1} of {len(quiz_questions)}")
            st.write(question_data["question"])

            if quiz_type == "mcq":
                options = question_data["options"] + ["I don't know"]
                choice = st.radio("Choose an answer:", options, key=f"mcq_{index}", index=None)

                col1, col2 = st.columns(2)
                if col1.button("Submit Answer", key=f"submit_{index}", disabled=(choice is None)):
                    topic = question_data["topic"]
                    if choice == question_data["correct_option"]:
                        record_result(topic, "correct")
                    elif choice == "I don't know":
                        record_result(topic, "idk")
                    else:
                        record_result(topic, "incorrect")
                    st.session_state["quiz_index"] += 1
                    st.rerun()

                if col2.button("Stop Quiz", key=f"stop_{index}"):
                    st.session_state["quiz_done"] = True
                    st.rerun()

            else:
                user_answer = st.text_input("Your answer:", key=f"answer_{index}")
                col1, col2, col3 = st.columns(3)

                if col1.button("Submit Answer", key=f"submit_{index}"):
                    topic = question_data["topic"]
                    if user_answer.strip().lower() in idk_phrases:
                        record_result(topic, "idk")
                    else:
                        def grade():
                            grading_system_prompt = (
                                "You are grading a student's quiz answer. The student's answer does NOT need to "
                                "match word-for-word — mark it CORRECT if it captures the same key idea or meaning "
                                "as the correct answer, even if phrased very differently, shorter, or missing minor "
                                "details. Only mark INCORRECT if the core idea is wrong or missing. Reply with ONLY "
                                "the single word 'CORRECT' or 'INCORRECT' — nothing else."
                            )
                            grading_user_prompt = f"Correct answer: {question_data['answer']}\nStudent's answer: {user_answer}"
                            return call_nemotron(grading_system_prompt, grading_user_prompt)

                        try:
                            grading_result = run_with_progress(grade, label="Grading your answer...")
                        except NemotronError as e:
                            st.error(f"⚠️ {e}")
                            st.stop()
                        correct = grading_result.strip().upper() == "CORRECT"
                        record_result(topic, "correct" if correct else "incorrect")

                    st.session_state["quiz_index"] += 1
                    st.rerun()

                if col2.button("I don't know", key=f"idk_{index}"):
                    record_result(question_data["topic"], "idk")
                    st.session_state["quiz_index"] += 1
                    st.rerun()

                if col3.button("Stop Quiz", key=f"stop_{index}"):
                    st.session_state["quiz_done"] = True
                    st.rerun()

        else:
            st.session_state["quiz_done"] = True

            if "round_folded_in" not in st.session_state:
                for topic, results in st.session_state["round_topic_results"].items():
                    st.session_state["history"].setdefault(
                        topic, {"attempts": 0, "mastered": False, "last_reviewed": None}
                    )
                    st.session_state["history"][topic]["attempts"] += 1
                    st.session_state["history"][topic]["mastered"] = (results["wrong"] == 0)
                    st.session_state["history"][topic]["last_reviewed"] = datetime.now().isoformat()

                overall = st.session_state["overall_score"]
                for key in ["correct", "incorrect", "idk"]:
                    overall[key] += st.session_state["score"][key]

                active_code = st.session_state.get("active_class_code")
                if active_code:
                    store = get_class_store()
                    if active_code in store:
                        score = st.session_state["score"]
                        total = score["correct"] + score["incorrect"] + score["idk"]
                        pct = round((score["correct"] / total) * 100) if total > 0 else 0
                        store[active_code]["quiz_results"].append({
                            "score_pct": pct,
                            "timestamp": datetime.now().isoformat()
                        })

                st.session_state["round_folded_in"] = True

            score = st.session_state["score"]
            total = score["correct"] + score["incorrect"] + score["idk"]
            percentage = round((score["correct"] / total) * 100) if total > 0 else 0

            st.success(f"📊 Quiz complete on **{quiz_label}**! {score['correct']} correct, {score['incorrect']} incorrect, {score['idk']} idk — {percentage}%")

            if len(st.session_state["weak_topics"]) == 0:
                st.success("🎉 No weak spots here — great job!")
                if st.button("Practice another topic"):
                    reset_quiz_state()
                    st.rerun()


# ---------- Teach-back section ----------

if st.session_state.get("quiz_done", False) and len(st.session_state.get("weak_topics", {})) > 0:
    weak_topics_list = list(st.session_state["weak_topics"].keys())

    st.divider()
    st.subheader("Teach-back Mode")
    st.write("For topics you struggled with, try explaining them to me like I'm a confused classmate!")

    if "tb_topic_index" not in st.session_state:
        st.session_state["tb_topic_index"] = 0
        st.session_state["tb_stage"] = "explain"
        st.session_state["tb_explanation"] = ""
        st.session_state["tb_followup_question"] = ""
        st.session_state["tb_done"] = False

    topic_index = st.session_state["tb_topic_index"]

    if not st.session_state["tb_done"] and topic_index < len(weak_topics_list):
        topic = weak_topics_list[topic_index]
        st.write(f"**Topic {topic_index + 1} of {len(weak_topics_list)}: {topic}**")

        past_attempts = st.session_state["history"].get(topic, {}).get("attempts", 0)
        depth_hint = (
            "This student has struggled with this topic across multiple attempts, so dig into a deeper, "
            "more specific angle than a beginner-level question."
            if past_attempts >= 2 else
            "Keep the question at a straightforward, first-pass level."
        )

        stage = st.session_state["tb_stage"]

        if stage == "explain":
            explanation = st.text_input(f"Can you explain '{topic}' to me?", key=f"explain_{topic_index}")

            col1, col2 = st.columns(2)
            if col1.button("Submit Explanation", key=f"submit_explain_{topic_index}"):
                st.session_state["tb_explanation"] = explanation

                def get_followup():
                    followup_system_prompt = (
                        f"You are a curious but confused student learning about {topic}. The user just tried "
                        f"to explain it to you. Ask ONE genuine, specific follow-up question about their "
                        f"explanation, like a real confused classmate would. Keep it short and natural — no "
                        f"more than 2 sentences. {depth_hint}"
                    )
                    return call_nemotron(followup_system_prompt, explanation)

                try:
                    followup_question = run_with_progress(get_followup, label="Thinking of a follow-up question...")
                except NemotronError as e:
                    st.error(f"⚠️ {e}")
                    st.stop()
                st.session_state["tb_followup_question"] = followup_question
                st.session_state["tb_stage"] = "followup"
                st.rerun()

            if col2.button("Skip this topic", key=f"skip_explain_{topic_index}"):
                st.session_state["tb_topic_index"] += 1
                st.session_state["tb_stage"] = "explain"
                st.rerun()

        elif stage == "followup":
            st.write("🤔", st.session_state["tb_followup_question"])
            followup_explanation = st.text_input("Your simpler explanation:", key=f"followup_{topic_index}")

            col1, col2 = st.columns(2)
            if col1.button("Submit", key=f"submit_followup_{topic_index}"):
                def get_verdict():
                    verdict_system_prompt = f"You are judging whether a student truly understands {topic}, based on their two explanations. Give brief, encouraging feedback (2-3 sentences) on whether their understanding seems solid or still surface-level, and why."
                    verdict_input = f"First explanation: {st.session_state['tb_explanation']}\nFollow-up explanation: {followup_explanation}"
                    return call_nemotron(verdict_system_prompt, verdict_input)

                try:
                    verdict = run_with_progress(get_verdict, label="Evaluating your understanding...")
                except NemotronError as e:
                    st.error(f"⚠️ {e}")
                    st.stop()
                st.session_state["tb_verdict"] = verdict
                st.session_state["tb_stage"] = "verdict"
                st.rerun()

            if col2.button("Skip this topic", key=f"skip_followup_{topic_index}"):
                st.session_state["tb_topic_index"] += 1
                st.session_state["tb_stage"] = "explain"
                st.rerun()

        elif stage == "verdict":
            st.write("✅", st.session_state["tb_verdict"])
            if st.button("Next Topic", key=f"next_{topic_index}"):
                st.session_state["tb_topic_index"] += 1
                st.session_state["tb_stage"] = "explain"
                st.rerun()

    else:
        st.session_state["tb_done"] = True
        st.balloons()
        st.success("🎉 Teach-back complete! Great work reviewing these topics.")
        if st.button("Practice another topic"):
            reset_quiz_state()
            st.rerun()