import streamlit as st
import os
import requests
from dotenv import load_dotenv
import json
import random
import string
import threading
import time
from pypdf import PdfReader

load_dotenv()

API_KEY = os.environ.get("NEBIUS_API_KEY")
API_URL = "https://api.tokenfactory.nebius.com/v1/chat/completions"
MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"


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
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def run_with_progress(worker_fn, label="Working..."):
    """
    Runs worker_fn() in a background thread while showing a simulated
    progress bar (0-90% while waiting, snaps to 100% when actually done).
    worker_fn takes no arguments and returns whatever result you want back.
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
    return json.loads(raw_response)


def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def group_by_topic(flashcards):
    grouped = {}
    for card in flashcards:
        topic = card["topic"]
        grouped.setdefault(topic, []).append(card)
    return grouped


def generate_quiz_questions(cards, quiz_type):
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
            "in options. No extra text, just the JSON."
        )
    else:
        system_prompt = (
            "You are a quiz writer. For each flashcard given (question and answer), reword ONLY the question "
            "using different wording than the original while keeping the exact same meaning and the same "
            "correct answer. Respond ONLY with a JSON array, one entry per flashcard, in the SAME order as "
            "given, in this exact format: [{\"question\": \"...\", \"answer\": \"...\"}]. No extra text."
        )

    raw_response = call_nemotron(system_prompt, json.dumps(cards_for_prompt))
    quiz_questions = json.loads(raw_response)

    safe_length = min(len(quiz_questions), len(cards))
    for i in range(safe_length):
        quiz_questions[i]["topic"] = cards[i]["topic"]
    return quiz_questions[:safe_length]


def reset_quiz_state():
    keys_to_clear = [
        "quiz_topic", "quiz_type", "quiz_questions", "quiz_index", "score",
        "weak_topics", "quiz_done", "round_topic_results",
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


@st.cache_resource
def get_class_store():
    return {}


def make_class_code():
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=5))


# ---------- Page setup ----------

st.title("📚 Notes to Mastery")
ensure_history_initialized()

# ---------- Sidebar: Class Mode + Progress Dashboard ----------

with st.sidebar:
    st.header("🏫 Class Mode")
    st.caption("Join a set someone shared with you, or create your own below after generating flashcards.")
    join_code = st.text_input("Enter a class code:", key="join_code_input").strip().upper()
    if st.button("Join"):
        store = get_class_store()
        if join_code in store:
            st.session_state["flashcards"] = store[join_code]["flashcards"]
            reset_quiz_state()
            st.success(f"Joined class code {join_code}! Scroll down to see the flashcards.")
        else:
            st.error("That code wasn't found. Double-check it with whoever shared it.")

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
            st.write(f"- {topic}: {status} ({data['attempts']} attempt{'s' if data['attempts'] != 1 else ''})")

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
        loaded = json.load(uploaded_progress)
        st.session_state["flashcards"] = loaded.get("flashcards", [])
        st.session_state["history"] = loaded.get("history", {})
        st.session_state["overall_score"] = loaded.get("overall_score", {"correct": 0, "incorrect": 0, "idk": 0})
        reset_quiz_state()
        st.success("Progress loaded!")
        st.rerun()


# ---------- Notes / PDF input ----------

st.write("Paste your notes below, or upload up to 10 PDFs, and I'll turn them into flashcards.")

notes_input = st.text_area("Your notes:", height=150)
uploaded_pdfs = st.file_uploader("Or upload up to 10 PDFs:", type=["pdf"], accept_multiple_files=True)

if uploaded_pdfs and len(uploaded_pdfs) > 10:
    st.warning("You can upload up to 10 PDFs at a time — only the first 10 will be used.")
    uploaded_pdfs = uploaded_pdfs[:10]

if st.button("Generate Flashcards"):
    combined_notes = notes_input.strip()

    if uploaded_pdfs:
        with st.spinner(f"Reading {len(uploaded_pdfs)} PDF(s)..."):
            for pdf_file in uploaded_pdfs:
                pdf_text = extract_text_from_pdf(pdf_file)
                combined_notes = (combined_notes + "\n" + pdf_text).strip()

    if combined_notes == "":
        st.warning("Please paste some notes or upload at least one PDF first!")
    else:
        flashcards = run_with_progress(
            lambda: generate_flashcards(combined_notes),
            label="Generating flashcards..."
        )
        st.session_state["flashcards"] = flashcards
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
                st.write(f"**Q:** {card['question']}")
                st.write(f"**A:** {card['answer']}")
                st.divider()

    st.write("**Sharing this set with a class or study group?**")
    if st.button("Create a class code for this set"):
        store = get_class_store()
        code = make_class_code()
        store[code] = {"flashcards": flashcards}
        st.success(f"Share this code with others: **{code}** (they enter it in the sidebar 'Class Mode' box)")


# ---------- Quiz section ----------

if "flashcards" in st.session_state:
    st.divider()
    st.subheader("Quiz Time")

    flashcards = st.session_state["flashcards"]
    topics_grouped = group_by_topic(flashcards)
    topic_names = list(topics_grouped.keys())

    idk_phrases = ["idk", "i don't know", "i dont know", "not sure", "no idea", ""]

    weak_pool_topics = [t for t, d in st.session_state["history"].items() if not d["mastered"] and d["attempts"] > 0]

    if "quiz_topic" not in st.session_state:
        st.write("What would you like to practice?")

        mode_options = ["Choose a specific topic"]
        if weak_pool_topics:
            mode_options.insert(0, f"Retry all weak topics ({len(weak_pool_topics)})")

        mode_choice = st.radio("Practice mode:", mode_options, key="practice_mode")

        if mode_choice.startswith("Retry all weak topics"):
            cards_for_quiz = [c for c in flashcards if c["topic"] in weak_pool_topics]
            chosen_label = "Weak Topics Review"
        else:
            selected_topic = st.selectbox("Topic:", topic_names, key="topic_selector")
            cards_for_quiz = topics_grouped[selected_topic]
            chosen_label = selected_topic

        quiz_type_label = st.radio("Quiz type:", options=["Open-ended", "Multiple Choice"], key="quiz_type_selector")
        quiz_type = "mcq" if quiz_type_label == "Multiple Choice" else "open"

        if st.button("Start Quiz"):
            quiz_questions = run_with_progress(
                lambda: generate_quiz_questions(cards_for_quiz, quiz_type),
                label="Preparing your quiz..."
            )

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

                        grading_result = run_with_progress(grade, label="Grading your answer...")
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
                    st.session_state["history"].setdefault(topic, {"attempts": 0, "mastered": False})
                    st.session_state["history"][topic]["attempts"] += 1
                    st.session_state["history"][topic]["mastered"] = (results["wrong"] == 0)

                overall = st.session_state["overall_score"]
                for key in ["correct", "incorrect", "idk"]:
                    overall[key] += st.session_state["score"][key]

                st.session_state["round_folded_in"] = True

            score = st.session_state["score"]
            total = score["correct"] + score["incorrect"] + score["idk"]
            percentage = round((score["correct"] / total) * 100) if total > 0 else 0

            st.success(f"📊 Quiz complete on **{quiz_label}**! {score['correct']} correct, {score['incorrect']} incorrect, {score['idk']} idk — {percentage}%")

            if len(st.session_state["weak_topics"]) == 0:
                st.success("🎉 No weak spots here — great job!")
                if st.button("Practice another topic"):
                    reset_quiz_state()
                    if "round_folded_in" in st.session_state:
                        del st.session_state["round_folded_in"]
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

                followup_question = run_with_progress(get_followup, label="Thinking of a follow-up question...")
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

                verdict = run_with_progress(get_verdict, label="Evaluating your understanding...")
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
            if "round_folded_in" in st.session_state:
                del st.session_state["round_folded_in"]
            st.rerun()