import streamlit as st
import os
import requests
from dotenv import load_dotenv
import json
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
    """Turns a flat list of flashcards into {topic_name: [cards...]}"""
    grouped = {}
    for card in flashcards:
        topic = card["topic"]
        grouped.setdefault(topic, []).append(card)
    return grouped


def reset_quiz_state():
    """Clears everything related to the current quiz + teach-back round,
    so the user can pick a fresh topic to practice."""
    keys_to_clear = [
        "quiz_topic", "quiz_cards", "quiz_index", "score", "weak_topics", "quiz_done",
        "tb_topic_index", "tb_stage", "tb_explanation", "tb_followup_question", "tb_verdict", "tb_done"
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


# ---------- The actual webpage starts here ----------

st.title("📚 Notes to Mastery")
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
        with st.spinner("Generating flashcards..."):
            flashcards = generate_flashcards(combined_notes)

        st.session_state["flashcards"] = flashcards
        reset_quiz_state()  # starting fresh flashcards means any old quiz progress is no longer valid
        st.success(f"Generated {len(flashcards)} flashcards!")

# ---------- Flashcard display, grouped by topic ----------

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


# ---------- Quiz section: pick a topic, then quiz on just that topic ----------

if "flashcards" in st.session_state:
    st.divider()
    st.subheader("Quiz Time")

    flashcards = st.session_state["flashcards"]
    topics_grouped = group_by_topic(flashcards)
    topic_names = list(topics_grouped.keys())

    idk_phrases = ["idk", "i don't know", "i dont know", "not sure", "no idea", ""]

    if "quiz_topic" not in st.session_state:
        # No topic chosen yet — show the picker
        st.write("Choose a topic to practice:")
        selected_topic = st.selectbox("Topic:", topic_names, key="topic_selector")

        if st.button("Start Quiz on this topic"):
            st.session_state["quiz_topic"] = selected_topic
            st.session_state["quiz_cards"] = topics_grouped[selected_topic]
            st.session_state["quiz_index"] = 0
            st.session_state["score"] = {"correct": 0, "incorrect": 0, "idk": 0}
            st.session_state["weak_topics"] = {}
            st.session_state["quiz_done"] = False
            st.rerun()

    else:
        # A topic has been chosen — run the quiz on just that topic's cards
        quiz_cards = st.session_state["quiz_cards"]
        index = st.session_state["quiz_index"]
        quiz_topic = st.session_state["quiz_topic"]

        if not st.session_state["quiz_done"] and index < len(quiz_cards):
            card = quiz_cards[index]
            st.write(f"**Topic: {quiz_topic}** — Question {index + 1} of {len(quiz_cards)}")
            st.write(card["question"])

            user_answer = st.text_input("Your answer:", key=f"answer_{index}")

            col1, col2, col3 = st.columns(3)

            if col1.button("Submit Answer", key=f"submit_{index}"):
                if user_answer.strip().lower() in idk_phrases:
                    st.session_state["score"]["idk"] += 1
                    st.session_state["weak_topics"][quiz_topic] = st.session_state["weak_topics"].get(quiz_topic, 0) + 1
                else:
                    grading_system_prompt = "You are grading a student's quiz answer. Given the correct answer and the student's answer, reply with ONLY the single word 'CORRECT' or 'INCORRECT' — nothing else."
                    grading_user_prompt = f"Correct answer: {card['answer']}\nStudent's answer: {user_answer}"
                    grading_result = call_nemotron(grading_system_prompt, grading_user_prompt)
                    correct = grading_result.strip().upper() == "CORRECT"

                    if correct:
                        st.session_state["score"]["correct"] += 1
                    else:
                        st.session_state["score"]["incorrect"] += 1
                        st.session_state["weak_topics"][quiz_topic] = st.session_state["weak_topics"].get(quiz_topic, 0) + 1

                st.session_state["quiz_index"] += 1
                st.rerun()

            if col2.button("I don't know", key=f"idk_{index}"):
                st.session_state["score"]["idk"] += 1
                st.session_state["weak_topics"][quiz_topic] = st.session_state["weak_topics"].get(quiz_topic, 0) + 1
                st.session_state["quiz_index"] += 1
                st.rerun()

            if col3.button("Stop Quiz", key=f"stop_{index}"):
                st.session_state["quiz_done"] = True
                st.rerun()

        else:
            st.session_state["quiz_done"] = True
            score = st.session_state["score"]
            total = score["correct"] + score["incorrect"] + score["idk"]
            percentage = round((score["correct"] / total) * 100) if total > 0 else 0

            st.success(f"📊 Quiz complete on **{quiz_topic}**! {score['correct']} correct, {score['incorrect']} incorrect, {score['idk']} idk — {percentage}%")

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

        stage = st.session_state["tb_stage"]

        if stage == "explain":
            explanation = st.text_input(f"Can you explain '{topic}' to me?", key=f"explain_{topic_index}")

            col1, col2 = st.columns(2)
            if col1.button("Submit Explanation", key=f"submit_explain_{topic_index}"):
                st.session_state["tb_explanation"] = explanation
                followup_system_prompt = f"You are a curious but confused student learning about {topic}. The user just tried to explain it to you. Ask ONE genuine, specific follow-up question about their explanation, like a real confused classmate would. Keep it short and natural — no more than 2 sentences."
                followup_question = call_nemotron(followup_system_prompt, explanation)
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
                verdict_system_prompt = f"You are judging whether a student truly understands {topic}, based on their two explanations. Give brief, encouraging feedback (2-3 sentences) on whether their understanding seems solid or still surface-level, and why."
                verdict_input = f"First explanation: {st.session_state['tb_explanation']}\nFollow-up explanation: {followup_explanation}"
                verdict = call_nemotron(verdict_system_prompt, verdict_input)
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
        st.success("🎉 Teach-back complete! Great work reviewing this topic.")
        if st.button("Practice another topic"):
            reset_quiz_state()
            st.rerun()