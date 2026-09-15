import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()  # reads the .env file and loads NEBIUS_API_KEY into the environment

API_KEY = os.environ.get("NEBIUS_API_KEY")
API_URL = "https://api.tokenfactory.nebius.com/v1/chat/completions"
MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"


def call_nemotron(system_prompt, user_prompt):
    """
    Sends a message to Nemotron via Nebius Token Factory and returns its text reply.
    system_prompt = instructions for how the AI should behave
    user_prompt = the actual input/question
    """
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
    response.raise_for_status()  # will raise an error if the request failed
    data = response.json()
    return data["choices"][0]["message"]["content"]


def generate_flashcards(notes_text):
    system_prompt = "You are a study assistant. Extract 3-5 key concepts from the given notes and turn them into flashcards. Respond ONLY with a JSON array like this: [{\"question\": \"...\", \"answer\": \"...\", \"topic\": \"...\"}]. No extra text, just the JSON."
    user_prompt = notes_text

    raw_response = call_nemotron(system_prompt, user_prompt)
    flashcards = json.loads(raw_response)
    return flashcards


notes = "Photosynthesis is how plants make energy from sunlight. The mitochondria is the powerhouse of the cell."
flashcards = generate_flashcards(notes)

for card in flashcards:
    print("Q:", card["question"])
    print("A:", card["answer"])
    print("Topic:", card["topic"])
    print("---")


def run_quiz(flashcards):
    weak_topics = {}
    score = {"correct": 0, "incorrect": 0, "idk": 0}

    # Any of these typed answers will count as "I don't know"
    idk_phrases = ["idk", "i don't know", "i dont know", "not sure", "no idea", ""]

    for card in flashcards:
        print("\nQuestion:", card["question"])
        user_answer = input("Your answer (type 'idk' if unsure, or 'quit' to stop the quiz): ").strip()

        if user_answer.lower() in ["quit", "exit", "stop"]:
            print("\n👋 Quiz ended early.")
            break

        if user_answer.lower() in idk_phrases:
            print("📝 No worries — we'll revisit this one.")
            score["idk"] += 1
            topic = card["topic"]
            weak_topics[topic] = weak_topics.get(topic, 0) + 1
            continue  # skip the AI grading call entirely, move to next question

        grading_system_prompt = "You are grading a student's quiz answer. Given the correct answer and the student's answer, reply with ONLY the single word 'CORRECT' or 'INCORRECT' — nothing else."
        grading_user_prompt = f"Correct answer: {card['answer']}\nStudent's answer: {user_answer}"

        grading_result = call_nemotron(grading_system_prompt, grading_user_prompt)
        correct = grading_result.strip().upper() == "CORRECT"

        if correct:
            print("✅ Correct!")
            score["correct"] += 1
        else:
            print("❌ Not quite. The answer was:", card["answer"])
            score["incorrect"] += 1
            topic = card["topic"]
            weak_topics[topic] = weak_topics.get(topic, 0) + 1

    return weak_topics, score


weak_topics, score = run_quiz(flashcards)

total_questions = score["correct"] + score["incorrect"] + score["idk"]
percentage = round((score["correct"] / total_questions) * 100) if total_questions > 0 else 0

print(f"\n📊 Score: {score['correct']} correct, {score['incorrect']} incorrect, {score['idk']} idk — {percentage}%")
print("Weak topics:", weak_topics)


def run_teach_back(weak_topics):
    print("\n--- Teach-back mode ---")
    print("For topics you struggled with, try explaining them to me like I'm a confused classmate!")

    for topic, mistake_count in weak_topics.items():
        print(f"\nLet's work on: {topic}")
        explanation = input(f"Can you explain '{topic}' to me? (or type 'quit' to stop): ")

        if explanation.strip().lower() in ["quit", "exit", "stop"]:
            print("\n👋 Teach-back session ended early.")
            break

        followup_system_prompt = f"You are a curious but confused student learning about {topic}. The user just tried to explain it to you. Ask ONE genuine, specific follow-up question about their explanation, like a real confused classmate would. Keep it short and natural — no more than 2 sentences."
        followup_question = call_nemotron(followup_system_prompt, explanation)
        print("🤔", followup_question)

        followup_explanation = input("Your simpler explanation (or type 'quit' to stop): ")

        if followup_explanation.strip().lower() in ["quit", "exit", "stop"]:
            print("\n👋 Teach-back session ended early.")
            break

        verdict_system_prompt = f"You are judging whether a student truly understands {topic}, based on their two explanations. Give brief, encouraging feedback (2-3 sentences) on whether their understanding seems solid or still surface-level, and why."
        verdict_input = f"First explanation: {explanation}\nFollow-up explanation: {followup_explanation}"
        verdict = call_nemotron(verdict_system_prompt, verdict_input)
        print("✅", verdict)


run_teach_back(weak_topics)