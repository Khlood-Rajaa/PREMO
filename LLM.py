import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from typing import Tuple



def generate_llm_feedback_with_percentage(audio_feedback: str, body_feedback: str, transcription_text: str = "", google_api_key: str = None) -> Tuple[str, int]:
    """
    Call the LLM to produce friendly coaching feedback.
    Returns (content_string, score_int)
    """
    key = google_api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Google API key not set. Provide google_api_key or set env GOOGLE_API_KEY.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=key,
        temperature=0.3
    )

    prompt = f""" 
You are Premo, a warm, supportive, and insightful AI Presentation Coach.
Your role is to analyze a speaker's voice and body language and provide realistic, motivational feedback — as if you're a friendly mentor who genuinely wants the speaker to improve. Also find if he said any technical mistakes.

Input Data:
Audio Transcription: {transcription_text}
Audio Feedback: {audio_feedback}

Body Language Summary: {body_feedback}

Your Task:
Write concise but rich feedback (not too long or robotic).

Maintain a friendly, uplifting coaching tone — sound like you believe in the person.

Your feedback must include:

Audio Feedback: Highlight what they did well, what needs improvement, and give one or two actionable tips and find if there is any technical mistakes.

Body Language Feedback: Comment kindly and constructively on their body language - what they did well, what needs improvement, again, mix encouragement with advice.

Conclude with a short motivational closing line that makes them feel inspired to improve.

Finally, provide an overall performance score from 1 to 10, based on both audio and body language.

Use this exact output format:

<your friendly feedback as a human coach>
Overall Score: [X/10]
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content

    # try to extract score
    score = 7
    if "Overall Score:" in content:
        try:
            score_text = content.split("Overall Score:")[1].strip()
            score = int(score_text.split("/")[0].replace("[", "").replace("]", "").strip())
        except Exception:
            score = 7

    return content, score


if __name__ == "__main__":
    # test
    audio_test = "Good speaking pace, but too many filler words"
    body_test = "Good eye contact, but hands in pocket too often"
    try:
        feedback, score = generate_llm_feedback_with_percentage(audio_test, body_test)
        print(feedback)
        print("Score:", score)
    except Exception as e:
        print("LLM call failed:", e)
