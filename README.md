
# 🎯 Premo — AI Presentation Coach
**Premo** is an AI-powered presentation coach that analyzes a recorded presentation video and returns moment-by-moment feedback on voice and body language, then generates human-like coaching delivered through a character (Premo) and a final performance score.

> Speak with confidence. Inspire with intelligence.

---

## 🔍 Project Summary

Premo accepts a user's presentation video and performs a multi-stage pipeline:

1. **Audio → Text & Audio Analysis**  
   - Transcribe speech using **Whisper**.  
   - Extract audio features (e.g., pitch, energy, tempo) using **Librosa**.  
   - Save analysis results to a structured file.

2. **Body Language Analysis**  
   - Process video frames using **MediaPipe** (pose & face landmarks) and **YOLO** (for object detection).  
   - Compute per-frame/body metrics (gestures, posture, eye contact, phone detection).  
   - Save time-aligned analysis results and produce an annotated video with overlays showing real-time feedback.

3. **Feedback Generation via LLM**  
   - Merge audio/text analysis and body-language analysis outputs.  
   - Send combined analysis to **Gemini (via API)** to generate human-like, actionable feedback.

4. **Character Rendering (Premo) & Output**  
   - Feed LLM feedback into the Premo-character API to produce a voiced narration (Premo persona).  
   - Produce final deliverables: annotated video, textual feedback report, and a performance **score** for the user.



---

## 🛠 Tech Stack

- **Language:** Python  
- **Audio / Speech:** Whisper , Librosa, moviepy 
- **Computer Vision:** OpenCV, MediaPipe, YOLO 
- **LLM / Feedback:** Gemini (via API)  
- **Frontend / Demo UI:** Streamlit 
- **Video processing / annotation:** MoviePy / OpenCV drawing utilities  
- **Character:** Replicate  API  


---

## ⚙️ Detailed Pipeline (what each step does)

1. **Ingest & Preprocess**
   - Accept uploaded video .  
   - Extract audio track (ffmpeg / MoviePy).

2. **Audio Transcription & Feature Extraction**
   - Transcribe audio with Whisper → `transcript.txt` (timestamped).  
   - Extract features via Librosa (tempo, energy, pitch,silence detection).  
   - Detect filler words.  
   - Output: `audio_analysis.txt` 

3. **Body Language Analysis**
   - Frame-by-frame processing:
     - Pose & facial landmarks with MediaPipe.
     - phone detection with YOLO.  
     - Compute metrics: gesture frequency, posture stability, eye contact, phone detection.  
   - Render annotated frames (overlay arrows, heatmaps, textual hints) and stitch annotated video: `video_annotated.mp4`.  
   - Output: `body_analysis.txt`.

4. **Merge & LLM Feedback**
   - combine `audio_analysis.txt` and `body_analysis.txt`.  
   - Prepare a structured prompt that summarizes detected issues and highlights strengths.  
   - Send prompt to Gemini API → receive step-by-step, human-readable feedback.  
   - Output: `feedback.txt` and `score`.

5. **Premo Character & Voice**
   - Use the feedback text.  
   - Call Premo-character API (Replicate) to synthesize spoken feedback.  

---

# 🔐 Configuration & API Keys
GEMINI_API_KEY=your_gemini_api_key

PREMO_TTS_API_KEY=Replicate_api_key  

---

# sample of the annoted video "output video"
<p align="center">
  <img src="https://github.com/SohaAshraff/PREMO/blob/main/example%20output.jpg" width="600">
  <br>
  <em>analyis the input video</em>
</p>

---

# Premo Character 
<p align="center">
  <img src="https://github.com/SohaAshraff/PREMO/blob/main/Premo.jpg" width="600">
  <br>
  <em>Premo Interface</em>
</p>

---

## demo while running the program

You can watch it here:  
[▶️ Watch tech Demo](https://github.com/SohaAshraff/PREMO/blob/main/techdemo.mp4)

---

## Premo example Video

You can watch premo feedback example here:  
[▶️ Watch Demo](https://github.com/SohaAshraff/PREMO/blob/main/Avatar%20IV%20Video.mp4)





