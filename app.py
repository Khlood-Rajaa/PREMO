import streamlit as st
import tempfile
import os
from moviepy.editor import VideoFileClip, AudioFileClip
from text_analyzer import PresentationAudioAnalyzer
from body_analyzer import process_video
from LLM import generate_llm_feedback_with_percentage


st.set_page_config(page_title="PREMO Analyzer", layout="wide")

st.markdown("""
    <style>
        body, .main {
            background-color: #f2f2f2; /* رمادي فاتح */
        }

        .big-title {
            text-align: center;
            color: #0d47a1;
            font-size: 5em;
            font-weight: 800;
            letter-spacing: 3px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            margin-top: 0.3em;
        }

        .subtitle {
            text-align: center;
            color: #555;
            font-size: 1.3em;
            margin-bottom: 1.5em;
        }

        .section {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }

        .video-container {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 2rem;
            flex-wrap: wrap;
        }

        video {
            border-radius: 15px;
            box-shadow: 0px 3px 10px rgba(0,0,0,0.2);
            width: 480px !important;
            height: 480px !important;
            object-fit: contain;
            background-color: #000;
        }

        .score-box {
            text-align: center;
            background-color: #d9edf7;
            border-left: 8px solid #1976d2;
            border-radius: 12px;
            padding: 1.5rem;
            font-size: 1.5em;
            font-weight: bold;
            color: #0d47a1;
            margin-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Header

st.markdown("<div class='big-title'>PREMO</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>🎤 AI Presentation Performance Coach</div>", unsafe_allow_html=True)

# Upload Section

uploaded_video = st.file_uploader("Upload your presentation video 🎥", type=["mp4", "mov", "avi"])

if uploaded_video:
    with st.spinner("Analyzing your presentation... please wait ⏳"):
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, uploaded_video.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())

        st.markdown("<div class='section'><h3>📽️ Uploaded Video</h3>", unsafe_allow_html=True)
        st.video(video_path)
        st.markdown("</div>", unsafe_allow_html=True)

        analyzer = PresentationAudioAnalyzer()
        metrics, results = analyzer.analyze(video_path, is_video=True)
        process_video(video_path, "final.mp4")

        video = VideoFileClip(r"C:\\Users\\Dell\\Desktop\\pre\\final.mp4")
        audio = AudioFileClip(r"outputs/temp/OMRAN_audio.wav")
        audio = audio.subclip(0, video.duration)
        final = video.set_audio(audio)
        final.write_videofile("finalvid.mp4", codec="libx264", audio_codec="aac")

        with open(r"C:\\Users\\Dell\\Desktop\\pre\\outputs\\audio_report.txt", "r", encoding="utf-8") as f:
            audio_test = f.read()

        with open(r"C:\\Users\\Dell\\Desktop\\pre\\analysis_report.txt", "r", encoding="utf-8") as f:
            body_test = f.read()

        feedback, score = generate_llm_feedback_with_percentage(audio_test, body_test)



        # Processed Videos
        st.markdown("<div class='section'><h3>🎬 Processed Videos</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            st.video(r"C:\\Users\\Dell\\Desktop\\pre\\finalvid.mp4")

        with col2:
            st.video(r"C:\\Users\\Dell\\Desktop\\pre\\Avatar IV Video.mp4")

        st.markdown("</div>", unsafe_allow_html=True)

        #  Feedback
        st.markdown("<div class='section'><h3>🧠 Feedback</h3>", unsafe_allow_html=True)
        st.write(feedback)
        st.markdown("</div>", unsafe_allow_html=True)

        # Score (from 10)
        st.markdown(f"""
            <div class='score-box'>
                🎯 <span style='font-size:1.8em;'>Score:</span> {score}/10
            </div>
        """, unsafe_allow_html=True)
