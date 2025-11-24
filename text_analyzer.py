import whisper
import librosa
import numpy as np
from pathlib import Path
from moviepy.editor import VideoFileClip
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Lightweight AudioMetrics dataclass so this file is standalone
@dataclass
class AudioMetrics:
    speaking_rate: float
    avg_pitch: float
    pitch_variance: float
    avg_volume: float
    volume_variance: float
    pause_count: int
    avg_pause_duration: float
    filler_words_count: int
    filler_words_rate: float
    total_duration: float
    speech_duration: float
    silence_ratio: float
    pitch_monotony_score: float
    energy_level: str
    confidence_score: float = 0.0

class PresentationAudioAnalyzer:
    """Comprehensive audio analysis for presentation coaching"""

    FILLER_WORDS = [
        'um', 'uh', 'like', 'you know', 'basically', 'actually',
        'literally', 'sort of', 'kind of', 'i mean', 'so', 'well',
        'right', 'okay', 'yeah', 'mhm'
    ]

    def __init__(self, model_size: str = "base"):
        print(f"Loading Whisper model ({model_size})...")
        self.whisper_model = whisper.load_model(model_size)
        print("✅ Whisper model loaded successfully!")

        # Ensure outputs folder exists
        self.output_dir = Path("outputs")
        self.temp_dir = self.output_dir / "temp"
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)

    def extract_audio_from_video(self, video_path: str) -> str:
        """Extract audio from video file and save inside outputs/temp"""
        base_name = Path(video_path).stem
        output_path = self.temp_dir / f"{base_name}_audio.wav"
        print(f"🎬 Extracting audio from: {video_path}")
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(str(output_path), logger=None)
        video.close()
        print(f"✅ Audio extracted → {output_path}")
        return str(output_path)

    def transcribe_with_timestamps(self, audio_path: str) -> Dict:
        """Transcribe audio with Whisper"""
        print("🎤 Transcribing audio...")
        result = self.whisper_model.transcribe(audio_path, word_timestamps=True, verbose=False)
        print("📝 Transcription done.")
        return result

    def analyze_speech_rate(self, transcription: Dict) -> Tuple[float, int]:
        words = transcription['text'].split()
        word_count = len(words)
        duration_minutes = transcription['segments'][-1]['end'] / 60 if transcription['segments'] else 0
        wpm = word_count / duration_minutes if duration_minutes > 0 else 0
        return wpm, word_count

    def detect_filler_words(self, transcription: Dict) -> Tuple[int, List[Dict]]:
        text = transcription['text'].lower()
        filler_instances = []
        total_count = 0
        for filler in self.FILLER_WORDS:
            count = text.count(f' {filler} ') + text.count(f' {filler},') + text.count(f' {filler}.')
            if count > 0:
                total_count += count
                filler_instances.append({'word': filler, 'count': count})
        return total_count, filler_instances

    def analyze_pauses(self, transcription: Dict, silence_threshold: float = 0.5) -> Tuple[int, float, List[float]]:
        segments = transcription['segments']
        pauses = []
        for i in range(len(segments) - 1):
            pause_duration = segments[i + 1]['start'] - segments[i]['end']
            if pause_duration > silence_threshold:
                pauses.append(pause_duration)
        pause_count = len(pauses)
        avg_pause = np.mean(pauses) if pauses else 0
        return pause_count, avg_pause, pauses

    def analyze_pitch(self, audio_path: str) -> Tuple[float, float, float]:
        y, sr = librosa.load(audio_path)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        if pitch_values:
            avg_pitch = np.mean(pitch_values)
            pitch_variance = np.var(pitch_values)
            pitch_std = np.std(pitch_values)
        else:
            avg_pitch = pitch_variance = pitch_std = 0
        return avg_pitch, pitch_variance, pitch_std

    def analyze_volume_energy(self, audio_path: str) -> Tuple[float, float, str]:
        y, sr = librosa.load(audio_path)
        rms = librosa.feature.rms(y=y)[0]
        avg_volume = np.mean(rms)
        volume_variance = np.var(rms)
        if avg_volume < 0.02:
            energy_level = "low"
        elif avg_volume < 0.05:
            energy_level = "medium"
        else:
            energy_level = "high"
        return avg_volume, volume_variance, energy_level

    def calculate_silence_ratio(self, audio_path: str, top_db: int = 30) -> Tuple[float, float]:
        y, sr = librosa.load(audio_path)
        intervals = librosa.effects.split(y, top_db=top_db)
        speech_duration = sum([(end - start) / sr for start, end in intervals])
        total_duration = len(y) / sr
        silence_ratio = 1 - (speech_duration / total_duration) if total_duration > 0 else 0
        return silence_ratio, speech_duration

    def calculate_confidence_score(self, metrics: Dict) -> float:
        score = 100.0
        if metrics['filler_words_rate'] > 10:
            score -= min(20, (metrics['filler_words_rate'] - 10) * 2)
        if metrics['avg_pause_duration'] > 2.0:
            score -= min(15, (metrics['avg_pause_duration'] - 2) * 5)
        score -= min(25, metrics['pitch_monotony_score'] / 4)
        ideal_rate = 150
        rate_deviation = abs(metrics['speaking_rate'] - ideal_rate)
        if rate_deviation > 30:
            score -= min(15, (rate_deviation - 30) / 5)
        if metrics['energy_level'] == 'low':
            score -= 10
        if metrics['silence_ratio'] > 0.3:
            score -= min(15, (metrics['silence_ratio'] - 0.3) * 50)
        return max(0, min(100, score))

    def generate_feedback(self, metrics: AudioMetrics) -> Dict[str, List[str]]:
        strengths, weaknesses, suggestions = [], [], []

        if 130 <= metrics.speaking_rate <= 170:
            strengths.append(f"Excellent speaking pace at {metrics.speaking_rate:.0f} words per minute")
        elif metrics.speaking_rate < 100:
            weaknesses.append(f"Speaking too slowly ({metrics.speaking_rate:.0f} WPM)")
            suggestions.append("Increase your pace to 130–170 WPM")
        elif metrics.speaking_rate > 190:
            weaknesses.append(f"Speaking too fast ({metrics.speaking_rate:.0f} WPM)")
            suggestions.append("Slow down to 130–170 WPM")

        if metrics.filler_words_rate < 3:
            strengths.append("Minimal use of filler words – great clarity")
        elif metrics.filler_words_rate > 8:
            weaknesses.append(f"High filler word usage ({metrics.filler_words_count} instances)")
            suggestions.append("Try pausing briefly instead of using filler words")

        if metrics.pitch_monotony_score < 30:
            strengths.append("Good vocal variety")
        elif metrics.pitch_monotony_score > 60:
            weaknesses.append("Monotonous tone detected")
            suggestions.append("Vary your tone to sound more engaging")

        if metrics.energy_level == "high":
            strengths.append("Strong vocal energy – confident delivery")
        elif metrics.energy_level == "low":
            weaknesses.append("Low vocal energy")
            suggestions.append("Project your voice more for engagement")

        if not strengths:
            strengths.append("Clear speech detected – keep it up!")

        return {'strengths': strengths, 'weaknesses': weaknesses, 'suggestions': suggestions}

    def analyze(self, audio_path: str, is_video: bool = False):
        if is_video:
            audio_path = self.extract_audio_from_video(audio_path)

        transcription = self.transcribe_with_timestamps(audio_path)
        total_duration = transcription['segments'][-1]['end'] if transcription['segments'] else 0

        speaking_rate, word_count = self.analyze_speech_rate(transcription)
        filler_count, _ = self.detect_filler_words(transcription)
        filler_rate = (filler_count / total_duration) * 60 if total_duration > 0 else 0
        pause_count, avg_pause, _ = self.analyze_pauses(transcription)
        avg_pitch, pitch_var, pitch_std = self.analyze_pitch(audio_path)
        pitch_monotony = max(0, 100 - (pitch_std / 10)) if pitch_std >= 0 else 0
        avg_volume, vol_var, energy_level = self.analyze_volume_energy(audio_path)
        silence_ratio, speech_duration = self.calculate_silence_ratio(audio_path)

        metrics_dict = {
            'speaking_rate': speaking_rate,
            'avg_pitch': avg_pitch,
            'pitch_variance': pitch_var,
            'avg_volume': avg_volume,
            'volume_variance': vol_var,
            'pause_count': pause_count,
            'avg_pause_duration': avg_pause,
            'filler_words_count': filler_count,
            'filler_words_rate': filler_rate,
            'total_duration': total_duration,
            'speech_duration': speech_duration,
            'silence_ratio': silence_ratio,
            'pitch_monotony_score': pitch_monotony,
            'energy_level': energy_level
        }

        confidence = self.calculate_confidence_score(metrics_dict)
        metrics_dict['confidence_score'] = confidence
        metrics = AudioMetrics(**metrics_dict)
        feedback = self.generate_feedback(metrics)

        results = {
            'metrics': metrics_dict,
            'feedback': feedback,
            'transcription': transcription['text']
        }

        base_name = Path(audio_path).stem.replace("_audio", "")
        report_path = self.output_dir / "audio_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== AUDIO ANALYSIS REPORT ===\n\n")
            f.write(f"Transcription:\n{transcription['text']}\n\n")
            f.write("=== METRICS ===\n")
            for k, v in metrics_dict.items():
                f.write(f"{k}: {v}\n")
            f.write("\n=== FEEDBACK ===\n")
            for section, items in feedback.items():
                f.write(f"\n[{section.upper()}]\n")
                for item in items:
                    f.write(f"- {item}\n")

        print(f"📄 Audio report saved to {report_path}")
        return metrics, results


