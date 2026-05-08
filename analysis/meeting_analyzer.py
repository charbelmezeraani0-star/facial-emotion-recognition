"""
Meeting Mood Analytics
======================
Processes any video file frame-by-frame, tracks emotions across all faces,
and generates a full PDF report with charts, statistics, and key moments.

Usage:
    python analysis/meeting_analyzer.py --video path/to/video.mp4
    python analysis/meeting_analyzer.py --video path/to/video.mp4 --model saved_models/custom_cnn_best.keras
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import argparse
import time
from collections import defaultdict
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import tensorflow as tf

from detection.face_detector import detect_faces_haar, crop_face, predict_emotion, EMOTIONS

# ── colours consistent with the rest of the project ────────────────────────
EMOTION_COLORS = {
    'angry':    '#e74c3c',
    'disgust':  '#8e44ad',
    'fear':     '#3498db',
    'happy':    '#f1c40f',
    'sad':      '#2980b9',
    'surprise': '#e67e22',
    'neutral':  '#95a5a6',
}

POSITIVE_EMOTIONS  = {'happy', 'surprise'}
NEGATIVE_EMOTIONS  = {'angry', 'disgust', 'fear', 'sad'}
NEUTRAL_EMOTIONS   = {'neutral'}


# ── data collection ─────────────────────────────────────────────────────────

class MeetingAnalyzer:
    def __init__(self, model_path, sample_rate=5):
        """
        model_path  : path to saved Keras model
        sample_rate : analyse every Nth frame (5 = good balance of speed/detail)
        """
        print(f"Loading model: {model_path}")
        self.model       = tf.keras.models.load_model(model_path, compile=False)
        self.sample_rate = sample_rate

        # Per-frame records
        self.timestamps    = []          # seconds
        self.frame_emotions = []         # dominant emotion per frame
        self.frame_probs   = []          # full prob vector per frame
        self.face_counts   = []          # number of faces per frame
        self.key_moments   = []          # (timestamp, emotion, confidence, frame_img)

    def analyze(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps        = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration   = total_frames / fps
        print(f"Video: {os.path.basename(video_path)}")
        print(f"Duration: {duration:.1f}s | FPS: {fps:.1f} | Frames: {total_frames:,}")
        print(f"Analysing every {self.sample_rate} frames...")

        prev_emotion    = None
        frame_idx       = 0
        processed       = 0
        t_start         = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % self.sample_rate != 0:
                continue

            timestamp = frame_idx / fps
            gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces     = detect_faces_haar(gray)

            if len(faces) == 0:
                self.timestamps.append(timestamp)
                self.frame_emotions.append('neutral')
                self.frame_probs.append(np.ones(7) / 7)
                self.face_counts.append(0)
                frame_idx += 1
                continue

            # Average probabilities across all detected faces in the frame
            all_probs = []
            for (x, y, w, h) in faces:
                face_arr = crop_face(gray, x, y, w, h)
                _, _, probs = predict_emotion(self.model, face_arr)
                all_probs.append(probs)

            avg_probs = np.mean(all_probs, axis=0)
            dominant  = EMOTIONS[avg_probs.argmax()]
            confidence = float(avg_probs.max())

            self.timestamps.append(timestamp)
            self.frame_emotions.append(dominant)
            self.frame_probs.append(avg_probs)
            self.face_counts.append(len(faces))

            # Record key moments: emotion transitions with high confidence
            if dominant != prev_emotion and confidence > 0.6:
                if len(self.key_moments) < 12:   # cap at 12 key moments
                    thumb = cv2.resize(frame, (160, 120))
                    self.key_moments.append((timestamp, dominant, confidence, thumb))
                prev_emotion = dominant

            processed += 1
            if processed % 50 == 0:
                elapsed = time.time() - t_start
                pct     = timestamp / duration * 100
                print(f"  {pct:.0f}%  t={timestamp:.1f}s  [{elapsed:.0f}s elapsed]")

        cap.release()
        print(f"Analysis complete — {processed} frames processed.")
        return self


    # ── statistics ───────────────────────────────────────────────────────────

    def compute_stats(self):
        probs_arr = np.array(self.frame_probs)    # (T, 7)
        counts    = defaultdict(int)
        for e in self.frame_emotions:
            counts[e] += 1
        total = max(len(self.frame_emotions), 1)

        stats = {
            'duration_s':     self.timestamps[-1] if self.timestamps else 0,
            'frames_analysed': len(self.timestamps),
            'emotion_pct':    {e: counts[e] / total * 100 for e in EMOTIONS},
            'mean_probs':     probs_arr.mean(axis=0),
            'dominant_overall': max(counts, key=counts.get) if counts else 'neutral',
            'engagement_score': sum(counts[e] for e in POSITIVE_EMOTIONS) / total * 100,
            'negativity_score': sum(counts[e] for e in NEGATIVE_EMOTIONS) / total * 100,
            'avg_faces':      float(np.mean(self.face_counts)) if self.face_counts else 0,
        }
        return stats


# ── chart generators ─────────────────────────────────────────────────────────

def plot_emotion_timeline(analyzer, save_path):
    """Stacked area chart of emotion probabilities over time."""
    probs = np.array(analyzer.frame_probs)    # (T, 7)
    times = np.array(analyzer.timestamps)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.stackplot(times,
                 [probs[:, i] for i in range(7)],
                 labels=EMOTIONS,
                 colors=[EMOTION_COLORS[e] for e in EMOTIONS],
                 alpha=0.85)
    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel('Emotion Probability', fontsize=11)
    ax.set_title('Emotion Timeline — Probability Over Time', fontsize=13, weight='bold')
    ax.legend(loc='upper right', fontsize=8, ncol=4)
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_dominant_emotion_bar(analyzer, stats, save_path):
    """Horizontal bar chart of time spent per emotion."""
    emotions = list(stats['emotion_pct'].keys())
    pcts     = [stats['emotion_pct'][e] for e in emotions]
    colors   = [EMOTION_COLORS[e] for e in emotions]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(emotions, pcts, color=colors, edgecolor='white', height=0.6)
    for bar, pct in zip(bars, pcts):
        if pct > 2:
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{pct:.1f}%', va='center', fontsize=9)
    ax.set_xlabel('Time (%)', fontsize=11)
    ax.set_title('Time Spent per Emotion', fontsize=13, weight='bold')
    ax.set_xlim(0, max(pcts) * 1.15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_pie_chart(stats, save_path):
    """Pie chart of overall emotion distribution."""
    emotions = [e for e in EMOTIONS if stats['emotion_pct'][e] > 0.5]
    sizes    = [stats['emotion_pct'][e] for e in emotions]
    colors   = [EMOTION_COLORS[e] for e in emotions]
    explode  = [0.05] * len(emotions)

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=emotions, colors=colors, explode=explode,
        autopct='%1.1f%%', startangle=140,
        textprops={'fontsize': 10})
    for at in autotexts:
        at.set_fontsize(9)
    ax.set_title('Overall Emotion Distribution', fontsize=13, weight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_engagement_gauge(stats, save_path):
    """Gauge-style chart showing engagement vs negativity scores."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    for ax, (label, score, color) in zip(axes, [
        ('Engagement Score', stats['engagement_score'], '#2ecc71'),
        ('Negativity Score', stats['negativity_score'], '#e74c3c'),
    ]):
        theta = np.linspace(0, np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), 'lightgrey', linewidth=15)
        fill  = np.linspace(0, np.pi * score / 100, 200)
        ax.plot(np.cos(fill),  np.sin(fill),  color,      linewidth=15)
        ax.text(0, -0.15, f'{score:.1f}%', ha='center', va='center',
                fontsize=22, weight='bold', color=color)
        ax.text(0, -0.45, label, ha='center', fontsize=11)
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.6, 1.2)
        ax.axis('off')

    plt.suptitle('Meeting Mood Scores', fontsize=13, weight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_face_count(analyzer, save_path):
    """Line chart of number of faces detected over time."""
    fig, ax = plt.subplots(figsize=(14, 2.5))
    ax.fill_between(analyzer.timestamps, analyzer.face_counts,
                    alpha=0.5, color='#3498db')
    ax.plot(analyzer.timestamps, analyzer.face_counts, '#2980b9', linewidth=1)
    ax.set_xlabel('Time (seconds)', fontsize=10)
    ax.set_ylabel('Faces', fontsize=10)
    ax.set_title('Number of Faces Detected Over Time', fontsize=11, weight='bold')
    ax.set_xlim(analyzer.timestamps[0], analyzer.timestamps[-1])
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ── PDF report ───────────────────────────────────────────────────────────────

def generate_pdf_report(analyzer, stats, video_path, output_dir, chart_paths):
    try:
        from fpdf import FPDF
    except ImportError:
        print("fpdf2 not installed — skipping PDF. Run: pip install fpdf2")
        return None

    class PDF(FPDF):
        def header(self):
            self.set_font('Helvetica', 'B', 14)
            self.set_fill_color(44, 62, 80)
            self.set_text_color(255, 255, 255)
            self.cell(0, 12, '  Meeting Mood Analytics Report', fill=True, ln=True)
            self.set_text_color(0, 0, 0)
            self.ln(3)

        def footer(self):
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10, f'Page {self.page_no()} | Generated by Emotion Detection System', align='C')

    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=18)

    # ── Title section ──
    pdf.set_font('Helvetica', 'B', 18)
    pdf.cell(0, 10, 'Meeting Mood Analytics', ln=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f'Video: {os.path.basename(video_path)}', ln=True)
    pdf.cell(0, 6, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # ── Summary stats box ──
    pdf.set_fill_color(236, 240, 241)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, 'Summary Statistics', fill=True, ln=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.ln(1)

    dur  = stats['duration_s']
    mins = int(dur // 60); secs = int(dur % 60)
    rows = [
        ('Duration',           f'{mins}m {secs}s'),
        ('Frames Analysed',    f"{stats['frames_analysed']:,}"),
        ('Dominant Emotion',   stats['dominant_overall'].upper()),
        ('Engagement Score',   f"{stats['engagement_score']:.1f}%  (happy + surprise)"),
        ('Negativity Score',   f"{stats['negativity_score']:.1f}%  (angry + fear + disgust + sad)"),
        ('Avg Faces / Frame',  f"{stats['avg_faces']:.1f}"),
    ]
    col_w = 65
    for label, value in rows:
        pdf.set_font('Helvetica', 'B', 9); pdf.cell(col_w, 7, label + ':', ln=False)
        pdf.set_font('Helvetica', '', 9);  pdf.cell(0, 7, value, ln=True)
    pdf.ln(4)

    # ── Emotion breakdown table ──
    pdf.set_fill_color(236, 240, 241)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, 'Emotion Breakdown', fill=True, ln=True)
    pdf.ln(1)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.cell(50, 7, 'Emotion', border='B'); pdf.cell(40, 7, 'Time %', border='B')
    pdf.cell(0, 7, 'Avg Confidence', border='B', ln=True)
    pdf.set_font('Helvetica', '', 9)
    for i, e in enumerate(EMOTIONS):
        pct  = stats['emotion_pct'][e]
        conf = float(stats['mean_probs'][i]) * 100
        pdf.cell(50, 6, e.capitalize())
        pdf.cell(40, 6, f'{pct:.1f}%')
        pdf.cell(0,  6, f'{conf:.1f}%', ln=True)
    pdf.ln(5)

    # ── Charts ──
    chart_labels = [
        ('emotion_timeline',  'Emotion Timeline'),
        ('emotion_bar',       'Time per Emotion'),
        ('emotion_pie',       'Emotion Distribution'),
        ('engagement_gauge',  'Mood Scores'),
        ('face_count',        'Face Count Over Time'),
    ]
    for key, title in chart_labels:
        path = chart_paths.get(key)
        if path and os.path.exists(path):
            pdf.set_font('Helvetica', 'B', 11)
            pdf.set_fill_color(236, 240, 241)
            pdf.cell(0, 8, title, fill=True, ln=True)
            pdf.ln(1)
            pdf.image(path, w=pdf.epw)
            pdf.ln(4)

    # ── Key moments ──
    if analyzer.key_moments:
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_fill_color(236, 240, 241)
        pdf.cell(0, 8, 'Key Emotion Moments', fill=True, ln=True)
        pdf.ln(2)
        pdf.set_font('Helvetica', '', 9)

        cols = 4
        thumb_w = (pdf.epw - (cols-1)*3) / cols
        for i, (ts, emotion, conf, thumb_bgr) in enumerate(analyzer.key_moments):
            # Save thumbnail temporarily
            thumb_path = os.path.join(output_dir, f'_thumb_{i}.jpg')
            cv2.imwrite(thumb_path, thumb_bgr)

            col = i % cols
            if col == 0 and i > 0:
                pdf.ln(thumb_w * 0.75 + 10)

            x = pdf.get_x() + col * (thumb_w + 3)
            y = pdf.get_y()
            pdf.image(thumb_path, x=x, y=y, w=thumb_w)

            mins_ts = int(ts // 60); secs_ts = int(ts % 60)
            label   = f"{mins_ts}:{secs_ts:02d} — {emotion} ({conf:.0%})"
            pdf.set_xy(x, y + thumb_w * 0.75 + 1)
            pdf.set_font('Helvetica', '', 7)
            pdf.cell(thumb_w, 4, label, align='C')
            if col < cols - 1:
                pdf.set_xy(x + thumb_w + 3, y)

        # Clean up temp thumbnails
        for i in range(len(analyzer.key_moments)):
            p = os.path.join(output_dir, f'_thumb_{i}.jpg')
            if os.path.exists(p):
                os.remove(p)

    pdf_path = os.path.join(output_dir, 'meeting_report.pdf')
    pdf.output(pdf_path)
    print(f"PDF report saved → {pdf_path}")
    return pdf_path


# ── main ─────────────────────────────────────────────────────────────────────

def run(video_path, model_path='saved_models/custom_cnn_best.keras',
        sample_rate=5, output_dir=None):

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Output folder named after video
    base       = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = output_dir or os.path.join('analysis', 'reports', base)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # ── Analyse ──
    analyzer = MeetingAnalyzer(model_path, sample_rate=sample_rate)
    analyzer.analyze(video_path)
    stats = analyzer.compute_stats()

    # ── Generate charts ──
    print("Generating charts...")
    chart_paths = {
        'emotion_timeline': os.path.join(output_dir, 'emotion_timeline.png'),
        'emotion_bar':      os.path.join(output_dir, 'emotion_bar.png'),
        'emotion_pie':      os.path.join(output_dir, 'emotion_pie.png'),
        'engagement_gauge': os.path.join(output_dir, 'engagement_gauge.png'),
        'face_count':       os.path.join(output_dir, 'face_count.png'),
    }
    plot_emotion_timeline(analyzer, chart_paths['emotion_timeline'])
    plot_dominant_emotion_bar(analyzer, stats, chart_paths['emotion_bar'])
    plot_pie_chart(stats, chart_paths['emotion_pie'])
    plot_engagement_gauge(stats, chart_paths['engagement_gauge'])
    plot_face_count(analyzer, chart_paths['face_count'])

    # ── Print summary ──
    print("\n" + "="*50)
    print("MEETING MOOD SUMMARY")
    print("="*50)
    print(f"  Duration          : {int(stats['duration_s']//60)}m {int(stats['duration_s']%60)}s")
    print(f"  Dominant Emotion  : {stats['dominant_overall'].upper()}")
    print(f"  Engagement Score  : {stats['engagement_score']:.1f}%")
    print(f"  Negativity Score  : {stats['negativity_score']:.1f}%")
    print(f"  Key Moments Found : {len(analyzer.key_moments)}")
    print("\n  Emotion Breakdown:")
    for e in EMOTIONS:
        bar = '█' * int(stats['emotion_pct'][e] / 2)
        print(f"    {e:10s} {stats['emotion_pct'][e]:5.1f}%  {bar}")
    print("="*50)

    # ── Generate PDF ──
    pdf_path = generate_pdf_report(analyzer, stats, video_path, output_dir, chart_paths)

    print(f"\nAll outputs saved to: {output_dir}/")
    if pdf_path:
        print(f"Open the report: xdg-open '{pdf_path}'")
    return stats


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Meeting Mood Analytics')
    p.add_argument('--video',       required=True, help='Path to video file')
    p.add_argument('--model',       default='saved_models/custom_cnn_best.keras')
    p.add_argument('--sample-rate', type=int, default=5,
                   help='Analyse every Nth frame (lower=slower but more detail)')
    p.add_argument('--output-dir',  default=None)
    args = p.parse_args()
    run(args.video, args.model, args.sample_rate, args.output_dir)
