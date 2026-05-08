import os, sys, tempfile, collections
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from PIL import Image
import keras
keras.config.enable_unsafe_deserialization()

from evaluation.gradcam import compute_gradcam, overlay_gradcam
from detection.face_detector import (detect_faces_haar, crop_face,
                                     predict_emotion, EMOTIONS)

# ── Constants ────────────────────────────────────────────────────────────────

EMOTION_COLORS = {
    'angry':    '#e74c3c',
    'disgust':  '#8e44ad',
    'fear':     '#3498db',
    'happy':    '#f1c40f',
    'sad':      '#2980b9',
    'surprise': '#e67e22',
    'neutral':  '#95a5a6',
}

AVAILABLE_MODELS = {
    name: path for name, path in {
        'Custom CNN v1  (Best checkpoint)': 'saved_models/custom_cnn_best.keras',
        'Custom CNN v2  (Fine-tuned best)': 'saved_models/custom_cnn_v2_best.keras',
        'Custom CNN v2  (SWA)':             'saved_models/custom_cnn_v2_swa.keras',
        'MobileNetV2    (SWA)':             'saved_models/mobilenetv2_swa.keras',
    }.items() if os.path.exists(path)
}

# Known benchmark results (all with 8-pass TTA unless noted)
BENCHMARK = {
    'DeepFace\n(pretrained)':              52.80,
    'MobileNetV2\nSWA + TTA':              68.19,
    'Custom CNN v1\nBest + TTA':           69.42,
    'Custom CNN v2\nBest + TTA':           69.73,
    'Custom CNN v2\nSWA + TTA':            69.54,
    'Ensemble\n(4 models + TTA)':          70.45,
}

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title='Emotion Detection System',
    page_icon='🎭',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown("""
<style>
  /* ── Sidebar background ── */
  [data-testid="stSidebar"] { background: #0f172a; }

  /* ── Sidebar ALL text → bright white ── */
  [data-testid="stSidebar"] * { color: #f1f5f9 !important; }

  /* Sidebar selectbox / dropdown text */
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] { color: #f1f5f9 !important; }

  /* Sidebar checkboxes */
  [data-testid="stSidebar"] .stCheckbox label p { color: #f1f5f9 !important; font-size: 1rem !important; }
  [data-testid="stSidebar"] .stCheckbox span     { color: #f1f5f9 !important; }

  /* Sidebar markdown text */
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3,
  [data-testid="stSidebar"] small,
  [data-testid="stSidebar"] .stCaption { color: #cbd5e1 !important; }

  /* Sidebar divider lines */
  [data-testid="stSidebar"] hr { border-color: #334155 !important; }

  /* ── Main page font sizes ── */
  .main-title { font-size: 2.8rem; font-weight: 800; color: #e2e8f0; line-height: 1.1 }
  .sub-title  { font-size: 1.15rem; color: #94a3b8; margin-bottom: 1.5rem }

  /* General body text bigger */
  .stMarkdown p, .stText  { font-size: 1.05rem !important; }
  .stMarkdown li          { font-size: 1.05rem !important; }

  /* Tab labels bigger */
  button[data-baseweb="tab"] p { font-size: 1rem !important; font-weight: 600 !important; }

  /* Section headers */
  h3 { font-size: 1.5rem !important; }
  h4 { font-size: 1.25rem !important; }

  .badge {
      display: inline-block; padding: 6px 16px; border-radius: 20px;
      font-size: 1rem; font-weight: 700; color: #fff; margin: 3px;
  }
  .metric-box {
      background: #1e293b; border-radius: 12px; padding: 18px 14px;
      border-left: 4px solid #6366f1; margin: 6px 0; text-align: center;
  }
  .metric-box .val { font-size: 2.1rem; font-weight: 800; color: #e2e8f0 }
  .metric-box .lbl { font-size: 0.88rem; color: #94a3b8; margin-top: 4px }

  section[data-testid="stFileUploadDropzone"] { border: 2px dashed #6366f1 !important }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎭 Emotion AI")
    st.markdown("---")

    if not AVAILABLE_MODELS:
        st.error("No trained models found.\nRun `python3 training/train.py` first.")
        st.stop()

    model_name = st.selectbox("🤖 Active model", list(AVAILABLE_MODELS.keys()))
    st.markdown("---")

    show_gradcam   = st.checkbox("🔥 Show Grad-CAM",         value=True)
    show_deepface  = st.checkbox("🆚 Compare with DeepFace",  value=False)
    st.markdown("---")

    st.markdown("### Dataset")
    st.markdown("RAF-DB + FER2013 + extra  \n~85k training images  \n7 emotion classes")
    st.markdown("**Best accuracy**")
    st.markdown(
        "<div class='metric-box'>"
        "<div class='val'>70.45%</div>"
        "<div class='lbl'>4-model ensemble + TTA</div>"
        "</div>", unsafe_allow_html=True)
    st.markdown("**vs DeepFace pretrained**")
    st.markdown(
        "<div class='metric-box'>"
        "<div class='val'>52.80%</div>"
        "<div class='lbl'>DeepFace (FER2013 backbone)</div>"
        "</div>", unsafe_allow_html=True)


# ── Model loader (cached) ─────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def load_model(path):
    m = tf.keras.models.load_model(path, compile=False)
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return m

model = load_model(AVAILABLE_MODELS[model_name])


# ── DeepFace (lazy, cached) ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading DeepFace…")
def get_deepface():
    from deepface import DeepFace
    return DeepFace

def deepface_predict(bgr_face_crop):
    """Run DeepFace on a BGR face crop; returns (emotion_str, conf, probs_array)."""
    try:
        DeepFace = get_deepface()
        rgb = cv2.cvtColor(bgr_face_crop, cv2.COLOR_BGR2RGB)
        res = DeepFace.analyze(rgb, actions=['emotion'],
                               enforce_detection=False, silent=True)
        em_dict = res[0]['emotion']
        dom     = res[0]['dominant_emotion']
        probs   = np.array([em_dict.get(e, 0) for e in EMOTIONS], dtype=np.float32) / 100.0
        return dom, float(em_dict[dom] / 100.0), probs
    except Exception as e:
        return 'neutral', 0.0, np.ones(7) / 7


# ── Helper: run detection on BGR frame ────────────────────────────────────────

def run_detection(frame_bgr):
    gray    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces   = detect_faces_haar(gray)
    results = []
    for (x, y, w, h) in faces:
        face_arr            = crop_face(gray, x, y, w, h)
        emotion, conf, probs = predict_emotion(model, face_arr)
        # keep raw BGR crop for DeepFace
        bgr_crop = frame_bgr[max(0,y):y+h, max(0,x):x+w]
        results.append({
            'box':       (int(x), int(y), int(w), int(h)),
            'face':      face_arr,
            'bgr_crop':  bgr_crop,
            'emotion':   emotion,
            'confidence': float(conf),
            'probs':     probs,
        })
    return results


# ── Helper: draw face boxes on frame ─────────────────────────────────────────

def annotate_frame(frame_bgr, results):
    out = frame_bgr.copy()
    for r in results:
        x, y, w, h = r['box']
        hex_c   = EMOTION_COLORS[r['emotion']]
        bgr_c   = tuple(int(hex_c.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
        cv2.rectangle(out, (x, y), (x+w, y+h), bgr_c, 2)
        label = f"{r['emotion'].upper()}  {r['confidence']:.0%}"
        cv2.putText(out, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, bgr_c, 2)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


# ── Helper: horizontal probability bar chart ──────────────────────────────────

def prob_bar_chart(probs, title='', highlight=None):
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    colors = [EMOTION_COLORS[e] for e in EMOTIONS]
    bars   = ax.barh(EMOTIONS, probs, color=colors, height=0.65,
                     edgecolor='none')
    for bar, p, em in zip(bars, probs, EMOTIONS):
        ax.text(min(bar.get_width() + 0.02, 1.08),
                bar.get_y() + bar.get_height() / 2,
                f'{p:.1%}', va='center', fontsize=8, color='#e2e8f0')
        if em == highlight:
            bar.set_edgecolor('white')
            bar.set_linewidth(2)

    ax.set_xlim(0, 1.18)
    ax.set_xlabel('Probability', color='#94a3b8', fontsize=8)
    ax.tick_params(colors='#94a3b8', labelsize=8)
    ax.spines[:].set_visible(False)
    ax.invert_yaxis()
    if title:
        ax.set_title(title, color='#e2e8f0', fontsize=9, pad=6)
    plt.tight_layout(pad=0.5)
    return fig


# ── Helper: render one detected face ─────────────────────────────────────────

def render_face_card(r, face_idx, show_df):
    emotion = r['emotion']
    conf    = r['confidence']
    hex_c   = EMOTION_COLORS[emotion]

    st.markdown(
        f"<div style='border-left:4px solid {hex_c}; padding:6px 10px; "
        f"background:#1e293b; border-radius:8px; margin-bottom:8px'>"
        f"<b style='color:{hex_c};font-size:1.15rem'>{emotion.upper()}</b>"
        f"<span style='color:#94a3b8; margin-left:10px'>{conf:.1%} confidence</span>"
        f"</div>", unsafe_allow_html=True)

    col_face, col_probs, col_cam = st.columns([1, 2, 1])

    with col_face:
        st.markdown("**Cropped face**")
        face_pil = Image.fromarray(np.uint8(r['face'] * 255))
        face_pil = face_pil.resize((120, 120), Image.NEAREST)
        st.image(face_pil)

    with col_probs:
        st.markdown("**Our model — probabilities**")
        fig = prob_bar_chart(r['probs'], highlight=emotion)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_cam:
        if show_gradcam:
            st.markdown("**Grad-CAM**")
            try:
                heatmap = compute_gradcam(model, r['face'], r['probs'].argmax())
                overlay = overlay_gradcam(r['face'], heatmap)
                st.image(overlay, use_container_width=True)
                st.caption("Red = attention focus")
            except Exception:
                st.caption("Grad-CAM unavailable")

    if show_df and r['bgr_crop'].size > 0:
        with st.spinner("Running DeepFace…"):
            df_em, df_conf, df_probs = deepface_predict(r['bgr_crop'])
        st.markdown("**DeepFace prediction**")
        df_hex = EMOTION_COLORS.get(df_em, '#94a3b8')
        st.markdown(
            f"<span class='badge' style='background:{df_hex}'>"
            f"{df_em.upper()}  {df_conf:.1%}</span>",
            unsafe_allow_html=True)
        fig2 = prob_bar_chart(df_probs, title='DeepFace', highlight=df_em)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

        agree = (df_em == emotion)
        if agree:
            st.success("Both models agree ✓")
        else:
            st.warning(f"Models disagree — ours: **{emotion}**, DeepFace: **{df_em}**")


# ── Main header ───────────────────────────────────────────────────────────────

st.markdown("<div class='main-title'>🎭 Emotion Detection System</div>",
            unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Real-time facial emotion recognition · "
            "RAF-DB + FER2013 · 7 emotions · Deep CNN + TTA Ensemble</div>",
            unsafe_allow_html=True)

tabs = st.tabs([
    "📷  Live Demo",
    "🎬  Video Analysis",
    "📊  Model Comparison",
    "📈  Training Curves",
    "ℹ️  About",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Demo
# ════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("### Detect emotions from an image or webcam snapshot")

    input_mode = st.radio("Input source", ["Upload image", "Webcam snapshot"],
                          horizontal=True)

    frame_bgr = None

    if input_mode == "Upload image":
        uploaded = st.file_uploader("Drop an image here",
                                    type=['jpg', 'jpeg', 'png', 'webp', 'bmp'])
        if uploaded:
            pil_img   = Image.open(uploaded).convert('RGB')
            frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        snap = st.camera_input("Take a snapshot")
        if snap:
            pil_img   = Image.open(snap).convert('RGB')
            frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    if frame_bgr is not None:
        with st.spinner("Detecting faces…"):
            results = run_detection(frame_bgr)

        if not results:
            st.warning("No faces detected. Try a clearer, well-lit photo "
                       "with a visible frontal face.")
        else:
            st.success(f"Found **{len(results)}** face(s)")
            annotated = annotate_frame(frame_bgr, results)
            col_img, col_pad = st.columns([2, 1])
            with col_img:
                st.image(annotated, caption="Detected faces", use_container_width=True)
            st.markdown("---")
            for i, r in enumerate(results):
                st.markdown(f"#### Face {i+1}")
                render_face_card(r, i, show_df=show_deepface)
                if i < len(results) - 1:
                    st.markdown("---")
    else:
        # Placeholder tip
        st.info("👆 Upload a photo or take a webcam snapshot to analyse emotions.")
        em_cols = st.columns(7)
        for col, em in zip(em_cols, EMOTIONS):
            col.markdown(
                f"<div style='background:{EMOTION_COLORS[em]};border-radius:8px;"
                f"padding:8px;text-align:center;color:white;font-weight:700;"
                f"font-size:.8rem'>{em}</div>",
                unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Video Analysis
# ════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### Emotion analysis from video")
    st.caption("Upload any video file (mp4, avi, mov, mkv). "
               "Frames are sampled at the chosen rate and each face is analysed.")

    video_file  = st.file_uploader("Upload a video",
                                   type=['mp4', 'avi', 'mov', 'mkv', 'webm'])
    sample_rate = st.slider("Sample every N frames", 1, 30, 5,
                            help="Lower = more detail but slower processing")

    if video_file and st.button("▶  Analyse video", type="primary"):
        # Save upload to a temp file (OpenCV needs a path)
        suffix = '.' + video_file.name.rsplit('.', 1)[-1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name

        cap     = cv2.VideoCapture(tmp_path)
        fps     = cap.get(cv2.CAP_PROP_FPS) or 25
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        timestamps, probs_over_time, dominant_over_time = [], [], []
        frame_idx = 0

        progress = st.progress(0, text="Processing video…")
        preview_slot = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % sample_rate != 0:
                continue

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces_haar(gray)

            if len(faces) > 0:
                face_probs = []
                for (x, y, w, h) in faces:
                    fa = crop_face(gray, x, y, w, h)
                    _, _, p = predict_emotion(model, fa)
                    face_probs.append(p)
                avg_probs = np.mean(face_probs, axis=0)
            else:
                avg_probs = np.ones(7) / 7   # no face → flat distribution

            timestamps.append(frame_idx / fps)
            probs_over_time.append(avg_probs)
            dominant_over_time.append(EMOTIONS[avg_probs.argmax()])

            pct = min(frame_idx / max(total_f, 1), 1.0)
            progress.progress(pct, text=f"Frame {frame_idx} / {total_f}  |  "
                                        f"Detected {len(faces)} face(s)")

            # Show an annotated preview every 30 sampled frames
            if len(timestamps) % 30 == 0:
                annotated = annotate_frame(frame, [
                    {'box': (int(x),int(y),int(w),int(h)),
                     'emotion': EMOTIONS[avg_probs.argmax()],
                     'confidence': float(avg_probs.max())}
                    for (x,y,w,h) in faces
                ])
                preview_slot.image(annotated, caption=f"t = {timestamps[-1]:.1f}s",
                                   use_container_width=True)

        cap.release()
        os.remove(tmp_path)
        progress.empty()
        preview_slot.empty()

        if not timestamps:
            st.error("Could not read any frames from the video.")
        else:
            times     = np.array(timestamps)
            probs_arr = np.array(probs_over_time)

            from collections import Counter
            counts   = Counter(dominant_over_time)
            total_s  = len(dominant_over_time)
            dominant = counts.most_common(1)[0][0]
            pos_pct  = (counts.get('happy', 0) + counts.get('surprise', 0)) / total_s * 100
            neg_pct  = sum(counts.get(e, 0) for e in
                           ['angry', 'disgust', 'fear', 'sad']) / total_s * 100

            # Summary metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"<div class='metric-box'><div class='val'>{dominant.upper()}</div>"
                        f"<div class='lbl'>Dominant emotion</div></div>",
                        unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-box'><div class='val'>{pos_pct:.1f}%</div>"
                        f"<div class='lbl'>Positive frames</div></div>",
                        unsafe_allow_html=True)
            c3.markdown(f"<div class='metric-box'><div class='val'>{neg_pct:.1f}%</div>"
                        f"<div class='lbl'>Negative frames</div></div>",
                        unsafe_allow_html=True)
            c4.markdown(f"<div class='metric-box'><div class='val'>{len(timestamps):,}</div>"
                        f"<div class='lbl'>Frames analysed</div></div>",
                        unsafe_allow_html=True)

            st.markdown("---")

            # ── Stacked area timeline ─────────────────────────────────────
            st.markdown("#### Emotion timeline")
            fig, ax = plt.subplots(figsize=(12, 4))
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#1e293b')
            ax.stackplot(times,
                         [probs_arr[:, i] for i in range(7)],
                         labels=EMOTIONS,
                         colors=[EMOTION_COLORS[e] for e in EMOTIONS],
                         alpha=0.85)
            ax.set_xlabel('Time (s)', color='#94a3b8')
            ax.set_ylabel('Probability', color='#94a3b8')
            ax.set_xlim(times[0], times[-1])
            ax.tick_params(colors='#94a3b8', labelsize=8)
            ax.spines[:].set_color('#334155')
            ax.legend(loc='upper right', ncol=4, fontsize=8,
                      facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0')
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # ── Per-emotion bar + pie ─────────────────────────────────────
            st.markdown("#### Emotion distribution")
            col_bar, col_pie = st.columns(2)

            with col_bar:
                fig2, ax2 = plt.subplots(figsize=(5, 3.5))
                fig2.patch.set_facecolor('#0f172a')
                ax2.set_facecolor('#1e293b')
                em_pcts = [counts.get(e, 0) / total_s * 100 for e in EMOTIONS]
                bars = ax2.barh(EMOTIONS, em_pcts,
                                color=[EMOTION_COLORS[e] for e in EMOTIONS],
                                edgecolor='none', height=0.6)
                for bar, v in zip(bars, em_pcts):
                    ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                             f'{v:.1f}%', va='center', fontsize=8, color='#e2e8f0')
                ax2.set_xlabel('% of frames', color='#94a3b8', fontsize=8)
                ax2.tick_params(colors='#94a3b8', labelsize=8)
                ax2.spines[:].set_visible(False)
                ax2.invert_yaxis()
                ax2.set_xlim(0, max(em_pcts) * 1.25 + 1)
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)

            with col_pie:
                fig3, ax3 = plt.subplots(figsize=(4, 3.5))
                fig3.patch.set_facecolor('#0f172a')
                sizes  = [counts.get(e, 0) for e in EMOTIONS]
                colors = [EMOTION_COLORS[e] for e in EMOTIONS]
                wedges, texts, autotexts = ax3.pie(
                    sizes, labels=EMOTIONS, colors=colors,
                    autopct=lambda p: f'{p:.1f}%' if p > 2 else '',
                    startangle=140, textprops={'color': '#e2e8f0', 'fontsize': 8})
                for at in autotexts:
                    at.set_fontsize(7)
                ax3.set_facecolor('#0f172a')
                st.pyplot(fig3, use_container_width=True)
                plt.close(fig3)

            # ── Frame-level dominant emotion over time ────────────────────
            st.markdown("#### Dominant emotion over time")
            fig4, ax4 = plt.subplots(figsize=(12, 1.8))
            fig4.patch.set_facecolor('#0f172a')
            ax4.set_facecolor('#1e293b')
            for t, em in zip(times, dominant_over_time):
                ax4.axvspan(t, t + (times[1]-times[0] if len(times)>1 else 1),
                            color=EMOTION_COLORS[em], alpha=0.8)
            ax4.set_xlim(times[0], times[-1])
            ax4.set_xlabel('Time (s)', color='#94a3b8', fontsize=8)
            ax4.set_yticks([])
            ax4.spines[:].set_color('#334155')
            ax4.tick_params(colors='#94a3b8', labelsize=7)
            legend_patches = [mpatches.Patch(color=EMOTION_COLORS[e], label=e)
                              for e in EMOTIONS]
            ax4.legend(handles=legend_patches, loc='upper right', ncol=7,
                       fontsize=7, facecolor='#1e293b', edgecolor='#334155',
                       labelcolor='#e2e8f0')
            st.pyplot(fig4, use_container_width=True)
            plt.close(fig4)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Comparison
# ════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### Model accuracy benchmark")
    st.caption("All custom models evaluated with 8-pass Test-Time Augmentation (TTA). "
               "DeepFace evaluated on a 500-image random sample of the test set.")

    # ── Accuracy bar chart ────────────────────────────────────────────────────
    labels = list(BENCHMARK.keys())
    accs   = [v / 100 for v in BENCHMARK.values()]
    bar_colors = ['#64748b', '#6366f1', '#6366f1', '#6366f1', '#6366f1', '#f59e0b']

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    bars = ax.bar(labels, [v * 100 for v in accs], color=bar_colors,
                  edgecolor='none', width=0.55)
    # Highlight best
    bars[-1].set_color('#f59e0b')
    bars[0].set_color('#64748b')

    ax.axhline(52.80, color='#64748b', linestyle='--', linewidth=1,
               alpha=0.6, label='DeepFace baseline')
    ax.axhline(70.45, color='#f59e0b', linestyle='--', linewidth=1,
               alpha=0.6, label='Our best ensemble')

    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f'{v*100:.2f}%', ha='center', va='bottom',
                fontsize=9, color='#e2e8f0', fontweight='bold')

    ax.set_ylim(40, 78)
    ax.set_ylabel('Test Accuracy (%)', color='#94a3b8')
    ax.tick_params(colors='#94a3b8', labelsize=8)
    ax.spines[:].set_visible(False)
    ax.set_facecolor('#1e293b')
    fig.patch.set_facecolor('#0f172a')

    legend = ax.legend(fontsize=8, facecolor='#1e293b',
                       edgecolor='#334155', labelcolor='#e2e8f0')
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("<div class='metric-box'><div class='val'>70.45%</div>"
                "<div class='lbl'>Best ensemble accuracy</div></div>",
                unsafe_allow_html=True)
    c2.markdown("<div class='metric-box'><div class='val'>+17.65%</div>"
                "<div class='lbl'>vs DeepFace pretrained</div></div>",
                unsafe_allow_html=True)
    c3.markdown("<div class='metric-box'><div class='val'>4 models</div>"
                "<div class='lbl'>in best ensemble</div></div>",
                unsafe_allow_html=True)
    c4.markdown("<div class='metric-box'><div class='val'>8 passes</div>"
                "<div class='lbl'>TTA augmentation</div></div>",
                unsafe_allow_html=True)

    # ── Data table ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Per-model breakdown")
    df_bench = pd.DataFrame({
        'Model':       [l.replace('\n', ' ') for l in labels],
        'Accuracy (%)': [round(v, 2) for v in BENCHMARK.values()],
        'Gap vs DeepFace': [round(v - 52.80, 2) for v in BENCHMARK.values()],
    })
    df_bench = df_bench.set_index('Model')
    st.dataframe(
        df_bench.style
            .background_gradient(subset=['Accuracy (%)'], cmap='YlGn', vmin=45, vmax=75)
            .format({'Accuracy (%)': '{:.2f}%', 'Gap vs DeepFace': '{:+.2f}%'}),
        use_container_width=True)

    st.markdown("---")
    st.markdown("### About DeepFace")
    st.markdown("""
DeepFace uses a **mini-XCEPTION** model trained on the original FER2013 dataset.
It runs without any task-specific fine-tuning on our RAF-DB test set, which explains the
lower accuracy. Our models were trained directly on the same distribution, giving them a
significant advantage — but it's still a meaningful baseline since DeepFace requires
**zero training effort**.

| | DeepFace | Our Ensemble |
|--|--|--|
| Training required | ❌ None | ✅ 100+ epochs |
| Dataset | FER2013 | RAF-DB + FER2013 + extra |
| Test accuracy | 52.80% | **70.45%** |
| TTA | — | 8 passes |
| Real-time capable | ✅ | ✅ |
""")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Training Curves
# ════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### Training history")

    def load_csv(path):
        full = os.path.join(ROOT, path)
        if not os.path.exists(full):
            return None
        return pd.read_csv(full)

    v1 = load_csv('logs/custom_cnn_history.csv')
    v2 = load_csv('logs/custom_cnn_v2_history.csv')

    def plot_history(df, title, color_train='#6366f1', color_val='#f59e0b'):
        if df is None:
            return None
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor('#0f172a')
        for ax in (ax1, ax2):
            ax.set_facecolor('#1e293b')
            ax.tick_params(colors='#94a3b8', labelsize=8)
            ax.spines[:].set_color('#334155')

        epochs = range(1, len(df) + 1)

        ax1.plot(epochs, df['accuracy'],     color=color_train, lw=2, label='Train acc')
        ax1.plot(epochs, df['val_accuracy'], color=color_val,   lw=2,
                 linestyle='--', label='Val acc')
        ax1.set_title(f'{title} — Accuracy', color='#e2e8f0', fontsize=10)
        ax1.set_xlabel('Epoch', color='#94a3b8')
        ax1.set_ylabel('Accuracy', color='#94a3b8')
        ax1.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0',
                   fontsize=8)
        best_val = df['val_accuracy'].max()
        best_ep  = df['val_accuracy'].idxmax() + 1
        ax1.axhline(best_val, color=color_val, linestyle=':', alpha=0.5)
        ax1.text(0.98, best_val + 0.003, f'best={best_val:.4f} (ep{best_ep})',
                 ha='right', va='bottom', color=color_val, fontsize=7,
                 transform=ax1.get_yaxis_transform())

        ax2.plot(epochs, df['loss'],     color=color_train, lw=2, label='Train loss')
        ax2.plot(epochs, df['val_loss'], color=color_val,   lw=2,
                 linestyle='--', label='Val loss')
        ax2.set_title(f'{title} — Loss', color='#e2e8f0', fontsize=10)
        ax2.set_xlabel('Epoch', color='#94a3b8')
        ax2.set_ylabel('Loss', color='#94a3b8')
        ax2.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0',
                   fontsize=8)

        plt.tight_layout(pad=1.0)
        return fig

    if v1 is not None:
        st.markdown("#### Custom CNN v1 — 100 epochs from scratch")
        st.caption(f"Final val accuracy: **{v1['val_accuracy'].max():.4f}**  "
                   f"| Best epoch: {int(v1['val_accuracy'].idxmax()) + 1}")
        fig = plot_history(v1, 'Custom CNN v1')
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("---")

    if v2 is not None:
        st.markdown("#### Custom CNN v2 — 30-epoch fine-tune from v1 checkpoint")
        st.caption(f"Final val accuracy: **{v2['val_accuracy'].max():.4f}**  "
                   f"| Best epoch: {int(v2['val_accuracy'].idxmax()) + 1}  "
                   f"| LR: 2e-4 → 0 (CosineDecay)")
        fig = plot_history(v2, 'Custom CNN v2', '#10b981', '#f59e0b')
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("---")
    st.markdown("#### MobileNetV2 — Transfer learning curves")
    mob_head = load_csv('logs/mobilenetv2_head_history.csv')
    mob_fine = load_csv('logs/mobilenetv2_fine_history.csv')

    c1, c2 = st.columns(2)
    with c1:
        if mob_head is not None:
            st.markdown("**Head training (top layers only)**")
            fig = plot_history(mob_head, 'MobileNetV2 head', '#ec4899', '#f59e0b')
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
    with c2:
        if mob_fine is not None:
            st.markdown("**Full fine-tune (all layers)**")
            fig = plot_history(mob_fine, 'MobileNetV2 fine', '#ec4899', '#f59e0b')
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.markdown("---")
    st.markdown("#### Key observations")
    st.markdown("""
- **Augmentation bug** (discovered at epoch 0): `RandomBrightness` + `RandomContrast`
  applied after per-image standardisation destroyed the training signal → model stuck at
  **14.3% (random baseline)**. Removing colour augmentations fixed it immediately.
- **Custom CNN v2** fine-tunes from v1's best checkpoint with 10× lower LR (2e-4 vs 2e-3),
  squeezing out an additional ~2% accuracy before applying SWA.
- **Stochastic Weight Averaging** (8 post-training epochs) further improves generalisation
  by averaging weights from the loss landscape flat region.
- **TTA** (8 random augmentation passes at test time) consistently adds **+1–2.5%** on top
  of each model's single-pass accuracy.
""")


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — About
# ════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("### Real-Time Emotion Detection System")
    st.markdown(
        "A full deep learning pipeline built for a university AI / Computer Vision course — "
        "targeting the highest possible accuracy on 7-class facial emotion recognition.")

    st.markdown("---")

    # Emotion grid
    st.markdown("#### 7 Emotions Detected")
    em_cols = st.columns(7)
    for col, em in zip(em_cols, EMOTIONS):
        col.markdown(
            f"<div style='background:{EMOTION_COLORS[em]};border-radius:10px;"
            f"padding:12px 4px;text-align:center;color:white;font-weight:700;"
            f"font-size:.85rem;margin:2px'>{em.upper()}</div>",
            unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Architecture — Custom CNN")
        st.markdown("""
- **Multi-scale** convolutional feature extraction (3 branches)
- **Squeeze-and-Excitation** (SE) attention blocks
- **Residual** connections throughout
- **Batch Normalisation** + He initialisation
- ~3.2M parameters
- Input: 48 × 48 × 1 (grayscale)
- Output: softmax over 7 emotions
""")
        st.markdown("#### Training techniques")
        st.markdown("""
- CosineDecay LR schedule (1e-3 → ~0)
- Adam optimizer + gradient clipping (clipnorm=1.0)
- No colour augmentation (incompatible with per-image standardisation)
- Stochastic Weight Averaging (SWA)
- Test-Time Augmentation (TTA, 8 passes)
- 4-model soft-voting ensemble
""")

    with col_b:
        st.markdown("#### Dataset")
        st.markdown("""
| Dataset | Split | Images |
|---------|-------|--------|
| RAF-DB  | Train | ~12,000 |
| FER2013 | Train | ~28,000 |
| Extra   | Train | ~2,700 |
| **Total train** | | **~84,500** |
| Val  | | 4,940 |
| **Test** | | **7,178** |

Minority classes oversampled to **12,000** per class (angry, disgust, fear, sad, surprise, neutral).
""")
        st.markdown("#### Preprocessing")
        st.markdown("""
1. Convert to grayscale
2. Face detection (Haar cascade / MTCNN)
3. Crop + resize to 48 × 48
4. Per-image standardisation (zero-mean, unit-variance → [0, 1])
""")

    st.markdown("---")
    st.markdown("#### How to run")
    st.code("""
# Train custom CNN (100 epochs)
python3 training/train.py

# Fine-tune from best checkpoint (+30 epochs + SWA)
python3 finetune_cnn.py

# Full evaluation: all models + TTA + ensemble + DeepFace
python3 evaluate_final.py

# Real-time webcam demo (Grad-CAM + timeline)
python3 realtime/realtime_webcam.py

# Real-time with DeepFace side-by-side comparison
python3 realtime/realtime_webcam.py --compare

# This web app
streamlit run app/app.py
""", language='bash')

    st.markdown("---")
    st.markdown("#### Results summary")

    res_data = {
        'Model': [l.replace('\n', ' ') for l in BENCHMARK.keys()],
        'Accuracy': [f"{v:.2f}%" for v in BENCHMARK.values()],
        'Method': ['Zero-shot pretrained', 'TTA only', 'TTA only',
                   'Fine-tune + TTA', 'SWA + TTA', '4-model soft-voting + TTA'],
    }
    st.dataframe(pd.DataFrame(res_data).set_index('Model'),
                 use_container_width=True)
