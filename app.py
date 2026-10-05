import streamlit as st
from pathlib import Path
import time
from PIL import Image

from algorithms.template_matching import run_template_matching
from algorithms.viola_jones import detect_faces
from algorithms.deepface import analyze_face, verify_faces, deepface_available
from algorithms.facenet import generate_embedding, compare_embeddings, facenet_available

BASE = Path(__file__).parent
DEMO = BASE / "demo_images"

st.set_page_config(page_title="VISIONLAB", page_icon="👁", layout="wide", initial_sidebar_state="expanded")

# ---------- Global CSS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #080d18; color: #eef4ff; }
.block-container { padding: 2.2rem 3rem 3rem; max-width: 1450px; }
[data-testid="stSidebar"] { background: #0b1220; border-right: 1px solid #1c2a42; }
[data-testid="stSidebar"] > div:first-child { padding: 1.3rem 1rem; }
.vl-brand { padding: 10px 8px 24px; border-bottom:1px solid #1d2a40; margin-bottom:18px; }
.vl-eye { color:#39d9ff; font-size:26px; }
.vl-brand-title { font-size:20px; font-weight:800; letter-spacing:1.5px; }
.vl-brand-sub { color:#8292ad; font-size:11px; margin-top:3px; line-height:1.4; }
.hero { background: radial-gradient(circle at 82% 18%, rgba(61,220,255,.15), transparent 34%), linear-gradient(135deg,#0d1729,#09111f 62%,#10172b); border:1px solid #203451; border-radius:24px; padding:52px 54px; position:relative; overflow:hidden; }
.hero:after { content:""; position:absolute; width:310px; height:310px; right:-100px; bottom:-150px; border:1px solid rgba(94,102,255,.25); border-radius:50%; box-shadow:0 0 0 45px rgba(94,102,255,.035),0 0 0 90px rgba(94,102,255,.025); }
.eyebrow { color:#42dfff; font-size:12px; font-weight:700; letter-spacing:2px; text-transform:uppercase; }
.hero h1 { font-size:52px; line-height:1; margin:12px 0 12px; letter-spacing:-2px; }
.hero p { max-width:650px; color:#a7b6ca; font-size:16px; line-height:1.7; }
.section-title { font-size:24px; font-weight:750; margin:34px 0 18px; }
.page-title { font-size:38px; margin:0 0 6px; letter-spacing:-1px; }
.page-sub { color:#8798b2; margin-bottom:26px; }
.card { background:#0e1727; border:1px solid #1e3049; border-radius:18px; padding:22px; height:100%; transition:.2s; }
.card:hover { border-color:#2b91b4; transform:translateY(-2px); }
.card h3 { margin:8px 0 7px; font-size:18px; }
.card p { color:#8798b2; font-size:13px; line-height:1.55; }
.icon { font-size:24px; color:#42dfff; }
.metric { background:#0d1727; border:1px solid #20334e; border-radius:16px; padding:18px 20px; }
.metric-label { color:#7f91ad; font-size:11px; text-transform:uppercase; letter-spacing:1.3px; }
.metric-value { font-size:25px; font-weight:750; margin-top:5px; color:#f4f8ff; }
.process { background:#0b1422; border:1px solid #1d2c44; border-radius:16px; padding:18px; text-align:center; }
.process .num { color:#42dfff; font-size:12px; font-weight:800; }
.process .name { font-weight:650; margin:7px 0 4px; }
.process .desc { color:#788ba7; font-size:12px; line-height:1.45; }
.small-muted { color:#71829b; font-size:12px; }
div.stButton > button { border-radius:10px; border:1px solid #2a4764; background:#122138; color:#eaf5ff; font-weight:600; min-height:42px; }
div.stButton > button:hover { border-color:#42dfff; color:#42dfff; }
div.stButton > button[kind="primary"] { background:#12384a; border-color:#2b9bbd; color:#eaffff; }
.stFileUploader { background:#0c1524; border-radius:14px; border:1px dashed #2a405d; }
hr { border-color:#1d2b42; }
.footer { margin-top:55px; padding-top:20px; border-top:1px solid #1b293f; color:#63758f; font-size:11px; display:flex; justify-content:space-between; }
[data-testid="stMetric"] { background:#0d1727; border:1px solid #20334e; padding:12px; border-radius:14px; }
</style>
""", unsafe_allow_html=True)

# ---------- Navigation ----------
if "page" not in st.session_state:
    st.session_state.page = "Home"

pages = ["Home", "Template Matching", "Viola-Jones", "DeepFace", "FaceNet", "Demo Gallery", "About"]
icons = {"Home":"⌂","Template Matching":"⌗","Viola-Jones":"◎","DeepFace":"◉","FaceNet":"◇","Demo Gallery":"▦","About":"ⓘ"}

with st.sidebar:
    st.markdown('<div class="vl-brand"><span class="vl-eye">◉</span><br><span class="vl-brand-title">VISIONLAB</span><div class="vl-brand-sub">Computer Vision<br>Algorithm Laboratory</div></div>', unsafe_allow_html=True)
    for p in pages:
        if st.button(f"{icons[p]}   {p}", key=f"nav_{p}", use_container_width=True):
            st.session_state.page = p
            st.rerun()
    st.markdown("<div style='height:90px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="small-muted">VISIONLAB<br>Computer Vision Laboratory</div>', unsafe_allow_html=True)

def header(title, subtitle):
    st.markdown(f'<div class="eyebrow">COMPUTER VISION LABORATORY</div><div class="page-title">{title}</div><div class="page-sub">{subtitle}</div>', unsafe_allow_html=True)

def metric_card(label, value):
    st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

def img_bytes(upload):
    return upload.getvalue() if upload else None

def navigate(page):
    st.session_state.page = page
    st.rerun()

# ---------- Home ----------
def home():
    st.markdown("""<div class="hero">
    <div class="eyebrow">INTERACTIVE COMPUTER VISION LABORATORY</div>
    <h1>VISIONLAB</h1>
    <div style="font-size:20px;font-weight:600;color:#d9e7f7;margin-bottom:10px">Computer Vision Algorithm Laboratory</div>
    <p>Explore, visualize and understand classical and deep-learning-based computer vision algorithms through interactive demonstrations.</p>
    </div>""", unsafe_allow_html=True)
    st.write("")
    c1,c2,_ = st.columns([1,1,4])
    with c1:
        if st.button("Explore Algorithms", type="primary", use_container_width=True): navigate("Template Matching")
    with c2:
        if st.button("Try Demo", use_container_width=True): navigate("Demo Gallery")

    st.markdown('<div class="section-title">Explore the Laboratory</div>', unsafe_allow_html=True)
    cards = [
        ("⌗","Template Matching","Locate a specific template or object inside a larger image.","Template Matching"),
        ("◎","Viola-Jones","Fast classical face detection using Haar Cascade classifiers.","Viola-Jones"),
        ("◉","DeepFace","Deep-learning-based facial analysis and verification.","DeepFace"),
        ("◇","FaceNet","Represent faces as numerical embedding vectors.","FaceNet"),
    ]
    cols=st.columns(4)
    for col,(ic,name,desc,target) in zip(cols,cards):
        with col:
            st.markdown(f'<div class="card"><div class="icon">{ic}</div><h3>{name}</h3><p>{desc}</p></div>',unsafe_allow_html=True)
            if st.button("Explore →", key=f"home_{name}", use_container_width=True): navigate(target)

    st.markdown('<div class="section-title">Why VisionLab?</div>', unsafe_allow_html=True)
    reasons=[("▣","Visual Results","See algorithm output directly on images."),
             ("↗","Interactive","Upload your own images or use prepared demonstrations."),
             ("◇","Educational","Understand what happens behind each algorithm."),
             ("⇄","Comparison","Explore the progression from classical computer vision to deep learning.")]
    cols=st.columns(4)
    for col,(ic,t,d) in zip(cols,reasons):
        with col: st.markdown(f'<div class="card"><div class="icon">{ic}</div><h3>{t}</h3><p>{d}</p></div>',unsafe_allow_html=True)

# ---------- Template ----------
def template_page():
    header("Template Matching","Find a small template inside a larger image.")
    st.markdown("### Inputs")
    left,right=st.columns(2)
    with left:
        st.markdown("**Main Image**")
        main_upload=st.file_uploader("Upload main image",type=["jpg","jpeg","png"],key="tm_main")
        main_demo=st.checkbox("Use Demo Image",value=not bool(main_upload),key="tm_main_demo")
        main = DEMO/"template_main.jpg" if main_demo or not main_upload else None
    with right:
        st.markdown("**Template Image**")
        temp_upload=st.file_uploader("Upload template",type=["jpg","jpeg","png"],key="tm_temp")
        temp_demo=st.checkbox("Use Demo Template",value=not bool(temp_upload),key="tm_temp_demo")
        temp = DEMO/"template.jpg" if temp_demo or not temp_upload else None
    if main_upload: main = Image.open(main_upload).convert("RGB")
    if temp_upload: temp = Image.open(temp_upload).convert("RGB")

    if main is not None and temp is not None:
        c1,c2=st.columns(2)
        with c1: st.image(main,caption="Main Image",use_container_width=True)
        with c2: st.image(temp,caption="Template",use_container_width=True)
        if st.button("Run Template Matching",type="primary",use_container_width=True):
            with st.spinner("Running algorithm..."):
                result=run_template_matching(main,temp)
            if not result["ok"]:
                st.error(result["message"])
            else:
                st.markdown("### Detection Result")
                st.image(result["image"],caption="Best matching region",use_container_width=True)
                m=st.columns(3)
                with m[0]: metric_card("Similarity",f'{result["similarity"]*100:.2f}%')
                with m[1]: metric_card("Location",f'({result["location"][0]}, {result["location"][1]})')
                with m[2]: metric_card("Template Size",f'{result["template_size"][0]} × {result["template_size"][1]} px')
    st.markdown("### How It Works")
    cols=st.columns(3)
    for col,num,title,desc in zip(cols,[1,2,3],["Select Template","Compare","Locate"],
        ["Choose the object or pattern to search for.","OpenCV compares the template across the main image.","The highest similarity region is highlighted."]):
        with col: st.markdown(f'<div class="process"><div class="num">STEP {num}</div><div class="name">{title}</div><div class="desc">{desc}</div></div>',unsafe_allow_html=True)

# ---------- Viola ----------
def viola_page():
    header("Viola-Jones","Classical real-time face detection using Haar Cascade classifiers.")
    demo=st.selectbox("Choose Demo Image",["Single Face","Multiple Faces","Group"],key="vj_demo")
    upload=st.file_uploader("Upload image (optional)",type=["jpg","jpeg","png"],key="vj_upload")
    path={"Single Face":"single_face.jpg","Multiple Faces":"multiple_faces.jpg","Group":"group.jpg"}[demo]
    image=Image.open(upload).convert("RGB") if upload else Image.open(DEMO/path).convert("RGB")
    st.image(image,caption="Input Image",use_container_width=False,width=620)
    if st.button("Detect Faces",type="primary",use_container_width=True):
        with st.spinner("Running Haar Cascade detector..."):
            r=detect_faces(image)
        if not r["ok"]: st.error(r["message"])
        else:
            c1,c2=st.columns(2)
            with c1: st.image(image,caption="Original Image",use_container_width=True)
            with c2: st.image(r["image"],caption="Detection Result",use_container_width=True)
            m=st.columns(3)
            with m[0]: metric_card("Faces Detected",str(r["face_count"]))
            with m[1]: metric_card("Processing Time",f'{r["time_ms"]:.1f} ms')
            with m[2]: metric_card("Image Resolution",f'{image.width} × {image.height}')
            if r["face_count"]==0: st.warning("No face detected in this image. Try another image.")
    st.markdown("### How Viola-Jones Works")
    cols=st.columns(5)
    steps=[("1","Haar-like Features"),("2","Integral Image"),("3","AdaBoost"),("4","Cascade Classifier"),("5","Face Detection")]
    for col,(n,t) in zip(cols,steps):
        with col: st.markdown(f'<div class="process"><div class="num">STEP {n}</div><div class="name">{t}</div></div>',unsafe_allow_html=True)

# ---------- DeepFace ----------
def deepface_page():
    header("DeepFace","Explore deep-learning-based facial analysis and verification.")
    op=st.radio("Operation",["Face Analysis","Face Verification"],horizontal=True)
    if op=="Face Analysis":
        upload=st.file_uploader("Upload Image",type=["jpg","jpeg","png"],key="df_one")
        demo=st.checkbox("Use Demo Face",value=not bool(upload),key="df_demo")
        image=Image.open(upload).convert("RGB") if upload else Image.open(DEMO/"single_face.jpg").convert("RGB")
        st.image(image,caption="Analysis Input",width=520)
        if st.button("Analyze Face",type="primary",use_container_width=True):
            with st.spinner("Analyzing image..."):
                r=analyze_face(image)
            if not r["ok"]:
                st.warning(r["message"])
            else:
                res=r["result"][0] if isinstance(r["result"],list) else r["result"]
                cols=st.columns(4)
                for col,key in zip(cols,["age","gender","dominant_emotion","dominant_race"]):
                    with col: metric_card(key.replace("_"," ").title(),str(res.get(key,"—")))
    else:
        a,b=st.columns(2)
        with a: up1=st.file_uploader("Image 1",type=["jpg","jpeg","png"],key="df_v1")
        with b: up2=st.file_uploader("Image 2",type=["jpg","jpeg","png"],key="df_v2")
        im1=Image.open(up1).convert("RGB") if up1 else Image.open(DEMO/"single_face.jpg").convert("RGB")
        im2=Image.open(up2).convert("RGB") if up2 else Image.open(DEMO/"single_face.jpg").convert("RGB")
        c1,c2=st.columns(2)
        with c1: st.image(im1,caption="Image 1",use_container_width=True)
        with c2: st.image(im2,caption="Image 2",use_container_width=True)
        if st.button("Verify Faces",type="primary",use_container_width=True):
            with st.spinner("Verifying faces..."):
                r=verify_faces(im1,im2)
            if not r["ok"]: st.warning(r["message"])
            else:
                verified=r["verified"]
                metric_card("Verification","VERIFIED" if verified else "NOT VERIFIED")
                st.caption(f"Distance: {r.get('distance','—')}  •  Threshold: {r.get('threshold','—')}")
    if not deepface_available():
        st.caption("DeepFace is optional and loaded only when this page needs it. If its model/dependency is unavailable, the page remains usable and reports the issue.")

# ---------- FaceNet ----------
def facenet_page():
    header("FaceNet","Represent faces as numerical embedding vectors.")
    st.markdown("Face → CNN → Embedding Vector → Similarity Comparison")
    op=st.radio("Mode",["Single Embedding","Compare Two Faces"],horizontal=True)
    if op=="Single Embedding":
        up=st.file_uploader("Image",type=["jpg","jpeg","png"],key="fn_one")
        image=Image.open(up).convert("RGB") if up else Image.open(DEMO/"single_face.jpg").convert("RGB")
        st.image(image,caption="Embedding Input",width=520)
        if st.button("Generate Embedding",type="primary",use_container_width=True):
            with st.spinner("Generating embedding..."):
                r=generate_embedding(image)
            if not r["ok"]: st.warning(r["message"])
            else:
                metric_card("Embedding Dimension",str(r["dimension"]))
                st.markdown("### Compact Embedding Visualization")
                st.line_chart(r["preview"])
    else:
        a,b=st.columns(2)
        with a: u1=st.file_uploader("Image 1",type=["jpg","jpeg","png"],key="fn1")
        with b: u2=st.file_uploader("Image 2",type=["jpg","jpeg","png"],key="fn2")
        im1=Image.open(u1).convert("RGB") if u1 else Image.open(DEMO/"single_face.jpg").convert("RGB")
        im2=Image.open(u2).convert("RGB") if u2 else Image.open(DEMO/"single_face.jpg").convert("RGB")
        c1,c2=st.columns(2)
        with c1: st.image(im1,caption="Image 1",use_container_width=True)
        with c2: st.image(im2,caption="Image 2",use_container_width=True)
        if st.button("Compare Embeddings",type="primary",use_container_width=True):
            with st.spinner("Comparing embeddings..."):
                r=compare_embeddings(im1,im2)
            if not r["ok"]: st.warning(r["message"])
            else:
                m=st.columns(3)
                with m[0]: metric_card("Similarity",f'{r["similarity"]:.4f}')
                with m[1]: metric_card("Distance",f'{r["distance"]:.4f}')
                with m[2]: metric_card("Match", "MATCH" if r["match"] else "NO MATCH")
    if not facenet_available():
        st.caption("FaceNet is implemented through the FaceNet-compatible Facenet512 model exposed by DeepFace. It is loaded on demand and reports model availability instead of crashing.")

# ---------- Gallery ----------
def gallery():
    header("Demo Gallery","Presentation-ready examples you can launch directly into each laboratory.")
    items=[
        ("single_face.jpg","Single Face","One clearly visible face for Viola-Jones and facial models.","Viola-Jones"),
        ("multiple_faces.jpg","Multiple Faces","Three separated faces for multi-face detection.","Viola-Jones"),
        ("group.jpg","Group","A four-face composite for detection demonstrations.","Viola-Jones"),
        ("template_main.jpg","Template Matching","A larger scene containing the exact template crop.","Template Matching"),
    ]
    cols=st.columns(2)
    for col,(fn,title,desc,target) in zip(cols*2,items):
        with col:
            st.markdown('<div class="card">',unsafe_allow_html=True)
            st.image(DEMO/fn,use_container_width=True)
            st.markdown(f"### {title}")
            st.markdown(f'<p>{desc}</p>',unsafe_allow_html=True)
            if st.button("Try Demo →",key=f"gal_{fn}",use_container_width=True): navigate(target)
            st.markdown('</div>',unsafe_allow_html=True)

# ---------- About ----------
def about():
    header("About VisionLab","An educational computer vision laboratory for practical algorithm exploration.")
    st.markdown("""<div class="card"><h3>What is VisionLab?</h3>
    <p style="font-size:15px">VisionLab is an educational computer vision laboratory designed to demonstrate how different computer vision techniques work on practical image-processing and facial-analysis tasks.</p></div>""",unsafe_allow_html=True)
    st.markdown("### Technology Stack")
    cols=st.columns(6)
    for col,t in zip(cols,["Python","OpenCV","NumPy","Streamlit","DeepFace","FaceNet"]):
        with col: st.markdown(f'<div class="process"><div class="name">{t}</div></div>',unsafe_allow_html=True)
    st.markdown("### Evolution of Computer Vision")
    cols=st.columns(4)
    for col,n,t in zip(cols,[1,2,3,4],["Template Matching","Viola-Jones","DeepFace","FaceNet"]):
        with col: st.markdown(f'<div class="process"><div class="num">0{n}</div><div class="name">{t}</div></div>',unsafe_allow_html=True)
    st.markdown("<p class='small-muted'>The project demonstrates a progression from classical image processing and object localization to face detection, deep facial analysis, and numerical face representation.</p>",unsafe_allow_html=True)

pages_fn={"Home":home,"Template Matching":template_page,"Viola-Jones":viola_page,"DeepFace":deepface_page,"FaceNet":facenet_page,"Demo Gallery":gallery,"About":about}
pages_fn[st.session_state.page]()
st.markdown('<div class="footer"><span>VISIONLAB • Computer Vision Algorithm Laboratory</span><span>Classical Vision → Deep Learning</span></div>',unsafe_allow_html=True)
