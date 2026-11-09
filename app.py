import streamlit as st

from rag import ask, load_index

st.set_page_config(page_title="MuseumMind", page_icon="🏛️", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Heebo:wght@400;600;800&display=swap');
html, body, .stApp { font-family:'Heebo',sans-serif; }
.block-container, [data-testid="stSidebarContent"] { direction:rtl; text-align:right; }
.stApp { background:linear-gradient(180deg,#1b1713 0%,#14110e 100%); }
#MainMenu, footer { visibility:hidden; }
header, [data-testid="stHeader"] { background:transparent !important; }
[data-testid="stSidebar"] *, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label { color:#f1e8d8 !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color:#b9ab94 !important; }
.stButton { width:100%; }
.stButton button { min-height:2.8rem; white-space:normal; }
[data-testid="stHorizontalBlock"] { direction:rtl; }
[data-testid="stSidebar"] { min-width:260px !important; max-width:300px !important; }
.block-container { max-width:760px; padding-top:2rem; }

.hero { text-align:center; padding:1.6rem 1rem 1.2rem; margin-bottom:1.2rem;
        border-radius:18px; background:radial-gradient(circle at 50% 0%,#3a2e1f,#201a14);
        border:1px solid #4a3b28; }
.hero h1 { font-weight:800; font-size:2.6rem; margin:0; color:#e9c46a; letter-spacing:1px; }
.hero p { color:#b9ab94; margin:.4rem 0 0; font-size:1.05rem; }

.stTextInput input { background:#241e17; color:#f1e8d8; border:1px solid #5a4930;
                     border-radius:12px; padding:.8rem 1rem; font-size:1.05rem; text-align:right; }
.stTextInput input:focus { border-color:#e9c46a; box-shadow:0 0 0 1px #e9c46a; }
.stTextInput label p { color:#d8c9ae; font-weight:600; }

.stButton button { width:100%; border-radius:999px; background:#2a2219; color:#e9c46a;
                   border:1px solid #5a4930; font-size:.9rem; }
.stButton button:hover { background:#3a2e1f; border-color:#e9c46a; color:#fff; }

.answer { background:#2a2219; border-right:5px solid #e9c46a; border-radius:12px;
          padding:1.1rem 1.3rem; color:#f6efe0; font-size:1.12rem; line-height:1.8;
          margin:.4rem 0 1.2rem; }
.noinfo { border-right-color:#c0584a; }
.section-title { display:block; direction:rtl; text-align:right; color:#e9c46a; font-weight:700; font-size:1.15rem; margin:1rem 0 .4rem; }

[data-testid="stExpander"] { background:#201a14; border:1px solid #4a3b28; border-radius:12px; }
[data-testid="stExpander"] summary, .stCheckbox, .stCaption { direction:rtl; }
.badge { background:#e9c46a; color:#1b1713; border-radius:999px; padding:1px 10px;
         font-size:.8rem; font-weight:700; margin-right:.5rem; }
[data-testid="stSidebar"] { background:#17130f; border-left:1px solid #4a3b28; }
</style>
<div class="hero">
  <h1>🏛️ MuseumMind</h1>
  <p>העוזר החכם של מוזיאון מסע בזמן · עונה רק מתוך מסמכי המוזיאון</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def get_index():
    return load_index()


try:
    index = get_index()
except Exception as e:
    st.error(str(e))
    st.stop()

with st.sidebar:
    st.markdown("### הגדרות")
    expand = st.checkbox("הרחבת שאלה בעזרת AI", value=False,
                         help="כבוי כברירת מחדל. הפעילי רק כשהשאלה קצרה או עמומה.")
    st.caption("חיפוש: Similarity · Top K = 5\n\nמודל: gpt-4o-mini")

EXAMPLES = ["מה שעות הפתיחה ביום שישי?", "איך טיפלו ביומן של נווט הזמן?",
            "כמה עולה כרטיס לסטודנט?", "מה קרה בקפסולת הזמן?"]
if "q" not in st.session_state:
    st.session_state.q = ""

st.markdown('<div class="section-title">שאלות לדוגמה</div>', unsafe_allow_html=True)
cols = st.columns(2)
for i, ex in enumerate(EXAMPLES):
    if cols[i % 2].button(ex, key=f"ex{i}"):
        st.session_state.q = ex

question = st.text_input("שאלי שאלה על המוזיאון:", key="q",
                         placeholder="למשל: מה שעות הפתיחה ביום שישי?")

if question:
    with st.spinner("מחפש במסמכי המוזיאון..."):
        res = ask(index, question, expand=expand)
    cls = "answer" if res["sources"] else "answer noinfo"
    st.markdown('<div class="section-title">תשובה</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="{cls}">{res["answer"]}</div>', unsafe_allow_html=True)
    if expand:
        st.caption(f"שאילתת החיפוש המורחבת: {res['search_query']}")
    if res["sources"]:
        st.markdown('<div class="section-title">📚 מקורות</div>', unsafe_allow_html=True)
        for s in res["sources"]:
            with st.expander(f"{s['file']}  ·  דמיון {s['score']}"):
                st.write(s["text"])
