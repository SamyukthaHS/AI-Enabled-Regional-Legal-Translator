# ============================
# Load model & tokenizer
# ============================
model_name = "ai4bharat/indictrans2-en-indic-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)

# ============================
# Translation functions
# ============================
def translate(sentence, src_lang="eng_Latn", tgt_lang="kan_Knda"):
    if not sentence:
        return "", "", None, None

    formatted_text = f"{src_lang} {tgt_lang} {sentence}"
    inputs = tokenizer(formatted_text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs, max_length=256, num_beams=5, use_cache=False)

    dev = tokenizer.decode(outputs[0], skip_special_tokens=True)
    kn = UnicodeIndicTransliterator.transliterate(dev, "hi", "kn")

    # TTS
    audio_file = "kannada_audio.mp3"
    tts = gTTS(text=kn, lang="kn")
    tts.save(audio_file)

    # Download .txt
    txt_file = "Kannada_Translation.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(kn)

    return dev, kn, audio_file, txt_file


def translate_file(file):
    if not file:
        return "", "", None, None

    if file.name.endswith(".txt"):
        text = file.read().decode("utf-8")
    elif file.name.endswith(".docx"):
        doc = Document(file.name)
        text = "\n".join(p.text for p in doc.paragraphs)
    else:
        return "Unsupported file!", "", None, None

    return translate(text)

# ============================
# CSS
# ============================
custom_css = """
body {
    background: url('https://images.unsplash.com/photo-1573497019940-1c28c88b4f63?w=1400')
    no-repeat center center fixed;
    background-size: cover;
    font-family: 'Segoe UI';
}

.gradio-container {
    background-color: rgba(255, 255, 255, 0.88) !important;
    border-radius: 15px;
    padding: 25px;
}

/* Brown Subtitle Bar (Restored) */
.subtitle-bar {
    background-color: #8B4513;
    color: white;
    padding: 8px;
    border-radius: 8px;
    text-align: center;
    font-size: 22px;
    margin-top: 0px;
    margin-bottom: 15px;
}

/* Label Styling */
#english_box label,
#dev_box label,
#kn_box label {
    font-weight: 800 !important;
    color: #e65100 !important;
    font-size: 18px !important;
}
#english_box label span,
#dev_box label span,
#kn_box label span {
    font-weight: 800 !important;
    color: #e65100 !important;
}

/* Textbox border + background */
#english_box textarea,
#dev_box textarea,
#kn_box textarea,
#file_box input {
    border: 2px solid #004d99 !important;
    border-radius: 8px !important;
    background-color: #f2f7ff !important;
    color: black ;
}

/* Buttons */
.animate-btn button {
    background-color: #004d99 !important;
    color: white !important;
    font-size: 18px !important;
    border-radius: 10px !important;
    font-weight: bold;
    transition: 0.2s;
}
.animate-btn button:hover {
    background-color: #0066cc !important;
    transform: scale(1.07);
}

/* Project cards */
.name-card {
    width: 48%;
    background: white;
    padding: 10px;
    border-radius: 8px;
    border: 2px solid #004d99;
    text-align: center;
    color: black;
}
"""

# ============================
# UI
# ============================
with gr.Blocks(css=custom_css) as demo:

    # ✅ New Blue Banner Header
    gr.HTML("""
    <div style="
        background: linear-gradient(to right, #0A2342, #0A2342);
        padding: 16px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 10px rgba(0,0,0,0.25);
        color: white;
        margin-bottom: 0px;
    ">
        <img src="/content/content/images.jpg" width="95" style="border-radius:50px;">

        <div style="text-align:center;">
            <h1 style="margin:0; font-size: 36px; font-weight:800; color:#E8E8E8;">
                MALNAD COLLEGE OF ENGINEERING
            </h1>
            <h2 style="margin:0; font-size:22px; font-weight:700; color:#CFCFCF;">
                CSE (Artificial Intelligence and Machine Learning)
            </h2>
        </div>

       <img src="https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.igauge.in%2Finstitution%2Fcolleges%2Fmalnad-college-of-engineering&psig=AOvVaw3emV08RUCIljplO_QMnbBT&ust=1763574699916000&source=images&cd=vfe&opi=89978449&ved=0CBUQjRxqFwoTCLimn9mh_ZADFQAAAAAdAAAAABAL" width="95">
    </div>
    """
    )

    # ✅ Brown subtitle restored
    gr.HTML("<div class='subtitle-bar'>⚖️ AI Enabled Regional Legal Translator</div>")

    with gr.Tab("✍️ Enter Text"):
        english = gr.Textbox(label="✍️ Enter English Legal Text", lines=7, elem_id="english_box")

    with gr.Tab("📄 Upload .txt / .docx"):
        file_in = gr.File(label="Choose File", elem_id="file_box")

    with gr.Row():
        dev_out = gr.Textbox(label="🔠 Devanagari Output", lines=7, elem_id="dev_box")
        kn_out = gr.Textbox(label="📝 Kannada Output", lines=7, elem_id="kn_box")

    audio = gr.Audio(label="🔊 Listen Kannada Speech")
    download_file = gr.File(label="📥 Download Kannada Output (.txt)")

    with gr.Row():
        translate_btn = gr.Button("Translate ✅", elem_classes=["animate-btn"])
        translate_file_btn = gr.Button("Translate Uploaded File 📑", elem_classes=["animate-btn"])

    translate_btn.click(translate, inputs=english, outputs=[dev_out, kn_out, audio, download_file])
    translate_file_btn.click(translate_file, inputs=file_in, outputs=[dev_out, kn_out, audio, download_file])

    # Project By
    gr.HTML("""
    <h3 style="text-align:center; margin-top:25px;">📌 Project By</h3>
    <div style="display:flex; justify-content:space-between;">
        <div class='name-card'>✅ Nithyashree CP<br>✅ Samyuktha HS</div>
        <div class='name-card'>✅ Archana K<br>✅ Avaneesh Honnappa</div>
    </div>
    """
    )

demo.launch()
