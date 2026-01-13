"""
🎬 AI Video Generator
אפליקציית Streamlit ליצירת וידאו עם VEO
"""

import streamlit as st
import time
import io
import os
from PIL import Image
from google import genai
from google.genai import types

# ==============================================================================
# הגדרות דף
# ==============================================================================

st.set_page_config(
    page_title="AI Video Generator",
    page_icon="🎬",
    layout="wide"
)

# ==============================================================================
# CSS מותאם
# ==============================================================================

st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #e7f3ff;
        border: 1px solid #b6d4fe;
        color: #084298;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# פונקציות עיבוד תמונה
# ==============================================================================

def process_image(uploaded_file, target_aspect_ratio: str = "16:9", method: str = "crop", crop_position: str = "center") -> bytes:
    """עיבוד תמונה - המרה, יחס, וגודל"""
    
    img = Image.open(uploaded_file)
    
    # המרה ל-RGB
    if img.mode in ('RGBA', 'P', 'LA', 'L'):
        if img.mode == 'LA' or img.mode == 'L':
            img = img.convert('RGB')
        else:
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'RGBA':
                background.paste(img, mask=img.split()[3])
            else:
                background.paste(img)
            img = background
    
    # יחס יעד
    target_ratio = 16/9 if target_aspect_ratio == "16:9" else 9/16
    current_ratio = img.width / img.height
    
    # התאמת יחס
    if abs(current_ratio - target_ratio) > 0.01:
        if method == "crop":
            if current_ratio > target_ratio:
                # התמונה רחבה מדי - חותכים מהצדדים
                new_width = int(img.height * target_ratio)
                left = (img.width - new_width) // 2  # תמיד מרכז לרוחב
                img = img.crop((left, 0, left + new_width, img.height))
            else:
                # התמונה גבוהה מדי - חותכים למעלה/למטה לפי בחירה
                new_height = int(img.width / target_ratio)
                if crop_position == "top":
                    top = 0
                elif crop_position == "bottom":
                    top = img.height - new_height
                else:  # center
                    top = (img.height - new_height) // 2
                img = img.crop((0, top, img.width, top + new_height))
        else:  # padding
            if current_ratio > target_ratio:
                new_height = int(img.width / target_ratio)
                new_img = Image.new('RGB', (img.width, new_height), (0, 0, 0))
                new_img.paste(img, (0, (new_height - img.height) // 2))
                img = new_img
            else:
                new_width = int(img.height * target_ratio)
                new_img = Image.new('RGB', (new_width, img.height), (0, 0, 0))
                new_img.paste(img, ((new_width - img.width) // 2, 0))
                img = new_img
    
    # הקטנה אם גדול מדי
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    size_mb = len(buffer.getvalue()) / (1024 * 1024)
    
    if size_mb > 7:
        scale = (7 / size_mb) ** 0.5
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # שמירה ל-bytes
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    
    return buffer.getvalue(), img.size


def bytes_to_genai_image(image_bytes: bytes) -> types.Image:
    """המרת bytes ל-Image של genai"""
    return types.Image(
        image_bytes=image_bytes,
        mime_type="image/jpeg"
    )

# ==============================================================================
# פונקציות יצירת וידאו
# ==============================================================================

def generate_video(client, params: dict, progress_placeholder):
    """יצירת וידאו עם עדכון התקדמות"""
    
    try:
        # הכנת התמונות
        config_params = {
            "aspect_ratio": params["aspect_ratio"],
            "duration_seconds": params["duration"],
        }
        
        if params["negative_prompt"]:
            config_params["negative_prompt"] = params["negative_prompt"]
        
        if params["resolution"] and params["model"] != "veo-2.0-generate-001":
            config_params["resolution"] = params["resolution"]
        
        # הכנת פרמטרים לפי מצב
        gen_params = {
            "model": params["model"],
            "prompt": params["prompt"],
        }
        
        # Image to Video
        if params["mode"] == "image_to_video" and params.get("start_image"):
            gen_params["image"] = bytes_to_genai_image(params["start_image"])
        
        # Interpolation (First + Last Frame)
        elif params["mode"] == "interpolation":
            if params.get("start_image") and params.get("end_image"):
                gen_params["image"] = bytes_to_genai_image(params["start_image"])
                config_params["last_frame"] = bytes_to_genai_image(params["end_image"])
                config_params["duration_seconds"] = 8  # חובה להיות 8 ב-interpolation
        
        gen_params["config"] = types.GenerateVideosConfig(**config_params)
        
        # שליחת הבקשה
        progress_placeholder.info("📤 שולח בקשה ל-API...")
        operation = client.models.generate_videos(**gen_params)
        
        # המתנה עם עדכון התקדמות
        start_time = time.time()
        while not operation.done:
            elapsed = int(time.time() - start_time)
            progress_placeholder.info(f"⏳ מייצר וידאו... ({elapsed} שניות)")
            time.sleep(10)
            operation = client.operations.get(operation)
        
        elapsed = int(time.time() - start_time)
        
        # בדיקת תוצאה
        if operation.response is None:
            return None, "❌ לא התקבלה תשובה מה-API"
        
        if operation.response.rai_media_filtered_reasons:
            reasons = operation.response.rai_media_filtered_reasons
            return None, f"❌ הוידאו נחסם: {reasons[0]}"
        
        if not operation.response.generated_videos:
            return None, "❌ לא נוצר וידאו"
        
        # הורדת הוידאו
        progress_placeholder.info("📥 מוריד וידאו...")
        video = operation.response.generated_videos[0]
        client.files.download(file=video.video)
        
        return video.video.video_bytes, f"✅ הוידאו נוצר בהצלחה! ({elapsed} שניות)"
        
    except Exception as e:
        return None, f"❌ שגיאה: {str(e)}"

# ==============================================================================
# ממשק משתמש
# ==============================================================================

def main():
    st.title("🎬 AI Video Generator")
    st.markdown("יצירת וידאו עם VEO של Google")
    
    # Sidebar - הגדרות
    with st.sidebar:
        st.header("⚙️ הגדרות")
        
        # API Key
        try:
            default_key = st.secrets.get("GOOGLE_API_KEY", "")
        except:
            default_key = ""
        api_key = st.text_input(
            "🔑 Google API Key",
            value=default_key,
            type="password",
            help="הכנס את מפתח ה-API של Google"
        )
        
        st.divider()
        
        # בחירת מודל
        st.subheader("📦 מודל")
        model = st.selectbox(
            "בחר מודל",
            options=[
                "veo-3.0-generate-001",
                "veo-3.1-generate-preview",
                "veo-3.0-fast-generate-001",
                "veo-3.1-fast-generate-preview",
                "veo-2.0-generate-001",
            ],
            index=0,
            help="VEO 3.1 תומך ב-Interpolation"
        )
        
        # בחירת מצב
        st.subheader("🎬 מצב יצירה")
        mode = st.selectbox(
            "בחר מצב",
            options=[
                ("text_to_video", "📝 Text to Video"),
                ("image_to_video", "🖼️ Image to Video"),
                ("interpolation", "🔄 First + Last Frame (VEO 3.1)"),
            ],
            format_func=lambda x: x[1],
            index=0
        )[0]
        
        # אזהרה על interpolation
        if mode == "interpolation" and "3.1" not in model:
            st.warning("⚠️ Interpolation דורש VEO 3.1!")
        
        st.divider()
        
        # פרמטרים
        st.subheader("🎛️ פרמטרים")
        
        aspect_ratio = st.selectbox(
            "יחס תמונה",
            options=["16:9", "9:16"],
            index=0
        )
        
        duration = st.selectbox(
            "אורך (שניות)",
            options=[4, 6, 8],
            index=2,
            disabled=(mode == "interpolation"),
            help="Interpolation תמיד 8 שניות"
        )
        if mode == "interpolation":
            duration = 8
        
        resolution = st.selectbox(
            "רזולוציה",
            options=["720p", "1080p"],
            index=0,
            help="1080p רק ל-8 שניות"
        )
        
        # שיטת עיבוד תמונה
        st.subheader("🖼️ עיבוד תמונה")
        image_method = st.radio(
            "התאמת יחס",
            options=["crop", "padding"],
            format_func=lambda x: "✂️ חיתוך" if x == "crop" else "📐 פסים שחורים",
            horizontal=True
        )
        
        # מיקום החיתוך
        if image_method == "crop":
            crop_position = st.select_slider(
                "מיקום חיתוך",
                options=["top", "center", "bottom"],
                value="center",
                format_func=lambda x: {"top": "⬆️ למעלה", "center": "⬌ מרכז", "bottom": "⬇️ למטה"}[x]
            )
        else:
            crop_position = "center"
    
    # אזור ראשי
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 פרומפט")
        prompt = st.text_area(
            "תאר את הוידאו",
            height=100,
            placeholder="A cinematic shot of a sunset over the ocean, with gentle waves..."
        )
        
        negative_prompt = st.text_input(
            "פרומפט שלילי (אופציונלי)",
            placeholder="blurry, low quality, distorted..."
        )
        
        # העלאת תמונות
        if mode in ["image_to_video", "interpolation"]:
            st.subheader("🖼️ תמונות")
            
            start_image_file = st.file_uploader(
                "תמונה התחלתית" if mode == "interpolation" else "תמונה",
                type=["png", "jpg", "jpeg", "webp"],
                key="start_image"
            )
            
            if start_image_file:
                processed_start, size_start = process_image(start_image_file, aspect_ratio, image_method, crop_position)
                st.image(processed_start, caption=f"מעובדת: {size_start[0]}x{size_start[1]}", width=300)
            
            if mode == "interpolation":
                end_image_file = st.file_uploader(
                    "תמונה סופית",
                    type=["png", "jpg", "jpeg", "webp"],
                    key="end_image"
                )
                
                if end_image_file:
                    processed_end, size_end = process_image(end_image_file, aspect_ratio, image_method, crop_position)
                    st.image(processed_end, caption=f"מעובדת: {size_end[0]}x{size_end[1]}", width=300)
    
    with col2:
        st.subheader("🎥 תוצאה")
        
        # כפתור יצירה
        generate_button = st.button(
            "🚀 צור וידאו",
            type="primary",
            use_container_width=True,
            disabled=not api_key or not prompt
        )
        
        # אזור התקדמות
        progress_placeholder = st.empty()
        video_placeholder = st.empty()
        
        if generate_button:
            # בדיקות
            if not api_key:
                st.error("❌ חסר API Key!")
                return
            
            if not prompt:
                st.error("❌ חסר פרומפט!")
                return
            
            if mode == "image_to_video" and not start_image_file:
                st.error("❌ חסרה תמונה!")
                return
            
            if mode == "interpolation" and (not start_image_file or not end_image_file):
                st.error("❌ חסרות תמונות (התחלה וסוף)!")
                return
            
            # יצירת client
            try:
                os.environ["GOOGLE_API_KEY"] = api_key
                client = genai.Client()
            except Exception as e:
                st.error(f"❌ שגיאה בחיבור ל-API: {e}")
                return
            
            # הכנת פרמטרים
            actual_model = model
            # החלפה אוטומטית ל-VEO 3.1 עבור Interpolation
            if mode == "interpolation" and "3.1" not in model:
                actual_model = "veo-3.1-generate-preview"
                st.info("ℹ️ הוחלף אוטומטית ל-VEO 3.1 (נדרש ל-Interpolation)")
            
            params = {
                "model": actual_model,
                "mode": mode,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "aspect_ratio": aspect_ratio,
                "duration": duration,
                "resolution": resolution,
            }
            
            # הוספת תמונות מעובדות
            if mode in ["image_to_video", "interpolation"] and start_image_file:
                params["start_image"], _ = process_image(start_image_file, aspect_ratio, image_method, crop_position)
            
            if mode == "interpolation" and end_image_file:
                params["end_image"], _ = process_image(end_image_file, aspect_ratio, image_method, crop_position)
            
            # יצירת הוידאו
            video_bytes, message = generate_video(client, params, progress_placeholder)
            
            if video_bytes:
                progress_placeholder.success(message)
                
                # הצגת הוידאו
                video_placeholder.video(video_bytes)
                
                # כפתור הורדה
                st.download_button(
                    "💾 הורד וידאו",
                    data=video_bytes,
                    file_name=f"video_{int(time.time())}.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
            else:
                progress_placeholder.error(message)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        🎬 AI Video Generator | Powered by Google VEO
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()