import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import boto3
from botocore.client import Config
from io import BytesIO
from streamlit_drawable_canvas import st_canvas
import streamlit.components.v1 as components

st.set_page_config(page_title="Pollen Curator", layout="wide")

BATCH_SIZE = 12

def get_secret(key, default=None):
    if hasattr(st, "secrets") and key in st.secrets:
        return st.secrets[key]
    return os.environ.get(key, default)

# S3 Configuration
@st.cache_resource
def get_s3_client():
    endpoint = get_secret("S3_ENDPOINT", "https://s3.cl4.du.cesnet.cz")
    access_key = get_secret("AWS_ACCESS_KEY_ID")
    secret_key = get_secret("AWS_SECRET_ACCESS_KEY")
    if not access_key or not secret_key:
        return None
    try:
        return boto3.client('s3',
                            endpoint_url=endpoint,
                            aws_access_key_id=access_key,
                            aws_secret_access_key=secret_key,
                            config=Config(signature_version='s3v4'))
    except Exception:
        return None

def get_bucket_name():
    return get_secret("S3_BUCKET", "bucket")

@st.cache_resource
def load_model():
    model_path = "best.pt"
    if os.path.exists(model_path):
        return YOLO(model_path)
    # Fallback to FastSAM if best.pt is not available yet
    from ultralytics import FastSAM
    return FastSAM("FastSAM-s.pt")

def filter_sam_results(results, orig_img):
    """Filter SAM masks based on area and color (purple hue)."""
    if not results or not results[0].masks:
        return results
        
    res = results[0]
    masks = res.masks.data.cpu().numpy()
    hsv_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)
    img_h, img_w = orig_img.shape[:2]
    img_area = img_h * img_w
    
    keep_indices = []
    for i, mask in enumerate(masks):
        mask_resized = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        area = np.sum(mask_resized)
        
        # Area filter to remove tiny specks
        if area < (img_area * 0.0005): 
            continue
            
        masked_hsv = hsv_img[mask_resized.astype(bool)]
        if len(masked_hsv) == 0:
            continue
            
        avg_h = np.mean(masked_hsv[:, 0])
        avg_s = np.mean(masked_hsv[:, 1])
        
        # Purple in OpenCV HSV is generally Hue between 110 and 170
        # Pollen is very saturated, dust is not. Increase S threshold to 80.
        if 110 <= avg_h <= 170 and avg_s > 80:
            keep_indices.append(i)
            
    if len(keep_indices) == 0:
        return [res[[]]] # Return empty results object
        
    return [res[keep_indices]]

def fetch_keys_from_s3():
    s3 = get_s3_client()
    if not s3:
        st.error("⚠️ S3 client is not configured. Please add AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in Streamlit Secrets.")
        return
    st.info("Fetching tiles from S3...")
    try:
        bucket = get_bucket_name()
        prefix = "Ostatni/Pollen_viability/tiles_640/"
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=200)
        new_keys = []
        if 'Contents' in response:
            for obj in response['Contents']:
                if not obj['Key'].endswith('/'):
                    new_keys.append(obj['Key'])
        st.session_state.s3_keys = new_keys
    except Exception as e:
        st.error(f"Error fetching from S3: {e}")

def get_dashboard_counts():
    categories = ["hard_positives", "needs_labeling", "hard_negatives", "discarded"]
    counts = {cat: 0 for cat in categories}
    try:
        s3 = get_s3_client()
        if not s3:
            return counts
        bucket = get_bucket_name()
        base_prefix = "Ostatni/Pollen_viability/active_learning/"
        for cat in categories:
            response = s3.list_objects_v2(Bucket=bucket, Prefix=f"{base_prefix}{cat}/")
            if 'Contents' in response:
                imgs = [obj for obj in response['Contents'] if not obj['Key'].endswith('/') and not obj['Key'].endswith('.txt')]
                counts[cat] = len(imgs)
    except Exception:
        pass
    return counts

# State Initialization
if "s3_keys" not in st.session_state:
    st.session_state.s3_keys = []
if "assignments" not in st.session_state:
    st.session_state.assignments = {}
if "batch_images" not in st.session_state:
    st.session_state.batch_images = {}
if "batch_results" not in st.session_state:
    st.session_state.batch_results = {}
if "keyboard_idx" not in st.session_state:
    st.session_state.keyboard_idx = 0

ACTION_MAP = {
    "🌟 Hard Positives": "hard_positives",
    "⚠️ Needs Labeling": "needs_labeling",
    "🌑 Hard Negatives": "hard_negatives",
    "🗑️ Discard": "discarded"
}
ACTIONS = list(ACTION_MAP.keys())

# Sidebar: Dashboard & Mode Selection
st.sidebar.title("🌸 Curator Dashboard")

if get_s3_client() is None:
    st.sidebar.error("⚠️ **S3 Credentials Required**\n\nPlease configure `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in **App settings -> Secrets**.")

counts = get_dashboard_counts()
st.sidebar.metric("🌟 Hard Positives", counts["hard_positives"])
st.sidebar.metric("⚠️ Needs Labeling", counts["needs_labeling"])
st.sidebar.metric("🌑 Hard Negatives", counts["hard_negatives"])
st.sidebar.metric("🗑️ Discard", counts["discarded"])

st.sidebar.markdown("---")
mode = st.sidebar.radio("Working Mode", ["⌨️ Keyboard Mode", "🎨 Canvas Mode", "📋 Grid Mode", "📱 Swipe Mode", "👀 Review & Submit"])

def process_submission():
    s3 = get_s3_client()
    bucket = get_bucket_name()
    current_batch = st.session_state.s3_keys[:BATCH_SIZE]
    
    with st.spinner("Uploading to S3..."):
        for key in current_batch:
            action_str = st.session_state.assignments.get(key, "🌑 Hard Negatives")
            action_type = ACTION_MAP[action_str]
            filename = os.path.basename(key)
            base_target = f"Ostatni/Pollen_viability/active_learning/{action_type}"
            
            s3.copy_object(
                Bucket=bucket,
                CopySource={'Bucket': bucket, 'Key': key},
                Key=f"{base_target}/{filename}"
            )
            
            if action_type == "hard_positives":
                res = st.session_state.batch_results.get(key)
                if res and len(res[0].boxes) > 0:
                    lines = []
                    if res[0].masks is not None:
                        for m_idx, mask_coords in enumerate(res[0].masks.xyn):
                            if len(mask_coords) == 0: continue
                            cls_id = int(res[0].boxes.cls[m_idx])
                            coords_str = " ".join([f"{x:.6f} {y:.6f}" for x, y in mask_coords])
                            lines.append(f"{cls_id} {coords_str}")
                    label_content = "\n".join(lines)
                    label_key = f"{base_target}/{os.path.splitext(filename)[0]}.txt"
                    s3.put_object(Bucket=bucket, Key=label_key, Body=label_content.encode('utf-8'))
            
            s3.delete_object(Bucket=bucket, Key=key)
            if key in st.session_state.s3_keys:
                st.session_state.s3_keys.remove(key)
                
    st.session_state.assignments = {}
    st.session_state.batch_results = {}
    st.session_state.batch_images = {}
    st.session_state.keyboard_idx = 0
    st.success("Batch Submitted!")

if not st.session_state.s3_keys:
    st.success("No pending tiles in queue!")
    if st.button("⬇️ Fetch Tiles from S3"):
        fetch_keys_from_s3()
        st.rerun()
    st.stop()

current_batch_keys = st.session_state.s3_keys[:BATCH_SIZE]
s3 = get_s3_client()
bucket = get_bucket_name()
model = load_model()

# Pre-load batch images and predict
for key in current_batch_keys:
    if key not in st.session_state.batch_images:
        response = s3.get_object(Bucket=bucket, Key=key)
        st.session_state.batch_images[key] = response['Body'].read()
    
    if key not in st.session_state.assignments:
        st.session_state.assignments[key] = "🌑 Hard Negatives"
        
    if model and key not in st.session_state.batch_results:
        img_bytes = st.session_state.batch_images[key]
        pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # Predict using YOLO or FastSAM
        # iou=0.3 enforces strict Non-Maximum Suppression (removes overlapping duplicates)
        results = model(cv_img, conf=0.25, iou=0.3, agnostic_nms=True, verbose=False)
        
        # Apply filtering if we are using FastSAM (which generates masks natively)
        if hasattr(model, 'task') and getattr(model, 'task', '') == 'segment' or type(model).__name__ == "FastSAM":
             results = filter_sam_results(results, cv_img)
             
        st.session_state.batch_results[key] = results
        if results and len(results[0].boxes) > 0:
            st.session_state.assignments[key] = "🌟 Hard Positives"

if mode == "⌨️ Keyboard Mode":
    st.markdown("### ⌨️ Keyboard Mode")
    st.info("💡 **Keyboard Controls:**\n- **Left/Right Arrows:** Change category\n- **Spacebar:** Advance to next tile\n- **Enter:** Submit batch")
    
    idx = st.session_state.keyboard_idx
    if idx >= len(current_batch_keys):
        st.success("Finished batch! Go to Review & Submit.")
        if st.button("Review & Submit"):
            st.session_state.keyboard_idx = 0
            st.rerun()
            
        # Also show history when finished with batch so they can edit before submit
        hist_keys = current_batch_keys
        if len(hist_keys) > 0:
            st.markdown("---")
            st.markdown("#### 🕒 Batch History (Click Edit to change)")
            display_keys = hist_keys[-8:] # Show up to 8
            hist_cols = st.columns(len(display_keys))
            for i, h_key in enumerate(display_keys):
                with hist_cols[i]:
                    h_img_bytes = st.session_state.batch_images[h_key]
                    h_pil_img = Image.open(BytesIO(h_img_bytes)).convert("RGB")
                    st.image(h_pil_img, use_container_width=True)
                    st.caption(st.session_state.assignments[h_key].split()[0]) # Just emoji
                    orig_idx = current_batch_keys.index(h_key)
                    if st.button("✏️ Edit", key=f"undo_end_{h_key}"):
                        st.session_state.keyboard_idx = orig_idx
                        st.rerun()
    else:
        key = current_batch_keys[idx]
        img_bytes = st.session_state.batch_images[key]
        pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
        
        # Render Image
        results = st.session_state.batch_results.get(key)
        if results and len(results[0].boxes) > 0:
            colA, colB = st.columns(2)
            with colA:
                show_labels = st.checkbox("Show Labels", value=False)
            with colB:
                opacity = st.slider("Mask Visibility", 0.0, 1.0, 1.0)
                
            orig_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            annotated_img = results[0].plot(boxes=False, labels=show_labels, conf=show_labels)
            
            if opacity < 1.0:
                annotated_img = cv2.addWeighted(annotated_img, opacity, orig_cv, 1 - opacity, 0)
                
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            st.image(annotated_img, width=640)
        else:
            st.image(pil_img, width=640)
            
        current_action = st.session_state.assignments[key]
        st.markdown(f"### Current Category: <span style='color:#0078D7'>{current_action}</span>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("⬅️ Prev Cat", key="btn_prev_cat"):
                c_idx = ACTIONS.index(st.session_state.assignments[key])
                st.session_state.assignments[key] = ACTIONS[(c_idx - 1) % 4]
                st.rerun()
        with col2:
            if st.button("➡️ Next Cat", key="btn_next_cat"):
                c_idx = ACTIONS.index(st.session_state.assignments[key])
                st.session_state.assignments[key] = ACTIONS[(c_idx + 1) % 4]
                st.rerun()
        with col3:
            if st.button("⏭️ Next Tile", key="btn_next_tile"):
                st.session_state.keyboard_idx += 1
                st.rerun()
        with col4:
            if st.button("🚀 Submit Batch", key="btn_submit"):
                process_submission()
                st.rerun()
                
        # History Line
        hist_keys = current_batch_keys[:idx]
        if len(hist_keys) > 0:
            st.markdown("---")
            st.markdown("#### 🕒 Recently Assigned")
            display_keys = hist_keys[-6:] # Show up to last 6
            hist_cols = st.columns(len(display_keys))
            for i, h_key in enumerate(display_keys):
                with hist_cols[i]:
                    h_img_bytes = st.session_state.batch_images[h_key]
                    h_pil_img = Image.open(BytesIO(h_img_bytes)).convert("RGB")
                    st.image(h_pil_img, use_container_width=True)
                    st.caption(st.session_state.assignments[h_key].split()[0]) # Just emoji
                    orig_idx = current_batch_keys.index(h_key)
                    if st.button("✏️ Edit", key=f"undo_{h_key}"):
                        st.session_state.keyboard_idx = orig_idx
                        st.rerun()

    # JS event listener mapping key presses to UI buttons
    components.html(
        """
        <script>
        const doc = window.parent.document;
        function clickButton(text) {
            const buttons = Array.from(doc.querySelectorAll('button'));
            const btn = buttons.find(b => b.innerText.includes(text));
            if (btn) btn.click();
        }
        doc.addEventListener('keydown', function(e) {
            if (e.key === 'ArrowLeft') { e.preventDefault(); clickButton('⬅️ Prev Cat'); }
            if (e.key === 'ArrowRight') { e.preventDefault(); clickButton('➡️ Next Cat'); }
            if (e.key === ' ') { e.preventDefault(); clickButton('⏭️ Next Tile'); }
            if (e.key === 'Enter') { e.preventDefault(); clickButton('🚀 Submit Batch'); }
        });
        </script>
        """,
        height=0
    )

elif mode == "🎨 Canvas Mode":
    st.markdown("### 🎨 Draw to Select")
    st.caption("Draw rectangles over the grid to select multiple tiles. Then click a category button below to assign them.")
    
    cell_w, cell_h = 320, 320
    cols, rows = 4, 3
    collage = Image.new('RGB', (cols * cell_w, rows * cell_h))
    
    for i, key in enumerate(current_batch_keys):
        c = i % cols
        r = i // cols
        img_bytes = st.session_state.batch_images[key]
        pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
        
        results = st.session_state.batch_results.get(key)
        if results and len(results[0].boxes) > 0:
            ann_img = results[0].plot()
            pil_img = Image.fromarray(cv2.cvtColor(ann_img, cv2.COLOR_BGR2RGB))
            
        pil_img = pil_img.resize((cell_w, cell_h))
        collage.paste(pil_img, (c * cell_w, r * cell_h))
        
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=2,
        stroke_color="#ff0000",
        background_image=collage,
        update_streamlit=True,
        height=rows * cell_h,
        width=cols * cell_w,
        drawing_mode="rect",
        key="canvas",
    )
    
    selected_indices = set()
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        for obj in objects:
            if obj["type"] == "rect":
                left, top = obj["left"], obj["top"]
                width, height = obj["width"], obj["height"]
                for i in range(len(current_batch_keys)):
                    c = i % cols
                    r = i // cols
                    cell_left, cell_top = c * cell_w, r * cell_h
                    cell_right, cell_bottom = cell_left + cell_w, cell_top + cell_h
                    rect_right, rect_bottom = left + width, top + height
                    if not (cell_right <= left or cell_left >= rect_right or cell_bottom <= top or cell_top >= rect_bottom):
                        selected_indices.add(i)
                        
    if selected_indices:
        st.info(f"**{len(selected_indices)} tiles selected.**")
        cat = st.selectbox("Assign selected to:", ACTIONS)
        if st.button("Apply Category to Selected"):
            for idx in selected_indices:
                st.session_state.assignments[current_batch_keys[idx]] = cat
            st.rerun()
            
    st.markdown("**Current Assignments:**")
    assign_cols = st.columns(4)
    for i, key in enumerate(current_batch_keys):
        with assign_cols[i % 4]:
            st.write(f"Tile {i+1}: {st.session_state.assignments[key]}")

elif mode == "📋 Grid Mode":
    st.markdown("### 📋 Grid Mode")
    cols = st.columns(4)
    for i, key in enumerate(current_batch_keys):
        with cols[i % 4]:
            img_bytes = st.session_state.batch_images[key]
            pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
            results = st.session_state.batch_results.get(key)
            if results and len(results[0].boxes) > 0:
                ann_img = results[0].plot()
                pil_img = Image.fromarray(cv2.cvtColor(ann_img, cv2.COLOR_BGR2RGB))
            st.image(pil_img, use_container_width=True)
            
            new_assign = st.radio("Action:", ACTIONS, index=ACTIONS.index(st.session_state.assignments[key]), key=f"rad_{key}")
            if new_assign != st.session_state.assignments[key]:
                st.session_state.assignments[key] = new_assign

elif mode == "👀 Review & Submit":
    st.markdown("### 👀 Review & Submit")
    
    for action in ACTIONS:
        keys_for_action = [k for k in current_batch_keys if st.session_state.assignments[k] == action]
        with st.expander(f"{action} ({len(keys_for_action)} tiles)", expanded=True):
            if not keys_for_action:
                st.write("None")
            else:
                cols = st.columns(6)
                for i, key in enumerate(keys_for_action):
                    with cols[i % 6]:
                        img_bytes = st.session_state.batch_images[key]
                        pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
                        st.image(pil_img, use_container_width=True)
                    
    if st.button("🚀 Submit Batch", type="primary", use_container_width=True):
        process_submission()

elif mode == "📱 Swipe Mode":
    st.markdown("### 📱 Swipe Mode (Individual Pollen Grains)")
    
    if "swipe_tile_idx" not in st.session_state:
        st.session_state.swipe_tile_idx = 0
    if "swipe_grain_idx" not in st.session_state:
        st.session_state.swipe_grain_idx = 0
    if "swipe_grains" not in st.session_state:
        st.session_state.swipe_grains = []
    if "swipe_labels" not in st.session_state:
        st.session_state.swipe_labels = {} # grain_id -> class_id
        
    # Get all tiles in current batch that have results
    valid_keys = [k for k in current_batch_keys if k in st.session_state.batch_results and len(st.session_state.batch_results[k][0].boxes) > 0]
    
    if not valid_keys:
        st.warning("No pollen grains detected in the current batch. Try a different batch or verify the model is working.")
    else:
        # Load grains for current tile if needed
        if st.session_state.swipe_tile_idx >= len(valid_keys):
            st.success("Finished all tiles in this batch!")
            if st.button("Review & Submit Batch"):
                st.session_state.swipe_tile_idx = 0
                st.rerun()
        else:
            current_key = valid_keys[st.session_state.swipe_tile_idx]
            
            # Extract grains if we haven't for this tile
            if getattr(st.session_state, '_current_swipe_key', None) != current_key:
                st.session_state._current_swipe_key = current_key
                st.session_state.swipe_grain_idx = 0
                st.session_state.swipe_labels = {}
                
                results = st.session_state.batch_results[current_key]
                img_bytes = st.session_state.batch_images[current_key]
                pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
                
                grains = []
                for idx, box in enumerate(results[0].boxes.xyxy):
                    x1, y1, x2, y2 = map(int, box.tolist())
                    # Add some padding
                    pad = 20
                    x1 = max(0, x1 - pad)
                    y1 = max(0, y1 - pad)
                    x2 = min(pil_img.width, x2 + pad)
                    y2 = min(pil_img.height, y2 + pad)
                    
                    cropped = pil_img.crop((x1, y1, x2, y2))
                    
                    # Original YOLO format coordinates (center x, center y, w, h normalized)
                    orig_x1, orig_y1, orig_x2, orig_y2 = map(float, box.tolist())
                    xc = ((orig_x1 + orig_x2) / 2) / pil_img.width
                    yc = ((orig_y1 + orig_y2) / 2) / pil_img.height
                    w = (orig_x2 - orig_x1) / pil_img.width
                    h = (orig_y2 - orig_y1) / pil_img.height
                    
                    grains.append({
                        "id": idx,
                        "image": cropped,
                        "yolo_coords": (xc, yc, w, h)
                    })
                st.session_state.swipe_grains = grains
            
            # Display current grain
            grains = st.session_state.swipe_grains
            
            if st.session_state.swipe_grain_idx >= len(grains):
                st.success(f"Finished {len(grains)} grains for this tile!")
                
                # Save YOLO labels to S3
                if st.button("Save Labels & Next Tile"):
                    s3 = get_s3_client()
                    bucket = get_bucket_name()
                    
                    # Generate YOLO string
                    lines = []
                    for g in grains:
                        gid = g["id"]
                        if gid in st.session_state.swipe_labels:
                            cls_id = st.session_state.swipe_labels[gid]
                            xc, yc, w, h = g["yolo_coords"]
                            lines.append(f"{cls_id} {xc} {yc} {w} {h}")
                    
                    if lines:
                        txt_content = "\n".join(lines)
                        base_name = os.path.basename(current_key)
                        # Replace .jpg or .czi with .txt
                        txt_key = current_key.rsplit('.', 1)[0] + '.txt'
                        s3.put_object(Bucket=bucket, Key=txt_key, Body=txt_content.encode('utf-8'))
                        st.toast(f"Saved {len(lines)} labels to S3!")
                        
                    st.session_state.swipe_tile_idx += 1
                    st.rerun()
            else:
                current_grain = grains[st.session_state.swipe_grain_idx]
                st.progress((st.session_state.swipe_grain_idx) / len(grains), text=f"Grain {st.session_state.swipe_grain_idx + 1} of {len(grains)}")
                
                # Big centered image
                st.image(current_grain["image"], use_container_width=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Giant Buttons
                col1, col2, col3 = st.columns(3)
                
                def classify_grain(cls_id):
                    st.session_state.swipe_labels[current_grain["id"]] = cls_id
                    st.session_state.swipe_grain_idx += 1
                
                with col1:
                    if st.button("🟩 Viable", use_container_width=True, key=f"btn_viable_{current_grain['id']}"):
                        classify_grain(0)
                        st.rerun()
                with col2:
                    if st.button("🟥 Non-Viable", use_container_width=True, key=f"btn_nonviable_{current_grain['id']}"):
                        classify_grain(1)
                        st.rerun()
                with col3:
                    if st.button("🟨 Aborted", use_container_width=True, key=f"btn_aborted_{current_grain['id']}"):
                        classify_grain(2)
                        st.rerun()
