import streamlit as st
st.set_page_config(page_title="Pollen Curator", layout="wide", initial_sidebar_state="expanded")

try:
    import os
    import json
    import cv2
    import numpy as np
    from PIL import Image, ImageEnhance
    import boto3
    from botocore.client import Config
    from io import BytesIO
    import streamlit.components.v1 as components
    import concurrent.futures
    import gc
except Exception as e:
    st.error(f"**Import Error:** `{type(e).__name__}: {e}`")
    import traceback
    st.code(traceback.format_exc())
    st.stop()


# Mobile and Touch Ergonomic Styling
st.markdown("""
<style>
    /* Global touch optimizations */
    div.stButton > button {
        width: 100% !important;
        min-height: 56px !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        margin: 4px 0px !important;
        transition: all 0.15s ease-in-out !important;
        touch-action: manipulation !important;
    }
    div.stButton > button:active {
        transform: scale(0.97) !important;
    }

    /* Viable Button */
    button[key*="btn_viable"] {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3) !important;
    }

    /* Non-Viable Button */
    button[key*="btn_nonviable"] {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3) !important;
    }

    /* Aborted Button */
    button[key*="btn_aborted"] {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.3) !important;
    }

    /* Undo Button */
    button[key*="btn_undo"] {
        background: linear-gradient(135deg, #4B5563 0%, #374151 100%) !important;
        color: #F9FAFB !important;
        border: 1px solid #6B7280 !important;
    }

    /* Discard & Relabel Buttons */
    button[key*="btn_discard_grain"], button[key*="btn_relabel_tile"], button[key*="btn_discard_tile"] {
        font-size: 1.05rem !important;
        min-height: 50px !important;
    }

    /* Top Navigation bar */
    .top-nav-container {
        display: flex;
        gap: 8px;
        margin-bottom: 16px;
        overflow-x: auto;
    }

    /* Responsive grid tweaks for mobile */
    @media (max-width: 768px) {
        .element-container, .stColumn {
            width: 100% !important;
        }
        div.stButton > button {
            min-height: 64px !important;
            font-size: 1.25rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

BATCH_SIZE = 6  # Keep low to stay within Streamlit Cloud 1GB RAM limit

def get_secret(key, default=None):
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
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



# ─── Lightweight detection result types ────────────────────────────────────────
# These mimic the YOLO result interface so all rendering code works unchanged,
# but they hold plain numpy arrays — no PyTorch, no ultralytics required.

class _Tensor:
    """Minimal tensor-like wrapper around a numpy array."""
    def __init__(self, arr):
        self._arr = np.array(arr, dtype=np.float32) if len(arr) else np.zeros((0, 4), dtype=np.float32)
    def __len__(self):
        return len(self._arr)
    def tolist(self):
        return self._arr.tolist()
    def __getitem__(self, idx):
        return self._arr[idx]
    # tensor-style column slices used for box coordinates
    @property
    def T(self):
        return self._arr.T
    def __repr__(self):
        return f"_Tensor({self._arr})"


class DetBoxes:
    """Mimics ultralytics.engine.results.Boxes."""
    def __init__(self, boxes_list):
        # boxes_list: [[x1,y1,x2,y2,conf,cls], ...]
        if boxes_list:
            arr = np.array(boxes_list, dtype=np.float32)
            self.xyxy = arr[:, :4]
            self.conf = arr[:, 4]
            self.cls  = arr[:, 5].astype(int)
        else:
            self.xyxy = np.zeros((0, 4), dtype=np.float32)
            self.conf = np.zeros(0, dtype=np.float32)
            self.cls  = np.zeros(0, dtype=int)

    def __len__(self):
        return len(self.xyxy)


class DetMasks:
    """Mimics ultralytics.engine.results.Masks."""
    def __init__(self, masks_xyn, img_w, img_h):
        # masks_xyn: list of [[xn, yn], ...] normalised polygons
        self.xyn = [np.array(m, dtype=np.float32) for m in masks_xyn]
        # pixel-space polygons (.xy)
        self.xy  = [np.array([[pt[0] * img_w, pt[1] * img_h] for pt in m], dtype=np.float32)
                    for m in masks_xyn]
        # .data as boolean masks for overlay code (lazy, uses polygons)
        self._w = img_w
        self._h = img_h

    @property
    def data(self):
        """Return boolean mask tensors (H×W) for each detection."""
        import numpy as np
        masks = []
        for poly in self.xy:
            mask = np.zeros((self._h, self._w), dtype=np.float32)
            if len(poly) >= 3:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(mask, [pts], 1.0)
            masks.append(mask)
        # Return as a simple list wrapped in a cpu()-able object
        return _MaskArray(masks)


class _MaskArray:
    """Minimal wrapper so .data.cpu().numpy() works."""
    def __init__(self, masks):
        self._masks = masks
    def cpu(self):
        return self
    def numpy(self):
        return np.array(self._masks, dtype=np.float32)
    def __len__(self):
        return len(self._masks)


class DetectionResult:
    """Mimics a single YOLO result object.  Holds pre-computed detections."""
    def __init__(self, det_json: dict, img_w: int, img_h: int):
        boxes_list   = det_json.get("boxes", [])
        masks_xyn    = det_json.get("masks_xyn", [])
        self.boxes   = DetBoxes(boxes_list)
        self.masks   = DetMasks(masks_xyn, img_w, img_h) if masks_xyn else None
        self._img_w  = img_w
        self._img_h  = img_h

    def __len__(self):
        return len(self.boxes)

    def __getitem__(self, idx):
        """Allow result[keep_idx] slicing used in box filtering."""
        import numpy as np
        subset = DetectionResult.__new__(DetectionResult)
        subset._img_w = self._img_w
        subset._img_h = self._img_h

        if hasattr(idx, '__len__') or isinstance(idx, (list, np.ndarray)):
            idx_list = list(idx)
        else:
            idx_list = [idx]

        boxes_arr = np.zeros((0, 6), dtype=np.float32)
        if len(self.boxes) > 0:
            full = np.column_stack([
                self.boxes.xyxy,
                self.boxes.conf[:, None],
                self.boxes.cls[:, None]
            ])
            boxes_arr = full[idx_list]
        subset.boxes = DetBoxes(boxes_arr.tolist())

        if self.masks is not None:
            subset_xyn = [self.masks.xyn[i].tolist() for i in idx_list if i < len(self.masks.xyn)]
            subset.masks = DetMasks(subset_xyn, self._img_w, self._img_h)
        else:
            subset.masks = None
        return subset

    def plot(self, boxes=True, labels=True, conf=True):
        """Minimal annotated-image renderer (replaces YOLO .plot())."""
        img = np.zeros((self._img_h, self._img_w, 3), dtype=np.uint8)
        cls_colors = {0: (0, 200, 0), 1: (0, 0, 220)}   # BGR: green, red
        default_color = (0, 200, 200)
        cls_names  = {0: "V", 1: "NV"}
        if self.masks is not None:
            overlay = img.copy()
            for i, poly in enumerate(self.masks.xy):
                cls_id = int(self.boxes.cls[i]) if i < len(self.boxes) else 0
                color = cls_colors.get(cls_id, default_color)
                pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
        if boxes:
            for i in range(len(self.boxes)):
                x1, y1, x2, y2 = [int(v) for v in self.boxes.xyxy[i]]
                cls_id = int(self.boxes.cls[i])
                cf = float(self.boxes.conf[i])
                color = cls_colors.get(cls_id, default_color)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                if labels:
                    lbl = f"{cls_names.get(cls_id, '?')} {cf:.0%}" if conf else cls_names.get(cls_id, '?')
                    cv2.putText(img, lbl, (x1, max(y1 - 6, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        return img

# ── end lightweight detection types ─────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_sample_viability_index_gui():
    json_path = os.path.join(os.path.dirname(__file__), "src", "sample_viability_index.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

sample_index = load_sample_viability_index_gui()

def extract_sample_id(key):
    """
    Extracts the matching sample ID from a file key or path using sample_index.
    """
    if not sample_index:
        return ""
    clean_k = os.path.basename(key)
    # Sort sample_index keys by length descending so longer IDs like '27-3-B' match before '7-3-B'
    sorted_s_ids = sorted(sample_index.keys(), key=len, reverse=True)
    for s_id in sorted_s_ids:
        if s_id in clean_k or s_id in key:
            return s_id
    parts = clean_k.replace('.jpg', '').replace('.czi', '').split('_')
    for part in parts:
        if part in sample_index:
            return part
    return parts[0]

def get_sample_rank(sample_id, strategy):
    if sample_id in sample_index:
        info = sample_index[sample_id]
        if strategy == "🎯 High Non-Viable Dense":
            return (info.get("non_viable", 0), info.get("non_viable_rate", 0.0))
        elif strategy == "🟩 Viable Dense":
            return (info.get("viable", 0), 1.0 - info.get("non_viable_rate", 0.0))
        elif strategy == "🌑 Hard Negatives (Low/Zero Pollen)":
            return (-info.get("total_grains", 0), -info.get("viable", 0))
    return (0, 0.0) if "High" in strategy or "Viable" in strategy else (99999, 99999)

@st.cache_data(ttl=120, show_spinner=False)
def get_all_s3_tile_keys():
    s3 = get_s3_client()
    if not s3:
        return {}
    bucket = get_bucket_name()
    prefix = "Ostatni/Pollen_viability/tiles_640/"
    valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp')
    
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")
        subfolders = []
        if 'CommonPrefixes' in response:
            subfolders = [p['Prefix'] for p in response['CommonPrefixes']]
            
        if subfolders:
            def fetch_folder_keys(folder_prefix):
                try:
                    s3_c = get_s3_client()
                    b_name = get_bucket_name()
                    f_resp = s3_c.list_objects_v2(Bucket=b_name, Prefix=folder_prefix, MaxKeys=100)
                    keys = []
                    if 'Contents' in f_resp:
                        for obj in f_resp['Contents']:
                            k = obj['Key']
                            if k.lower().endswith(valid_extensions):
                                keys.append(k)
                    return folder_prefix, keys
                except Exception:
                    return folder_prefix, []

            folder_keys_map = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                results = executor.map(fetch_folder_keys, subfolders)
                for f_prefix, keys in results:
                    if keys:
                        folder_keys_map[f_prefix] = keys
            return folder_keys_map
        else:
            all_keys = []
            paginator = s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        k = obj['Key']
                        if k.lower().endswith(valid_extensions):
                            all_keys.append(k)
            return all_keys
    except Exception:
        return {}

def fetch_keys_from_s3():
    s3 = get_s3_client()
    if not s3:
        st.error("⚠️ S3 client is not configured. Please add AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in Streamlit Secrets.")
        return
    st.info("Fetching tiles from S3...")
    try:
        strategy = getattr(st.session_state, "queue_strategy_select", "🎯 High Non-Viable Dense")
        folder_data = get_all_s3_tile_keys()

        # ── Source image filter ───────────────────────────────────────────────
        selected_folders = st.session_state.get("selected_image_folders", [])
        if isinstance(folder_data, dict) and selected_folders:
            folder_data = {k: v for k, v in folder_data.items() if k in selected_folders}

        valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp')
        new_keys = []
        
        if isinstance(folder_data, dict):
            reverse_sort = "High" in strategy or "Viable" in strategy
            sorted_folders = sorted(
                folder_data.keys(),
                key=lambda folder: get_sample_rank(extract_sample_id(folder), strategy),
                reverse=reverse_sort
            )
            for folder in sorted_folders:
                if len(new_keys) >= 500:
                    break
                new_keys.extend(folder_data[folder])
        elif isinstance(folder_data, list):
            reverse_sort = "High" in strategy or "Viable" in strategy
            new_keys = sorted(folder_data, key=lambda k: get_sample_rank(extract_sample_id(k), strategy), reverse=reverse_sort)

        st.session_state.s3_keys = new_keys[:500]
        st.session_state.batch_images = {}
        st.session_state.batch_results = {}
        st.session_state.assignments = {}
    except Exception as e:
        st.error(f"Error fetching from S3: {e}")

@st.cache_data(ttl=30, show_spinner=False)
def get_grain_and_tile_counts():
    categories = ["hard_positives", "needs_labeling", "hard_negatives", "discarded"]
    tile_counts = {cat: 0 for cat in categories}
    grain_counts = {"viable": 0, "non_viable": 0, "aborted": 0, "total": 0}
    try:
        s3 = get_s3_client()
        if not s3:
            return tile_counts, grain_counts
        bucket = get_bucket_name()
        base_prefix = "Ostatni/Pollen_viability/active_learning/"
        
        txt_keys = []
        for cat in categories:
            response = s3.list_objects_v2(Bucket=bucket, Prefix=f"{base_prefix}{cat}/")
            if 'Contents' in response:
                for obj in response['Contents']:
                    key = obj['Key']
                    if not key.endswith('/'):
                        if key.endswith('.txt'):
                            txt_keys.append(key)
                        else:
                            tile_counts[cat] += 1
                            
        def process_txt(key):
            counts = {0: 0, 1: 0, 2: 0}
            try:
                res = s3.get_object(Bucket=bucket, Key=key)
                lines = res['Body'].read().decode('utf-8').splitlines()
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        if cls_id in counts:
                            counts[cls_id] += 1
            except Exception:
                pass
            return counts

        if txt_keys:
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                results = executor.map(process_txt, txt_keys)
                for r in results:
                    grain_counts["viable"] += r[0]
                    grain_counts["non_viable"] += r[1]
                    grain_counts["aborted"] += r[2]

        grain_counts["total"] = grain_counts["viable"] + grain_counts["non_viable"] + grain_counts["aborted"]
    except Exception:
        pass
    return tile_counts, grain_counts

@st.cache_data(ttl=30, show_spinner=False)
def list_category_keys(category: str, page: int = 0, page_size: int = 24):
    """Return (total_count, page_keys) for tiles in a given active_learning category."""
    s3 = get_s3_client()
    if not s3:
        return 0, []
    bucket = get_bucket_name()
    prefix = f"Ostatni/Pollen_viability/active_learning/{category}/"
    valid_ext = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp')
    try:
        all_keys = []
        paginator = s3.get_paginator('list_objects_v2')
        for resp in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in resp.get('Contents', []):
                k = obj['Key']
                if k.lower().endswith(valid_ext):
                    all_keys.append(k)
        all_keys.sort()
        start = page * page_size
        return len(all_keys), all_keys[start : start + page_size]
    except Exception:
        return 0, []

def move_s3_file(key, target_category):
    try:
        s3 = get_s3_client()
        bucket = get_bucket_name()
        filename = os.path.basename(key)
        target_key = f"Ostatni/Pollen_viability/active_learning/{target_category}/{filename}"
        s3.copy_object(
            Bucket=bucket,
            CopySource={'Bucket': bucket, 'Key': key},
            Key=target_key
        )
        s3.delete_object(Bucket=bucket, Key=key)
        
        # Clean up session state
        if key in st.session_state.s3_keys:
            st.session_state.s3_keys.remove(key)
        if key in st.session_state.batch_images:
            del st.session_state.batch_images[key]
        if key in st.session_state.batch_results:
            del st.session_state.batch_results[key]
        if key in st.session_state.assignments:
            del st.session_state.assignments[key]
            
        get_grain_and_tile_counts.clear()
        return True
    except Exception as e:
        st.error(f"Error moving tile '{key}': {e}")
        return False

# State Initialization
if "s3_keys" not in st.session_state:
    st.session_state.s3_keys = []
    if get_s3_client() is not None:
        try:
            fetch_keys_from_s3()
        except Exception:
            pass

if "assignments" not in st.session_state:
    st.session_state.assignments = {}
if "batch_images" not in st.session_state:
    st.session_state.batch_images = {}
if "batch_results" not in st.session_state:
    st.session_state.batch_results = {}
if "keyboard_idx" not in st.session_state:
    st.session_state.keyboard_idx = 0
if "mode" not in st.session_state:
    st.session_state.mode = "📋 Grid Mode"
if "selected_image_folders" not in st.session_state:
    st.session_state.selected_image_folders = []

def set_active_mode(new_mode):
    st.session_state._pending_mode = new_mode

ACTION_MAP = {
    "🌟 Hard Positives": "hard_positives",
    "⚠️ Needs Labeling": "needs_labeling",
    "🌑 Hard Negatives": "hard_negatives",
    "🗑️ Discard": "discarded"
}
ACTIONS = list(ACTION_MAP.keys())
MODES = ["📱 Swipe Mode", "⌨️ Keyboard Mode", "🎨 Canvas Mode", "📋 Grid Mode", "👀 Review & Submit", "🗂️ Browse Categories", "📖 Tutorial & Guide"]

# Sidebar Dashboard & Working Mode
st.sidebar.title("🌸 Curator Dashboard")

if get_s3_client() is None:
    st.sidebar.error("⚠️ **S3 Credentials Required**\n\nPlease configure `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in **App settings -> Secrets**.")

tile_counts, grain_counts = get_grain_and_tile_counts()

st.sidebar.markdown("#### 🌾 Individual Pollen Grains")
st.sidebar.metric("🟩 Viable Grains", f"{grain_counts['viable']:,}")
st.sidebar.metric("🟥 Non-Viable Grains", f"{grain_counts['non_viable']:,}")
st.sidebar.metric("🟨 Aborted Grains", f"{grain_counts['aborted']:,}")
st.sidebar.caption(f"📊 **Total Pollen Count:** {grain_counts['total']:,}")

st.sidebar.markdown("---")
st.sidebar.markdown("#### 🖼️ Active Learning Tiles")
st.sidebar.metric("🌟 Hard Positives", tile_counts["hard_positives"])
st.sidebar.metric("⚠️ Needs Labeling", tile_counts["needs_labeling"])
st.sidebar.metric("🌑 Hard Negatives", tile_counts["hard_negatives"])
st.sidebar.metric("🗑️ Discarded Tiles", tile_counts["discarded"])

st.sidebar.markdown("---")
st.sidebar.markdown("#### 🎯 Dataset Balance Controls")

def on_queue_strategy_change():
    fetch_keys_from_s3()

queue_strategy = st.sidebar.selectbox(
    "🎯 Queue Filter & Priority",
    options=[
        "🎯 High Non-Viable Dense",
        "🟩 Viable Dense",
        "🌑 Hard Negatives (Low/Zero Pollen)",
        "🎲 All Tiles (Natural Mix)"
    ],
    index=0,
    key="queue_strategy_select",
    on_change=on_queue_strategy_change,
    help="Controls which tiles appear at the top of your batch queue in Grid & Swipe modes."
)

if sample_index:
    with st.sidebar.expander("📊 Non-Viable Sample Leaderboard", expanded=False):
        st.caption("Historical non-viable pollen yields per sample:")
        sorted_samples = sorted(
            sample_index.items(),
            key=lambda item: (item[1].get("non_viable", 0), item[1].get("non_viable_rate", 0)),
            reverse=True
        )[:12]
        for s_id, s_info in sorted_samples:
            pct = s_info.get('non_viable_rate', 0) * 100
            st.markdown(f"**`{s_id}`**: `{s_info.get('non_viable')} non-viable` ({pct:.1f}%)")

st.sidebar.markdown("---")

# ── Source Image Filter ────────────────────────────────────────────────────────
st.sidebar.markdown("#### 🔬 Source Image Filter")
st.sidebar.caption("Restrict tile queue to specific CZI source images.")

_folder_data_for_filter = get_all_s3_tile_keys()
if isinstance(_folder_data_for_filter, dict) and _folder_data_for_filter:
    _TILE_PREFIX = "Ostatni/Pollen_viability/tiles_640/"
    # Build display_name → full folder prefix map
    _folder_map = {
        f.rstrip("/").replace(_TILE_PREFIX, "").rstrip("/"): f
        for f in _folder_data_for_filter.keys()
    }
    _image_names = sorted(_folder_map.keys())

    # Determine current selection (convert stored prefixes back to display names)
    _current_selected = st.session_state.get("selected_image_folders", [])
    _current_names = [
        name for name, prefix in _folder_map.items()
        if prefix in _current_selected
    ]

    _selected_names = st.sidebar.multiselect(
        "Source CZI images",
        options=_image_names,
        default=_current_names,
        placeholder="All images (no filter applied)",
        key="image_filter_multiselect",
        help="Type to search. Select one or more images to restrict the queue.",
    )

    _fi_col1, _fi_col2 = st.sidebar.columns(2)
    with _fi_col1:
        if st.button("🔄 Apply", use_container_width=True, key="btn_apply_image_filter",
                     help="Reload queue with selected images only"):
            st.session_state.selected_image_folders = [
                _folder_map[n] for n in _selected_names
            ]
            fetch_keys_from_s3()
            get_grain_and_tile_counts.clear()
            st.rerun()
    with _fi_col2:
        _filter_active = bool(st.session_state.get("selected_image_folders", []))
        if st.button("✖ Clear", use_container_width=True, key="btn_clear_image_filter",
                     disabled=not _filter_active,
                     help="Remove filter and show all images"):
            st.session_state.selected_image_folders = []
            fetch_keys_from_s3()
            get_grain_and_tile_counts.clear()
            st.rerun()
else:
    st.sidebar.caption("*(Image list not available — check S3 connection)*")

st.sidebar.markdown("---")

# Apply any pending mode change BEFORE the radio widget renders
if "_pending_mode" in st.session_state:
    st.session_state.mode = st.session_state._pending_mode
    del st.session_state._pending_mode

st.sidebar.radio(
    "Working Mode", 
    MODES, 
    key="mode"
)


# Top Horizontal Navigation for Phone Ergonomics
st.markdown("### 🌸 Pollen Curator")
top_cols = st.columns(len(MODES))
for i, m_name in enumerate(MODES):
    with top_cols[i]:
        btn_type = "primary" if st.session_state.mode == m_name else "secondary"
        if st.button(m_name, key=f"top_nav_btn_{i}", type=btn_type, use_container_width=True):
            set_active_mode(m_name)
            st.rerun()

# Top Summary Card Cards for Pollen Grain Analytics
m_col1, m_col2, m_col3, m_col4 = st.columns(4)
with m_col1:
    st.metric("🟩 Viable Grains", f"{grain_counts['viable']:,}")
with m_col2:
    st.metric("🟥 Non-Viable Grains", f"{grain_counts['non_viable']:,}")
with m_col3:
    st.metric("🟨 Aborted Grains", f"{grain_counts['aborted']:,}")
with m_col4:
    st.metric("🖼️ Pending Batch", len(st.session_state.s3_keys))

# Active filter badge
_active_filter_folders = st.session_state.get("selected_image_folders", [])
if _active_filter_folders:
    _TILE_PFX = "Ostatni/Pollen_viability/tiles_640/"
    _active_names = [
        f.rstrip("/").replace(_TILE_PFX, "").rstrip("/")
        for f in _active_filter_folders
    ]
    _names_str = "`, `".join(_active_names)
    st.info(f"🔬 **Source filter active** — showing tiles from: `{_names_str}` "
            f"({len(_active_filter_folders)} image{'s' if len(_active_filter_folders) != 1 else ''}). "
            "Use the sidebar to change or clear.", icon="🔬")

st.markdown("---")
mode = st.session_state.mode

if mode == "📖 Tutorial & Guide":
    st.markdown("### 📖 Pollen Curator Interactive Guide & Tutorial")
    st.caption("Learn how to navigate, curate pollen viability, balance datasets, and use mobile tools.")
    
    t_tab1, t_tab2, t_tab3, t_tab4 = st.tabs(["📱 Mobile Curation", "📋 Grid & Keyboard Modes", "🎯 Dataset Balancing", "❓ FAQ & Rules"])
    
    with t_tab1:
        st.markdown("""
        #### 📱 Mobile Swipe Mode (Individual Pollen Grains)
        
        Designed specifically for fast, comfortable single-thumb operation on mobile phones.
        
        1. **View Grain Crop**: The screen displays a magnified crop of each detected pollen grain alongside SAM outline contours and confidence scores.
        2. **Classification Buttons**:
           - 🟩 **Viable**: Stained dark red/magenta, plump, full cytoplasm.
           - 🟥 **Non-Viable**: Pale green, empty shell, shriveled, unfertilized.
           - 🟨 **Aborted**: Faint pink/yellowish, incomplete cytoplasm.
        3. **Ergonomic Actions**:
           - **`↩️ Undo Last`**: Tapping this immediately restores your previous choice and steps back one grain or tile.
           - **`🗑️ Discard Label`**: Skips saving a label for bad or ambiguous crops without affecting the tile.
           - **`⚠️ Send Tile to Relabel`**: Moves the whole tile to `needs_labeling` for expert re-annotation.
           - **`🗑️ Discard Whole Tile`**: Removes the entire tile from active learning queue if out of focus or debris.
        """)
        
    with t_tab2:
        st.markdown("""
        #### 📋 Phone-Friendly Grid Mode (Tile Pollen Confirmation)
        - Select column density: `📱 2 Columns` (Recommended for phones) or `🖥️ 4 Columns` (Desktops).
        - Tap tile confirmation buttons below each card (`🌟 Pollen Present`, `🌑 No Pollen`, `⚠️ Needs Review`, `🗑️ Discard Tile`).
        - Use top quick buttons `🌟 Mark All as Pollen Present` or `🚀 Submit Tile Queue to S3`.
        - Tap `📱 Curate Grains in Swipe Mode` on any tile card to jump straight into grain-by-grain viability curation.
        
        #### ⌨️ Desktop Keyboard Mode
        - **Left / Right Arrow Keys**: Cycle categories.
        - **Spacebar**: Advance to next tile.
        - **Enter**: Submit current batch to S3.
        - **Undo Tile**: Step back tile index.
        """)
        
    with t_tab3:
        st.markdown("""
        #### 🎯 Dataset Balancing & Non-Viable Prioritization
        
        In natural microscope scans, **~96.8%** of grains are viable, leading to heavy dataset imbalance.
        
        - The Curator automatically cross-references historical sample rates from **`src/sample_viability_index.json`**.
        - Top high non-viable samples (e.g. `1-6-J` at **88.4%**, `7-9-F` at **66.9%**, `6-1-F` at **53.8%**) are automatically sorted to the top of your queue when **`🎯 Prioritize High Non-Viable Samples`** is checked.
        """)
        
    with t_tab4:
        st.markdown("""
        #### ❓ Frequently Asked Questions
        
        * **Where are my labeled tiles stored in S3?**
          They are moved to `Ostatni/Pollen_viability/active_learning/{hard_positives|needs_labeling|hard_negatives|discarded}/`.
        * **How are YOLO mask labels saved?**
          When submitting hard positives, `.txt` segmentation files are generated and uploaded alongside `.jpg` tiles.
        * **How do I switch modes on mobile?**
          Use the top horizontal navigation buttons (`📱 Swipe Mode`, `📋 Grid Mode`, `📖 Tutorial & Guide`) directly at the top of the main screen!
        """)
    st.stop()

def process_submission():
    s3 = get_s3_client()
    bucket = get_bucket_name()
    current_batch = st.session_state.s3_keys[:BATCH_SIZE]
    
    with st.spinner("Uploading batch to S3..."):
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
    get_grain_and_tile_counts.clear()
    st.success("Batch Submitted!")

valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp')
st.session_state.s3_keys = [k for k in st.session_state.s3_keys if k.lower().endswith(valid_extensions)]

if not st.session_state.s3_keys:
    st.success("🎉 No pending tiles in queue!")
    if st.button("⬇️ Fetch Tiles from S3", use_container_width=True):
        fetch_keys_from_s3()
        get_grain_and_tile_counts.clear()
        st.rerun()
    st.stop()

def is_tile_matching_strategy(key, strategy):
    if strategy == "🎲 All Tiles (Natural Mix)":
        return True

    res = st.session_state.batch_results.get(key)

    # If no detection results exist for this tile (no model loaded),
    # we can't filter — show all tiles regardless of strategy
    if res is None:
        return True

    num_nonviable = 0
    num_viable = 0
    total_boxes = 0
    if res and len(res) > 0 and hasattr(res[0], 'boxes') and res[0].boxes is not None:
        total_boxes = len(res[0].boxes)
        if hasattr(res[0].boxes, 'cls') and res[0].boxes.cls is not None:
            classes = res[0].boxes.cls.tolist()
            num_nonviable = sum(1 for c in classes if int(c) == 1)
            num_viable = sum(1 for c in classes if int(c) == 0)

    if strategy == "🌑 Hard Negatives (Low/Zero Pollen)":
        # EXCLUSIVE: MUST have 0 detected pollen grains!
        return total_boxes == 0
    elif strategy == "🎯 High Non-Viable Dense":
        # EXCLUSIVE: MUST contain detected non-viable grains OR come from high non-viable sample
        if num_nonviable > 0:
            return True
        s_id = extract_sample_id(key)
        s_info = sample_index.get(s_id, {})
        return (s_info.get("non_viable", 0) >= 5 or s_info.get("non_viable_rate", 0.0) >= 0.05) and total_boxes > 0
    elif strategy == "🟩 Viable Dense":
        # EXCLUSIVE: MUST contain viable pollen grains!
        return num_viable > 0
        
    return True

strategy = getattr(st.session_state, "queue_strategy_select", "🎯 High Non-Viable Dense")

s3 = get_s3_client()
bucket = get_bucket_name()

@st.cache_data(ttl=180, max_entries=50, show_spinner=False)
def load_s3_image_bytes(key):
    try:
        s3_c = get_s3_client()
        b_name = get_bucket_name()
        resp = s3_c.get_object(Bucket=b_name, Key=key)
        return resp['Body'].read()
    except Exception:
        return None


@st.cache_data(ttl=300, max_entries=200, show_spinner=False)
def load_s3_detection_json(tile_key):
    """Load pre-computed detection JSON (_det.json) from S3.
    Returns a dict {boxes, masks_xyn} or None if not found.
    """
    det_key = tile_key.rsplit('.', 1)[0] + '_det.json'
    try:
        s3_c = get_s3_client()
        b_name = get_bucket_name()
        resp = s3_c.get_object(Bucket=b_name, Key=det_key)
        return json.loads(resp['Body'].read().decode('utf-8'))
    except Exception:
        return None  # JSON not yet generated — tile shows without annotations


def fetch_tile_and_detections(key):
    """Fetch image bytes and detection JSON for one tile in parallel."""
    img_bytes = load_s3_image_bytes(key)
    det_json  = load_s3_detection_json(key)
    return key, img_bytes, det_json

# Load images + pre-computed detections; scan up to MAX_SCAN_TILES tiles
matching_keys = []
candidate_chunk_size = 6
scanned_count = 0
MAX_SCAN_TILES = 12

while len(matching_keys) < BATCH_SIZE and scanned_count < min(MAX_SCAN_TILES, len(st.session_state.s3_keys)):
    chunk_keys = st.session_state.s3_keys[scanned_count : scanned_count + candidate_chunk_size]
    scanned_count += candidate_chunk_size
    if not chunk_keys:
        break

    # Fetch images + detection JSONs in parallel (both are tiny S3 GETs)
    keys_to_fetch = [k for k in chunk_keys if k not in st.session_state.batch_images]
    if keys_to_fetch:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            for k, img_bytes, det_json in executor.map(fetch_tile_and_detections, keys_to_fetch):
                if img_bytes:
                    st.session_state.batch_images[k] = img_bytes
                if det_json is not None and k not in st.session_state.batch_results:
                    # Build a DetectionResult from the cached JSON
                    try:
                        pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
                        det_result = DetectionResult(det_json, pil_img.width, pil_img.height)
                        st.session_state.batch_results[k] = [det_result]
                        if len(det_result.boxes) > 0:
                            st.session_state.assignments[k] = "🌟 Hard Positives"
                    except Exception:
                        pass

    for key in chunk_keys:
        if key not in st.session_state.assignments:
            st.session_state.assignments[key] = "🌑 Hard Negatives"

        if is_tile_matching_strategy(key, strategy):
            if key not in matching_keys:
                matching_keys.append(key)
            if len(matching_keys) >= BATCH_SIZE:
                break

# Fallback: if no matching keys found, show whatever was scanned
if not matching_keys:
    matching_keys = st.session_state.s3_keys[:BATCH_SIZE]

current_batch_keys = matching_keys[:BATCH_SIZE]

# Prune session state to only the active batch (keep RAM tight)
active_set = set(current_batch_keys)
st.session_state.batch_images  = {k: v for k, v in st.session_state.batch_images.items()  if k in active_set}
st.session_state.batch_results = {k: v for k, v in st.session_state.batch_results.items() if k in active_set}
gc.collect()

# MODE IMPLEMENTATIONS

if mode == "⌨️ Keyboard Mode":
    st.markdown("### ⌨️ Keyboard Mode")
    st.info("💡 **Keyboard Controls:**\n- **Left/Right Arrows:** Change category\n- **Spacebar:** Advance to next tile\n- **Enter:** Submit batch")
    
    idx = st.session_state.keyboard_idx
    if idx >= len(current_batch_keys):
        st.success("Finished batch! Go to Review & Submit.")
        if st.button("Review & Submit", type="primary", use_container_width=True):
            st.session_state.keyboard_idx = 0
            set_active_mode("👀 Review & Submit")
            st.rerun()
            
        hist_keys = current_batch_keys
        if len(hist_keys) > 0:
            st.markdown("---")
            st.markdown("#### 🕒 Batch History (Click Edit to change)")
            display_keys = hist_keys[-8:]
            hist_cols = st.columns(len(display_keys))
            for i, h_key in enumerate(display_keys):
                with hist_cols[i]:
                    try:
                        h_img_bytes = st.session_state.batch_images[h_key]
                        h_pil_img = Image.open(BytesIO(h_img_bytes)).convert("RGB")
                        st.image(h_pil_img, use_container_width=True)
                        st.caption(st.session_state.assignments[h_key].split()[0])
                    except Exception:
                        pass
                    orig_idx = current_batch_keys.index(h_key)
                    if st.button("✏️ Edit", key=f"undo_end_{h_key}"):
                        st.session_state.keyboard_idx = orig_idx
                        st.rerun()
    else:
        key = current_batch_keys[idx]
        if key not in st.session_state.batch_images:
            st.error(f"Image data for tile '{key}' not available.")
            st.stop()
        try:
            img_bytes = st.session_state.batch_images[key]
            pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            st.error(f"Cannot render image tile '{key}': {e}")
            st.stop()
        
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
            st.image(annotated_img, use_container_width=True)
        else:
            st.image(pil_img, use_container_width=True)
            
        current_action = st.session_state.assignments[key]
        st.markdown(f"### Current Category: <span style='color:#0078D7'>{current_action}</span>", unsafe_allow_html=True)
        
        # Action Bar & Ergonomic Control Buttons
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button("⬅️ Prev Cat", key="btn_prev_cat", use_container_width=True):
                c_idx = ACTIONS.index(st.session_state.assignments[key])
                st.session_state.assignments[key] = ACTIONS[(c_idx - 1) % 4]
                st.rerun()
        with c2:
            if st.button("➡️ Next Cat", key="btn_next_cat", use_container_width=True):
                c_idx = ACTIONS.index(st.session_state.assignments[key])
                st.session_state.assignments[key] = ACTIONS[(c_idx + 1) % 4]
                st.rerun()
        with c3:
            if st.button("↩️ Undo Tile", key="btn_undo_kb", use_container_width=True):
                if st.session_state.keyboard_idx > 0:
                    st.session_state.keyboard_idx -= 1
                st.rerun()
        with c4:
            if st.button("⏭️ Next Tile", key="btn_next_tile", use_container_width=True):
                st.session_state.keyboard_idx += 1
                st.rerun()
                
        # Direct Relabel & Discard buttons for whole tile
        rc1, rc2 = st.columns(2)
        with rc1:
            if st.button("⚠️ Send Tile to Relabel", key="btn_relabel_tile_kb", use_container_width=True):
                move_s3_file(key, "needs_labeling")
                st.toast("Tile moved to Needs Labeling!")
                st.rerun()
        with rc2:
            if st.button("🗑️ Discard Tile", key="btn_discard_tile_kb", use_container_width=True):
                move_s3_file(key, "discarded")
                st.toast("Tile discarded!")
                st.rerun()
                
        # History Line
        hist_keys = current_batch_keys[:idx]
        if len(hist_keys) > 0:
            st.markdown("---")
            st.markdown("#### 🕒 Recently Assigned")
            display_keys = hist_keys[-6:]
            hist_cols = st.columns(len(display_keys))
            for i, h_key in enumerate(display_keys):
                with hist_cols[i]:
                    h_img_bytes = st.session_state.batch_images[h_key]
                    h_pil_img = Image.open(BytesIO(h_img_bytes)).convert("RGB")
                    st.image(h_pil_img, use_container_width=True)
                    st.caption(st.session_state.assignments[h_key].split()[0])
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
    try:
        from streamlit_drawable_canvas import st_canvas
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
    except Exception as e:
        st.warning(f"⚠️ Canvas drawing is disabled due to component compatibility: {e}")
        canvas_result = None
    
    selected_indices = set()
    if canvas_result is not None and canvas_result.json_data is not None:
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
        if st.button("Apply Category to Selected", type="primary", use_container_width=True):
            for idx in selected_indices:
                st.session_state.assignments[current_batch_keys[idx]] = cat
            st.rerun()
            
    st.markdown("**Current Assignments:**")
    assign_cols = st.columns(4)
    for i, key in enumerate(current_batch_keys):
        with assign_cols[i % 4]:
            st.write(f"Tile {i+1}: {st.session_state.assignments[key]}")

elif mode == "📋 Grid Mode":
    st.markdown("### 📋 Grid Mode (Tile Pollen Confirmation)")
    st.info("💡 **Tile-Level Confirmation Only**: Grid Mode confirms whether a tile contains pollen grains (`Pollen Present`) vs empty background (`No Pollen`). Individual grain viability (**Viable** 🟩 / **Non-Viable** 🟥 / **Aborted** 🟨) is identified per grain in **📱 Swipe Mode**.")
    
    # Grid Mode Quick Batch Bar
    g_ctrl1, g_ctrl2, g_ctrl3, g_ctrl4 = st.columns([2, 2, 2, 3])
    with g_ctrl1:
        grid_cols_num = st.radio("Grid Columns:", [2, 1, 4], index=0, horizontal=True, key="grid_cols_choice", help="Select grid column count for comfortable phone viewing.")
    with g_ctrl2:
        if st.button("🌟 Mark All as Pollen Present", use_container_width=True, key="btn_all_pos"):
            for k in current_batch_keys:
                st.session_state.assignments[k] = "🌟 Hard Positives"
            st.rerun()
    with g_ctrl3:
        if st.button("⏭️ Skip Batch", use_container_width=True, key="btn_skip_batch"):
            # Rotate s3_keys: move current batch to the back so next scan gets fresh tiles
            skip_count = min(MAX_SCAN_TILES, len(st.session_state.s3_keys))
            st.session_state.s3_keys = st.session_state.s3_keys[skip_count:] + st.session_state.s3_keys[:skip_count]
            st.session_state.batch_images = {}
            st.session_state.batch_results = {}
            st.session_state.assignments = {}
            get_grain_and_tile_counts.clear()
            st.rerun()
    with g_ctrl4:
        if st.button("🚀 Submit Tile Queue to S3", type="primary", use_container_width=True, key="btn_grid_submit"):
            process_submission()
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    grid_cols = st.columns(grid_cols_num)
    for i, key in enumerate(current_batch_keys):
        with grid_cols[i % grid_cols_num]:
            curr_assign = st.session_state.assignments.get(key, "🌑 Hard Negatives")
            st.markdown(f"**Tile {i+1} / {len(current_batch_keys)}** &nbsp;|&nbsp; Tile Status: `{curr_assign.split()[0]}`")
            
            img_bytes = st.session_state.batch_images.get(key)
            if img_bytes:
                pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
                results = st.session_state.batch_results.get(key)
                if results and len(results[0].boxes) > 0:
                    # Draw color-coded detections: Green=Viable, Red=Non-Viable, Yellow=Other
                    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    boxes = results[0].boxes
                    class_names = {0: "V", 1: "NV"}
                    class_colors = {0: (0, 200, 0), 1: (0, 0, 220)}  # BGR: green, red
                    default_color = (0, 200, 200)  # yellow
                    
                    # Draw masks if available
                    if hasattr(results[0], 'masks') and results[0].masks is not None:
                        overlay = cv_img.copy()
                        for m_idx, mask_xy in enumerate(results[0].masks.xy):
                            cls_id = int(boxes.cls[m_idx])
                            color = class_colors.get(cls_id, default_color)
                            pts = np.array(mask_xy, np.int32).reshape((-1, 1, 2))
                            cv2.fillPoly(overlay, [pts], color)
                        cv2.addWeighted(overlay, 0.3, cv_img, 0.7, 0, cv_img)
                    
                    # Draw boxes and labels
                    for b_idx in range(len(boxes)):
                        cls_id = int(boxes.cls[b_idx])
                        conf = float(boxes.conf[b_idx])
                        x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[b_idx].tolist()]
                        color = class_colors.get(cls_id, default_color)
                        label = f"{class_names.get(cls_id, '?')} {conf:.0%}"
                        cv2.rectangle(cv_img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(cv_img, label, (x1, max(y1 - 6, 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
                st.image(pil_img, use_container_width=True)
            
            # Ergonomic Touch Action Buttons for Tile Confirmation
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                pos_type = "primary" if curr_assign == "🌟 Hard Positives" else "secondary"
                if st.button("🌟 Pollen Present", key=f"g_pos_{key}", type=pos_type, use_container_width=True):
                    st.session_state.assignments[key] = "🌟 Hard Positives"
                    st.rerun()
            with btn_col2:
                neg_type = "primary" if curr_assign == "🌑 Hard Negatives" else "secondary"
                if st.button("🌑 No Pollen", key=f"g_neg_{key}", type=neg_type, use_container_width=True):
                    st.session_state.assignments[key] = "🌑 Hard Negatives"
                    st.rerun()

            btn_col3, btn_col4 = st.columns(2)
            with btn_col3:
                rel_type = "primary" if curr_assign == "⚠️ Needs Labeling" else "secondary"
                if st.button("⚠️ Needs Review", key=f"g_rel_{key}", type=rel_type, use_container_width=True):
                    st.session_state.assignments[key] = "⚠️ Needs Labeling"
                    st.rerun()
            with btn_col4:
                dis_type = "primary" if curr_assign == "🗑️ Discard" else "secondary"
                if st.button("🗑️ Discard Tile", key=f"g_dis_{key}", type=dis_type, use_container_width=True):
                    st.session_state.assignments[key] = "🗑️ Discard"
                    st.rerun()

            # Shortcut to jump directly to Swipe Mode for grain-level viability curation
            if st.button(f"📱 Curate Grains in Swipe Mode", key=f"g_swipe_{key}", use_container_width=True):
                valid_k = [k for k in current_batch_keys if k in st.session_state.batch_results and len(st.session_state.batch_results[k][0].boxes) > 0]
                if key in valid_k:
                    st.session_state.swipe_tile_idx = valid_k.index(key)
                    st.session_state._current_swipe_key = None
                set_active_mode("📱 Swipe Mode")
                st.rerun()

            st.markdown("---")

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
                        img_bytes = st.session_state.batch_images.get(key)
                        if not img_bytes:
                            try:
                                resp = s3.get_object(Bucket=bucket, Key=key)
                                img_bytes = resp['Body'].read()
                                st.session_state.batch_images[key] = img_bytes
                            except Exception:
                                continue
                        pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
                        st.image(pil_img, use_container_width=True)
                    
    if st.button("🚀 Submit Batch", type="primary", use_container_width=True):
        process_submission()

elif mode == "📱 Swipe Mode":
    st.markdown("### 📱 Mobile Swipe Mode (Individual Pollen Grains)")
    
    with st.expander("⚙️ Image Controls & SAM Outline Opacity", expanded=False):
        b_col, c_col, o_col = st.columns(3)
        with b_col:
            brightness = st.slider("☀️ Brightness", 0.8, 3.0, 1.4, 0.1, key="swipe_brightness")
        with c_col:
            contrast = st.slider("🔍 Contrast", 0.8, 2.5, 1.2, 0.1, key="swipe_contrast")
        with o_col:
            mask_opacity = st.slider("👁️ Outline Opacity", 0.0, 1.0, 0.5, 0.1, key="swipe_mask_opacity")
    
    if "swipe_tile_idx" not in st.session_state:
        st.session_state.swipe_tile_idx = 0
    if "swipe_grain_idx" not in st.session_state:
        st.session_state.swipe_grain_idx = 0
    if "swipe_grains" not in st.session_state:
        st.session_state.swipe_grains = []
    if "swipe_labels" not in st.session_state:
        st.session_state.swipe_labels = {} # grain_id -> class_id
    if "swipe_history" not in st.session_state:
        st.session_state.swipe_history = [] # list of {"grain_idx": G, "label": prev_label}
        
    valid_keys = [k for k in current_batch_keys if k in st.session_state.batch_results and len(st.session_state.batch_results[k][0].boxes) > 0]
    
    if not valid_keys:
        st.warning("⚠️ No pollen grains detected in the current batch. Try fetching a new batch or switching modes.")
    else:
        if st.session_state.swipe_tile_idx >= len(valid_keys):
            st.success("🎉 Finished all tiles in this batch!")
            if st.button("🚀 Go to Review & Submit Batch", type="primary", use_container_width=True):
                st.session_state.swipe_tile_idx = 0
                set_active_mode("👀 Review & Submit")
                st.rerun()
        else:
            current_key = valid_keys[st.session_state.swipe_tile_idx]
            
            # Load grains for tile if not loaded
            if getattr(st.session_state, '_current_swipe_key', None) != current_key:
                st.session_state._current_swipe_key = current_key
                st.session_state.swipe_grain_idx = 0
                st.session_state.swipe_labels = {}
                st.session_state.swipe_history = []
                
                results = st.session_state.batch_results.get(current_key)
                img_bytes = st.session_state.batch_images.get(current_key)
                
                if not img_bytes:
                    try:
                        resp = s3.get_object(Bucket=bucket, Key=current_key)
                        img_bytes = resp['Body'].read()
                        st.session_state.batch_images[current_key] = img_bytes
                    except Exception:
                        st.error(f"Failed to load image for '{current_key}'")
                        st.session_state.swipe_tile_idx += 1
                        st.rerun()
                        
                if not results:
                    st.warning(f"Detection results missing for '{current_key}'. Skipping tile...")
                    st.session_state.swipe_tile_idx += 1
                    st.rerun()
                    
                pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
                cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                
                overlay_cv = cv_img.copy()
                if results[0].masks is not None and len(results[0].masks.data) > 0:
                    masks_data = results[0].masks.data.cpu().numpy()
                    img_h, img_w = cv_img.shape[:2]
                    for m_idx, mask in enumerate(masks_data):
                        mask_resized = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(overlay_cv, contours, -1, (255, 255, 0), 2)
                else:
                    for box in results[0].boxes.xyxy:
                        bx1, by1, bx2, by2 = map(int, box.tolist())
                        cv2.rectangle(overlay_cv, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                        
                annotated_pil = Image.fromarray(cv2.cvtColor(overlay_cv, cv2.COLOR_BGR2RGB))
                
                grains = []
                for idx, box in enumerate(results[0].boxes.xyxy):
                    x1, y1, x2, y2 = map(int, box.tolist())
                    pad = 20
                    x1 = max(0, x1 - pad)
                    y1 = max(0, y1 - pad)
                    x2 = min(pil_img.width, x2 + pad)
                    y2 = min(pil_img.height, y2 + pad)
                    
                    cropped_raw = pil_img.crop((x1, y1, x2, y2))
                    cropped_overlay = annotated_pil.crop((x1, y1, x2, y2))
                    
                    conf = 1.0
                    if hasattr(results[0].boxes, 'conf') and len(results[0].boxes.conf) > idx:
                        conf = float(results[0].boxes.conf[idx])
                    
                    orig_x1, orig_y1, orig_x2, orig_y2 = map(float, box.tolist())
                    xc = ((orig_x1 + orig_x2) / 2) / pil_img.width
                    yc = ((orig_y1 + orig_y2) / 2) / pil_img.height
                    w = (orig_x2 - orig_x1) / pil_img.width
                    h = (orig_y2 - orig_y1) / pil_img.height
                    
                    grains.append({
                        "id": idx,
                        "image_raw": cropped_raw,
                        "image_overlay": cropped_overlay,
                        "conf": conf,
                        "yolo_coords": (xc, yc, w, h)
                    })
                st.session_state.swipe_grains = grains
            
            grains = st.session_state.swipe_grains
            
            if st.session_state.swipe_grain_idx >= len(grains):
                st.success(f"✅ Categorized {len(st.session_state.swipe_labels)} of {len(grains)} grains for this tile!")
                
                s_col1, s_col2 = st.columns(2)
                with s_col1:
                    if st.button("↩️ Undo Last Grain", key="btn_undo_end_grain", use_container_width=True):
                        if st.session_state.swipe_grain_idx > 0:
                            st.session_state.swipe_grain_idx -= 1
                            if st.session_state.swipe_history:
                                last_item = st.session_state.swipe_history.pop()
                                gid = last_item["grain_id"]
                                if last_item["label"] is None:
                                    st.session_state.swipe_labels.pop(gid, None)
                                else:
                                    st.session_state.swipe_labels[gid] = last_item["label"]
                        st.rerun()
                with s_col2:
                    if st.button("💾 Save Labels & Next Tile", type="primary", key="btn_save_next_tile", use_container_width=True):
                        lines = []
                        _tile_results = st.session_state.batch_results.get(current_key)
                        _has_masks = (_tile_results and _tile_results[0].masks is not None
                                      and len(_tile_results[0].masks.xyn) > 0)
                        for g in grains:
                            gid = g["id"]
                            if gid in st.session_state.swipe_labels:
                                cls_id = st.session_state.swipe_labels[gid]
                                # Prefer SAM polygon mask over bbox when available
                                if _has_masks and gid < len(_tile_results[0].masks.xyn):
                                    poly = _tile_results[0].masks.xyn[gid]
                                    if len(poly) >= 3:
                                        coords_str = " ".join(
                                            f"{float(pt[0]):.6f} {float(pt[1]):.6f}"
                                            for pt in poly
                                        )
                                        lines.append(f"{cls_id} {coords_str}")
                                        continue
                                # Fallback: YOLO bbox format
                                xc, yc, w, h = g["yolo_coords"]
                                lines.append(f"{cls_id} {xc} {yc} {w} {h}")
                        
                        if lines:
                            txt_content = "\n".join(lines)
                            txt_key = current_key.rsplit('.', 1)[0] + '.txt'
                            s3.put_object(Bucket=bucket, Key=txt_key, Body=txt_content.encode('utf-8'))
                            st.toast(f"Saved {len(lines)} labels to S3!")
                            get_grain_and_tile_counts.clear()
                            
                        st.session_state.swipe_tile_idx += 1
                        st.rerun()
            else:
                current_grain = grains[st.session_state.swipe_grain_idx]
                st.progress((st.session_state.swipe_grain_idx) / len(grains), text=f"Grain {st.session_state.swipe_grain_idx + 1} of {len(grains)}")
                
                conf_pct = current_grain.get('conf', 1.0) * 100
                st.markdown(f"**🎯 SAM / YOLO Confidence:** `{conf_pct:.1f}%` &nbsp;|&nbsp; **Tile {st.session_state.swipe_tile_idx + 1}/{len(valid_keys)}**")
                
                raw_img = current_grain.get("image_raw")
                overlay_img = current_grain.get("image_overlay")
                
                if mask_opacity > 0.0 and overlay_img is not None:
                    img_to_show = Image.blend(raw_img, overlay_img, mask_opacity)
                else:
                    img_to_show = raw_img
                    
                if brightness != 1.0:
                    img_to_show = ImageEnhance.Brightness(img_to_show).enhance(brightness)
                if contrast != 1.0:
                    img_to_show = ImageEnhance.Contrast(img_to_show).enhance(contrast)
                
                st.image(img_to_show, use_container_width=True)
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Giant Classification Buttons for Mobile Ergonomics
                col1, col2, col3 = st.columns(3)
                
                def classify_grain(cls_id):
                    gid = current_grain["id"]
                    prev_val = st.session_state.swipe_labels.get(gid)
                    st.session_state.swipe_history.append({"grain_id": gid, "label": prev_val})
                    if cls_id is not None:
                        st.session_state.swipe_labels[gid] = cls_id
                    else:
                        st.session_state.swipe_labels.pop(gid, None)
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

                st.markdown("<br>", unsafe_allow_html=True)
                
                # Undo & Discard Grain Row
                ctrl_col1, ctrl_col2 = st.columns(2)
                with ctrl_col1:
                    if st.button("↩️ Undo Last", key=f"btn_undo_{current_grain['id']}", use_container_width=True):
                        if st.session_state.swipe_grain_idx > 0:
                            st.session_state.swipe_grain_idx -= 1
                            if st.session_state.swipe_history:
                                last_item = st.session_state.swipe_history.pop()
                                gid = last_item["grain_id"]
                                if last_item["label"] is None:
                                    st.session_state.swipe_labels.pop(gid, None)
                                else:
                                    st.session_state.swipe_labels[gid] = last_item["label"]
                        elif st.session_state.swipe_tile_idx > 0:
                            st.session_state.swipe_tile_idx -= 1
                            st.session_state._current_swipe_key = None
                        st.rerun()
                with ctrl_col2:
                    if st.button("🗑️ Discard Label", key=f"btn_discard_grain_{current_grain['id']}", use_container_width=True):
                        classify_grain(None)
                        st.toast("Discarded grain label")
                        st.rerun()
                        
                # Relabel Tile & Discard Tile Row
                tile_ctrl1, tile_ctrl2 = st.columns(2)
                with tile_ctrl1:
                    if st.button("⚠️ Send Tile to Relabel", key=f"btn_relabel_tile_{current_grain['id']}", use_container_width=True):
                        move_s3_file(current_key, "needs_labeling")
                        st.session_state._current_swipe_key = None
                        st.session_state.swipe_grain_idx = 0
                        st.session_state.swipe_labels = {}
                        st.toast("Tile moved to Needs Labeling!")
                        st.rerun()
                with tile_ctrl2:
                    if st.button("🗑️ Discard Whole Tile", key=f"btn_discard_tile_{current_grain['id']}", use_container_width=True):
                        move_s3_file(current_key, "discarded")
                        st.session_state._current_swipe_key = None
                        st.session_state.swipe_grain_idx = 0
                        st.session_state.swipe_labels = {}
                        st.toast("Tile moved to Discarded!")
                        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# 🗂️ BROWSE CATEGORIES MODE
# Shows tiles already sorted into active_learning/{category}/ with paging
# and per-tile or bulk reassignment to any other category.
# ═══════════════════════════════════════════════════════════════════════════════

elif mode == "🗂️ Browse Categories":
    st.markdown("### 🗂️ Browse Accepted Tiles by Category")
    st.caption("Review tiles already moved to an active-learning category and reassign them if needed.")

    CAT_DISPLAY = {
        "hard_positives": "🌟 Hard Positives",
        "needs_labeling": "⚠️ Needs Labeling",
        "hard_negatives": "🌑 Hard Negatives",
        "discarded":      "🗑️ Discarded",
    }
    CAT_KEYS = list(CAT_DISPLAY.keys())
    PAGE_SIZE = 24

    # Per-category counts shown in tab labels
    tc, _ = get_grain_and_tile_counts()
    tab_labels = [f"{CAT_DISPLAY[c]} ({tc.get(c, '?')})" for c in CAT_KEYS]
    tabs = st.tabs(tab_labels)

    if "browse_page" not in st.session_state:
        st.session_state.browse_page = {c: 0 for c in CAT_KEYS}

    for tab, cat in zip(tabs, CAT_KEYS):
        with tab:
            page = st.session_state.browse_page.get(cat, 0)
            total, page_keys = list_category_keys(cat, page, PAGE_SIZE)

            if total == 0:
                st.info(f"No tiles in **{CAT_DISPLAY[cat]}** yet.")
                continue

            n_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)

            # ── Top bar: paging + bulk actions ────────────────────────────────
            bar_l, bar_m, bar_r = st.columns([3, 2, 3])
            with bar_l:
                st.caption(
                    f"Page **{page + 1}** / {n_pages} &nbsp;|&nbsp; "
                    f"**{total:,}** tiles total"
                )
            with bar_m:
                pcols = st.columns(2)
                with pcols[0]:
                    if st.button("◀ Prev", key=f"prev_{cat}", use_container_width=True,
                                 disabled=(page == 0)):
                        st.session_state.browse_page[cat] = page - 1
                        list_category_keys.clear()
                        st.rerun()
                with pcols[1]:
                    if st.button("Next ▶", key=f"next_{cat}", use_container_width=True,
                                 disabled=(page >= n_pages - 1)):
                        st.session_state.browse_page[cat] = page + 1
                        list_category_keys.clear()
                        st.rerun()
            with bar_r:
                other_cats = [c for c in CAT_KEYS if c != cat]
                bulk_target = st.selectbox(
                    "Bulk move page to",
                    options=other_cats,
                    format_func=lambda c: CAT_DISPLAY[c],
                    key=f"bulk_target_{cat}",
                    label_visibility="collapsed",
                )
                if st.button(
                    f"↪️ Move all {len(page_keys)} → {CAT_DISPLAY[bulk_target]}",
                    key=f"bulk_move_{cat}", use_container_width=True, type="primary"
                ):
                    moved = 0
                    with st.spinner(f"Moving {len(page_keys)} tiles…"):
                        for k in page_keys:
                            if move_s3_file(k, bulk_target):
                                moved += 1
                    list_category_keys.clear()
                    get_grain_and_tile_counts.clear()
                    st.toast(f"✅ Moved {moved} tiles → {CAT_DISPLAY[bulk_target]}")
                    st.rerun()

            st.markdown("---")

            # ── Fetch images + detections in parallel for this page ───────────
            keys_missing = [k for k in page_keys if k not in st.session_state.batch_images]
            if keys_missing:
                with st.spinner(f"Loading {len(keys_missing)} tile images…"):
                    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                        for k, img_bytes, det_json in executor.map(fetch_tile_and_detections, keys_missing):
                            if img_bytes:
                                st.session_state.batch_images[k] = img_bytes
                            if det_json is not None and k not in st.session_state.batch_results:
                                try:
                                    pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
                                    det_result = DetectionResult(det_json, pil_img.width, pil_img.height)
                                    st.session_state.batch_results[k] = [det_result]
                                except Exception:
                                    pass

            # ── 4-column tile grid ────────────────────────────────────────────
            n_cols = 4
            grid = st.columns(n_cols)

            for i, key in enumerate(page_keys):
                with grid[i % n_cols]:
                    filename = os.path.basename(key)

                    # Image with optional detection overlay
                    img_bytes = st.session_state.batch_images.get(key)
                    if img_bytes:
                        try:
                            pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
                            det_res  = st.session_state.batch_results.get(key)
                            if det_res and len(det_res[0].boxes) > 0:
                                cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                                boxes = det_res[0].boxes
                                cls_colors = {0: (0, 200, 0), 1: (0, 0, 220)}
                                for b in range(len(boxes)):
                                    x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[b]]
                                    color = cls_colors.get(int(boxes.cls[b]), (0, 200, 200))
                                    cv2.rectangle(cv_img, (x1, y1), (x2, y2), color, 2)
                                pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
                            short_name = filename[:28] + ("…" if len(filename) > 28 else "")
                            st.image(pil_img, use_container_width=True, caption=short_name)
                        except Exception:
                            st.caption("*(render error)*")
                    else:
                        st.caption("*(unavailable)*")

                    # Detection grain summary
                    det_res = st.session_state.batch_results.get(key)
                    if det_res and len(det_res[0].boxes) > 0:
                        n_v  = sum(1 for c in det_res[0].boxes.cls if int(c) == 0)
                        n_nv = sum(1 for c in det_res[0].boxes.cls if int(c) == 1)
                        st.caption(f"🟢 {n_v} viable  🔴 {n_nv} non-viable")

                    # Per-tile reassignment
                    safe_id = f"{cat}_{i}_{hash(key) % 99999}"
                    new_cat = st.selectbox(
                        "Move to →",
                        options=[c for c in CAT_KEYS if c != cat],
                        format_func=lambda c: CAT_DISPLAY[c],
                        key=f"sel_{safe_id}",
                        label_visibility="collapsed",
                    )
                    if st.button("↪️ Move", key=f"mv_{safe_id}", use_container_width=True):
                        with st.spinner("Moving tile…"):
                            move_s3_file(key, new_cat)
                        list_category_keys.clear()
                        get_grain_and_tile_counts.clear()
                        st.session_state.batch_images.pop(key, None)
                        st.session_state.batch_results.pop(key, None)
                        st.toast(f"✅ Moved → {CAT_DISPLAY[new_cat]}")
                        st.rerun()

                    st.markdown("<div style='margin-bottom:12px'></div>", unsafe_allow_html=True)
