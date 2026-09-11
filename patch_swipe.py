import re

with open("app_gui.py", "r") as f:
    content = f.read()

# Update sidebar radio
content = content.replace('["⌨️ Keyboard Mode", "🎨 Canvas Mode", "📋 Grid Mode", "👀 Review & Submit"]',
                          '["⌨️ Keyboard Mode", "🎨 Canvas Mode", "📋 Grid Mode", "📱 Swipe Mode", "👀 Review & Submit"]')

# Append Swipe Mode logic at the end
swipe_code = """
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
            if not getattr(st.session_state, '_current_swipe_key', None) == current_key:
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
                        txt_content = "\\n".join(lines)
                        # Replace .jpg or .czi with .txt
                        base_name = os.path.splitext(os.path.basename(current_key))[0]
                        # Upload directly to same prefix but as .txt
                        txt_key = current_key.rsplit('.', 1)[0] + '.txt'
                        s3.put_object(Bucket=bucket, Key=txt_key, Body=txt_content.encode('utf-8'))
                        st.toast(f"Saved {len(lines)} labels to S3!")
                        
                    st.session_state.swipe_tile_idx += 1
                    st.rerun()
            else:
                current_grain = grains[st.session_state.swipe_grain_idx]
                st.progress((st.session_state.swipe_grain_idx) / len(grains), text=f"Grain {st.session_state.swipe_grain_idx + 1} of {len(grains)}")
                
                # Big centered image
                st.image(current_grain["image"], use_column_width=True)
                
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
"""

if "elif mode == \\"📱 Swipe Mode\\":" not in content:
    content += swipe_code

with open("app_gui.py", "w") as f:
    f.write(content)
