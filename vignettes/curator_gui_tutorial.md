# 🌸 Pollen Curator GUI — User Guide & Tutorial

Welcome to the **Pollen Curator GUI**! This web application allows biologists and data annotators to inspect, curate, and balance pollen viability datasets on both mobile phones and desktop computers.

---

## 💡 Key Architectural Concepts

- **📋 Grid Mode (Tile-Level Presence Confirmation)**:
  - Grid Mode is used **strictly for tile-level filtering**: confirming whether a 640x640 tile contains pollen grains (`🌟 Pollen Present`) or is empty background (`🌑 No Pollen`).
  - Viability is **NEVER assumed automatically** at the tile level.
- **📱 Swipe Mode (Individual Pollen Grain Viability Identification)**:
  - Viability (Viable 🟩 vs Non-Viable 🟥 vs Aborted 🟨) is identified **per individual pollen grain** in Swipe Mode.

---

## 📱 Mobile Curation Workflows

### 📱 Swipe Mode (Individual Pollen Grain Viability)
Swipe Mode is optimized for single-thumb mobile curation:

1. **Magnified Grain View**: Each pollen grain crop is rendered with optional SAM (Segment Anything Model) outline overlays and model confidence.
2. **One-Tap Viability Classification**:
   - 🟩 **Viable**: Magenta/dark purple, plump, full cytoplasm.
   - 🟥 **Non-Viable**: Green/pale, empty shell, shriveled.
   - 🟨 **Aborted**: Faint pink/yellow, partial cytoplasm.
3. **Control Actions**:
   - **`↩️ Undo Last`**: Reverts your last grain classification or steps back to the previous grain/tile.
   - **`🗑️ Discard Label`**: Omits an ambiguous crop without assigning a class.
   - **`⚠️ Send Tile to Relabel`**: Moves the whole tile to `active_learning/needs_labeling/` in S3.
   - **`🗑️ Discard Whole Tile`**: Moves blurry or debris-filled tiles to `active_learning/discarded/`.

---

## 📋 Grid & Keyboard Modes

### 📋 Phone-Friendly Grid Mode (Pollen Confirmation)
- Choose column density (`📱 2 Columns` for mobile screens, `🖥️ 4 Columns` for desktop).
- Tap high-contrast tile confirmation buttons:
  - **`🌟 Pollen Present`**: Tile contains pollen grains (moved to positive active learning batch).
  - **`🌑 No Pollen`**: Empty background tile.
  - **`⚠️ Needs Review`**: Tile requires expert inspection.
  - **`🗑️ Discard Tile`**: Tile is unusable (debris/out of focus).
- **`📱 Curate Grains in Swipe Mode`**: Tap directly on any tile card to jump straight into Swipe Mode for grain-by-grain viability curation.

### ⌨️ Desktop Keyboard Mode
- **Left / Right Arrow Keys**: Cycle tile classification category.
- **Spacebar**: Advance to the next tile in batch.
- **Enter**: Submit batch to S3.
- **`↩️ Undo Tile`**: Step back tile index.

---

## 🎯 Dataset Balancing & Sample Prioritization

Natural pollen samples are heavily imbalanced (~96.8% viable vs ~3.2% non-viable).

To accelerate training data collection for rare non-viable grains:
- Historical dataset statistics ([src/sample_viability_index.json](file:///home/meow/Documents/Antigravity/Pollen_viability/src/sample_viability_index.json)) track per-sample non-viable yields.
- **Top Non-Viable Samples**:
  - `1-6-J`: **88.4% non-viable** (975 non-viable grains)
  - `7-9-F`: **66.9% non-viable** (176 non-viable grains)
  - `6-1-F`: **53.8% non-viable** (154 non-viable grains)
- **Automatic Queue Sorting**: Enabling **`🎯 Prioritize High Non-Viable Samples`** sorts incoming S3 tiles so tiles rich in non-viable grains appear first.

---

## ☁️ S3 Storage Structure

Tiles and labels are organized in S3 as follows:
- Pending tiles: `Ostatni/Pollen_viability/tiles_640/`
- Active learning output: `Ostatni/Pollen_viability/active_learning/`
  - `hard_positives/`: Verified tiles + `.txt` segmentation masks
  - `needs_labeling/`: Tiles requiring expert review
  - `hard_negatives/`: Background tiles with zero pollen
  - `discarded/`: Rejected tiles
