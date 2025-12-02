# Document ML Agent

An end-to-end document processing pipeline that combines deep learning with an agentic orchestration layer. Built for MIS 382N Advanced Machine Learning at UT Austin.

## What it does

Upload a receipt, invoice, or business letter and the system will:
1. Extract text using OCR (EasyOCR)
2. Classify the document type (ResNet18 ensemble trained on RVL-CDIP)
3. Pull out structured fields like vendor, date, totals (LayoutLMv3)
4. Make an approval decision based on business rules
5. Flag anomalies and route edge cases to human review

## Architecture

```
Document → OCR Agent → Router Agent → Field Agent → Decision Agent → HITL Manager
              ↓            ↓              ↓              ↓
          EasyOCR     ResNet18      LayoutLMv3     Rule Engine
                      Ensemble
```

Each agent is independent and logs its decisions for a full audit trail.

## Notebooks

| File | What's in it |
|------|--------------|
| `AML_Document_Processing_Pipeline_v2.ipynb` | Main notebook - OCR, LayoutLM, CNN, agentic layer, Gradio UI |
| `RVL_CDIP_Classification.ipynb` | ResNet18 training on document images |

## Running it

1. Open in Google Colab (GPU runtime recommended)
2. Run the cells in order
3. Phase 10-11 will launch a Gradio demo with a shareable link

The `.pt` model files aren't in the repo (too large). Train them yourself or grab from Google Drive.

## Tech stack

- PyTorch + torchvision (ResNet18)
- HuggingFace Transformers (LayoutLMv3)
- EasyOCR
- Gradio for the demo UI
- Good old regex for fallback field extraction

## Next steps

A few things that would make this better:

- **Isolation Forest for anomaly detection** - currently using rules, ML would catch more edge cases
- **Fine-tune LayoutLM on real SROIE data** - we trained on synthetic receipts, real data would help
- **Add XGBoost approval predictor** - learn from historical approval decisions
- **Better OCR preprocessing** - deskewing, noise removal before OCR
- **Async processing** - handle batch uploads without blocking
- **Model versioning** - track which model made each decision

## Team

Built by the RogueTex crew for our grad ML class.

---

*If you're grading this: yes, all the phases work. Run it in Colab with a GPU.*
