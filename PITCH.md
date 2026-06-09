# 5-Minute Pitch, Car-Type Recognition

**Live demo:** https://HanfuZhao781-car-recognition.hf.space
**Code:** https://github.com/hanfuzhao/Module1-CarRecognition

Speaker notes for a 5-minute, hard-stop pitch. Four beats: Problem, Approach, Live Demo, Results/Insights. All numbers are real (`data/outputs/metrics.json`).


## 1. Problem & Motivation (~45s)
- "What car is this?" from one photo, not just *car*, but the exact
 **make / model / year** (e.g. *Aston Martin Virage Coupe 2012*). 196 classes.
- This is **fine-grained**: classes differ by a grille, a badge, a trim. It
 powers real consumer features (marketplace listings, insurance, valuation).
- Two failure modes that matter in production:
 1. **Look-alikes**, rebadged twins and same-model trims cap accuracy.
 2. **Bad photos**, and a *confident wrong answer* is worse for a user than an
 honest "I'm not sure."

## 2. Approach Overview (~60s)
- Three models, as required, on the **real Stanford Cars** test set (8,041 imgs):
 - **Naive** (majority / random), the floor.
 - **Classical**, HOG features, linear SVM (no learned features).
 - **Deep (deployed)**, **frozen ImageNet ResNet50 as a feature extractor +
 a trained MLP head** (transfer learning). Freezing keeps the served
 artifact tiny (~4.6 MB) and the backbone is fetched at runtime.
- **Novelty = insight, not chasing SOTA:** I profile **corruption robustness**
 and add a **confidence-gating** abstain mechanism, directly targeting the two
 failure modes above.

## 3. Live Demo (~90s)
- Open the live app. Upload a clean car photo, correct top-1 with a high
 confidence bar + top-5 list.
- Upload a hard/look-alike or low-quality photo, confidence drops and the app
 **abstains**: *"I'm not sure, try a clearer, larger photo."*
- Talking point while it loads: inference only, public URL, runs on Hugging
 Face Spaces.

## 4. Results, Insights, Key Findings (~75s)
- **Accuracy:** Deep model **52% top-1 / 80% top-5**, vs **4.4%** for HOG+SVM and
 **0.85%** naive, transfer learning is doing real work; HOG can't crack
 fine-grained.
- **The model learned real structure:** its top confusions are
 *Chevrolet Express and GMC Savana* and *Dodge and Mercedes Sprinter*, literally
 the **same vans sold under two brands**.
- **Robustness experiment (key finding):** robust to JPEG (+2pt) and motion blur
 (+1pt), -6pt to noise, but **pixelation is catastrophic: -51pt to near-chance.**
 Thumbnails/over-compressed inputs fail silently.
- **Confidence gating recovers trust:** abstaining on low-confidence inputs lifts
 accuracy to **79% @ 43% coverage** and **90% @ 19% coverage**, a 52% model
 becomes a 90%-on-accepted product experience, no retraining.

## Closing line
"It's an honest, deployable assistant: it tells you its best guesses, shows how
sure it is, and asks for a better photo instead of bluffing."


### Q&A backup facts
- Why frozen backbone? 3-day budget + tiny deployable artifact; fine-tuning is
 the obvious next step (expected +25-35 pts).
- Why robustness, not long-tail? Stanford Cars is near-balanced (max/min = 2.83;
 measured head/tail gap 0.1%), so long-tail isn't a real effect here.
- Data: Stanford Cars (US-market, <=2012), a stated limitation in the report.
