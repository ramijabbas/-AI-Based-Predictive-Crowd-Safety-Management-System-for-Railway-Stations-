# RailVision — AI Crowd & Safety Management at Railway Stations

RailVision transforms existing CCTV infrastructure from a **passive surveillance system** into an **AI-powered predictive safety and crowd-management system**, helping railway authorities identify risks *before* they develop into major incidents.

> 🚀 **Live demo:** https://tamalika093.github.io/railvision/

---

## 🎯 Problem Statement

Railway stations face crowd-management and safety challenges during peak hours, festivals, government exams, and large events. Risks include:

- **Overcrowding / stampede** risk at platforms and entry/exit zones
- **Track incursions** and people crossing tracks
- **Abandoned / unattended baggage**
- **Sudden panic-like crowd movement**
- **Unexpected platform changes** onto already-crowded platforms
- **Gaps** between ticketing/entry data and the number of people physically present

Existing CCTV is used only reactively (for investigation) rather than proactively for live safety decisions.

---

## 💡 Solution

Using **existing CCTV camera footage + AI / computer vision**, RailVision provides:

- Real-time **people counting** and **crowd-density** estimation per platform/area
- **Heat maps** of crowded zones to guide staff deployment
- **Early-warning alerts** when density or movement patterns indicate an **overcrowding / stampede risk**
- **Ticketing correlation** — compares CCTV headcount vs. ticket/entry data to flag unusual differences and prioritise monitoring/checking
- **Crowd forecasting** — uses historical + real-time data (festivals, exams, holidays) to predict crowd levels and peak timings, enabling planning for extra trains, staff, platform allocation, and entry/exit management
- **Unexpected platform-change alerts** — instant alert if a train arrives on an already-crowded platform
- Safety modules for **track-incursion detection**, **abandoned baggage**, and **abnormal/panic movement**
- A **central control-room dashboard** showing live density, risk levels, alerts, and predictions

---

## 🖥️ The Dashboard (Member 4 — Main Dashboard)

The dashboard shown to the judges, built with **HTML + CSS + JavaScript (React-compatible single-file app)**.

### Features
| Feature | What it shows |
|---------|---------------|
| **Live Crowd Count** | Real-time total headcount across all platforms (vs. capacity) |
| **Platform Status** | 8 platforms with occupancy bars & NORMAL / MODERATE / HIGH / CRITICAL status |
| **Crowd Heatmap** | Color-coded live heat cells per platform |
| **Crowd Graphs** | Live density trend vs. normal-peak baseline |
| **Risk Level** | Overall system risk indicator (LOW → CRITICAL) |
| **Safety Alerts** | Live feed: stampede risk, track incursion, abandoned baggage, panic movement, unexpected platform change, ticketing mismatch |
| **Crowd Prediction** | Next-6-hour forecast + staff / train / platform recommendations |
| **Smart Safety Modules** | Track incursion, baggage, panic, people counting, ticketing correlation |
| **Live CCTV Feed** | AI-detection visualizer cycling through platform cameras |

> **Demo mode:** all data is realistically *simulated live* in the browser — no real CCTV needed — so it showcases every feature working end-to-end. It updates automatically every ~1.5s and is fully responsive (works on phones/tablets).

---

## 🛠️ Tech Stack

- **Frontend:** HTML5, CSS3, JavaScript (vanilla — no build step, single deployable file)
- **Data simulation:** In-browser real-time generator (density, alerts, forecasting, CCTV visualization)
- **Hosting:** GitHub Pages (CI/CD via GitHub Actions)
- **Charts:** Native HTML5 Canvas (no external libraries required)

---

## 📁 Project Structure

```
RailVision/
├── index.html                  # The complete self-contained dashboard
└── .github/
    └── workflows/
        └── deploy.yml          # CI/CD — auto-deploys to GitHub Pages on push
```

---

## 🚀 How to Run / Deploy

### Run locally
Open `index.html` directly in any modern browser. No server required.

Or serve it locally:
```bash
# Python
python -m http.server 8000

# Node
npx serve .
```
Then visit `http://localhost:8000`.

### Deploy to GitHub Pages
1. Push this repo to GitHub.
2. The included GitHub Actions workflow (`.github/workflows/deploy.yml`) auto-builds and deploys to Pages.
3. Go to **Settings → Pages → Build and deployment → Source: GitHub Actions**. The site goes live at:
   `https://<username>.github.io/<repo>/`
4. Redeploy automatically on every `git push`.

---

## 🔁 Customising the Demo

| Want to change… | Where in `index.html` |
|------------------|------------------------|
| Number/capacity of platforms | `const P = [...]` (platform definitions) |
| Update frequency | `setInterval(step, 1500)` |
| Alert types | `const alertTypes = [...]` |
| Forecast scenario (festival build-up) | `renderPrediction()` |
| Camera rotation | `setInterval(()=>{ cameraIndex=(cameraIndex+1)%8 }, 4000)` |

---

## 🌟 Future Work / Real Implementation

- Train a YOLO/DETR model on station-specific CCTV for robust people counting & tracking (ByteTrack/DeepSORT)
- Integrate live CCTV RTSP feeds and IRCTC/ticketing APIs (UTS, reservation data)
- Add density thresholds tuned to station layout & safety norms
- Send alerts via SMS / dashboard to a central control room
- Multi-station aggregation for zonal control centres

---

## 👩‍💻 Team

- **Member 1 —** Computer Vision / Crowd Counting
- **Member 2 —** Alert & Prediction Engine
- **Member 3 —** Ticketing Correlation & Backend
- **Member 4 —** Main Dashboard (this repository's focus)

---

Made with ❤️ for the Hackathon · **AI Crowd & Safety Management at Railway Stations**
