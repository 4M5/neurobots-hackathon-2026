# Exam Behavior Anomaly Detection

An AI-powered system to detect cheating behavior in online exams using **Isolation Forest** algorithm.

## 📋 Overview

This project demonstrates anomaly detection for online exam proctoring by analyzing behavioral patterns:

- **Normal Students**: Consistent typing, minimal tab switches, focused exam-taking
- **Cheating Behavior**: Copy-pasting, frequent tab switches, irregular typing patterns

The system uses unsupervised machine learning (IsolationForest) to learn what "normal" looks like, then flags deviations as potential cheating.

## 🔧 Setup Instructions

### Step 1: Create the Conda Environment

Open Anaconda Prompt (or terminal) and run:

```bash
# Create new environment with Python 3.10
conda create -n exam_hack python=3.10 -y

# Activate the environment
conda activate exam_hack
```

### Step 2: Install Dependencies

```bash
# Install required packages
pip install numpy pandas scikit-learn joblib flask google-generativeai python-dotenv
```

Or using conda:

```bash
conda install numpy pandas scikit-learn joblib -y
pip install flask google-generativeai python-dotenv
```

### Step 3: Setup Gemini API (Optional)

1. Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a `.env` file in the project root:
   ```bash
   cp .env.example .env
   ```
3. Edit `.env` and add your key:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

### Step 4: Navigate to Project Folder

```bash
cd path/to/project_root
```

## 🚀 Running the Project

Run the scripts **in order**:

### 1️⃣ Generate Dataset

```bash
python generate_dataset.py
```

**What it does:**
- Creates synthetic exam behavior data
- Generates 800 normal + 200 cheating samples
- Saves as `behavior_dataset.csv`

**Output:** You'll see statistics about the generated dataset

### 2️⃣ Train the Model

```bash
python train_model.py
```

**What it does:**
- Loads the dataset
- Trains IsolationForest on NORMAL samples only
- Saves model as `behavior_model.pkl`

**Output:** Training progress and model evaluation

### 3️⃣ Test the Model

```bash
python test_model.py
```

**What it does:**
- Loads the trained model
- Tests on suspicious sample cases
- Evaluates on full dataset
- Optional: Interactive custom testing

**Output:** Per-sample predictions and accuracy metrics

## 📊 Understanding the Output

### Predictions

| Output | Meaning |
|--------|---------|
| ✅ NORMAL | Behavior is within expected patterns |
| 🚨 ANOMALY | Behavior deviates significantly (potential cheating) |

### Anomaly Score

- **Higher scores (closer to 0)**: More normal behavior
- **Lower scores (negative)**: More anomalous behavior

### Features Analyzed

| Feature | Description |
|---------|-------------|
| `avg_key_interval` | Average time between keystrokes (seconds) |
| `std_key_interval` | Consistency of typing rhythm |
| `typing_speed` | Characters per minute |
| `backspace_rate` | Ratio of corrections to total keystrokes |
| `paste_count` | Number of paste operations |
| `tab_switch` | Number of tab/window switches |
| `focus_loss` | Times the exam window lost focus |
| `avg_idle` | Average idle time between actions |
| `max_idle` | Longest idle period |
| `edit_count` | Number of answer modifications |
| `time_to_first_key` | Time from question display to first key |
| `answer_duration` | Total time spent on answer |

## 🧠 How IsolationForest Works

1. **Training Phase**: The model builds "isolation trees" from normal behavior samples
2. **Each tree** randomly selects features and split points to isolate data points
3. **Normal points** are harder to isolate (require more splits)
4. **Anomalies** are easier to isolate (require fewer splits)
5. **Detection**: New samples requiring few splits → flagged as anomalies

## 📁 Project Structure

```
/
├── README.md              # This file
├── generate_dataset.py    # Creates synthetic training data
├── train_model.py         # Trains the IsolationForest model
├── test_model.py          # Tests and evaluates the model
├── behavior_dataset.csv   # Generated after running generate_dataset.py
├── behavior_model.pkl     # Generated after running train_model.py
├── server.py              # Flask server with Gemini API integration
├── .env.example           # Template for environment variables
├── index.html             # Web demo interface
├── app.js                 # Web demo logic
└── styles.css             # Web demo styling
```

## ✨ Key Features

- ✅ **Fully Offline**: No internet required after setup
- ✅ **No Deep Learning**: Simple, interpretable IsolationForest
- ✅ **Clean Code**: Well-commented and easy to understand
- ✅ **Demo Ready**: Works out of the box for hackathon demos
- ✅ **Interactive Testing**: Test custom behavior patterns

## 📝 Notes for Hackathon

- The dataset is **synthetic** for demonstration purposes
- In production, you would collect real behavioral data
- The model can be retrained as more data becomes available
- Parameters can be tuned for different sensitivity levels

## 🔬 Extending the Project

Ideas for enhancement:
- Add real-time monitoring capabilities
- Create a web dashboard for proctors
- Add more behavioral features (mouse movement, webcam analysis)
- Implement ensemble methods with multiple anomaly detectors

---

**Built for Hackathon Demo** | Python 3.10 | IsolationForest | scikit-learn

