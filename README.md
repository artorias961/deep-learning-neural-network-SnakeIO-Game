# 🐍 Deep Learning Neural Network SnakeIO Game

This repository contains the code and notebooks for training and evaluating a **deep learning neural network to play a SnakeIO-style game**.  
This project was completed as the **final project for my Neural Networks course**.

---

## 🚀 Project Overview

The goal of this project is to design and train a neural network capable of autonomously playing a Snake-style game.  
The model learns how to navigate the environment, collect food, and avoid collisions based on game-state inputs.

This project demonstrates:
- Neural network design and implementation
- Training and evaluation of a deep learning model
- Application of machine learning concepts to a game environment

---

## 🧠 High-Level Architecture

Typical data flow in this project:

Game State → Neural Network → Action Output  
(e.g., direction to move the snake)

A block diagram of the architecture is included in the repository as:

```
┌──────────────────────┐
│  Game Environment    │
│  (Snake Grid World)  │
└──────────┬───────────┘
           │  current state
           ▼
┌──────────────────────────────┐
│  State Extraction / Encoding │
│  (features from game state)  │
└──────────┬───────────────────┘
           │  state vector
           ▼
┌──────────────────────────────┐
│ Neural Network Policy /      │
│ Q-Model (my_model.py)        │
└──────────┬───────────────────┘
           │  predicted action
           ▼
┌──────────────────────────────┐
│        Action Selection      │
│   (Up / Down / Left / Right) │
└──────────┬───────────────────┘
           │  apply action
           ▼
┌──────────────────────┐
│  Game Environment    │
│ (next state, reward) │
└──────────┬───────────┘
           │  (state, action,
           │   reward, next_state)
           ▼
┌────────────────────────────────────┐
│ Training Data / Experience Replay  │
│ (collected transitions)            │
└──────────┬─────────────────────────┘
           │  training batch
           ▼
┌──────────────────────────────┐
│      Loss + Optimizer        │
│   (backpropagation update)   │
└──────────┬───────────────────┘
           │  updated weights
           ▼
┌──────────────────────────────┐
│ Neural Network Policy /      │
│ Q-Model (updated)            │
└──────────┬───────────────────┘
           │  save / load
           ▼
┌──────────────────────────────┐
│        Model Checkpoints     │
│      (saved parameters)      │
└──────────────────────────────┘

```

---

## 🗂 Repository Structure

| File / Folder | Description |
|---------------|-------------|
| `final_project_v*.ipynb` | Jupyter notebooks for training and evaluation |
| `my_model.py` | Neural network model implementation |
| `block_diagram.drawio` | High-level architecture diagram |
| `README.md` | Project documentation |

---

## 🧪 How to Run

1. **Clone the repository**
```bash
git clone https://github.com/artorias961/deep-learning-neural-network-SnakeIO-Game.git
cd deep-learning-neural-network-SnakeIO-Game
```

2. **Install dependencies**
```bash
pip install numpy pandas matplotlib pytorch
```

3. **Run the notebooks**
Open the final project notebooks (`final_project_v*.ipynb`) and execute the cells to train and evaluate the model.

---

## 📊 Results

The notebooks include visualizations such as:
- Training loss curves
- Performance metrics over time
- Behavioral evaluation of the snake agent

These plots demonstrate how the neural network improves its gameplay through training.

---

## 📚 Academic Context

This project was developed as the **final deliverable for a Neural Networks course**.  
It focuses on applying theoretical concepts learned in class to a practical, interactive problem.

---

## 🛠️ Future Improvements

- Reinforcement learning with reward shaping
- Real-time graphical visualization of gameplay
- Hyperparameter tuning and model optimization

---

## 🧑‍💻 Author

Created by **artorias961**  
Final project – Neural Networks course
