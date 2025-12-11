# [DIGITAL IMAGE PROCESSING] Chess Move Tracking

This project is a computer vision-based system to automatically track chess moves from a video recording of a game. It detects the chessboard, recognizes the pieces, and records the moves in standard chess notation (PGN).

## Main Idea

The core of this project is a pipeline that transforms video frames into a structured chess game history. This is achieved through the following steps:

1.  **Board Perception**: The system first analyzes the video to detect the location and orientation of the chessboard.
2.  **Piece Recognition**: A YOLOv11 model, fine-tuned for chess pieces, is used to detect the position of each piece on the board in every frame.
3.  **State Generation**: The detected piece positions are used to generate a Forsyth-Edwards Notation (FEN) string for each frame. FEN is a standard notation for describing a particular board position.
4.  **Move Derivation**: By comparing consecutive FEN strings, the system identifies the moves made during the game.
5.  **PGN Generation**: The sequence of identified moves is cleaned and compiled into a final Portable Game Notation (PGN) file, which can be used in any standard chess software.

This repository contains the source code for Team XYZ, which participated in the CU Chess Detection 2025 Kaggle competition. The competition has concluded, and the leaderboard can be viewed here: https://www.kaggle.com/competitions/cu-chess-detection-2025/leaderboard.

![Chessboard Detection](https://via.placeholder.com/600x400.png?text=Replace+with+your+image+of+chessboard+detection)

## Installation

This project uses `uv` for package management and specifies the Python version.

1.  **Clone the Repository**: Start by cloning the repository to your local machine.

    ```bash
    git clone https://github.com/imtorrr/chess-move-tracking.git
    cd chess-move-tracking
    ```

2.  **Install Dependencies**: Use `uv` to install the dependencies listed in `pyproject.toml`.

    ```bash
    uv sync
    source .venv/bin/activate
    ```

## How to Run

1.  **Download Data and Models**: The project requires pre-trained models and sample videos. Use the provided scripts to download them.

    ```bash
    bash scripts/download_videos.sh
    ```

2.  **Run Processing**: To run the move tracking on the sample videos, execute the `main.py` script.

    ```bash
    python main.py
    ```

    The output will be a CSV file located at `results/output.csv` containing the PGN for each processed video.

3.  **(Optional) Train the Model**: You can re-train the YOLO model using the provided training script.

    ```bash
    bash scripts/download_chess-pieces.sh
    bash scripts/download_video_labelled.sh
    bash scripts/download_xcorners.sh

    bash scripts/train.sh
    ```
    The training results will be saved in the `runs/detect/` directory.

## Project Structure

```
.
├── data/                  # Data for training and processing
├── results/               # Output from the processing
├── runs/                  # Output from model training
├── scripts/               # Helper scripts for downloading data, training, etc.
├── weights/               # Pre-trained model weights
├── board_perception.py    # Logic for detecting the chessboard
├── chesspieces.py         # Utilities related to chess pieces
├── main.py                # Main script to run the processing pipeline
├── processing.py          # Core processing logic
└── pyproject.toml         # Project dependencies
```
