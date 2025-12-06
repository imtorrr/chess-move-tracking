import os

import pandas as pd

from processing import (
    clean_move_history,
    derive_moves_from_history,
    generate_fen_history,
    generate_pgn,
    get_board_from_video,
)

if __name__ == "__main__":
    video_paths = [
        "data/videos/2_move_student.mp4",
        "data/videos/2_Move_rotate_student.mp4",
        "data/videos/4_Move_studet.mp4",
        "data/videos/6_Move_student.mp4",
        "data/videos/8_Move_student.mp4",
        "data/videos/Bonus Long Video Label.mp4",
    ]
    all_pgns = []
    model_path = "weights/chess-piece-yolo11l-tuned.pt"
    for video_path in video_paths:
        print(f"Processing video: {video_path}")
        filename = os.path.splitext(os.path.basename(video_path))[0]
        cache_board_path = f"stubs/{filename}.board.joblib"
        cache_history_path = f"stubs/{filename}.history.joblib"

        # Step 1: Detect or load the chessboard
        board = get_board_from_video(video_path, cache_board_path)
        if not board:
            print(f"Could not find or load a board for {video_path}. Skipping.")
            all_pgns.append("")
            continue

        # Step 2: Generate FEN history from the video
        history = generate_fen_history(video_path, board, cache_history_path, model_path, output_path=video_path.replace("data/videos", "results"))
        if not history:
            print(f"Could not generate FEN history for {video_path}. Skipping.")
            all_pgns.append("")
            continue

        # Step 3: Derive moves from FEN history
        moves = derive_moves_from_history(history)

        # Step 4: Clean up detected moves
        cleaned_moves = clean_move_history(moves)

        # Step 5: Generate PGN from the cleaned moves
        pgn_string = generate_pgn(history, cleaned_moves)
        print(f"Generated PGN: {pgn_string}\n")
        all_pgns.append(pgn_string)

    # Final Step: Save results to CSV
    print("Saving results to output.csv")
    filenames = [os.path.basename(video_path) for video_path in video_paths]
    df = pd.DataFrame({"row_id": filenames, "output": all_pgns})

    os.makedirs("results", exist_ok=True)
    df.to_csv("results/output.csv", index=False)
    print("Processing complete.")
