import argparse
import os
import pandas as pd


def process_video(
    video_path: str, model_path: str, results_dir: str = "results", print_ascii=False
) -> str | None:
    """
    Processes a single video to extract chess moves and generate a PGN string.

    Args:
        video_path: Path to the video file.
        model_path: Path to the chess piece detection model.
        results_dir: Directory to save result videos.

    Returns:
        The generated PGN string, or None if processing fails.
    """
    print(f"Processing video: {video_path}")
    from processing import (
        clean_move_history,
        derive_moves_from_history,
        generate_fen_history,
        generate_pgn,
        get_board_from_video,
    )

    filename = os.path.splitext(os.path.basename(video_path))[0]

    stubs_dir = "stubs"
    os.makedirs(stubs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    cache_board_path = os.path.join(stubs_dir, f"{filename}.board.joblib")
    cache_history_path = os.path.join(stubs_dir, f"{filename}.history.joblib")

    # Step 1: Detect or load the chessboard
    board = get_board_from_video(video_path, cache_board_path)
    if not board:
        print(f"Could not find or load a board for {video_path}. Skipping.")
        return None

    # Step 2: Generate FEN history from the video
    video_output_path = os.path.join(results_dir, os.path.basename(video_path))
    history = generate_fen_history(
        video_path,
        board,
        cache_history_path,
        model_path,
        output_path=video_output_path,
        print_ascii=print_ascii
    )
    if not history:
        print(f"Could not generate FEN history for {video_path}. Skipping.")
        return None

    # Step 3: Derive moves from FEN history
    moves = derive_moves_from_history(history)

    # Step 4: Clean up detected moves
    cleaned_moves = clean_move_history(moves)

    # Step 5: Generate PGN from the cleaned moves
    pgn_string = generate_pgn(history, cleaned_moves)

    return pgn_string


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process chess game videos to generate PGNs."
    )
    parser.add_argument(
        "--video-inputs",
        required=True,
        nargs="+",
        help="Paths to video files or directories containing video files.",
    )
    parser.add_argument(
        "--model-path",
        default="weights/chess-piece-yolo11l-tuned.pt",
        help="Path to the chess piece detection model.",
    )
    parser.add_argument(
        "--output-file",
        help="Path to save the output CSV file. If not provided, will print PGNs to stdout.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory to save annotated videos.",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="StdOut to console with ascii characters.",
    )

    args = parser.parse_args()
    
    all_pgns = []
    filenames = []

    video_files = []
    for video_input in args.video_inputs:
        if os.path.isdir(video_input):
            for filename in os.listdir(video_input):
                if filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                    video_files.append(os.path.join(video_input, filename))
        else:
            video_files.append(video_input)

    for video_path in video_files:
        pgn = process_video(video_path, args.model_path, args.results_dir, print_ascii=args.ascii)
        all_pgns.append(pgn if pgn else "")
        filenames.append(os.path.basename(video_path))

    if args.output_file:
        # Ensure the output directory exists
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        df = pd.DataFrame({"row_id": filenames, "output": all_pgns})
        df.to_csv(args.output_file, index=False)
        print(f"Results saved to {args.output_file}")
    else:
        for filename, pgn in zip(filenames, all_pgns):
            print(f"\n--- PGN for {filename} ---")
            print(pgn)

    print("Processing complete.")
