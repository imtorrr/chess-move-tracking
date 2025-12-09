import os

import chess
import cv2
import joblib
import torch
from ultralytics import YOLO

from board_perception import BoardPerception
from chessboard import ChessBoard
from chesspieces import ChessPieces

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def get_board_from_video(video_path: str, cache_path: str) -> ChessBoard | None:
    """
    Detects or loads a cached chessboard from a video.

    Args:
        video_path: Path to the video file.
        cache_path: Path to the cache file for the board.

    Returns:
        A ChessBoard object or None if not found.
    """
    if os.path.exists(cache_path):
        print(f"Loading cached board from {cache_path}")
        return joblib.load(cache_path)

    print("Cached board not found, detecting from video...")
    cap = cv2.VideoCapture(video_path)
    board = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        board = ChessBoard()
        if board.find_board_with_rotation_correction(frame):
            print(f"Board found with orientation: {board.orientation}")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            joblib.dump(board, cache_path)
            print(f"Board cached to {cache_path}")
            break
        else:
            print("Could not find chessboard in this frame.")
    cap.release()
    return board


def generate_fen_history(
    video_path: str, board: ChessBoard, cache_path: str, model_path: str, output_path: str | None = None,
) -> list[str]:
    """
    Generates a history of board states in FEN format from a video.

    Args:
        video_path: Path to the video file.
        board: The ChessBoard object for the video.
        cache_path: Path to the cache file for the FEN history.

    Returns:
        A list of FEN strings.
    """
    if os.path.exists(cache_path):
        print(f"Load cached history from {cache_path}")
        return joblib.load(cache_path)

    chess_pieces_detector = ChessPieces(model_path=model_path)
    board_perceptor = BoardPerception(buffer_size=10)
    person_detector = YOLO("yolo11l.pt")
    history = []

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = None
    if output_path is not None:
        frame_rate = 30
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (frame_width, frame_height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

        # Note: isColor is set to False because we are writing a grayscale video.
        out = cv2.VideoWriter(output_path, fourcc, frame_rate, frame_size, isColor=True)

        print(f"Writing video to {output_path} using 'mp4v' codec at {frame_rate} FPS...")
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if board.rotation is not None:
            frame = cv2.rotate(frame, board.rotation)
        i += 1

        if i % 3 != 0:
            continue

        found_person = False
        results = person_detector.predict(
            frame, save=False, verbose=False, device=device
        )
        for box in results[0].boxes:
            person = person_detector.names[int(box.cls)]
            if person == "person":
                found_person = True
                break

        draw_frame = frame.copy()

        if not found_person and board is not None:
            board_idx, draw_frame = chess_pieces_detector.detect_frame(
                frame, return_plot=True, board=board
            )
            fen_curr = chess_pieces_detector.board_to_fen(board_idx, board)
            board_curr = chess.Board(fen_curr) # A chess library for Python
            if (
                board_curr.is_valid()
                or board_curr.status() == chess.STATUS_OPPOSITE_CHECK
            ):
                board_perceptor.add_frame_observation(board_idx)
            else:
                print(f"Invalid board state: {board_curr.status()}")

            stable_grid, is_stable = board_perceptor.get_stable_board()

            if stable_grid is not None and is_stable:
                fen_string = chess_pieces_detector.board_to_fen(stable_grid, board)
                history.append(fen_string)
                print(f"frame: {i}/{total_frames} (stable)")
        # Optional: display frame for debugging
        cv2.imshow("frame", draw_frame)
        if out is not None:
            out.write(draw_frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    if out is not None:
        out.release()
    joblib.dump(history, cache_path)
    print(f"Save history to cached to {cache_path}")
    cv2.destroyAllWindows()
    return history


def derive_moves_from_history(history: list[str]) -> list[tuple[chess.Move, bool]]:
    """
    Analyzes the FEN history to detect chess moves.

    Args:
        history: A list of FEN strings.

    Returns:
        A list of tuples containing the detected move and the turn.
    """
    moves = []
    for i in range(len(history) - 1):
        fen_prev = history[i]
        fen_curr = history[i + 1]

        board_prev = chess.Board(fen_prev)
        board_curr = chess.Board(fen_curr)

        map_prev = board_prev.piece_map()
        map_curr = board_curr.piece_map()

        detected_turn = None
        diffs = []
        for square in range(64):
            piece_prev = map_prev.get(square)
            piece_curr = map_curr.get(square)

            if piece_prev != piece_curr:
                diffs.append(square)
                if piece_prev is not None:
                    if piece_curr is None:
                        detected_turn = piece_prev.color
                    elif piece_curr.color != piece_prev.color:
                        detected_turn = piece_curr.color

        if detected_turn is not None:
            board_prev.turn = detected_turn
            found_move = None
            for move in board_prev.legal_moves:
                if board_prev.piece_at(move.from_square) == board_curr.piece_at(
                    move.to_square
                ):
                    found_move = move
                    break

            if len(diffs) == 2 and found_move is None:
                sq_1, sq_2 = diffs[0], diffs[1]
                candidates = [(sq_1, sq_2), (sq_2, sq_1)]
                for from_sq, to_sq in candidates:
                    piece = board_prev.piece_at(from_sq)
                    if piece is None:
                        continue
                    move = chess.Move(from_sq, to_sq)
                    if board_prev.piece_at(move.from_square) == board_curr.piece_at(
                        move.to_square
                    ):
                        found_move = move

            if found_move:
                moves.append((found_move, detected_turn))
    return moves


def clean_move_history(
    moves: list[tuple[chess.Move, bool]],
) -> list[tuple[chess.Move, bool]]:
    """
    Filters out noisy or incorrect move detections.

    Args:
        moves: The list of detected moves.

    Returns:
        A cleaned list of moves.
    """
    cleaned_moves = moves.copy()
    for i in range(len(cleaned_moves) - 1):
        move_curr, turn_curr = cleaned_moves[i]
        if move_curr is not None and turn_curr is not None:
            move_next, turn_next = cleaned_moves[i + 1]
            if turn_curr == turn_next:
                if (move_curr.from_square == move_next.to_square) and (
                    move_curr.to_square == move_next.from_square
                ):
                    cleaned_moves[i] = (None, None)
                    cleaned_moves[i + 1] = (None, None)
    return [m for m in cleaned_moves if m[0] is not None]


def generate_pgn(history: list[str], legal_moves: list[tuple[chess.Move, bool]]) -> str:
    """
    Converts a clean move list into a standard PGN string.

    Args:
        history: The FEN history (for starting position).
        legal_moves: The cleaned list of moves.

    Returns:
        The PGN string.
    """
    if not legal_moves:
        return ""

    
    try:
        ini_move, ini_turn = legal_moves[0]
        board_ini = chess.Board(history[0])
        board_ini.turn = ini_turn
        pgn_string = board_ini.variation_san([ move for (move, _) in legal_moves])
        pgn_string = pgn_string.replace("...", "... ") # if black move first
        print("board.variation_san success")
    except chess.IllegalMoveError as e:
        print(f"boarc.variation_san errors: {e}")
        ini_move, ini_turn = legal_moves[0]
        board_ini = chess.Board(history[0])
        board_ini.turn = ini_turn
        n_move = 0
        if not ini_turn:  # Black moves first
            legal_moves.pop(0)
            san = board_ini.san(ini_move)
            board_ini.push(ini_move)
            pgn_string = f"1... {san} "
            n_move = 1

        for i in range(0, len(legal_moves), 2):
            n_move += 1
            try:
                white_move, _ = legal_moves[i]
                san_white = board_ini.san(white_move)
                board_ini.push(white_move)

                san_black = ""
                if i + 1 < len(legal_moves):
                    black_move, _ = legal_moves[i + 1]
                    san_black = board_ini.san(black_move)
                    board_ini.push(black_move)

                pgn_string += f"{n_move}. {san_white} {san_black} "
            except chess.IllegalMoveError:
                print("Warning: Illegal move found during PGN generation. Skipping.")
                continue
            except IndexError:
                pgn_string += f"{n_move}. {san_white} "

    return pgn_string.strip()
