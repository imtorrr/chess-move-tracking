from chessboard import ChessBoard
import cv2
import joblib
import os
import numpy as np
from chesspeices import ChessPieces
from board_perception import BoardPerception
from ultralytics import YOLO
import chess
import pandas as pd


def pad_image_to_target_resolution(image, target_width, target_height, border_color=(0, 0, 0)):
    """
    Pads an image to a target resolution, centering the original image.

    Args:
        image (numpy.ndarray): The input image (BGR format).
        target_width (int): The desired width of the padded image.
        target_height (int): The desired height of the padded image.
        border_color (tuple): The color of the padding (BGR format). Default is black.

    Returns:
        numpy.ndarray: The padded image.
    """
    h, w = image.shape[:2]

    # Calculate padding amounts
    diff_vert = target_height - h
    pad_top = diff_vert // 2
    pad_bottom = diff_vert - pad_top

    diff_hori = target_width - w
    pad_left = diff_hori // 2
    pad_right = diff_hori - pad_left

    # Apply padding
    padded_image = cv2.copyMakeBorder(
        image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=border_color
    )
    return padded_image

if __name__ == "__main__":
    video_paths = [
        # "data/2_move_student.mp4",
        # "data/2_Move_rotate_student.mp4",
        # "data/4_Move_studet.mp4",
        # "data/6_Move_student.mp4",
        # "data/8_Move_student.mp4",
        "data/videos/Bonus Long Video Label.mp4",
    ]
    output = []
    for video_path in video_paths:
        filename = os.path.splitext(os.path.basename(video_path))[0]
        cache_board_path = f"stubs/{filename}-board.joblib"
        cache_history_path = f"stubs/{filename}-history.joblib"

        board = None
        # Step 1: Detect board orientation, files and ranks
        if os.path.exists(cache_board_path):
            print(f"Loading cached board from {cache_board_path}")
            board = joblib.load(cache_board_path)
        else:
            print("Cached board not found, detecting from video...")
            cap = cv2.VideoCapture(video_path)
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                frame = pad_image_to_target_resolution(frame, 1080, 1920)
                board = ChessBoard()
                if board.find_board2(frame):
                    print(f"Board found with orientation: {board.orientation}")
                    os.makedirs(
                        os.path.dirname(cache_board_path), exist_ok=True
                    )
                    # joblib.dump(board, cache_board_path)
                    print(f"Board cached to {cache_board_path}")
                    break
                else:
                    print("Could not find chessboard in this frame.")
            cap.release()

        # Detection chess pieces
        chess_pieces_detector = ChessPieces()
        board_perceptor = BoardPerception(buffer_size=10)
        person_detector = YOLO("yolo11l.pt")
        history = []
        if board and board.H is not None:
            # Now you can use the board object on new frames
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            id_to_name = np.vectorize(
                lambda cid: chess_pieces_detector.model.names[int(cid)]
                if cid != -1
                else None
            )
            if os.path.exists(cache_history_path):
                # if False:
                print(f"Load cached history from {cache_history_path}")
                history = joblib.load(cache_history_path)
            else:
                i = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if board.rotation is not None:
                        frame = cv2.rotate(frame, board.rotation)
                    frame = pad_image_to_target_resolution(frame, 1080, 1920)
                    i += 1

                    if i % 3 != 0:
                        continue
                    

                    # look for person
                    found_person = False
                    results = person_detector.predict(
                        frame, save=False, verbose=False, device="mps")
                    for box in results[0].boxes:
                        person = person_detector.names[int(box.cls)]
                        if person == "person":
                            found_person = True
                            break

                    draw_frame = frame.copy()
                    if found_person and board is None:
                        cv2.imshow("frame", draw_frame)
                    elif found_person and board is not None:
                        board.draw_projected_centers(
                            draw_frame, 8, 8, color=(0, 255, 0)
                        )
                        board.draw_projected_grid(
                            draw_frame, 0, 8, 0, 8, color=(0, 255, 255)
                        )

                        cv2.imshow("frame", draw_frame)
                    elif not found_person:
                        board_idx, draw_frame = (
                            chess_pieces_detector.detect_frame(
                                frame, return_plot=True, board=board
                            )
                        )
                        fen_curr = chess_pieces_detector.board_to_fen(
                            board_idx, board
                        )
                        board_curr = chess.Board(fen_curr)
                        if (
                            board_curr.is_valid()
                            or board_curr.status()
                            == chess.STATUS_OPPOSITE_CHECK
                        ):
                            board_perceptor.add_frame_observation(board_idx)
                        else:
                            print(board_curr.status())

                        stable_grid, is_stable = (
                            board_perceptor.get_stable_board()
                        )

                        if stable_grid is not None:
                            # print("--- Stable Board State ---")

                            if is_stable:
                                # print("Board is stable.")
                                fen_string = chess_pieces_detector.board_to_fen(
                                    stable_grid, board
                                )
                                history.append(fen_string)
                                chess_board = chess.Board(fen_string)
                                print(f"frame: {i}/{total_frames} (stable)")

                                print(
                                    chess_board.transform(
                                        chess.flip_horizontal
                                    ).transform(chess.flip_vertical)
                                )
                                # print(f"FEN: {fen_string}")
                                # print(chess_board)
                            else:
                                print(f"frame: {i}/{total_frames}")
                            # print("--------------------------")

                        board.draw_projected_centers(
                            draw_frame, 8, 8, color=(0, 255, 0)
                        )
                        board.draw_projected_grid(
                            draw_frame, 0, 8, 0, 8, color=(0, 255, 255)
                        )

                        cv2.imshow("frame", draw_frame)
                    if cv2.waitKey(1) == ord("q"):
                        break
                cap.release()
                joblib.dump(history, cache_history_path)
                print(f"Save history to cached to {cache_history_path}")
        else:
            print("Failed to load or find a board.")

        cv2.destroyAllWindows()

        moves = []
        detected_turns = []
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
                    # if i == 135:
                    #     import pdb;pdb.set_trace()
                    diffs.append(square)

                    if piece_prev is not None:
                        if piece_curr is None:  # Move to empty square
                            detected_turn = piece_prev.color
                        elif piece_curr.color != piece_prev.color:
                            detected_turn = piece_curr.color

            if detected_turn is not None:
                detected_turns.append(detected_turn)
                # if i == 2840:
                #     detected_turn = True
                board_prev.turn = detected_turn
                found_move = None
                for move in board_prev.legal_moves:
                    # board_prev.push(move)
                    # todo: more strict
                    # if board_prev.fen().split(" ")[0] == fen_curr:
                    if board_prev.piece_at(
                        move.from_square
                    ) == board_curr.piece_at(move.to_square):
                        found_move = move
                        break
                    # board_prev.pop()

                # ILLEGAL MOVE
                if len(diffs) == 2 and found_move is None:
                    sq_1, sq_2 = diffs[0], diffs[1]
                    candidates = [(sq_1, sq_2), (sq_2, sq_1)]
                    for from_sq, to_sq in candidates:
                        piece = board_prev.piece_at(from_sq)
                        if piece is None:
                            continue
                        move = chess.Move(from_sq, to_sq)
                        if board_prev.piece_at(
                            move.from_square
                        ) == board_curr.piece_at(move.to_square):
                            # if board_prev.piece_at(move.from_square):
                            found_move = move
                move_data = {}

                if found_move:
                    moving_piece = board_prev.piece_at(found_move.from_square)
                    is_capture = board_prev.is_capture(found_move)

                    captured_piece = None
                    if is_capture:
                        if board_prev.is_en_passant(found_move):
                            captured_piece = chess.Piece(
                                chess.PAWN, not detected_turn
                            )
                        else:
                            captured_piece = board_prev.piece_at(
                                found_move.to_square
                            )

                    move_data = {
                        "turn": chess.WHITE
                        if detected_turn == chess.WHITE
                        else chess.BLACK,
                        "uci": found_move.uci(),
                        "piece": moving_piece.symbol(),
                        "from_sq": chess.square_name(found_move.from_square),
                        "to_sq": chess.square_name(found_move.to_square),
                        "is_capture": is_capture,
                        "captured_piece": captured_piece.symbol()
                        if is_capture
                        else None,
                        "is_castling": board_prev.is_castling(found_move),
                        "is_en_passant": board_prev.is_en_passant(found_move),
                        "is_promotion": found_move.promotion is not None,
                    }
                    moves.append((found_move, detected_turn))
                print(move_data)
                if not move_data:
                    import pdb;pdb.set_trace()
        for i in range(len(moves) - 1):
            move_curr, turn_curr = moves[i]
            if move_curr is not None and turn_curr is not None:
                move_next, turn_next = moves[i + 1]
                if turn_curr == turn_next:
                    if (move_curr.from_square == move_next.to_square) and (
                        move_curr.to_square == move_next.from_square
                    ):
                        moves[i] = (None, None)
                        moves[i + 1] = (None, None)
        legal_moves = [m for m in moves if m[0] is not None]
        
        ini_move, ini_turn = legal_moves[0]
        board_ini = chess.Board(history[0])
        board_ini.turn = ini_turn

        if not ini_turn:
            legal_moves.pop(0)
            san = board_ini.san(ini_move)
            board_ini.push(ini_move)
            pgn_string = f"1... {san} "
            n_move = 1
        else:
            pgn_string = ""
            n_move = 0
        for i in range(0, len(legal_moves), 2):
            try:
                white_move, _ = legal_moves[i]
                san_white = board_ini.san(white_move)
                board_ini.push(white_move)

                black_move, _ = legal_moves[i + 1]
                san_black = board_ini.san(black_move)
                board_ini.push(black_move)
            except chess.IllegalMoveError:
                raise ValueError("IllegalMove was found")
            except IndexError:
                san_black = ""
            n_move += 1
            
            pgn_string += f"{n_move}. {san_white} {san_black} "
        output.append(pgn_string.strip())
    filenames = [ os.path.basename(video_path) for video_path in video_paths]
    df = pd.DataFrame({"row_id": filenames, "output": output})
    df.to_csv("result.csv", index=False)
        # for move, turn in legal_moves:
        #     board_ini.turn = turn
        #     san = board_ini.san(move)
        #     board_ini.push(move)
        #     print(san)

