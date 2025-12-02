import cv2
import numpy as np
from ultralytics import YOLO
from chessboard import ChessBoard
import torch

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class ChessPieces:
    def __init__(self, model_path="weights/chess-piece-yolo11l-tuned.pt"):
        self.model = YOLO(model_path)

    def detect_frame(
        self, frame, return_plot=False, board: ChessBoard | None = None
    ):
        if return_plot:
            draw_frame = frame.copy()

        results = self.model.track(
            frame,
            save=False,
            verbose=False,
            agnostic_nms=True,
            device=device,
            persist=True,
            iou=0.7,
        )

        board_idx = np.zeros((8, 8), dtype=int) - 1
        confidence_map = np.zeros((8, 8), dtype=float)

        for box in results[0].boxes:
            confidence = float(box.conf)
            x, y, w, h = map(int, box.xywh[0])
            xc, yc = (int(x), int(y + 0.25 * h))  # center at bottom
            cv2.circle(draw_frame, (xc, yc), 5, (255, 0, 255), -1)
            if board is not None and board.H is not None:
                r, c = board.project_xy_to_rc(xc, yc)
                if r is not None and 0 <= r < 8 and 0 <= c < 8:
                    if confidence > confidence_map[r, c]:
                        confidence_map[r, c] = confidence
                        board_idx[r, c] = int(box.cls)

        if return_plot and board is not None:
            for r in range(8):
                for c in range(8):
                    if board_idx[r, c] != -1:
                        label = self.model.names[board_idx[r, c]]
                        uv = board.project_rc_center(r, c)
                        if uv is not None:
                            u, v = map(int, uv)

                            cv2.putText(
                                draw_frame,
                                label,
                                (u + 5, v - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 255),
                                1,
                            )

        if return_plot:
            return board_idx, draw_frame
        return board_idx

    def board_to_fen(self, board_idx, chessboard_obj):
        # Define the mapping from model names to FEN characters
        piece_id_to_fen_char = {
            "black-bishop": "b",
            "black-king": "k",
            "black-knight": "n",
            "black-pawn": "p",
            "black-queen": "q",
            "black-rook": "r",
            "white-bishop": "B",
            "white-king": "K",
            "white-knight": "N",
            "white-pawn": "P",
            "white-queen": "Q",
            "white-rook": "R",
        }
        # Initialize a 2D array to hold FEN characters, ordered by FEN rank (8-1) and FEN file (a-h)
        board_chars = [["" for _ in range(8)] for _ in range(8)]
        for r_idx in range(
            8
        ):  # Iterate through the board_idx array's rows (0-7)
            for c_idx in range(
                8
            ):  # Iterate through the board_idx array's columns (0-7)
                piece_id = board_idx[r_idx, c_idx]
                pgn_col, pgn_row = chessboard_obj.get_pgn_from_rc(r_idx, c_idx)

                # FEN ranks are 8 down to 1, so pgn_row '8' is index 0, '1' is index 7
                fen_rank_idx = 8 - int(pgn_row)
                # FEN files are 'a' to 'h', so 'a' is index 0, 'h' is index 7
                fen_file_idx = ord(pgn_col) - ord("a")

                if piece_id != -1:
                    piece_name = self.model.names[piece_id]
                    fen_char = piece_id_to_fen_char.get(
                        piece_name, "U"
                    )  # 'U' for unknown
                    board_chars[fen_rank_idx][fen_file_idx] = fen_char
                else:
                    board_chars[fen_rank_idx][fen_file_idx] = (
                        "1"  # Placeholder for empty square
                    )

        # Condense empty squares and build final FEN ranks
        final_fen_ranks = []
        for rank_array in board_chars:
            condensed_rank = ""
            empty_count = 0
            for char in rank_array:
                if char == "1":
                    empty_count += 1
                else:
                    if empty_count > 0:
                        condensed_rank += str(empty_count)
                        empty_count = 0
                    condensed_rank += char
            if empty_count > 0:
                condensed_rank += str(empty_count)
            final_fen_ranks.append(condensed_rank)

        # Join ranks with '/'
        piece_placement_fen = "/".join(final_fen_ranks)

        # For now, return only the piece placement part.
        # Other FEN fields (active color, castling, en passant, halfmove, fullmove)
        # would need more logic based on game state, which is not available here.
        return piece_placement_fen
