import cv2
import numpy as np
import torch
from ultralytics import YOLO

from chessboard import ChessBoard

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

COLOR_MAPPING = {
    "black-bishop": (0, 100, 150),
    "black-king": (100, 0, 150),
    "black-knight": (50, 50, 150),
    "black-pawn": (0, 0, 100),
    "black-queen": (150, 0, 100),
    "black-rook": (0, 50, 100),
    "white-bishop": (255, 180, 0),
    "white-king": (255, 255, 0),
    "white-knight": (200, 255, 50),
    "white-pawn": (255, 220, 100),
    "white-queen": (255, 150, 150),
    "white-rook": (150, 255, 150)
}

PIECE_ID_TO_FEN_CHAR = {
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

class ChessPieces:
    def __init__(self, model_path: str = "weights/chess-piece-yolo11l-tuned.pt"):
        """
        Initializes the ChessPieces object with a YOLO model for piece detection.

        Args:
            model_path: The path to the YOLO model file.
        """
        self.model = YOLO(model_path)

    def detect_frame(
        self,
        frame: np.ndarray,
        return_plot: bool = False,
        board: ChessBoard | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Detects chess pieces in a given frame and maps them to board positions.

        Args:
            frame: The input image frame.
            return_plot: Whether to return a frame with detections drawn on it.
            board: A ChessBoard object for projecting piece coordinates.

        Returns:
            An 8x8 NumPy array representing the board, or a tuple with the board and the plotted frame.
        """
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
            board.draw_projected_centers(draw_frame, 8, 8, color=(0, 255, 0))
            board.draw_projected_grid(draw_frame, 0, 8, 0, 8, color=(0, 0, 0))
            for r in range(8):
                for c in range(8):
                    if board_idx[r, c] != -1:
                        label = self.model.names[board_idx[r, c]]
                        uv = board.project_rc_center(r, c)
                        if uv is not None:
                            u, v = map(int, uv)

                            buffer_color = (255, 255, 255) if "black" in label else (0, 0, 0)

                            cv2.putText(
                                draw_frame,
                                PIECE_ID_TO_FEN_CHAR[label],
                                (u-20, v),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                buffer_color,
                                4,
                            )   
                            
                            cv2.putText(
                                draw_frame,
                                PIECE_ID_TO_FEN_CHAR[label],
                                (u-20, v),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                COLOR_MAPPING[label],
                                2,
                            )

        return board_idx, draw_frame

    def board_to_fen(self, board_idx: np.ndarray, chessboard_obj: ChessBoard) -> str:
        """
        Converts the detected board configuration into a FEN (Forsyth-Edwards Notation) string.

        Args:
            board_idx: An 8x8 NumPy array with class indices of the detected pieces.
            chessboard_obj: The ChessBoard object used for coordinate conversions.

        Returns:
            The piece placement part of a FEN string.
        """
        # Define the mapping from model names to FEN characters
        
        # Initialize a 2D array to hold FEN characters, ordered by FEN rank (8-1) and FEN file (a-h)
        board_chars = [["" for _ in range(8)] for _ in range(8)]
        for r_idx in range(8):  # Iterate through the board_idx array's rows (0-7)
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
                    fen_char = PIECE_ID_TO_FEN_CHAR.get(
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
