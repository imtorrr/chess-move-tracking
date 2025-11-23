from chessboard import ChessBoard
import cv2
import joblib
import os
import numpy as np
from collections import deque, Counter
from chesspeices import ChessPieces
from board_perception import BoardPerception
from ultralytics import YOLO
import chess

def detect_move_with_engine(fen_prev, fen_curr, turn):
    # 1. Create a board from the PREVIOUS valid state
    board = chess.Board(fen_prev)
    board.turn = turn
    
    # 2. Extract just the "Piece Placement" part of the new FEN
    #    (YOLO often gets move counters/flags wrong, so we only trust the grid)
    target_placement = fen_curr

    # 3. Iterate through every single LEGAL move available
    for move in board.legal_moves:
        board.push(move) # Make the move virtually
        
        # 4. Check if the resulting board matches your YOLO detection
        candidate_fen = board.fen()
        candidate_placement = candidate_fen.split(" ")[0]
        
        if candidate_placement == target_placement:
            return move # Found it! (e.g., "e2e4" or "e1g1")
            
        board.pop() # Undo and try the next one
        
    return None # No legal move matches what YOLO sees (Detection Error)

if __name__ == "__main__":
    video_path = "data/8_Move_student.mp4"
    # video_path = "data/4_Move_studet.mp4"
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
            
            board = ChessBoard()
            if board.find_board2(frame):
                print(f"Board found with orientation: {board.orientation}")
                os.makedirs(os.path.dirname(cache_board_path), exist_ok=True)
                joblib.dump(board, cache_board_path)
                print(f"Board cached to {cache_board_path}")
                break
            else:
                print("Could not find chessboard in this frame.")
        cap.release()
        
    # Detection chess pieces
    chess_pieces_detector = ChessPieces()
    board_perceptor = BoardPerception()
    person_detector = YOLO("yolo11n.pt")
    history = []
    if board and board.H is not None:
        # Now you can use the board object on new frames
        cap = cv2.VideoCapture(video_path)
        
        id_to_name = np.vectorize(lambda cid: chess_pieces_detector.model.names[int(cid)] if cid != -1 else None)
        if os.path.exists(cache_history_path):
            print(f"Load cached history from {cache_history_path}")
            history = joblib.load(cache_history_path)
        else:
            i = 0
            
            while True:
                ret, frame = cap.read()
                if board.rotation is not None:
                    frame = cv2.rotate(frame, board.rotation)
                i += 1
                
                if i % 3 != 0:
                    continue
                if not ret:
                    break
                
                # look for person
                found_person = False
                results = person_detector.predict(frame, save=False, verbose=False)
                for box in results[0].boxes:
                    person = person_detector.names[int(box.cls)]
                    if person == "person":
                        found_person = True
                        break
                
                draw_frame = frame.copy()  
                if found_person and board is None:
                    cv2.imshow('frame', draw_frame)
                elif found_person and board is not None:
                    board.draw_projected_centers(draw_frame, 8, 8, color=(0, 255, 0))
                    board.draw_projected_grid(draw_frame, 0, 8, 0, 8, color=(0, 255, 255))
                    
                    cv2.imshow('frame', draw_frame)
                elif not found_person:
                    board_idx, draw_frame = chess_pieces_detector.detect_frame(frame, return_plot=True, board = board)
                    board_perceptor.add_frame_observation(board_idx)
                    
                    stable_grid, is_stable = board_perceptor.get_stable_board()
                    
                    if stable_grid is not None:
                        # print("--- Stable Board State ---")
                        
                        if is_stable:
                            # print("Board is stable.")
                            fen_string = chess_pieces_detector.board_to_fen(stable_grid, board)
                            history.append(fen_string)
                            chess_board = chess.Board(fen_string)
                            print(f"frame: {i} (stable)", end="\r")
                            # print(f"FEN: {fen_string}")
                            # print(chess_board)
                        else:
                            print(f"frame: {i}", end="\r")
                        # print("--------------------------")

                    board.draw_projected_centers(draw_frame, 8, 8, color=(0, 255, 0))
                    board.draw_projected_grid(draw_frame, 0, 8, 0, 8, color=(0, 255, 255))
                    
                    cv2.imshow('frame', draw_frame)
                if cv2.waitKey(1) == ord('q'):
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
                    if piece_curr is None: # Move to empty square
                        detected_turn = piece_prev.color
                    elif piece_curr.color != piece_prev.color:
                        detected_turn = piece_curr.color
                       
        if detected_turn is not None:
            detected_turns.append(detected_turn)
            board_prev.turn = detected_turn
            found_move = None
            for move in board_prev.legal_moves:
                # board_prev.push(move)
                # if board_prev.fen().split(" ")[0] == fen_curr:
                if board_prev.piece_at(move.from_square) == board_curr.piece_at(move.to_square):
                    found_move = move
                    print(board_prev.san(move))
                    break
                # board_prev.pop()
            print(diffs, i)
            if i == 6566:
                import pdb;pdb.set_trace()
            # ILLEGAL MOVE
            if len(diffs) == 2:
                sq_1, sq_2 = diffs[0], diffs[1]
                candidates = [(sq_1, sq_2), (sq_2, sq_1)]
                for from_sq, to_sq in candidates:
                    piece = board_prev.piece_at(from_sq)
                    if piece is  None:
                        continue
                    move = chess.Move(from_sq, to_sq)
                    # if board_prev.piece_at(move.from_square) == board_curr.piece_at(move.to_square):
                    if board_prev.piece_at(move.from_square):
                        found_move = move
                        print(board_prev.san(move))
            move_data = {}
            
            if found_move:
                moving_piece = board_prev.piece_at(found_move.from_square)
                is_capture = board_prev.is_capture(found_move)

                captured_piece = None
                if is_capture:
                    if board_prev.is_en_passant(found_move):
                        captured_piece = chess.Piece(chess.PAWN, not detected_turn)
                    else:
                        captured_piece = board_prev.piece_at(found_move.to_square) 
                
                move_data = {
                    "turn": chess.WHITE if detected_turn == chess.WHITE else chess.BLACK,
                    "uci": found_move.uci(),
                    "piece": moving_piece.symbol(),
                    "from_sq": chess.square_name(found_move.from_square),
                    "to_sq": chess.square_name(found_move.to_square),
                    "is_capture": is_capture,
                    "captured_piece": captured_piece.symbol() if is_capture else None,
                    "is_castling": board_prev.is_castling(found_move),
                    "is_en_passant": board_prev.is_en_passant(found_move),
                    "is_promotion": found_move.promotion is not None,  
                } 
                moves.append(found_move)
            print(move_data)
    board_ini = chess.Board(history[0])
    board_ini.turn = detected_turns[0]
    
    print(board_ini.variation_san(moves))
    
            

