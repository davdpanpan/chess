import chess
import chess.pgn
import pandas as pd
import zstandard as zstd
from io import StringIO
import numpy as np

def get_material_features(board):
    """Extract material-related features"""
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                   chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    
    white_material = sum(piece_values[piece.piece_type] for piece in board.piece_map().values() if piece.color == chess.WHITE)
    black_material = sum(piece_values[piece.piece_type] for piece in board.piece_map().values() if piece.color == chess.BLACK)
    
    return {
        'white_material': white_material,
        'black_material': black_material,
        'material_diff': white_material - black_material
    }

def get_piece_count_features(board):
    """Extract piece count features"""
    features = {}
    for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        white_count = len(board.pieces(piece_type, chess.WHITE))
        black_count = len(board.pieces(piece_type, chess.BLACK))
        piece_name = chess.piece_name(piece_type)
        features[f'white_{piece_name}s'] = white_count
        features[f'black_{piece_name}s'] = black_count
    return features

def get_king_safety_features(board):
    """Extract king safety features"""
    white_king_square = board.king(chess.WHITE)
    black_king_square = board.king(chess.BLACK)
    
    features = {}
    
    if white_king_square:
        white_king_attacks = len(board.attackers(chess.BLACK, white_king_square))
        features['white_king_attackers'] = white_king_attacks
        
        # Count pawns in front of white king
        king_file = chess.square_file(white_king_square)
        king_rank = chess.square_rank(white_king_square)
        white_pawn_shield = 0
        
        # Check files: king's file and adjacent files
        for file_offset in [-1, 0, 1]:
            check_file = king_file + file_offset
            if 0 <= check_file <= 7:  # Valid file
                # Look for pawns in front of king (higher ranks)
                for rank in range(king_rank + 1, 8):
                    square = chess.square(check_file, rank)
                    piece = board.piece_at(square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == chess.WHITE:
                        white_pawn_shield += 1
                        break  # Only count first pawn in each file
        
        features['white_pawn_shield'] = white_pawn_shield
    else:
        features['white_king_attackers'] = 0
        features['white_pawn_shield'] = 0
        
    if black_king_square:
        black_king_attacks = len(board.attackers(chess.WHITE, black_king_square))
        features['black_king_attackers'] = black_king_attacks
        
        # Count pawns in front of black king
        king_file = chess.square_file(black_king_square)
        king_rank = chess.square_rank(black_king_square)
        black_pawn_shield = 0
        
        # Check files: king's file and adjacent files
        for file_offset in [-1, 0, 1]:
            check_file = king_file + file_offset
            if 0 <= check_file <= 7:  # Valid file
                # Look for pawns in front of king (lower ranks for black)
                for rank in range(king_rank - 1, -1, -1):
                    square = chess.square(check_file, rank)
                    piece = board.piece_at(square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == chess.BLACK:
                        black_pawn_shield += 1
                        break  # Only count first pawn in each file
        
        features['black_pawn_shield'] = black_pawn_shield
    else:
        features['black_king_attackers'] = 0
        features['black_pawn_shield'] = 0
    
    return features

def get_mobility_features(board):
    """Extract mobility features"""
    return {
        'to_move': 1 if board.turn == chess.WHITE else 0,
        'legal_moves': board.legal_moves.count()
    }

def get_game_state_features(board):
    """Extract game state features"""
    return {
        'is_check': int(board.is_check()),
        'is_checkmate': int(board.is_checkmate()),
        'is_stalemate': int(board.is_stalemate())
    }

def extract_board_features(board):
    """Extract features from a chess board position"""
    features = {}
    
    # Combine all feature categories
    features.update(get_material_features(board))
    features.update(get_piece_count_features(board))
    features.update(get_king_safety_features(board))
    features.update(get_mobility_features(board))
    features.update(get_game_state_features(board))
    
    return features

def process_pgn_file(filename, target_move=20, max_games=None):
    """Process PGN file and extract features from specified move"""
    games_data = []
    games_processed = 0
    
    # Handle compressed files
    if filename.endswith('.zst'):
        with open(filename, 'rb') as compressed_file:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(compressed_file) as reader:
                text_stream = StringIO(reader.read().decode('utf-8'))
    else:
        text_stream = open(filename, 'r')
    
    try:
        while True:
            game = chess.pgn.read_game(text_stream)
            if game is None:
                break
                
            # Get game result
            result = game.headers.get("Result", "*")
            if result == "*":  # Skip unfinished games
                continue
                
            # Convert result to target variable
            if result == "1-0":
                target = 1  # White wins
            elif result == "0-1":
                target = 0  # Black wins
            elif result == "1/2-1/2":
                target = 0.5  # Draw
            else:
                continue
            
            # Get player ratings
            white_elo = game.headers.get("WhiteElo", "")
            black_elo = game.headers.get("BlackElo", "")
            
            # Skip games without ratings
            if not white_elo or not black_elo or white_elo == "?" or black_elo == "?":
                continue
                
            try:
                white_elo = int(white_elo)
                black_elo = int(black_elo)
            except ValueError:
                continue
            
            # Play through the game to move 20
            board = game.board()
            move_count = 0
            
            for move in game.mainline_moves():
                if move_count >= target_move * 2:  # 20 moves = 40 half-moves
                    break
                board.push(move)
                move_count += 1
            
            # Skip if game didn't reach target move
            if move_count < target_move * 2:
                continue
            
            # Extract features from the board position
            features = extract_board_features(board)
            
            # Add metadata
            features['white_elo'] = white_elo
            features['black_elo'] = black_elo
            features['elo_diff'] = white_elo - black_elo
            features['result'] = target
            
            games_data.append(features)
            games_processed += 1
            
            # if games_processed % 1000 == 0:
            #     print(f"Processed {games_processed} games...")
            
            if max_games and games_processed >= max_games:
                break
                
    finally:
        if hasattr(text_stream, 'close'):
            text_stream.close()
    
    return games_data