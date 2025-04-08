import pandas as pd
import numpy as np
from pathlib import Path

def get_feature_stores():
    """Returns paths to feature stores"""
    feature_dir = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'features'
    feature_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        'h2h': feature_dir / 'head_to_head.h5',
        'player_stats': feature_dir / 'player_stats.h5'
    }

def get_features_store():
    """Returns path to features HDF5 store"""
    feature_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
    feature_dir.mkdir(parents=True, exist_ok=True)
    return feature_dir / 'features.h5'

def get_feature_dir():
    """Returns path to features directory, creating it if needed"""
    feature_dir = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'features'
    feature_dir.mkdir(parents=True, exist_ok=True)
    return feature_dir

def build_head_to_head_features(df):
    """
    Creates head-to-head statistics for all player pairs.
    
    Args:
        df: DataFrame containing match data with winner_id, loser_id columns
        
    Returns:
        DataFrame with one row per unique player pair and their matchup statistics
    """
    # Create player name lookup first
    name_lookup = pd.concat([
        df[['winner_id', 'winner_name']].rename(columns={'winner_id': 'player_id', 'winner_name': 'name'}),
        df[['loser_id', 'loser_name']].rename(columns={'loser_id': 'player_id', 'loser_name': 'name'})
    ]).drop_duplicates('player_id').set_index('player_id')['name']
    
    # Create lists of all matchups
    matchups = []
    for _, row in df.iterrows():
        # Always store player IDs in sorted order for consistency
        p1, p2 = sorted([row['winner_id'], row['loser_id']])
        winner = row['winner_id']
        matchups.append({
            'player1_id': p1,
            'player2_id': p2,
            'winner_id': winner
        })
    
    h2h_df = pd.DataFrame(matchups)
    
    # Group by player pairs and calculate statistics
    h2h_stats = h2h_df.groupby(['player1_id', 'player2_id']).agg({
        'winner_id': [
            ('total_matches', 'count'),
            ('p1_wins', lambda x: (x == x.iloc[0]).sum()),
            ('p2_wins', lambda x: (x == x.iloc[-1]).sum())
        ]
    }).reset_index()
    
    # Flatten column names
    h2h_stats.columns = ['player1_id', 'player2_id', 'total_matches', 'p1_wins', 'p2_wins']
    
    # Calculate win percentages
    h2h_stats['p1_win_pct'] = h2h_stats['p1_wins'] / h2h_stats['total_matches']
    h2h_stats['p2_win_pct'] = h2h_stats['p2_wins'] / h2h_stats['total_matches']
    
    # Add player names using the lookup
    h2h_stats['player1_name'] = h2h_stats['player1_id'].map(name_lookup)
    h2h_stats['player2_name'] = h2h_stats['player2_id'].map(name_lookup)
    
    return h2h_stats

def build_player_overall_stats(df):
    """
    Creates career statistics for each player.
    
    Args:
        df: DataFrame containing match data
        
    Returns:
        DataFrame with one row per player containing career stats
    """
    # Create separate dataframes for wins and losses to calculate stats
    wins = df[['winner_id', 'winner_name', 'surface', 'tourney_level',
               'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
               'w_SvGms', 'w_bpSaved', 'w_bpFaced']].copy()
    losses = df[['loser_id', 'loser_name', 'surface', 'tourney_level',
                 'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon',
                 'l_SvGms', 'l_bpSaved', 'l_bpFaced']].copy()
    
    # Rename columns to common names
    wins.columns = ['player_id', 'name', 'surface', 'level', 
                   'aces', 'dfs', 'serve_pts', 'first_in', 'first_won',
                   'second_won', 'serve_games', 'bp_saved', 'bp_faced']
    losses.columns = wins.columns
    
    # Add win/loss column
    wins['won'] = 1
    losses['won'] = 0
    
    # Combine wins and losses
    all_matches = pd.concat([wins, losses])
    
    # Group by player
    stats = []
    for player_id, player_matches in all_matches.groupby('player_id'):
        matches_count = len(player_matches)
        if matches_count < 1:
            continue
            
        # Get player name from first match
        player_name = player_matches['name'].iloc[0]
            
        # Basic info
        player_stats = {
            'player_id': player_id,
            'player_name': player_name,  # Added player name
            'matches_played': matches_count,
            
            # Overall win rate
            'win_rate': player_matches['won'].mean(),
            
            # Surface win rates
            'hard_win_rate': player_matches[player_matches['surface'] == 'Hard']['won'].mean(),
            'clay_win_rate': player_matches[player_matches['surface'] == 'Clay']['won'].mean(),
            'grass_win_rate': player_matches[player_matches['surface'] == 'Grass']['won'].mean(),
            
            # Tournament level win rates
            'grand_slam_win_rate': player_matches[player_matches['level'] == 'G']['won'].mean(),
            'masters_win_rate': player_matches[player_matches['level'] == 'M']['won'].mean(),
            
            # Serve stats
            'first_serve_pct': player_matches['first_in'].sum() / player_matches['serve_pts'].sum(),
            'first_serve_won_pct': player_matches['first_won'].sum() / player_matches['first_in'].sum(),
            'second_serve_won_pct': (player_matches['second_won'].sum() / 
                                   (player_matches['serve_pts'].sum() - player_matches['first_in'].sum())),
            
            # Ace and double fault rates (per service game)
            'ace_rate': player_matches['aces'].sum() / player_matches['serve_games'].sum(),
            'double_fault_rate': player_matches['dfs'].sum() / player_matches['serve_games'].sum(),
            
            # Return stats
            'break_points_per_game': player_matches['bp_faced'].sum() / player_matches['serve_games'].sum(),
            'break_point_save_pct': player_matches['bp_saved'].sum() / player_matches['bp_faced'].sum()
        }
        
        # Handle division by zero cases
        player_stats = {k: (v if not pd.isna(v) else 0) for k, v in player_stats.items()}
        stats.append(player_stats)
    
    return pd.DataFrame(stats)

def build_all_features(df, force_recompute=False):
    """Builds and saves all feature sets"""
    stores = get_feature_stores()
    
    # Initialize variables
    h2h = None
    stats = None
    
    # Build head to head features
    h2h_path = stores['h2h']
    try:
        if force_recompute or not h2h_path.exists():
            print("Computing head-to-head features...")
            h2h = build_head_to_head_features(df)
            if h2h is not None and not h2h.empty:
                with pd.HDFStore(h2h_path, mode='w') as store:
                    store.put('h2h', h2h, format='fixed')  # Changed to fixed format
                print("Head-to-head features saved successfully")
        else:
            try:
                with pd.HDFStore(h2h_path, mode='r') as store:
                    if '/h2h' in store:
                        h2h = store.get('h2h')
                    else:
                        print("H2H data not found in store, recomputing...")
                        h2h = build_head_to_head_features(df)
                        with pd.HDFStore(h2h_path, mode='w') as store:
                            store.put('h2h', h2h, format='fixed')
            except Exception as e:
                print(f"Error reading H2H store: {str(e)}, recomputing...")
                h2h = build_head_to_head_features(df)
                with pd.HDFStore(h2h_path, mode='w') as store:
                    store.put('h2h', h2h, format='fixed')
    except Exception as e:
        print(f"Error in H2H feature computation: {str(e)}")
        raise

    # Build player stats features
    stats_path = stores['player_stats']
    try:
        if force_recompute or not stats_path.exists():
            print("Computing player stats features...")
            stats = build_player_overall_stats(df)
            if stats is not None and not stats.empty:
                with pd.HDFStore(stats_path, mode='w') as store:
                    store.put('player_stats', stats, format='fixed')  # Changed to fixed format
                print("Player stats features saved successfully")
        else:
            try:
                with pd.HDFStore(stats_path, mode='r') as store:
                    if '/player_stats' in store:
                        stats = store.get('player_stats')
                    else:
                        print("Player stats not found in store, recomputing...")
                        stats = build_player_overall_stats(df)
                        with pd.HDFStore(stats_path, mode='w') as store:
                            store.put('player_stats', stats, format='fixed')
            except Exception as e:
                print(f"Error reading player stats store: {str(e)}, recomputing...")
                stats = build_player_overall_stats(df)
                with pd.HDFStore(stats_path, mode='w') as store:
                    store.put('player_stats', stats, format='fixed')
    except Exception as e:
        print(f"Error in player stats computation: {str(e)}")
        raise

    return h2h, stats
