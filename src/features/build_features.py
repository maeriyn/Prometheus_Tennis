import pandas as pd
import numpy as np
from pathlib import Path

def get_feature_stores():
    """Returns paths to feature stores"""
    feature_dir = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'features'
    feature_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        'h2h': feature_dir / 'head_to_head.h5',
        'player_stats': feature_dir / 'player_stats.h5',
        'time_stats': feature_dir / 'time_dependent.h5'
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
    
    return h2h_stats

def build_player_overall_stats(df):
    """
    Creates overall career statistics for each player.
    
    Args:
        df: DataFrame containing match data
        
    Returns:
        DataFrame with aggregate statistics per player
    """
    # Initialize stats for winners and losers
    winner_stats = df.groupby('winner_id').agg({
        'tourney_id': 'count',  # total matches won
        'winner_rank': 'mean',  # average rank when winning
        'w_ace': 'mean',       # average aces per match
        'w_df': 'mean',        # average double faults
        'w_svpt': 'mean',      # average service points
        'w_1stIn': 'mean',     # average first serves in
        'w_1stWon': 'mean',    # average first serve points won
        'w_2ndWon': 'mean',    # average second serve points won
        'w_bpSaved': 'mean',   # average break points saved
    }).rename(columns={
        'tourney_id': 'matches_won',
        'winner_rank': 'avg_rank_when_winning'
    })

    loser_stats = df.groupby('loser_id').agg({
        'tourney_id': 'count',  # total matches lost
        'loser_rank': 'mean',   # average rank when losing
        'l_ace': 'mean',
        'l_df': 'mean',
        'l_svpt': 'mean',
        'l_1stIn': 'mean',
        'l_1stWon': 'mean',
        'l_2ndWon': 'mean',
        'l_bpSaved': 'mean'
    }).rename(columns={
        'tourney_id': 'matches_lost',
        'loser_rank': 'avg_rank_when_losing'
    })

    # Combine winner and loser stats
    all_stats = pd.merge(
        winner_stats,
        loser_stats,
        left_index=True,
        right_index=True,
        suffixes=('_winning', '_losing'),
        how='outer'
    ).fillna(0)

    # Calculate career metrics
    all_stats['total_matches'] = all_stats['matches_won'] + all_stats['matches_lost']
    all_stats['win_rate'] = all_stats['matches_won'] / all_stats['total_matches']
    all_stats['avg_rank'] = (all_stats['avg_rank_when_winning'] * all_stats['matches_won'] + 
                            all_stats['avg_rank_when_losing'] * all_stats['matches_lost']) / all_stats['total_matches']

    return all_stats

def build_time_dependent_stats(df, window_sizes=[10, 30, 90]):
    """Creates rolling statistics for each player over different time windows."""
    df['tourney_date'] = pd.to_datetime(df['tourney_date'])
    
    # Prepare the base stats DataFrame
    stats_dfs = []
    for role in ['winner', 'loser']:
        is_winner = (role == 'winner')
        stats_df = pd.DataFrame({
            'player_id': df[f'{role}_id'],
            'tourney_date': df['tourney_date'],
            'won_match': 1 if is_winner else 0,
            'aces': df[f'{role[0]}_ace'],
            'double_faults': df[f'{role[0]}_df'],
            'first_serve_in': df[f'{role[0]}_1stIn'],
            'first_serve_won': df[f'{role[0]}_1stWon'],
            'second_serve_won': df[f'{role[0]}_2ndWon'],
            'break_points_saved': df[f'{role[0]}_bpSaved']
        })
        stats_dfs.append(stats_df)
    
    # Combine and sort chronologically
    all_stats = pd.concat(stats_dfs, ignore_index=True)
    all_stats.sort_values(['player_id', 'tourney_date'], inplace=True)
    
    # Initialize result DataFrame
    result = all_stats[['player_id', 'tourney_date']].copy()
    
    # Calculate rolling stats for each window size
    for window in window_sizes:
        grouped = all_stats.groupby('player_id')
        roll = grouped.rolling(window=f'{window}D', on='tourney_date', min_periods=1)
        
        # Add stats for this window
        stats_dict = {
            f'matches_played_{window}d': roll['won_match'].count(),
            f'win_rate_{window}d': roll['won_match'].mean(),
            f'avg_aces_{window}d': roll['aces'].mean(),
            f'avg_dfs_{window}d': roll['double_faults'].mean(),
            f'first_serve_pct_{window}d': roll['first_serve_in'].mean(),
            f'first_serve_won_pct_{window}d': roll['first_serve_won'].mean(),
            f'second_serve_won_pct_{window}d': roll['second_serve_won'].mean(),
            f'break_points_saved_pct_{window}d': roll['break_points_saved'].mean()
        }
        
        for col, values in stats_dict.items():
            result[col] = values.reset_index(level=0, drop=True)
    
    return result

def build_all_features(df, force_recompute=False):
    """
    Main function to build all feature sets.
    
    Args:
        df: Clean DataFrame with match data
        force_recompute: If True, recomputes all features even if cached
        
    Returns:
        Tuple of (h2h_features, player_overall_stats, time_dependent_stats)
    """
    stores = get_feature_stores()
    
    # Handle head-to-head features
    with pd.HDFStore(stores['h2h']) as store:
        if not force_recompute and '/features' in store:
            print("Loading cached head-to-head features...")
            h2h_features = store.get('/features')
        else:
            print("Building head-to-head features...")
            h2h_features = build_head_to_head_features(df)
            store.put('/features', h2h_features)
    
    # Handle player career statistics
    with pd.HDFStore(stores['player_stats']) as store:
        if not force_recompute and '/features' in store:
            print("Loading cached player career statistics...")
            player_stats = store.get('/features')
        else:
            print("Building player career statistics...")
            player_stats = build_player_overall_stats(df)
            player_stats = player_stats.reset_index().rename(columns={'index': 'player_id'})
            store.put('/features', player_stats)
    
    # Handle time-dependent stats (always recomputed)
    print("Building time-dependent performance statistics...")
    time_stats = build_time_dependent_stats(df)
    with pd.HDFStore(stores['time_stats']) as store:
        store.put('/features', time_stats)
    
    return h2h_features, player_stats, time_stats
