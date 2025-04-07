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
    Creates overall career statistics for each player.
    
    Args:
        df: DataFrame containing match data
        
    Returns:
        DataFrame with aggregate statistics per player
    """
    # Create player name lookup first
    name_lookup = pd.concat([
        df[['winner_id', 'winner_name']].rename(columns={'winner_id': 'player_id', 'winner_name': 'name'}),
        df[['loser_id', 'loser_name']].rename(columns={'loser_id': 'player_id', 'loser_name': 'name'})
    ]).drop_duplicates('player_id').set_index('player_id')['name']
    
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

    # Add player names using the lookup
    all_stats = all_stats.reset_index().rename(columns={'index': 'player_id'})
    all_stats['player_name'] = all_stats['player_id'].map(name_lookup)

    return all_stats

def build_all_features(df, force_recompute=False):
    """
    Main function to build all feature sets.
    
    Args:
        df: Clean DataFrame with match data
        force_recompute: If True, recomputes all features even if cached
        
    Returns:
        Tuple of (h2h_features, player_overall_stats)
    """
    stores = get_feature_stores()
    
    # Handle head-to-head features
    print("Building head-to-head features...")
    h2h_features = build_head_to_head_features(df)
    with pd.HDFStore(stores['h2h']) as store:
        store.put('/features', h2h_features)
    
    # Handle player career statistics
    print("Building player career statistics...")
    player_stats = build_player_overall_stats(df)
    with pd.HDFStore(stores['player_stats']) as store:
        store.put('/features', player_stats)
    
    return h2h_features, player_stats
