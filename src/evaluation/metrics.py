"""Evaluation metrics for Truco agents.

Implements Elo rating computation, win rate statistics, and
other evaluation metrics for comparing trained agents.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import math


@dataclass
class EloRating:
    """Elo rating tracker for an agent."""
    name: str
    rating: float = 1500.0
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    
    @property
    def win_rate(self) -> float:
        """Compute win rate."""
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played
    
    @property
    def record(self) -> str:
        """Return W-L-D record string."""
        return f"{self.wins}-{self.losses}-{self.draws}"


def expected_score(rating_a: float, rating_b: float) -> float:
    """Compute expected score for player A against player B.
    
    Args:
        rating_a: Elo rating of player A.
        rating_b: Elo rating of player B.
        
    Returns:
        Expected score (probability of winning + 0.5 * probability of draw).
    """
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update_elo(
    winner_rating: float,
    loser_rating: float,
    k_factor: float = 32.0,
    is_draw: bool = False,
) -> Tuple[float, float]:
    """Update Elo ratings after a match.
    
    Args:
        winner_rating: Current rating of winner (or player A if draw).
        loser_rating: Current rating of loser (or player B if draw).
        k_factor: K-factor for rating update magnitude.
        is_draw: Whether the match was a draw.
        
    Returns:
        Tuple of (new_winner_rating, new_loser_rating).
    """
    expected_winner = expected_score(winner_rating, loser_rating)
    expected_loser = 1.0 - expected_winner
    
    if is_draw:
        actual_winner = 0.5
        actual_loser = 0.5
    else:
        actual_winner = 1.0
        actual_loser = 0.0
    
    new_winner = winner_rating + k_factor * (actual_winner - expected_winner)
    new_loser = loser_rating + k_factor * (actual_loser - expected_loser)
    
    return new_winner, new_loser


def compute_elo(
    match_results: List[Tuple[str, str, int]],
    initial_rating: float = 1500.0,
    k_factor: float = 32.0,
) -> Dict[str, EloRating]:
    """Compute Elo ratings from a list of match results.
    
    Args:
        match_results: List of (player_a, player_b, result) tuples.
            result: 1 if player_a won, -1 if player_b won, 0 if draw.
        initial_rating: Starting Elo rating for new players.
        k_factor: K-factor for rating updates.
        
    Returns:
        Dictionary mapping player names to EloRating objects.
    """
    ratings: Dict[str, EloRating] = {}
    
    for player_a, player_b, result in match_results:
        # Initialize ratings if needed
        if player_a not in ratings:
            ratings[player_a] = EloRating(name=player_a, rating=initial_rating)
        if player_b not in ratings:
            ratings[player_b] = EloRating(name=player_b, rating=initial_rating)
        
        rating_a = ratings[player_a]
        rating_b = ratings[player_b]
        
        # Update game counts
        rating_a.games_played += 1
        rating_b.games_played += 1
        
        if result == 1:  # Player A won
            rating_a.wins += 1
            rating_b.losses += 1
            new_a, new_b = update_elo(rating_a.rating, rating_b.rating, k_factor)
        elif result == -1:  # Player B won
            rating_a.losses += 1
            rating_b.wins += 1
            new_b, new_a = update_elo(rating_b.rating, rating_a.rating, k_factor)
        else:  # Draw
            rating_a.draws += 1
            rating_b.draws += 1
            new_a, new_b = update_elo(rating_a.rating, rating_b.rating, k_factor, is_draw=True)
        
        rating_a.rating = new_a
        rating_b.rating = new_b
    
    return ratings


def compute_win_rate(
    wins: int,
    total: int,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Compute win rate with confidence interval.
    
    Uses Wilson score interval for binomial proportion.
    
    Args:
        wins: Number of wins.
        total: Total number of games.
        confidence: Confidence level (default 95%).
        
    Returns:
        Tuple of (win_rate, lower_bound, upper_bound).
    """
    if total == 0:
        return 0.0, 0.0, 0.0
    
    p = wins / total
    
    # Z-score for confidence level
    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
    
    # Wilson score interval
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    
    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    
    return p, lower, upper


@dataclass
class MatchStatistics:
    """Statistics from a head-to-head evaluation."""
    player_a: str
    player_b: str
    num_games: int
    wins_a: int
    wins_b: int
    draws: int
    total_points_a: float = 0.0
    total_points_b: float = 0.0
    avg_game_length: float = 0.0
    
    @property
    def win_rate_a(self) -> float:
        return self.wins_a / self.num_games if self.num_games > 0 else 0.0
    
    @property
    def win_rate_b(self) -> float:
        return self.wins_b / self.num_games if self.num_games > 0 else 0.0
    
    def win_rate_with_ci(self, player: str, confidence: float = 0.95) -> Tuple[float, float, float]:
        """Get win rate with confidence interval for a player."""
        if player == self.player_a:
            return compute_win_rate(self.wins_a, self.num_games, confidence)
        else:
            return compute_win_rate(self.wins_b, self.num_games, confidence)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "player_a": self.player_a,
            "player_b": self.player_b,
            "num_games": self.num_games,
            "wins_a": self.wins_a,
            "wins_b": self.wins_b,
            "draws": self.draws,
            "win_rate_a": self.win_rate_a,
            "win_rate_b": self.win_rate_b,
            "total_points_a": self.total_points_a,
            "total_points_b": self.total_points_b,
            "avg_game_length": self.avg_game_length,
        }
    
    def __str__(self) -> str:
        wr_a, lo_a, hi_a = self.win_rate_with_ci(self.player_a)
        wr_b, lo_b, hi_b = self.win_rate_with_ci(self.player_b)
        return (
            f"{self.player_a} vs {self.player_b}: "
            f"{self.wins_a}-{self.wins_b}-{self.draws} "
            f"(WR: {wr_a:.1%} [{lo_a:.1%}-{hi_a:.1%}])"
        )


@dataclass  
class TournamentResults:
    """Results from a round-robin tournament."""
    agents: List[str]
    match_stats: Dict[Tuple[str, str], MatchStatistics] = field(default_factory=dict)
    elo_ratings: Dict[str, EloRating] = field(default_factory=dict)
    
    def add_match(self, stats: MatchStatistics):
        """Add match statistics."""
        key = (stats.player_a, stats.player_b)
        self.match_stats[key] = stats
    
    def compute_rankings(self) -> List[Tuple[str, float]]:
        """Compute final rankings by Elo."""
        if not self.elo_ratings:
            return [(agent, 1500.0) for agent in self.agents]
        
        rankings = [(name, rating.rating) for name, rating in self.elo_ratings.items()]
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = ["Tournament Results", "=" * 40]
        
        rankings = self.compute_rankings()
        lines.append("\nRankings:")
        for i, (name, elo) in enumerate(rankings, 1):
            rating = self.elo_ratings.get(name)
            if rating:
                lines.append(f"  {i}. {name}: {elo:.0f} ({rating.record})")
            else:
                lines.append(f"  {i}. {name}: {elo:.0f}")
        
        lines.append("\nHead-to-Head:")
        for (a, b), stats in self.match_stats.items():
            lines.append(f"  {stats}")
        
        return "\n".join(lines)
