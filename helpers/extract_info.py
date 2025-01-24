import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
from matplotlib import patches
from helpers.data_handlers import map_coordinates_to_18_zones

def extract_all_display_names(simplified_qualifiers):
    """ 
    Extracts all the display names from a list of simplified qualifiers.
    Args:
        simplified_qualifiers: List of simplified qualifiers.
    Returns:
        List of display names.
    """

    display_names = []
    for qualifiers in simplified_qualifiers:
            display_names.append(qualifiers['displayNames'])
    return display_names

def extract_values_for_qualifier(qualifiers, display_name):
    """
    Extracts the values for a given display_name from a list of qualifiers.

    Parameters:
        qualifiers: List or array of qualifiers dictionaries.
        display_name: display_name to extract from the qualifiers.
    Returns:
        List of values for the given display_name.
    """
    
    extracted_values = []
    for qualifier in qualifiers:
        
        display_names = qualifier["displayNames"]
        display_values = qualifier["values"]
        
        if display_name in display_names:
            idx = display_names.index(display_name)
            extracted_values.append(display_values[idx])
    
    return extracted_values

def extract_next_event_dict(next_event, playerName = None):
    """
    Takes the next event and extracts the relevant information from it.
    Args:
        next_event (pandas.Series): the next event that happened after the current event
        playerName (str): the name of the player who took the current event
    Returns:
        dict: the extracted information
    """

    try:
        qualifiers = ast.literal_eval(next_event['simplified_qualifiers'])
    except:
        qualifiers = next_event['simplified_qualifiers']
        
    next_event_dict = {
        "event_type": next_event['type'],
        "event_success": next_event['outcomeType'],
        "event_location": map_coordinates_to_18_zones(next_event['x'], next_event['y']),
        "event_end_location": map_coordinates_to_18_zones(next_event['endX'], next_event['endY']),
        "event_qualifiers": qualifiers,
        "event_taker": playerName,
        "event_taker_team": next_event['teamName'],
        "event_taker_position": next_event['position'],
        "event_success": "successfully" if next_event['outcomeType'] == "Successful" else "unsuccessfully"
    }
    return next_event_dict

def extract_player_rating(player_ratings, minute, original_minute=None, direction="backward"):
    """
    Extract the player rating at a given minute. Handles cases where the player rating is not available
    by first looking backward one minute at a time and then resetting to the original minute and looking forward if necessary.
    
    Args:
        player_ratings (dict): Dictionary containing the player ratings with minutes as keys.
        minute (int): The minute at which the rating is required.
        original_minute (int, optional): Tracks the original minute to reset if no ratings are found going backward.
        direction (str, optional): Tracks whether we are searching "backward" or "forward". Defaults to "backward".
        
    Returns:
        float: The rating of the player at the given minute.
    """
    if original_minute is None:
        original_minute = minute

    try:
        return player_ratings[str(minute)]
    except KeyError:
        if direction == "backward":
            if minute > 0:
                return extract_player_rating(player_ratings, minute - 1, original_minute, "backward")
            else:
                return extract_player_rating(player_ratings, original_minute, original_minute, "forward")
        elif direction == "forward":
            if minute < 100:  # Assuming the maximum possible minute is 100
                return extract_player_rating(player_ratings, minute + 1, original_minute, "forward")
            else:
                return None

def extract_player_stats(player_stats: dict) -> dict:
    """
    Extracts and aggregates key metrics from the player's stats dictionary, handling missing or inconsistent keys dynamically.
    :param player_stats: Dictionary of stats for a player.
    :return: Aggregated key metrics as a dictionary.
    """
    def safe_get(stat_dict, key, default=0):
        "Safely get the value from a nested dictionary, or return a default."
        return stat_dict.get(key, default) if isinstance(stat_dict, dict) else default

    def sum_stats(stat_dict):
        "Safely sum values from a dictionary, handling None or missing keys."
        return int(sum(stat_dict.values())) if isinstance(stat_dict, dict) else 0

    def get_last_value(stat_dict):
        "Safely get the last value from a dictionary, handling None or missing keys."
        return list(stat_dict.values())[-1] if isinstance(stat_dict, dict) and stat_dict else 0

    result = {
        'total_saves': sum_stats(player_stats.get('totalSaves', {})),
        'collected': sum_stats(player_stats.get('collected', {})),
        'parried_safe': sum_stats(player_stats.get('parriedSafe', {})),
        'claims_high': sum_stats(player_stats.get('claimsHigh', {})),
        'possession': sum_stats(player_stats.get('possession', {})),
        'ratings': get_last_value(player_stats.get('ratings', {})),
        'clearances': sum_stats(player_stats.get('clearances', {})),
        'touches': sum_stats(player_stats.get('touches', {})),
        'passes_total': sum_stats(player_stats.get('passesTotal', {})),
        'passes_accurate': sum_stats(player_stats.get('passesAccurate', {})),
        'pass_success': safe_get(player_stats, 'passSuccess', 0),
        'aerials_total': sum_stats(player_stats.get('aerialsTotal', {})),
        'aerials_won': sum_stats(player_stats.get('aerialsWon', {})),
        'defensive_aerials': sum_stats(player_stats.get('defensiveAerials', {})),
        'interceptions': sum_stats(player_stats.get('interceptions', {})),
        'passes_key': sum_stats(player_stats.get('passesKey', {})),
        'aerial_success': safe_get(player_stats, 'aerialSuccess', 0),
        'offensive_aerials': sum_stats(player_stats.get('offensiveAerials', {})),
        'fouls_committed': sum_stats(player_stats.get('foulsCommited', {})),
        'tackles_total': sum_stats(player_stats.get('tacklesTotal', {})),
        'tackles_successful': sum_stats(player_stats.get('tackleSuccessful', {})),
        'tackle_success': safe_get(player_stats, 'tackleSuccess', 0),
        'dribbles_won': sum_stats(player_stats.get('dribblesWon', {})),
        'dribbles_attempted': sum_stats(player_stats.get('dribblesAttempted', {})),
        'dribble_success': safe_get(player_stats, 'dribbleSuccess', 0),
        'corners_total': sum_stats(player_stats.get('cornersTotal', {})),
        'throw_ins_total': sum_stats(player_stats.get('throwInsTotal', {})),
        'throw_ins_accurate': sum_stats(player_stats.get('throwInsAccurate', {})),
        'dribbles_lost': sum_stats(player_stats.get('dribblesLost', {})),
        'tackles_unsuccessful': sum_stats(player_stats.get('tackleUnsuccesful', {})),
        'dribbled_past': sum_stats(player_stats.get('dribbledPast', {})),
        'shots_total': sum_stats(player_stats.get('shotsTotal', {})),
        'shots_off_target': sum_stats(player_stats.get('shotsOffTarget', {})),
        'shots_off_target': sum_stats(player_stats.get('shotsOffTarget', {})),
        'shots_blocked': sum_stats(player_stats.get('shotsBlocked', {})),
        'shots_on_target': sum_stats(player_stats.get('shotsOnTarget', {})),
        'offsides_caught': sum_stats(player_stats.get('offsidesCaught', {})),
        'dispossessed': sum_stats(player_stats.get('dispossessed', {}))
    }
    return result

def extract_team_standing(league_standing: pd.DataFrame, teamName: str)-> int:
    """ 
    Extracts the team standing from the league standings dataframe for the given teamId
    
    Args:
        league_standing: pd.DataFrame - the league standings dataframe
        teamId: int - the teamId of the team
    Returns:
        str - the standing of the team written in words
    """
    
    # ordinal mapping for the standing
    ordinal_mapping = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth",
    11: "eleventh",
    12: "twelfth",
    13: "thirteenth",
    14: "fourteenth",
    15: "fifteenth",
    16: "sixteenth",
    17: "seventeenth",
    18: "eighteenth"
    }
    
    return ordinal_mapping[int(league_standing[league_standing['teamName'] == teamName].index[0] + 1)]

def extract_team_strength(league_standing: pd.DataFrame, teamName: str)-> str:
    """
    Based on the league standings, extracts the strength of the team.
    Args:
        league_standing: pd.DataFrame - the league standings dataframe
        teamName: str - the teamName of the team
    Returns:
        str - the strength of the team
    """
    
    # if condition that checks the standing of the team and returns the strength
    standing = int(league_standing[league_standing['teamName'] == teamName].index[0] + 1)
    if standing <= 6:
        return "strong"
    elif standing <= 12:
        return "average"
    else:
        return "weak"

def extract_team_goals(league_standing: pd.DataFrame, teamName: str) -> dict:
    """
    Extracts goals for, goals against, goal difference, and averages from the league standings.
    
    Args:
        league_standing: pd.DataFrame - the league standings dataframe
        teamId: int - the teamId of the team
    
    Returns:
        dict - goals metrics for the team
    
    Raises:
        ValueError: If the teamId is not found in the league standings.
    """
    # Extract the team row
    team_row = league_standing[league_standing['teamName'] == teamName]
    if team_row.empty:
        raise ValueError(f"Team Name {teamName} not found in league standings.")
    
    # Retrieve team data
    team_data = team_row.iloc[0]
    games_played = int(team_data['gamesPlayed'])
    goals_for = int(team_data['goalsFor'])
    goals_against = int(team_data['goalsAgainst'])
    goal_difference = goals_for - goals_against
    
    # Compute averages safely
    avg_goals_for = round(goals_for / games_played, 2) if games_played > 0 else 0
    avg_goals_against = round(goals_against / games_played, 2) if games_played > 0 else 0

    return {
        "goals_for": goals_for,
        "goals_against": goals_against,
        "goal_difference": goal_difference,
        "avg_goals_for": avg_goals_for,
        "avg_goals_against": avg_goals_against
    }

def extract_team_form(league_standing: pd.DataFrame, teamName: str)-> str:
    """
    Based on the last five column from the league standings, extracts the form of the team.
    Args:
        league_standing: pd.DataFrame - the league standings dataframe
        teamId: int - the teamId of the team
    Returns:
        str - the form of the team
    """
    
    # extract the last five columns of the team
    team_form = ast.literal_eval(league_standing[league_standing['teamName'] == teamName]['lastFive'].values[0])[-3:]
    
    # if condition that checks the form of the team and returns the form
    if team_form.count("W") >= 2:
        return "good"
    elif team_form.count("D") >= 2:
        return "average"
    else:
        return "bad"
    
def extract_team_info(information: dict)-> dict:
    """
    Extracts information from the league standings dataframe for the given teamId. Used for building chunks.
    Args:
        information: dict - the information about the event
    Returns:
        dict - the modified information dictionary
    """

    league_standing = information['league_standings']
    
    # extract the team strength
    information['event_taker_team_standing'] = extract_team_standing(league_standing, information['event_taker_team'])
    information['event_taker_team_strength'] = extract_team_strength(league_standing, information['event_taker_team'])
    # extract the opponent team strength
    information['opponent_team_standing'] = extract_team_standing(league_standing, information['opponent_team'])
    information['opponent_team_strength'] = extract_team_strength(league_standing, information['opponent_team'])
    
    # extract the team form
    information['opponent_team_form'] = extract_team_form(league_standing, information['opponent_team'])
    information['event_taker_team_form'] = extract_team_form(league_standing, information['event_taker_team'])

    # Extract team goals
    event_taker_goals = extract_team_goals(league_standing, information['event_taker_team'])
    information['event_taker_team_goals_for'] = event_taker_goals['goals_for']
    information['event_taker_team_goals_against'] = event_taker_goals['goals_against']
    information['event_taker_team_goal_difference'] = event_taker_goals['goal_difference']
    information['event_taker_team_avg_goals_for'] = event_taker_goals['avg_goals_for']
    information['event_taker_team_avg_goals_against'] = event_taker_goals['avg_goals_against']

    opponent_goals = extract_team_goals(league_standing, information['opponent_team'])
    information['opponent_team_goals_for'] = opponent_goals['goals_for']
    information['opponent_team_goals_against'] = opponent_goals['goals_against']
    information['opponent_team_goal_difference'] = opponent_goals['goal_difference']
    information['opponent_team_avg_goals_for'] = opponent_goals['avg_goals_for']
    information['opponent_team_avg_goals_against'] = opponent_goals['avg_goals_against']

    return information

def check_extra_time(minute, period):
    """
    Checks if the event is in extra time. Does this by looking at the minute and the period of the game.
    Args:
        minute: int - the minute of the event
        period: str - the period of the game ('FirstHalf' or 'SecondHalf')
    Returns:
        bool - True if the event is in extra time, False otherwise
    """
    # if condition that checks if the minute is greater than 45 and the period is 'FirstHalf' or the minute is greater than 90
    if (minute >= 45 and period == 'FirstHalf') or minute > 90:
        return True
    return False

def draw_customized_football_pitch_with_zones():
    """
    Draws a detailed football pitch with requested customization and zones labeled with numbers only.
    """
    # Define the pitch dimensions
    pitch_length = 120  # Adjusted length for a standard football pitch in meters
    pitch_width = 80    # Adjusted width for a standard football pitch

    # Define thresholds for dividing the field into 6 vertical and 3 horizontal sections
    x_thresholds = [0, 20, 40, 60, 80, 100, 120]  # 6 columns
    y_thresholds = [0, 26.67, 53.33, 80]          # 3 rows

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, pitch_length)
    ax.set_ylim(0, pitch_width)
    ax.set_aspect('equal')

    # Draw pitch boundaries
    ax.plot([0, pitch_length, pitch_length, 0, 0],
            [0, 0, pitch_width, pitch_width, 0], color="gray", linewidth=3)

    # Center circle and center line
    center_circle = patches.Circle((pitch_length / 2, pitch_width / 2), 9.15, color="gray", fill=False, linewidth=2)
    ax.add_patch(center_circle)
    ax.plot([pitch_length / 2, pitch_length / 2], [0, pitch_width], color="gray", linewidth=2, linestyle="--")

    # Add center spot
    ax.scatter(pitch_length / 2, pitch_width / 2, color="gray", s=50)

    # Penalty areas and goals
    # Left penalty area
    ax.plot([0, 16.5, 16.5, 0], [pitch_width / 2 - 20.15, pitch_width / 2 - 20.15, pitch_width / 2 + 20.15, pitch_width / 2 + 20.15],
            color="gray", linewidth=2)
    # Right penalty area
    ax.plot([pitch_length, pitch_length - 16.5, pitch_length - 16.5, pitch_length],
            [pitch_width / 2 - 20.15, pitch_width / 2 - 20.15, pitch_width / 2 + 20.15, pitch_width / 2 + 20.15],
            color="gray", linewidth=2)

    # Add goals
    ax.plot([0, -2, -2, 0], [pitch_width / 2 - 7.32 / 2, pitch_width / 2 - 7.32 / 2, pitch_width / 2 + 7.32 / 2, pitch_width / 2 + 7.32 / 2],
            color="gray", linewidth=2)
    ax.plot([pitch_length, pitch_length + 2, pitch_length + 2, pitch_length],
            [pitch_width / 2 - 7.32 / 2, pitch_width / 2 - 7.32 / 2, pitch_width / 2 + 7.32 / 2, pitch_width / 2 + 7.32 / 2],
            color="gray", linewidth=2)

    # Add penalty spots
    ax.scatter(11, pitch_width / 2, color="gray", s=50)
    ax.scatter(pitch_length - 11, pitch_width / 2, color="gray", s=50)

    # Add "D" arcs at the penalty areas
    left_arc = patches.Arc((16.5, pitch_width / 2), height=18.3, width=18.3,
                        angle=0, theta1=308, theta2=52, color="gray", linewidth=2)
    ax.add_patch(left_arc)
    right_arc = patches.Arc((pitch_length - 16.5, pitch_width / 2), height=18.3, width=18.3,
                            angle=0, theta1=128, theta2=232, color="gray", linewidth=2)
    ax.add_patch(right_arc)

    # Draw the vertical lines (x thresholds) and horizontal lines (y thresholds)
    for x in x_thresholds:
        ax.plot([x, x], [0, pitch_width], color="lightgray", linewidth=2, linestyle="--")
    for y in y_thresholds:
        ax.plot([0, pitch_length], [y, y], color="lightgray", linewidth=2, linestyle="--")

    # Add labels to each zone with numbers only
    zone_number = 1
    for row in range(len(y_thresholds) - 1):
        for col in range(len(x_thresholds) - 1):
            # Calculate the center of the zone for text placement
            center_x = (x_thresholds[col] + x_thresholds[col + 1]) / 2
            center_y = (y_thresholds[row] + y_thresholds[row + 1]) / 2
            ax.text(center_x, center_y, f"{zone_number}", color="cornflowerblue",
                    fontsize=18, ha='center', va='center', weight='bold')
            zone_number += 1

    # Add axis labels
    ax.set_xlabel("Length of the Field (meters)", fontsize=12)
    ax.set_ylabel("Width of the Field (meters)", fontsize=12)

    # Add grass color to the pitch background
    ax.set_facecolor("white")

    plt.show()