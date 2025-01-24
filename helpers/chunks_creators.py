import pandas as pd
import numpy as np
import json
import ast
from helpers.extract_info import extract_next_event_dict, extract_team_info, extract_player_stats

def get_information_template() -> dict:
    """
    Returns a dictionary template for the information about an event.
    Returns:
        dict: dictionary template
    """
    
    return {
        "game_week": None,
        "event_type": None,
        "event_success": None,
        "event_type_mapped": None,
        "event_description": None,
        "event_qualifiers": None,
        "event_period": None,
        "event_time": None,
        "extra_time": None,
        "event_location": None,
        "event_end_location": None,
        "event_taker": None,
        "event_taker_rating": None,
        "event_taker_team": None,
        "event_taker_position": None,
        "match_score": None,
        "opponent_team": None,
        "next_event": None
    }
    

# ================= Team-Level Chunk ================= #
def build_team_chunk(information: dict) -> str:
    """
    Builds a team chunk that contains information about the team's performance in the until the current game week.
    Args:
        information (dict): Information about the event as dictionary. Use method 'get_information_template()' to get an empty dictionary.
    Returns:
        str: the team chunk
    """
    
    # extracting info from league standings
    information = extract_team_info(information) if information['game_week'] != 1 else information
    
    # League standings context
    if information['game_week'] != 1:
        league_standings = (
            f"As of the game week {information['game_week']}, {information['event_taker_team']} is placed {information['event_taker_team_standing']} "
            f"in the league. They are considered a '{information['event_taker_team_strength']}' Bundesliga team. "
            f"{information['event_taker_team']} have scored {information['event_taker_team_goals_for']} goals and conceded {information['event_taker_team_goals_against']} goals so far, "
            f"with an average of {information['event_taker_team_avg_goals_for']} goals scored and {information['event_taker_team_avg_goals_against']} goals conceded per game. "
            f"Their form in the last three matches is '{information['event_taker_team_form']}'.\n\n"
            f"The opponent, {information['opponent_team']}, is placed {information['opponent_team_standing']} "
            f"in the league. They are considered a '{information['opponent_team_strength']}' Bundesliga team. "
            f"{information['opponent_team']} have scored {information['opponent_team_goals_for']} goals and conceded {information['opponent_team_goals_against']} goals so far, "
            f"with an average of {information['opponent_team_avg_goals_for']} goals scored and {information['opponent_team_avg_goals_against']} goals conceded per game. "
            f"Their form in the last three matches is '{information['opponent_team_form']}'."
        )
    else:
        league_standings = "This is the first game of the season, so no standings or form data is available."

    chunk = (
        f"Bundesliga Season 2023/2024 │ Game Week: {information['game_week']}\n\n"
        
        f"{information['event_taker_team']} is playing against {information['opponent_team']}. {league_standings}\n\n"

    )

    return chunk

# ================= Event-Level Chunk ================= #

def build_event_chunk(information: dict) -> str:
    """ 
    Builds a basic event chunk.
    Args:
        information (dict): Information about the event as dictionary. Use method 'get_information_template()' to get an empty dictionary.
        This needs to contain the following keys:
        - game_week (int): the game week of the event
        - event_type (str): the type of the event. E.g. 'Goal', 'Card', 'Pass' etc.
        - event_success (str): the success of the event. 'successfully' or 'unsuccessfully'.
        - event_type_mapped (str): the type of the event mapped to a more natural title for the event. E.g. 'Foul' -> 'Foul Committed', 'CornerAwarded' -> 'Corner Kick Awarded' etc. Mapping is from the event_mapping.json file.
        - event_description (str): the description of the event. E.g. 'scored a goal', 'received a yellow card', 'performed a pass' etc.
        - event_qualifiers (dict): simplified qualifiers of the event that were generated using helpers.data_handlers method "simplify_qualifiers". Includes specific information about event type. These need to be parsed to be in form of a dictionary.
        - event_period (str): the period of the event. 'First Half' or 'Second Half'.
        - event_time (str): the time of the event
        - extra_time (boolean): True if the event happened in extra time, False otherwise
        - event_location (str): the location of the event
        - event_end_location (str): the end location of the event
        - event_taker (str): the person who is taking the event
        - event_taker_rating (str): the rating of the person who is taking the event
        - event_taker_team (str): the team of the person who is taking the event
        - match_score (dict): the score of the with team names as keys and scores as values
        - event_taker_position (str): the position of the person who is taking the event
        - opponent_team (str): the opponent team
        - next_event (pandas.Series): the next event that happened after this event. Used to determine the context of the event.
    Returns:
        str: the event chunk
    """

    # events without a location
    no_location = ["FormationSet", "FormationChange", "Start", "End", "MatchWon", "MatchLost", "MatchDraw", "SubstitutionOn", "SubstitutionOff"]
    location = "" if information['event_type'] in no_location else f" The {information['event_type_mapped'].lower()} took place in {information['event_location'].lower()}."
    
    extra_time = "extra time" if information['extra_time'] else "regular play"
    
    # extracting info from league standings
    information = extract_team_info(information) if information['game_week'] != 1 else information
    
    # build the specific event
    event = build_specific_event(information, location)
    
    strength = f"- {information['event_taker_team']} ({information['event_taker_team_strength']} team in {information['event_taker_team_form']} form, placed {information['event_taker_team_standing']}) is playing against {information['opponent_team']} ({information['opponent_team_strength']} team in {information['opponent_team_form']} form, placed {information['opponent_team_standing']}).\n" if information['game_week'] != 1 else "It is the first game of the season, so no standings or form data is available.\n"

    game_context = (
        "Game context:"
        f"{strength}"
        f"- The game is in the {information['event_period']} and it is {extra_time}.\n"
        f"- Score at minute {information['event_time']} is {information['event_taker_team']} {information['match_score'][str(information['event_taker_team'])]}:{information['match_score'][str(information['opponent_team'])]} {information['opponent_team']}.\n\n"
    )
    
    chunk = (
        f"Bundesliga Season 2023/2024 │ Game Week: {information['game_week']} │ Time (in minutes): {information['event_time']} │ Event: {information['event_type_mapped']} \n\n"
        f"{game_context}"
        f"In-game event:\n{event}\n"
    )
    
    return chunk

def build_specific_event(information: dict, location: str) -> str:
    """ 
    Builds the specific event to be added to the event chunk. It considers all different event types to generate specific descriptions.
    
    Args:
        information (dict): Information about the event as dictionary. Use method 'get_information_template()' to get an empty dictionary.
        location (str): Sentence about the location of the event.
    Returns:
        str: the specific event
    
    """

    # events without a taker (e.g. formation changes, start and end of the game) - event Foul sometimes does not have a taker (we are unsure why)
    no_taker = ["FormationChange", "Start", "MatchWon", "MatchLost", "MatchDraw"]
    
    # add rating of the player if the event has a taker
    if information['event_time'] == None:
        rating = ""
    elif information['event_type'] not in no_taker :
        
        if information['event_taker_rating'] is None:
            rating = ""
        else:
            # extracting the minute of the game from the string
            minute = int(information['event_time'].split(":")[0])
            # we give players time to warm up and get into the game before we start rating them
            if minute <= 15:
                good_bad_performance = f" and he is already having an impact on the game." if information['event_taker_rating'] > 7 else f" and he is yet to make a major impact on the game."
            else:
                good_bad_performance = f". {information['event_taker']} is having a good game." if information['event_taker_rating'] > 6.5 else f". {information['event_taker']} is having a bad game."

            rating = f" {information['event_taker']} rating is currently {information['event_taker_rating']}{good_bad_performance}"
            
    # current score of the match
    current_score = f"\n\nCurrently the score is {information['event_taker_team']} {information['match_score'][str(information['event_taker_team'])]}:{information['match_score'][str(information['opponent_team'])]} {information['opponent_team']}."
    
    if information['event_type'] in no_taker:
        return build_no_taker_event(information, current_score)
    elif information['event_type'] == "Foul":
        return build_foul_event(information) + location + rating + current_score
    elif information['event_type'] == "Goal":
        return build_goal_event(information) + location + rating + current_score
    elif information['event_type'] == "Pass":
        return build_pass_event(information) + rating + current_score
    elif information['event_type'] == "Save":
        return build_save_event(information) + rating + current_score
    else:
        return f"{information['event_taker']} from {information['event_taker_team']} {information['event_success']} {information['event_description']}." + location +  rating + current_score

def build_no_taker_event(information: dict, current_score: str) -> str:
    """
    Builds an event without a taker.
    Args:
        information (dict): Information about the event as dictionary. Use method 'get_information_template()' to get an empty dictionary.
        current_score (str): the current score of the match - relevant for formation changes
    Returns:
        str: the event
    """

    if information['event_type'] == "FormationSet":
        return f"The coaching staff {information['event_description']}." + current_score
    
    elif information['event_type'] == "FormationChange":
        return f"The coaching staff {information['event_description']}." + current_score
    
    elif information['event_type'] == "Start":
        return f"The game between the two teams has started."
    
    elif information['event_type'] == "MatchWon":
        return f"The game between the two teams has ended in the {information['event_time']} minute. {information['event_taker_team']} won the match by scoring {information['event_qualifiers']['goalsScored']} and conceding {information['event_qualifiers']['goalsConceded']}."
    
    elif information['event_type'] == "MatchLost":
        return f"The game between the two teams has ended in the {information['event_time']} minute. {information['event_taker_team']} lost the match by scoring {information['event_qualifiers']['goalsScored']} and conceding {information['event_qualifiers']['goalsConceded']}."
    
    elif information['event_type'] == "MatchDraw":
        return f"The game between the two teams has ended in the {information['event_time']} minute. The result of the game is a draw. Both teams scored {information['event_qualifiers']['goalsScored']} goals."

def build_foul_event(information: dict) -> str:
    """
    Builds a foul event.
    Args:
        information (dict): Information about the event as dictionary. Use method 'get_information_template()' to get an empty dictionary.
    Returns:
        str: the foul event
    """
    
    if information['event_taker'] is None:
        return f"{information['opponent_team']} {information['event_description']}. There is no information on who the player that committed the foul is."
    else:
        return f"{information['event_taker']} from {information['event_taker_team']} {information['event_description']}."

def build_goal_event(information: dict) -> str:
    """ 
    Builds a goal event.
    Args:
        information (dict): Information about the event as dictionary. Use method 'get_information_template()' to get an empty dictionary.
    Returns:
        str: the goal event
    """
    qualifiers = information.get('event_qualifiers', [])
    # Add specific details based on simplified qualifiers
    display_names = qualifiers.get("displayNames", [])
    values = qualifiers.get("values", [])
    
    # Initialize description with base event
    if 'OwnGoal' in display_names:
        description = f"{information['event_taker']} from {information['opponent_team']} playing as a {information['event_taker_position']} scored an own goal from {information['event_location'].lower()} "
    else:
        description = f"{information['event_taker']} from {information['event_taker_team']} playing as a {information['event_taker_position']} {information['event_description']} from {information['event_location'].lower()} "

    if 'LeftFoot' in display_names:
        description += "with his left foot. "
    if 'RightFoot' in display_names:
        description += "with the right foot. "
    if 'Head' in display_names:
        description += "by heading the ball into the net. "
    if 'Penalty' in display_names:
        description += "The goal was scored from a penalty. "
    if 'FastBreak' in display_names:
        description += "This goal was the result of a quick counter-attack. "
    if 'Volley' in display_names:
        description += "It was a spectacular volley finish. "
    if 'BigChance' in display_names:
        description += "The goal came from a significant scoring opportunity. "
    if 'FromCorner' in display_names:
        description += "The goal was scored following a well-placed corner kick. "
    if 'Assist' in display_names:
        description += "The goal was assisted by a teammate. "
    if 'IndividualPlay' in display_names:
        description += "The goal was a result of outstanding individual skill. "
    if 'SmallBoxCentre' in display_names:
        description += "The goal was scored from close range, right in the center of the box. "
    if 'BoxLeft' in display_names:
        description += "The goal was taken from the left side of the box. "
    if 'BoxRight' in display_names:
        description += "The goal was taken from the right side of the box. "
    if 'OutOfBoxCentre' in display_names:
        description += "The goal was a strike from outside the box. "
    if 'HighCentre' in display_names:
        description += "The shot went high into the center of the net. "
    if 'HighLeft' in display_names:
        description += "The shot went high into the top left corner. "
    if 'HighRight' in display_names:
        description += "The shot went high into the top right corner. "
    if 'LowCentre' in display_names:
        description += "The ball was placed low into the center of the net. "
    if 'LowLeft' in display_names:
        description += "The ball was placed low into the bottom left corner. "
    if 'LowRight' in display_names:
        description += "The ball was placed low into the bottom right corner. "
    if 'DeepBoxLeft' in display_names:
        description += "The goal was scored from deep on the left side of the box. "
    if 'DeepBoxRight' in display_names:
        description += "The goal was scored from deep on the right side of the box. "
    if 'ThrowinSetPiece' in display_names:
        description += "The goal was scored following a throw-in set piece. "
    if 'DirectFreekick' in display_names:
        description += "The goal came from a direct free kick. "
    if 'SetPiece' in display_names:
        description += "The goal was scored from a set piece. "
    if 'ThirtyFivePlusCentre' in display_names:
        description += "The goal was a strike from over 35 yards out. "
    if 'FirstTouch' in display_names:
        description += "The goal was scored with a first touch. "

    return description

def build_pass_event(information: dict) -> str:
    """
    Builds a pass event.
    Args:
        information (dict): Information about the event as dictionary. Use method 'get_information_template()' to get an empty dictionary.
    Returns:
        str: the pass event
    """
    qualifiers = information.get('event_qualifiers', [])
    # Add specific details based on simplified qualifiers
    display_names = qualifiers.get("displayNames", [])
    values = qualifiers.get("values", [])
    
    # checking if the next event has a player
    if information['next_event']['event_taker'] is None:
        pass_receiver = " "
    # the next event is from the same team
    elif information['next_event']['event_taker_team'] == information['event_taker_team']:
        pass_receiver = f" The pass was received by teammate {information['next_event']['event_taker']}. "
    else:
        pass_receiver = " "
    
    # checking the type of the next event
    next_event_type = information['next_event']['event_type']
    
    # pass can have multiple types
    pass_type = ""
    if 'Offensive' in display_names:
        pass_type += "offensive "
    if 'Longball' in display_names:
        pass_type += "long ball "
    if 'Chipped' in display_names:
        pass_type += "chipped "
    if 'ThrowIn' in display_names:
        pass_type += "throw-in. "
        
    
    # assigning the pass type to the event description - we overwrite the static event description
    information['event_description'] = f"performed a {pass_type}pass"
    
    # additional information based on the way the pass was made
    if 'LeftFoot' in display_names:
        information['event_description'] += " with the left foot"
    if 'RightFoot' in display_names:
        information['event_description'] += " with the right foot"
    if 'HeadPass' in display_names:
        information['event_description'] += " with the head"
    
    # Construct the base description
    description = (
        f"{information['event_taker']} from {information['event_taker_team']} playing as a {information['event_taker_position']} "
        f"{information['event_success']} performed a {pass_type}pass "
        f"from {information['event_location'].lower()} to {information['event_end_location'].lower()}."
    )

    # Add qualifiers that provide additional context
    if 'ShotAssist' in display_names:
        description += f" The pass assisted a shot taken by {information['next_event']['event_taker']}."
    if 'KeyPass' in display_names:
        description += " The pass was a key moment in play."
    if 'GoalKick' in display_names:
        description += " The pass originated from a goal kick."
    if 'Cross' in display_names:
        description += " The pass was a cross."
    if 'FreekickTaken' in display_names:
        description += " The pass was played from a freekick."
        
    # Add the pass receiver if available
    description += pass_receiver
    
    # next event taker
    next_event_taker = information['next_event']['event_taker']
    
    if next_event_type == 'Pass' and information['next_event']['event_taker_team'] == information['event_taker_team']:
        description += f"The pass reciever {next_event_taker} performed another pass to continue the play. "
    elif next_event_type == 'Pass' and information['next_event']['event_taker_team'] != information['event_taker_team']:
        description += f"The pass was lost to the opposing player {next_event_taker} who then performed a pass. "
    elif next_event_type == 'BallRecovery':
        description += "The pass led to a ball recovery by the opposing team. "
    elif next_event_type == 'BallTouch' and information['next_event']['event_taker_team'] != information['event_taker_team']:
        description += f"The pass was briefly touched by the opposing player {next_event_taker}. "
    elif next_event_type == 'Aerial':
        description += "The pass initiated an aerial duel. "
    elif next_event_type == 'Clearance':
        description += f"The pass was cleared by {next_event_taker} from {information['next_event']['event_taker_team']}. "
    elif next_event_type == 'TakeOn' and information['next_event']['event_taker_team'] == information['event_taker_team']:
        description += f"The pass set up the teammate {next_event_taker} for a take-on attempt. "
    elif next_event_type == 'BlockedPass':
        description += f"The pass was blocked by the opposing player {next_event_taker}. "
    elif next_event_type == 'Interception' and information['next_event']['event_taker_team'] != information['event_taker_team']:
        description += "The pass was intercepted by the opposing team. "
    elif next_event_type == 'Foul' and information['next_event']['event_taker_team'] != information['event_taker_team']:
        description += "The play following the pass resulted in a foul by the opponent. "
    elif next_event_type == 'Foul' and information['next_event']['event_taker_team'] == information['event_taker_team']:
        description += f"The play following the pass resulted in a foul by a teammate. "
    elif next_event_type == 'Dispossessed':
        description += "The pass led to a dispossession by the opposing team. "
    elif next_event_type == 'SavedShot':
        description += f"The pass set up teammate {next_event_taker} for a shot. The attempted shot was saved by the goalkeeper. "
    elif next_event_type == 'KeeperPickup':
        description += "The pass was picked up by the opposing goalkeeper. "
    elif next_event_type == 'Challenge':
        description += "The pass initiated a challenge for possession. "
    elif next_event_type == 'MissedShots':
        description += f"The pass led to a shot by {next_event_taker}. The attempted shot missed the target. "
    elif next_event_type == 'OffsidePass':
        description += "The pass was mistimed and it resulted in an offside call. "
    elif next_event_type == 'Goal':
        description += f"The pass directly contributed to a goal by {next_event_taker}. "
    elif next_event_type == 'OffsideGiven':
        description += f"The play following the pass was ruled offside for {information['next_event']['event_taker_team']}. "
    elif next_event_type == 'Claim':
        description += "The pass was claimed by the opposing goalkeeper. "
    elif next_event_type == 'End':
        description += "The pass was the last action of the match. "
    elif next_event_type == 'KeeperSweeper':
        description += "The pass was intercepted by a sweeper-keeper action. "
    elif next_event_type == 'SubstitutionOff':
        description += "The pass was followed by a player substitution. "
    elif next_event_type == 'ShieldBallOpp':
        description += "The pass led to the shielding of the ball by an opponent. "
    elif next_event_type == 'Punch':
        description += "The pass led to a punch clearance by the goalkeeper. "
    elif next_event_type == 'Error' and information['next_event']['event_taker_team'] == information['event_taker_team']:
        if information['next_event']['event_success'] == 'unsuccessfully':
            description += "The pass was mishandled by the teammate. "
        else:
            description += "The pass was successful but the play following it resulted in an error. "
    elif next_event_type == 'Error' and information['next_event']['event_taker_team'] != information['event_taker_team']:
        description += "The pass was intercepted by the opposing team. "
    elif next_event_type == 'ShotOnPost':
        description += f"The pass set up teammate {next_event_taker} for a shot. The attempted shot hit the post. "
    elif next_event_type == 'Smother':
        description += "The pass was smothered by the goalkeeper. "
    elif next_event_type == 'CornerAwarded' and information['next_event']['event_taker_team'] == information['event_taker_team']:
        description += f"The pass resulted in a corner kick for the team {information['next_event']['event_taker_team']}. "
    elif next_event_type == 'CornerAwarded' and information['next_event']['event_taker_team'] != information['event_taker_team']:
        description += f"The pass was played out of bounds and resulted in a corner kick for the opposing team. "
    elif next_event_type == 'Card':
        description += "The play following the pass resulted in a card. "
    elif next_event_type == 'ChanceMissed':
        description += f"The pass led to a missed scoring opportunity by {information['next_event']['event_taker_team']}. "
    elif next_event_type == 'CrossNotClaimed':
        description += "The pass was a cross that was not claimed. "
    elif next_event_type == 'Tackle' and information['next_event']['event_taker_team'] != information['event_taker_team']:
        description += f"The passer was {information['next_event']['event_success']} tackled by {next_event_taker}. "
    elif next_event_type == 'Save':
        description += "The pass led to a save. "
        
    return description


def build_save_event(information: dict) -> str:
    """
    Builds a save event. Specific method needed as saves can be made by goalkeepers or outfield players.
    Args:
        information (dict): Information about the event as dictionary. Use method 'get_information_template()' to get an empty dictionary.
    Returns:
        str: the save event
    """
    
    if information['event_taker_position'] == 'Goalkeeper':
        return f"{information['event_taker_team']} {information['event_taker_position']} {information['event_taker']} {information['event_success']} made a goalkeeper save. The goalkeeper performed the save in {information['event_location'].lower()}."
    else:
        return f"{information['event_taker_team']} {information['event_taker_position']} {information['event_taker']} {information['event_success']} saved the ball. The save was made in {information['event_location'].lower()}."

# ================= Player-Level Chunk ================= #

def build_player_chunk(player_row: pd.Series, team_mapping: dict, position_mapping: dict, gameweek: int) -> str:
    """
    Builds a chunk of text for a player's stats with enhanced details.
    :param player_row: A row from the player DataFrame containing player information and stats.
    :param team_mapping: A dictionary mapping team IDs to team names.
    :param position_mapping: A dictionary mapping position IDs to position names.
    :param gameweek: The current game week.
    :return: A string chunk summarizing the player's stats.
    """
    import numpy as np
    import ast

    player_name = player_row['name']
    player_team = team_mapping.get(player_row['teamId'], "Unknown Team")
    opponent_team_id = next((team_id for team_id in team_mapping if team_id != player_row['teamId']), None)
    opponent_team = team_mapping.get(opponent_team_id, "Unknown Opponent")
    position = position_mapping.get(player_row['position'], "Unknown Position")
    
    # Extract and parse player stats
    player_stats = player_row['stats'].values[0] if isinstance(player_row['stats'], pd.Series) else player_row['stats']
    stats = ast.literal_eval(player_stats) if isinstance(player_stats, str) else player_stats
    aggregated_stats = extract_player_stats(stats)

    # Determine player status
    if player_row.get('isFirstEleven', False):
        player_status = "started the match"
    elif not np.isnan(player_row.get('subbedInPlayerId', float('nan'))):
        player_status = "was substituted in during the game"
    else:
        player_status = "did not play"

    # Performance description
    performance_rating = aggregated_stats['ratings']
    if performance_rating > 8:
        performance = "an outstanding"
    elif performance_rating > 7:
        performance = "a very good"
    elif performance_rating > 6:
        performance = "a decent"
    else:
        performance = "a below-average"

    # Generate performance sentences dynamically
    pass_accuracy = (
        f"{aggregated_stats['passes_accurate']} out of {aggregated_stats['passes_total']} passes were accurate "
        f"({np.round((aggregated_stats['passes_accurate'] / aggregated_stats['passes_total']) * 100, 2)}% success rate)."
        if aggregated_stats['passes_total'] > 0 else "No passes attempted."
    )
    
    aerial_duels = (
        f"Contested {aggregated_stats['aerials_total']} aerial duels, winning {aggregated_stats['aerials_won']} "
        f"({np.round((aggregated_stats['aerials_won'] / aggregated_stats['aerials_total']) * 100, 2)}% success rate)."
        if aggregated_stats['aerials_total'] > 0 else "Did not engage in aerial duels."
    )
    
    tackles = (
        f"Attempted {aggregated_stats['tackles_total']} tackles, with {aggregated_stats['tackles_successful']} being successful "
        f"({np.round((aggregated_stats['tackles_successful'] / aggregated_stats['tackles_total']) * 100, 2)}% success rate)."
        if aggregated_stats['tackles_total'] > 0 else "No tackles attempted."
    )
    
    dribbles = (
        f"Attempted {aggregated_stats['dribbles_attempted']} dribbles, winning {aggregated_stats['dribbles_won']} "
        f"({np.round((aggregated_stats['dribbles_won'] / aggregated_stats['dribbles_attempted']) * 100, 2)}% success rate)."
        if aggregated_stats['dribbles_attempted'] > 0 else "No dribbles attempted."
    )

    shooting = (
        f"Had {aggregated_stats['shots_total']} shots, with {aggregated_stats['shots_on_target']} on target and "
        f"{aggregated_stats['shots_off_target']} off target."
        if aggregated_stats['shots_total'] > 0 else "Did not attempt any shots."
    )

    fouls = (
        f"Committed {aggregated_stats['fouls_committed']} fouls."
        if aggregated_stats['fouls_committed'] > 0 else "The player did not commit any fouls"
    )

    offsides = (
        f"Was caught offside {aggregated_stats['offsides_caught']} times." if aggregated_stats['offsides_caught'] > 0 else "Was not caught offside."
    )
    
    interceptions = (
        f"Recorded {aggregated_stats['interceptions']} interception(s)." if aggregated_stats['interceptions'] > 0 else "Did not record any interceptions."
    )
    
    clearances = (
        f"Completed {aggregated_stats['clearances']} clearance(s)." if aggregated_stats['clearances'] > 0 else "Did not complete any clearances."
    )
    
    if position.lower() in {'gk', 'goalkeeper'}:
        save_contributions = (
            f"- Saves: Made {aggregated_stats['total_saves']} goalkeeper save(s).\n"
            f"- Claims High: Claimed {aggregated_stats['claims_high']} high ball(s).\n"
            f"- Parried Safely: Successfully parried {aggregated_stats['parried_safe']} shot(s).\n"
            f"- Collected: Gathered the ball {aggregated_stats['collected']} time(s).\n"
        )
    else:
        save_contributions = (
            f"- Defensive Saves: {aggregated_stats['total_saves']} save(s) made to prevent goals.\n" if aggregated_stats['total_saves'] > 0 else "- Defensive Saves: Did not make any defensive saves.\n" 
        )

    # Build the final chunk
    if player_status == "did not play":
        return (
            f"Bundesliga 2023/2024 │ Game Week: {gameweek} │ Player Performance: {player_name}\n\n"
            f"{player_name} did not participate in the game against {opponent_team}.\n"
        )

    # check if player is the mvp
    if player_row['isManOfTheMatch']:
        mvp = f" {player_name} was awarded the Man of the Match award for his performance in the game."
    else:
        mvp = ""
    
    return (
        f"Bundesliga 2023/2024 │ Game Week: {gameweek} │ Player Performance: {player_name}\n\n"
        f"{player_name}, playing as a {position}, {player_status} for {player_team} against {opponent_team}. "
        f"The player delivered {performance} performance with a final rating of {performance_rating:.1f}.{mvp}\n\n"
        f"Performance Breakdown:\n"
        f"- Total Touches: {aggregated_stats['touches']}\n"
        f"- Passing: {pass_accuracy}\n"
        f"- Dribbles: {dribbles}\n"
        f"- Shooting: {shooting}\n"
        f"- Aerial Duels: {aerial_duels}\n"
        f"- Tackles: {tackles}\n"
        f"- Interceptions: {interceptions}\n"
        f"- Clearances: {clearances}\n"
        f"- Fouls: {fouls}\n"
        f"- Offsides: {offsides}\n"
        f"{save_contributions}\n"
    )
    
def process_player(player_df: pd.DataFrame, team_mapping: dict, position_mapping: dict, gameweek: int) -> list:
    """
    Processes a player DataFrame and prints a summary chunk for each player.
    :param player_df: DataFrame containing player information and stats.
    :param team_mapping: A dictionary mapping team IDs to team names.
    :param position_mapping: A dictionary mapping position IDs to position names.
    :param gameweek: The current game week.
    :return: A list of player summary chunks.
    """
    player_summaries = []
    for _, row in player_df.iterrows():
        chunk = build_player_chunk(row, team_mapping, position_mapping, gameweek)
        player_summaries.append(chunk)
    return player_summaries
