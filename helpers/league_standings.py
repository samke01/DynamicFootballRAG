import pandas as pd
import numpy as np
import ast
import os
import traceback   
from helpers.data_handlers import safe_json_loads

def _check_if_kickoff_next(event_df: pd.DataFrame, goal_indices: list, teamId: int, n: int = 5) -> list:
    """
    Checks if any of the next `n` events are kickoff events for a list of goal indices.

    Parameters:
    - event_df (pd.DataFrame): DataFrame containing match event data.
    - goal_indices (list): Indices of the goal events to validate.
    - teamId (int): The team ID of the current event (scoring team).
    - n (int): Number of subsequent events to check.

    Returns:
    - list: A list of booleans where True indicates the goal is followed by a valid kickoff within the next `n` events.
    """
    kickoff_flags = [False] * len(goal_indices)

    for i, index in enumerate(goal_indices):
        try:
            # Check up to the next `n` events
            for offset in range(1, n + 1):
                if index + offset >= len(event_df):
                    break  # Stop if we reach the end of the DataFrame

                next_event = event_df.iloc[index + offset]

                # Validate if the next event is a kickoff
                if (
                    next_event["type"] == "Pass" and
                    abs(next_event["x"] - 50) <= 3 and
                    abs(next_event["y"] - 50) <= 3 and
                    next_event["teamId"] != teamId
                ):
                    kickoff_flags[i] = True
                    break  # Stop checking further events for this goal
                elif next_event["type"] == "End":
                    kickoff_flags[i] = True # Assume the match (or half) ends after a goal
                    break # Stop checking further events for this goal
        except KeyError as e:
            print(f"KeyError: Missing column {e} in event DataFrame.")
        except Exception as e:
            print(f"Unexpected error at index {index}: {e}")

    return kickoff_flags

def get_match_outcome(event_df: pd.DataFrame, error_log_path: str, update=False, n=5, validate_goal = False) -> pd.DataFrame:
    """
    Determines the winner and loser of the match based on event data, including own goals.
    Creates new events: "MatchLost", "MatchWon", "MatchDraw".
    Adds "goalsScored", "goalsConceded", and "pointsGained" to qualifiers.

    Parameters:
    - event_df (pd.DataFrame): DataFrame containing match event data with columns 'teamId', 'type', and 'qualifiers'.
    - error_log_path (str): Path to the error log file with error.txt file name included.
    - update (bool): If True, the match outcome events will be overwritten if they already exist.
    - n (int): Number of subsequent events to check for a valid kickoff.
    - validate_goal (bool): If True, the function will check if the goal is followed by a valid kickoff event.
    
    Returns:
    - pd.DataFrame: Updated DataFrame with match outcome events added.
    """
    # Clear error log file if update=True
    if update:
        with open(error_log_path, "w") as error_file:
            error_file.write("Error log cleared. Starting fresh for this run.\n")

    def _is_own_goal(qualifiers: str) -> bool:
        """
        Determines if a goal is an own goal based on the 'OwnGoal' qualifier.

        Parameters:
        - qualifiers (str): String representation of qualifiers list from the 'qualifiers' column.

        Returns:
        - bool: True if the goal is an own goal, False otherwise.
        """
        try:
            # Parse the qualifiers string
            qualifiers_list = ast.literal_eval(qualifiers)
            # Check if any of the qualifiers indicate an own goal - the logic checks for any qualifier with type "OwnGoal"
            return any(q["type"]["displayName"] == "OwnGoal" for q in qualifiers_list)
        
        except Exception as e:
            # Log parsing errors
            with open(error_log_path, "a") as error_file:
                error_file.write(f"Error parsing qualifiers: {qualifiers}\n")
                error_file.write(f"Exception: {str(e)}\n")
                error_file.write(traceback.format_exc())
            return False

    if "MatchLost" in event_df["type"].values or "MatchWon" in event_df["type"].values or "MatchDraw" in event_df["type"].values:
        if not update:
            print("Match outcome already determined. Skipping.")
            return event_df  # Skip if match outcome already determined and update is False
        else:
            event_df = event_df[~event_df["type"].isin(["MatchLost", "MatchWon", "MatchDraw"])]

    # Get unique team IDs
    teams = event_df["teamId"].unique()
    
    all_goals = event_df[(event_df["type"] == "Goal")]
    all_goals_indices = all_goals.index.tolist()

    # Checking if any of the goals are own goals
    owngoal_condition = [_is_own_goal(event_df.loc[index, "qualifiers"]) for index in all_goals_indices]

    ## Reassigning goals to the opposing team if an own goal is detected
    # checking if any of the goals are own goals
    if owngoal_condition.count(True) > 0:
        print("Own goal/s detected. Reassigning goal to opposing team...")
        # indices of the own goals
        owngoals_indices = all_goals_indices[owngoal_condition.index(True)]
        # which team/s scored the own goal
        try:
            owngoal_teams = event_df.loc[owngoals_indices, "teamId"].unique()
        except Exception as e:
            print(f"Error getting teamId for own goal: {str(e)}.")
            if type(owngoals_indices) is int or len(owngoals_indices) == 1:
                print("This is due to a single own goal being detected.")
                owngoal_teams = [event_df.loc[owngoals_indices, "teamId"]]
        
        for team in owngoal_teams:
            # get the opposing team
            opposing_team_id = teams[teams != team][0]
            event_df.loc[owngoals_indices, "teamId"] = opposing_team_id # reassigning teamId to the opposing team
            event_df.loc[owngoals_indices, "teamName"] = event_df[event_df["teamId"] == opposing_team_id]["teamName"].iloc[0] # reassigning teamName to the opposing team
            print(f"Team {team} goals reassigned to team {opposing_team_id} ({event_df[event_df['teamId'] == opposing_team_id]['teamName'].iloc[0]})")
            
        with open(error_log_path, "a") as error_file:
            error_file.write("------- WARNING LOG ENTRY -------\n")
            error_file.write(f"Own goal detected at index {owngoals_indices}.\n")
            error_file.write(f"Team IDs: {owngoal_teams[0]}\n")
            error_file.write(f"Opposing Team ID: {opposing_team_id}\n")
            error_file.write("Goals reassigned to opposing team.\n")
            error_file.write("-------------------------------\n\n")
            
    # Initialize dictionary to store team goals
    team_goals = {team: 0 for team in teams}
    # Loop through each goal event
    for team in teams:
        try:
            goal_count = 0  # Initialize goal_count to a default value

            # Identify goal indices for the team
            goal_indices = event_df[(event_df["teamId"] == team) & (event_df["type"] == "Goal")].index.tolist()

            if validate_goal:
                # Check if each goal is followed by a valid kickoff within the next `n` events
                kickoff_condition = _check_if_kickoff_next(event_df, goal_indices, team, n)

                # Logging in case of goals not being confirmed by a valid kickoff
                if sum(kickoff_condition) != len(kickoff_condition):
                    with open(error_log_path, "a") as error_file:
                        error_file.write("------- WARNING LOG ENTRY -------\n")
                        error_file.write(f"Team ID: {team}\n")
                        try:
                            team_name = event_df[event_df["teamId"] == team]["teamName"].iloc[0]
                            error_file.write(f"Team Name: {team_name}\n")
                        except:
                            error_file.write("Team Name: Unknown\n")
                        error_file.write(f"Warning: Not all goals for Team {team} were confirmed by a valid kickoff.\n")
                        error_file.write(f"Goal Indices: {goal_indices}\n")
                        error_file.write(f"Kickoff Condition: {kickoff_condition}\n")
                        error_file.write("-------------------------------\n\n")

                # Count goals that are followed by a valid kickoff
                goal_count = sum(kickoff_condition)
            else:
                goal_count = len(goal_indices)

        except Exception as e:
            # Log detailed error information to the error log file
            with open(error_log_path, "a") as error_file:
                error_file.write("------- ERROR LOG ENTRY -------\n")
                error_file.write(f"Team ID: {team}\n")
                try:
                    team_name = event_df[event_df["teamId"] == team]["teamName"].iloc[0]
                    error_file.write(f"Team Name: {team_name}\n")
                except:
                    error_file.write("Team Name: Unknown\n")
                error_file.write(f"Goal Indices: {goal_indices}\n")
                error_file.write("Exception Message:\n")
                error_file.write(f"{str(e)}\n")
                error_file.write("Stack Trace:\n")
                error_file.write(traceback.format_exc())
                error_file.write("-------------------------------\n\n")

            goal_count = 0  # Default goal count in case of error

        # Update team goals
        team_goals[team] = goal_count

        
    # Determine match outcome
    if team_goals[teams[0]] > team_goals[teams[1]]:
        outcome = {
            teams[0]: ("MatchWon", 3, team_goals[teams[0]], team_goals[teams[1]]),
            teams[1]: ("MatchLost", 0, team_goals[teams[1]], team_goals[teams[0]])
        }
    elif team_goals[teams[0]] < team_goals[teams[1]]:
        outcome = {
            teams[0]: ("MatchLost", 0, team_goals[teams[0]], team_goals[teams[1]]),
            teams[1]: ("MatchWon", 3, team_goals[teams[1]], team_goals[teams[0]])
        }
    else:
        outcome = {
            teams[0]: ("MatchDraw", 1, team_goals[teams[0]], team_goals[teams[1]]),
            teams[1]: ("MatchDraw", 1, team_goals[teams[1]], team_goals[teams[0]])
        }

    
    # Append match outcome events
    new_rows = []
    for team, (result, points, scored, conceded) in outcome.items():
        
        # handling max time for the new event
        minute = max(event_df["minute"])
        second = max(event_df[event_df["minute"]==minute]["second"])
        if second == 59:
            minute += 1
            second = 0
            
        new_rows.append({
            "id": max(event_df["id"]) + 1,
            "eventId": max(event_df["eventId"]) + 1,
            "teamId": team,
            "teamName": event_df[event_df["teamId"] == team]["teamName"].iloc[0],
            "minute": minute,
            "second": second,
            "type": result,
            "period": "EndMatch",
            "qualifiers": {
                "goalsScored": scored,
                "goalsConceded": conceded,
                "pointsGained": points
            }
        })

    event_df = pd.concat([event_df, pd.DataFrame(new_rows)], ignore_index=True)
    return event_df

def get_standings(gameweek, input_folder, teams):
    """
    Initialize the league standings for the given gameweek.

    Parameters:
    - gameweek (int): The current gameweek being processed.
    - input_folder (str): The folder path where the gameweek data is stored.
    - teams (pd.DataFrame): A dataframe containing the team data with columns 'teamId' and 'teamName'.
    Returns:
    - pd.DataFrame: A dataframe containing the league standings with columns:
                    ['teamId', 'teamName', 'points', 'goalsFor', 'goalsAgainst', 'goalDifference', 'gamesWon', 'gamesLost', 'gamesPlayed', 'lastFive'].

    If it is the first gameweek, a new empty dataframe is created.
    Otherwise, the previous gameweek's standings are loaded from a CSV file.
    """
    try:
        if gameweek == 1:
            print("Initializing new empty for Gameweek 1...")
            return teams[['teamId', 'teamName']].copy().assign(points=0, goalsFor=0, goalsAgainst=0, goalDifference=0, gamesWon=0, gamesDrawn=0, gamesLost=0, gamesPlayed=0, lastFive="[]")
        else:
            print("Loading previous standings...")
            standings_path = os.path.join(input_folder + str(gameweek - 1), "league_standings.csv")
            
            # Check if file exists before attempting to read
            if os.path.exists(standings_path):
                return pd.read_csv(standings_path)
    except Exception as e:
        print(f"Error loading previous standings: {str(e)}")
        print(traceback.format_exc())
        
def process_match_results(match_results, gw_standings):
    """
    Update the league standings with results from the current gameweek's matches.
    
    Parameters:
    - match_results (pd.DataFrame): A dataframe containing the match results for the gameweek.
                                    Must include columns 'teamId', 'teamName', and 'qualifiers'.
    - gw_standings (pd.DataFrame): The league standings dataframe to be updated.
    - teams (pd.DataFrame): A dataframe containing the team data with columns 'teamId' and 'teamName'.
    Returns:
    - pd.DataFrame: The updated league standings dataframe.
    
    For each team in the match results:
    1. Check if the team is already in the standings, and if not, add a new entry.
    2. Iterate through the match results for each team to update goals and points.
    """
    
    # Check if the input parameters are valid DataFrames
    if not isinstance(gw_standings, pd.DataFrame):
        raise ValueError("The 'gw_standings' parameter must be a valid DataFrame.")
    
    # List to hold new team entries to be appended to the standings
    new_teams = []
    
    # Iterate through each team in the match results
    for team_id in match_results['teamId'].unique():
        
        try:
            # Filter match data for the current team
            team_data = match_results[match_results['teamId'] == team_id]
            
            # Parse the qualifiers dictionary (if stored as string)
            try:
                qualifiers = ast.literal_eval(team_data['qualifiers'].values[0])
            except:
                print("Error parsing qualifiers.")
                if type(qualifiers) is dict:
                    print("Qualifiers are already in dictionary format.")
                    qualifiers = team_data['qualifiers']
                
            update_team_stats(gw_standings, team_id, qualifiers)
        except Exception as e:
            print(f"Error processing team {team_id}: {str(e)}")
    # After processing all the teams, concatenate new teams to the standings if any new teams were added
    if new_teams:
        new_teams_df = pd.DataFrame(new_teams)  # Convert the list of new teams into a DataFrame
        gw_standings = pd.concat([gw_standings, new_teams_df], ignore_index=True)
    
    # Return the updated standings
    return gw_standings

def update_team_stats(standings, team_id, qualifiers):
    """
    Update the statistics for a specific team based on match qualifiers.

    Parameters:
    - standings (pd.DataFrame): The league standings dataframe to be updated.
    - team_id (int or str): The unique identifier for the team.
    - qualifiers (dict): A dictionary containing match statistics such as:
                        'goalsScored', 'goalsConceded', and 'pointsGained'.

    This method updates the team's total goals scored, goals conceded, points,
    and goal difference in the standings dataframe.
    """
    
    def _update_last_five(last_five, result):
        """
        Update the last five matches for a team based on the match result. The last five matches are stored as a list.

        Parameters:
        - last_five (list): A list of the last five match results for the team. Needs to be parsed from a string.
        - result (str): The result of the current match ('W', 'D', or 'L').

        Returns:
        - list: The updated list of the last five match results.
        """

        # Convert the string representation of the last five matches to a list
        last_five = ast.literal_eval(last_five)
        return last_five[-4:] + [result]
    
    goals_scored = qualifiers['goalsScored']
    goals_conceded = qualifiers['goalsConceded']
    points_gained = qualifiers['pointsGained']
    
    # Update goals scored, goals conceded, and points
    standings.loc[standings['teamId'] == team_id, 'goalsFor'] += goals_scored
    standings.loc[standings['teamId'] == team_id, 'goalsAgainst'] += goals_conceded
    standings.loc[standings['teamId'] == team_id, 'points'] += points_gained
    # Update goal difference
    standings.loc[standings['teamId'] == team_id, 'goalDifference'] += (goals_scored - goals_conceded)
    standings.loc[standings['teamId'] == team_id, 'gamesPlayed'] += 1
    
    if points_gained == 1: # Draw
        standings.loc[standings['teamId'] == team_id, 'gamesDrawn'] += 1
        standings.loc[standings['teamId'] == team_id, 'lastFive'] = standings.loc[standings['teamId'] == team_id, 'lastFive'].apply(lambda x: str(_update_last_five(x, 'D')))
    elif points_gained == 3: # Win
        standings.loc[standings['teamId'] == team_id, 'gamesWon'] += 1
        standings.loc[standings['teamId'] == team_id, 'lastFive'] = standings.loc[standings['teamId'] == team_id, 'lastFive'].apply(lambda x: str(_update_last_five(x, 'W')))
    else: # Loss
        standings.loc[standings['teamId'] == team_id, 'gamesLost'] += 1
        standings.loc[standings['teamId'] == team_id, 'lastFive'] = standings.loc[standings['teamId'] == team_id, 'lastFive'].apply(lambda x: str(_update_last_five(x, 'L')))


def save_standings(standings, gameweek, input_folder):
    """
    Save the current gameweek's league standings to a CSV file.

    Parameters:
    - standings (pd.DataFrame): The league standings dataframe to be saved.
    - gameweek (int): The current gameweek number.
    - input_folder (str): The folder path where the standings CSV should be saved.

    The standings are saved as 'league_standings.csv' in the directory for the current gameweek.
    """
    standings_save_path = os.path.join(input_folder + str(gameweek), "league_standings.csv")
    standings.to_csv(standings_save_path, index=False)
    print(f"League standings saved to: {standings_save_path}")
    

def track_match_score(event_df: pd.DataFrame, error_log_path: str, n: int = 5) -> pd.DataFrame:
    """
    Adds a new column 'matchScore' to the event_df that tracks the match score over time.
    Validates goals using the _check_if_kickoff_next method.

    Parameters:
    - event_df (pd.DataFrame): DataFrame containing match event data.
    - error_log_path (str): Path to the error log file with error.txt file name included.
    - n (int): Number of subsequent events to check for a valid kickoff.

    Returns:
    - pd.DataFrame: Updated DataFrame with a new 'matchScore' column.
    """
    try:
        # Initialize the score tracker as a dictionary with both teams' scores set to 0
        teams = event_df["teamId"].unique()
        if len(teams) != 2:
            raise ValueError("There should be exactly 2 unique team IDs in the data.")
        match_score = {str(teams[0]): 0, str(teams[1]): 0}

        # List to store match scores for each event
        match_score_column = []

        # Identify goal events and validate them
        goal_events = event_df[event_df["type"] == "Goal"]
        goal_indices = goal_events.index.tolist()
        valid_goals = _check_if_kickoff_next(event_df, goal_indices, None, n)

        for index, event in event_df.iterrows():
            # If the event is a validated goal, update the score
            if index in goal_indices and valid_goals[goal_indices.index(index)]:
                scoring_team = int(event["teamId"])
                match_score[str(scoring_team)] += 1

            # Append a copy of the current match score
            match_score_column.append(match_score.copy())

        # Add the new column to the DataFrame
        event_df["matchScore"] = match_score_column

    except Exception as e:
        # Log errors to the error log file
        with open(error_log_path, "a") as error_file:
            error_file.write("------- ERROR LOG ENTRY -------\n")
            error_file.write(f"Error while tracking match score:\n{str(e)}\n")
            error_file.write(traceback.format_exc())
            error_file.write("-------------------------------\n\n")

    return event_df