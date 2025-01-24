import json
import pandas as pd
import ast

# Function to safely handle JSON parsing and empty lists
def safe_json_loads(val):
    """
    Safely parse a JSON string and return None for invalid or empty lists.
    Parameters:
        val: JSON string to parse.
    Returns:
        Parsed JSON object or None if the JSON is invalid or an empty list.
    """
    try:
        if val.strip() == "[]":  # Handle empty list case
            return []
        # Replace single quotes with double quotes and parse JSON
        fixed_val = val.replace("'", '"')
        return json.loads(fixed_val)
    except Exception:
        return None  # Return None for invalid or unparsable JSON
    
def load_event_mapping(file_path: str) -> dict:
    """
    Loads a mapping from a json file.
    Args:
        file_path (str): the path to the file
    Returns:
        dict: the mapping
    """
    with open(file_path, 'r') as file:
        mapping = json.load(file)
    return mapping

def basic_player_data(player_df: pd.DataFrame, columns = ['teamId', 'playerId', 'name', 'shirtNo', 'age', 'height', 'position']) -> pd.DataFrame:
    """
    Modifies the player data to be unique and contain constant information on player attributes
    
    Args:
        player_df: DataFrame containing player data
        columns: List of columns of interest. These columns will be extracted and the rest will be dropped
    Returns:
        player_df: Modified DataFrame
    """

    return player_df[columns].drop_duplicates().reset_index(drop=True)

def simplify_qualifiers(event_df: pd.DataFrame, skip_events: list = None) -> pd.DataFrame:
    """
    Simplify the qualifiers column of the event dataframe.
    Extracts the 'displayName' and 'value' for each qualifier and simplifies the structure.
    Skips events in the `skip_events` list, directly transferring the qualifiers as-is.

    Args:
        event_df (pd.DataFrame): The event dataframe with a 'qualifiers' column.
        skip_events (list): List of event types to skip simplification for.

    Returns:
        pd.DataFrame: Updated dataframe with simplified 'qualifiers'.
    """
    def process_qualifiers(row):
        # Skip events listed in skip_events
        if skip_events and row['type'] in skip_events:
            return row['qualifiers']

        # Simplify qualifiers for other events
        try:
            qualifiers = ast.literal_eval(row['qualifiers']) if isinstance(row['qualifiers'], str) else row['qualifiers']
            if not qualifiers:
                return {"displayNames": [], "values": []}
            display_names = [q["type"]["displayName"] for q in qualifiers]
            values = [q.get("value") if not isinstance(q.get("value"), list) else ", ".join(map(str, q.get("value"))) for q in qualifiers]
            return {"displayNames": display_names, "values": values}
        except (ValueError, SyntaxError, KeyError, TypeError):
            return {"displayNames": [], "values": []}

    # Apply the helper function row-wise
    event_df["simplified_qualifiers"] = event_df.apply(process_qualifiers, axis=1)
    
    return event_df

def get_ordner(data_state: str) -> str:
    """
    Get the path to the folder containing the data for the given data state and data type.
    Args:
        data_state: The data state to get the path for.
    Returns:
        The path to the folder containing the data for the given
    """
    return f"data/Bundesliga/{data_state}/GW"

def map_coordinates_to_18_zones(x, y):
    """
    Maps the x and y coordinates to one of 18 zones on the football field.
    
    The field is divided into 6 vertical zones (columns) and 3 horizontal zones (rows):
        - Zones are labeled Zone1 to Zone18, numbered from top-left to bottom-right.
    
    Args:
        x (float): The x-coordinate (0-100 range).
        y (float): The y-coordinate (0-100 range).
    
    Returns:
        str: The zone label (Zone1 to Zone18).
    """
    # Define thresholds for dividing the field into 6 vertical and 3 horizontal sections
    x_thresholds = [0, 16.67, 33.33, 50, 66.67, 83.33, 100]  # 6 columns
    y_thresholds = [0, 33.33, 66.67, 100]                    # 3 rows

    # Determine the column (x_zone) and row (y_zone) based on thresholds
    x_zone = sum([x > t for t in x_thresholds])
    y_zone = sum([y > t for t in y_thresholds])

    # Calculate the zone number (1-18) based on x_zone and y_zone
    # subtract 1 from y_zone to start numbering from 0 then multiply by 6 (as 1 y_zone is 6 x_zones) and then add x_zone
    calculated_zone = (y_zone - 1) * 6 + x_zone
    
    return f"Field Zone {(y_zone - 1) * 6 + x_zone}" if calculated_zone > 0 else "Out of Play"
