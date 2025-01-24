import pandas as pd
import numpy as np


def find_event_sequence(event_data, event_1, event_2):
    """
    Identifies occurrences where event_2 directly follows event_1.

    Parameters:
        event_data (pd.DataFrame): DataFrame containing event data with a 'type' column.
        event_1 (str): The type of the first event to look for.
        event_2 (str): The type of the second event to look for.

    Returns:
        pd.DataFrame: DataFrame containing the rows of event_1 that are followed by event_2.
    """
    indices = []

    for i, event in enumerate(event_data.iterrows()):
        event = event[1]
        if event["type"] != event_1:
            continue
        # Ensure the next event exists before checking
        if i + 1 < len(event_data) and event_data.iloc[i + 1]["type"] == event_2:
            indices.append(i)

    return event_data.iloc[indices]


def find_events_with_qualifier(event_qualifiers, qualifier_name):
    """
    Identifies events where a certain qualifier is found in display names.

    Parameters:
        event_qualifiers (list of dict): List of event qualifiers, each containing a 'displayNames' field.
        qualifier_name (str): The qualifier name to search for in 'displayNames'.

    Returns:
        numpy.array: A numpy.array of booleans where each entry corresponds to whether the qualifier_name
            is found in the 'displayNames' of the respective event.
    """
    filter = np.zeros(len(event_qualifiers), dtype=bool)

    for i, q in enumerate(event_qualifiers):
        if qualifier_name in q["displayNames"]:
            filter[i] = True

    return filter
