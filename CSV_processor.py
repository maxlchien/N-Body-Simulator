
import re
import pandas as pd
import numpy as np

def csv_processor(csv_file):
    """
    Process a CSV file containing N-body simulation data.

    The CSV file should have:
    - First row: constants (G, dt, t_total) in the format key=value
    - Second row: header
    - Remaining rows: numerical data with positions and velocities

    Parameters
    ----------
    csv_file : str
        The CSV results file (filename).

    Returns
    -------
    G : float
        Gravitational constant from the CSV.
    dt : float
        Time step from the CSV.
    num_bodies : int
        Number of bodies in the simulation.
    df : pandas.DataFrame
        DataFrame containing iteration, time, and positions for each body.
    """

    FIRST_BODY = 1
    FIRST_BODY_PX = 2
    FIRST_BODY_PY = 3
    COLS_PER_BODY = 6

    with open(csv_file, "r") as f:
        lines = f.readlines()

    # Reading in position values for each body
    values = []
    for line in lines[2:]:
        values.append(list(map(float,line.strip().split(","))))

    # Reading in constants and simulation paramaters
    const_vals = lines[0].strip().split(",")
    G = float(const_vals[0].split('=')[-1])
    dt = float(const_vals[1].split('=')[-1])
    t_total = float(const_vals[2].split('=')[-1])

    # Calculating number of bodies in the simulation
    num_cols = len(values[0])
    num_bodies = int ((num_cols - 2) / COLS_PER_BODY)
    
    values = np.array(values)
    
    # Creating the DataFrame for plotting
    df = pd.DataFrame()
    df['iteration'] = values[:,0]
    df['time'] = values[:,1]

    x = FIRST_BODY_PX
    y = FIRST_BODY_PY
    body = FIRST_BODY
    while (y < num_cols):
        df[f'x{body}'] = values[:,x]
        df[f'y{body}'] = values[:,y]
        x = x + COLS_PER_BODY
        y = y + COLS_PER_BODY
        body = body + 1

    return G, dt, num_bodies, df