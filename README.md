# General Description:
The **nbody** package is a Python-based framework that handles the forward propagation of bodies that interact gravitationally given their initial position and velocity measurements. Given the initial configuration, the simulation enables one of two propagators, Euler or Barnes-Hut to calculate the positions and velocities of the bodies at every time step (user-defined) in a given time frame. The outputs are stored in a .csv file, and the time evolution can be visualized as a .mp4 video.



## Installation:

Copy the repository to your local environment and run
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt #add -e flag for editability
pytest
```
## File Layout
nbody/
  model/
    __init__.py
    body.py
  engine/
    euler_propagator.py
    barnes_hut.py
  utility/        #functions for reading in the files
    preprocess.py
    generate_bodies.py
  config/         #store config files
    simulation.yaml
    test*.yaml
   results/       #stores simulation output
    *.csv
   __init__.py
   main.py #stores main workflow for API

tests/
  test*.py

## Base Config
Running a basic simulation with the default config can be done straight away:
```
nbody
```
For a specific config with the Euler propagator, you can run
```
nbody --input [yourpath]/[your config].yaml --output-dir ~/
```
Testing the Euler simulation with the numba augumentation:
```
nbody --numba
```
Or with the Barnes Hut propagator:
```
nbody --barnes_hut
```
Finally, to visualize, one must run:
```
nbody --visualize
```

## Example
Beyond the testing suite, we provide a more advanced scenario in which the N-Body simulation can be used, where running the simulation might answer a question for which the analytical case would pose considerable difficulties. The example we provide is: "will a Hohmann transfer orbit still achieve its target in the presence of additional gravitational influences?"

In order to test this example, simply run:
```
python scripts/hohmann.py
```
Output:
Earth → Mars Hohmann transfer result
----------------------------------
Minimum distance to Mars: 0.00631 AU
Occurred at step: 2810
Arrival threshold: 0.01 AU
STATUS: ARRIVED (first entry at step 2764)
----------------------------------
