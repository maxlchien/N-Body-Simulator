# General Description:
The **nbody** package is a Python-based framework that handles the forward propagation of bodies that interact gravitationally given their initial position and velocity measurements. Given the initial configuration, the simulation enables one of two propagators, Euler or Barnes-Hut to calculate the positions and velocities of the bodies at every time step (user-defined) in a given time frame. The outputs are stored in a .csv file, and the time evolution can be visualized as a .mp4 video.



## Installation:

Copy the repository to your local environment via
```
git clone https://github.com/maxlchien/N-Body-Simulator/
cd N-Body-Simulator
```
To install with pip:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt #add -e flag for editability
```
With uv:
```
uv venv
uv pip install nbody
```

## File Layout
* `nbody/`
  * `model/`
    * `body.py`
  * `engine/`
    * `euler_propagator.py`
    * `barnes_hut.py`
  * `utility/`:  Functions for reading in the files
    * `preprocess.py`
    * `generate_bodies.py`
    * `plotter.py`
    * `CSV_processor.py`
  * `main.py`: Stores main workflow for CLI
  * `example/`: Examples
    * `hohmann.py`
    * `example_hohmann.yaml`
  * `config/`: Config files for internal use (tests, etc)
* `config/`: Store config files
  * `simulation.yaml`
  * `test*.yaml`
* `results/`: Stores simulation output
  * `*.csv`


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

## Example: Hohmann Transfer
Beyond the testing suite, we provide a more advanced scenario in which the N-Body simulation can be used, where running the simulation might answer a question for which the analytical case would pose considerable difficulties. The example we provide is: "will a Hohmann transfer orbit still achieve its target in the presence of additional gravitational influences?"

In order to test this example, simply run:
```
python src/nbody/example/hohmann.py
```
or
```
uv run src/nbody/example/hohmann.py
```
Output:
```
Earth → Mars Hohmann transfer result
----------------------------------
Minimum distance to Mars: 0.00631 AU
Occurred at step: 2810
Arrival threshold: 0.01 AU
STATUS: ARRIVED (first entry at step 2764)
----------------------------------
```
