from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from nbody.model.body import Body


class Tree:
    """
    The tree of dyadic cubes produced by the Barnes-Hut algorithm.

    An initial dyadic cube [a_1, b_1) x [a_2, b_2) (all cubes are half-open) is repeatedly subdivided
    until each cube contains exactly one point. Empty nodes are culled. The center of mass and total mass in each
    cube is computed in O(N logN) time from the bottom up.

    Attributes:
        Node (class): A tree node representing a dyadic cube, with links to the points contained, center of mass,
            total mass, and children.
        root (Node): The initial dyadic cube.

        MAX_DEPTH (int): Maximum tree depth before terminating splitting. Defaults to 500.
        MIN_LENGTH (float): Minimum sielength before terminating splitting. Defaults to 1e-8.
    """

    max_depth: int = 500
    min_length: float = 1e-8

    @dataclass
    class Node:
        """
        A tree node representing a dyadic cube.

        Attributes:
            coord (np.ndarray): The bottom left corner, i.e. (a_1, a_2) if the cube is [a_1, b_1) x [a_2, b_2).
            length (float): The sidelength of the cube.
            points (list[Body]): A list of the bodies contained in the cube.
            parent (Node): The parent dyadic cube.
            center_of_mass (np.ndarray, optional): The center of mass of the points in the cube.
            mass (float, optional): The total mass of the points in the cube.
            children (list, optional): A list of the child nodes.
            depth (int): The tree depth of the Node
        """

        coord: np.ndarray
        length: float
        points: list[Body]
        parent: Tree.Node | None

        center_of_mass: np.ndarray | None = None
        mass: float | None = None
        children: list | None = None

        depth: int = 0

        def compute_masses(self) -> None:
            """
            Recursively compute center of mass and mass values from child nodes.
            """
            if not self.children:  # nodes with no children should have exactly 1 point TODO: write a test for this
                self.center_of_mass = self.points[0].position
                self.mass = self.points[0].mass
                return
            for child in self.children:  # traverse down the tree
                child.compute_masses()
            com = np.zeros(2)
            m = 0
            for child in self.children:
                com += child.mass * child.center_of_mass  # traverse back up
                m += child.mass
            self.center_of_mass = com / m
            self.mass = m

        def __str__(self) -> str:
            if len(self.points) == 1:
                s = f"Node: {self.points}"
            else:
                s = f"Node: mass {self.mass}, CoM {self.center_of_mass}, points: {len(self.points)}"
            if self.children is None:
                return s
            for child in self.children:
                # indent
                lines = str(child).split("\n")
                indented_lines = []
                for line in lines:
                    indented_lines.append("- " + line)
                s += "\n" + "\n".join(indented_lines)
            return s

    root: Node

    def split(self, node: Node) -> list[Node]:
        """
        If the passed node contains more than one point, subdivide into eight subdyadic cubes and sort the points into
            the subcubes. Cull the empty subcubes.

        Arguments:
            node (Node): The node to subdivide.

        Returns:
            list[Node]: All child nodes of the passed node with at least one child.
        """
        if len(node.points) <= 1:
            return []
        if node.depth > self.max_depth:
            return []
        if node.length < self.min_length:
            return []
        x = node.coord[0]
        y = node.coord[1]
        xmid = node.coord[0] + node.length / 2
        ymid = node.coord[1] + node.length / 2

        nodell = Tree.Node(
            coord=np.array([x, y]),
            length=node.length / 2,
            points=[],
            parent=node,
            depth=node.depth + 1,
        )
        nodelr = Tree.Node(
            coord=np.array([x, ymid]),
            length=node.length / 2,
            points=[],
            parent=node,
            depth=node.depth + 1,
        )
        noderl = Tree.Node(
            coord=np.array([xmid, y]),
            length=node.length / 2,
            points=[],
            parent=node,
            depth=node.depth + 1,
        )
        noderr = Tree.Node(
            coord=np.array([xmid, ymid]),
            length=node.length / 2,
            points=[],
            parent=node,
            depth=node.depth + 1,
        )

        for point in node.points:  # use the convention of half open intervals [a,b)
            p = point.position
            if p[0] < xmid:
                n = nodell if p[1] < ymid else nodelr
            else:
                n = noderl if p[1] < ymid else noderr
            n.points.append(point)

        node.children = [
            nodell,
            nodelr,
            noderl,
            noderr,
        ]  # flatten list
        node.children = [
            n for n in node.children if len(n.points) > 0
        ]  # cull empty nodes
        return node.children

    def recursive_split(self, root: Node) -> None:
        """
        Recursively split the root node until the tree is completely generated.

        Arguments:
            root (Node): The dyadic cube containing all points in the system, serving as the base of the tree.
        """
        stack = [root]
        while stack:
            node = stack.pop()
            stack += self.split(node)

    def __init__(
        self,
        points: list[Body],
        max_depth: int | None = None,
        min_length: float | None = None,
    ):
        """
        Arguments:
            points (list[Body]): The bodies in the system to be subdivided.

            max_depth (int, optional): The maximum depth before terminating node splitting. May be infinity.
            min_length (float, optional): The minimum cube sidelength before terminating node splitting. May be zero.
        """
        if max_depth is not None:
            self.max_depth = max_depth
        if min_length is not None:
            self.min_length = min_length

        xvals = [p.position[0] for p in points]
        yvals = [p.position[1] for p in points]

        xmin = min(xvals)
        xmax = max(xvals)
        ymin = min(yvals)
        ymax = max(yvals)

        base_length = 2 * max(
            xmax - xmin, ymax - ymin
        )  # double length to avoid issues with the half interval convention
        self.root = self.Node(
            coord=np.array([xmin, ymin]),
            length=base_length,
            points=points,
            parent=None,
        )
        self.recursive_split(self.root)
        self.root.compute_masses()

    def compute_accel(
        self, other: Body, theta: float = 1, G: float = 6.6743e-11
    ) -> np.ndarray:
        """
        Compute the acceleration felt by a given Body object from the system using the Barnes-Hut algorithm.

        Arguments:
            other (Body): The Body experiencing acceleration. This will usually be a body in the system but computation
                is supported for bodies outside the system.
            theta (float, optional): An accuracy parameter used by the Barnes-Hut grouping algorithm. Defaults to 1.
            G (float, optional): The gravitational constant. Defaults to 6.6743e-11.

        Returns:
            np.ndarray: The acceleration vector.

        Examples:
            >>> moon = Body("Moon", 1, 1, np.zeros(2), np.zeros(2))
            >>> earth = Body("Earth", 1e6, 100, np.array([1.0, 0]), np.zeros(2))
            >>> tree = Tree([moon, earth])
            >>> tree.compute_accel(moon)
            [6.676973582469296e-05, 0.0]
            >>> tree2 = Tree([earth])
            >>> tree2.compute_accel(moon)
            [6.676973582469296e-05, 0.0]
        """
        accel = np.zeros(2)

        stack = [self.root]

        while stack:
            node = stack.pop()
            dist = np.linalg.norm(node.center_of_mass - other.position)
            # accept any cube with l/D < theta
            length = node.length
            if np.isclose(
                dist, 0
            ):  # if a cube happens to have center of mass at the point, examine children, unless
                # it is a point cube
                if node.children:
                    stack.extend(node.children)
                else:
                    continue
            if length / dist < theta:
                dir = (node.center_of_mass - other.position) / dist
                accel += dir * G * node.mass / (dist**2)
            else:
                if node.children:
                    stack.extend(node.children)
                else:  # point cubes should always be accepted
                    dir = (node.center_of_mass - other.position) / dist
                    accel += dir * G * node.mass / (dist**2)
        return accel

    def __str__(self):
        return f"Tree\n{self.root!s}"


class BarnesHutPropagator:
    """
    Simulate an orbital system using the Barnes-Hut N logN algorithm for force computation.
    """

    bodies: list[Body]
    theta: float
    G: float
    dt: float
    t_total: float
    num_steps: int
    output_name: str
    n_bodies: int
    state_vector_size: int
    states: np.ndarray

    def __init__(
        self,
        bodies: list[Body],
        params: dict | None,
        output_name: str = "OUTPUT_FILES/traces.csv",
    ):
        """
        Arguments:
            bodies (list[Body]): The bodies in the system to be simulated.
            params (dict): Simulation parameters. Supported keys are "dt", "t_max", "theta", and "G"
                (defaults to 1, 100, 1, and 6.6743e-11, resp.)
            output_file (str): The filename that traces are written to. Defaults to `OUTPUT_FILES/traces.csv`.
        """
        if params is None:
            params = {}
        self.bodies = bodies
        self.G = params.get("G", 6.6743e-11)
        self.dt = params.get("dt", 1.0)
        self.theta = params.get("theta", 1)
        self.t_total = params.get("t_total", 1000.0)
        self.num_steps = int(self.t_total / self.dt)
        self.output_name = output_name

        self.n_bodies = len(self.bodies)
        self.state_vector_size = 6  # x, y, vx, vy, ax, ay

        # 3D array to store states: time x body x state_vector
        self.states = np.zeros((self.num_steps, self.n_bodies, self.state_vector_size))

    def propagate(self, timeout_ns: int | None = None):
        """
        Propagate the system forward in time using the Barnes-Hut algorithm.

        Returns
        -------
        states : np.ndarray, shape (num_steps, n_bodies, 6)
            Simulated state history. For each time step and each body:
            [x, y, vx, vy, ax, ay].

        Notes
        -----
        This modifies internal state arrays in-place.
        """
        start_time = time.perf_counter_ns()
        for step in range(self.num_steps):
            # generate the dyadic tree
            tree = Tree(self.bodies)
            accels = np.zeros([len(self.bodies), 2])
            for j, body in enumerate(self.bodies):
                accels[j] = tree.compute_accel(body, self.theta, self.G)
                if (
                    timeout_ns is not None
                    and time.perf_counter_ns() - start_time > timeout_ns
                ):
                    msg = f"Timedout after {timeout_ns} ns"
                    raise TimeoutError(msg)
            for j, body in enumerate(self.bodies):
                # need a second loop because the tree reference current positions/velocities
                self.states[step, j, 0:2] = body.position
                self.states[step, j, 2:4] = body.velocity
                self.states[step, j, 4:6] = accels[j]
                # update position/velocity
                body.position += self.dt * body.velocity
                body.velocity += self.dt * accels[j]
        return self.states

    def write_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_name}_{timestamp}.csv"
        path = Path(filename)
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            # The header line
            param_row = [f"G={self.G}", f"dt={self.dt}", f"t_total={self.t_total}"]
            header = ["Iteration", "Time"] + [
                f"Body {i} {prop} ({axis})"
                for i in range(1, len(self.bodies) + 1)
                for prop in ("position", "velocity", "acceleration")
                for axis in ("x", "y")
            ]
            writer.writerow(param_row)
            writer.writerow(header)

            for step in range(self.num_steps):
                row = [
                    step,
                    step * self.dt,
                ]
                for b in range(self.n_bodies):
                    row.extend(self.states[step, b, :])
                writer.writerow(row)
        return filename
