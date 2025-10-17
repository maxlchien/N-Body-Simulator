from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np
import Path
from body import Body


class Tree:
    """
    The tree of dyadic cubes produced by the Barnes-Hut algorithm.

    An initial dyadic cube [a_1, b_1) x [a_2, b_2) x [a_3, b_3) (all cubes are half-open) is repeatedly subdivided
    until each cube contains exactly one point. Empty nodes are culled. The center of mass and total mass in each
    cube is computed in O(N logN) time from the bottom up.

    Attributes:
        Node (class): A tree node representing a dyadic cube, with links to the points contained, center of mass,
            total mass, and children.
        root (Node): The initial dyadic cube.
    """

    @dataclass
    class Node:
        """
        A tree node representing a dyadic cube.

        Attributes:
            coord (np.ndarray): The bottom left corner, i.e. (a_1, a_2, a_3) if the cube is [a_1, b_1) x [a_2, b_2) x [a_3, b_3).
            length (float): The sidelength of the cube.
            points (list[Body]): A list of the bodies contained in the cube.
            parent (Node): The parent dyadic cube.
            center_of_mass (np.ndarray, optional): The center of mass of the points in the cube.
            mass (float, optional): The total mass of the points in the cube.
            children (list, optional): A list of the child nodes.
        """

        coord: np.ndarray
        length: float
        points: list[Body]
        parent: Node

        center_of_mass: np.ndarray = None
        mass: float = None
        children: list = None

        def compute_masses(self):
            """
            Recursively compute center of mass and mass values from child nodes.
            """
            if not self.children:  # nodes with no children should have exactly 1 point TODO: write a test for this
                self.center_of_mass = self.points[0].position
                self.mass = self.points[0].mass
                return
            for child in self.children:  # traverse down the tree
                child.compute_masses()
            com = np.zeros(3)
            m = 0
            for child in self.children:
                com += child.mass * child.center_of_mass  # traverse back up
                m += child.mass
            self.center_of_mass = com / m
            self.mass = m

        def __str__(self):
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

    def split(node: Node) -> list[Node]:
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

        x = node.coord[0]
        y = node.coord[1]
        z = node.coord[2]
        xmid = node.coord[0] + node.length / 2
        ymid = node.coord[1] + node.length / 2
        zmid = node.coord[2] + node.length / 2

        nodelll = Tree.Node(
            coord=np.array([x, y, z]), length=node.length / 2, points=[], parent=node
        )
        nodellr = Tree.Node(
            coord=np.array([x, y, zmid]), length=node.length / 2, points=[], parent=node
        )
        nodelrl = Tree.Node(
            coord=np.array([x, ymid, z]), length=node.length / 2, points=[], parent=node
        )
        nodelrr = Tree.Node(
            coord=np.array([x, ymid, zmid]),
            length=node.length / 2,
            points=[],
            parent=node,
        )
        noderll = Tree.Node(
            coord=np.array([xmid, y, z]), length=node.length / 2, points=[], parent=node
        )
        noderlr = Tree.Node(
            coord=np.array([xmid, y, zmid]),
            length=node.length / 2,
            points=[],
            parent=node,
        )
        noderrl = Tree.Node(
            coord=np.array([xmid, ymid, z]),
            length=node.length / 2,
            points=[],
            parent=node,
        )
        noderrr = Tree.Node(
            coord=np.array([xmid, ymid, zmid]),
            length=node.length / 2,
            points=[],
            parent=node,
        )

        for point in node.points:  # use the convention of half open intervals [a,b)
            p = point.position
            if p[0] < xmid:
                if p[1] < ymid:
                    n = nodelll if p[2] < zmid else nodellr
                else:
                    n = nodelrl if p[2] < zmid else nodelrr
            else:
                if p[1] < ymid:
                    n = noderll if p[2] < zmid else noderlr
                else:
                    n = noderrl if p[2] < zmid else noderrr
            n.points.append(point)

        node.children = [
            nodelll,
            nodellr,
            nodelrl,
            nodelrr,
            noderll,
            noderlr,
            noderrl,
            noderrr,
        ]  # flatten list
        node.children = [
            n for n in node.children if len(n.points) > 0
        ]  # cull empty nodes
        return node.children

    def recursive_split(root: Node):
        """
        Recursively split the root node until the tree is completely generated.

        Arguments:
            root (Node): The dyadic cube containing all points in the system, serving as the base of the tree.
        """
        stack = [root]
        while stack:
            node = stack.pop()
            stack += Tree.split(node)

    def __init__(self, points: list[Body]):
        """
        Arguments:
            points (list[Body]): The bodies in the system to be subdivided.
        """
        xvals = [p.position[0] for p in points]
        yvals = [p.position[1] for p in points]
        zvals = [p.position[2] for p in points]

        xmin = min(xvals)
        xmax = max(xvals)
        ymin = min(yvals)
        ymax = max(yvals)
        zmin = min(zvals)
        zmax = max(zvals)

        base_length = 2 * max(
            xmax - xmin, ymax - ymin, zmax - zmin
        )  # double length to avoid issues with the half interval convention
        self.root = Tree.Node(
            coord=np.array([xmin, ymin, zmin]),
            length=base_length,
            points=points,
            parent=None,
        )
        Tree.recursive_split(self.root)
        self.root.compute_masses()

    def compute_accel(self, other: Body, theta: float = 1, G: float = 6.6743e-11):
        """
        Compute the acceleration felt by a given Body object from the system using the Barnes-Hut algorithm.

        Arguments:
            other (Body): The Body experiencing acceleration.
            theta (float, optional): An accuracy parameter used by the Barnes-Hut grouping algorithm. Defaults to 1.
            G (float, optional): The gravitational constant. Defaults to 6.6743e-11.

        Returns:
            np.ndarray: The acceleration vector.
        """
        accel = np.zeros(3)

        stack = [self.root]

        while stack:
            node = stack.pop()
            dist = np.linalg.norm(node.center_of_mass - other.position)
            # accept any cube with l/D < theta
            length = node.length
            if dist == 0:
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

    def __init__(
        self,
        bodies: list[Body],
        params: dict | None,
        output_file: str = "OUTPUT_FILES/traces.csv",
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
        dt = params.get("dt", 1.0)
        t_max = params.get("t_max", 100)
        theta = params.get("theta", 1)
        G = params.get("G", 6.6743e-11)

        path = Path(output_file)
        path.parent.mkdir(exist_ok=True, parents=True)
        with Path.open(path, "w") as f:
            writer = csv.writer(f)
            # The header line
            header = ["Iteration", "t"]
            for i in range(1, len(bodies) + 1):
                header.extend(
                    [
                        f"Body {i} position (x)",
                        f"Body {i} position (y)",
                        f"Body {i} position (z)",
                        f"Body {i} velocity (x)",
                        f"Body {i} velocity (y)",
                        f"Body {i} velocity (z)",
                        f"Body {i} acceleration (x)",
                        f"Body {i} acceleration (y)",
                        f"Body {i} acceleration (z)",
                    ]
                )
            writer.writerow(header)

            for i in range(t_max):
                # generate the dyadic tree
                tree = Tree(bodies)

                trace = [i, dt * i]
                accels = np.zeros([len(bodies), 3])
                for j, body in enumerate(bodies):
                    # record previous position/velocity
                    trace.extend(body.position)
                    trace.extend(body.velocity)
                    accels[j] = tree.compute_accel(body, theta, G)
                    trace.extend(accels[j])
                for j, body in enumerate(bodies):
                    # update position/velocity
                    body.position += dt * body.velocity
                    body.velocity += dt * accels[j]
                writer.writerow(trace)


# sample code to demonstrate utility
if __name__ == "__main__":
    bodies = [
        Body(1, np.zeros(3), np.zeros(3)),
        Body(1000, np.array([1.0, 0, 0]), np.zeros(3)),
    ]
    BarnesHutPropagator(bodies, {})
