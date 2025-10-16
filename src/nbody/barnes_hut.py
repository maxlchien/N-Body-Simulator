from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class Tree:
    @dataclass
    class Node:
        coord: np.ndarray
        length: float
        points: list[np.ndarray]
        parent: Node

        center_of_mass: np.ndarray = None
        mass: float = None
        children: list = None

        def compute_masses(self):
            """
            Update center of mass and mass values from child nodes.
            """
            if not self.children:  # nodes with no children should have exactly 1 point TODO: write a test for this
                self.center_of_mass = self.points[0]
                self.mass = 1  ##TODO: update this to use the Body mass parameter
                return
            for child in self.children:  # traverse down the tree
                child.compute_masses()
            com = np.zeros(3)
            m = 0
            for child in self.children:
                com += child.mass * child.center_of_mass  # traverse back up
                m += child.mass
            self.center_of_mass = com
            self.mass = m

    root: Node

    def split(node: Node) -> list[Node]:
        """ """
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

        for p in node.points:  # use the convention of half open intervals [a,b)
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
            n.points.append(p)

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
        stack = [root]
        while stack:
            node = stack.pop()
            stack += Tree.split(node)

    def __init__(self, points: list[np.ndarray]):
        xvals = [p[0] for p in points]
        yvals = [p[1] for p in points]
        zvals = [p[2] for p in points]

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
