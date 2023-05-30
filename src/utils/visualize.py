import sys
from typing import cast, Any, Union, TypeVar, Generic, NamedTuple, Optional, Tuple, List, Dict
import numpy as np
from numpy.typing import NDArray
from igraph import Graph
from plotly.graph_objects import Scatter3d, Scatter, Layout, Figure
from plotly.graph_objs.layout import Slider

Coordinate2D = Tuple[float, float]
Coordinate3D = Tuple[float, float, float]

TCoordinate = TypeVar("TCoordinate", Coordinate2D, Coordinate3D)
class NodeMeta(Generic[TCoordinate]):
    def __init__(self, name: str, coordinate: Optional[TCoordinate] = None, group: Optional[int] = None):
        self.name = name
        self.coordinate = coordinate
        self.group = group

class PlotStyle(NamedTuple):
    node: Dict[str, Any] = dict()
    edge: Dict[str, Any] = dict()
    axis: Dict[str, Any] = dict()
    font: Dict[str, Any] = dict()
    layout: Dict[str, Any] = dict()

def visualize_matrix(
    connectivity_matrix: NDArray,
    node_metas: List[NodeMeta[TCoordinate]],
    connectivity_threshold: float = sys.float_info.epsilon,
    slider_step: Union[int, List[float]] = 20,
    style: PlotStyle = PlotStyle()) -> Figure:
    """
    Visualize connectivity matrix with node metas
    """
    # Validate input parameters
    assert len(connectivity_matrix.shape) == 2 and connectivity_matrix.shape[0] == connectivity_matrix.shape[1], "Connectivity matrix must be a square matrix"
    length = connectivity_matrix.shape[0]
    assert length == len(node_metas), "Connectivity matrix and node metas must have the same length"
    for meta in node_metas[1:]:
        if node_metas[0].coordinate is None:
            assert meta.coordinate is None, "All node metas must have coordinate or none of them have coordinate"
        else:
            assert meta.coordinate is not None, "All node metas must have coordinate or none of them have coordinate"
            assert len(node_metas[0].coordinate) == len(meta.coordinate), "All node metas must use coordinate with the same dimension"
    generated_layout = node_metas[0].coordinate is None
    dimension = len(node_metas[0].coordinate) if node_metas[0].coordinate is not None else 3
    
    # Initialize node and edge coordinates
    edges: list[tuple[int, int, float]] = []
    for i in range(1, length):
        for j in range(i):
            weight = connectivity_matrix[i, j]
            if weight == 0:
                weight = connectivity_matrix[j, i]
            if weight > connectivity_threshold:
                edges += [(i, j, weight)]
    nodes: list[tuple[str, TCoordinate, Optional[int]]]
    if not generated_layout:
        nodes = [(node_meta.name, node_meta.coordinate, node_meta.group) for node_meta in node_metas] #type: ignore
    else:
        graph = Graph(directed=False)
        for node_meta in node_metas:
            graph.add_vertex(node_meta.name)
        for [i, j, _] in edges:
            graph.add_edge(i, j)
        graph_layout = graph.layout("circle", dim=3)
        nodes = []
        for i in range(length):
            nodes += [(node_metas[i].name, graph_layout[i], node_metas[i].group)] #type: ignore
    
    # Set default style
    def set_if_none(target: Dict[str, Any], key: str, value: Any):
        if key not in target or target[key] is None:
            target[key] = value
    set_if_none(style.node, "symbol", "circle")
    set_if_none(style.node, "size", 8)
    set_if_none(style.node, "colorscale", "Viridis")
    
    set_if_none(style.edge, "width", 4)
    set_if_none(style.edge, "color", "#888")
    
    set_if_none(style.axis, "visible", False)
    
    set_if_none(style.layout, "width", 512)
    set_if_none(style.layout, "height", 512)
    set_if_none(style.layout, "showlegend", False)
    set_if_none(style.layout, "hovermode", "closest")
    set_if_none(style.layout, "margin", dict(t=32, b=32, l=32, r=32))
    
    # Create figure traces
    def parse_color(color: str) -> Tuple[int, int, int]:
        if color.startswith("#"):
            color = color[1:]
            if len(color) == 3:
                color = "".join([c * 2 for c in color])
            return tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        elif color.startswith("rgb("):
            color = color[4:-1]
            return tuple(int(c) for c in color.split(","))
        else:
            raise ValueError("Color must be in hex or rgb format")
    
    base_color = parse_color(style.edge["color"])
    def get_edge_attributes(edge: Tuple[int, int, float]):
        [i, j, w] = edge
        attributes= dict(
            mode="lines",
            line={**style.edge, **dict(
                width=w * style.edge["width"],
                color=f"rgba({base_color[0]}, {base_color[1]}, {base_color[2]}, {(0 if w == 0 else (0.5 + w / 2)):.2f})"
            )},
            x=[nodes[i][1][0], nodes[j][1][0], None],
            y=[nodes[i][1][1], nodes[j][1][1], None],
            text=[f"{(w * 100):.2f}%"],
            textfont=style.font,
            hoverinfo="text"
        )
        if dimension == 3:
            attributes["z"] = [nodes[i][1][2], nodes[j][1][2], None]
        return attributes
        
    plotType = Scatter3d if dimension == 3 else Scatter
    marker_style = style.node.copy()
    if all([node[2] is not None for node in nodes]):
        marker_style["color"] = [node[2] for node in nodes]
    marker_mode = "markers"
    if marker_style["always_show_text"] == True:
        marker_mode += "+text"
        del marker_style["always_show_text"]
    node_trace = plotType(
        mode=marker_mode,
        marker=marker_style,
        x=[node[1][0] for node in nodes],
        y=[node[1][1] for node in nodes],
        z=[node[1][2] for node in nodes] if dimension == 3 else None,
        text=[node[0] for node in nodes],
        textfont=style.font,
        hoverinfo="text" if marker_mode == "markers" else "none"
    )
    edge_traces = [plotType(**get_edge_attributes(edge)) for edge in edges]
        
    # Create figure layout
    layout_scene = dict(xaxis=dict(style.axis), yaxis=dict(style.axis))
    if dimension == 3:
        layout_scene["zaxis"] = dict(style.axis)
    layout = Layout(**style.layout, scene=layout_scene)
    figure = Figure(data=edge_traces + [node_trace], layout=layout)
    
    # Create slider
    matrix_mask = np.full(connectivity_matrix.shape, True, dtype=bool)
    np.fill_diagonal(matrix_mask, False)
    min = connectivity_matrix.min(initial=sys.float_info.max, where=matrix_mask)
    max = connectivity_matrix.max(initial=sys.float_info.min, where=matrix_mask)
    slider_values = slider_step if type(slider_step) == list else np.linspace(min, max, cast(int, slider_step) + 1).tolist()
    steps = []
    for value in slider_values:
        rescaled = [0 if w < value else (1 if max == value else (w - value) / (max - value)) for [_, _, w] in edges]
        step = dict(
            method="restyle",
            args=[{
                "line.width": [w * style.edge["width"] for w in rescaled] + [None],
                "line.color": [f"rgba({base_color[0]}, {base_color[1]}, {base_color[2]}, {(0 if w == 0 else (0.5 + w / 2)):.2f})" for w in rescaled] + [None],
                "hoverinfo": ["none" if w < value else "text" for [_, _, w] in edges] + ["text"]
            }],
            label=f"{(value * 100):.1f}%"
        )
        steps.append(step)
    slider = Slider(
        active=0,
        currentvalue={"prefix": "Threshold: "},
        pad={"t": 32},
        steps=steps,
        font=style.font
    )
    figure.update_layout(sliders=[slider])
    return figure