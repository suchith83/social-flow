import dash
from dash import dcc, html
import plotly.graph_objs as go
from .config import settings


def create_dashboard(data_source):
    """
    Creates a real-time dashboard using Dash + Plotly.
    `data_source` should be a callable returning latest metrics.
    """
    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H1("📊 Real-Time Analytics Dashboard"),
        dcc.Interval(id="interval", interval=settings.refresh_interval * 1000, n_intervals=0),
        dcc.Graph(id="live-graph"),
    ])

    @app.callback(
        dash.Output("live-graph", "figure"),
        [dash.Input("interval", "n_intervals")]
    )
    def update_graph(_):
        metrics = data_source()
        return go.Figure(
            data=[go.Scatter(y=metrics.get("series", []), mode="lines+markers")],
            layout=go.Layout(title="Real-Time Metrics", xaxis=dict(title="Time"), yaxis=dict(title="Value"))
        )

    return app
