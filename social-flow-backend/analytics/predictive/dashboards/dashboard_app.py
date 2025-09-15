import dash
from dash import html
from flask import Flask
from .config import dashboard_settings
from .auth import require_auth
from .data_service import DataService
from .visualizations import user_growth_chart, video_engagement_chart
from .monitoring import DashboardMonitor
from .utils import logger


server = Flask(__name__)
app = dash.Dash(__name__, server=server, suppress_callback_exceptions=True, title=dashboard_settings.DASHBOARD_TITLE)

data_service = DataService()
monitor = DashboardMonitor()


@app.callback(
    dash.dependencies.Output("dashboard-content", "children"),
    [dash.dependencies.Input("interval-component", "n_intervals")],
)
@require_auth
def update_dashboard(_):
    """Refresh dashboard content periodically"""
    user_growth_df = data_service.fetch_user_growth_forecast()
    engagement_df = data_service.fetch_video_engagement_forecast()
    monitor.push_metrics(active_user_count=1)  # simplistic: count = 1 active session
    return [
        html.H2("Predictive Analytics Dashboard"),
        user_growth_chart(user_growth_df),
        video_engagement_chart(engagement_df),
    ]


app.layout = html.Div(
    [
        html.Div(id="dashboard-content"),
        dash.dcc.Interval(id="interval-component", interval=60 * 1000, n_intervals=0),
    ]
)


def run():
    logger.info(f"Starting dashboard at {dashboard_settings.DASHBOARD_HOST}:{dashboard_settings.DASHBOARD_PORT}")
    app.run_server(host=dashboard_settings.DASHBOARD_HOST, port=dashboard_settings.DASHBOARD_PORT)


if __name__ == "__main__":
    run()
