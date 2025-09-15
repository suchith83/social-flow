import plotly.graph_objs as go
from dash import dcc


def user_growth_chart(df):
    """Line chart for user growth forecast"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["forecast_date"], y=df["predicted_users"], mode="lines", name="Predicted Users"))
    fig.add_trace(go.Scatter(x=df["forecast_date"], y=df["lower_ci"], mode="lines", name="Lower CI", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=df["forecast_date"], y=df["upper_ci"], mode="lines", name="Upper CI", line=dict(dash="dot")))
    fig.update_layout(title="User Growth Forecast", xaxis_title="Date", yaxis_title="Users")
    return dcc.Graph(figure=fig)


def video_engagement_chart(df):
    """Multi-line chart for video engagement predictions"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["forecast_date"], y=df["predicted_views"], mode="lines", name="Predicted Views"))
    fig.add_trace(go.Scatter(x=df["forecast_date"], y=df["predicted_likes"], mode="lines", name="Predicted Likes"))
    fig.add_trace(go.Scatter(x=df["forecast_date"], y=df["predicted_comments"], mode="lines", name="Predicted Comments"))
    fig.update_layout(title="Video Engagement Forecast", xaxis_title="Date", yaxis_title="Counts")
    return dcc.Graph(figure=fig)
