from dash import dcc, html
from dash.exceptions import PreventUpdate
from flask import request
from .config import dashboard_settings


def check_auth(username, password):
    """Validate username/password"""
    return (
        username == dashboard_settings.ADMIN_USER
        and password == dashboard_settings.ADMIN_PASSWORD
    )


def login_layout():
    """Login form layout"""
    return html.Div(
        [
            html.H2("Login Required"),
            dcc.Input(id="username", type="text", placeholder="Username"),
            dcc.Input(id="password", type="password", placeholder="Password"),
            html.Button("Login", id="login-button"),
            html.Div(id="login-status"),
        ]
    )


def require_auth(func):
    """Decorator to enforce authentication"""

    def wrapper(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            raise PreventUpdate("Unauthorized")
        return func(*args, **kwargs)

    return wrapper
