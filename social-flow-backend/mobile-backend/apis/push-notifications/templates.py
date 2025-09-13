# Notification templates (text, HTML, rich content)
"""
Notification templates and simple templating helpers.
In production you might store templates in DB or a template service.
"""

from typing import Dict, Any
import jinja2

# Example in-memory templates
TEMPLATES = {
    "welcome": {
        "title": "Welcome, {{ name }}!",
        "body": "Thanks for joining {{ app_name }}. Tap to complete your profile."
    },
    "message_received": {
        "title": "{{ sender_name }} sent a message",
        "body": "{{ snippet }} - open the app to reply."
    }
}

_template_env = jinja2.Environment(autoescape=True)


def render_template(template_name: str, context: Dict[str, Any]) -> Dict[str, str]:
    tmpl = TEMPLATES.get(template_name)
    if not tmpl:
        raise ValueError("Template not found")
    title = _template_env.from_string(tmpl["title"]).render(**context)
    body = _template_env.from_string(tmpl["body"]).render(**context)
    return {"title": title, "body": body}
