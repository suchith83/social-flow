"""Public façade for AI/ML service access.

Thin wrapper kept separate from ``__init__`` so that importing the package
does not *require* immediate ML graph construction; advanced heavy libraries
remain lazily instantiated by ``MLService`` itself.

Usage:
    from app.ai_ml_services.facade import get_ai_ml_service
    ml = get_ai_ml_service()
    recs = await ml.generate_recommendations(user_id)
"""

from app.ai_ml_services import get_ai_ml_service, AIServiceFacade  # re-export

__all__ = ["get_ai_ml_service", "AIServiceFacade"]
