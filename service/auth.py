"""
API key check for the fraud service.

Only requests that carry the correct secret key in the 'X-API-Key' header are
allowed through. The expected key is read from the FRAUD_API_KEY environment
variable, which is loaded from a local .env file that is NOT committed to git.

This is the "only authorized eyes may use the system" requirement, kept
deliberately simple - a header check, not an enterprise identity system.
"""

import hmac
import os
from functools import wraps

from dotenv import load_dotenv
from flask import jsonify, request

load_dotenv()  # read key/value pairs from a local .env file into the environment

API_KEY_ENV = "FRAUD_API_KEY"
API_KEY_HEADER = "X-API-Key"


def require_api_key(view):
    """Decorator: block the request unless it carries the correct API key."""

    @wraps(view)
    def wrapped(*args, **kwargs):
        expected = os.getenv(API_KEY_ENV)
        provided = request.headers.get(API_KEY_HEADER, "")
        if not expected:
            # The server was started without a key configured - fail safe.
            return jsonify({"error": "Server has no API key configured."}), 500
        # compare_digest avoids leaking the key length/content via timing.
        if not hmac.compare_digest(provided, expected):
            return jsonify({"error": "Unauthorized: missing or invalid API key."}), 401
        return view(*args, **kwargs)

    return wrapped