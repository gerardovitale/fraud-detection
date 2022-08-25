from flask import Flask

from main import app


def create_test_client():
    app.config.update({
        "TESTING": True,
    })
    return app.test_client()
