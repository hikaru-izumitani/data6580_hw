import logging
import os


class Config:

    # In-file sqlite database
    SQLALCHEMY_DATABASE_URI = 'sqlite:///data.db'

    # In memory database
    # SQLALCHEMY_DATABASE_URI = ':memory:///data.db'
    SQLALCHEMY_ECHO = True
    SQLALCHEMY_TRACK_MODIFICATIONS = False