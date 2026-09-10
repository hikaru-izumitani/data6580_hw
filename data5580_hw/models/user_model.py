from dataclasses import dataclass


@dataclass
class User(object):
    id: str
    name: str
    email: str
