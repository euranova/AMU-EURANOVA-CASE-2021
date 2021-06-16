from setuptools import setup, find_packages

exec(open('event_extraction/version.py').read())

setup(
    name = "event_extraction",
    version = __version__,
    packages = find_packages(),
)
