from control_plane.api import create_app
from control_plane.settings import Settings


app = create_app(Settings.from_env())
