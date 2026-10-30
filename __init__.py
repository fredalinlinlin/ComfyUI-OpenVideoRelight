"""OpenRelight: keyframed, physically guided video relighting for ComfyUI."""

from .nodes import OpenRelightExtension

# the light stage editor on the Light Rig node (web/js/lightstage.js)
WEB_DIRECTORY = "./web"

try:  # presets and look names for the editor; ComfyUI's server is absent when imported by the tests
    from aiohttp import web
    from server import PromptServer

    from .openrelight import rig as _rig
    from .openrelight.prompt import FILMS, GRADES, LOOKS

    @PromptServer.instance.routes.get("/openrelight/presets")
    async def _openrelight_presets(request):
        return web.json_response({
            "presets": {name: _rig.load_rig(None, preset=name) for name in _rig.preset_names()},
            "labels": _rig.PRESET_LABELS, "looks": LOOKS, "films": FILMS, "grades": GRADES,
        })
except Exception:  # noqa: BLE001
    pass


async def comfy_entrypoint() -> OpenRelightExtension:
    return OpenRelightExtension()
