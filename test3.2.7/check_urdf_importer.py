from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.kit.commands
import sys

cmds = omni.kit.commands.get_all_commands()
urdf_cmds = [c for c in cmds if 'URDF' in c or 'urdf' in c]
print("=== URDF related commands ===")
for c in urdf_cmds:
    print(f"  {c}")

try:
    from isaacsim.asset.importer.urdf import URDFImporter
    print("=== URDFImporter methods ===")
    for attr in dir(URDFImporter):
        if not attr.startswith('__'):
            print(f"  {attr}")
except Exception as e:
    print(f"URDFImporter import error: {e}")

try:
    import omni.importer.urdf
    print("=== omni.importer.urdf contents ===")
    print(dir(omni.importer.urdf))
except Exception as e:
    print(f"omni.importer.urdf error: {e}")

simulation_app.close()
