from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

# Now import URDF stuff
from isaacsim.asset.importer.urdf import URDFImporter
print("URDFImporter methods:", [m for m in dir(URDFImporter) if not m.startswith('_')])

# Also check the commands
import omni.kit.commands
cmds = omni.kit.commands.get_all_commands()
urdf_cmds = [c for c in cmds if 'URDF' in c or 'urdf' in c]
print("URDF related commands:", urdf_cmds)

simulation_app.close()
