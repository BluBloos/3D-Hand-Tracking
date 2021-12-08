import pybullet as pbullet
import time
import pybullet_data as pdata

# Using pbullet.GUI for graphical version
# PyBullet is client-server API. There is a physics server, we send commands.
physicsClient = pbullet.connect(pbullet.GUI)

# Allow pbullet to use standard data shipped with pybullet.
pbullet.setAdditionalSearchPath(pdata.getDataPath())

# Load in ground plane.
planeId = pbullet.loadURDF("plane.urdf")

# Make a box that will fall.
startPos = [0,0,1]
startOrientation = pbullet.getQuaternionFromEuler([0,0,0])
boxId = pbullet.loadURDF("r2d2.urdf",startPos, startOrientation)

pbullet.setGravity(0,0,-9.81)

frame_begin = time.time()
target_frame_time = 1 / 800;
last_text_id = None

for i in range(1000):
    pbullet.stepSimulation()
    frame_end = time.time()
    delta_time = frame_end - frame_begin

    pbullet.removeAllUserDebugItems()
    if (delta_time < target_frame_time):
        delta = target_frame_time - delta_time
        print(delta_time)
        time.sleep(delta) # Sets the framerate
        last_text_id = pbullet.addUserDebugText("FPS: " + str(1 / target_frame_time), [0,0,2], textColorRGB=[1,0,0] )
    else:
        last_text_id = pbullet.addUserDebugText("FPS: " + str(1 / delta_time), [0,0,2], textColorRGB=[1,0,0] )
    frame_begin = time.time()


pbullet.disconnect()
