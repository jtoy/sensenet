import time,os,math,inspect
import pybullet as p
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)
pi = 3.14159265
#first try to connect to shared memory (VR), if it fails use local GUI
c = p.connect(p.SHARED_MEMORY)
print(c)
if (c<0):
  p.connect(p.GUI)

obj_to_classify = p.loadURDF("mesh.urdf",0,0,-1)

		

p.setGravity(0,0,0)
#load the MuJoCo MJCF hand
objects = p.loadMJCF("MPL/MPL.xml",flags=0)

hand=objects[0]
#clamp in range 400-600
#minV = 400
#maxV = 600
minVarray = [275,280,350,290]
maxVarray = [450,550,500,400]

pinkId = 0
middleId = 1
indexId = 2
thumbId = 3
ring_id = 4
def convertSensor(bla,bla2):
  return 0

p.setRealTimeSimulation(1)


words=[300,300,300,300,300]
while (1):#len(words)==6):
  pink = convertSensor(words[0],pinkId)
  middle = convertSensor(words[1],middleId)
  index = convertSensor(words[3],indexId)
  thumb = convertSensor(words[4],thumbId)
  ring = convertSensor(words[0],thumbId)

  p.setJointMotorControl2(hand,7,p.POSITION_CONTROL,pi/4.)	
  p.setJointMotorControl2(hand,9,p.POSITION_CONTROL,thumb+pi/10)
  p.setJointMotorControl2(hand,11,p.POSITION_CONTROL,thumb)
  p.setJointMotorControl2(hand,13,p.POSITION_CONTROL,thumb)

  p.setJointMotorControl2(hand,17,p.POSITION_CONTROL,index)
  p.setJointMotorControl2(hand,19,p.POSITION_CONTROL,index)
  p.setJointMotorControl2(hand,21,p.POSITION_CONTROL,index)

  p.setJointMotorControl2(hand,24,p.POSITION_CONTROL,middle)
  p.setJointMotorControl2(hand,26,p.POSITION_CONTROL,middle)
  p.setJointMotorControl2(hand,28,p.POSITION_CONTROL,middle)

  p.setJointMotorControl2(hand,40,p.POSITION_CONTROL,pink)
  p.setJointMotorControl2(hand,42,p.POSITION_CONTROL,pink)
  p.setJointMotorControl2(hand,44,p.POSITION_CONTROL,pink)

  ringpos = 0.5*(pink+middle)
  p.setJointMotorControl2(hand,32,p.POSITION_CONTROL,ringpos)
  p.setJointMotorControl2(hand,34,p.POSITION_CONTROL,ringpos)
  p.setJointMotorControl2(hand,36,p.POSITION_CONTROL,ringpos)

else:
  print("Cannot find port")
