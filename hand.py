import time,os,math,inspect,re
import random
import matplotlib.pyplot as plt
import pybullet as p
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)
pi = 3.14159265
debug = False
#first try to connect to shared memory (VR), if it fails use local GUI
#c = p.connect(p.SHARED_MEMORY)
#print(c)
p.connect(p.GUI)
#p.connect(p.DIRECT)
obj_to_classify = p.loadURDF("mesh.urdf",0,-1,0)
#obj_to_classify = p.loadURDF("mesh.urdf",0,0,-1)

p.setGravity(0,0,0)
#load the MuJoCo MJCF hand
objects = p.loadMJCF("MPL/MPL.xml",flags=0)
hand=objects[0]  #1 total
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
def convertSensor(bla,finger_index):
  if finger_index == indexId: 
    return random.uniform(-1,1)
    #return 0
  else:
    return 0
    #return random.random()

p.setRealTimeSimulation(0)

joint_count = p.getNumJoints(hand)
print("num of joints: ",joint_count)
pre = re.compile('index')
for id in range(joint_count):
  info = p.getJointInfo(hand,id)
  if pre.match(info[1].decode('utf-8')):
    print("joint info:", info)
words=[300,300,300,300,300]
while (1):
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


  #get camera position from index finger tip, indexId, 21
  link_state = p.getLinkState(hand,indexId)
  link_p = link_state[0]
  link_o = link_state[1]
  print("link position :",link_p)
  print("link orientation :",link_o)

  aspect = 1
  camTargetPos = [0,0,0]
  nearPlane = 0.01
  farPlane = 1000
  yaw = 40
  pitch = 10.0
  roll=0
  upAxisIndex = 2
  camDistance = 4
  pixelWidth = 320
  pixelHeight = 240
  nearPlane = 0.01
  farPlane = 1000
  lightDirection = [0,1,0]
  lightColor = [1,1,1]#optional
  fov = 60
  dist1 = 1.
  dist0 = 0.3
  #viewMatrix = p.computeViewMatrixFromYawPitchRoll(camTargetPos,camDistance,yaw,pitch,roll,upAxisIndex)
  #handpos,handorn = p.getBasePositionAndOrientation(hand)
  handmat = p.getMatrixFromQuaternion(link_o)
  #invhandPos,invhandOrn = p.invertTransform(handpos,handorn)
  #linkPosInHand,linkOrnInHand = self._p.multiplyTransforms(invHandPos,invHandOrn,link_p,link_o)
  target_pos = [link_p[0]+dist1*handmat[0],link_p[1]+dist1*handmat[3],link_p[2]+dist1*handmat[6]+0.3]
  #target_pos = [handpos[0]+dist1*handmat[0],handpos[1]+dist1*handmat[3],handpos[2]+dist1*handmat[6]+0.3]
  #eye_pos = [handpos[0]+dist0*handmat[0],handpos[1]+dist0*handmat[3],handpos[2]+dist0*handmat[6]+0.3]
  up = [handmat[2],handmat[5],handmat[8]]
  viewMatrix = p.computeViewMatrix(link_p,target_pos,up)
  projectionMatrix = p.computeProjectionMatrixFOV(fov,aspect,nearPlane,farPlane)
  img_arr = p.getCameraImage(200,200, viewMatrix,projectionMatrix, lightDirection,lightColor,renderer=p.ER_TINY_RENDERER)


  #why isnt the index finger all red?
  #p.setDebugObjectColor(hand,indexId,(255,0,0))
  #p.setDebugObjectColor(hand,middleId,(255,0,0))  #works
  #p.setDebugObjectColor(hand,thumbId,(255,0,0))  #works
  #joint info: (15, b'index_ABD', 0, 14, 13, 1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, b'link0_17')
  #joint info: (17, b'index_MCP', 0, 15, 14, 1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, b'link0_19')
  #joint info: (19, b'index_PIP', 0, 16, 15, 1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, b'link0_21')
  #joint info: (21, b'index_DIP', 0, 17, 16, 1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, b'link0_23')
  print("index id: ",indexId)
  p.setDebugObjectColor(hand,21,(255,0,0))
  p.setDebugObjectColor(hand,15,(255,0,0))
  p.setDebugObjectColor(hand,2,(255,0,0))
  p.setDebugObjectColor(hand,17,(255,0,0))
  p.setDebugObjectColor(hand,19,(255,0,0))

  p.stepSimulation()
  

  aabb = p.getAABB(hand,21) #should be 
  #print("AABB: ",aabb)

p.disconnect()
