import time,os,math,inspect,re
import random
import matplotlib.pyplot as plt
import pybullet as p
import numpy as np

pi = 3.1415926535

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)
#c = p.connect(p.SHARED_MEMORY)
#print(c)
p.connect(p.GUI)
#p.connect(p.DIRECT)
obj_to_classify = p.loadURDF("mesh.urdf",0,-1,0)

move = 0.01
moveSlider = p.addUserDebugParameter("move",0.001,1,0.01)
near_plane_slider = p.addUserDebugParameter("nearplane",0.001,1,0.01)
far_plane_slider = p.addUserDebugParameter("farplane",0.001,1,0.05)
fov_slider = p.addUserDebugParameter("fov",-100,100,50)
aspect_slider= p.addUserDebugParameter("aspect",-10,10,1)
p.setGravity(0,0,0)
#load the MuJoCo MJCF hand
objects = p.loadMJCF("MPL/MPL.xml",flags=0)
hand=objects[0]  #1 total

p.setRealTimeSimulation(0)

offset = 0.02 # Offset from basic position
offset1 = 0.2 # Offset from basic position

indexEndID = 21 # Need get position and orientation from index finger parts


while (1):
  pink = 0
  middle = 0
  index = 0
  thumb = 0
  ring = 0
  p.setJointMotorControl2(hand,7,p.POSITION_CONTROL,pi/4.)	
  p.setJointMotorControl2(hand,9,p.POSITION_CONTROL,thumb+pi/10)
  p.setJointMotorControl2(hand,11,p.POSITION_CONTROL,thumb)
  p.setJointMotorControl2(hand,13,p.POSITION_CONTROL,thumb)

  # That's index finger parts
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


  key = p.getKeyboardEvents()
  for k in key.keys():
    if k == 32 and key[k] == 3:
      pass
    else:
      hand_po = p.getBasePositionAndOrientation(hand)
      move = p.readUserDebugParameter(moveSlider)
      if k == 65298: #down
        p.resetBasePositionAndOrientation(hand,(hand_po[0][0]+move,hand_po[0][1],hand_po[0][2]),hand_po[1])
      elif k == 65297: #up
        p.resetBasePositionAndOrientation(hand,(hand_po[0][0]-move,hand_po[0][1],hand_po[0][2]),hand_po[1])
      elif k == 65295: #left
        p.resetBasePositionAndOrientation(hand,(hand_po[0][0],hand_po[0][1]+move,hand_po[0][2]),hand_po[1])
      elif k == 65296: #right
        p.resetBasePositionAndOrientation(hand,(hand_po[0][0],hand_po[0][1]-move,hand_po[0][2]),hand_po[1])
      elif k == 44: #<
        p.resetBasePositionAndOrientation(hand,(hand_po[0][0],hand_po[0][1],hand_po[0][2]+move),hand_po[1])
      elif k == 46: #>
        p.resetBasePositionAndOrientation(hand,(hand_po[0][0],hand_po[0][1],hand_po[0][2]-move),hand_po[1])

  link_state = p.getLinkState(hand,indexEndID)
  link_p = link_state[0]
  link_o = link_state[1]
  handmat = p.getMatrixFromQuaternion(link_o)

  axisX = [handmat[0],handmat[3],handmat[6]]
  axisY = [-handmat[1],-handmat[4],-handmat[7]] # Negative Y axis
  axisZ = [handmat[2],handmat[5],handmat[8]]

  eye_pos    = [link_p[0],link_p[1],link_p[2]]
  #eye_pos    = [link_p[0]+offset*axisY[0],link_p[1]+offset*axisY[1],link_p[2]+offset*axisY[2]]
  target_pos = [link_p[0]+offset1,link_p[1],link_p[2]] # target position based by axisY, not X
  #target_pos = [link_p[0]+offset1*axisY[0],link_p[1]+offset1*axisY[1],link_p[2]+offset1*axisY[2]] # target position based by axisY, not X
  up = axisZ # Up is Z axis
  viewMatrix = p.computeViewMatrix(eye_pos,target_pos,up)

  #p.addUserDebugLine(link_p,[link_p[0]+0.1*axisY[0],link_p[1]+0.1*axisY[1],link_p[2]+0.1*axisY[2]],[1,0,0],2,0.05) # Debug line in camera direction
  p.addUserDebugLine(link_p,[link_p[0]+0.1*axisY[0],link_p[1]+0.1*axisY[1],link_p[2]+0.1*axisY[2]],[1,0,0],2,0.2)

  aspect = 1
  camTargetPos = [0,0,0]
  nearPlane = 0.01
  farPlane = 1000
  nearPlane = 0.01
  nearPlane = p.readUserDebugParameter(near_plane_slider)
  farPlane = 0.05
  farPlane = p.readUserDebugParameter(far_plane_slider)
  lightDirection = [0,1,0]
  lightColor = [1,1,1]#optional
  fov = 50  #10 or 50
  fov = p.readUserDebugParameter(fov_slider)
  aspect = p.readUserDebugParameter(aspect_slider)
  projectionMatrix = p.computeProjectionMatrixFOV(fov,aspect,nearPlane,farPlane)
  w,h,img_arr,depths,mask = p.getCameraImage(200,200, viewMatrix,projectionMatrix, lightDirection,lightColor,renderer=p.ER_TINY_RENDERER)

  #p.resetDebugVisualizerCamera( cameraDistance=cDistance, cameraYaw=cYaw, cameraPitch=cPitch, cameraTargetPosition=[0,0,0])
  p.stepSimulation()

