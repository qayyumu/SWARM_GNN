import os
import argparse
import json
import time

import numpy as np

from util_swarm_func import *
"""
To generate gif of given timesteps

"""
def animates(boids,save_name,steps,pos1,env, region):
    import matplotlib.pyplot as plt
    from matplotlib import animation
    plt.rcParams['animation.html'] = 'html5'
    plt.rcParams['animation.ffmpeg_path'] = 'C:\FFMPEG\bin\ffmpeg'

    def animate(i,boids,pos1, scats):
        pos=pos1[i]
        boid_positions = [pos[3+k][:2]for k in range(boids)]
        #print(boid_positions)
        if boid_positions:
            scats[0].set_offsets(boid_positions)

        goal_positions = [pos[0][:2]]
        scats[2].set_offsets(goal_positions)

        return scats

    xmin, xmax, ymin, ymax = region

    fig, ax = plt.subplots()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')

    scats = [ax.scatter([], [], color='b'),
             ax.scatter([], [], color='m'),
             ax.scatter([], [], color='g')]

    for obstacle in env.obstacles:
        if not isinstance(obstacle, Wall):
            circle = plt.Circle(obstacle.position,
                                obstacle.size, color='r', fill=False)
            ax.add_patch(circle)

    anim = animation.FuncAnimation(fig, animate,
                                   fargs=(boids,pos1,scats),
                                   frames=steps, interval=20, blit=True)
    #my_writer=animation.PillowWriter(fps=20, codec='libx264', bitrate=2)
    anim.save(save_name + '.gif', dpi=80, writer='imagemagick')
    #anim.save(ARGS.save_name + '.gif', dpi=80, writer=my_writer)

def environmentsetup(filename,boids,save_name,prediction):
    region = (-100, 100, -100, 100)
    env = Environment2D(region)
    pos=np.load(f'{filename}.npy')
    print(pos.shape)
    pos=pos[0]
    pos_i=pos[0]
    goal = Goal(pos_i[0][:2], None, ndim=2)
    env.add_goal(goal)
    for i in range(boids):
        position = pos_i[3+i][:2]
        velocity = pos_i[3+i][2:]

        agent = Boid(position, velocity, ndim=2, size=3, max_speed=10, max_acceleration=20)
        agent.set_goal(goal)
        env.add_agent(agent)

    # Create a sphere obstacle 
    sphere = Sphere(8, pos_i[1][:2], ndim=2)
    env.add_obstacle(sphere)
    sphere = Sphere(8, pos_i[2][:2], ndim=2)
    env.add_obstacle(sphere)
    animates(boids,save_name,prediction,pos,env,region)
    return env


def environmentsetup_new(filename,boids,save_name,prediction):
    region = (-100, 100, -100, 100)
    env = Environment2D(region)
    pos=np.load(f'{filename}.npy')
    pos=np.transpose(pos,[1,0,2,3])
    print(pos.shape)
    pos=pos[0]
    pos_i=pos[0]
    goal = Goal(pos_i[0][:2], None, ndim=2)
    env.add_goal(goal)
    for i in range(boids):
        position = pos_i[3+i][:2]
        velocity = pos_i[3+i][2:]

        agent = Boid(position, velocity, ndim=2, size=3, max_speed=10, max_acceleration=20)
        agent.set_goal(goal)
        env.add_agent(agent)

    # Create a sphere obstacle 
    sphere = Sphere(8, pos_i[1][:2], ndim=2)
    env.add_obstacle(sphere)
    sphere = Sphere(8, pos_i[2][:2], ndim=2)
    env.add_obstacle(sphere)
    animates(boids,save_name,prediction,pos[-prediction:],env,region)
    return env



def main(): 
    environmentsetup(ARGS.filename,ARGS.boids,ARGS.save_name,ARGS.steps)


if __name__ == '__main__':
    
    
    class EmptyClass():
        pass
    ARGS = EmptyClass()

    ARGS.boids = 4
    ARGS.vicseks=0
    ARGS.obstacles=2
    ARGS.steps=500
    ARGS.dt=0.02
    ARGS.config='.'
    ARGS.save_name='_test'
    ARGS.filename= 'prediction_500'
    main()
