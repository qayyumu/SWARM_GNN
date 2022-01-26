import os
import argparse
import json

import numpy as np
from util_swarm_func import *
import matplotlib.pyplot as plt

num_of_simulations = 10
obstacle_fixed  = 1            ## if we want the obstacles to be at fixed location 
animation_steps = 2000

def random_obstacle(position1, position2, r):
    '''
    Generate a random obstacle of radius r randomly placed between position1 and position2.
    
    :param position1: The position of the first obstacle
    :param position2: the position of the obstacle
    :param r: radius of the obstacle
    :return: A list of obstacles.
    '''
    """Return an obstacle of radius r randomly placed between position1 and position2"""
    d = position1 - position2
    d_len = np.sqrt(d.dot(d))
    cos = d[0] / d_len
    sin = d[1] / d_len

    if(obstacle_fixed):
    # Generat random x and y assuming d is aligned with x axis.
        x = 13 #np.random.uniform(.1 + r, d_len - r)
        y = 8 #np.random.uniform(.1 * r, 1 * r)
    else:
        x = np.random.uniform(2 + r, d_len - r)
        y = np.random.uniform(-2 * r, 2 * r)


    # Rotate the alignment back to the actural d.
    true_x = x * cos + y * sin + position2[0]
    true_y = x * sin - y * cos + position2[1]
    


    return Sphere(r, [true_x, true_y], ndim=2)


def system_edges(obstacles, boids, vicseks):
    """Edge types of the directed graph representing the influences between
    elements of the system.
|      |Goal|Obst|Obst 1|Obst 2|Boid 1|Boid 2|  
|Goal  | 0  | 0  | 0    | 0    | 1    | 1    |  
|Obst 1| 0  | 0  | 0    | 0    | 2    | 2    |
|Obst 2| 0  | 0  | 0    | 0    | 2    | 2    |
|Boid 1| 0  | 0  | 0    | 0    | 0    | 3    |  
|Boid 2| 0  | 0  | 0    | 0    | 3    | 0    |  
    """

    # If boids == 0, edges would be same as if vicseks were boids
    if boids == 0:
        boids, vicseks = vicseks, boids

    particles = 1 + obstacles + boids + vicseks
    edges = np.zeros((particles, particles), dtype=int)

    up_to_goal = 1
    up_to_obs = up_to_goal + obstacles
    up_to_boids = up_to_obs + boids

    edges[0, up_to_obs:up_to_boids] = 1  # influence from goal to boid.
    edges[up_to_goal:up_to_obs, up_to_obs:up_to_boids] = 2  # influence from obstacle to boid.
    edges[up_to_obs:up_to_boids, up_to_obs:up_to_boids] = 3  # influence from boid to boid.
    edges[up_to_boids:, up_to_obs:up_to_boids] = 4  # influence from vicsek to boid.

    edges[0, up_to_boids:] = 5  # influence from goal to vicsek.
    edges[up_to_goal:up_to_obs, up_to_boids:] = 6  # influence from obstacle to vicsek.
    edges[up_to_obs:up_to_boids, up_to_boids:] = 7  # influence from obstacle to agent.
    edges[up_to_boids:, up_to_boids:] = 8  # influence from viscek to viscek.

    np.fill_diagonal(edges, 0)
    return edges


def simulation(args, _):
    np.random.seed()

    region = (-100, 100, -100, 100)

    env = Environment2D(region)

    goal = Goal(np.random.uniform(-40, 40, 2), ndim=2)
    env.add_goal(goal)

    for _ in range(args.boids):
        position = np.random.uniform(-80, 80, 2)
        velocity = np.random.uniform(-15, 15, 2)

        agent = Boid(position, velocity, ndim=2, vision=args.vision, size=args.size,
                     max_speed=10, max_acceleration=5)
        agent.set_goal(goal)

        env.add_agent(agent)
   # Create a sphere obstacle near segment between avg boids position and goal position.
    avg_boids_position = np.mean(
        np.vstack([agent.position for agent in env.population]), axis=0)

    spheres = []
    for _ in range(args.obstacles):
        
        if(obstacle_fixed):         
            sphere = random_obstacle(np.array([-54,64]), np.array([33,17]), 8)
        else:
            sphere = random_obstacle(avg_boids_position, goal.position, 8)
        spheres.append(sphere)
        env.add_obstacle(sphere)

    position_data = []
    velocity_data = []
    time_data = []
    t = 0
    for _ in range(args.steps):
        env.update(args.dt)
        position_data.append([goal.position for goal in env.goals] +
                             [sphere.position for sphere in spheres] +
                             [agent.position.copy() for agent in env.population])
        velocity_data.append([np.zeros(2) for goal in env.goals] +
                             [np.zeros(2) for sphere in spheres] +
                             [agent.velocity.copy() for agent in env.population])
        time_data.append(t)
        t += args.dt

    position_data, velocity_data = np.asarray(position_data), np.asarray(velocity_data)
    timeseries_data = np.concatenate([position_data, velocity_data], axis=-1)

    edge_data = system_edges(args.obstacles, args.boids, args.vicseks)

    return timeseries_data, edge_data, time_data

def maximize(plt):
    plot_backend = plt.get_backend()
    mng = plt.get_current_fig_manager()
    if plot_backend == 'TkAgg':
        mng.resize(*mng.window.maxsize())
    elif plot_backend == 'wxAgg':
        mng.frame.Maximize(True)
    elif plot_backend == 'Qt5Agg':
        mng.window.showMaximized()



def main():
    if not os.path.exists(ARGS.save_dir):
        os.makedirs(ARGS.save_dir)

    file_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(file_dir, ARGS.config)) as f:
        model_config = json.load(f)

    if ARGS.boids > 0:
        Boid.set_model(model_config["boid"])
    

    for i in range(num_of_simulations):
        
        timeseries_data_all, edge_data_all, time_data_all = \
            run_simulation(simulation, ARGS, ARGS.instances, ARGS.processes, ARGS.batch_size)

        
        
        ### plot the trajectory here
        plot_timeseries_data = np.array(timeseries_data_all)
        
        
        ## get the target goal position
        target_pos = plot_timeseries_data[0,:,0,0:2]   ## get zero index -target position
        obj_pos = plot_timeseries_data[0,:,1:3,0:2]   ## get zero index -obstacle position
        boid_pos = plot_timeseries_data[0,:,3:5,0:2]   ## get zero index -boids position
        
        
        ###plot goal
        plt.plot(target_pos[:,0],target_pos[:,1],'r*',marker="*",markersize=25)
        ###plot obstacle
        plt.plot(obj_pos[:,0],obj_pos[:,1],'bo',marker="o",markersize=25)
        ###plot boid -1
        plt.plot(boid_pos[0,0,0],boid_pos[0,0,1],'g*',marker="*",markersize=15)   ## starting point of boid-1
        plt.plot(boid_pos[:,0,0],boid_pos[:,0,1],'g.')
        ###plot boid 2
        plt.plot(boid_pos[0,1,0],boid_pos[0,1,1],'y*',marker="*",markersize=15)   ##starting point of boid 
        plt.plot(boid_pos[:,1,0],boid_pos[:,1,1],'y.')
        
        plt.xlim([-80, 80])
        plt.ylim([-80, 80])
        maximize(plt)
        # plt.show()
        plt.title(['Simulation # '+ str(i)])
        plt.show(block=False)
        plt.savefig(str(i)+'.png')
        
        plt.pause(1)
        plt.close()
        
        
        newsuffix = ARGS.suffix+str(i)
        np.save(os.path.join(ARGS.save_dir,
                            f'{ARGS.prefix}_timeseries{newsuffix}.npy'), timeseries_data_all)
        np.save(os.path.join(ARGS.save_dir, f'{ARGS.prefix}_edge{newsuffix}.npy'), edge_data_all)
        np.save(os.path.join(ARGS.save_dir, f'{ARGS.prefix}_time{newsuffix}.npy'), time_data_all)
        
     # Build GIF
    import imageio
    with imageio.get_writer('mygif.gif', mode='I') as writer:
        for filename in range(num_of_simulations):
            image = imageio.imread(str(filename)+'.png')
            writer.append_data(image)
            
    
    # from matplotlib.animation import FuncAnimation, PillowWriter
    # fig = plt.figure()
    # fig.show()
    # # axis = plt.axes(xlim =(-80, 80), ylim =(-80, 80))
    # line, = plt.plot([], [], lw = 3)
    # line.set_data([], [])
    
    
    # def animate(i):
    #     x = boid_pos[i,1,0]
    #     # plots a sine graph
    #     y = boid_pos[i,1,1]
    #     line.set_data(x, y)
    #     return line,
    
    
    # anim = FuncAnimation(fig, animate,
    #                 frames = 2000,
    #                 interval = 100,
    #                 blit = True)
    # # plt.show()
    # anim.save("TLI.gif", dpi=300, writer=PillowWriter(fps=25))    
        
    
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--boids', type=int, default=2,
                        help='number of boid agents')
    parser.add_argument('--vicseks', type=int, default=0,
                        help='number of vicsek agents')
    parser.add_argument('--obstacles', type=int, default=2,
                        help='number of obstacles')
    parser.add_argument('--vision', type=float, default=None,
                        help='vision range to determine range of interaction')
    parser.add_argument('--size', type=float, default=3,
                        help='agent size')
    parser.add_argument('--steps', type=int, default=animation_steps,
                        help='number of simulation steps')
    parser.add_argument('--instances', type=int, default=1,
                        help='number of simulation instances')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='time resolution')
    parser.add_argument('--config', type=str, default='boid_default.json',
                        help='path to config file')
    parser.add_argument('--save-dir', type=str, default='.',
                        help='name of the save directory')
    parser.add_argument('--prefix', type=str, default='',
                        help='prefix for save files')
    parser.add_argument('--suffix', type=str, default='savedata_',
                        help='suffix for save files')
    parser.add_argument('--processes', type=int, default=1,
                        help='number of parallel processes')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='number of simulation instances for each process')

    ARGS = parser.parse_args()

    ARGS.save_dir = os.path.expanduser(ARGS.save_dir)

    main()
