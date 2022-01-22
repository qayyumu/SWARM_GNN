# Swarms Simulation (Data generated from this repo will be used for GNN training)
Models of artificial swarms (implementation of Boid Model)


Original implementation is taken from Siyu Zhou (https://github.com/siyuzhou)

## Data generation

```bash
python obstacle_avoidance_sim.py --boids B --obstacles OBST --steps N_STEPS --save-dir /path/to/save/location/
```
B: No. of Boids  default:2  
OBST: No. of obstacles  default:2
N_STEPS: No. of steps for which data is generated  default:200 

### Files Generated
*_edge.npy: Edge type data   
*_time.npy: Time steps(dt)  
*_timeseries.npy: N_STEPS x N x D tensor  

N: Goal + No. of obstacles + No. of Boids + No. of Vicseks  
D: State vector in form of (position, velocity)

## Edge types
Edge types of the directed graph represent the influences between
elements of the system.
```


|      |Goal|Obst|Obst 1|Obst 2|Boid 1|Boid 2|  
|Goal  | 0  | 0  | 0    | 0    | 1    | 1    |  
|Obst 1| 0  | 0  | 0    | 0    | 2    | 2    |
|Obst 2| 0  | 0  | 0    | 0    | 2    | 2    |
|Boid 1| 0  | 0  | 0    | 0    | 0    | 3    |  
|Boid 2| 0  | 0  | 0    | 0    | 3    | 0    |  


```

## Boid

A simple implementation of Craig Reynolds' [Boids](https://www.red3d.com/cwr/boids/) model.  

![2D Flocking w/ Goal and Obstacle](demo/boid_goal_obstacle.gif)
![2D Flocking w/ Goal and Obstacle2](demo/boid_goal_obstacle2.gif)
![2D Flocking w/ Goal and Obstacle2](demo/boid_goal_obstacle3.gif)

