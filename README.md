The paper we worked on contains 2 simulation trajectories, "Spiral" and "Loop and corner". 
We applied both of them for the software part "simulation" each in two cases, with and without disturbances, using "Python". 
The results showed significant differences in the performance of the software with and without disturbances, especially for the "Loop and corner" , 
and even after applying tuning, the results weren't good enough to apply in hardware. On the other hand, the "Spiral" trajectory simulation shows promising results for the software.
All you need to run the code and see the results is to have installed the required libraries , "matplotlib", "numpy" and "scipy". 
"Results" file contains the software simulation results, 
S1NoDisturbances: Simulation 1 with no disturbances,
S1-Disturbances: Simulation 1 with applied disturbances,
S2NoDisturbances: Simulation 2 with no disturbances,
S2-Disturbances: Simulation 2 with applied disturbances.

For the hardware part, as we get great results in the simulated "Spiral" trajectory, so we applied the hardware only on this trajectory. 
We worked on the DJI Tello drone in the lab, we installed the required libraries to work with python.
All you need to do to test the hardware part is to connect to the DJI Tello drone via WIFI then run the Main.py file. 
When working on hardware we got 2 results one of them follows the trajectory in a good way but the issue was that it was tracking the trajectory from a distance that is too close to the floor 
"Issues with the take off ,mainly with z". Other result the drone takes off successfully it reaches some point and then starts tacking the trajectory, 
but in this case we got an oscillating path due to unstability in z, in order to solve this problem it needs tuning so we can have better results, but for now this is all we got. 
You can see the hardware plot results in the hardware files that contain the code and the plots observed.
