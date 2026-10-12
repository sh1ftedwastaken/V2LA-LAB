## TODOS:
[Main Package](./bev2/)
- Debug MPC Controller. 
    - Controller output (velocity, omega) doesn't match the require physical output.
    - Tune MPC variables.
    - Car bumper is not on the BEV canvas. Therefore, some solution are: use past frames to calculate velocities, use bumper as origin coords and find bev canvas bottom value to do calculations further down the image.

Currently:
[Testing Package](./bev_testing/)
- Refactoring Lane Analyzer and Controller.
