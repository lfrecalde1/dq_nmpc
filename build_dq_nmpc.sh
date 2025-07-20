#!/bin/bash
echo ""
echo "Let's build the NMPC!"
echo "enter your platform_type"
echo "default: race"
echo 'options: race race2 race_S voxl2 raxl2 iris eagle'
echo ""
read platform_type
platform_type=${platform_type:-race}
echo 'thank you!'
echo ""

python3 dq_nmpc/dq_controller.py /home/ros2_ws/src/arpl_autonomy_stack/config/eagle/default/dq_control_simulator_force.yaml

cp c_generated_code/libacados_ocp_solver_quadrotor.so /home/ros2_ws/install/dq_cpp/lib

echo "Deleting old Files"
rm -rf /home/ros2_ws/src/dq_cpp/c_generated_code
mv -f c_generated_code /home/ros2_ws/src/dq_cpp/


cd /home/ros2_ws/
source ~/.bashrc
colcon build --symlink-install --packages-select dq_cpp
source install/setup.bash
