#!/bin/bash
echo ""
echo "Let's build the NMPC!"
echo "enter your platform_type"
echo "default: race"
echo 'options: race race2 race_S voxl2 raxl2 iris'
echo ""
read platform_type
platform_type=${platform_type:-race}
echo 'thank you!'
echo ""

python3 dq_nmpc/dq_controller.py ../arpl_autonomy_stack/config/$platform_type/default/dq_control.yaml
source ~/.bashrc
cd ../..
colcon build --packages-select arpl_nmpc
source install/setup.bash
cd src/arpl_nmpc

echo ""
echo "copying acados lib in workspace"
cp c_generated_code/libacados_ocp_solver_quadrotor.so ../../install/arpl_nmpc/lib

