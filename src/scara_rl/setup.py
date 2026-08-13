import os
from glob import glob

from setuptools import setup

package_name = "scara_rl"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="hahahibaa",
    maintainer_email="shahi785khan@gmail.com",
    description="RL-trained colour pick-and-place for a 4-DOF SCARA.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "sim_node = scara_rl.sim_node:main",
            "vision_node = scara_rl.vision_node:main",
            "policy_node = scara_rl.policy_node:main",
            "policy_bridge = scara_rl.policy_bridge:main",
            "grasp_node = scara_rl.grasp_node:main",
        ],
    },
)
