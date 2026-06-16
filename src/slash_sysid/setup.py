from setuptools import setup
import os
from glob import glob

package_name = 'slash_sysid'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jetson',
    maintainer_email='jetson@todo.todo',
    description='底盘系统辨识 + g-g-v 标定（纵向已实测；绕圆 bag 补横向，创新点3地基）',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 标定指令发生器（ROS 节点）
            'circle_commander = slash_sysid.circle_commander:main',
            # 离线辨识脚本也注册为可执行（也可直接 python3 chassis_sysid.py 运行）
            'chassis_sysid = slash_sysid.chassis_sysid:main',
        ],
    },
)
