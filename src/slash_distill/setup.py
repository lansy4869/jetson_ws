from setuptools import setup
import os
from glob import glob

package_name = 'slash_distill'

setup(
    name=package_name,
    version='0.0.1',
    packages=[
        package_name,
        package_name + '.common',
        package_name + '.sim',
        package_name + '.experts',
        package_name + '.models',
        package_name + '.nodes',
    ],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jetson',
    maintainer_email='jetson@todo.todo',
    description='W5-W8 特权蒸馏竞速：sysid 接地仿真 + 特权专家 + BC/DAgger 学生 + 部署 + 消融（创新点4 与 §5 消融）',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 部署节点（W7）
            'student_policy_node = slash_distill.nodes.student_policy_node:main',
            # 离线脚本也注册成 console_scripts，便于 ros2 run（也可直接 python -m）
            'collect_demos = slash_distill.collect_demos:main',
            'train_bc = slash_distill.train_bc:main',
            'eval_closed_loop = slash_distill.eval_closed_loop:main',
            'dagger = slash_distill.dagger:main',
            'export_onnx = slash_distill.export_onnx:main',
            'run_ablations = slash_distill.run_ablations:main',
            'make_figures = slash_distill.make_figures:main',
        ],
    },
)
