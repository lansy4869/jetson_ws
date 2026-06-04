from setuptools import setup, find_packages

package_name = 'roboracer_china_2025'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/battle_fast2.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jetson',
    maintainer_email='jetson@todo.todo',
    description='Roboracer battle fast2 node',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'battle_fast2_node = roboracer_china_2025.battle_fast2_node:main',
        ],
    },
)
