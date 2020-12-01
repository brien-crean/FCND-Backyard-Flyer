import argparse
import time
from enum import Enum

import numpy as np
import visdom

from udacidrone import Drone
from udacidrone.connection import MavlinkConnection, WebSocketConnection  # noqa: F401
from udacidrone.messaging import MsgID


class States(Enum):
    MANUAL = 0
    ARMING = 1
    TAKEOFF = 2
    WAYPOINT = 3
    LANDING = 4
    DISARMING = 5


class BackyardFlyer(Drone):

    def __init__(self, connection):
        super().__init__(connection)
        self.target_position = np.array([0.0, 0.0, 0.0])
        self.all_waypoints = []
        self.in_mission = True
        self.check_state = {}
        # enabling more than 2 plots at a time can cause performance issues
        self.is_postion_plotted = True
        self.is_velocity_plotted = False

        # Plotting
        self.v = visdom.Visdom()
        assert self.v.check_connection()

        if self.is_postion_plotted:
            self.plot_ned_position()
        elif self.is_velocity_plotted:
            self.plot_ne_velocity()

        # initial state
        self.flight_state = States.MANUAL

        # TODO: Register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)
    
    def plot_ned_position(self):
        # Plot North/East Position
        ne = np.array([self.local_position[0], self.local_position[1]]).reshape(1, -1)
        self.ne_plot = self.v.scatter(ne, opts=dict(
            title="Local position (north, east)", 
            xlabel='North', 
            ylabel='East'
        ))
        # Plot Down (Altitude) Position
        d = np.array([self.local_position[2]])
        self.t = 0
        self.d_plot = self.v.line(d, X=np.array([self.t]), opts=dict(
            title="Altitude (meters)", 
            xlabel='Timestep', 
            ylabel='Down'
        ))

        self.register_callback(MsgID.LOCAL_POSITION, self.update_ne_plot)
        self.register_callback(MsgID.LOCAL_POSITION, self.update_d_plot)

    def plot_ne_velocity(self):
        # Plot North/East Velocity
        velocity = np.array([np.linalg.norm(self.local_velocity[0:2])])
        self.t = 0
        self.velocity_plot = self.v.line(velocity, X=np.array([self.t]), opts=dict(
            title="Velocity (m/s)", 
            xlabel='Timestep', 
            ylabel='Velocity'
        ))
        self.register_callback(MsgID.LOCAL_VELOCITY, self.update_ne_velocity_plot)

    def update_ne_plot(self):
        ne = np.array([self.local_position[0], self.local_position[1]]).reshape(1, -1)
        self.v.scatter(ne, win=self.ne_plot, update='append')

    def update_d_plot(self):
        d = np.array([self.local_position[2]])
        # update timestep
        self.t += 1
        self.v.line(d, X=np.array([self.t]), win=self.d_plot, update='append')

    def update_ne_velocity_plot(self):
        velocity = np.array([np.linalg.norm(self.local_velocity[0:2])])
        # update timestep
        self.t += 1
        self.v.line(velocity, X=np.array([self.t]), win=self.velocity_plot, update='append')

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if abs(self.local_position[2]) > 0.95 * self.target_position[2]:
                print('reached flight altitude')
                self.all_waypoints = self.calculate_box()
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 1.0:
                print('waypoint reached')
                if self.all_waypoints:
                    self.waypoint_transition()
                else:
                    print('no more waypoints')
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        print('slowed sufficiently to land')
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if ((self.global_position[2] - self.global_home[2] < 0.1) and
                    abs(self.local_position[2]) < 0.01):
                self.disarming_transition()

    def state_callback(self):
        if not self.in_mission:
            return
        if self.flight_state == States.MANUAL:
            self.arming_transition()
        elif self.flight_state == States.ARMING:
            self.takeoff_transition()
        elif self.flight_state == States.DISARMING:
            self.manual_transition()

    def calculate_box(self):
        return [[15.0, 0.0, 3.0], [15.0, 15.0, 3.0], [0.0, 15.0, 3.0], [0.0, 0.0, 3.0]]

    def arming_transition(self):
        print('arming transition')
        self.take_control()
        self.arm()
        self.set_home_position(self.global_position[0], 
                               self.global_position[1], 
                               self.global_position[2])
        self.flight_state = States.ARMING
        
    def takeoff_transition(self):
        print('takeoff transition')
        print('home', self.global_home)
        target_altitude = 3.0
        self.target_position[2] = target_altitude
        self.takeoff(target_altitude)
        self.flight_state = States.TAKEOFF

    def waypoint_transition(self):
        print('waypoint transition')
        self.target_position = self.all_waypoints.pop(0)
        print('going to waypoint', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], 0.0)
        self.flight_state = States.WAYPOINT

    def landing_transition(self):
        print('landing transition')
        self.land()
        self.flight_state = States.LANDING

    def disarming_transition(self):
        print('disarm transition')
        self.disarm()
        self.flight_state = States.DISARMING

    def manual_transition(self):
        print('manual transition')
        self.release_control()
        self.stop()
        self.in_mission = False
        self.flight_state = States.MANUAL

    def start(self):
        print("Creating log file")
        self.start_log("Logs", "NavLog.txt")
        print("starting connection")
        self.connection.start()
        print("Closing log file")
        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), threaded=False, PX4=False)
    #conn = WebSocketConnection('ws://{0}:{1}'.format(args.host, args.port))
    drone = BackyardFlyer(conn)
    time.sleep(2)
    drone.start()
