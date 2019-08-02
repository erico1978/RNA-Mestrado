__author__ = "RoboFEI-HT"
__authors__ = "Aislan C. Almeida"
__license__ = "GNU General Public License v3.0"

from Viewer import *
from MCL import *
import time

# To pass arguments to the function
import argparse
# Import a shared memory
import sys
sys.path.append('../../Blackboard/src/')
from SharedMemory import SharedMemory  # noqa E402

try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser

# To parse arguments on execution
parser = argparse.ArgumentParser(
    description='Robot Localization',
    epilog= 'Implements particle filters to self-localize a robot on the field.')  # noqa E501
parser.add_argument(
    '-g', '--graphs',
    action="store_true",
    help='Shows graphical interface which visualizes the particles.')
parser.add_argument(
    '-l', '--log',
    action="store_true",
    help='Print variable logs.')
parser.add_argument(
    '-i', '--robot_id',
    type=int,
    help='Robot id which will be simulated.')

args = parser.parse_args()


#   Class implementing the Core of the Localization Process
class Localization():
    #   Class constructor and pre-processing.
    def __init__(self):
        self.bkb = SharedMemory()
        config = ConfigParser()

        self.args = parser.parse_args()
        
        try:
            config.read('../../Control/Data/config.ini')
            self.robot_id = int(config.get('Communication', 'no_player_robofei'))
            mem_key = self.robot_id * 100
        except:  # noqa E722
            if self.args.robot_id:
                self.robot_id = self.args.robot_id
                mem_key = self.robot_id * 100
                print "Error loading ConfigParser."
                print "Resuming with robot_id", self.robot_id
            else:
                print "Neither ConfigParser worked nor robot_id was given."
                print "Finishing execution of Localization."
                sys.exit()

        self.Mem = self.bkb.shd_constructor(mem_key)

        # Timestamp to use on the time step used for motion
        self.timestamp = time.time()

    #   Localization's main method.
    def main(self):
        screen = Screen(self.args.graphs)

        if self.args.graphs:
            simul = Simulation(screen)
            field = SoccerField(screen)
            simul.field = field

        PF = MonteCarlo(robot_id=self.robot_id)

        # Erasing Shared memories values.
        obs = ['LANDMARK_L_1', 'LANDMARK_L_2', 'LANDMARK_L_3', 'LANDMARK_L_4', 'LANDMARK_T_1', 'LANDMARK_T_2', 'LANDMARK_X_1', 'LANDMARK_X_2', 'LANDMARK_P']
        for i in obs:
            self.bkb.write_int(self.Mem, i, -999)

        std = 100
        hp = -999

        weight = 1

        # Main loop
        while True:
            landmarks = []

            self.bkb.write_int(self.Mem, 'LOCALIZATION_WORKING', 1)

            # Process interactions events
            if self.args.graphs:
                simul.perform_events()

            # Gets the motion command from the blackboard.
            u = self.GetU(self.bkb.read_int(self.Mem, 'CONTROL_ACTION'))

            # Gets the measured variable from the blackboard,
            z = []
            auxz = []
            for i in obs[:4]:
                aux = self.bkb.read_int(self.Mem, i)
                if aux == -999:
                    aux = None
                auxz.append(aux)
            z.append(auxz)

            auxz = []
            for i in obs[4:6]:
                aux = self.bkb.read_int(self.Mem, i)
                if aux == -999:
                    aux = None
                auxz.append(aux)
            z.append(auxz)

            auxz = []
            for i in obs[6:8]:
                aux = self.bkb.read_int(self.Mem, i)
                if aux == -999:
                    aux = None
                auxz.append(aux)
            z.append(auxz)

            z.append([self.bkb.read_int(self.Mem, obs[-1])])

            z.append(self.bkb.read_int(self.Mem, "IMU_EULER_Z",))

            # and free them.
            for i in obs:
                self.bkb.write_int(self.Mem, i, -999)

            pos, std = PF.main(u, z)

            if PF.meanweight < 1:
                weight = np.log(0.05)/np.log(PF.meanweight)

            if self.args.log:
                print '\t\x1b[32mRobot at',
                print 'ent\x1b[32m[x:\x1b[34m{} cm'.format(int(pos[0])),
                print '\x1b[32m| y:\x1b[34m{} cm'.format(int(pos[1])),
                print u'\x1b[32m| \u03B8:\x1b[34m{}\u00B0'.format(int(pos[2])),
                print u'\x1b[32m| \u03C3:\x1b[34m{} cm\x1b[32m]'.format(int(std))  # noqa E501

            # Write the robot's position on Black Board to be read by telemetry
            self.bkb.write_int(self.Mem, 'LOCALIZATION_X', int(pos[0]))
            self.bkb.write_int(self.Mem, 'LOCALIZATION_Y', int(pos[1]))
            self.bkb.write_int(self.Mem, 'LOCALIZATION_THETA', int(pos[2]))
            self.bkb.write_float(self.Mem, 'LOCALIZATION_RBT01_X', std)

            if self.args.graphs:
                # Redraws the screen background
                field.draw_soccer_field()
                simul.DrawStd(pos, std, weight, hp)

                # Draws all particles on screen
                simul.display_update(PF.particles)

            # Updates for the next clock
            screen.clock.tick(5)

    #   This method returns a command instruction to the particles.
    def GetU(self, Action):
        if Action in [0, 4, 5, 12, 13, 19, 20, 21, 22]:
            return (0, 0, 0, 0, self.dt())  # Stop or kick
        elif Action == 11:
            return (0, 0, 0, 1, self.dt())  # Gait
        elif Action == 1:
            return (20, 0, 0, 1, self.dt())  # Fast Walk Forward
        elif Action == 8:
            return (10, 0, 0, 1, self.dt())  # Slow Walk Forward
        elif Action == 17:
            return (-20, 0, 0, 1, self.dt())  # Fast Walk Backward
        elif Action == 18:
            return (-10, 0, 0, 1, self.dt())  # Slow Walk Backward
        elif Action == 6:
            return (0, -10, 0, 1, self.dt())  # Walk Left
        elif Action == 7:
            return (0, 10, 0, 1, self.dt())  # Walk Right
        elif Action == 3:
            return (0, 0, 18.7, 1, self.dt())  # Turn Right
        elif Action == 2:
            return (0, 0, -18.7, 1, self.dt())  # Turn Left
        elif Action == 9:
            return (0, -10, -20, 1, self.dt())  # Turn Left Around the Ball
        elif Action == 14:
            return (0, 10, 20, 1, self.dt())  # Turn Right Around the Ball
        elif Action == 16:
            return (0, 0, 0, 2, self.dt())  # Get up, back up
        elif Action == 15:
            return (0, 0, 0, 3, self.dt())  # Get up, front up
        elif Action == 10:
            print "ERROR - Please, edit Localization.GetU() for Goalkeeper before resuming!"  # noqa E501
            return (0, 0, 0, 0, self.dt())

    #   This method returns the time since the last update
    def dt(self):
        auxtime = time.time()
        timer = auxtime - self.timestamp
        self.timestamp = auxtime
        return timer


if __name__ == "__main__":
    Loc = Localization()
    Loc.main()
