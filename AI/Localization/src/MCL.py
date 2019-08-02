__author__ = "RoboFEI-HT"
__authors__ = "Aislan C. Almeida"
__license__ = "GNU General Public License v3.0"

import numpy as np
from particle import *


#   This class implements the Monte Carlo's Particle Filter
class MonteCarlo():
    #   Constructor of the particle filter
    def __init__(self, max_qtd=1000, min_qtd=30, robot_id=0):
        # Holds the particles objects
        self.particles = []

        # Limits the quantity of particles the filter will have
        self.max_qtd = max_qtd
        self.min_qtd = min_qtd

        # Initializes with the max quantity of particles
        self.qtd = max_qtd

        if robot_id == 1:
            factors = [0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0]
            normals = [[[200, 5], [70, 5], [-90, 5]]]
        else:
            factors = None
            normals = None

        # self.particles.append(Particle(350,350,0,factors=15*[0]))
        for i in range(self.qtd):
            # Randomly generates n particles
            self.particles.append(Particle(
                factors=factors,
                normals=normals
            ))

        self.totalweight = 0.
        self.meanweight = 0

        self.mean = [0, 0, 0]
        self.std = 1.

        self.VQPsigma = 3
        self.VQPmean = 10

    #   Prediction step
    def Prediction(self, u=None):
        # If there was movement, run the prediction step
        if u is not None:
            for particle in self.particles:
                particle.Movement(*u, meanw=self.meanweight)

    #   Update step
    def Update(self, z=None):
        # Clears the last total weight
        self.totalweight = 0

        # Applies the observation model to each particle
        for particle in self.particles:
            self.totalweight += particle.Sensor(z)

    #   Resample step
    def Resample(self, qtd):
        try:
            parts = []

            np.random.shuffle(self.particles)

            step = self.totalweight / (qtd + 1.)
            s = 0

            poses = []

            i = 1
            j = len(self.particles)

            self.meanweight = 0
            while i <= qtd and j >= 0:
                if step * i > s:
                    j -= 1
                    s += self.particles[j].weight
                else:
                    i += 1
                    p = self.particles[j]
                    parts.append(Particle(*p.copy(), factors=p.factors))
                    self.meanweight += p.weight
                    poses.append([p.x, p.y, np.cos(np.radians(p.rotation)) +
                                  np.sin(np.radians(p.rotation)) * 1j])

            self.particles = parts
            self.qtd = len(self.particles)
            self.totalweight = self.meanweight
            self.meanweight /= self.qtd

            m = np.mean(poses, 0)

            self.mean[0] = int(np.real(m[0]))
            self.mean[1] = int(np.real(m[1]))
            self.mean[2] = int(np.angle(m[2], True))

            poses = np.matrix(poses - m)
            self.std = np.power(np.sqrt(np.abs(np.linalg.det(((poses.T * poses) / (self.qtd+1))))), 1/3.)  # noqa E501
        except:
            print s, step, i, j
            exit()

    #   Main algorithm
    def main(self, u=None, z=None):
        self.Prediction(u)
        self.Update(z)
        self.Resample(self.qtd)
        return self.mean, self.std
