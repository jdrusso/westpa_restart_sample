#!/usr/bin/env python

import sys

# OpenMM Imports
import simtk.openmm as mm
import simtk.openmm.app as app

# ParmEd Imports
# from parmed import load_file
from parmed import unit as u

from numpy import sqrt, mean


# Load the Gromacs files
print("Loading Gromacs files...")

gro = app.GromacsGroFile("step4.1_equilibration.gro")
top = app.GromacsTopFile(
    "topol.top", periodicBoxVectors=gro.getPeriodicBoxVectors(), includeDir="."
)


# Create the OpenMM system
print("Creating OpenMM System")
system = top.createSystem(
    nonbondedMethod=app.PME,
    nonbondedCutoff=1.2 * u.nanometer,
    constraints=app.HBonds,
    #                        implicitSolvent=app.GBn2,
    #                        implicitSolventSaltConc=0.1*u.moles/u.liter,
)


# Create the integrator to do Langevin dynamics
print("Creating integrator")
integrator = mm.LangevinIntegrator(
    300 * u.kelvin,  # Temperature of heat bath
    1.0 / u.picoseconds,  # Friction coefficient
    1.0 * u.femtoseconds,  # Time step
)

# Define the platform to use; CUDA, OpenCL, CPU, or Reference. Or do not
# specify the platform to use the default (fastest) platform
platform = mm.Platform.getPlatformByName("OpenCL")
prop = {}


print("Preparing simulation")
# Create the Simulation object
sim = app.Simulation(top.topology, system, integrator, platform, prop)

# Set the particle positions
sim.context.setPositions(gro.positions)

pre_min = sim.context.getState(getEnergy=True).getPotentialEnergy()

# Minimize the energy
print("Minimizing energy...")

sim.minimizeEnergy(maxIterations=5000)

state = sim.context.getState(getForces=True, getEnergy=True)
forces = state.getForces(asNumpy=True)

print(f"Energy now at: {state.getPotentialEnergy()}")
max_force = max(sqrt(mean(forces ** 2, axis=1)))
print(f"Max RMS force is: {max_force:2f}")


post_min = sim.context.getState(getEnergy=True).getPotentialEnergy()

print("Energy minimized from %r to %r" % (pre_min, post_min))


# Serialize openmm objects
with open("minimized_system.xml", "w") as f:
    f.write(mm.XmlSerializer.serialize(system))

with open("minimized_integrator.xml", "w") as f:
    f.write(mm.XmlSerializer.serialize(integrator))

with open("minimized_pos.gro", "w") as f:
    app.PDBFile.writeFile(
        sim.topology, sim.context.getState(getPositions=True).getPositions(), f
    )


# sim.context.setPositions(gro.positions)s
sim.context.setVelocitiesToTemperature(300 * u.kelvin)

# Set up the reporters to report energies and coordinates every 100 steps
sim.reporters.append(
    app.StateDataReporter(
        sys.stdout,
        200,
        step=True,
        potentialEnergy=True,
        kineticEnergy=True,
        temperature=True,
    )
)


# Run dynamics to let system equilibrate
print("Running dynamics")

# for i in range(500):
sim.step(30000)

# Serialize openmm objects
with open("system.xml", "w") as f:
    f.write(mm.XmlSerializer.serialize(system))

with open("integrator.xml", "w") as f:
    f.write(mm.XmlSerializer.serialize(integrator))

with open("pos.gro", "w") as f:
    # gf.write(sim.topology, f)
    app.PDBFile.writeFile(
        sim.topology, sim.context.getState(getPositions=True).getPositions(), f
    )
